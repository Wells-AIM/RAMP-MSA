#!/usr/bin/env python3
"""Train PDER on frozen aligned MOSI/MOSEI features using official splits."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset

from src.models.progressive_diagnostic_evidence_model import (
    ProgressiveDiagnosticEvidenceReasoningModel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mosi", "mosei"), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--base-config", default="configs/eatd_pder_full.json")
    parser.add_argument("--variant", choices=("full", "g_only"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--min-epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--refinement-start-epoch", type=int, default=0)
    parser.add_argument("--refinement-ramp-epochs", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def fit_stream_stats(split: dict[str, Any], key: str) -> tuple[np.ndarray, np.ndarray]:
    width = int(np.asarray(split[key][0]).shape[-1])
    total = np.zeros(width, dtype=np.float64)
    square = np.zeros(width, dtype=np.float64)
    count = 0
    for value in split[key]:
        array = np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if not array.size:
            continue
        total += array.sum(axis=0, dtype=np.float64)
        square += np.square(array, dtype=np.float64).sum(axis=0, dtype=np.float64)
        count += int(array.shape[0])
    mean = total / max(count, 1)
    variance = np.maximum(square / max(count, 1) - np.square(mean), 1e-6)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


class SentimentEvidenceDataset(Dataset):
    def __init__(
        self,
        split: dict[str, Any],
        audio_stats: tuple[np.ndarray, np.ndarray],
        vision_stats: tuple[np.ndarray, np.ndarray],
        max_length: int,
        smoke: bool = False,
    ) -> None:
        labels = np.asarray(split["regression_labels"], dtype=np.float32).reshape(-1)
        self.indices = np.flatnonzero(labels != 0).tolist()
        if smoke:
            self.indices = self.indices[: min(len(self.indices), 192)]
        self.split = split
        self.labels = labels
        self.audio_mean, self.audio_std = audio_stats
        self.vision_mean, self.vision_std = vision_stats
        self.max_length = int(max_length)
        self.input_dim = 768

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, Any]:
        index = self.indices[position]
        text_raw = np.nan_to_num(
            np.asarray(self.split["text"][index], dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        audio_raw = np.nan_to_num(
            np.asarray(self.split["audio"][index], dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        vision_raw = np.nan_to_num(
            np.asarray(self.split["vision"][index], dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        text = np.zeros((self.max_length, self.input_dim), dtype=np.float32)
        text_mask = np.zeros(self.max_length, dtype=np.bool_)
        text_length = min(text_raw.shape[0], self.max_length)
        text[:text_length] = text_raw[:text_length]
        text_mask[:text_length] = True

        nonverbal = np.zeros((self.max_length, self.input_dim), dtype=np.float32)
        nonverbal_mask = np.zeros(self.max_length, dtype=np.bool_)
        aligned_length = min(audio_raw.shape[0], vision_raw.shape[0], self.max_length)
        audio = (audio_raw[:aligned_length] - self.audio_mean) / self.audio_std
        vision = (vision_raw[:aligned_length] - self.vision_mean) / self.vision_std
        joined = np.concatenate([audio, vision], axis=-1)
        nonverbal[:aligned_length, : joined.shape[-1]] = np.clip(joined, -10.0, 10.0)
        nonverbal_mask[:aligned_length] = True

        regression_label = float(self.labels[index])
        return {
            "text": torch.from_numpy(text).unsqueeze(0),
            "text_mask": torch.from_numpy(text_mask).unsqueeze(0),
            "nonverbal": torch.from_numpy(nonverbal).unsqueeze(0),
            "nonverbal_mask": torch.from_numpy(nonverbal_mask).unsqueeze(0),
            "label": torch.tensor(1 if regression_label > 0 else 0, dtype=torch.long),
            "regression_label": torch.tensor(regression_label, dtype=torch.float32),
            "id": str(self.split["id"][index]),
        }


def build_model(
    base_config_path: str,
    variant: str,
    refinement_start_epoch: int,
    refinement_ramp_epochs: int,
) -> ProgressiveDiagnosticEvidenceReasoningModel:
    with open(base_config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    model = config["model"]
    model.update(
        {
            "input_dim": 768,
            "hidden_dim": 256,
            "max_evidence_units": 1,
            "queries_per_class": 4,
            "num_queries_per_class": 4,
            "num_query_layers": 4,
            "num_heads": 4,
            "dropout": 0.25,
            "token_dropout": 0.05,
            "pder_general_layers": 2,
            "pder_consensus_scale": 0.08 if variant == "full" else 0.0,
            "pder_complement_scale": 0.06 if variant == "full" else 0.0,
            "pder_evidence_registration_timing": "pre",
            "use_reconstruction_loss": False,
            "class_logit_bias_init": "zeros",
            "class_logit_bias_scale": 0.0,
        }
    )
    config["training"].update(
        {
            "pder_refinement_start_epoch": int(refinement_start_epoch),
            "pder_refinement_ramp_epochs": int(refinement_ramp_epochs),
            "teacher_leading_loss_weight": 0.0,
        }
    )
    config["cross_sample"] = {"use_cross_sample": False}
    config["embedding_augmentation"] = {"use_embedding_augmentation": False}
    return ProgressiveDiagnosticEvidenceReasoningModel(config)


def binary_metrics(labels: list[int], preds: list[int], probs: list[float]) -> dict[str, Any]:
    precision, recall, f1_positive, _ = precision_recall_fscore_support(
        labels, preds, labels=[1], average=None, zero_division=0
    )
    return {
        "acc2_nonzero": float(accuracy_score(labels, preds)),
        "f1_positive": float(f1_positive[0]),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "precision_positive": float(precision[0]),
        "recall_positive": float(recall[0]),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
        "n": len(labels),
        "positive_rate": float(np.mean(labels)),
        "predicted_positive_rate": float(np.mean(preds)),
        "mean_positive_probability": float(np.mean(probs)),
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        batch["text"].to(device, non_blocking=True),
        batch["nonverbal"].to(device, non_blocking=True),
        batch["text_mask"].to(device, non_blocking=True),
        batch["nonverbal_mask"].to(device, non_blocking=True),
        batch["label"].to(device, non_blocking=True),
    )


@torch.no_grad()
def evaluate(
    model: ProgressiveDiagnosticEvidenceReasoningModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    labels: list[int] = []
    preds: list[int] = []
    probs: list[float] = []
    ids: list[str] = []
    regression_labels: list[float] = []
    diagnostic_sums: dict[str, float] = {}
    diagnostic_count = 0
    for batch in loader:
        text, nonverbal, text_mask, nonverbal_mask, target = move_batch(batch, device)
        logits, details = model(text, nonverbal, text_mask, nonverbal_mask)
        probability = torch.softmax(logits, dim=-1)[:, 1]
        prediction = logits.argmax(dim=-1)
        labels.extend(target.cpu().tolist())
        preds.extend(prediction.cpu().tolist())
        probs.extend(probability.cpu().tolist())
        ids.extend([str(value) for value in batch["id"]])
        regression_labels.extend(batch["regression_label"].cpu().tolist())
        for key, value in details.items():
            if not torch.is_tensor(value) or value.numel() == 0:
                continue
            if key.startswith("pder_") and torch.isfinite(value).all():
                diagnostic_sums[key] = diagnostic_sums.get(key, 0.0) + float(value.float().mean().cpu())
        diagnostic_count += 1
    metrics = binary_metrics(labels, preds, probs)
    metrics["diagnostics"] = {
        key: value / max(diagnostic_count, 1) for key, value in diagnostic_sums.items()
    }
    predictions = {
        "id": ids,
        "regression_label": regression_labels,
        "binary_label": labels,
        "positive_probability": probs,
        "prediction": preds,
    }
    return metrics, predictions


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.data, "rb") as handle:
        payload = pickle.load(handle)
    audio_stats = fit_stream_stats(payload["train"], "audio")
    vision_stats = fit_stream_stats(payload["train"], "vision")
    datasets = {
        split: SentimentEvidenceDataset(
            payload[split], audio_stats, vision_stats, args.max_length, smoke=args.smoke
        )
        for split in ("train", "valid", "test")
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        ),
        "valid": DataLoader(
            datasets["valid"], batch_size=args.batch_size, num_workers=0, pin_memory=True
        ),
        "test": DataLoader(
            datasets["test"], batch_size=args.batch_size, num_workers=0, pin_memory=True
        ),
    }

    model = build_model(
        args.base_config,
        args.variant,
        args.refinement_start_epoch,
        args.refinement_ramp_epochs,
    ).to(device)
    train_dataset = datasets["train"]
    train_labels = [
        1 if float(train_dataset.labels[index]) > 0 else 0
        for index in train_dataset.indices
    ]
    counts = np.bincount(train_labels, minlength=2).astype(np.float32)
    class_weights = torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1.0)), device=device)
    if hasattr(model, "set_class_weights"):
        model.set_class_weights(class_weights)
    if hasattr(model, "set_class_counts"):
        model.set_class_counts(torch.tensor(counts, device=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_epoch = -1
    stale_epochs = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        if hasattr(model, "set_embedding_augmentation_epoch"):
            model.set_embedding_augmentation_epoch(epoch, args.epochs, len(datasets["train"]))
        running_loss = 0.0
        seen = 0
        train_labels_epoch: list[int] = []
        train_preds_epoch: list[int] = []
        for batch in loaders["train"]:
            text, nonverbal, text_mask, nonverbal_mask, target = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(text, nonverbal, text_mask, nonverbal_mask)
            loss = F.cross_entropy(logits, target, weight=class_weights, label_smoothing=0.05)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.detach().cpu()) * target.size(0)
            seen += target.size(0)
            train_labels_epoch.extend(target.detach().cpu().tolist())
            train_preds_epoch.extend(logits.detach().argmax(dim=-1).cpu().tolist())
        scheduler.step()

        valid_metrics, _ = evaluate(model, loaders["valid"], device)
        score = valid_metrics["f1_macro"]
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(seen, 1),
            "train_acc": float(accuracy_score(train_labels_epoch, train_preds_epoch)),
            "valid": valid_metrics,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=True), flush=True)
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch + 1
            stale_epochs = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": best_epoch}, output_dir / "best_model.pth")
        else:
            stale_epochs += 1
        if epoch + 1 >= args.min_epochs and stale_epochs >= args.patience:
            break

    checkpoint = torch.load(output_dir / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    valid_metrics, valid_predictions = evaluate(model, loaders["valid"], device)
    test_metrics, test_predictions = evaluate(model, loaders["test"], device)
    result = {
        "dataset": args.dataset,
        "variant": args.variant,
        "seed": args.seed,
        "protocol": "official split, nonzero sentiment binary classification",
        "modalities": "frozen BERT text plus train-normalized concatenated audio-vision evidence",
        "best_epoch": best_epoch,
        "elapsed_seconds": time.time() - start_time,
        "split_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "class_counts_train": counts.astype(int).tolist(),
        "valid": valid_metrics,
        "test": test_metrics,
        "args": vars(args),
    }
    with open(output_dir / "result.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=True)
    with open(output_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, ensure_ascii=True)
    with open(output_dir / "valid_predictions.json", "w", encoding="utf-8") as handle:
        json.dump(valid_predictions, handle, ensure_ascii=True)
    with open(output_dir / "test_predictions.json", "w", encoding="utf-8") as handle:
        json.dump(test_predictions, handle, ensure_ascii=True)
    with open(output_dir / "normalization.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "audio_mean": audio_stats[0].tolist(),
                "audio_std": audio_stats[1].tolist(),
                "vision_mean": vision_stats[0].tolist(),
                "vision_std": vision_stats[1].tolist(),
            },
            handle,
            indent=2,
        )
    (output_dir / "COMPLETE.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
