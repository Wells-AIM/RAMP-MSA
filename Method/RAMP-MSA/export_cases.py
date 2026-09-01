#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import torch

from ramp_msa.config import apply_overrides, load_config
from ramp_msa.data import build_dataloaders
from ramp_msa.losses import task_loss_per_sample
from ramp_msa.model import RAMPModel
from ramp_msa.utils import to_device


POLICY_NAMES = ["T", "A", "V", "TA", "TV", "AV", "TAV"]


def main():
    p = argparse.ArgumentParser(description="Export high-gain / harmful RAMP retrieval cases")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-path", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--output", default="retrieval_cases.json")
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    if args.data_path:
        cfg.data.path = args.data_path
        cfg.data.synthetic = False
    device_str = args.device or cfg.experiment.get("device", "cuda")
    if str(device_str).startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    loaders, info = build_dataloaders(cfg)
    model = RAMPModel(info.text_dim, info.audio_dim, info.vision_dim, cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    cases = []
    with torch.no_grad():
        for batch in loaders["test"]:
            ids = batch["id"]
            texts = batch.get("raw_text", [""] * len(ids))
            batch = to_device(batch, device)
            out = model(batch, use_memory=True)
            y = batch["label"]
            base_loss = task_loss_per_sample(out["base_logits"], y, str(cfg.task.type))
            final_loss = task_loss_per_sample(out["final_logits"], y, str(cfg.task.type))
            gain = base_loss - final_loss
            diff = out["components"][:, 3:].norm(dim=-1).sum(dim=-1)
            regime = out["route_probs"].argmax(dim=-1)
            if str(cfg.task.type) == "regression":
                base_pred = out["base_logits"].squeeze(-1)
                final_pred = out["final_logits"].squeeze(-1)
            else:
                base_pred = out["base_logits"].argmax(dim=-1)
                final_pred = out["final_logits"].argmax(dim=-1)
            for i in range(len(ids)):
                cases.append({
                    "id": ids[i],
                    "raw_text": texts[i],
                    "label": float(y[i].item()) if str(cfg.task.type) == "regression" else int(y[i].item()),
                    "base_prediction": float(base_pred[i].item()) if str(cfg.task.type) == "regression" else int(base_pred[i].item()),
                    "final_prediction": float(final_pred[i].item()) if str(cfg.task.type) == "regression" else int(final_pred[i].item()),
                    "memory_gain": float(gain[i].item()),
                    "interaction_difficulty": float(diff[i].item()),
                    "memory_gate": float(out["memory_gate"][i].item()),
                    "top_regime": int(regime[i].item()),
                    "retrieved_policy": {name: float(out["retrieved_policy"][i, j].item()) for j, name in enumerate(POLICY_NAMES)},
                })

    cases_sorted = sorted(cases, key=lambda x: x["memory_gain"], reverse=True)
    report = {
        "most_helpful": cases_sorted[: args.top],
        "most_harmful": list(reversed(cases_sorted[-args.top :])),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
