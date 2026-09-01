"""Training loop for text+audio query-adapter models."""
import copy
import fnmatch
import json
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from src.models.modality_decoupled_query_model import TokenPrependedModalityDecoupledQueryModel
from src.models.unified_referential_evidence_query_model import RefFormerAudioTextEmotionModel
from src.models.progressive_diagnostic_evidence_model import ProgressiveDiagnosticEvidenceReasoningModel
from src.preprocess.unified_evidence_dataset import get_unified_evidence_dataloaders, resolve_label_space_auto_value
from src.training.config import create_experiment_dir, get_device, load_config, set_seed

logger = logging.getLogger(__name__)


def binary_positive_metrics(
    labels: Sequence[int],
    preds: Sequence[int],
    positive_label: int = 1,
) -> Dict[str, float]:
    """Return positive-class metrics used to reject all-negative checkpoints."""
    tp = sum(1 for label, pred in zip(labels, preds) if label == positive_label and pred == positive_label)
    fp = sum(1 for label, pred in zip(labels, preds) if label != positive_label and pred == positive_label)
    fn = sum(1 for label, pred in zip(labels, preds) if label == positive_label and pred != positive_label)
    support = sum(1 for label in labels if label == positive_label)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall <= 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "positive_precision": float(precision),
        "positive_recall": float(recall),
        "positive_f1": float(f1),
        "positive_support": float(support),
        "positive_tp": float(tp),
        "positive_fp": float(fp),
        "positive_fn": float(fn),
    }


DIAGNOSTIC_SCALAR_KEYS = {
    "num_queries_per_class",
    "query_pool_entropy",
    "query_update_norm_ratio",
    "query_class_cosine",
    "query_adaption_disabled",
    "query_updates_frozen",
    "referential_gate_mean",
    "modal_pool_text_norm",
    "modal_pool_audio_norm",
    "qa_gate_mean",
    "text_sp_norm",
    "text_sh_norm",
    "audio_sp_norm",
    "audio_sh_norm",
    "num_encoder_layers",
}


def collect_query_diagnostics(logits_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    diagnostics: Dict[str, torch.Tensor] = {}
    for key, value in logits_dict.items():
        if not (
            key in DIAGNOSTIC_SCALAR_KEYS
            or key.startswith("sp_sh_query_")
            or key.startswith("sidebranch_sp_sh_query_")
        ):
            continue
        if not torch.is_tensor(value) or value.numel() == 0:
            continue
        scalar = value.detach()
        if not torch.is_floating_point(scalar):
            scalar = scalar.float()
        scalar = torch.nan_to_num(scalar, nan=0.0, posinf=1e4, neginf=-1e4).mean()
        if torch.isfinite(scalar).item():
            diagnostics[key] = scalar
    return diagnostics


def is_loggable_loss_scalar(metric_name: str) -> bool:
    if metric_name in {"total_loss", "main_loss", "aux_loss", "layer_loss", "global_loss", "light_loss"}:
        return True
    return metric_name.startswith((
        "loss_",
        "cross_sample_",
        "sp_sh_",
        "hqa_",
        "teacher_",
        "init_logit_distill_",
        "query_contrastive_",
    ))


class ModelEMA:
    """Maintain an exponential moving average of one model's weights."""

    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = float(decay)
        self.shadow = {key: value.detach().clone() for key, value in model.state_dict().items()}
        self.backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        for key, value in model_state.items():
            if key not in self.shadow:
                self.shadow[key] = value.detach().clone()
                continue
            if torch.is_floating_point(value):
                self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[key].copy_(value.detach())

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {key: value.detach().clone() for key, value in self.shadow.items()}

    def store(self, model: nn.Module) -> None:
        self.backup = {key: value.detach().clone() for key, value in model.state_dict().items()}

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=False)

    def restore(self, model: nn.Module) -> None:
        if self.backup:
            model.load_state_dict(self.backup, strict=False)
            self.backup = {}


def average_checkpoint_state_dicts(checkpoint_paths: List[str]) -> Dict[str, torch.Tensor]:
    """Average floating-point tensors from several checkpoints into one model state."""
    if not checkpoint_paths:
        raise ValueError("checkpoint_paths must not be empty")

    averaged_state: Optional[Dict[str, torch.Tensor]] = None
    checkpoint_count = 0
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if averaged_state is None:
            averaged_state = {}
            for key, value in state_dict.items():
                tensor = value.detach().cpu()
                averaged_state[key] = tensor.float().clone() if torch.is_floating_point(tensor) else tensor.clone()
        else:
            for key, value in state_dict.items():
                if key not in averaged_state:
                    continue
                tensor = value.detach().cpu()
                if torch.is_floating_point(tensor) and torch.is_floating_point(averaged_state[key]):
                    averaged_state[key].add_(tensor.float())
        checkpoint_count += 1

    assert averaged_state is not None
    for value in averaged_state.values():
        if torch.is_floating_point(value):
            value.div_(checkpoint_count)
    return averaged_state


def build_query_model(config: Dict[str, object]) -> nn.Module:
    """Build either stage of the validated QA4 warm-to-full pipeline."""
    model_type = config.get("model_type", "refformer_audio_text_emotion")
    if model_type == "refformer_audio_text_emotion":
        return RefFormerAudioTextEmotionModel(config)
    if model_type == "text_audio_token_prepended_modality_decoupled_query":
        return TokenPrependedModalityDecoupledQueryModel(config)
    if model_type == "progressive_diagnostic_evidence_reasoning":
        return ProgressiveDiagnosticEvidenceReasoningModel(config)
    raise ValueError(f"Unsupported model_type: {model_type!r}")


def load_model_state_compatible(model: nn.Module, state_dict: Dict[str, torch.Tensor], context: str) -> None:
    """Load checkpoints across compatible QA ablations."""
    model_state = model.state_dict()
    compatible_state = {}
    skipped_shape = []
    for key, value in state_dict.items():
        if key in model_state and tuple(model_state[key].shape) != tuple(value.shape):
            skipped_shape.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        compatible_state[key] = value
    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    if missing or unexpected:
        logger.info(
            "%s checkpoint兼容加载: missing=%d, unexpected=%d",
            context,
            len(missing),
            len(unexpected),
        )
    if skipped_shape:
        preview = ", ".join(
            f"{key}: {old}->{new}" for key, old, new in skipped_shape[:8]
        )
        logger.info(
            "%s checkpoint跳过shape不匹配参数: %d (%s)",
            context,
            len(skipped_shape),
            preview,
        )


def get_query_dataloaders(config: Dict[str, object]):
    """Build the unified cached-feature dataloaders used by EATD and DAIC."""
    data_config = config.get("data", {})
    feature_format = str(data_config.get("feature_format", "")).lower()
    if feature_format != "unified_evidence":
        raise ValueError("Only data.feature_format='unified_evidence' is supported")
    return get_unified_evidence_dataloaders(config)


def compute_class_counts(dataset, num_classes: int) -> torch.Tensor:
    labels = [int(item["label_id"]) for item in dataset.items]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    return torch.tensor(counts, dtype=torch.float32)


def resolve_class_weight_power(dataset, num_classes: int, power_config) -> float:
    labels = torch.tensor([int(item["label_id"]) for item in dataset.items], dtype=torch.long)
    power = resolve_label_space_auto_value("class_weight_power", power_config, labels, num_classes)
    return max(float(power), 0.0)


def resolve_learning_rate(num_classes: int, lr_config) -> float:
    if isinstance(lr_config, str):
        if lr_config.lower() == "label_space_auto":
            return 5e-4 if num_classes <= 2 else 5e-5
        return float(lr_config)
    return float(lr_config)


def resolve_prior_bias_scale(num_classes: int, scale_config) -> float:
    if isinstance(scale_config, str):
        if scale_config.lower() == "label_space_auto":
            return 1.0 if num_classes <= 2 else 0.0
        return float(scale_config)
    return float(scale_config)


def compute_class_weights(dataset, num_classes: int, power=1.0) -> torch.Tensor:
    labels = [int(item["label_id"]) for item in dataset.items]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    resolved_power = resolve_class_weight_power(dataset, num_classes, power)
    total = counts.sum()
    weights = total / (num_classes * counts)
    weights = np.power(weights, resolved_power)
    weights = weights / weights.mean()
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    dataset.resolved_class_weight_power = resolved_power
    print(f"QA class counts: {counts.astype(int).tolist()}")
    print(f"QA class weights: {weights_tensor} (power={resolved_power}, config={power})")
    return weights_tensor


def resolve_auto_train_epochs(train_loader, training_config: Dict[str, object]) -> Tuple[int, Dict[str, object]]:
    configured_epochs = int(training_config["epochs"])
    auto_config = training_config.get("auto_train_steps")
    steps_per_epoch = max(len(train_loader), 1)
    if not auto_config:
        return configured_epochs, {
            "enabled": False,
            "configured_epochs": configured_epochs,
            "resolved_epochs": configured_epochs,
            "steps_per_epoch": steps_per_epoch,
            "target_update_steps": configured_epochs * steps_per_epoch,
            "scheduler_reference_steps": configured_epochs * steps_per_epoch,
            "scheduler_epochs": configured_epochs,
        }
    min_steps = int(auto_config.get("min_steps", 320))
    epoch_multiplier = float(auto_config.get("epoch_multiplier", 6.0))
    target_update_steps = int(np.ceil(max(float(min_steps), epoch_multiplier * steps_per_epoch)))
    resolved_epochs = int(np.ceil(target_update_steps / steps_per_epoch))
    max_epochs = auto_config.get("max_epochs")
    if max_epochs is not None:
        resolved_epochs = min(resolved_epochs, int(max_epochs))
    resolved_epochs = max(resolved_epochs, 1)

    scheduler_reference_limit = int(auto_config.get("scheduler_reference_max_steps_per_epoch", 12))
    if steps_per_epoch <= scheduler_reference_limit:
        scheduler_reference_steps = int(auto_config.get("scheduler_reference_steps", target_update_steps))
    else:
        scheduler_reference_steps = target_update_steps
    scheduler_reference_steps = max(scheduler_reference_steps, target_update_steps, steps_per_epoch)
    scheduler_epochs = max(int(np.ceil(scheduler_reference_steps / steps_per_epoch)), resolved_epochs)
    return resolved_epochs, {
        "enabled": True,
        "configured_epochs": configured_epochs,
        "resolved_epochs": resolved_epochs,
        "steps_per_epoch": steps_per_epoch,
        "target_update_steps": target_update_steps,
        "min_steps": min_steps,
        "epoch_multiplier": epoch_multiplier,
        "max_epochs": max_epochs,
        "scheduler_reference_steps": scheduler_reference_steps,
        "scheduler_reference_max_steps_per_epoch": scheduler_reference_limit,
        "scheduler_epochs": scheduler_epochs,
    }


def move_batch(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {
        "text_features": batch["text_features"].to(device).float(),
        "text_attention_mask": batch["text_attention_mask"].to(device).bool(),
        "audio_features": batch["audio_features"].to(device).float(),
        "audio_attention_mask": batch["audio_attention_mask"].to(device).bool(),
        "labels": batch["emotion"].to(device),
        "sample_ids": batch.get("sample_id", batch.get("subject_id", batch.get("id"))),
    }


def predict_from_logits(
    logits: torch.Tensor,
    num_classes: int,
    decision_threshold: Optional[float] = None,
) -> torch.Tensor:
    """Predict labels, with optional calibrated threshold for binary QA models."""
    if decision_threshold is not None and num_classes == 2:
        depressed_probs = torch.softmax(logits, dim=-1)[:, 1]
        return (depressed_probs >= float(decision_threshold)).long()
    return torch.argmax(logits, dim=1)


def apply_inference_logit_adjustment(
    model: nn.Module,
    logits: torch.Tensor,
    tau: float = 0.0,
) -> torch.Tensor:
    """Apply a fixed train-prior logit correction for metric-time argmax only."""
    if abs(float(tau)) < 1e-12:
        return logits
    class_log_prior = getattr(model, "class_log_prior", None)
    if class_log_prior is None or class_log_prior.numel() != logits.size(-1):
        return logits
    return logits + float(tau) * class_log_prior.to(logits.device)


def _soft_target_cross_entropy(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    if class_weights is not None and class_weights.numel() == logits.size(-1):
        weights = class_weights.to(device=logits.device, dtype=logits.dtype).view(1, -1)
        return -(soft_targets * weights * log_probs).sum(dim=-1).mean()
    return -(soft_targets * log_probs).sum(dim=-1).mean()


def _batch_mixup_auxiliary_loss(
    model: nn.Module,
    data: Dict[str, object],
    num_classes: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    labels = data["labels"]
    batch_size = int(labels.size(0))
    device = labels.device
    zero = torch.zeros((), device=device)
    if batch_size < 2:
        return zero, {
            "mixup_lambda": zero,
            "mixup_pair_label_diff": zero,
            "mixup_loss": zero,
        }
    alpha = max(float(getattr(model, "cross_sample_mixup_alpha", 0.2)), 1e-6)
    min_lambda = float(getattr(model, "cross_sample_mixup_min_lambda", 0.05))
    beta = torch.distributions.Beta(
        torch.tensor(alpha, device=device),
        torch.tensor(alpha, device=device),
    )
    lam_tensor = beta.sample().to(dtype=data["text_features"].dtype)
    lam_tensor = lam_tensor.clamp(min_lambda, 1.0 - min_lambda)
    lam = lam_tensor.view(1, *([1] * (data["text_features"].dim() - 1)))
    perm = torch.randperm(batch_size, device=device)
    text_mixed = lam * data["text_features"] + (1.0 - lam) * data["text_features"][perm]
    audio_lam = lam_tensor.view(1, *([1] * (data["audio_features"].dim() - 1)))
    audio_mixed = audio_lam * data["audio_features"] + (1.0 - audio_lam) * data["audio_features"][perm]
    text_mask = data["text_attention_mask"] | data["text_attention_mask"][perm]
    audio_mask = data["audio_attention_mask"] | data["audio_attention_mask"][perm]
    target_a = F.one_hot(labels, num_classes=num_classes).to(dtype=text_mixed.dtype)
    target_b = F.one_hot(labels[perm], num_classes=num_classes).to(dtype=text_mixed.dtype)
    soft_targets = lam_tensor * target_a + (1.0 - lam_tensor) * target_b
    mixed_logits, _ = model(text_mixed, audio_mixed, text_mask, audio_mask)
    mixed_logits = torch.nan_to_num(mixed_logits, nan=0.0, posinf=1e4, neginf=-1e4)
    class_weights = getattr(model, "class_weights", None)
    loss = _soft_target_cross_entropy(mixed_logits, soft_targets, class_weights)
    pair_label_diff = (labels != labels[perm]).to(dtype=text_mixed.dtype).mean()
    return loss, {
        "mixup_lambda": lam_tensor.detach(),
        "mixup_pair_label_diff": pair_label_diff.detach(),
        "mixup_loss": loss.detach(),
    }



def _rank_mixup_auxiliary_loss(
    model: nn.Module,
    data: Dict[str, object],
    logits: torch.Tensor,
    num_classes: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """RankMixup-style train-only cross-sample loss.

    Mixed samples are intentionally treated as less confident than raw samples;
    we avoid trusting ordinary mixup soft labels as a full target.
    """
    labels = data["labels"]
    batch_size = int(labels.size(0))
    device = labels.device
    zero = torch.zeros((), device=device)
    if batch_size < 2:
        return zero, {
            "mixup_lambda": zero,
            "mixup_pair_label_diff": zero,
            "mixup_rank_loss": zero,
            "mixup_ce_loss": zero,
            "mixup_rank_active": zero,
            "mixup_raw_conf": zero,
            "mixup_mixed_conf": zero,
        }

    alpha = max(float(getattr(model, "cross_sample_mixup_alpha", 0.4)), 1e-6)
    min_lambda = float(getattr(model, "cross_sample_mixup_min_lambda", 0.65))
    min_lambda = min(max(min_lambda, 0.5), 0.99)
    rank_margin = float(getattr(model, "cross_sample_rank_mixup_margin", 0.03))
    pair_margin = float(getattr(model, "cross_sample_rank_mixup_pair_margin", 0.05))
    ce_weight = float(getattr(model, "cross_sample_rank_mixup_ce_weight", 0.10))
    min_anchor_mass = float(getattr(model, "cross_sample_rank_mixup_target_min_anchor_mass", 0.80))
    min_anchor_mass = min(max(min_anchor_mass, 0.5), 1.0)

    beta = torch.distributions.Beta(
        torch.tensor(alpha, device=device),
        torch.tensor(alpha, device=device),
    )
    lam_tensor = beta.sample().to(dtype=data["text_features"].dtype)
    lam_tensor = torch.maximum(lam_tensor, 1.0 - lam_tensor).clamp(min_lambda, 1.0)
    lam = lam_tensor.view(1, *([1] * (data["text_features"].dim() - 1)))
    perm = torch.randperm(batch_size, device=device)
    text_mixed = lam * data["text_features"] + (1.0 - lam) * data["text_features"][perm]
    audio_lam = lam_tensor.view(1, *([1] * (data["audio_features"].dim() - 1)))
    audio_mixed = audio_lam * data["audio_features"] + (1.0 - audio_lam) * data["audio_features"][perm]
    text_mask = data["text_attention_mask"] | data["text_attention_mask"][perm]
    audio_mask = data["audio_attention_mask"] | data["audio_attention_mask"][perm]

    mixed_logits, _ = model(text_mixed, audio_mixed, text_mask, audio_mask)
    mixed_logits = torch.nan_to_num(mixed_logits, nan=0.0, posinf=1e4, neginf=-1e4)
    raw_probs = torch.softmax(torch.nan_to_num(logits.detach(), nan=0.0, posinf=1e4, neginf=-1e4), dim=-1)
    mixed_probs = torch.softmax(mixed_logits, dim=-1)
    raw_conf = raw_probs.gather(1, labels.view(-1, 1)).squeeze(1)
    mixed_anchor_conf = mixed_probs.gather(1, labels.view(-1, 1)).squeeze(1)
    partner_labels = labels[perm]
    mixed_partner_conf = mixed_probs.gather(1, partner_labels.view(-1, 1)).squeeze(1)
    pair_label_diff = (labels != partner_labels).to(dtype=mixed_logits.dtype)

    # RankMixup intuition: a mixed sample should be less confident than the raw
    # dominant source, and for different-label pairs its dominant-label confidence
    # should stay above the partner-label confidence by a lambda-dependent margin.
    raw_rank = F.relu(mixed_anchor_conf - raw_conf + rank_margin)
    pair_rank_margin = pair_margin * (2.0 * lam_tensor.to(mixed_logits.dtype) - 1.0).clamp(min=0.0)
    pair_rank = F.relu(mixed_partner_conf - mixed_anchor_conf + pair_rank_margin) * pair_label_diff
    rank_loss = raw_rank.mean() + pair_rank.mean()

    anchor_mass = torch.maximum(
        lam_tensor.to(mixed_logits.dtype),
        mixed_logits.new_tensor(min_anchor_mass),
    ).clamp(max=1.0)
    target_a = F.one_hot(labels, num_classes=num_classes).to(dtype=mixed_logits.dtype)
    target_b = F.one_hot(partner_labels, num_classes=num_classes).to(dtype=mixed_logits.dtype)
    same_label = (labels == partner_labels).to(dtype=mixed_logits.dtype).view(-1, 1)
    soft_targets = same_label * target_a + (1.0 - same_label) * (
        anchor_mass.view(-1, 1) * target_a + (1.0 - anchor_mass).view(-1, 1) * target_b
    )
    class_weights = getattr(model, "class_weights", None)
    ce_loss = _soft_target_cross_entropy(mixed_logits, soft_targets, class_weights)
    loss = rank_loss + ce_weight * ce_loss
    rank_active = ((raw_rank > 0).to(dtype=mixed_logits.dtype).mean() + (pair_rank > 0).to(dtype=mixed_logits.dtype).mean()) * 0.5
    return loss, {
        "mixup_lambda": lam_tensor.detach(),
        "mixup_pair_label_diff": pair_label_diff.mean().detach(),
        "mixup_rank_loss": rank_loss.detach(),
        "mixup_ce_loss": ce_loss.detach(),
        "mixup_rank_active": rank_active.detach(),
        "mixup_raw_conf": raw_conf.mean().detach(),
        "mixup_mixed_conf": mixed_anchor_conf.mean().detach(),
        "mixup_loss": loss.detach(),
    }

def _use_cross_sample_gradient_guard(model: nn.Module) -> bool:
    guard = str(getattr(model, "cross_sample_gradient_guard", "none") or "none").lower()
    return guard in {"pcgrad", "drop_conflict", "gradient_guard", "drop"}


def _grad_overlap_stats(main_grads, cs_grads) -> Tuple[float, float, float, float]:
    dot = 0.0
    main_norm_sq = 0.0
    cs_norm_sq = 0.0
    overlap_count = 0.0
    for main_grad, cs_grad in zip(main_grads, cs_grads):
        if main_grad is None or cs_grad is None:
            continue
        main_flat = main_grad.detach().float().reshape(-1)
        cs_flat = cs_grad.detach().float().reshape(-1)
        dot += float(torch.dot(main_flat, cs_flat).cpu())
        main_norm_sq += float(torch.dot(main_flat, main_flat).cpu())
        cs_norm_sq += float(torch.dot(cs_flat, cs_flat).cpu())
        overlap_count += 1.0
    main_norm = main_norm_sq ** 0.5
    cs_norm = cs_norm_sq ** 0.5
    denom = max(main_norm * cs_norm, 1e-12)
    grad_cos = dot / denom if overlap_count > 0 else 0.0
    return grad_cos, main_norm, cs_norm, overlap_count


def _assign_combined_grads(params, main_grads, cs_grads, keep_cross_sample: bool) -> None:
    for param, main_grad, cs_grad in zip(params, main_grads, cs_grads):
        grad = None
        if main_grad is not None:
            grad = main_grad.detach().clone()
        if keep_cross_sample and cs_grad is not None:
            if grad is None:
                grad = cs_grad.detach().clone()
            else:
                grad.add_(cs_grad.detach())
        param.grad = grad


def _backward_with_cross_sample_gradient_guard(
    model: nn.Module,
    main_loss: torch.Tensor,
    cross_sample_term: torch.Tensor,
) -> Dict[str, float]:
    params = [param for param in model.parameters() if param.requires_grad]
    main_grads = torch.autograd.grad(main_loss, params, retain_graph=True, allow_unused=True)
    cs_grads = torch.autograd.grad(cross_sample_term, params, retain_graph=False, allow_unused=True)
    grad_cos, main_norm, cs_norm, overlap_count = _grad_overlap_stats(main_grads, cs_grads)
    min_cos = float(getattr(model, "cross_sample_grad_guard_min_cos", 0.0))
    keep_cross_sample = bool(overlap_count <= 0.0 or grad_cos >= min_cos)
    _assign_combined_grads(params, main_grads, cs_grads, keep_cross_sample)
    return {
        "grad_cos": grad_cos,
        "main_grad_norm": main_norm,
        "cs_grad_norm": cs_norm,
        "guard_drop": 0.0 if keep_cross_sample else 1.0,
        "guard_overlap_params": overlap_count,
    }


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer: optim.Optimizer,
    device: torch.device,
    clip_grad_val: float,
    num_classes: int,
    prediction_distribution_loss_weight: float = 0.0,
    prediction_distribution_target: Optional[torch.Tensor] = None,
    ema: Optional[ModelEMA] = None,
    epoch_index: int = 0,
    init_teacher_model: Optional[nn.Module] = None,
    init_logit_distill_weight: float = 0.0,
    init_logit_distill_temperature: float = 2.0,
    init_logit_distill_min_confidence: float = 0.0,
    init_logit_distill_label_filter: str = "all",
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    skipped_batches = 0
    all_preds: List[int] = []
    all_labels: List[int] = []
    scalar_metric_sums: Dict[str, float] = {}

    if hasattr(model, "set_cross_sample_epoch"):
        model.set_cross_sample_epoch(epoch_index)

    pbar = tqdm(train_loader, desc="QA Training")
    for batch_idx, batch in enumerate(pbar):
        data = move_batch(batch, device)
        forward_kwargs = {}
        if getattr(model, "use_cross_sample", False):
            forward_kwargs["labels"] = data["labels"]
            forward_kwargs["sample_ids"] = data.get("sample_ids")
        logits, logits_dict = model(
            data["text_features"],
            data["audio_features"],
            data["text_attention_mask"],
            data["audio_attention_mask"],
            **forward_kwargs,
        )
        losses = model.calculate_losses(logits, logits_dict, data["labels"])
        loss = losses["total_loss"]
        if prediction_distribution_loss_weight > 0:
            mean_probs = torch.softmax(logits, dim=-1).mean(dim=0)
            if prediction_distribution_target is not None:
                target_probs = prediction_distribution_target.to(logits.device, dtype=mean_probs.dtype)
            else:
                target_probs = torch.full_like(mean_probs, 1.0 / max(num_classes, 1))
            distribution_loss = torch.mean((mean_probs - target_probs).pow(2))
            loss = loss + float(prediction_distribution_loss_weight) * distribution_loss

        if init_teacher_model is not None and float(init_logit_distill_weight) > 0.0:
            init_teacher_model.eval()
            with torch.no_grad():
                teacher_logits, _ = init_teacher_model(
                    data["text_features"],
                    data["audio_features"],
                    data["text_attention_mask"],
                    data["audio_attention_mask"],
                )
            if teacher_logits.shape == logits.shape:
                temperature = max(float(init_logit_distill_temperature), 1e-6)
                student_log_probs = F.log_softmax(logits / temperature, dim=-1)
                teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=-1)
                per_sample_init_distill = F.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction="none",
                ).sum(dim=-1) * (temperature ** 2)
                min_confidence = float(init_logit_distill_min_confidence)
                distill_mask = torch.ones_like(data["labels"], dtype=torch.bool, device=logits.device)
                label_filter = str(init_logit_distill_label_filter or "all").lower()
                if label_filter in {"negative", "negative_only", "non_depressed", "label0", "class0"}:
                    distill_mask = data["labels"].to(device=logits.device).long().eq(0)
                elif label_filter in {"positive", "positive_only", "depressed", "label1", "class1"}:
                    distill_mask = data["labels"].to(device=logits.device).long().eq(1)
                if min_confidence > 0.0:
                    teacher_confidence = teacher_probs.max(dim=-1).values
                    distill_mask = distill_mask & (teacher_confidence >= min_confidence)
                if distill_mask.any():
                    init_distill_loss = per_sample_init_distill[distill_mask].mean()
                    losses["init_logit_distill_active_ratio"] = (
                        distill_mask.to(dtype=logits.dtype).mean().detach()
                    )
                else:
                    init_distill_loss = logits.new_tensor(0.0)
                    losses["init_logit_distill_active_ratio"] = logits.new_tensor(0.0)
                loss = loss + float(init_logit_distill_weight) * init_distill_loss
                losses["init_logit_distill_loss"] = init_distill_loss.detach()
                losses["init_logit_distill_weight"] = logits.new_tensor(float(init_logit_distill_weight))
                losses["init_logit_distill_label_filter_negative"] = logits.new_tensor(
                    1.0
                    if label_filter
                    in {"negative", "negative_only", "non_depressed", "label0", "class0"}
                    else 0.0
                )

        cross_sample_mode = str(getattr(model, "cross_sample_mode", "") or "").lower()
        if (
            getattr(model, "use_cross_sample", False)
            and cross_sample_mode in {"batch_mixup_aux", "rank_mixup_aux"}
        ):
            if hasattr(model, "_cross_sample_loss_factor"):
                mixup_factor = model._cross_sample_loss_factor()
            else:
                mixup_factor = logits.new_tensor(1.0)
            if torch.is_tensor(mixup_factor):
                mixup_factor_value = float(mixup_factor.detach().cpu())
            else:
                mixup_factor_value = float(mixup_factor)
                mixup_factor = logits.new_tensor(mixup_factor_value)
            if mixup_factor_value > 0.0:
                if cross_sample_mode == "rank_mixup_aux":
                    mixup_loss, mixup_diagnostics = _rank_mixup_auxiliary_loss(model, data, logits, num_classes)
                else:
                    mixup_loss, mixup_diagnostics = _batch_mixup_auxiliary_loss(model, data, num_classes)
                mixup_applied = mixup_factor * mixup_loss
                loss = loss + float(getattr(model, "cross_sample_lambda", 1.0)) * mixup_applied
                losses["cross_sample_loss"] = mixup_loss
                losses["cross_sample_loss_l3"] = mixup_loss
                losses["cross_sample_loss_factor"] = mixup_factor
                losses["cross_sample_loss_applied"] = mixup_applied
                for diag_name, diag_value in mixup_diagnostics.items():
                    losses[f"cross_sample_{diag_name}"] = diag_value
            else:
                zero = logits.new_tensor(0.0)
                losses["cross_sample_loss"] = zero
                losses["cross_sample_loss_l3"] = zero
                losses["cross_sample_loss_factor"] = mixup_factor
                losses["cross_sample_loss_applied"] = zero

        if not torch.isfinite(loss) or not torch.isfinite(logits).all():
            skipped_batches += 1
            logger.warning(
                "跳过非有限QA训练batch: batch=%s, loss=%s, logits_finite=%s",
                batch_idx,
                float(loss.detach().cpu()) if torch.isfinite(loss.detach()).item() else "non-finite",
                bool(torch.isfinite(logits).all().detach().cpu().item()),
            )
            optimizer.zero_grad(set_to_none=True)
            pbar.set_postfix({"Skipped": skipped_batches})
            continue

        optimizer.zero_grad(set_to_none=True)
        used_gradient_guard = False
        if _use_cross_sample_gradient_guard(model):
            cross_sample_applied = losses.get("cross_sample_loss_applied")
            if torch.is_tensor(cross_sample_applied) and cross_sample_applied.requires_grad:
                cross_sample_term = float(getattr(model, "cross_sample_lambda", 1.0)) * cross_sample_applied
                if torch.isfinite(cross_sample_term.detach()).item() and abs(float(cross_sample_term.detach().cpu())) > 1e-12:
                    main_loss = loss - cross_sample_term
                    guard_metrics = _backward_with_cross_sample_gradient_guard(model, main_loss, cross_sample_term)
                    used_gradient_guard = True
                    for guard_name, guard_value in guard_metrics.items():
                        losses[f"cross_sample_{guard_name}"] = logits.new_tensor(float(guard_value))
        if not used_gradient_guard:
            loss.backward()
        grad_norm = None
        if clip_grad_val > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_val)
        nonfinite_grad = grad_norm is not None and not torch.isfinite(torch.as_tensor(grad_norm)).item()
        if nonfinite_grad:
            skipped_batches += 1
            logger.warning("跳过非有限QA梯度batch: batch=%s, grad_norm=%s", batch_idx, grad_norm)
            optimizer.zero_grad(set_to_none=True)
            pbar.set_postfix({"Skipped": skipped_batches})
            continue
        optimizer.step()
        if ema is not None:
            ema.update(model)

        batch_size = data["labels"].size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        for metric_name, metric_value in losses.items():
            if not is_loggable_loss_scalar(metric_name):
                continue
            if not torch.is_tensor(metric_value) or metric_value.numel() != 1:
                continue
            metric_scalar = metric_value.detach()
            if torch.isfinite(metric_scalar).item():
                scalar_metric_sums[metric_name] = scalar_metric_sums.get(metric_name, 0.0) + float(metric_scalar.cpu()) * batch_size
        all_preds.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
        all_labels.extend(data["labels"].detach().cpu().tolist())
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Avg_Loss": f"{total_loss / total_samples:.4f}", "Skipped": skipped_batches})

    metrics = {
        "loss": total_loss / max(total_samples, 1),
        "acc": accuracy_score(all_labels, all_preds) if all_labels else 0.0,
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0) if all_labels else 0.0,
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted", zero_division=0) if all_labels else 0.0,
        "skipped_batches": skipped_batches,
    }
    for metric_name, metric_sum in scalar_metric_sums.items():
        metrics[metric_name] = metric_sum / max(total_samples, 1)
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader,
    device: torch.device,
    num_classes: int,
    decision_threshold: Optional[float] = None,
    inference_logit_adjustment_tau: float = 0.0,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_subjects: List[str] = []
    all_positive_probs: List[float] = []
    diagnostic_sums: Dict[str, float] = {}
    scalar_metric_sums: Dict[str, float] = {}

    for batch in tqdm(data_loader, desc="QA Evaluating"):
        data = move_batch(batch, device)
        forward_kwargs = {}
        if getattr(model, "use_cross_sample", False) and getattr(model, "eval_use_cross_sample_ref", False):
            forward_kwargs["labels"] = data["labels"]
            forward_kwargs["sample_ids"] = data.get("sample_ids")
        logits, logits_dict = model(
            data["text_features"],
            data["audio_features"],
            data["text_attention_mask"],
            data["audio_attention_mask"],
            **forward_kwargs,
        )
        losses = model.calculate_losses(logits, logits_dict, data["labels"])

        if not torch.isfinite(logits).all():
            logger.warning("评估时发现非有限QA logits，已用有限值替换以避免指标计算崩溃")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        batch_loss = losses["total_loss"]
        if not torch.isfinite(batch_loss):
            logger.warning("评估时发现非有限QA loss，当前batch loss按0计入")
            batch_loss = torch.zeros_like(batch_loss)

        batch_size = data["labels"].size(0)
        total_loss += batch_loss.item() * batch_size
        total_samples += batch_size
        for metric_name, metric_value in losses.items():
            if not is_loggable_loss_scalar(metric_name):
                continue
            if not torch.is_tensor(metric_value) or metric_value.numel() != 1:
                continue
            metric_scalar = metric_value.detach()
            if torch.isfinite(metric_scalar).item():
                scalar_metric_sums[metric_name] = scalar_metric_sums.get(metric_name, 0.0) + float(metric_scalar.cpu()) * batch_size
        for key, value in collect_query_diagnostics(logits_dict).items():
            diagnostic_sums[key] = diagnostic_sums.get(key, 0.0) + float(value.cpu()) * batch_size
        metric_logits = apply_inference_logit_adjustment(model, logits, inference_logit_adjustment_tau)
        if num_classes == 2:
            positive_probs = torch.softmax(metric_logits, dim=1)[:, 1]
            all_positive_probs.extend(positive_probs.detach().cpu().tolist())
        preds = predict_from_logits(metric_logits, num_classes, decision_threshold)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(data["labels"].detach().cpu().tolist())
        all_subjects.extend(batch.get("subject_id", []))

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    label_class_count = max(len(set(all_labels)), 1)
    expected_class_count = max(min(num_classes, label_class_count), 1)
    predicted_class_count = len(set(all_preds))
    prediction_coverage = min(1.0, predicted_class_count / expected_class_count)
    balanced_score = (acc + f1_macro + f1_weighted) / 3.0
    positive_metrics = binary_positive_metrics(all_labels, all_preds, positive_label=min(1, num_classes - 1))
    metrics = {
        "loss": total_loss / max(total_samples, 1),
        "acc": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "balanced_acc": balanced_acc,
        "acc_macro_mean": 0.5 * (acc + f1_macro),
        "balanced_score": balanced_score,
        "prediction_coverage": prediction_coverage,
        "coverage_balanced_score": balanced_score * prediction_coverage,
        "confusion_matrix": confusion_matrix(all_labels, all_preds, labels=list(range(num_classes))).tolist(),
        "predictions": all_preds,
        "labels": all_labels,
        "subject_ids": all_subjects,
        "positive_probabilities": all_positive_probs,
        "inference_logit_adjustment_tau": float(inference_logit_adjustment_tau),
        **positive_metrics,
    }
    for key, value in diagnostic_sums.items():
        metrics[f"diag_{key}"] = value / max(total_samples, 1)
    for key, value in scalar_metric_sums.items():
        metrics[key] = value / max(total_samples, 1)
    return metrics


def build_scheduler(optimizer: optim.Optimizer, config: Dict[str, object], epochs: int):
    training_config = config.get("training", {})
    scheduler_name = training_config.get("scheduler", "cosine")
    if scheduler_name == "none":
        return None
    if scheduler_name == "linear":
        return LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=max(epochs, 1))
    warmup_epochs = int(epochs * training_config.get("warmup_ratio", 0.0))
    cosine_epochs = max(epochs - warmup_epochs, 1)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs)
    if warmup_epochs <= 0:
        return cosine
    warmup = LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_epochs)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])



def _stage_patterns_match(name: str, patterns) -> bool:
    if not patterns:
        return False
    for pattern in patterns:
        pattern = str(pattern)
        if not pattern:
            continue
        if fnmatch.fnmatch(name, pattern) or name.startswith(pattern) or pattern in name:
            return True
    return False


def _resolve_trainable_stage(training_config: Dict[str, object], epoch_number: int):
    stages = training_config.get("trainable_stages") or training_config.get("staged_trainable") or []
    if not stages:
        return None, None
    active_index = None
    for idx, stage in enumerate(stages):
        start_epoch = int(stage.get("start_epoch", stage.get("start", 1)))
        if epoch_number >= start_epoch:
            active_index = idx
    if active_index is None:
        active_index = 0
    return active_index, stages[active_index]


def _apply_trainable_stage(model: nn.Module, stage: Dict[str, object], logger=None) -> int:
    if not stage:
        return sum(param.numel() for param in model.parameters() if param.requires_grad)
    trainable_patterns = list(stage.get("trainable_patterns", stage.get("trainable_prefixes", [])) or [])
    frozen_patterns = list(stage.get("frozen_patterns", stage.get("freeze_patterns", stage.get("frozen_prefixes", []))) or [])
    if "trainable_keywords" in stage:
        trainable_patterns.extend(stage.get("trainable_keywords") or [])
    if "frozen_keywords" in stage:
        frozen_patterns.extend(stage.get("frozen_keywords") or [])
    default_trainable = bool(stage.get("default_trainable", not bool(trainable_patterns)))
    trainable_names = []
    frozen_names = []
    for name, param in model.named_parameters():
        trainable = default_trainable
        if trainable_patterns:
            trainable = _stage_patterns_match(name, trainable_patterns)
        if frozen_patterns and _stage_patterns_match(name, frozen_patterns):
            trainable = False
        param.requires_grad_(trainable)
        if trainable:
            trainable_names.append(name)
        else:
            frozen_names.append(name)
    trainable_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    if logger is not None:
        logger.info(
            "应用训练阶段: name=%s, default_trainable=%s, trainable_params=%d, frozen_params=%d, trainable_numel=%d",
            stage.get("name", "unnamed"),
            default_trainable,
            len(trainable_names),
            len(frozen_names),
            trainable_count,
        )
        logger.info("阶段可训练参数前缀示例: %s", trainable_names[:30])
    return trainable_count

def train_query_model(config_path: str) -> Tuple[nn.Module, Dict[str, List[float]]]:
    config = load_config(config_path)
    set_seed(config["random_seed"])
    device = get_device()
    logger.info(f"使用设备: {device}")

    exp_dir = create_experiment_dir(config)
    logger.info(f"实验目录: {exp_dir}")

    dataloaders = get_query_dataloaders(config)
    train_loader = dataloaders["train"]
    dev_loader = dataloaders["dev"]
    test_loader = dataloaders.get("test")

    model = build_query_model(config).to(device)
    num_classes = int(config["model"].get("output", {}).get("num_classes", 2))
    training_config = config.get("training", {})
    evaluation_config = config.get("evaluation", {})
    evaluate_independent_test = bool(
        evaluation_config.get("evaluate_independent_test", True)
    )
    decision_threshold = evaluation_config.get("decision_threshold")
    if decision_threshold is not None:
        decision_threshold = float(decision_threshold)
        logger.info(f"使用二分类校准阈值: depressed_prob >= {decision_threshold:.4f}")
    inference_logit_adjustment_tau = float(
        evaluation_config.get(
            "inference_logit_adjustment_tau",
            training_config.get("inference_logit_adjustment_tau", 0.0),
        )
        or 0.0
    )
    if abs(inference_logit_adjustment_tau) > 1e-12:
        logger.info(f"使用统一训练先验logit校准: tau={inference_logit_adjustment_tau:.4f}")
    class_weights = None
    if training_config.get("class_weights", True):
        class_weight_power_config = training_config.get("class_weight_power", 1.0)
        class_weights = compute_class_weights(
            train_loader.dataset,
            num_classes,
            class_weight_power_config,
        ).to(device)
        model.set_class_weights(class_weights)
        logger.info(
            "类别权重: %s, class_weight_power_config=%s, resolved=%.4f",
            class_weights,
            class_weight_power_config,
            float(getattr(train_loader.dataset, "resolved_class_weight_power", 1.0)),
        )
    class_counts_for_prior = compute_class_counts(train_loader.dataset, num_classes).to(device)
    if hasattr(model, "set_class_counts"):
        model.set_class_counts(class_counts_for_prior)
        logger.info(f"类别样本数: {class_counts_for_prior}")

    prior_bias_config = config.get("model", {}).get("class_logit_bias_init")
    if isinstance(prior_bias_config, str) and prior_bias_config.lower() == "train_prior":
        with torch.no_grad():
            prior = class_counts_for_prior.float() / class_counts_for_prior.sum().clamp_min(1.0)
            prior_bias = prior.clamp_min(1e-8).log()
            prior_bias = prior_bias - prior_bias.mean()
            prior_bias_scale = resolve_prior_bias_scale(
                num_classes, config.get("model", {}).get("class_logit_bias_scale", 1.0)
            )
            model.class_logit_bias.copy_(prior_bias_scale * prior_bias.to(model.class_logit_bias.device))
        logger.info(
            "使用训练集先验初始化class_logit_bias: scale=%.4f, bias=%s",
            prior_bias_scale,
            model.class_logit_bias.detach().cpu(),
        )

    prediction_distribution_target = None
    prediction_distribution_target_config = str(
        training_config.get("prediction_distribution_target", "uniform")
    ).lower()
    if prediction_distribution_target_config == "train_prior":
        prediction_distribution_target = class_counts_for_prior.float() / class_counts_for_prior.sum().clamp_min(1.0)
        logger.info(f"预测分布正则目标: train_prior={prediction_distribution_target}")
    else:
        logger.info("预测分布正则目标: uniform")

    init_checkpoint_path = training_config.get("init_from_checkpoint")
    if init_checkpoint_path:
        checkpoint = torch.load(init_checkpoint_path, map_location=device, weights_only=False)
        init_state = checkpoint.get("model_state_dict", checkpoint)
        init_clone_query_layers = training_config.get("init_clone_query_layers", {})
        if init_clone_query_layers:
            if not isinstance(init_clone_query_layers, dict):
                raise ValueError("training.init_clone_query_layers must map target layers to source layers")
            init_state = dict(init_state)
            cloned_count = 0
            for target_layer, source_layer in init_clone_query_layers.items():
                target_index = int(target_layer) - 1
                source_index = int(source_layer) - 1
                if target_index < 0 or source_index < 0:
                    raise ValueError("training.init_clone_query_layers uses 1-based positive layer indices")
                source_prefix = f"query_layers.{source_index}."
                target_prefix = f"query_layers.{target_index}."
                source_items = [
                    (key, value) for key, value in init_state.items() if key.startswith(source_prefix)
                ]
                if not source_items:
                    raise ValueError(
                        f"No checkpoint parameters found for query layer {int(source_layer)}"
                    )
                for key, value in source_items:
                    init_state[target_prefix + key[len(source_prefix) :]] = value
                    cloned_count += 1
            logger.info(
                "初始化checkpoint复制QA层: mapping=%s, cloned=%d",
                init_clone_query_layers,
                cloned_count,
            )
        init_exclude_patterns = training_config.get("init_exclude_patterns", [])
        if isinstance(init_exclude_patterns, str):
            init_exclude_patterns = [init_exclude_patterns]
        if init_exclude_patterns:
            excluded_keys = [
                key
                for key in init_state
                if any(fnmatch.fnmatch(key, pattern) for pattern in init_exclude_patterns)
            ]
            init_state = {
                key: value for key, value in init_state.items() if key not in excluded_keys
            }
            logger.info(
                "初始化checkpoint按配置排除参数: patterns=%s, excluded=%d",
                init_exclude_patterns,
                len(excluded_keys),
            )
        if hasattr(model, "load_layerwise_checkpoint_state"):
            missing, unexpected, loaded_count = model.load_layerwise_checkpoint_state(init_state)
            logger.info(
                f"从Layerwise checkpoint初始化: {init_checkpoint_path}, "
                f"加载参数项={loaded_count}, missing={len(missing)}, unexpected={len(unexpected)}"
            )
        else:
            load_model_state_compatible(model, init_state, f"从checkpoint初始化: {init_checkpoint_path}")

    init_teacher_model = None
    init_logit_distill_weight = float(training_config.get("init_logit_distill_weight", 0.0) or 0.0)
    init_logit_distill_temperature = float(training_config.get("init_logit_distill_temperature", 2.0) or 2.0)
    init_logit_distill_min_confidence = float(
        training_config.get("init_logit_distill_min_confidence", 0.0) or 0.0
    )
    if init_logit_distill_weight > 0.0:
        if not init_checkpoint_path:
            logger.warning("配置了init_logit_distill_weight，但没有init_from_checkpoint，已跳过初始模型蒸馏")
            init_logit_distill_weight = 0.0
        else:
            teacher_config = copy.deepcopy(config)
            if bool(training_config.get("init_logit_distill_disable_sp_sh_query", True)):
                teacher_model_config = teacher_config.setdefault("model", {})
                teacher_model_config["use_sp_sh_query_update"] = False
                teacher_model_config["use_sidebranch_sp_sh_query_residual"] = False
                teacher_model_config["samplewise_sidebranch_sp_sh_query_gate"] = False
            init_teacher_model = build_query_model(teacher_config).to(device)
            if config.get("training", {}).get("class_weights", True):
                init_teacher_model.set_class_weights(class_weights)
            if hasattr(init_teacher_model, "set_class_counts"):
                init_teacher_model.set_class_counts(class_counts_for_prior)
            teacher_checkpoint = torch.load(init_checkpoint_path, map_location=device, weights_only=False)
            teacher_state = teacher_checkpoint.get("model_state_dict", teacher_checkpoint)
            load_model_state_compatible(init_teacher_model, teacher_state, "初始logit蒸馏teacher")
            init_teacher_model.eval()
            for teacher_param in init_teacher_model.parameters():
                teacher_param.requires_grad_(False)
            logger.info(
                "启用初始模型logit蒸馏: weight=%.4f, temperature=%.3f, disable_sp_sh_query=%s",
                init_logit_distill_weight,
                init_logit_distill_temperature,
                bool(training_config.get("init_logit_distill_disable_sp_sh_query", True)),
            )

    if training_config.get("freeze_pretrained_base", False):
        if hasattr(model, "freeze_pretrained_base"):
            trainable_count = model.freeze_pretrained_base()
        else:
            trainable_count = 0
            for param in model.parameters():
                param.requires_grad = False
        logger.info(f"冻结预训练主干，仅训练新增交互参数，可训练参数量: {trainable_count}")

    resolved_lr = resolve_learning_rate(num_classes, training_config["lr"])
    logger.info("学习率解析: lr_config=%s -> %.6g", training_config["lr"], resolved_lr)
    base_weight_decay = training_config.get("weight_decay", 0.01)
    loss_lr_multiplier = float(
        training_config.get(
            "learnable_loss_lr_multiplier",
            training_config.get("learnable_loss_param_lr_multiplier", 1.0),
        )
        or 1.0
    )

    active_stage_index, active_stage = _resolve_trainable_stage(training_config, 1)
    if active_stage is not None:
        _apply_trainable_stage(model, active_stage, logger)

    def _build_optimizer_for_current_trainable():
        named_trainable_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
        trainable_params = [param for _, param in named_trainable_params]
        trainable_param_count = sum(param.numel() for param in trainable_params)
        logger.info(f"QA可训练参数量: {trainable_param_count}")
        if not trainable_params:
            raise ValueError("没有可训练参数，请检查 trainable_stages / freeze 配置")
        if loss_lr_multiplier != 1.0:
            loss_weight_params = [
                param for name, param in named_trainable_params
                if "learnable_loss_" in name
            ]
            loss_weight_param_ids = {id(param) for param in loss_weight_params}
            base_params = [
                param for _, param in named_trainable_params
                if id(param) not in loss_weight_param_ids
            ]
            optimizer_param_groups = []
            if base_params:
                optimizer_param_groups.append({
                    "params": base_params,
                    "lr": resolved_lr,
                    "weight_decay": base_weight_decay,
                })
            if loss_weight_params:
                loss_lr = resolved_lr * loss_lr_multiplier
                optimizer_param_groups.append({
                    "params": loss_weight_params,
                    "lr": loss_lr,
                    "weight_decay": 0.0,
                })
                logger.info(
                    "可学习loss权重参数使用独立学习率: params=%d, lr=%.6g, multiplier=%.3g",
                    sum(param.numel() for param in loss_weight_params),
                    loss_lr,
                    loss_lr_multiplier,
                )
            else:
                logger.info("配置了learnable_loss_lr_multiplier，但未找到可学习loss权重参数")
                optimizer_param_groups.append({
                    "params": trainable_params,
                    "lr": resolved_lr,
                    "weight_decay": base_weight_decay,
                })
            return optim.AdamW(optimizer_param_groups)
        return optim.AdamW(
            trainable_params,
            lr=resolved_lr,
            weight_decay=base_weight_decay,
        )

    optimizer = _build_optimizer_for_current_trainable()
    epochs, resolved_training = resolve_auto_train_epochs(train_loader, training_config)
    train_num_samples = len(train_loader.dataset) if hasattr(train_loader, 'dataset') else None
    if hasattr(model, 'set_embedding_augmentation_epoch') and train_num_samples is not None:
        model.set_embedding_augmentation_epoch(
            0,
            total_epochs=resolved_training.get('resolved_epochs', epochs),
            num_train_samples=train_num_samples,
        )
    resolved_training.update({
        "sampling_alpha_config": config.get("data", {}).get("sampling_alpha", 1.0),
        "resolved_sampling_alpha": float(getattr(train_loader.dataset, "resolved_sampling_alpha", -1.0)),
        "class_weight_power_config": training_config.get("class_weight_power", 1.0),
        "resolved_class_weight_power": float(getattr(train_loader.dataset, "resolved_class_weight_power", -1.0)),
        "lr_config": training_config.get("lr"),
        "resolved_lr": resolved_lr,
    })
    config["resolved_training"] = resolved_training
    logger.info(
        "自动训练步数解析: enabled=%s, epochs %s -> %s, steps/epoch=%s, target_update_steps=%s, scheduler_epochs=%s",
        resolved_training["enabled"],
        resolved_training["configured_epochs"],
        resolved_training["resolved_epochs"],
        resolved_training["steps_per_epoch"],
        resolved_training["target_update_steps"],
        resolved_training.get("scheduler_epochs", epochs),
    )
    logger.info(
        "自动不平衡控制解析: sampling_alpha=%s -> %.4f, class_weight_power=%s -> %.4f",
        resolved_training["sampling_alpha_config"],
        resolved_training["resolved_sampling_alpha"],
        resolved_training["class_weight_power_config"],
        resolved_training["resolved_class_weight_power"],
    )
    with open(os.path.join(exp_dir, "config_resolved.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    scheduler_epochs = int(resolved_training.get("scheduler_epochs", epochs))
    scheduler = build_scheduler(optimizer, config, scheduler_epochs)
    ema_decay = float(training_config.get("ema_decay", 0.0) or 0.0)
    ema_start_epoch = int(training_config.get("ema_start_epoch", 1))
    use_ema_for_eval = bool(training_config.get("use_ema_for_eval", ema_decay > 0.0))
    ema = ModelEMA(model, ema_decay) if ema_decay > 0.0 else None
    if ema is not None:
        logger.info(
            "启用EMA权重平均: decay=%.5f, start_epoch=%d, use_ema_for_eval=%s",
            ema_decay,
            ema_start_epoch,
            use_ema_for_eval,
        )
    topk_swa_enabled = bool(training_config.get("topk_swa_enabled", False))
    topk_swa_k = int(training_config.get("topk_swa_k", 3))
    topk_swa_start_epoch = int(training_config.get("topk_swa_start_epoch", 1))
    topk_swa_min_prediction_coverage = float(
        training_config.get("topk_swa_min_prediction_coverage", 0.0) or 0.0
    )
    topk_swa_dir = os.path.join(exp_dir, "topk_swa_candidates")
    topk_swa_records: List[Dict[str, object]] = []
    if topk_swa_enabled:
        os.makedirs(topk_swa_dir, exist_ok=True)
        logger.info(
            "启用Top-K SWA单模型权重平均: k=%d, start_epoch=%d, min_prediction_coverage=%.4f",
            topk_swa_k,
            topk_swa_start_epoch,
            topk_swa_min_prediction_coverage,
        )
    resolved_training.update({
        "ema_decay": ema_decay,
        "ema_start_epoch": ema_start_epoch if ema is not None else None,
        "use_ema_for_eval": use_ema_for_eval if ema is not None else False,
        "topk_swa_enabled": topk_swa_enabled,
        "topk_swa_k": topk_swa_k if topk_swa_enabled else None,
        "topk_swa_start_epoch": topk_swa_start_epoch if topk_swa_enabled else None,
        "topk_swa_min_prediction_coverage": topk_swa_min_prediction_coverage if topk_swa_enabled else None,
    })

    def _evaluate_for_selection():
        if ema is not None and use_ema_for_eval:
            ema.store(model)
            ema.copy_to(model)
            try:
                return evaluate(model, dev_loader, device, num_classes, decision_threshold, inference_logit_adjustment_tau)
            finally:
                ema.restore(model)
        return evaluate(model, dev_loader, device, num_classes, decision_threshold, inference_logit_adjustment_tau)

    def _checkpoint_state_dict():
        if ema is not None and use_ema_for_eval:
            return ema.state_dict()
        return {key: value.detach().clone() for key, value in model.state_dict().items()}

    metric_for_best = training_config.get("metric_for_best", "f1_macro")
    early_stopping_patience = int(training_config.get("early_stopping_patience", 0))
    epochs_without_improvement = 0

    history = {key: [] for key in [
        "train_loss", "train_acc", "train_f1_macro", "train_f1_weighted",
        "val_loss", "val_acc", "val_f1_macro", "val_f1_weighted",
    ]}
    def _selection_score(metrics):
        if metric_for_best == "acc_then_macro":
            return (float(metrics["acc"]), float(metrics["f1_macro"]))
        if metric_for_best == "acc_macro_score":
            acc = float(metrics.get("acc", 0.0))
            f1_macro = float(metrics.get("f1_macro", 0.0))
            f1_weighted = float(metrics.get("f1_weighted", 0.0))
            return (acc + 0.25 * f1_macro + 0.10 * f1_weighted, acc, f1_macro, f1_weighted)
        if metric_for_best == "coverage_balanced_score":
            return (
                float(metrics.get("coverage_balanced_score", 0.0)),
                float(metrics.get("acc", 0.0)),
                float(metrics.get("f1_macro", 0.0)),
            )
        if metric_for_best in {"positive_f1_then_macro", "depressed_f1_then_macro"}:
            return (
                float(metrics.get("positive_f1", 0.0)),
                float(metrics.get("positive_recall", 0.0)),
                float(metrics.get("f1_macro", 0.0)),
                float(metrics.get("acc", 0.0)),
            )
        if metric_for_best in {"macro_posf1_score", "macro_depressed_f1_score"}:
            f1_macro = float(metrics.get("f1_macro", 0.0))
            positive_f1 = float(metrics.get("positive_f1", 0.0))
            acc = float(metrics.get("acc", 0.0))
            return (f1_macro + 0.25 * positive_f1, f1_macro, positive_f1, acc)
        if metric_for_best in {"positive_macro_hmean", "depressed_macro_hmean"}:
            positive_f1 = float(metrics.get("positive_f1", 0.0))
            f1_macro = float(metrics.get("f1_macro", 0.0))
            hmean = 0.0 if positive_f1 + f1_macro <= 0.0 else (
                2.0 * positive_f1 * f1_macro / (positive_f1 + f1_macro)
            )
            return (
                hmean,
                positive_f1,
                f1_macro,
                float(metrics.get("positive_recall", 0.0)),
                float(metrics.get("acc", 0.0)),
            )
        return (float(metrics.get(metric_for_best, metrics["f1_macro"])),)

    def _score_is_finite(score: Tuple[float, ...]) -> bool:
        return all(np.isfinite(value) for value in score)

    def _update_topk_swa_records(epoch_index: int, metrics: Dict[str, object], score: Tuple[float, ...]) -> None:
        if not topk_swa_enabled:
            return
        epoch_number = epoch_index + 1
        if epoch_number < topk_swa_start_epoch:
            return
        if not _score_is_finite(score):
            return
        prediction_coverage = float(metrics.get("prediction_coverage", 1.0))
        if prediction_coverage + 1e-12 < topk_swa_min_prediction_coverage:
            logger.info(
                "跳过Top-K SWA候选: epoch=%d, prediction_coverage=%.4f < %.4f",
                epoch_number,
                prediction_coverage,
                topk_swa_min_prediction_coverage,
            )
            return

        candidate_path = os.path.join(topk_swa_dir, f"epoch_{epoch_number:03d}.pth")
        torch.save(
            {
                "epoch": epoch_index,
                "model_state_dict": _checkpoint_state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": {k: v for k, v in metrics.items() if k not in {"predictions", "labels", "subject_ids"}},
                "selection_score": list(score),
                "config": config,
            },
            candidate_path,
        )
        topk_swa_records.append({"epoch": epoch_index, "score": list(score), "path": candidate_path})
        topk_swa_records.sort(key=lambda item: tuple(item["score"]), reverse=True)
        while len(topk_swa_records) > max(1, topk_swa_k):
            removed = topk_swa_records.pop()
            removed_path = str(removed["path"])
            if os.path.exists(removed_path):
                os.remove(removed_path)
        logger.info(
            "Top-K SWA候选: epoch=%d, score=%s, 当前TopK=%s",
            epoch_number,
            score,
            [(int(item["epoch"]) + 1, item["score"]) for item in topk_swa_records],
        )

    best_metric = (
        (-1.0, -1.0) if metric_for_best == "acc_then_macro"
        else (-1.0, -1.0, -1.0, -1.0) if metric_for_best == "acc_macro_score"
        else (-1.0, -1.0, -1.0) if metric_for_best == "coverage_balanced_score"
        else (-1.0, -1.0, -1.0, -1.0) if metric_for_best in {"positive_f1_then_macro", "depressed_f1_then_macro"}
        else (-1.0, -1.0, -1.0, -1.0) if metric_for_best in {"macro_posf1_score", "macro_depressed_f1_score"}
        else (-1.0, -1.0, -1.0, -1.0, -1.0) if metric_for_best in {"positive_macro_hmean", "depressed_macro_hmean"}
        else (-1.0,)
    )
    best_model_path = os.path.join(exp_dir, "best_model.pth")
    save_initial_as_best = bool(training_config.get("save_initial_as_best", True))

    if training_config.get("evaluate_before_training", False):
        initial_metrics = _evaluate_for_selection()
        initial_metric = _selection_score(initial_metrics)
        logger.info(
            f"初始dev指标: Acc: {initial_metrics['acc']:.4f}, "
            f"Macro F1: {initial_metrics['f1_macro']:.4f}, Weighted F1: {initial_metrics['f1_weighted']:.4f}"
        )
        if save_initial_as_best:
            best_metric = initial_metric
            torch.save(
                {
                    "epoch": -1,
                    "model_state_dict": _checkpoint_state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": {k: v for k, v in initial_metrics.items() if k not in {"predictions", "labels", "subject_ids"}},
                    "config": config,
                },
                best_model_path,
            )
            logger.info(f"保存初始QA模型作为当前最佳，{metric_for_best}: {best_metric}")
        else:
            logger.info(f"仅记录初始QA模型，不参与best选择，{metric_for_best}: {initial_metric}")

    for epoch in range(epochs):
        if hasattr(model, 'set_embedding_augmentation_epoch'):
            model.set_embedding_augmentation_epoch(
                epoch,
                total_epochs=resolved_training.get('resolved_epochs', epochs),
                num_train_samples=train_num_samples,
            )
        stage_index, stage_config = _resolve_trainable_stage(training_config, epoch + 1)
        if stage_config is not None and stage_index != active_stage_index:
            active_stage_index = stage_index
            _apply_trainable_stage(model, stage_config, logger)
            optimizer = _build_optimizer_for_current_trainable()
            remaining_scheduler_epochs = max(1, scheduler_epochs - epoch)
            scheduler = build_scheduler(optimizer, config, remaining_scheduler_epochs)
            logger.info(
                "训练阶段切换到 %s (epoch=%d)，已重建optimizer/scheduler，remaining_scheduler_epochs=%d",
                stage_config.get("name", stage_index),
                epoch + 1,
                remaining_scheduler_epochs,
            )
        logger.info(f"QA Epoch {epoch + 1}/{epochs}")
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            config["training"].get("gradient_clip_val", 1.0),
            num_classes,
            float(config["training"].get("prediction_distribution_loss_weight", 0.0) or 0.0),
            prediction_distribution_target,
            ema if (ema is not None and (epoch + 1) >= ema_start_epoch) else None,
            epoch,
            init_teacher_model,
            init_logit_distill_weight,
            init_logit_distill_temperature,
            init_logit_distill_min_confidence,
            str(training_config.get("init_logit_distill_label_filter", "all")),
        )
        val_metrics = _evaluate_for_selection()
        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["train_f1_macro"].append(train_metrics["f1_macro"])
        history["train_f1_weighted"].append(train_metrics["f1_weighted"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])
        history["val_f1_weighted"].append(val_metrics["f1_weighted"])
        for metric_name, metric_value in train_metrics.items():
            if is_loggable_loss_scalar(metric_name):
                history.setdefault(f"train_{metric_name}", []).append(metric_value)
        for metric_name, metric_value in val_metrics.items():
            if is_loggable_loss_scalar(metric_name):
                history.setdefault(f"val_{metric_name}", []).append(metric_value)

        logger.info(
            f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}, "
            f"Macro F1: {train_metrics['f1_macro']:.4f}"
        )
        loss_metric_items = {
            key: value for key, value in train_metrics.items() if key.startswith("loss_")
        }
        if loss_metric_items:
            logger.info(
                "Train loss components: %s",
                ", ".join(f"{key}={value:.4f}" for key, value in sorted(loss_metric_items.items())),
            )
        cross_metric_items = {
            key: value for key, value in train_metrics.items() if key.startswith("cross_sample_")
        }
        if cross_metric_items:
            logger.info(
                "Cross-sample train diagnostics: %s",
                ", ".join(f"{key}={value:.4f}" for key, value in sorted(cross_metric_items.items())),
            )
        if train_metrics.get("skipped_batches", 0) > 0:
            logger.warning("QA训练本epoch跳过非有限batch数: %s", train_metrics["skipped_batches"])
        logger.info(
            f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}, "
            f"Macro F1: {val_metrics['f1_macro']:.4f}, Weighted F1: {val_metrics['f1_weighted']:.4f}"
        )
        logger.info(f"Val Confusion Matrix: {val_metrics['confusion_matrix']}")

        current_metric = _selection_score(val_metrics)
        _update_topk_swa_records(epoch, val_metrics, current_metric)
        if current_metric > best_metric:
            best_metric = current_metric
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": _checkpoint_state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": {k: v for k, v in val_metrics.items() if k not in {"predictions", "labels", "subject_ids"}},
                    "config": config,
                },
                best_model_path,
            )
            logger.info(f"保存最佳QA模型，{metric_for_best}: {best_metric}")
        else:
            epochs_without_improvement += 1
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                logger.info(
                    f"Early stopping triggered: {metric_for_best} has not improved for "
                    f"{early_stopping_patience} epochs. Best={best_metric}"
                )
                break

    if bool(training_config.get("save_last_model", False)):
        last_model_path = os.path.join(exp_dir, "last_model.pth")
        torch.save(
            {
                "epoch": len(history["train_loss"]) - 1,
                "model_state_dict": _checkpoint_state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": {
                    k: v for k, v in val_metrics.items()
                    if k not in {"predictions", "labels", "subject_ids"}
                },
                "config": config,
                "checkpoint_role": "fixed_training_endpoint",
            },
            last_model_path,
        )
        resolved_training["last_model_path"] = last_model_path
        resolved_training["last_model_epoch"] = len(history["train_loss"])
        logger.info("Saved fixed-endpoint QA checkpoint: %s", last_model_path)

    best_model_path = os.path.join(exp_dir, "best_model.pth")
    if topk_swa_enabled and topk_swa_records:
        topk_swa_paths = [str(item["path"]) for item in topk_swa_records]
        topk_swa_state = average_checkpoint_state_dicts(topk_swa_paths)
        topk_swa_metadata = {
            "records": [
                {"epoch": int(item["epoch"]) + 1, "score": item["score"], "path": str(item["path"])}
                for item in topk_swa_records
            ],
            "k": topk_swa_k,
            "metric_for_best": metric_for_best,
        }
        torch.save(
            {
                "epoch": "topk_swa",
                "model_state_dict": topk_swa_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": {"topk_swa_metadata": topk_swa_metadata},
                "topk_swa_metadata": topk_swa_metadata,
                "config": config,
            },
            best_model_path,
        )
        resolved_training["topk_swa_records"] = topk_swa_metadata["records"]
        logger.info("已用Top-K SWA平均候选覆盖最终best_model.pth: %s", topk_swa_metadata["records"])

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    load_model_state_compatible(model, checkpoint["model_state_dict"], "最佳QA模型")
    final_metrics = evaluate(
        model, dev_loader, device, num_classes, decision_threshold, inference_logit_adjustment_tau
    )

    final_metrics["resolved_training"] = resolved_training
    with open(os.path.join(exp_dir, "dev_results.json"), "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=4, ensure_ascii=False)
    if evaluate_independent_test and test_loader is not None and test_loader is not dev_loader:
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            num_classes,
            decision_threshold,
            inference_logit_adjustment_tau,
        )
        test_metrics["resolved_training"] = resolved_training
        with open(os.path.join(exp_dir, "test_results.json"), "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=4, ensure_ascii=False)
        logger.info(
            f"QA independent test指标: Acc: {test_metrics['acc']:.4f}, "
            f"F1 Macro: {test_metrics['f1_macro']:.4f}, "
            f"F1 Weighted: {test_metrics['f1_weighted']:.4f}"
        )
    elif test_loader is not None and test_loader is not dev_loader:
        logger.info("已按配置跳过独立测试集评估；当前阶段只使用验证集")
    with open(os.path.join(exp_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
    with open(os.path.join(exp_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"resolved_training": resolved_training}, f, indent=4, ensure_ascii=False)

    logger.info(
        f"QA dev指标: Acc: {final_metrics['acc']:.4f}, "
        f"F1 Macro: {final_metrics['f1_macro']:.4f}, F1 Weighted: {final_metrics['f1_weighted']:.4f}"
    )
    return model, history


def evaluate_query_checkpoint(config_path: str, checkpoint_path: str, split: str = "dev") -> Dict[str, object]:
    config = load_config(config_path)
    set_seed(config["random_seed"])
    device = get_device()
    dataloaders = get_query_dataloaders(config)
    if split not in dataloaders:
        split = "dev" if "dev" in dataloaders else next(iter(dataloaders))
        logger.warning(f"未找到请求划分，改用{split}划分评估")

    model = build_query_model(config).to(device)
    num_classes = int(config["model"].get("output", {}).get("num_classes", 2))
    if config.get("training", {}).get("class_weights", True):
        class_weight_power = config.get("training", {}).get("class_weight_power", 1.0)
        model.set_class_weights(
            compute_class_weights(dataloaders["train"].dataset, num_classes, class_weight_power).to(device)
        )
    if hasattr(model, "set_class_counts"):
        model.set_class_counts(compute_class_counts(dataloaders["train"].dataset, num_classes).to(device))

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    load_model_state_compatible(model, checkpoint["model_state_dict"], "QA评估模型")
    evaluation_config = config.get("evaluation", {})
    decision_threshold = evaluation_config.get("decision_threshold")
    if decision_threshold is not None:
        decision_threshold = float(decision_threshold)
        logger.info(f"使用二分类校准阈值: depressed_prob >= {decision_threshold:.4f}")
    inference_logit_adjustment_tau = float(
        evaluation_config.get(
            "inference_logit_adjustment_tau",
            config.get("training", {}).get("inference_logit_adjustment_tau", 0.0),
        )
        or 0.0
    )
    if abs(inference_logit_adjustment_tau) > 1e-12:
        logger.info(f"使用统一训练先验logit校准: tau={inference_logit_adjustment_tau:.4f}")
    metrics = evaluate(
        model, dataloaders[split], device, num_classes, decision_threshold, inference_logit_adjustment_tau
    )

    output_tags = []
    if decision_threshold is not None:
        output_tags.append(f"thr_{decision_threshold:.3f}".replace(".", "p"))
    if abs(inference_logit_adjustment_tau) > 1e-12:
        output_tags.append(f"prior_tau_{inference_logit_adjustment_tau:.3f}".replace("-", "m").replace(".", "p"))
    output_suffix = "_" + "_".join(output_tags) if output_tags else ""
    output_path = os.path.join(os.path.dirname(checkpoint_path), f"{split}_qa_evaluation{output_suffix}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    logger.info(
        f"QA {split}指标: Acc: {metrics['acc']:.4f}, "
        f"F1 Macro: {metrics['f1_macro']:.4f}, F1 Weighted: {metrics['f1_weighted']:.4f}"
    )
    logger.info(f"QA评估结果已保存至: {output_path}")
    return metrics
