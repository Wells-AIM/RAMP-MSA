"""Losses for RefQFormer-style audio-text emotion recognition."""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_bool(config: Dict[str, object], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def build_refqformer_loss_outputs(
    logits: torch.Tensor,
    logits_dict: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Build the optional dict-style forward payload without changing old callers."""
    outputs: Dict[str, torch.Tensor] = {}
    if logits_dict:
        outputs.update(logits_dict)
    outputs["logits"] = logits

    if "final_queries" not in outputs and "queries" in outputs:
        outputs["final_queries"] = outputs["queries"]

    layer_logits = outputs.get("layer_logits")
    # Active-layer models bind auxiliary logits to physical QA layer numbers.
    # A missing key therefore means that layer was skipped, not that positional
    # fallback should relabel a later active layer as QA2 or QA3.
    has_active_layer_identity = "active_query_layer_mask" in outputs
    if layer_logits is not None and not has_active_layer_identity:
        if "aux_logits_q2" not in outputs:
            aux_q2 = _layer_logits_at(layer_logits, 1)
            if aux_q2 is not None:
                outputs["aux_logits_q2"] = aux_q2
        if "aux_logits_q3" not in outputs:
            aux_q3 = _layer_logits_at(layer_logits, 2)
            if aux_q3 is not None:
                outputs["aux_logits_q3"] = aux_q3

    alias_pairs = {
        "text_sp": "text_specific",
        "text_sh": "text_shared",
        "audio_sp": "audio_specific",
        "audio_sh": "audio_shared",
    }
    for target, source in alias_pairs.items():
        if target not in outputs and source in outputs:
            outputs[target] = outputs[source]
    return outputs


def _layer_logits_at(layer_logits, index: int) -> Optional[torch.Tensor]:
    if torch.is_tensor(layer_logits):
        if layer_logits.dim() >= 3 and layer_logits.size(0) > index:
            return layer_logits[index]
        return None
    if isinstance(layer_logits, (list, tuple)) and len(layer_logits) > index:
        value = layer_logits[index]
        return value if torch.is_tensor(value) else None
    return None


class RefQFormerLoss(nn.Module):
    """Composite loss for QA + SP/SH experiments.

    The module is intentionally standalone so existing models can opt in by
    delegating from their current ``calculate_losses`` method.
    """

    LOSS_KEYS = (
        "loss_total",
        "loss_main",
        "loss_aux_q2",
        "loss_aux_q3",
        "loss_sp_sh",
        "loss_sh_align",
        "loss_orth",
        "loss_qdiv",
        "loss_supcon",
        "loss_recon",
        "loss_recon_text",
        "loss_recon_audio",
    )

    def __init__(self, cfg: Dict[str, object]):
        super().__init__()
        model_cfg = cfg.get("model", {})
        data_cfg = cfg.get("data", {})
        train_cfg = cfg.get("training", {})
        loss_cfg = cfg.get("loss", {})
        output_cfg = model_cfg.get("output", {}) if isinstance(model_cfg.get("output", {}), dict) else {}

        self.num_classes = int(
            loss_cfg.get("num_classes", output_cfg.get("num_classes", data_cfg.get("num_classes", 2)))
        )
        self.queries_per_class = int(
            loss_cfg.get(
                "queries_per_class",
                model_cfg.get("num_queries_per_class", model_cfg.get("queries_per_class", 1)),
            )
        )
        self.focal_gamma = float(loss_cfg.get("focal_gamma", 2.0))
        self.focal_pt_mode = str(loss_cfg.get("focal_pt_mode", "legacy_weighted_ce")).lower()
        self.label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
        self.use_class_weights = _as_bool(loss_cfg, "use_class_weights", bool(train_cfg.get("class_weights", True)))

        self.lambda_aux_q2 = float(loss_cfg.get("lambda_aux_q2", 0.2))
        self.lambda_aux_q3 = float(loss_cfg.get("lambda_aux_q3", 0.3))
        self.lambda_sp_sh = float(loss_cfg.get("lambda_sp_sh", 0.05))
        self.lambda_sh_align_inside_sp_sh = float(
            loss_cfg.get("lambda_sh_align_inside_sp_sh", 1.0)
        )
        self.lambda_orth_inside_sp_sh = float(loss_cfg.get("lambda_orth_inside_sp_sh", 0.5))
        self.lambda_qdiv = float(loss_cfg.get("lambda_qdiv", 0.03))
        self.lambda_supcon = float(loss_cfg.get("lambda_supcon", 0.05))
        self.lambda_recon = float(loss_cfg.get("lambda_recon", loss_cfg.get("lambda_reconstruction", 0.0)))
        self.supcon_temperature = float(loss_cfg.get("supcon_temperature", 0.1))

        self.use_aux_loss = _as_bool(loss_cfg, "use_aux_loss", True)
        self.use_sp_sh_loss = _as_bool(loss_cfg, "use_sp_sh_loss", True)
        self.use_query_diversity_loss = _as_bool(loss_cfg, "use_query_diversity_loss", True)
        self.use_supcon_loss = _as_bool(loss_cfg, "use_supcon_loss", True)
        self.use_reconstruction_loss = _as_bool(
            loss_cfg,
            "use_reconstruction_loss",
            self.lambda_recon > 0.0,
        )

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logits = outputs["logits"]
        labels = labels.to(device=logits.device).long().view(-1)
        class_weights = self._class_weights(class_weights, logits)
        zero = logits.sum() * 0.0

        loss_main = self._focal_ce(logits, labels, class_weights)

        loss_aux_q2 = zero
        if self.use_aux_loss and self.lambda_aux_q2 > 0.0:
            aux_logits = outputs.get("aux_logits_q2")
            if aux_logits is not None:
                loss_aux_q2 = self._focal_ce(aux_logits, labels, class_weights)

        loss_aux_q3 = zero
        if self.use_aux_loss and self.lambda_aux_q3 > 0.0:
            aux_logits = outputs.get("aux_logits_q3")
            if aux_logits is not None:
                loss_aux_q3 = self._focal_ce(aux_logits, labels, class_weights)

        loss_sp_sh = zero
        loss_sh_align = zero
        loss_orth = zero
        if self.use_sp_sh_loss and self.lambda_sp_sh > 0.0:
            loss_sp_sh, loss_sh_align, loss_orth = self._sp_sh_loss(outputs, zero)

        loss_qdiv = zero
        if self.use_query_diversity_loss and self.lambda_qdiv > 0.0:
            loss_qdiv = self._query_diversity_loss(outputs.get("final_queries", outputs.get("queries")), zero)

        loss_supcon = zero
        if self.use_supcon_loss and self.lambda_supcon > 0.0:
            loss_supcon = self._supervised_contrastive_loss(outputs.get("class_reps"), labels, zero)

        loss_recon = zero
        loss_recon_text = zero
        loss_recon_audio = zero
        if self.use_reconstruction_loss and self.lambda_recon > 0.0:
            loss_recon = self._scalar_loss_from_outputs(
                outputs,
                zero,
                "loss_recon",
                "reconstruction_loss",
            )
            loss_recon_text = self._scalar_loss_from_outputs(
                outputs,
                zero,
                "loss_recon_text",
                "reconstruction_text_loss",
            )
            loss_recon_audio = self._scalar_loss_from_outputs(
                outputs,
                zero,
                "loss_recon_audio",
                "reconstruction_audio_loss",
            )

        total = (
            loss_main
            + self.lambda_aux_q2 * loss_aux_q2
            + self.lambda_aux_q3 * loss_aux_q3
            + self.lambda_sp_sh * loss_sp_sh
            + self.lambda_qdiv * loss_qdiv
            + self.lambda_supcon * loss_supcon
            + self.lambda_recon * loss_recon
        )
        total = torch.nan_to_num(total, nan=0.0, posinf=1e4, neginf=-1e4)

        loss_dict = {
            "loss_total": total,
            "total_loss": total,
            "loss_main": loss_main,
            "main_loss": loss_main,
            "loss_aux_q2": loss_aux_q2,
            "loss_aux_q3": loss_aux_q3,
            "loss_sp_sh": loss_sp_sh,
            "loss_sh_align": loss_sh_align,
            "loss_orth": loss_orth,
            "loss_qdiv": loss_qdiv,
            "loss_supcon": loss_supcon,
            "loss_recon": loss_recon,
            "loss_recon_text": loss_recon_text,
            "loss_recon_audio": loss_recon_audio,
            "reconstruction_loss": loss_recon,
            "reconstruction_text_loss": loss_recon_text,
            "reconstruction_audio_loss": loss_recon_audio,
        }
        return total, loss_dict

    def _scalar_loss_from_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
        zero: torch.Tensor,
        *names: str,
    ) -> torch.Tensor:
        for name in names:
            value = outputs.get(name)
            if torch.is_tensor(value):
                if value.numel() == 0:
                    return zero
                if value.dim() > 0:
                    value = value.mean()
                return torch.nan_to_num(value, nan=0.0, posinf=1e4, neginf=-1e4)
        return zero

    def _class_weights(self, class_weights: Optional[torch.Tensor], logits: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.use_class_weights or class_weights is None or not torch.is_tensor(class_weights):
            return None
        if class_weights.numel() != max(int(logits.size(-1)), 1):
            return None
        return class_weights.to(device=logits.device, dtype=logits.dtype)

    def _focal_ce(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        class_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        if logits.dim() == 1 or (logits.dim() == 2 and logits.size(-1) == 1):
            return self._focal_bce(logits.view(-1), labels, class_weights)

        ce = F.cross_entropy(
            logits,
            labels,
            weight=class_weights,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        if self.focal_gamma > 0.0:
            if self.focal_pt_mode in {"true_class_probability", "standard", "true_prob", "softmax"}:
                pt = F.softmax(logits.detach(), dim=-1).gather(1, labels.view(-1, 1)).squeeze(1)
                pt = pt.clamp(min=1e-8, max=1.0)
            else:
                # Legacy behavior retained for exact reproducibility of previous runs.
                # Note: with class weights this is not the textbook focal p_t.
                pt = torch.exp(-ce.detach()).clamp(min=1e-8, max=1.0)
            ce = (1.0 - pt).pow(self.focal_gamma) * ce
        return torch.nan_to_num(ce.mean(), nan=0.0, posinf=1e4, neginf=-1e4)

    def _focal_bce(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        class_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        targets = labels.to(dtype=logits.dtype)
        if self.label_smoothing > 0.0:
            smooth = min(max(self.label_smoothing, 0.0), 1.0)
            targets = targets * (1.0 - smooth) + 0.5 * smooth
        pos_weight = None
        if class_weights is not None and class_weights.numel() >= 2:
            pos_weight = (class_weights[1] / class_weights[0].clamp_min(1e-8)).view(1)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
            reduction="none",
        )
        if self.focal_gamma > 0.0:
            if self.focal_pt_mode in {"true_class_probability", "standard", "true_prob", "sigmoid"}:
                hard_targets = labels.to(dtype=logits.dtype)
                prob = torch.sigmoid(logits.detach())
                pt = prob * hard_targets + (1.0 - prob) * (1.0 - hard_targets)
                pt = pt.clamp(min=1e-8, max=1.0)
            else:
                pt = torch.exp(-loss.detach()).clamp(min=1e-8, max=1.0)
            loss = (1.0 - pt).pow(self.focal_gamma) * loss
        return torch.nan_to_num(loss.mean(), nan=0.0, posinf=1e4, neginf=-1e4)

    def _feature(self, outputs: Dict[str, torch.Tensor], *names: str) -> Optional[torch.Tensor]:
        for name in names:
            value = outputs.get(name)
            if torch.is_tensor(value):
                return value
        return None

    def _masked_or_mean_pool(
        self,
        feature: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feature = torch.nan_to_num(feature, nan=0.0, posinf=1e4, neginf=-1e4)
        if feature.dim() <= 2:
            return feature
        if feature.dim() > 3:
            feature = feature.reshape(feature.size(0), -1, feature.size(-1))
        if mask is not None and torch.is_tensor(mask):
            mask = mask.to(device=feature.device).bool()
            if mask.dim() > 2:
                mask = mask.reshape(mask.size(0), -1)
            if mask.shape[:2] == feature.shape[:2]:
                weights = mask.to(dtype=feature.dtype).unsqueeze(-1)
                return (feature * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return feature.mean(dim=1)

    def _sp_sh_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        zero: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_sp = self._feature(outputs, "text_sp", "text_specific")
        text_sh = self._feature(outputs, "text_sh", "text_shared")
        audio_sp = self._feature(outputs, "audio_sp", "audio_specific")
        audio_sh = self._feature(outputs, "audio_sh", "audio_shared")
        if text_sp is None or text_sh is None or audio_sp is None or audio_sh is None:
            return zero, zero, zero

        text_mask = outputs.get("text_unit_mask")
        audio_mask = outputs.get("audio_unit_mask")
        text_sp = F.normalize(self._masked_or_mean_pool(text_sp, text_mask), dim=-1)
        text_sh = F.normalize(self._masked_or_mean_pool(text_sh, text_mask), dim=-1)
        audio_sp = F.normalize(self._masked_or_mean_pool(audio_sp, audio_mask), dim=-1)
        audio_sh = F.normalize(self._masked_or_mean_pool(audio_sh, audio_mask), dim=-1)

        loss_sh_align = (1.0 - F.cosine_similarity(text_sh, audio_sh, dim=-1)).mean()
        loss_orth = (
            F.cosine_similarity(text_sp, text_sh, dim=-1).pow(2).mean()
            + F.cosine_similarity(audio_sp, audio_sh, dim=-1).pow(2).mean()
        )
        loss_sp_sh = (
            self.lambda_sh_align_inside_sp_sh * loss_sh_align
            + self.lambda_orth_inside_sp_sh * loss_orth
        )
        return (
            torch.nan_to_num(loss_sp_sh, nan=0.0, posinf=1e4, neginf=-1e4),
            torch.nan_to_num(loss_sh_align, nan=0.0, posinf=1e4, neginf=-1e4),
            torch.nan_to_num(loss_orth, nan=0.0, posinf=1e4, neginf=-1e4),
        )

    def _query_diversity_loss(self, queries: Optional[torch.Tensor], zero: torch.Tensor) -> torch.Tensor:
        if queries is None or not torch.is_tensor(queries):
            return zero
        if self.queries_per_class <= 1:
            return zero
        if queries.dim() == 3:
            expected = self.num_classes * self.queries_per_class
            if queries.size(1) != expected:
                return zero
            queries = queries.reshape(queries.size(0), self.num_classes, self.queries_per_class, queries.size(-1))
        elif queries.dim() != 4:
            return zero
        if queries.size(2) <= 1:
            return zero
        q = F.normalize(torch.nan_to_num(queries, nan=0.0, posinf=1e4, neginf=-1e4), dim=-1)
        gram = torch.matmul(q, q.transpose(-1, -2))
        eye = torch.eye(q.size(2), device=q.device, dtype=q.dtype).view(1, 1, q.size(2), q.size(2))
        off_diag = gram - eye
        return torch.nan_to_num(off_diag.pow(2).mean(), nan=0.0, posinf=1e4, neginf=-1e4)

    def _supervised_contrastive_loss(
        self,
        class_reps: Optional[torch.Tensor],
        labels: torch.Tensor,
        zero: torch.Tensor,
    ) -> torch.Tensor:
        if class_reps is None or not torch.is_tensor(class_reps) or class_reps.dim() != 3:
            return zero
        batch_size = class_reps.size(0)
        if batch_size < 2:
            return zero
        valid_labels = (labels >= 0) & (labels < class_reps.size(1))
        if not valid_labels.all():
            return zero
        sample_ids = torch.arange(batch_size, device=class_reps.device)
        z = class_reps[sample_ids, labels]
        z = F.normalize(torch.nan_to_num(z, nan=0.0, posinf=1e4, neginf=-1e4), dim=-1)
        temperature = max(float(self.supcon_temperature), 1e-6)
        similarity = torch.matmul(z, z.transpose(0, 1)) / temperature
        self_mask = torch.eye(batch_size, device=z.device, dtype=torch.bool)
        positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
        valid = positive_mask.any(dim=1)
        if not valid.any():
            return zero
        similarity = torch.nan_to_num(similarity, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
        similarity = similarity.masked_fill(self_mask, -1e4)
        log_prob = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
        positive_count = positive_mask.to(dtype=z.dtype).sum(dim=1).clamp_min(1.0)
        per_anchor = -(log_prob * positive_mask.to(dtype=z.dtype)).sum(dim=1) / positive_count
        return torch.nan_to_num(per_anchor[valid].mean(), nan=0.0, posinf=1e4, neginf=-1e4)
