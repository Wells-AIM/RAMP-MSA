"""Batch-local cross-sample utilities for RefFormer audio-text emotion models."""
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossSampleProjector(nn.Module):
    """Project per-layer query/text/audio summaries into retrieval space."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.projector = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        q_rep: torch.Tensor,
        text_rep: torch.Tensor,
        audio_rep: torch.Tensor,
    ) -> torch.Tensor:
        z = self.projector(torch.cat([q_rep, text_rep, audio_rep], dim=-1))
        return F.normalize(z, dim=-1)



def _finite_cross_sample_tensor(tensor: torch.Tensor, limit: float = 50.0) -> torch.Tensor:
    return torch.nan_to_num(tensor, nan=0.0, posinf=limit, neginf=-limit).clamp(min=-limit, max=limit)


class BatchRelationEncoder(nn.Module):
    """BatchFormer-style sample relation encoder over the batch dimension."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, nodes: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        nodes = _finite_cross_sample_tensor(nodes)
        batch_size = nodes.size(0)
        zero = nodes.new_tensor(0.0)
        if batch_size <= 1:
            return nodes, {"attention_entropy": zero, "offdiag_attention_mass": zero}
        x = nodes.unsqueeze(0)
        attn_out, attn_weights = self.attention(
            x, x, x, need_weights=True, average_attn_weights=True
        )
        x = self.norm1(x + self.dropout(_finite_cross_sample_tensor(attn_out)))
        x = self.norm2(x + self.dropout(_finite_cross_sample_tensor(self.ffn(x))))
        relation_nodes = _finite_cross_sample_tensor(x.squeeze(0))
        weights = attn_weights.squeeze(0).clamp_min(1e-8)
        entropy = -(weights * weights.log()).sum(dim=-1).mean()
        entropy = entropy / nodes.new_tensor(float(batch_size)).log().clamp_min(1e-8)
        eye = torch.eye(batch_size, device=nodes.device, dtype=torch.bool)
        offdiag_mass = weights.masked_fill(eye, 0.0).sum(dim=-1).mean()
        return relation_nodes, {"attention_entropy": entropy, "offdiag_attention_mass": offdiag_mass}


def build_batch_relation_nodes(
    q_rep: torch.Tensor,
    text_rep: torch.Tensor,
    audio_rep: torch.Tensor,
    projector: Optional[nn.Module] = None,
) -> torch.Tensor:
    """Build unlabeled batch-relation nodes from query/text/audio summaries."""
    node_input = torch.cat([q_rep, text_rep, audio_rep], dim=-1)
    node = projector(node_input) if projector is not None else node_input
    return F.normalize(_finite_cross_sample_tensor(node), dim=-1)


def relation_auxiliary_ce_loss(
    relation_logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    logit_adjustment_tau: float = 0.0,
    class_log_prior: Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Auxiliary CE for train-only relation logits."""
    zero = labels.new_tensor(0.0, dtype=torch.float32)
    if relation_logits is None or relation_logits.numel() == 0 or labels.numel() == 0:
        return zero, {"aux_acc": zero, "aux_confidence": zero}
    logits = _finite_cross_sample_tensor(relation_logits, limit=1e4)
    if logit_adjustment_tau != 0.0 and class_log_prior is not None and class_log_prior.numel() == logits.size(-1):
        logits = logits + float(logit_adjustment_tau) * class_log_prior.to(device=logits.device, dtype=logits.dtype)
    weights = class_weights.to(device=logits.device, dtype=logits.dtype) if class_weights is not None and class_weights.numel() > 0 else None
    per_sample = F.cross_entropy(
        logits,
        labels,
        weight=weights,
        label_smoothing=float(label_smoothing),
        reduction="none",
    )
    if sample_weights is not None:
        sample_weights = _finite_cross_sample_tensor(sample_weights.to(device=logits.device, dtype=logits.dtype)).clamp_min(0.0)
        loss = (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
    else:
        loss = per_sample.mean()
    probs = torch.softmax(logits.detach(), dim=-1)
    return loss, {
        "aux_acc": logits.detach().argmax(dim=-1).eq(labels).to(dtype=logits.dtype).mean(),
        "aux_confidence": probs.max(dim=-1).values.mean(),
    }


def relation_aware_loss_weights(
    node: torch.Tensor,
    text_key: torch.Tensor,
    audio_key: torch.Tensor,
    labels: torch.Tensor,
    sample_ids=None,
    memory_node: torch.Tensor = None,
    memory_text_key: torch.Tensor = None,
    memory_audio_key: torch.Tensor = None,
    memory_labels: torch.Tensor = None,
    memory_sample_ids: torch.Tensor = None,
    exclude_same_sample: bool = True,
    min_weight: float = 0.90,
    max_weight: float = 1.15,
    risk_sim_threshold: float = 0.35,
    safe_margin: float = 0.15,
    temperature: float = 0.10,
    node_weight: float = 0.50,
    text_weight: float = 0.30,
    audio_weight: float = 0.20,
    normalize_mean: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute conservative per-sample CE weights from relation similarity."""
    batch_size = labels.numel()
    device = labels.device
    dtype = node.dtype
    one = node.new_tensor(1.0)
    if batch_size <= 1:
        weights = node.new_ones(batch_size)
        return weights, {
            "mean_weight": one,
            "min_weight": one,
            "max_weight": one,
            "avg_hard_negative_sim": node.new_tensor(0.0),
            "avg_same_label_sim": node.new_tensor(0.0),
            "high_risk_ratio": node.new_tensor(0.0),
        }
    node = F.normalize(_finite_cross_sample_tensor(node), dim=-1)
    text_key = F.normalize(_finite_cross_sample_tensor(text_key.to(device=device, dtype=dtype)), dim=-1)
    audio_key = F.normalize(_finite_cross_sample_tensor(audio_key.to(device=device, dtype=dtype)), dim=-1)
    labels = labels.view(-1).to(device=device, dtype=torch.long)
    has_memory = (
        memory_node is not None
        and memory_node.numel() > 0
        and memory_labels is not None
        and memory_labels.numel() == memory_node.size(0)
        and memory_text_key is not None
        and memory_audio_key is not None
    )
    if has_memory:
        memory_node = F.normalize(_finite_cross_sample_tensor(memory_node.detach().to(device=device, dtype=dtype)), dim=-1)
        memory_text_key = F.normalize(_finite_cross_sample_tensor(memory_text_key.detach().to(device=device, dtype=dtype)), dim=-1)
        memory_audio_key = F.normalize(_finite_cross_sample_tensor(memory_audio_key.detach().to(device=device, dtype=dtype)), dim=-1)
        memory_labels = memory_labels.detach().view(-1).to(device=device, dtype=torch.long)
    else:
        memory_node = node.new_empty(0, node.size(-1))
        memory_text_key = text_key.new_empty(0, text_key.size(-1))
        memory_audio_key = audio_key.new_empty(0, audio_key.size(-1))
        memory_labels = labels.new_empty(0)
    candidate_node = torch.cat([node, memory_node], dim=0)
    candidate_text = torch.cat([text_key, memory_text_key], dim=0)
    candidate_audio = torch.cat([audio_key, memory_audio_key], dim=0)
    candidate_labels = torch.cat([labels, memory_labels], dim=0)
    candidate_count = candidate_labels.numel()
    sim = (
        float(node_weight) * torch.matmul(node, candidate_node.transpose(0, 1))
        + float(text_weight) * torch.matmul(text_key, candidate_text.transpose(0, 1))
        + float(audio_weight) * torch.matmul(audio_key, candidate_audio.transpose(0, 1))
    )
    exclusion = torch.zeros(batch_size, candidate_count, device=device, dtype=torch.bool)
    exclusion[:, :batch_size] |= torch.eye(batch_size, device=device, dtype=torch.bool)
    if exclude_same_sample:
        exclusion[:, :batch_size] |= _same_sample_mask(sample_ids, batch_size, device)
        if memory_sample_ids is not None and memory_sample_ids.numel() == memory_labels.numel():
            if torch.is_tensor(sample_ids) and sample_ids.numel() == batch_size:
                anchor_ids = sample_ids.detach().view(-1).to(device=device, dtype=memory_sample_ids.dtype)
            elif isinstance(sample_ids, Sequence) and len(sample_ids) == batch_size:
                anchor_ids = torch.tensor([hash(str(item)) & 0x7FFFFFFFFFFFFFFF for item in sample_ids], device=device, dtype=memory_sample_ids.dtype)
            else:
                anchor_ids = torch.full((batch_size,), -1, device=device, dtype=memory_sample_ids.dtype)
            memory_sample_ids = memory_sample_ids.detach().view(-1).to(device=device, dtype=memory_sample_ids.dtype)
            valid_ids = anchor_ids[:, None].ge(0) & memory_sample_ids[None, :].ge(0)
            exclusion[:, batch_size:] |= valid_ids & anchor_ids[:, None].eq(memory_sample_ids[None, :])
    same_label = labels[:, None].eq(candidate_labels[None, :]) & ~exclusion
    diff_label = labels[:, None].ne(candidate_labels[None, :]) & ~exclusion
    hard_diff = sim.masked_fill(~diff_label, -1e9).max(dim=1).values
    hard_same = sim.masked_fill(~same_label, -1e9).max(dim=1).values
    has_diff = hard_diff.gt(-1e8)
    has_same = hard_same.gt(-1e8)
    temp = max(float(temperature), 1e-6)
    risk = torch.sigmoid((hard_diff - float(risk_sim_threshold)) / temp) * has_diff.to(dtype=dtype)
    safe = torch.sigmoid((hard_same - hard_diff - float(safe_margin)) / temp)
    safe = safe * (has_same & has_diff).to(dtype=dtype) * (1.0 - risk)
    weights = 1.0 + (float(max_weight) - 1.0) * risk - (1.0 - float(min_weight)) * safe
    weights = weights.clamp(float(min_weight), float(max_weight))
    if normalize_mean and weights.numel() > 0:
        weights = (weights / weights.mean().clamp_min(1e-8)).clamp(float(min_weight), float(max_weight))
    valid_diff_sim = hard_diff[has_diff]
    valid_same_sim = hard_same[has_same]
    return weights.detach(), {
        "mean_weight": weights.mean(),
        "min_weight": weights.min(),
        "max_weight": weights.max(),
        "avg_hard_negative_sim": valid_diff_sim.mean() if valid_diff_sim.numel() > 0 else node.new_tensor(0.0),
        "avg_same_label_sim": valid_same_sim.mean() if valid_same_sim.numel() > 0 else node.new_tensor(0.0),
        "high_risk_ratio": risk.gt(0.5).to(dtype=dtype).mean(),
    }


def boundary_logit_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    text_key: torch.Tensor,
    audio_key: torch.Tensor,
    sample_ids=None,
    min_sim: float = 0.85,
    margin: float = 0.35,
    temperature: float = 0.05,
    confidence_low: float = 0.50,
    confidence_high: float = 0.85,
    max_active_ratio: float = 1.0,
    text_weight: float = 0.70,
    audio_weight: float = 0.30,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Train-only boundary regularizer from high-similarity different-label neighbors.

    This does not pull representations together. It uses batch relations only to
    identify likely decision-boundary anchors, then weakly asks the true-label
    logit to stay ahead of the most similar different-label neighbor's class.
    """
    batch_size = labels.numel()
    zero = logits.new_tensor(0.0)
    if batch_size <= 1 or logits.numel() == 0:
        return zero, {
            "valid_anchor_ratio": zero,
            "avg_hard_negative_sim": zero,
            "avg_confidence_gate": zero,
            "avg_boundary_weight": zero,
            "avg_margin_violation": zero,
            "active_margin_ratio": zero,
            "memory_size": zero,
        }

    labels = labels.view(-1).to(device=logits.device, dtype=torch.long)
    dtype = logits.dtype
    text_key = F.normalize(_finite_cross_sample_tensor(text_key.to(device=logits.device, dtype=dtype)), dim=-1)
    audio_key = F.normalize(_finite_cross_sample_tensor(audio_key.to(device=logits.device, dtype=dtype)), dim=-1)
    sim = float(text_weight) * torch.matmul(text_key, text_key.transpose(0, 1))
    sim = sim + float(audio_weight) * torch.matmul(audio_key, audio_key.transpose(0, 1))

    exclusion = torch.eye(batch_size, device=logits.device, dtype=torch.bool)
    exclusion |= _same_sample_mask(sample_ids, batch_size, logits.device)
    diff_label = labels[:, None].ne(labels[None, :]) & ~exclusion
    hard_sim, hard_index = sim.masked_fill(~diff_label, -1e9).max(dim=1)
    has_hard_negative = hard_sim.gt(-1e8)
    hard_labels = labels.gather(0, hard_index.clamp(0, batch_size - 1))

    safe_temperature = max(float(temperature), 1e-6)
    relation_gate = torch.sigmoid((hard_sim - float(min_sim)) / safe_temperature)
    relation_gate = relation_gate * has_hard_negative.to(dtype=dtype)

    with torch.no_grad():
        probs = torch.softmax(_finite_cross_sample_tensor(logits.detach(), limit=1e4), dim=-1)
        true_conf = probs.gather(1, labels[:, None]).squeeze(1)
        low = float(confidence_low)
        high = max(float(confidence_high), low + 1e-6)
        confidence_gate = ((high - true_conf) / (high - low)).clamp(0.0, 1.0)

    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    hard_negative_logits = logits.gather(1, hard_labels[:, None]).squeeze(1)
    margin_violation = F.relu(float(margin) - (true_logits - hard_negative_logits))
    boundary_weight = relation_gate * confidence_gate.to(dtype=dtype)
    if class_weights is not None and class_weights.numel() > int(labels.max().detach().cpu()) >= 0:
        class_weight = class_weights.to(device=logits.device, dtype=dtype).gather(0, labels)
        class_weight = class_weight / class_weight.mean().clamp_min(1e-8)
        boundary_weight = boundary_weight * class_weight.detach()

    candidate = boundary_weight.gt(1e-6) & margin_violation.gt(0)
    max_ratio = max(0.0, min(1.0, float(max_active_ratio)))
    if max_ratio < 1.0 and candidate.any():
        max_active = max(1, int(batch_size * max_ratio + 0.999))
        candidate_score = (boundary_weight * margin_violation.detach()).masked_fill(~candidate, -1e9)
        top_scores, top_indices = torch.topk(candidate_score, k=min(max_active, batch_size), dim=0)
        selected = torch.zeros_like(candidate)
        valid = top_scores.gt(-1e8)
        if valid.any():
            selected.scatter_(0, top_indices[valid], True)
        boundary_weight = boundary_weight * selected.to(dtype=dtype)

    active = boundary_weight.gt(1e-6)
    denom = boundary_weight.sum().clamp_min(1e-8)
    loss = (margin_violation * boundary_weight).sum() / denom
    valid_hard_sim = hard_sim[has_hard_negative]
    return loss, {
        "valid_anchor_ratio": active.to(dtype=dtype).mean(),
        "avg_hard_negative_sim": valid_hard_sim.mean() if valid_hard_sim.numel() > 0 else zero,
        "avg_confidence_gate": confidence_gate.to(dtype=dtype).mean(),
        "avg_boundary_weight": boundary_weight.mean(),
        "avg_margin_violation": margin_violation[active].mean() if active.any() else zero,
        "candidate_margin_ratio": candidate.to(dtype=dtype).mean(),
        "active_margin_ratio": (margin_violation.gt(0) & active).to(dtype=dtype).mean(),
        "max_active_ratio": logits.new_tensor(max_ratio),
        "memory_size": zero,
    }


def _same_sample_mask(sample_ids, batch_size: int, device: torch.device) -> torch.Tensor:
    if sample_ids is None:
        return torch.zeros(batch_size, batch_size, device=device, dtype=torch.bool)
    if torch.is_tensor(sample_ids):
        if sample_ids.numel() != batch_size:
            return torch.zeros(batch_size, batch_size, device=device, dtype=torch.bool)
        sample_ids = sample_ids.detach().view(-1).to(device)
        return sample_ids[:, None].eq(sample_ids[None, :])
    if isinstance(sample_ids, Sequence) and len(sample_ids) == batch_size:
        ids = [str(item) for item in sample_ids]
        return torch.tensor(
            [[ids[row] == ids[col] for col in range(batch_size)] for row in range(batch_size)],
            device=device,
            dtype=torch.bool,
        )
    return torch.zeros(batch_size, batch_size, device=device, dtype=torch.bool)


def _positive_and_exclusion_masks(
    labels: torch.Tensor,
    sample_ids,
    exclude_same_sample: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = labels.view(-1)
    batch_size = labels.size(0)
    eye = torch.eye(batch_size, device=labels.device, dtype=torch.bool)
    same_sample = _same_sample_mask(sample_ids, batch_size, labels.device) if exclude_same_sample else eye
    exclusion_mask = eye | (same_sample & ~eye)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~exclusion_mask
    return positive_mask, exclusion_mask


def _empty_contrastive_diagnostics(z: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "valid_anchor_count": z.new_tensor(0.0),
        "valid_anchor_ratio": z.new_tensor(0.0),
        "avg_pos_per_anchor": z.new_tensor(0.0),
    }


def _filter_positive_mask(
    positive_mask: torch.Tensor,
    similarity: torch.Tensor,
    positive_top_k: int = 0,
    positive_sim_threshold: Optional[float] = None,
) -> torch.Tensor:
    """Keep only reliable same-label positives before averaging SupCon terms."""
    if positive_sim_threshold is not None:
        positive_mask = positive_mask & similarity.ge(float(positive_sim_threshold))

    max_positive_count = int(positive_top_k or 0)
    if max_positive_count <= 0 or not positive_mask.any():
        return positive_mask

    k = min(max_positive_count, positive_mask.size(1))
    masked_similarity = similarity.masked_fill(~positive_mask, -1e9)
    top_values, top_indices = torch.topk(masked_similarity, k=k, dim=1)
    topk_mask = torch.zeros_like(positive_mask)
    topk_mask.scatter_(1, top_indices, top_values.gt(-1e8))
    return positive_mask & topk_mask


def supervised_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    sample_ids=None,
    class_balanced: bool = False,
    exclude_same_sample: bool = False,
    positive_top_k: int = 0,
    positive_sim_threshold: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Supervised contrastive loss with optional class balance and sample de-duplication."""
    if z.size(0) <= 1:
        return z.new_tensor(0.0), _empty_contrastive_diagnostics(z)

    z = F.normalize(z, dim=-1)
    labels = labels.view(-1)
    positive_mask, exclusion_mask = _positive_and_exclusion_masks(
        labels,
        sample_ids,
        bool(exclude_same_sample),
    )
    similarity = torch.matmul(z, z.transpose(0, 1))
    positive_mask = _filter_positive_mask(positive_mask, similarity, positive_top_k, positive_sim_threshold)
    valid_anchor_mask = positive_mask.any(dim=1)
    valid_anchor_count = valid_anchor_mask.sum()
    pos_counts = positive_mask.sum(dim=1).to(dtype=z.dtype)
    diagnostics = {
        "valid_anchor_count": valid_anchor_count.to(dtype=z.dtype),
        "valid_anchor_ratio": valid_anchor_mask.to(dtype=z.dtype).mean(),
        "avg_pos_per_anchor": (
            pos_counts[valid_anchor_mask].mean() if valid_anchor_mask.any() else z.new_tensor(0.0)
        ),
    }
    if valid_anchor_count.item() == 0:
        return z.new_tensor(0.0), diagnostics

    logits = similarity / max(float(temperature), 1e-6)
    logits = logits.masked_fill(exclusion_mask, torch.finfo(logits.dtype).min)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_prob = (log_prob * positive_mask.float()).sum(dim=1)
    positive_count = positive_mask.sum(dim=1).clamp_min(1).float()
    per_anchor_loss = -positive_log_prob / positive_count
    if class_balanced:
        class_losses = []
        for label in labels[valid_anchor_mask].unique(sorted=True):
            class_mask = valid_anchor_mask & labels.eq(label)
            if class_mask.any():
                class_losses.append(per_anchor_loss[class_mask].mean())
        loss = torch.stack(class_losses).mean() if class_losses else z.new_tensor(0.0)
    else:
        loss = per_anchor_loss[valid_anchor_mask].mean()
    return loss, diagnostics


def supervised_contrastive_loss_with_memory(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    memory_z: torch.Tensor = None,
    memory_labels: torch.Tensor = None,
    sample_ids=None,
    memory_sample_ids: torch.Tensor = None,
    class_balanced: bool = False,
    exclude_same_sample: bool = False,
    positive_top_k: int = 0,
    positive_sim_threshold: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Supervised contrastive loss where anchors are current batch and candidates include memory."""
    if memory_z is None or memory_labels is None or memory_z.numel() == 0:
        loss, diagnostics = supervised_contrastive_loss(
            z,
            labels,
            temperature,
            sample_ids=sample_ids,
            class_balanced=class_balanced,
            exclude_same_sample=exclude_same_sample,
            positive_top_k=positive_top_k,
            positive_sim_threshold=positive_sim_threshold,
        )
        diagnostics["memory_size"] = z.new_tensor(0.0)
        return loss, diagnostics

    if z.size(0) == 0:
        return z.new_tensor(0.0), _empty_contrastive_diagnostics(z)

    z = F.normalize(z, dim=-1)
    labels = labels.view(-1)
    memory_z = F.normalize(memory_z.detach().to(device=z.device, dtype=z.dtype), dim=-1)
    memory_labels = memory_labels.detach().view(-1).to(device=z.device, dtype=labels.dtype)
    batch_size = z.size(0)
    memory_size = memory_z.size(0)

    candidates = torch.cat([z, memory_z], dim=0)
    candidate_labels = torch.cat([labels, memory_labels], dim=0)
    positive_mask = labels[:, None].eq(candidate_labels[None, :])
    exclusion_mask = torch.zeros(batch_size, batch_size + memory_size, device=z.device, dtype=torch.bool)
    exclusion_mask[:, :batch_size] |= torch.eye(batch_size, device=z.device, dtype=torch.bool)

    if exclude_same_sample:
        batch_same = _same_sample_mask(sample_ids, batch_size, z.device)
        exclusion_mask[:, :batch_size] |= batch_same
        if memory_sample_ids is not None and memory_sample_ids.numel() == memory_size:
            if torch.is_tensor(sample_ids) and sample_ids.numel() == batch_size:
                anchor_ids = sample_ids.detach().view(-1).to(device=z.device, dtype=memory_sample_ids.dtype)
            elif isinstance(sample_ids, Sequence) and len(sample_ids) == batch_size:
                anchor_ids = torch.tensor(
                    [hash(str(item)) & 0x7FFFFFFFFFFFFFFF for item in sample_ids],
                    device=z.device,
                    dtype=memory_sample_ids.dtype,
                )
            else:
                anchor_ids = torch.full((batch_size,), -1, device=z.device, dtype=memory_sample_ids.dtype)
            memory_sample_ids = memory_sample_ids.detach().view(-1).to(z.device)
            valid_ids = anchor_ids[:, None].ge(0) & memory_sample_ids[None, :].ge(0)
            exclusion_mask[:, batch_size:] |= valid_ids & anchor_ids[:, None].eq(memory_sample_ids[None, :])

    positive_mask = positive_mask & ~exclusion_mask
    similarity = torch.matmul(z, candidates.transpose(0, 1))
    positive_mask = _filter_positive_mask(positive_mask, similarity, positive_top_k, positive_sim_threshold)
    valid_anchor_mask = positive_mask.any(dim=1)
    pos_counts = positive_mask.sum(dim=1).to(dtype=z.dtype)
    diagnostics = {
        "valid_anchor_count": valid_anchor_mask.sum().to(dtype=z.dtype),
        "valid_anchor_ratio": valid_anchor_mask.to(dtype=z.dtype).mean(),
        "avg_pos_per_anchor": pos_counts[valid_anchor_mask].mean() if valid_anchor_mask.any() else z.new_tensor(0.0),
        "memory_size": z.new_tensor(float(memory_size)),
    }
    if not valid_anchor_mask.any():
        return z.new_tensor(0.0), diagnostics

    logits = similarity / max(float(temperature), 1e-6)
    logits = logits.masked_fill(exclusion_mask, torch.finfo(logits.dtype).min)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_prob = (log_prob * positive_mask.float()).sum(dim=1)
    positive_count = positive_mask.sum(dim=1).clamp_min(1).float()
    per_anchor_loss = -positive_log_prob / positive_count
    if class_balanced:
        class_losses = []
        for label in labels[valid_anchor_mask].unique(sorted=True):
            class_mask = valid_anchor_mask & labels.eq(label)
            if class_mask.any():
                class_losses.append(per_anchor_loss[class_mask].mean())
        loss = torch.stack(class_losses).mean() if class_losses else z.new_tensor(0.0)
    else:
        loss = per_anchor_loss[valid_anchor_mask].mean()

    return loss, diagnostics


def build_reliable_positive_weights(
    labels: torch.Tensor,
    filter_text: torch.Tensor,
    filter_audio: torch.Tensor,
    candidate_labels: torch.Tensor,
    candidate_filter_text: torch.Tensor,
    candidate_filter_audio: torch.Tensor,
    exclusion_mask: torch.Tensor,
    positive_top_k: int = 2,
    positive_filter_quantile: float = 0.75,
    positive_min_filter_sim: float = 0.15,
    positive_weight_temperature: float = 0.20,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Build weighted same-label positives from reliable text/audio similarity."""
    batch_size = labels.numel()
    candidate_count = candidate_labels.numel()
    if batch_size == 0 or candidate_count == 0:
        empty = filter_text.new_zeros(batch_size, candidate_count)
        diagnostics = {
            "valid_anchor_ratio": filter_text.new_tensor(0.0),
            "valid_anchor_count": filter_text.new_tensor(0.0),
            "avg_pos_per_anchor": filter_text.new_tensor(0.0),
            "avg_filter_sim": filter_text.new_tensor(0.0),
        }
        return empty, empty, diagnostics

    filter_text = F.normalize(filter_text.detach(), dim=-1)
    filter_audio = F.normalize(filter_audio.detach(), dim=-1)
    candidate_filter_text = F.normalize(candidate_filter_text.detach(), dim=-1)
    candidate_filter_audio = F.normalize(candidate_filter_audio.detach(), dim=-1)
    text_sim = torch.matmul(filter_text, candidate_filter_text.transpose(0, 1))
    audio_sim = torch.matmul(filter_audio, candidate_filter_audio.transpose(0, 1))
    filter_score = 0.6 * text_sim + 0.4 * audio_sim

    positive_mask = labels.view(-1, 1).eq(candidate_labels.view(1, -1)) & ~exclusion_mask.bool()
    positive_weights = filter_score.new_zeros(batch_size, candidate_count)
    top_k = int(positive_top_k or 0)
    quantile = min(1.0, max(0.0, float(positive_filter_quantile)))
    min_sim = float(positive_min_filter_sim)
    weight_temperature = max(float(positive_weight_temperature), 1e-6)

    for row in range(batch_size):
        row_positive = positive_mask[row]
        if not row_positive.any():
            continue
        row_scores = filter_score[row]
        positive_scores = row_scores[row_positive]
        threshold = torch.quantile(positive_scores.float(), quantile).to(dtype=row_scores.dtype)
        threshold = torch.maximum(threshold, row_scores.new_tensor(min_sim))
        reliable_mask = row_positive & row_scores.ge(threshold)
        if not reliable_mask.any():
            continue
        if top_k > 0:
            k = min(top_k, int(reliable_mask.sum().item()))
            selected_scores, selected_indices = torch.topk(row_scores.masked_fill(~reliable_mask, -1e9), k=k)
            valid = selected_scores.gt(-1e8)
            if not valid.any():
                continue
            selected_scores = selected_scores[valid]
            selected_indices = selected_indices[valid]
        else:
            selected_indices = reliable_mask.nonzero(as_tuple=False).view(-1)
            selected_scores = row_scores[selected_indices]
        weights = torch.softmax(selected_scores / weight_temperature, dim=0)
        positive_weights[row, selected_indices] = weights

    selected_mask = positive_weights.gt(0)
    valid_anchor_mask = selected_mask.any(dim=1)
    pos_counts = selected_mask.sum(dim=1).to(dtype=filter_score.dtype)
    weighted_filter = (positive_weights * filter_score).sum(dim=1)
    diagnostics = {
        "valid_anchor_count": valid_anchor_mask.sum().to(dtype=filter_score.dtype),
        "valid_anchor_ratio": valid_anchor_mask.to(dtype=filter_score.dtype).mean(),
        "avg_pos_per_anchor": pos_counts[valid_anchor_mask].mean() if valid_anchor_mask.any() else filter_score.new_tensor(0.0),
        "avg_filter_sim": weighted_filter[valid_anchor_mask].mean() if valid_anchor_mask.any() else filter_score.new_tensor(0.0),
    }
    return positive_weights, filter_score, diagnostics


def weighted_supervised_contrastive_loss_with_memory(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    filter_text: torch.Tensor,
    filter_audio: torch.Tensor,
    memory_z: torch.Tensor = None,
    memory_labels: torch.Tensor = None,
    memory_filter_text: torch.Tensor = None,
    memory_filter_audio: torch.Tensor = None,
    sample_ids=None,
    memory_sample_ids: torch.Tensor = None,
    class_balanced: bool = False,
    exclude_same_sample: bool = False,
    positive_top_k: int = 2,
    positive_filter_quantile: float = 0.75,
    positive_min_filter_sim: float = 0.15,
    positive_weight_temperature: float = 0.20,
    drop_ambiguous_negatives: bool = False,
    negative_filter_sim_threshold: float = 0.65,
    anchor_weights: torch.Tensor = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Reliable weighted SupCon with optional detached memory candidates."""
    if z.size(0) == 0:
        return z.new_tensor(0.0), _empty_contrastive_diagnostics(z)

    z = F.normalize(z, dim=-1)
    labels = labels.detach().view(-1).to(device=z.device)
    batch_size = z.size(0)
    has_memory = memory_z is not None and memory_labels is not None and memory_z.numel() > 0
    if has_memory:
        memory_z = F.normalize(memory_z.detach().to(device=z.device, dtype=z.dtype), dim=-1)
        memory_labels = memory_labels.detach().view(-1).to(device=z.device, dtype=labels.dtype)
        memory_size = memory_z.size(0)
        if memory_filter_text is None or memory_filter_text.numel() != memory_size * filter_text.size(-1):
            memory_filter_text = memory_z.new_zeros(memory_size, filter_text.size(-1))
        if memory_filter_audio is None or memory_filter_audio.numel() != memory_size * filter_audio.size(-1):
            memory_filter_audio = memory_z.new_zeros(memory_size, filter_audio.size(-1))
        memory_filter_text = memory_filter_text.detach().to(device=z.device, dtype=filter_text.dtype).view(memory_size, -1)
        memory_filter_audio = memory_filter_audio.detach().to(device=z.device, dtype=filter_audio.dtype).view(memory_size, -1)
    else:
        memory_z = z.new_empty(0, z.size(-1))
        memory_labels = labels.new_empty(0)
        memory_filter_text = filter_text.new_empty(0, filter_text.size(-1))
        memory_filter_audio = filter_audio.new_empty(0, filter_audio.size(-1))
        memory_size = 0

    candidates = torch.cat([z, memory_z], dim=0)
    candidate_labels = torch.cat([labels, memory_labels], dim=0)
    candidate_filter_text = torch.cat([filter_text.detach().to(z.device), memory_filter_text], dim=0)
    candidate_filter_audio = torch.cat([filter_audio.detach().to(z.device), memory_filter_audio], dim=0)

    exclusion_mask = torch.zeros(batch_size, batch_size + memory_size, device=z.device, dtype=torch.bool)
    exclusion_mask[:, :batch_size] |= torch.eye(batch_size, device=z.device, dtype=torch.bool)
    if exclude_same_sample:
        batch_same = _same_sample_mask(sample_ids, batch_size, z.device)
        exclusion_mask[:, :batch_size] |= batch_same
        if memory_sample_ids is not None and memory_sample_ids.numel() == memory_size:
            if torch.is_tensor(sample_ids) and sample_ids.numel() == batch_size:
                anchor_ids = sample_ids.detach().view(-1).to(device=z.device, dtype=memory_sample_ids.dtype)
            elif isinstance(sample_ids, Sequence) and len(sample_ids) == batch_size:
                anchor_ids = torch.tensor(
                    [hash(str(item)) & 0x7FFFFFFFFFFFFFFF for item in sample_ids],
                    device=z.device,
                    dtype=memory_sample_ids.dtype,
                )
            else:
                anchor_ids = torch.full((batch_size,), -1, device=z.device, dtype=memory_sample_ids.dtype)
            memory_sample_ids = memory_sample_ids.detach().view(-1).to(z.device)
            valid_ids = anchor_ids[:, None].ge(0) & memory_sample_ids[None, :].ge(0)
            exclusion_mask[:, batch_size:] |= valid_ids & anchor_ids[:, None].eq(memory_sample_ids[None, :])

    positive_weights, filter_score, diagnostics = build_reliable_positive_weights(
        labels,
        filter_text.to(device=z.device, dtype=z.dtype),
        filter_audio.to(device=z.device, dtype=z.dtype),
        candidate_labels,
        candidate_filter_text.to(dtype=z.dtype),
        candidate_filter_audio.to(dtype=z.dtype),
        exclusion_mask,
        positive_top_k=positive_top_k,
        positive_filter_quantile=positive_filter_quantile,
        positive_min_filter_sim=positive_min_filter_sim,
        positive_weight_temperature=positive_weight_temperature,
    )
    valid_anchor_mask = positive_weights.sum(dim=1).gt(0)
    diagnostics["memory_size"] = z.new_tensor(float(memory_size))
    if anchor_weights is None:
        anchor_gate = z.new_ones(batch_size)
    else:
        anchor_gate = anchor_weights.detach().view(-1).to(device=z.device, dtype=z.dtype).clamp(0.0, 1.0)
    diagnostics["avg_anchor_gate"] = anchor_gate[valid_anchor_mask].mean() if valid_anchor_mask.any() else z.new_tensor(0.0)
    if not valid_anchor_mask.any():
        diagnostics["dropped_negative_ratio"] = z.new_tensor(0.0)
        return z.new_tensor(0.0), diagnostics

    denominator_exclusion = exclusion_mask.clone()
    negative_mask = labels.view(-1, 1).ne(candidate_labels.view(1, -1)) & ~exclusion_mask
    if drop_ambiguous_negatives:
        ambiguous_negative_mask = negative_mask & filter_score.ge(float(negative_filter_sim_threshold))
        denominator_exclusion |= ambiguous_negative_mask
        diagnostics["dropped_negative_ratio"] = (
            ambiguous_negative_mask.sum().to(dtype=z.dtype) / negative_mask.sum().clamp_min(1).to(dtype=z.dtype)
        )
    else:
        diagnostics["dropped_negative_ratio"] = z.new_tensor(0.0)

    logits = torch.matmul(z, candidates.transpose(0, 1)) / max(float(temperature), 1e-6)
    logits = logits.masked_fill(denominator_exclusion, torch.finfo(logits.dtype).min)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    per_anchor_loss = -(positive_weights * log_prob).sum(dim=1) * anchor_gate
    if class_balanced:
        class_losses = []
        for label in labels[valid_anchor_mask].unique(sorted=True):
            class_mask = valid_anchor_mask & labels.eq(label)
            if class_mask.any():
                class_losses.append(per_anchor_loss[class_mask].mean())
        loss = torch.stack(class_losses).mean() if class_losses else z.new_tensor(0.0)
    else:
        loss = per_anchor_loss[valid_anchor_mask].mean()
    return loss, diagnostics





def build_query_relation_graph(
    node: torch.Tensor,
    text_key: torch.Tensor,
    audio_key: torch.Tensor,
    labels: torch.Tensor = None,
    memory_node: torch.Tensor = None,
    memory_text_key: torch.Tensor = None,
    memory_audio_key: torch.Tensor = None,
    memory_labels: torch.Tensor = None,
    sample_ids=None,
    memory_sample_ids: torch.Tensor = None,
    exclude_same_sample: bool = True,
    top_k: int = 4,
    min_sim: float = 0.10,
    same_label_bias: float = 0.05,
    node_weight: float = 0.50,
    text_weight: float = 0.30,
    audio_weight: float = 0.20,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Build a train-time sample relation graph and pooled neighbor context."""
    batch_size = node.size(0)
    if batch_size == 0:
        empty_context = node.new_zeros(node.shape)
        empty_weights = node.new_zeros(0, 0)
        zero = node.new_tensor(0.0)
        return empty_context, empty_weights, {
            "valid_anchor_count": zero,
            "valid_anchor_ratio": zero,
            "avg_neighbors": zero,
            "avg_relation_sim": zero,
            "avg_same_label_edge_ratio": zero,
            "memory_size": zero,
        }

    device = node.device
    dtype = node.dtype
    node = F.normalize(node, dim=-1)
    text_key = F.normalize(text_key.to(device=device, dtype=dtype), dim=-1)
    audio_key = F.normalize(audio_key.to(device=device, dtype=dtype), dim=-1)
    labels = labels.detach().view(-1).to(device=device, dtype=torch.long) if labels is not None else None

    has_memory = memory_node is not None and memory_node.numel() > 0
    if has_memory:
        memory_size = memory_node.size(0)
        memory_ok = (
            memory_text_key is not None
            and memory_audio_key is not None
            and memory_text_key.numel() == memory_size * text_key.size(-1)
            and memory_audio_key.numel() == memory_size * audio_key.size(-1)
        )
        if memory_ok:
            memory_node = F.normalize(memory_node.detach().to(device=device, dtype=dtype), dim=-1)
            memory_text_key = F.normalize(memory_text_key.detach().to(device=device, dtype=dtype).view(memory_size, -1), dim=-1)
            memory_audio_key = F.normalize(memory_audio_key.detach().to(device=device, dtype=dtype).view(memory_size, -1), dim=-1)
            if memory_labels is not None and memory_labels.numel() == memory_size:
                memory_labels = memory_labels.detach().view(-1).to(device=device, dtype=torch.long)
            else:
                memory_labels = None
        else:
            memory_node = node.new_empty(0, node.size(-1))
            memory_text_key = text_key.new_empty(0, text_key.size(-1))
            memory_audio_key = audio_key.new_empty(0, audio_key.size(-1))
            memory_labels = None
            memory_size = 0
    else:
        memory_node = node.new_empty(0, node.size(-1))
        memory_text_key = text_key.new_empty(0, text_key.size(-1))
        memory_audio_key = audio_key.new_empty(0, audio_key.size(-1))
        memory_labels = None
        memory_size = 0

    candidate_node = torch.cat([node, memory_node], dim=0)
    candidate_text = torch.cat([text_key, memory_text_key], dim=0)
    candidate_audio = torch.cat([audio_key, memory_audio_key], dim=0)
    candidate_count = candidate_node.size(0)

    node_sim = torch.matmul(node, candidate_node.transpose(0, 1))
    text_sim = torch.matmul(text_key, candidate_text.transpose(0, 1))
    audio_sim = torch.matmul(audio_key, candidate_audio.transpose(0, 1))
    relation_sim = float(node_weight) * node_sim + float(text_weight) * text_sim + float(audio_weight) * audio_sim

    exclusion_mask = torch.zeros(batch_size, candidate_count, device=device, dtype=torch.bool)
    exclusion_mask[:, :batch_size] |= torch.eye(batch_size, device=device, dtype=torch.bool)
    if exclude_same_sample:
        exclusion_mask[:, :batch_size] |= _same_sample_mask(sample_ids, batch_size, device)
        if memory_sample_ids is not None and memory_sample_ids.numel() == memory_size:
            if torch.is_tensor(sample_ids) and sample_ids.numel() == batch_size:
                anchor_ids = sample_ids.detach().view(-1).to(device=device, dtype=memory_sample_ids.dtype)
            elif isinstance(sample_ids, Sequence) and len(sample_ids) == batch_size:
                anchor_ids = torch.tensor(
                    [hash(str(item)) & 0x7FFFFFFFFFFFFFFF for item in sample_ids],
                    device=device,
                    dtype=memory_sample_ids.dtype,
                )
            else:
                anchor_ids = torch.full((batch_size,), -1, device=device, dtype=memory_sample_ids.dtype)
            memory_sample_ids = memory_sample_ids.detach().view(-1).to(device=device)
            valid_ids = anchor_ids[:, None].ge(0) & memory_sample_ids[None, :].ge(0)
            exclusion_mask[:, batch_size:] |= valid_ids & anchor_ids[:, None].eq(memory_sample_ids[None, :])

    candidate_labels = None
    same_label_mask = torch.zeros(batch_size, candidate_count, device=device, dtype=torch.bool)
    if labels is not None:
        if memory_labels is None:
            candidate_labels = torch.cat([labels, labels.new_full((memory_size,), -1)], dim=0)
        else:
            candidate_labels = torch.cat([labels, memory_labels], dim=0)
        same_label_mask = labels.view(-1, 1).eq(candidate_labels.view(1, -1)) & ~exclusion_mask
        relation_sim = relation_sim + float(same_label_bias) * same_label_mask.to(dtype=dtype)

    valid_edge_mask = ~exclusion_mask & relation_sim.ge(float(min_sim))
    top_k = max(1, int(top_k or 1))
    k = min(top_k, candidate_count)
    top_values, top_indices = torch.topk(relation_sim.masked_fill(~valid_edge_mask, -1e9), k=k, dim=1)
    edge_mask = top_values.gt(-1e8)
    relation_weights = node.new_zeros(batch_size, candidate_count)
    if edge_mask.any():
        safe_values = top_values.masked_fill(~edge_mask, -1e9)
        local_weights = torch.softmax(safe_values, dim=1) * edge_mask.to(dtype=dtype)
        denom = local_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        local_weights = local_weights / denom
        relation_weights.scatter_(1, top_indices, local_weights)

    selected_mask = relation_weights.gt(0)
    valid_anchor_mask = selected_mask.any(dim=1)
    relation_context = torch.matmul(relation_weights, candidate_node)
    neighbor_counts = selected_mask.to(dtype=dtype).sum(dim=1)
    weighted_sim = (relation_weights * relation_sim).sum(dim=1)
    if labels is not None:
        same_label_ratio = (relation_weights * same_label_mask.to(dtype=dtype)).sum(dim=1)
    else:
        same_label_ratio = relation_weights.new_zeros(batch_size)
    zero = node.new_tensor(0.0)
    diagnostics = {
        "valid_anchor_count": valid_anchor_mask.sum().to(dtype=dtype),
        "valid_anchor_ratio": valid_anchor_mask.to(dtype=dtype).mean(),
        "avg_neighbors": neighbor_counts[valid_anchor_mask].mean() if valid_anchor_mask.any() else zero,
        "avg_relation_sim": weighted_sim[valid_anchor_mask].mean() if valid_anchor_mask.any() else zero,
        "avg_same_label_edge_ratio": same_label_ratio[valid_anchor_mask].mean() if valid_anchor_mask.any() else zero,
        "memory_size": node.new_tensor(float(memory_size)),
    }
    return relation_context, relation_weights, diagnostics


def relation_logit_consistency_loss(
    logits: torch.Tensor,
    relation_weights: torch.Tensor,
    memory_logits: torch.Tensor = None,
    temperature: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Weak KL consistency to detached logits of relation neighbors."""
    batch_size, num_classes = logits.shape
    zero = logits.new_tensor(0.0)
    if relation_weights is None or relation_weights.numel() == 0 or batch_size == 0:
        return zero, {"valid_anchor_count": zero, "valid_anchor_ratio": zero, "teacher_entropy": zero}
    if memory_logits is not None and memory_logits.numel() > 0:
        memory_logits = memory_logits.detach().to(device=logits.device, dtype=logits.dtype).view(-1, num_classes)
    else:
        memory_logits = logits.new_empty(0, num_classes)
    candidate_logits = torch.cat([logits.detach(), memory_logits], dim=0)
    if relation_weights.size(1) != candidate_logits.size(0):
        return zero, {"valid_anchor_count": zero, "valid_anchor_ratio": zero, "teacher_entropy": zero}
    valid_anchor_mask = relation_weights.sum(dim=1).gt(0)
    diagnostics = {
        "valid_anchor_count": valid_anchor_mask.sum().to(dtype=logits.dtype),
        "valid_anchor_ratio": valid_anchor_mask.to(dtype=logits.dtype).mean(),
        "teacher_entropy": zero,
    }
    if not valid_anchor_mask.any():
        return zero, diagnostics
    safe_temperature = max(float(temperature), 1e-6)
    neighbor_probs = torch.softmax(candidate_logits / safe_temperature, dim=-1)
    teacher = torch.matmul(relation_weights.to(dtype=logits.dtype), neighbor_probs)
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1e-8)
    log_probs = F.log_softmax(logits / safe_temperature, dim=-1)
    per_anchor_loss = F.kl_div(log_probs, teacher.detach(), reduction="none").sum(dim=1) * (safe_temperature ** 2)
    entropy = -(teacher * teacher.clamp_min(1e-8).log()).sum(dim=1)
    diagnostics["teacher_entropy"] = entropy[valid_anchor_mask].mean()
    return per_anchor_loss[valid_anchor_mask].mean(), diagnostics

def reliable_logit_distillation_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    filter_text: torch.Tensor,
    filter_audio: torch.Tensor,
    memory_logits: torch.Tensor = None,
    memory_labels: torch.Tensor = None,
    memory_filter_text: torch.Tensor = None,
    memory_filter_audio: torch.Tensor = None,
    sample_ids=None,
    memory_sample_ids: torch.Tensor = None,
    exclude_same_sample: bool = True,
    positive_top_k: int = 3,
    positive_filter_quantile: float = 0.70,
    positive_min_filter_sim: float = 0.15,
    positive_weight_temperature: float = 0.20,
    neighbor_label_prob_threshold: float = 0.35,
    binary_neighbor_label_prob_threshold: float = 0.60,
    teacher_temperature: float = 2.0,
    teacher_true_prob: float = 0.90,
    text_weight: float = 0.70,
    audio_weight: float = 0.30,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Detached-neighbor logit distillation without pulling representation geometry."""
    if logits.size(0) == 0:
        zero = logits.new_tensor(0.0)
        return zero, {
            "valid_anchor_count": zero,
            "valid_anchor_ratio": zero,
            "avg_neighbors": zero,
            "avg_filter_sim": zero,
            "avg_neighbor_label_prob": zero,
            "avg_anchor_gate": zero,
            "teacher_true_prob": zero,
            "memory_size": zero,
        }

    device = logits.device
    dtype = logits.dtype
    labels = labels.detach().view(-1).to(device=device, dtype=torch.long)
    batch_size, num_classes = logits.shape
    filter_text = F.normalize(filter_text.detach().to(device=device, dtype=dtype), dim=-1)
    filter_audio = F.normalize(filter_audio.detach().to(device=device, dtype=dtype), dim=-1)

    batch_logits = logits.detach()
    batch_labels = labels
    has_memory = memory_logits is not None and memory_labels is not None and memory_logits.numel() > 0
    if has_memory:
        memory_logits = memory_logits.detach().to(device=device, dtype=dtype).view(-1, num_classes)
        memory_labels = memory_labels.detach().view(-1).to(device=device, dtype=torch.long)
        memory_size = memory_logits.size(0)
        filters_ok = (
            memory_filter_text is not None
            and memory_filter_audio is not None
            and memory_filter_text.numel() == memory_size * filter_text.size(-1)
            and memory_filter_audio.numel() == memory_size * filter_audio.size(-1)
        )
        if filters_ok:
            memory_filter_text = F.normalize(
                memory_filter_text.detach().to(device=device, dtype=dtype).view(memory_size, -1), dim=-1
            )
            memory_filter_audio = F.normalize(
                memory_filter_audio.detach().to(device=device, dtype=dtype).view(memory_size, -1), dim=-1
            )
        else:
            memory_logits = logits.new_empty(0, num_classes)
            memory_labels = labels.new_empty(0)
            memory_filter_text = filter_text.new_empty(0, filter_text.size(-1))
            memory_filter_audio = filter_audio.new_empty(0, filter_audio.size(-1))
            memory_size = 0
    else:
        memory_logits = logits.new_empty(0, num_classes)
        memory_labels = labels.new_empty(0)
        memory_filter_text = filter_text.new_empty(0, filter_text.size(-1))
        memory_filter_audio = filter_audio.new_empty(0, filter_audio.size(-1))
        memory_size = 0

    candidate_logits = torch.cat([batch_logits, memory_logits], dim=0)
    candidate_labels = torch.cat([batch_labels, memory_labels], dim=0)
    candidate_filter_text = torch.cat([filter_text, memory_filter_text], dim=0)
    candidate_filter_audio = torch.cat([filter_audio, memory_filter_audio], dim=0)
    candidate_count = candidate_logits.size(0)

    text_sim = torch.matmul(filter_text, candidate_filter_text.transpose(0, 1))
    audio_sim = torch.matmul(filter_audio, candidate_filter_audio.transpose(0, 1))
    filter_score = float(text_weight) * text_sim + float(audio_weight) * audio_sim

    exclusion_mask = torch.zeros(batch_size, candidate_count, device=device, dtype=torch.bool)
    exclusion_mask[:, :batch_size] |= torch.eye(batch_size, device=device, dtype=torch.bool)
    if exclude_same_sample:
        exclusion_mask[:, :batch_size] |= _same_sample_mask(sample_ids, batch_size, device)
        if memory_sample_ids is not None and memory_sample_ids.numel() == memory_size:
            if torch.is_tensor(sample_ids) and sample_ids.numel() == batch_size:
                anchor_ids = sample_ids.detach().view(-1).to(device=device, dtype=memory_sample_ids.dtype)
            elif isinstance(sample_ids, Sequence) and len(sample_ids) == batch_size:
                anchor_ids = torch.tensor(
                    [hash(str(item)) & 0x7FFFFFFFFFFFFFFF for item in sample_ids],
                    device=device,
                    dtype=memory_sample_ids.dtype,
                )
            else:
                anchor_ids = torch.full((batch_size,), -1, device=device, dtype=memory_sample_ids.dtype)
            memory_sample_ids = memory_sample_ids.detach().view(-1).to(device=device)
            valid_ids = anchor_ids[:, None].ge(0) & memory_sample_ids[None, :].ge(0)
            exclusion_mask[:, batch_size:] |= valid_ids & anchor_ids[:, None].eq(memory_sample_ids[None, :])

    candidate_probs = torch.softmax(candidate_logits, dim=-1)
    candidate_label_prob = candidate_probs.gather(1, candidate_labels.clamp(0, num_classes - 1).view(-1, 1)).squeeze(1)
    threshold = float(binary_neighbor_label_prob_threshold) if num_classes <= 2 else float(neighbor_label_prob_threshold)
    positive_mask = labels.view(-1, 1).eq(candidate_labels.view(1, -1)) & ~exclusion_mask
    positive_mask = positive_mask & candidate_label_prob.view(1, -1).ge(threshold)

    top_k = max(1, int(positive_top_k or 1))
    quantile = min(1.0, max(0.0, float(positive_filter_quantile)))
    min_sim = float(positive_min_filter_sim)
    weight_temperature = max(float(positive_weight_temperature), 1e-6)
    neighbor_weights = filter_score.new_zeros(batch_size, candidate_count)
    for row in range(batch_size):
        row_positive = positive_mask[row]
        if not row_positive.any():
            continue
        row_scores = filter_score[row]
        pos_scores = row_scores[row_positive]
        dynamic_threshold = torch.quantile(pos_scores.float(), quantile).to(dtype=dtype)
        dynamic_threshold = torch.maximum(dynamic_threshold, row_scores.new_tensor(min_sim))
        reliable = row_positive & row_scores.ge(dynamic_threshold)
        if not reliable.any():
            continue
        k = min(top_k, int(reliable.sum().item()))
        top_values, top_indices = torch.topk(row_scores.masked_fill(~reliable, -1e9), k=k)
        valid = top_values.gt(-1e8)
        if not valid.any():
            continue
        top_values = top_values[valid]
        top_indices = top_indices[valid]
        neighbor_weights[row, top_indices] = torch.softmax(top_values / weight_temperature, dim=0)

    selected_mask = neighbor_weights.gt(0)
    valid_anchor_mask = selected_mask.any(dim=1)
    neighbor_counts = selected_mask.to(dtype=dtype).sum(dim=1)

    current_probs = torch.softmax(logits.detach(), dim=-1)
    current_label_prob = current_probs.gather(1, labels.clamp(0, num_classes - 1).view(-1, 1)).squeeze(1)
    chance = 1.0 / max(float(num_classes), 1.0)
    gate_denom = max(0.50 - chance, 1e-6)
    anchor_gate = ((current_label_prob - chance) / gate_denom).clamp(0.0, 1.0)

    zero = logits.new_tensor(0.0)
    diagnostics = {
        "valid_anchor_count": valid_anchor_mask.sum().to(dtype=dtype),
        "valid_anchor_ratio": valid_anchor_mask.to(dtype=dtype).mean(),
        "avg_neighbors": neighbor_counts[valid_anchor_mask].mean() if valid_anchor_mask.any() else zero,
        "avg_filter_sim": (neighbor_weights * filter_score).sum(dim=1)[valid_anchor_mask].mean() if valid_anchor_mask.any() else zero,
        "avg_neighbor_label_prob": (neighbor_weights * candidate_label_prob.view(1, -1)).sum(dim=1)[valid_anchor_mask].mean() if valid_anchor_mask.any() else zero,
        "avg_anchor_gate": anchor_gate[valid_anchor_mask].mean() if valid_anchor_mask.any() else zero,
        "teacher_true_prob": zero,
        "memory_size": logits.new_tensor(float(memory_size)),
    }
    active_mask = valid_anchor_mask & anchor_gate.gt(0)
    if not active_mask.any():
        return zero, diagnostics

    safe_temperature = max(float(teacher_temperature), 1e-6)
    neighbor_probs = torch.softmax(candidate_logits.detach() / safe_temperature, dim=-1)
    dark_teacher = torch.matmul(neighbor_weights, neighbor_probs)
    dark_teacher = dark_teacher / dark_teacher.sum(dim=1, keepdim=True).clamp_min(1e-8)
    true_mass = min(max(float(teacher_true_prob), 0.0), 1.0)
    hard = F.one_hot(labels, num_classes=num_classes).to(dtype=dtype)
    teacher = (1.0 - true_mass) * dark_teacher + true_mass * hard
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1e-8)
    diagnostics["teacher_true_prob"] = teacher.gather(1, labels.view(-1, 1)).squeeze(1)[active_mask].mean()

    log_probs = F.log_softmax(logits / safe_temperature, dim=-1)
    per_anchor_loss = F.kl_div(log_probs, teacher.detach(), reduction="none").sum(dim=1) * (safe_temperature ** 2)
    loss = (per_anchor_loss[active_mask] * anchor_gate[active_mask]).mean()
    return loss, diagnostics

def ccr_margin_retrieval_loss_with_memory(
    anchor: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 1.0,
    memory_anchor: torch.Tensor = None,
    memory_labels: torch.Tensor = None,
    sample_ids=None,
    memory_sample_ids: torch.Tensor = None,
    exclude_same_sample: bool = True,
    anchor_weights: torch.Tensor = None,
    positive_top_k: int = 1,
    positive_only: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Hard positive retrieval with an optional hard-negative margin term.
    if anchor.size(0) == 0:
        diagnostics = {
            "valid_anchor_count": anchor.new_tensor(0.0),
            "valid_anchor_ratio": anchor.new_tensor(0.0),
            "avg_pos_sim": anchor.new_tensor(0.0),
            "avg_neg_sim": anchor.new_tensor(0.0),
            "margin_active_ratio": anchor.new_tensor(0.0),
            "avg_anchor_gate": anchor.new_tensor(0.0),
            "memory_size": anchor.new_tensor(0.0),
        }
        return anchor.new_tensor(0.0), diagnostics

    anchor = F.normalize(anchor, dim=-1)
    labels = labels.detach().view(-1).to(device=anchor.device)
    batch_size = anchor.size(0)
    has_memory = memory_anchor is not None and memory_labels is not None and memory_anchor.numel() > 0
    if has_memory:
        memory_anchor = F.normalize(memory_anchor.detach().to(device=anchor.device, dtype=anchor.dtype), dim=-1)
        memory_labels = memory_labels.detach().view(-1).to(device=anchor.device, dtype=labels.dtype)
        memory_size = memory_anchor.size(0)
    else:
        memory_anchor = anchor.new_empty(0, anchor.size(-1))
        memory_labels = labels.new_empty(0)
        memory_size = 0

    candidates = torch.cat([anchor, memory_anchor], dim=0)
    candidate_labels = torch.cat([labels, memory_labels], dim=0)
    candidate_count = candidates.size(0)
    exclusion_mask = torch.zeros(batch_size, candidate_count, device=anchor.device, dtype=torch.bool)
    exclusion_mask[:, :batch_size] |= torch.eye(batch_size, device=anchor.device, dtype=torch.bool)

    if exclude_same_sample:
        batch_same = _same_sample_mask(sample_ids, batch_size, anchor.device)
        exclusion_mask[:, :batch_size] |= batch_same
        if memory_sample_ids is not None and memory_sample_ids.numel() == memory_size:
            if torch.is_tensor(sample_ids) and sample_ids.numel() == batch_size:
                anchor_ids = sample_ids.detach().view(-1).to(device=anchor.device, dtype=memory_sample_ids.dtype)
            elif isinstance(sample_ids, Sequence) and len(sample_ids) == batch_size:
                anchor_ids = torch.tensor(
                    [hash(str(item)) & 0x7FFFFFFFFFFFFFFF for item in sample_ids],
                    device=anchor.device,
                    dtype=memory_sample_ids.dtype,
                )
            else:
                anchor_ids = torch.full((batch_size,), -1, device=anchor.device, dtype=memory_sample_ids.dtype)
            memory_sample_ids = memory_sample_ids.detach().view(-1).to(anchor.device)
            valid_ids = anchor_ids[:, None].ge(0) & memory_sample_ids[None, :].ge(0)
            exclusion_mask[:, batch_size:] |= valid_ids & anchor_ids[:, None].eq(memory_sample_ids[None, :])

    similarity = torch.matmul(anchor, candidates.transpose(0, 1))
    positive_mask = labels.view(-1, 1).eq(candidate_labels.view(1, -1)) & ~exclusion_mask
    negative_mask = labels.view(-1, 1).ne(candidate_labels.view(1, -1)) & ~exclusion_mask
    valid_anchor_mask = positive_mask.any(dim=1)
    if not positive_only:
        valid_anchor_mask = valid_anchor_mask & negative_mask.any(dim=1)
    if anchor_weights is not None:
        anchor_gate = anchor_weights.detach().view(-1).to(device=anchor.device, dtype=anchor.dtype).clamp(0.0, 1.0)
        if anchor_gate.numel() != batch_size:
            anchor_gate = anchor.new_ones(batch_size)
    else:
        anchor_gate = anchor.new_ones(batch_size)

    diagnostics = {
        "valid_anchor_count": valid_anchor_mask.sum().to(dtype=anchor.dtype),
        "valid_anchor_ratio": valid_anchor_mask.to(dtype=anchor.dtype).mean(),
        "avg_pos_sim": anchor.new_tensor(0.0),
        "avg_neg_sim": anchor.new_tensor(0.0),
        "margin_active_ratio": anchor.new_tensor(0.0),
        "avg_anchor_gate": anchor.new_tensor(0.0),
        "memory_size": anchor.new_tensor(float(memory_size)),
    }
    if not valid_anchor_mask.any():
        return anchor.new_tensor(0.0), diagnostics

    top_k = max(1, int(positive_top_k or 1))
    top_k = min(top_k, candidate_count)
    pos_sim_values, pos_idx = similarity.masked_fill(~positive_mask, -1e9).topk(top_k, dim=1)
    pos_valid = pos_sim_values.gt(-1e8)
    pos_vectors = candidates[pos_idx]
    pos_dist_values = (anchor[:, None, :] - pos_vectors).pow(2).sum(dim=-1)
    pos_count = pos_valid.to(dtype=anchor.dtype).sum(dim=1).clamp_min(1.0)
    pos_dist_sq = (pos_dist_values * pos_valid.to(dtype=anchor.dtype)).sum(dim=1) / pos_count
    pos_sim = (pos_sim_values.masked_fill(~pos_valid, 0.0) * pos_valid.to(dtype=anchor.dtype)).sum(dim=1) / pos_count
    if positive_only:
        neg_sim = anchor.new_zeros(batch_size)
        margin_term = anchor.new_zeros(batch_size)
        per_anchor_loss = pos_dist_sq
    else:
        neg_sim, neg_idx = similarity.masked_fill(~negative_mask, -1e9).max(dim=1)
        neg = candidates[neg_idx]
        neg_dist_sq = (anchor - neg).pow(2).sum(dim=-1)
        margin_term = F.relu(anchor.new_tensor(float(margin)) - neg_dist_sq)
        per_anchor_loss = pos_dist_sq + margin_term
    valid_gate = anchor_gate[valid_anchor_mask]
    loss = (per_anchor_loss[valid_anchor_mask] * valid_gate).mean()
    diagnostics["avg_pos_sim"] = pos_sim[valid_anchor_mask].mean()
    diagnostics["avg_neg_sim"] = neg_sim[valid_anchor_mask].mean()
    diagnostics["margin_active_ratio"] = margin_term[valid_anchor_mask].gt(0).to(dtype=anchor.dtype).mean()
    diagnostics["avg_anchor_gate"] = valid_gate.mean()
    return loss, diagnostics


@torch.no_grad()
def _positive_topk_indices(
    z: torch.Tensor,
    labels: torch.Tensor,
    top_k: int,
    sample_ids=None,
    sim_threshold: float = -1.0,
    exclude_same_sample: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    batch_size = z.size(0)
    if top_k <= 0 or batch_size == 0:
        empty_indices = torch.empty(batch_size, 0, device=z.device, dtype=torch.long)
        empty_mask = torch.empty(batch_size, 0, device=z.device, dtype=torch.bool)
        empty_values = torch.empty(batch_size, 0, device=z.device, dtype=z.dtype)
        diagnostics = {
            "ref_coverage": z.new_tensor(0.0),
            "avg_ref_sim": z.new_tensor(0.0),
            "avg_pos_per_anchor": z.new_tensor(0.0),
        }
        return empty_indices, empty_mask, empty_values, diagnostics

    labels = labels.view(-1)
    z = F.normalize(z, dim=-1)
    positive_mask, _ = _positive_and_exclusion_masks(labels, sample_ids, bool(exclude_same_sample))
    similarity = torch.matmul(z, z.transpose(0, 1)).masked_fill(~positive_mask, -1e9)
    k = min(int(top_k), max(batch_size - 1, 0))
    top_values, top_indices = torch.topk(similarity, k=k, dim=1)
    ref_mask = (top_values > -1e8) & (top_values >= float(sim_threshold))
    if k < top_k:
        pad = top_k - k
        top_indices = F.pad(top_indices, (0, pad), value=0)
        ref_mask = F.pad(ref_mask, (0, pad), value=False)
        top_values = F.pad(top_values, (0, pad), value=0.0)
    diagnostics = {
        "ref_coverage": ref_mask.any(dim=1).to(dtype=z.dtype).mean(),
        "avg_ref_sim": top_values[ref_mask].mean() if ref_mask.any() else z.new_tensor(0.0),
        "avg_pos_per_anchor": positive_mask.sum(dim=1).to(dtype=z.dtype).mean(),
    }
    return top_indices, ref_mask, top_values, diagnostics


def build_batch_positive_refs(
    z: torch.Tensor,
    labels: torch.Tensor,
    text_pool: torch.Tensor,
    audio_pool: torch.Tensor,
    top_k: int,
    sample_ids=None,
    sim_threshold: float = -1.0,
    detach_refs: bool = False,
    exclude_same_sample: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Build [B,K,D] same-label references from current batch positives."""
    indices, ref_mask, _, diagnostics = _positive_topk_indices(
        z.detach(),
        labels,
        int(top_k),
        sample_ids=sample_ids,
        sim_threshold=float(sim_threshold),
        exclude_same_sample=bool(exclude_same_sample),
    )
    if indices.numel() == 0:
        empty_shape = (z.size(0), 0, text_pool.size(-1))
        return text_pool.new_zeros(empty_shape), audio_pool.new_zeros(empty_shape), ref_mask, diagnostics

    text_source = text_pool.detach() if detach_refs else text_pool
    audio_source = audio_pool.detach() if detach_refs else audio_pool
    ref_text = text_source[indices] * ref_mask.unsqueeze(-1).to(dtype=text_pool.dtype)
    ref_audio = audio_source[indices] * ref_mask.unsqueeze(-1).to(dtype=audio_pool.dtype)
    return ref_text, ref_audio, ref_mask, diagnostics


@torch.no_grad()
def train_memory_retrieval_logit_adapter(
    base_logits: torch.Tensor,
    text_key: torch.Tensor,
    audio_key: torch.Tensor,
    memory_text_key: torch.Tensor = None,
    memory_audio_key: torch.Tensor = None,
    memory_labels: torch.Tensor = None,
    num_classes: int = 2,
    alpha: float = 0.05,
    top_k_per_class: int = 5,
    temperature: float = 0.10,
    prior_beta: float = 0.5,
    min_sim: float = 0.20,
    sim_gate_width: float = 0.30,
    text_weight: float = 0.6,
    audio_weight: float = 0.4,
    logit_clamp: float = 3.0,
    train_class_prior: torch.Tensor = None,
    sample_ids: torch.Tensor = None,
    memory_sample_ids: torch.Tensor = None,
    exclude_same_sample: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Post-hoc train-memory retrieval prior for evaluation-time logits.

    The adapter never changes the backbone representation. It retrieves from a
    fixed train-split memory and adds a small, gated class prior to logits.
    """
    batch_size = base_logits.size(0)
    device = base_logits.device
    dtype = base_logits.dtype
    empty_retrieval = base_logits.new_zeros(batch_size, int(num_classes))

    def _empty_result() -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        diagnostics = {
            "gate_mean": base_logits.new_tensor(0.0),
            "active_gate_ratio": base_logits.new_tensor(0.0),
            "max_sim_mean": base_logits.new_tensor(0.0),
            "retrieval_entropy": base_logits.new_tensor(0.0),
            "retrieval_logit_abs_mean": base_logits.new_tensor(0.0),
            "memory_size": base_logits.new_tensor(0.0),
        }
        return base_logits, empty_retrieval, diagnostics

    if (
        memory_text_key is None
        or memory_audio_key is None
        or memory_labels is None
        or memory_labels.numel() == 0
        or batch_size == 0
        or int(num_classes) <= 0
        or abs(float(alpha)) < 1e-12
    ):
        final_logits, retrieval_logits, diagnostics = _empty_result()
        if memory_labels is not None:
            diagnostics["memory_size"] = base_logits.new_tensor(float(memory_labels.numel()))
        return final_logits, retrieval_logits, diagnostics

    text_key = F.normalize(text_key.to(device=device, dtype=dtype), dim=-1)
    audio_key = F.normalize(audio_key.to(device=device, dtype=dtype), dim=-1)
    memory_text_key = F.normalize(memory_text_key.detach().to(device=device, dtype=dtype), dim=-1)
    memory_audio_key = F.normalize(memory_audio_key.detach().to(device=device, dtype=dtype), dim=-1)
    memory_labels = memory_labels.detach().view(-1).to(device=device, dtype=torch.long)
    memory_size = memory_labels.numel()
    if memory_text_key.size(0) != memory_size or memory_audio_key.size(0) != memory_size:
        return _empty_result()

    text_sim = torch.matmul(text_key, memory_text_key.transpose(0, 1))
    audio_sim = torch.matmul(audio_key, memory_audio_key.transpose(0, 1))
    similarity = float(text_weight) * text_sim + float(audio_weight) * audio_sim

    if exclude_same_sample and sample_ids is not None and memory_sample_ids is not None:
        if torch.is_tensor(sample_ids) and sample_ids.numel() == batch_size:
            anchor_ids = sample_ids.detach().view(-1).to(device=device, dtype=torch.long)
        else:
            anchor_ids = torch.full((batch_size,), -1, device=device, dtype=torch.long)
        memory_sample_ids = memory_sample_ids.detach().view(-1).to(device=device, dtype=torch.long)
        if memory_sample_ids.numel() == memory_size:
            valid_ids = anchor_ids[:, None].ge(0) & memory_sample_ids[None, :].ge(0)
            similarity = similarity.masked_fill(valid_ids & anchor_ids[:, None].eq(memory_sample_ids[None, :]), -1e9)

    class_scores = base_logits.new_full((batch_size, int(num_classes)), -20.0)
    class_top_sim = base_logits.new_full((batch_size, int(num_classes)), -1.0)
    top_k = max(1, int(top_k_per_class or 1))
    safe_temperature = max(float(temperature), 1e-6)
    for class_idx in range(int(num_classes)):
        class_mask = memory_labels.eq(class_idx)
        if not class_mask.any():
            continue
        class_sim = similarity[:, class_mask]
        k = min(top_k, class_sim.size(1))
        top_values = torch.topk(class_sim, k=k, dim=1).values
        valid = top_values.ge(float(min_sim))
        valid_count = valid.to(dtype=dtype).sum(dim=1)
        masked_values = top_values.masked_fill(~valid, 0.0)
        mean_sim = masked_values.sum(dim=1) / valid_count.clamp_min(1.0)
        class_top_sim[:, class_idx] = top_values[:, 0]
        class_score = mean_sim / safe_temperature
        if train_class_prior is not None and train_class_prior.numel() == int(num_classes):
            prior = train_class_prior.to(device=device, dtype=dtype).view(-1).clamp_min(1e-8)
            class_score = class_score - float(prior_beta) * prior[class_idx].log()
        class_scores[:, class_idx] = torch.where(valid_count.gt(0), class_score, class_scores[:, class_idx])

    max_sim = class_top_sim.max(dim=1).values
    retrieval_probs = torch.softmax(class_scores, dim=-1)
    entropy = -(retrieval_probs * retrieval_probs.clamp_min(1e-8).log()).sum(dim=-1)
    log_classes = base_logits.new_tensor(float(max(int(num_classes), 2))).log()
    entropy_gate = (1.0 - entropy / log_classes.clamp_min(1e-8)).clamp(0.0, 1.0)
    sim_gate = ((max_sim - float(min_sim)) / max(float(sim_gate_width), 1e-6)).clamp(0.0, 1.0)
    gate = float(alpha) * sim_gate * entropy_gate

    retrieval_logits = class_scores - class_scores.mean(dim=1, keepdim=True)
    retrieval_logits = retrieval_logits.clamp(-float(logit_clamp), float(logit_clamp)).detach()
    final_logits = base_logits + gate.unsqueeze(-1) * retrieval_logits
    diagnostics = {
        "gate_mean": gate.mean(),
        "active_gate_ratio": gate.gt(1e-8).to(dtype=dtype).mean(),
        "max_sim_mean": max_sim.mean(),
        "retrieval_entropy": entropy.mean(),
        "retrieval_logit_abs_mean": retrieval_logits.abs().mean(),
        "memory_size": base_logits.new_tensor(float(memory_size)),
    }
    return final_logits, retrieval_logits, diagnostics
