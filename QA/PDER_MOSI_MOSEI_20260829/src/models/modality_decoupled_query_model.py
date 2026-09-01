"""MDReID-inspired modality-decoupled query model for audio-text emotion recognition.

This model keeps the existing RefFormer audio-text training interface and cached
feature format. It adds a small decoupling head that extracts modality-specific
and modality-shared tokens from final text/audio evidence representations:

    [Text_sp, Audio_sp, Text_sh, Audio_sh]

The head follows the MDReID idea in a form suitable for classification rather
than retrieval: a representation orthogonality loss separates shared/specific
channels, and a knowledge discrepancy loss encourages the combined
shared+specific representation to be more discriminative than either part alone.
"""
from typing import Dict, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cross_sample import supervised_contrastive_loss
from src.models.unified_referential_evidence_query_model import QueryAdaptionModule, RefFormerAudioTextEmotionModel


def _finite_tensor(tensor: torch.Tensor, limit: float = 50.0) -> torch.Tensor:
    if tensor is None:
        return tensor
    return torch.nan_to_num(tensor, nan=0.0, posinf=limit, neginf=-limit).clamp(min=-limit, max=limit)


def _masked_mean_sequence(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    tokens = _finite_tensor(tokens)
    mask = mask.bool()
    if mask.dim() > 2:
        mask = mask.squeeze(-1)
    if mask.size(1) != tokens.size(1):
        return tokens.mean(dim=1)
    if not mask.any(dim=1).all():
        mask = mask.clone()
        tokens = tokens.clone()
        missing_rows = ~mask.any(dim=1)
        mask[missing_rows, 0] = True
        tokens[missing_rows, 0, :] = 0.0
    weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
    return _finite_tensor((tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0))


class EvidenceTokenDecoupler(nn.Module):
    """Extract specific/shared tokens from one modality's evidence-unit memory."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(2, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, memory: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = memory.size(0)
        mask = mask.bool()
        missing_rows = ~mask.any(dim=1)
        if missing_rows.any():
            mask = mask.clone()
            memory = memory.clone()
            mask[missing_rows, 0] = True
            memory[missing_rows, 0, :] = 0.0
        query = self.tokens.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.attn(
            query,
            _finite_tensor(memory),
            _finite_tensor(memory),
            key_padding_mask=~mask,
            need_weights=False,
        )
        tokens = self.norm1(query + _finite_tensor(attended))
        tokens = self.norm2(tokens + self.ffn(tokens))
        return _finite_tensor(tokens[:, 0, :]), _finite_tensor(tokens[:, 1, :])


class UnitSPShMILHead(nn.Module):
    """Segment-level SP/SH evidence head with multiple-instance pooling.

    Depression evidence is often sparse: one or two interview segments can carry
    the decisive cue.  This head keeps the evidence-unit dimension and scores
    each unit before aggregating, instead of averaging all units into one global
    SP/SH vector.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        num_heads: int,
        dropout: float,
        topk: int = 3,
        residual_scale: float = 0.05,
        gate_init: float = -4.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.topk = max(1, int(topk))
        self.residual_scale = float(residual_scale)
        self.text_specific = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.audio_specific = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.text_shared = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        self.audio_shared = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        self.shared_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        unit_input_dim = hidden_dim * 7
        self.unit_fusion = nn.Sequential(
            nn.LayerNorm(unit_input_dim),
            nn.Linear(unit_input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.unit_scorer = nn.Linear(hidden_dim, num_classes)
        self.unit_reliability = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, max(16, hidden_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, hidden_dim // 2), 1),
        )
        self.evidence_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.class_delta = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * max(num_classes, 1)),
        )
        gate_input_dim = hidden_dim * 2 + num_classes * 2 + 3
        self.gate = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, max(16, hidden_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, hidden_dim // 2), 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, float(gate_init))

    def forward(
        self,
        text_units: torch.Tensor,
        audio_units: torch.Tensor,
        class_reps: torch.Tensor,
        unit_mask: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        text_units = _finite_tensor(text_units)
        audio_units = _finite_tensor(audio_units)
        class_reps = _finite_tensor(class_reps)
        unit_mask = unit_mask.bool()
        if not unit_mask.any(dim=1).all():
            unit_mask = unit_mask.clone()
            unit_mask[~unit_mask.any(dim=1), 0] = True

        pooled_query = class_reps.mean(dim=1)
        query_units = pooled_query.unsqueeze(1).expand(-1, text_units.size(1), -1)
        text_sp = _finite_tensor(self.text_specific(text_units))
        audio_sp = _finite_tensor(self.audio_specific(audio_units))
        text_sh = _finite_tensor(self.text_shared(text_units))
        audio_sh = _finite_tensor(self.audio_shared(audio_units))
        shared_pair = torch.cat(
            [text_sh, audio_sh, torch.abs(text_sh - audio_sh), text_sh * audio_sh],
            dim=-1,
        )
        shared = _finite_tensor(self.shared_fusion(_finite_tensor(shared_pair)))
        unit_input = torch.cat(
            [
                query_units,
                shared,
                text_sp,
                audio_sp,
                torch.abs(text_sp - audio_sp),
                text_sp * audio_sp,
                torch.abs(shared - 0.5 * (text_sp + audio_sp)),
            ],
            dim=-1,
        )
        unit_evidence = _finite_tensor(self.unit_fusion(_finite_tensor(unit_input)))
        unit_evidence = unit_evidence.masked_fill(~unit_mask.unsqueeze(-1), 0.0)

        unit_logits = _finite_tensor(self.unit_scorer(unit_evidence), limit=1e4)
        reliability = torch.sigmoid(self.unit_reliability(unit_evidence)).squeeze(-1)
        reliability = reliability.masked_fill(~unit_mask, 0.0)

        masked_unit_logits = unit_logits.masked_fill(~unit_mask.unsqueeze(-1), -1e4)
        if self.num_classes >= 2:
            positive_scores = masked_unit_logits[..., 1] + torch.log(reliability.clamp_min(1e-6))
            valid_counts = unit_mask.sum(dim=1).clamp_min(1)
            k = min(self.topk, int(unit_mask.size(1)))
            topk_values = torch.topk(positive_scores, k=k, dim=1).values
            topk_mask = torch.arange(k, device=unit_mask.device).unsqueeze(0) < valid_counts.clamp(max=k).unsqueeze(1)
            positive_logit = (
                topk_values.masked_fill(~topk_mask, 0.0).sum(dim=1)
                / topk_mask.to(dtype=topk_values.dtype).sum(dim=1).clamp_min(1.0)
            )
            negative_scores = masked_unit_logits[..., 0]
            negative_logit = (
                negative_scores.masked_fill(~unit_mask, 0.0).sum(dim=1)
                / unit_mask.to(dtype=negative_scores.dtype).sum(dim=1).clamp_min(1.0)
            )
            mil_logits = torch.stack([negative_logit, positive_logit], dim=-1)
        else:
            mil_logits = (
                masked_unit_logits.masked_fill(~unit_mask.unsqueeze(-1), 0.0).sum(dim=1)
                / unit_mask.to(dtype=unit_logits.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
            )

        evidence_context, evidence_weights = self.evidence_attention(
            pooled_query.unsqueeze(1),
            unit_evidence,
            unit_evidence,
            key_padding_mask=~unit_mask,
            need_weights=True,
        )
        evidence_context = _finite_tensor(evidence_context.squeeze(1))
        base_probs = torch.softmax(_finite_tensor(base_logits, limit=1e4), dim=-1)
        mil_probs = torch.softmax(_finite_tensor(mil_logits, limit=1e4), dim=-1)
        sorted_base = base_probs.sort(dim=-1, descending=True).values
        margin = sorted_base[:, :1] - sorted_base[:, 1:2] if sorted_base.size(-1) > 1 else sorted_base[:, :1]
        entropy = -(base_probs * base_probs.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
        entropy = entropy / base_logits.new_tensor(math.log(max(int(base_probs.size(-1)), 2)))
        gate_input = torch.cat(
            [
                pooled_query,
                evidence_context,
                base_probs,
                mil_probs,
                margin,
                entropy,
                reliability.sum(dim=1, keepdim=True) / unit_mask.to(dtype=reliability.dtype).sum(dim=1, keepdim=True).clamp_min(1.0),
            ],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate(_finite_tensor(gate_input)))
        class_delta = self.class_delta(evidence_context).view(-1, self.num_classes, self.hidden_dim)
        fused_class_reps = _finite_tensor(
            class_reps + class_reps.new_tensor(self.residual_scale) * gate.unsqueeze(-1) * class_delta
        )
        diagnostics = {
            "unit_spsh_mil_gate": gate.detach().mean(),
            "unit_spsh_mil_reliability": (
                reliability.detach().sum(dim=1)
                / unit_mask.to(dtype=reliability.dtype).sum(dim=1).clamp_min(1.0)
            ).mean(),
            "unit_spsh_mil_top_positive": (
                positive_scores.detach().masked_fill(~unit_mask, -1e4).max(dim=1).values.mean()
                if self.num_classes >= 2
                else unit_logits.detach().mean()
            ),
            "unit_spsh_mil_attention_max": evidence_weights.detach().max(dim=-1).values.mean(),
        }
        return fused_class_reps, _finite_tensor(mil_logits, limit=1e4), diagnostics


class HQAQueryBlock(nn.Module):
    """Self/cross-attention block used by hierarchical affective queries."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, use_self_attention: bool = True):
        super().__init__()
        self.use_self_attention = bool(use_self_attention)
        if self.use_self_attention:
            self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            self.self_norm = nn.LayerNorm(hidden_dim)
            self.self_out_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.cross_out_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_out_norm = nn.LayerNorm(hidden_dim)

    def _prepare_memory(self, memory: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        memory = _finite_tensor(memory)
        mask = mask.bool()
        if not mask.any(dim=1).all():
            memory = memory.clone()
            mask = mask.clone()
            missing_rows = ~mask.any(dim=1)
            mask[missing_rows, 0] = True
            memory[missing_rows, 0, :] = 0.0
        return memory, mask

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        queries = _finite_tensor(queries)
        if queries.size(1) <= 0:
            return queries, queries.new_zeros(queries.size(0), 0, queries.size(-1))
        if self.use_self_attention and queries.size(1) > 1:
            self_context, _ = self.self_attn(
                self.self_norm(queries),
                self.self_norm(queries),
                self.self_norm(queries),
                need_weights=False,
            )
            queries = self.self_out_norm(queries + _finite_tensor(self_context))
        memory, mask = self._prepare_memory(memory, mask)
        context, _ = self.cross_attn(
            self.cross_norm(queries),
            memory,
            memory,
            key_padding_mask=~mask,
            need_weights=False,
        )
        queries = self.cross_out_norm(queries + _finite_tensor(context))
        queries = self.ffn_out_norm(queries + self.ffn(queries))
        return _finite_tensor(queries), _finite_tensor(context)


class HQAInteractionLayer(nn.Module):
    """One local-global HQA update layer over private/shared SP/SH memories."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.text_private = HQAQueryBlock(hidden_dim, num_heads, dropout, use_self_attention=True)
        self.audio_private = HQAQueryBlock(hidden_dim, num_heads, dropout, use_self_attention=True)
        self.shared = HQAQueryBlock(hidden_dim, num_heads, dropout, use_self_attention=True)
        self.global_cross = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.global_norm = nn.LayerNorm(hidden_dim)
        self.global_out_norm = nn.LayerNorm(hidden_dim)
        self.global_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 6),
            nn.Linear(hidden_dim * 6, max(16, hidden_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, hidden_dim // 2), 1),
        )
        nn.init.zeros_(self.global_gate[-1].weight)
        nn.init.constant_(self.global_gate[-1].bias, -1.0)
        self.global_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.global_ffn_norm = nn.LayerNorm(hidden_dim)

    def _global_memory(
        self,
        q_text: torch.Tensor,
        q_audio: torch.Tensor,
        q_shared: torch.Tensor,
        class_reps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        parts = []
        for item in (q_text, q_audio, q_shared, class_reps):
            if item is not None and item.size(1) > 0:
                parts.append(_finite_tensor(item))
        if not parts:
            return class_reps, torch.ones(class_reps.shape[:2], device=class_reps.device, dtype=torch.bool)
        memory = torch.cat(parts, dim=1)
        mask = torch.ones(memory.shape[:2], device=memory.device, dtype=torch.bool)
        return _finite_tensor(memory), mask

    def _pool_or_zero(self, queries: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if queries is None or queries.size(1) <= 0:
            return reference.new_zeros(reference.size(0), reference.size(-1))
        return _finite_tensor(queries).mean(dim=1)

    def forward(
        self,
        q_text: torch.Tensor,
        q_audio: torch.Tensor,
        q_shared: torch.Tensor,
        q_global: torch.Tensor,
        text_memory: torch.Tensor,
        text_memory_mask: torch.Tensor,
        audio_memory: torch.Tensor,
        audio_memory_mask: torch.Tensor,
        shared_memory: torch.Tensor,
        shared_memory_mask: torch.Tensor,
        class_reps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        q_text, text_context = self.text_private(q_text, text_memory, text_memory_mask)
        q_audio, audio_context = self.audio_private(q_audio, audio_memory, audio_memory_mask)
        q_shared, shared_context = self.shared(q_shared, shared_memory, shared_memory_mask)

        global_memory, global_mask = self._global_memory(q_text, q_audio, q_shared, class_reps)
        global_context, _ = self.global_cross(
            self.global_norm(q_global),
            global_memory,
            global_memory,
            key_padding_mask=~global_mask,
            need_weights=False,
        )
        reference = q_global
        class_pool = class_reps.mean(dim=1)
        gate_input = torch.cat(
            [
                self._pool_or_zero(q_global, reference),
                _finite_tensor(global_context).mean(dim=1),
                self._pool_or_zero(q_text, reference),
                self._pool_or_zero(q_audio, reference),
                self._pool_or_zero(q_shared, reference),
                class_pool,
            ],
            dim=-1,
        )
        global_gate = torch.sigmoid(self.global_gate(_finite_tensor(gate_input))).view(-1, 1, 1)
        q_global = self.global_out_norm(q_global + global_gate * _finite_tensor(global_context))
        q_global = self.global_ffn_norm(q_global + self.global_ffn(q_global))

        diagnostics = {
            "global_gate": global_gate.detach().mean(),
            "text_context_norm": (
                text_context.detach().norm(dim=-1).mean()
                if text_context.numel() > 0
                else q_global.new_tensor(0.0)
            ),
            "audio_context_norm": (
                audio_context.detach().norm(dim=-1).mean()
                if audio_context.numel() > 0
                else q_global.new_tensor(0.0)
            ),
            "shared_context_norm": (
                shared_context.detach().norm(dim=-1).mean()
                if shared_context.numel() > 0
                else q_global.new_tensor(0.0)
            ),
        }
        return _finite_tensor(q_text), _finite_tensor(q_audio), _finite_tensor(q_shared), _finite_tensor(q_global), diagnostics


class HierarchicalSPShQueryInteractor(nn.Module):
    """Hierarchical query-mediated shared/private interaction head.

    It reads final text/audio evidence units plus per-unit SP/SH registers and
    produces a representation-level fusion logit. Local SP/SH evidence affects
    classification through query states, not through a direct logit residual.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        num_heads: int,
        dropout: float,
        depth: int = 2,
        num_private_queries: int = 2,
        num_shared_queries: int = 2,
        num_global_queries: int = 1,
        router_gate: bool = True,
        feature_modulation: bool = False,
        feature_modulation_scale: float = 0.05,
        mamba_mixer: bool = False,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.num_private_queries = max(0, int(num_private_queries))
        self.num_shared_queries = max(0, int(num_shared_queries))
        self.num_global_queries = max(1, int(num_global_queries))
        self.router_gate = bool(router_gate)
        self.feature_modulation = bool(feature_modulation)
        self.feature_modulation_scale = float(feature_modulation_scale)
        self.mamba_mixer = bool(mamba_mixer)

        self.text_private_queries = nn.Parameter(torch.randn(self.num_private_queries, hidden_dim) * 0.02)
        self.audio_private_queries = nn.Parameter(torch.randn(self.num_private_queries, hidden_dim) * 0.02)
        self.shared_queries = nn.Parameter(torch.randn(self.num_shared_queries, hidden_dim) * 0.02)
        self.global_queries = nn.Parameter(torch.randn(self.num_global_queries, hidden_dim) * 0.02)
        self.layers = nn.ModuleList(
            [HQAInteractionLayer(hidden_dim, num_heads, dropout) for _ in range(max(1, int(depth)))]
        )
        router_hidden = max(16, hidden_dim // 2)
        self.router = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5),
            nn.Linear(hidden_dim * 5, router_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden, 4),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        classifier_input_dim = hidden_dim * 5 + 4
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Linear(classifier_input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )
        self.aux_classifier = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes))
        if self.feature_modulation:
            self.class_modulator = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 2),
            )
        else:
            self.class_modulator = None

    def _expand_queries(self, queries: torch.Tensor, batch_size: int) -> torch.Tensor:
        return queries.unsqueeze(0).expand(batch_size, -1, -1)

    def _ensure_mask(self, mask: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        mask = mask.bool()
        if mask.size(1) != tokens.size(1):
            mask = torch.ones(tokens.shape[:2], device=tokens.device, dtype=torch.bool)
        if not mask.any(dim=1).all():
            mask = mask.clone()
            mask[~mask.any(dim=1), 0] = True
        return mask

    def _weighted_unit_loss(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = self._ensure_mask(mask, values.unsqueeze(-1))
        weights = mask.to(dtype=values.dtype)
        return _finite_tensor((values * weights).sum() / weights.sum().clamp_min(1.0))

    def _shared_alignment_loss(
        self,
        text_shared_units: torch.Tensor,
        audio_shared_units: torch.Tensor,
        text_mask: torch.Tensor,
        audio_mask: torch.Tensor,
    ) -> torch.Tensor:
        pair_mask = text_mask.bool() & audio_mask.bool()
        similarity = F.cosine_similarity(
            F.normalize(_finite_tensor(text_shared_units), dim=-1),
            F.normalize(_finite_tensor(audio_shared_units), dim=-1),
            dim=-1,
        )
        return self._weighted_unit_loss(1.0 - similarity, pair_mask)

    def _private_orth_loss(
        self,
        text_specific_units: torch.Tensor,
        text_shared_units: torch.Tensor,
        audio_specific_units: torch.Tensor,
        audio_shared_units: torch.Tensor,
        text_mask: torch.Tensor,
        audio_mask: torch.Tensor,
    ) -> torch.Tensor:
        text_sp = F.normalize(_finite_tensor(text_specific_units), dim=-1)
        text_sh = F.normalize(_finite_tensor(text_shared_units), dim=-1)
        audio_sp = F.normalize(_finite_tensor(audio_specific_units), dim=-1)
        audio_sh = F.normalize(_finite_tensor(audio_shared_units), dim=-1)
        text_loss = self._weighted_unit_loss(F.cosine_similarity(text_sp, text_sh, dim=-1).pow(2), text_mask)
        audio_loss = self._weighted_unit_loss(F.cosine_similarity(audio_sp, audio_sh, dim=-1).pow(2), audio_mask)
        pair_mask = text_mask.bool() & audio_mask.bool()
        cross_loss = self._weighted_unit_loss(F.cosine_similarity(text_sp, audio_sp, dim=-1).pow(2), pair_mask)
        return _finite_tensor((text_loss + audio_loss + cross_loss) / 3.0)

    def _pool_queries(self, query_parts, reference: torch.Tensor) -> torch.Tensor:
        valid_parts = [part for part in query_parts if part is not None and part.size(1) > 0]
        if not valid_parts:
            return reference.new_zeros(reference.size(0), reference.size(-1))
        return _finite_tensor(torch.cat(valid_parts, dim=1)).mean(dim=1)

    def forward(
        self,
        text_units: torch.Tensor,
        audio_units: torch.Tensor,
        text_specific_units: torch.Tensor,
        text_shared_units: torch.Tensor,
        audio_specific_units: torch.Tensor,
        audio_shared_units: torch.Tensor,
        class_reps: torch.Tensor,
        unit_mask: torch.Tensor,
        text_unit_mask: torch.Tensor = None,
        audio_unit_mask: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        text_units = _finite_tensor(text_units)
        audio_units = _finite_tensor(audio_units)
        class_reps = _finite_tensor(class_reps)
        batch_size = class_reps.size(0)
        unit_mask = self._ensure_mask(unit_mask, text_units)
        text_unit_mask = self._ensure_mask(unit_mask if text_unit_mask is None else text_unit_mask, text_units)
        audio_unit_mask = self._ensure_mask(unit_mask if audio_unit_mask is None else audio_unit_mask, audio_units)

        q_text = self._expand_queries(self.text_private_queries, batch_size)
        q_audio = self._expand_queries(self.audio_private_queries, batch_size)
        q_shared = self._expand_queries(self.shared_queries, batch_size)
        q_global = self._expand_queries(self.global_queries, batch_size)

        text_memory = torch.cat([text_units, _finite_tensor(text_specific_units)], dim=1)
        text_memory_mask = torch.cat([text_unit_mask, text_unit_mask], dim=1)
        audio_memory = torch.cat([audio_units, _finite_tensor(audio_specific_units)], dim=1)
        audio_memory_mask = torch.cat([audio_unit_mask, audio_unit_mask], dim=1)
        shared_memory = torch.cat([_finite_tensor(text_shared_units), _finite_tensor(audio_shared_units)], dim=1)
        shared_memory_mask = torch.cat([text_unit_mask, audio_unit_mask], dim=1)

        layer_gates = []
        text_context_norms = []
        audio_context_norms = []
        shared_context_norms = []
        for layer in self.layers:
            q_text, q_audio, q_shared, q_global, diag = layer(
                q_text,
                q_audio,
                q_shared,
                q_global,
                text_memory,
                text_memory_mask,
                audio_memory,
                audio_memory_mask,
                shared_memory,
                shared_memory_mask,
                class_reps,
            )
            layer_gates.append(diag["global_gate"])
            text_context_norms.append(diag["text_context_norm"])
            audio_context_norms.append(diag["audio_context_norm"])
            shared_context_norms.append(diag["shared_context_norm"])

        h_text = _masked_mean_sequence(text_units, text_unit_mask)
        h_audio = _masked_mean_sequence(audio_units, audio_unit_mask)
        h_class = _finite_tensor(class_reps).mean(dim=1)
        h_query = self._pool_queries([q_text, q_audio, q_shared, q_global], h_class)
        h_global = _finite_tensor(q_global).mean(dim=1)
        if self.class_modulator is not None:
            scale_shift = self.class_modulator(h_global).view(batch_size, 2, self.hidden_dim)
            scale = torch.tanh(scale_shift[:, 0, :]).unsqueeze(1)
            shift = torch.tanh(scale_shift[:, 1, :]).unsqueeze(1)
            class_reps = _finite_tensor(
                class_reps
                + class_reps.new_tensor(self.feature_modulation_scale)
                * (scale * F.layer_norm(class_reps, (class_reps.size(-1),)) + shift)
            )
            h_class = class_reps.mean(dim=1)

        router_input = torch.cat([h_class, h_text, h_audio, h_query, torch.abs(h_text - h_audio)], dim=-1)
        if self.router_gate:
            router_weights = torch.softmax(_finite_tensor(self.router(router_input)), dim=-1)
        else:
            router_weights = h_class.new_full((batch_size, 4), 0.25)
        components = torch.stack([h_class, h_text, h_audio, h_query], dim=1)
        routed_feature = _finite_tensor((router_weights.unsqueeze(-1) * components).sum(dim=1))
        classifier_input = torch.cat([h_class, h_text, h_audio, h_query, routed_feature, router_weights], dim=-1)
        logits = _finite_tensor(self.classifier(_finite_tensor(classifier_input)), limit=1e4)
        aux_logits = _finite_tensor(self.aux_classifier(h_global), limit=1e4)
        shared_alignment_loss = self._shared_alignment_loss(
            text_shared_units,
            audio_shared_units,
            text_unit_mask,
            audio_unit_mask,
        )
        private_orth_loss = self._private_orth_loss(
            text_specific_units,
            text_shared_units,
            audio_specific_units,
            audio_shared_units,
            text_unit_mask,
            audio_unit_mask,
        )
        return {
            "hqa_logits": logits,
            "hqa_mlp_logits": logits,
            "hqa_aux_logits": aux_logits,
            "hqa_class_reps": _finite_tensor(class_reps),
            "hqa_rep": _finite_tensor(h_query),
            "hqa_global_rep": _finite_tensor(h_global),
            "hqa_text_private_queries": _finite_tensor(q_text),
            "hqa_audio_private_queries": _finite_tensor(q_audio),
            "hqa_shared_queries": _finite_tensor(q_shared),
            "hqa_global_queries": _finite_tensor(q_global),
            "hqa_router_weights": router_weights.detach(),
            "hqa_router_base_weight": router_weights[:, 0].detach().mean(),
            "hqa_router_text_weight": router_weights[:, 1].detach().mean(),
            "hqa_router_audio_weight": router_weights[:, 2].detach().mean(),
            "hqa_router_query_weight": router_weights[:, 3].detach().mean(),
            "hqa_global_gate": torch.stack(layer_gates).mean() if layer_gates else logits.new_tensor(0.0),
            "hqa_text_context_norm": (
                torch.stack(text_context_norms).mean() if text_context_norms else logits.new_tensor(0.0)
            ),
            "hqa_audio_context_norm": (
                torch.stack(audio_context_norms).mean() if audio_context_norms else logits.new_tensor(0.0)
            ),
            "hqa_shared_context_norm": (
                torch.stack(shared_context_norms).mean() if shared_context_norms else logits.new_tensor(0.0)
            ),
            "hqa_shared_alignment_loss": shared_alignment_loss,
            "hqa_private_orth_loss": private_orth_loss,
            "hqa_mamba_requested": logits.new_tensor(1.0 if self.mamba_mixer else 0.0),
            "hqa_feature_modulation_enabled": logits.new_tensor(1.0 if self.feature_modulation else 0.0),
        }


class FingerprintReconstructionDecoder(nn.Module):
    """Masked-token decoder that reconstructs modality tokens from Q/FP bottleneck tokens."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))

    def forward(
        self,
        target_queries: torch.Tensor,
        source_tokens: torch.Tensor,
        source_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        decoded = _finite_tensor(target_queries)
        source_tokens = _finite_tensor(source_tokens)
        key_padding_mask = None if source_mask is None else ~source_mask.bool()
        for layer in self.layers:
            decoded = layer(decoded, source_tokens, memory_key_padding_mask=key_padding_mask)
            decoded = _finite_tensor(decoded)
        return _finite_tensor(self.output(decoded))


class DirectSPShQueryAdapter(nn.Module):
    # SP/SH assists Q through cross-attention and an FFN residual, without a sigmoid gate.
    # Final classification can still remain Q-only; this module only rewrites Q.

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        scale: float = 1.0,
        ffn_ratio: float = 2.0,
    ):
        super().__init__()
        self.scale = float(scale)
        ffn_dim = max(hidden_dim, int(hidden_dim * float(ffn_ratio)))
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.query_to_sp_sh = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.delta = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        queries: torch.Tensor,
        sp_sh_memory: torch.Tensor,
        sp_sh_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.scale <= 0:
            zeros = queries.new_zeros(queries.shape)
            return queries, zeros, zeros
        queries = _finite_tensor(queries)
        memory_mask = sp_sh_mask.bool()
        memory = _finite_tensor(sp_sh_memory).masked_fill(~memory_mask.unsqueeze(-1), 0.0)
        context, _ = self.query_to_sp_sh(
            self.query_norm(queries),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=~memory_mask,
            need_weights=False,
        )
        context = _finite_tensor(context)
        delta_input = torch.cat(
            [queries, context, torch.abs(queries - context), queries * context],
            dim=-1,
        )
        delta = _finite_tensor(self.delta(_finite_tensor(delta_input)))
        updated = _finite_tensor(self.out_norm(queries + self.scale * delta))
        return updated, context, delta


class IdentityInitSPShQueryAdapter(nn.Module):
    """Identity-initialized late SP/SH memory update for stable warm-starting.

    The ordinary direct adapter normalizes the residual output, so even a tiny
    delta can move a checkpoint that is already strong. This adapter keeps the
    residual path mathematically zero at initialization and applies no output
    LayerNorm, letting SP/SH tokens learn only the corrections supported by the
    held-out fold.
    """

    is_identity_sp_sh_query_adapter = True

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        scale: float = 0.10,
        ffn_ratio: float = 1.0,
    ):
        super().__init__()
        self.scale = float(scale)
        ffn_dim = max(hidden_dim, int(hidden_dim * float(ffn_ratio)))
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.query_to_sp_sh = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.delta_norm = nn.LayerNorm(hidden_dim * 4)
        self.delta_fc1 = nn.Linear(hidden_dim * 4, ffn_dim)
        self.delta_act = nn.GELU()
        self.delta_dropout = nn.Dropout(dropout)
        self.delta_fc2 = nn.Linear(ffn_dim, hidden_dim)
        self.output_dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.delta_fc2.weight)
        nn.init.zeros_(self.delta_fc2.bias)

    def _safe_memory(
        self,
        memory: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = mask.bool()
        missing_rows = ~mask.any(dim=1)
        memory = _finite_tensor(memory)
        if missing_rows.any():
            memory = memory.clone()
            mask = mask.clone()
            memory[missing_rows, 0, :] = 0.0
            mask[missing_rows, 0] = True
        memory = memory.masked_fill(~mask.unsqueeze(-1), 0.0)
        return memory, mask

    def forward(
        self,
        queries: torch.Tensor,
        sp_sh_memory: torch.Tensor,
        sp_sh_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.scale <= 0:
            zeros = queries.new_zeros(queries.shape)
            return queries, zeros, zeros
        queries = _finite_tensor(queries)
        memory, memory_mask = self._safe_memory(sp_sh_memory, sp_sh_mask)
        norm_memory = self.memory_norm(memory)
        context, _ = self.query_to_sp_sh(
            self.query_norm(queries),
            norm_memory,
            norm_memory,
            key_padding_mask=~memory_mask,
            need_weights=False,
        )
        context = _finite_tensor(context)
        delta_input = _finite_tensor(
            torch.cat([queries, context, torch.abs(queries - context), queries * context], dim=-1)
        )
        delta = self.delta_fc2(
            self.delta_dropout(self.delta_act(self.delta_fc1(self.delta_norm(delta_input))))
        )
        delta = _finite_tensor(self.output_dropout(delta))
        updated = _finite_tensor(queries + queries.new_tensor(self.scale) * delta)
        return updated, context, delta


class ClassFeatureResidualLogitAdapter(nn.Module):
    """Small identity-initialized residual classifier over final class features."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        adapter_dim: int = 128,
        gate_init: float = -4.0,
        max_scale: float = 0.50,
    ):
        super().__init__()
        self.max_scale = float(max_scale)
        adapter_dim = max(16, int(adapter_dim))
        input_dim = hidden_dim * 8
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, adapter_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(adapter_dim, 1)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        base_logits: torch.Tensor,
        class_reps: torch.Tensor,
        text_pool: torch.Tensor,
        audio_pool: torch.Tensor,
        global_rep: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        class_reps = _finite_tensor(class_reps)
        batch_size, num_classes, hidden_dim = class_reps.shape
        text_pool = _finite_tensor(text_pool).unsqueeze(1).expand(batch_size, num_classes, hidden_dim)
        audio_pool = _finite_tensor(audio_pool).unsqueeze(1).expand(batch_size, num_classes, hidden_dim)
        global_rep = _finite_tensor(global_rep).unsqueeze(1).expand(batch_size, num_classes, hidden_dim)
        adapter_input = torch.cat(
            [
                class_reps,
                text_pool,
                audio_pool,
                global_rep,
                torch.abs(class_reps - text_pool),
                torch.abs(class_reps - audio_pool),
                class_reps * text_pool,
                class_reps * audio_pool,
            ],
            dim=-1,
        )
        delta = self.fc2(self.dropout(self.act(self.fc1(self.norm(_finite_tensor(adapter_input))))))
        delta = _finite_tensor(delta.squeeze(-1), limit=1e4)
        gate = base_logits.new_tensor(self.max_scale) * torch.sigmoid(
            self.gate_logit.to(device=base_logits.device, dtype=base_logits.dtype)
        )
        logits = _finite_tensor(base_logits + gate * delta, limit=1e4)
        return logits, {
            "class_feature_residual_delta": delta.detach(),
            "class_feature_residual_delta_norm": delta.detach().norm(dim=-1).mean(),
            "class_feature_residual_gate": gate.detach(),
        }


class ParalinguisticStyleQueryAdapter(nn.Module):
    """Residual logit adapter for non-lexical speaker/style cues.

    The cached datasets do not contain explicit age, speaker attitude, or tone
    labels. This module therefore derives a style memory from available audio
    embeddings and audio-text mismatch:

    - audio center: speaker/timbre/prosody aggregate;
    - audio variation: within-subject temporal variation over evidence units;
    - audio-text gap: how much vocal evidence disagrees with lexical evidence.

    Class queries attend to this memory, then produce a zero-initialized residual
    logit. At initialization the full model is identical to the source
    checkpoint.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        adapter_dim: int = 128,
        gate_init: float = -2.0,
        max_scale: float = 0.50,
    ):
        super().__init__()
        self.max_scale = float(max_scale)
        adapter_dim = max(16, int(adapter_dim))
        self.audio_var_norm = nn.LayerNorm(hidden_dim)
        self.gap_norm = nn.LayerNorm(hidden_dim)
        self.memory_mixer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.query_to_style = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        input_dim = hidden_dim * 5
        self.delta = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, adapter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, 1),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))

    @staticmethod
    def _masked_mean_var(tokens: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = _finite_tensor(tokens)
        mask = mask.bool()
        missing_rows = ~mask.any(dim=1)
        if missing_rows.any():
            mask = mask.clone()
            mask[missing_rows, 0] = True
        weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        mean = (tokens * weights).sum(dim=1) / denom
        centered = (tokens - mean.unsqueeze(1)).masked_fill(~mask.unsqueeze(-1), 0.0)
        var = (centered.pow(2) * weights).sum(dim=1) / denom
        return _finite_tensor(mean), _finite_tensor(torch.sqrt(var.clamp_min(1e-6)))

    def forward(
        self,
        logits: torch.Tensor,
        class_reps: torch.Tensor,
        text_units: torch.Tensor,
        audio_units: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        class_reps = _finite_tensor(class_reps)
        text_mean, text_var = self._masked_mean_var(text_units, unit_mask)
        audio_mean, audio_var = self._masked_mean_var(audio_units, unit_mask)
        gap = _finite_tensor(torch.abs(audio_mean - text_mean))
        style_tokens = torch.stack(
            [
                audio_mean,
                self.audio_var_norm(audio_var),
                self.gap_norm(gap),
                _finite_tensor(audio_mean * text_mean),
            ],
            dim=1,
        )
        style_tokens = _finite_tensor(self.memory_mixer(style_tokens))
        context, attn = self.query_to_style(
            self.query_norm(class_reps),
            self.memory_norm(style_tokens),
            self.memory_norm(style_tokens),
            need_weights=True,
        )
        context = _finite_tensor(context)
        interaction = torch.cat(
            [
                class_reps,
                context,
                torch.abs(class_reps - context),
                class_reps * context,
                gap.unsqueeze(1).expand(-1, class_reps.size(1), -1),
            ],
            dim=-1,
        )
        delta = _finite_tensor(self.delta(_finite_tensor(interaction)).squeeze(-1), limit=1e4)
        gate = logits.new_tensor(self.max_scale) * torch.sigmoid(
            self.gate_logit.to(device=logits.device, dtype=logits.dtype)
        )
        adapted_logits = _finite_tensor(logits + gate * delta, limit=1e4)
        diagnostics = {
            "paralinguistic_style_delta": delta.detach(),
            "paralinguistic_style_delta_norm": delta.detach().norm(dim=-1).mean(),
            "paralinguistic_style_gate": gate.detach(),
            "paralinguistic_audio_var_norm": audio_var.detach().norm(dim=-1).mean(),
            "paralinguistic_audio_text_gap_norm": gap.detach().norm(dim=-1).mean(),
        }
        if attn is not None:
            diagnostics["paralinguistic_style_attention"] = attn.detach()
        return adapted_logits, diagnostics


class AcousticStyleQueryAdapter(nn.Module):
    """Query adapter over raw acoustic style statistics.

    This module reads the cached frame/token-level audio embedding directly,
    before the RefFormer evidence stack compresses it.  It builds per-unit style
    tokens from the audio center, within-unit variation, and adjacent-frame
    change magnitude. These statistics are rough but available proxies for
    tone, timbre, speaking dynamics, and other non-lexical speaker factors.
    """

    def __init__(
        self,
        audio_input_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        adapter_dim: int = 128,
        gate_init: float = -1.0,
        max_scale: float = 0.50,
    ):
        super().__init__()
        self.max_scale = float(max_scale)
        adapter_dim = max(16, int(adapter_dim))
        self.stat_projector = nn.Sequential(
            nn.LayerNorm(audio_input_dim * 3),
            nn.Linear(audio_input_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.query_to_style = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        input_dim = hidden_dim * 5
        self.delta = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, adapter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, 1),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))

    @staticmethod
    def _audio_stats(
        audio_features: torch.Tensor,
        audio_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_features = _finite_tensor(audio_features)
        mask = audio_attention_mask.bool()
        if mask.dim() != 3:
            raise ValueError("audio_attention_mask must have shape [batch, units, frames]")
        unit_valid = mask.any(dim=-1)
        if not unit_valid.any(dim=1).all():
            unit_valid = unit_valid.clone()
            mask = mask.clone()
            missing_rows = ~unit_valid.any(dim=1)
            unit_valid[missing_rows, 0] = True
            mask[missing_rows, 0, 0] = True

        weights = mask.to(dtype=audio_features.dtype).unsqueeze(-1)
        denom = weights.sum(dim=2).clamp_min(1.0)
        mean = (audio_features * weights).sum(dim=2) / denom
        centered = (audio_features - mean.unsqueeze(2)).masked_fill(~mask.unsqueeze(-1), 0.0)
        var = (centered.pow(2) * weights).sum(dim=2) / denom
        std = torch.sqrt(var.clamp_min(1e-6))

        if audio_features.size(2) > 1:
            pair_mask = mask[:, :, 1:] & mask[:, :, :-1]
            diffs = torch.abs(audio_features[:, :, 1:, :] - audio_features[:, :, :-1, :])
            pair_weights = pair_mask.to(dtype=audio_features.dtype).unsqueeze(-1)
            pair_denom = pair_weights.sum(dim=2).clamp_min(1.0)
            delta = (diffs * pair_weights).sum(dim=2) / pair_denom
            delta = delta.masked_fill(~pair_mask.any(dim=2, keepdim=True), 0.0)
        else:
            delta = torch.zeros_like(mean)

        stats = torch.cat([mean, std, delta], dim=-1)
        stats = stats.masked_fill(~unit_valid.unsqueeze(-1), 0.0)
        return _finite_tensor(stats), unit_valid

    @staticmethod
    def _masked_mean_var(tokens: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = mask.bool()
        weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        mean = (tokens * weights).sum(dim=1) / denom
        centered = (tokens - mean.unsqueeze(1)).masked_fill(~mask.unsqueeze(-1), 0.0)
        var = (centered.pow(2) * weights).sum(dim=1) / denom
        return _finite_tensor(mean), _finite_tensor(torch.sqrt(var.clamp_min(1e-6)))

    def forward(
        self,
        logits: torch.Tensor,
        class_reps: torch.Tensor,
        audio_features: torch.Tensor,
        audio_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        class_reps = _finite_tensor(class_reps)
        stats, unit_valid = self._audio_stats(audio_features, audio_attention_mask)
        style_units = _finite_tensor(self.stat_projector(stats))
        global_mean, global_var = self._masked_mean_var(style_units, unit_valid)
        memory = torch.cat([style_units, global_mean.unsqueeze(1), global_var.unsqueeze(1)], dim=1)
        summary_mask = torch.ones(
            unit_valid.size(0),
            2,
            dtype=torch.bool,
            device=unit_valid.device,
        )
        memory_mask = torch.cat([unit_valid, summary_mask], dim=1)
        context, attn = self.query_to_style(
            self.query_norm(class_reps),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=~memory_mask,
            need_weights=True,
        )
        context = _finite_tensor(context)
        global_context = global_mean.unsqueeze(1).expand(-1, class_reps.size(1), -1)
        interaction = torch.cat(
            [
                class_reps,
                context,
                torch.abs(class_reps - context),
                class_reps * context,
                global_context,
            ],
            dim=-1,
        )
        delta = _finite_tensor(self.delta(_finite_tensor(interaction)).squeeze(-1), limit=1e4)
        gate = logits.new_tensor(self.max_scale) * torch.sigmoid(
            self.gate_logit.to(device=logits.device, dtype=logits.dtype)
        )
        adapted_logits = _finite_tensor(logits + gate * delta, limit=1e4)
        diagnostics = {
            "acoustic_style_delta": delta.detach(),
            "acoustic_style_delta_norm": delta.detach().norm(dim=-1).mean(),
            "acoustic_style_gate": gate.detach(),
            "acoustic_style_unit_valid_mean": unit_valid.to(dtype=logits.dtype).mean().detach(),
            "acoustic_style_global_mean_norm": global_mean.detach().norm(dim=-1).mean(),
            "acoustic_style_global_var_norm": global_var.detach().norm(dim=-1).mean(),
        }
        if attn is not None:
            diagnostics["acoustic_style_attention"] = attn.detach()
        return adapted_logits, diagnostics


class RoutedSPShQueryAdapter(nn.Module):
    """Reliability-routed direct SP/SH memory update for referential queries.

    The adapter lets SP/SH tokens participate in the main QAM path while keeping
    the residual small when the SP/SH memory is uncertain.  It is intentionally
    query-wise: each class query can accept or reject the SP/SH evidence
    independently.
    """

    is_routed_sp_sh_query_adapter = True

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        scale: float = 0.10,
        gate_init: float = -2.5,
        ffn_ratio: float = 2.0,
    ):
        super().__init__()
        self.scale = float(scale)
        ffn_dim = max(hidden_dim, int(hidden_dim * float(ffn_ratio)))
        gate_hidden = max(16, hidden_dim // 2)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.query_to_sp_sh = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.delta = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.reliability_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5),
            nn.Linear(hidden_dim * 5, gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 1),
        )
        nn.init.zeros_(self.reliability_gate[-1].weight)
        nn.init.constant_(self.reliability_gate[-1].bias, float(gate_init))
        self.out_norm = nn.LayerNorm(hidden_dim)

    def _safe_memory(
        self,
        memory: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = mask.bool()
        missing_rows = ~mask.any(dim=1)
        memory = _finite_tensor(memory)
        if missing_rows.any():
            memory = memory.clone()
            mask = mask.clone()
            memory[missing_rows, 0, :] = 0.0
            mask[missing_rows, 0] = True
        memory = memory.masked_fill(~mask.unsqueeze(-1), 0.0)
        return memory, mask

    def forward(
        self,
        queries: torch.Tensor,
        sp_sh_memory: torch.Tensor,
        sp_sh_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.scale <= 0:
            zeros = queries.new_zeros(queries.size(0), queries.size(1), 1)
            return queries, queries.new_zeros(queries.shape), zeros
        queries = _finite_tensor(queries)
        memory, memory_mask = self._safe_memory(sp_sh_memory, sp_sh_mask)
        norm_memory = self.memory_norm(memory)
        context, _ = self.query_to_sp_sh(
            self.query_norm(queries),
            norm_memory,
            norm_memory,
            key_padding_mask=~memory_mask,
            need_weights=False,
        )
        context = _finite_tensor(context)
        interaction = _finite_tensor(
            torch.cat([queries, context, torch.abs(queries - context), queries * context], dim=-1)
        )
        delta = _finite_tensor(self.delta(interaction))
        weights = memory_mask.to(dtype=memory.dtype).unsqueeze(-1)
        memory_pool = (memory * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        memory_pool = memory_pool.unsqueeze(1).expand(-1, queries.size(1), -1)
        gate_input = _finite_tensor(
            torch.cat(
                [
                    queries,
                    context,
                    torch.abs(queries - context),
                    queries * context,
                    memory_pool,
                ],
                dim=-1,
            )
        )
        gate = torch.sigmoid(self.reliability_gate(gate_input))
        updated = _finite_tensor(queries + queries.new_tensor(self.scale) * gate * delta)
        return updated, context, gate


class VisualPromptMoESPShQueryAdapter(nn.Module):
    """Token-mixing MoE SP/SH adapter inspired by recent vision prompt designs.

    The adapter keeps the RefFormer QA loop intact: it rewrites only the
    referential queries using SP/SH memory tokens.  Unlike the old gated variant,
    it does not rely on a low sigmoid gate.  The SP/SH memory is first refined by
    a small Transformer token mixer, then each query attends to that memory and a
    softmax router combines several lightweight delta experts.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        scale: float = 1.0,
        num_experts: int = 3,
        ffn_ratio: float = 2.0,
        mixer_layers: int = 1,
    ):
        super().__init__()
        self.scale = float(scale)
        self.num_experts = max(1, int(num_experts))
        ffn_dim = max(hidden_dim, int(hidden_dim * float(ffn_ratio)))
        mixer_layers = max(0, int(mixer_layers))
        self.memory_mixer = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=max(hidden_dim, hidden_dim * 2),
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(mixer_layers)
            ]
        )
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.query_to_prompt = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        interaction_dim = hidden_dim * 4
        self.router = nn.Sequential(
            nn.LayerNorm(interaction_dim),
            nn.Linear(interaction_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, self.num_experts),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(interaction_dim),
                    nn.Linear(interaction_dim, ffn_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_dim, hidden_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(self.num_experts)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def _mix_memory(self, memory: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mixed = _finite_tensor(memory)
        key_padding_mask = ~mask.bool()
        for layer in self.memory_mixer:
            mixed = _finite_tensor(layer(mixed, src_key_padding_mask=key_padding_mask))
        return mixed

    def forward(
        self,
        queries: torch.Tensor,
        sp_sh_memory: torch.Tensor,
        sp_sh_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        queries = _finite_tensor(queries)
        memory_mask = sp_sh_mask.bool()
        missing_rows = ~memory_mask.any(dim=1)
        memory = _finite_tensor(sp_sh_memory)
        if missing_rows.any():
            memory = memory.clone()
            memory_mask = memory_mask.clone()
            memory[missing_rows, 0, :] = 0.0
            memory_mask[missing_rows, 0] = True
        memory = self._mix_memory(memory, memory_mask)
        context, _ = self.query_to_prompt(
            self.query_norm(queries),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=~memory_mask,
            need_weights=False,
        )
        context = _finite_tensor(context)
        interaction = _finite_tensor(
            torch.cat([queries, context, torch.abs(queries - context), queries * context], dim=-1)
        )
        router_weights = torch.softmax(_finite_tensor(self.router(interaction)), dim=-1)
        expert_deltas = torch.stack(
            [_finite_tensor(expert(interaction)) for expert in self.experts],
            dim=-2,
        )
        delta = torch.sum(router_weights.unsqueeze(-1) * expert_deltas, dim=-2)
        updated = _finite_tensor(self.out_norm(queries + queries.new_tensor(self.scale) * delta))
        return updated, context, router_weights


class DualPathSPShQueryAdapter(nn.Module):
    """Sample-wise mixture of gated QA and direct QA over SP/SH register tokens."""

    is_dual_path_sp_sh_query_adapter = True

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        gated_scale: float = 0.03,
        direct_scale: float = 0.05,
        ffn_ratio: float = 2.0,
        gate_init: float = -0.5,
        gate_mode: str = "learned",
        fixed_direct_weight: Optional[float] = None,
    ):
        super().__init__()
        self.gate_mode = str(gate_mode or "learned").lower()
        if self.gate_mode == "fixed_gated":
            fixed_direct_weight = 0.0
        elif self.gate_mode == "fixed_direct":
            fixed_direct_weight = 1.0
        if fixed_direct_weight is not None:
            fixed_direct_weight = max(0.0, min(1.0, float(fixed_direct_weight)))
        self.fixed_direct_weight = fixed_direct_weight
        self.gated_adapter = QueryAdaptionModule(hidden_dim, num_heads, dropout, gated_scale)
        self.direct_adapter = DirectSPShQueryAdapter(
            hidden_dim,
            num_heads,
            dropout,
            scale=direct_scale,
            ffn_ratio=ffn_ratio,
        )
        gate_hidden = max(16, hidden_dim // 2)
        self.path_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5),
            nn.Linear(hidden_dim * 5, gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 1),
        )
        nn.init.zeros_(self.path_gate[-1].weight)
        nn.init.constant_(self.path_gate[-1].bias, float(gate_init))
        self.out_norm = nn.LayerNorm(hidden_dim)

    def combine_paths(
        self,
        queries: torch.Tensor,
        gated_queries: torch.Tensor,
        gated_context: torch.Tensor,
        direct_queries: torch.Tensor,
        direct_context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        queries = _finite_tensor(queries)
        gated_queries = _finite_tensor(gated_queries)
        direct_queries = _finite_tensor(direct_queries)
        gated_context = _finite_tensor(gated_context)
        direct_context = _finite_tensor(direct_context)
        query_pool = queries.mean(dim=1)
        gated_pool = gated_context.mean(dim=1)
        direct_pool = direct_context.mean(dim=1)
        if self.gate_mode in {"fixed", "constant", "fixed_direct", "fixed_gated"} or self.fixed_direct_weight is not None:
            direct_weight = queries.new_full((queries.size(0), 1, 1), float(self.fixed_direct_weight or 0.0))
        else:
            gate_input = torch.cat(
                [
                    query_pool,
                    gated_pool,
                    direct_pool,
                    torch.abs(gated_pool - direct_pool),
                    gated_pool * direct_pool,
                ],
                dim=-1,
            )
            direct_weight = torch.sigmoid(self.path_gate(_finite_tensor(gate_input))).view(-1, 1, 1)
        updated = (1.0 - direct_weight) * gated_queries + direct_weight * direct_queries
        return _finite_tensor(updated), direct_weight

    def forward(
        self,
        queries: torch.Tensor,
        sp_sh_memory: torch.Tensor,
        sp_sh_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gated_queries, gated_context, _ = self.gated_adapter(queries, sp_sh_memory, sp_sh_mask)
        direct_queries, direct_context, _ = self.direct_adapter(queries, sp_sh_memory, sp_sh_mask)
        updated, direct_weight = self.combine_paths(
            queries,
            gated_queries,
            gated_context,
            direct_queries,
            direct_context,
        )
        mixed_context = (1.0 - direct_weight) * gated_context + direct_weight * direct_context
        return updated, _finite_tensor(mixed_context), direct_weight


class ModalityDecoupledQueryModel(RefFormerAudioTextEmotionModel):
    """RefFormer + MDReID-style shared/specific modality decoupling."""

    def __init__(self, config: Dict[str, object]):
        super().__init__(config)
        model_config = config.get("model", {})
        training_config = config.get("training", {})
        num_heads = int(model_config.get("num_heads", 4))

        self.decoupled_logit_scale = float(model_config.get("decoupled_logit_scale", 0.25))
        self.learnable_decoupled_logit_gate = bool(model_config.get("learnable_decoupled_logit_gate", False))
        self.decoupled_logit_gate_max = float(
            model_config.get("decoupled_logit_gate_max", self.decoupled_logit_scale)
        )
        self.decoupled_logit_gate_init = float(model_config.get("decoupled_logit_gate_init", -6.0))
        if self.learnable_decoupled_logit_gate:
            self.decoupled_logit_gate = nn.Parameter(
                torch.tensor(self.decoupled_logit_gate_init, dtype=torch.float32)
            )
        self.sp_sh_current_epoch = 0
        self.sp_sh_residual_start_epoch = int(
            training_config.get(
                "sp_sh_residual_start_epoch",
                model_config.get("sp_sh_residual_start_epoch", 0),
            )
            or 0
        )
        self.sp_sh_residual_ramp_epochs = int(
            training_config.get(
                "sp_sh_residual_ramp_epochs",
                model_config.get("sp_sh_residual_ramp_epochs", 0),
            )
            or 0
        )
        self.sp_sh_aux_start_epoch = int(
            training_config.get(
                "sp_sh_aux_start_epoch",
                model_config.get("sp_sh_aux_start_epoch", 0),
            )
            or 0
        )
        self.sp_sh_aux_ramp_epochs = int(
            training_config.get(
                "sp_sh_aux_ramp_epochs",
                model_config.get("sp_sh_aux_ramp_epochs", 0),
            )
            or 0
        )
        self.samplewise_decoupled_logit_gate = bool(
            model_config.get("samplewise_decoupled_logit_gate", False)
        )
        sample_gate_hidden = max(16, self.hidden_dim // 2)
        self.decoupled_sample_gate = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, sample_gate_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(sample_gate_hidden, 1),
        )
        sample_gate_init = float(
            model_config.get("decoupled_sample_gate_init", self.decoupled_logit_gate_init)
        )
        nn.init.zeros_(self.decoupled_sample_gate[-1].weight)
        nn.init.constant_(self.decoupled_sample_gate[-1].bias, sample_gate_init)
        self.final_logit_source = str(model_config.get("final_logit_source", "base_plus_decoupled")).lower()
        source_aliases = {
            "base+decoupled": "base_plus_decoupled",
            "base_decoupled": "base_plus_decoupled",
            "sp_sh": "decoupled",
            "sp-sh": "decoupled",
            "decoupled_only": "decoupled",
            "q_only": "sidebranch_sp_sh_query",
            "query_only": "sidebranch_sp_sh_query",
            "sp_sh_query": "sidebranch_sp_sh_query",
            "sidebranch_query": "sidebranch_sp_sh_query",
            "q_plus_base": "sidebranch_sp_sh_query_plus_base_residual",
            "query_plus_base": "sidebranch_sp_sh_query_plus_base_residual",
            "sp_sh_query_plus_base": "sidebranch_sp_sh_query_plus_base_residual",
            "hqa": "hqa_spsh",
            "hqa_sp_sh": "hqa_spsh",
            "hierarchical_query": "hqa_spsh",
            "query": "base",
            "qa": "base",
            "static_q": "static_query",
            "static": "static_query",
            "pooled_modalities": "modal_pool",
            "modal_pooled": "modal_pool",
            "no_q": "modal_pool",
            "no_query": "modal_pool",
        }
        self.final_logit_source = source_aliases.get(self.final_logit_source, self.final_logit_source)
        if self.final_logit_source not in {
            "base_plus_decoupled",
            "base",
            "static_query",
            "modal_pool",
            "decoupled",
            "sidebranch_sp_sh_query",
            "sidebranch_sp_sh_query_plus_base_residual",
            "hqa_spsh",
        }:
            raise ValueError(
                "model.final_logit_source must be one of: "
                "base_plus_decoupled, base, static_query, modal_pool, decoupled, "
                "sidebranch_sp_sh_query, sidebranch_sp_sh_query_plus_base_residual, hqa_spsh"
            )
        self.decoupled_classifier_weight = float(training_config.get("decoupled_classifier_weight", 0.10))
        self.rol_loss_weight = float(training_config.get("rol_loss_weight", 0.05))
        self.decoupled_only_loss = bool(
            training_config.get("decoupled_only_loss", model_config.get("decoupled_only_loss", False))
        )
        self.kdl_loss_weight = float(training_config.get("kdl_loss_weight", 0.05))
        self.kdl_temperature = float(training_config.get("kdl_temperature", 1.0))
        self.sp_sh_private_loss_weight = float(training_config.get("sp_sh_private_loss_weight", 0.0))
        self.sp_sh_shared_loss_weight = float(training_config.get("sp_sh_shared_loss_weight", 0.0))
        self.sp_sh_consistency_loss_weight = float(training_config.get("sp_sh_consistency_loss_weight", 0.0))
        self.sp_sh_disentangle_loss_weight = float(training_config.get("sp_sh_disentangle_loss_weight", 0.0))
        self.sp_sh_shared_contrastive_loss_weight = float(
            training_config.get("sp_sh_shared_contrastive_loss_weight", 0.0)
        )
        self.sp_sh_shared_contrastive_temperature = float(
            training_config.get("sp_sh_shared_contrastive_temperature", 0.07)
        )
        self.sp_sh_shared_contrastive_class_balanced = bool(
            training_config.get("sp_sh_shared_contrastive_class_balanced", True)
        )
        self.use_unit_spsh_mil_head = bool(model_config.get("use_unit_spsh_mil_head", False))
        self.unit_spsh_mil_loss_weight = float(training_config.get("unit_spsh_mil_loss_weight", 0.0))
        self.unit_spsh_mil_topk = int(model_config.get("unit_spsh_mil_topk", 3) or 3)
        self.unit_spsh_mil_residual_scale = float(model_config.get("unit_spsh_mil_residual_scale", 0.05))
        self.unit_spsh_mil_gate_init = float(model_config.get("unit_spsh_mil_gate_init", -4.0))
        self.unit_spsh_mil_logit_blend = float(model_config.get("unit_spsh_mil_logit_blend", 0.0))
        self.unit_spsh_mil_output_mode = str(model_config.get("unit_spsh_mil_output_mode", "replace")).lower()
        self.unit_spsh_mil_logit_residual_scale = float(
            model_config.get("unit_spsh_mil_logit_residual_scale", 1.0)
        )
        self.use_class_feature_residual_adapter = bool(
            model_config.get("use_class_feature_residual_adapter", False)
        )
        self.class_feature_residual_adapter_loss_weight = float(
            training_config.get("class_feature_residual_adapter_loss_weight", 0.0)
        )
        self.use_paralinguistic_style_adapter = bool(
            model_config.get("use_paralinguistic_style_adapter", False)
        )
        self.use_hqa_spsh_interactor = bool(model_config.get("use_hqa_spsh_interactor", False))
        self.hqa_depth = int(model_config.get("hqa_depth", 2) or 2)
        self.hqa_num_private_queries = int(model_config.get("hqa_num_private_queries", 2) or 0)
        self.hqa_num_shared_queries = int(model_config.get("hqa_num_shared_queries", 2) or 0)
        self.hqa_num_global_queries = int(model_config.get("hqa_num_global_queries", 1) or 1)
        self.hqa_heads = int(model_config.get("hqa_heads", num_heads) or num_heads)
        self.hqa_feature_modulation = bool(model_config.get("hqa_feature_modulation", False))
        self.hqa_feature_modulation_scale = float(model_config.get("hqa_feature_modulation_scale", 0.05))
        self.hqa_router_gate = bool(model_config.get("hqa_router_gate", True))
        self.hqa_mamba_mixer = bool(model_config.get("hqa_mamba_mixer", False))
        self.hqa_classifier_mode = str(model_config.get("hqa_classifier_mode", "mlp")).lower()
        self.hqa_logit_blend = float(model_config.get("hqa_logit_blend", 0.50))
        self.hqa_aux_cls_loss_weight = float(training_config.get("hqa_aux_cls_weight", 0.0))
        self.hqa_shared_alignment_weight = float(training_config.get("hqa_shared_alignment_weight", 0.0))
        self.hqa_private_orth_weight = float(training_config.get("hqa_private_orth_weight", 0.0))
        self.sp_sh_query_cls_loss_weight = float(
            training_config.get("sp_sh_query_cls_loss_weight", 0.0)
        )
        self.sidebranch_sp_sh_query_loss_weight = float(
            training_config.get("sidebranch_sp_sh_query_loss_weight", 0.0)
        )
        self.sp_sh_reconstruction_loss_weight = float(
            training_config.get("sp_sh_reconstruction_loss_weight", training_config.get("sp_sh_recon_loss_weight", 0.0))
        )
        self.sp_sh_reconstruction_normalize_targets = bool(training_config.get("sp_sh_reconstruction_normalize_targets", True))
        self.reconstruction_loss_weight = float(training_config.get("reconstruction_loss_weight", 0.0))
        self.use_reconstruction_loss = bool(
            training_config.get("use_reconstruction_loss", self.reconstruction_loss_weight > 0.0)
        )
        self.reconstruction_source = str(training_config.get("reconstruction_source", "queries_decoupled")).lower()
        self.reconstruction_decoder_layers = int(training_config.get("reconstruction_decoder_layers", 1) or 1)
        self.reconstruction_max_tokens_per_modality = int(
            training_config.get("reconstruction_max_tokens_per_modality", 128) or 0
        )
        self.reconstruction_target_mode = str(
            training_config.get(
                "reconstruction_target_mode",
                model_config.get("reconstruction_target_mode", "full_tokens"),
            )
        ).lower()
        self.reconstruction_salient_topk = int(
            training_config.get(
                "reconstruction_salient_topk",
                model_config.get("reconstruction_salient_topk", 2),
            )
            or 2
        )
        self.reconstruction_loss_type = str(training_config.get("reconstruction_loss_type", "mse")).lower()
        self.reconstruction_normalize_targets = bool(
            training_config.get("reconstruction_normalize_targets", False)
        )
        self.reconstruction_max_positions = int(model_config.get("reconstruction_max_positions", 512) or 512)
        self.reconstruction_exclude_tail_tokens = int(model_config.get("reconstruction_exclude_tail_tokens", 0) or 0)
        self.tcr_diversity_scope = str(
            training_config.get("tcr_diversity_scope", model_config.get("tcr_diversity_scope", "")) or ""
        ).lower()
        self.tcr_diversity_epsilon = float(
            training_config.get("tcr_diversity_epsilon", model_config.get("tcr_diversity_epsilon", 16.0))
        )
        self.tcr_diversity_normalize_tokens = bool(
            training_config.get(
                "tcr_diversity_normalize_tokens",
                model_config.get("tcr_diversity_normalize_tokens", True),
            )
        )
        self.tcr_diversity_center_tokens = bool(
            training_config.get(
                "tcr_diversity_center_tokens",
                model_config.get("tcr_diversity_center_tokens", False),
            )
        )
        self.use_learnable_loss_weighting = bool(training_config.get("learnable_loss_weighting", False))
        self.learnable_loss_component_mode = str(
            training_config.get("learnable_loss_component_mode", "branch")
        ).lower()
        self.learnable_loss_log_var_bounds = (
            float(training_config.get("learnable_loss_log_var_min", -6.0)),
            float(training_config.get("learnable_loss_log_var_max", 6.0)),
        )
        self.learnable_loss_branch_names = ("base_ref", "decoupled_classifier", "rol", "kdl")
        branch_init_loss_weights = training_config.get(
            "learnable_loss_initial_weights",
            [1.0, self.decoupled_classifier_weight, self.rol_loss_weight, self.kdl_loss_weight],
        )
        if not isinstance(branch_init_loss_weights, (list, tuple)):
            branch_init_loss_weights = [
                1.0,
                self.decoupled_classifier_weight,
                self.rol_loss_weight,
                self.kdl_loss_weight,
            ]
        branch_init_loss_weights = list(branch_init_loss_weights)[:4]
        while len(branch_init_loss_weights) < 4:
            branch_init_loss_weights.append(1.0)

        self.learnable_loss_all_default_weights = {
            "main": 1.0,
            "aux": float(getattr(self, "aux_loss_weight", 0.0)),
            "layer": float(getattr(self, "layer_loss_weight", 0.0)),
            "global": float(getattr(self, "global_loss_weight", 0.0)),
            "light": float(getattr(self, "light_loss_weight", 0.0)),
            "rank": float(getattr(self, "rank_loss_weight", 0.0)),
            "query_contrastive": float(getattr(self, "query_contrastive_loss_weight", 0.0)),
            "alignment": float(getattr(self, "alignment_loss_weight", 0.0)),
            "diversity": float(getattr(self, "diversity_loss_weight", 0.0)),
            "cross_sample": float(getattr(self, "cross_sample_lambda", 0.0)),
            "teacher_leading": float(getattr(self, "teacher_leading_loss_weight", 0.0)),
            "decoupled_classifier": self.decoupled_classifier_weight,
            "rol": self.rol_loss_weight,
            "kdl": self.kdl_loss_weight,
            "sp_sh_private": self.sp_sh_private_loss_weight,
            "sp_sh_shared": self.sp_sh_shared_loss_weight,
            "sp_sh_consistency": self.sp_sh_consistency_loss_weight,
            "sp_sh_disentangle": self.sp_sh_disentangle_loss_weight,
            "sp_sh_shared_contrastive": self.sp_sh_shared_contrastive_loss_weight,
            "sp_sh_query_cls": self.sp_sh_query_cls_loss_weight,
            "sidebranch_sp_sh_query": self.sidebranch_sp_sh_query_loss_weight,
            "unit_spsh_mil": self.unit_spsh_mil_loss_weight,
            "hqa_aux_cls": self.hqa_aux_cls_loss_weight,
            "hqa_shared_alignment": self.hqa_shared_alignment_weight,
            "hqa_private_orth": self.hqa_private_orth_weight,
            "sp_sh_reconstruction": self.sp_sh_reconstruction_loss_weight,
            "reconstruction": self.reconstruction_loss_weight,
        }
        configured_loss_names = training_config.get("learnable_loss_names")
        if isinstance(configured_loss_names, (list, tuple)) and configured_loss_names:
            active_loss_names = [str(name) for name in configured_loss_names]
        else:
            active_loss_names = [
                name
                for name, weight in self.learnable_loss_all_default_weights.items()
                if name == "main" or float(weight) > 0.0
            ]
        init_by_name = training_config.get("learnable_loss_initial_weights_by_name", {})
        if not isinstance(init_by_name, dict):
            init_by_name = {}
        self.learnable_loss_names = tuple(active_loss_names)

        self.learnable_loss_scale_min = float(training_config.get("learnable_loss_scale_min", 0.5))
        self.learnable_loss_scale_max = float(training_config.get("learnable_loss_scale_max", 1.5))
        if self.use_learnable_loss_weighting:
            if self.learnable_loss_component_mode in {"bounded", "bounded_sum", "bounded_fixed"}:
                self.learnable_loss_scale_logits = nn.Parameter(
                    torch.zeros(len(self.learnable_loss_names), dtype=torch.float32)
                )
            else:
                if self.learnable_loss_component_mode == "all":
                    init_loss_weights = [
                        float(init_by_name.get(name, self.learnable_loss_all_default_weights.get(name, 1.0)))
                        for name in self.learnable_loss_names
                    ]
                else:
                    init_loss_weights = branch_init_loss_weights
                init_loss_log_vars = [-math.log(max(float(weight), 1e-6)) for weight in init_loss_weights]
                self.learnable_loss_log_vars = nn.Parameter(torch.tensor(init_loss_log_vars, dtype=torch.float32))

        self.text_decoupler = EvidenceTokenDecoupler(self.hidden_dim, num_heads, self.dropout_rate)
        self.audio_decoupler = EvidenceTokenDecoupler(self.hidden_dim, num_heads, self.dropout_rate)
        self.decoupled_fusion = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 6),
            nn.Linear(self.hidden_dim * 6, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.decoupled_classifier = self._make_vector_classifier(hidden_layers=True)
        self.text_specific_classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.num_classes),
        )
        self.audio_specific_classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.num_classes),
        )
        self.shared_evidence_classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.num_classes),
        )
        self.unit_spsh_mil_head = None
        if self.use_unit_spsh_mil_head:
            self.unit_spsh_mil_head = UnitSPShMILHead(
                self.hidden_dim,
                self.num_classes,
                num_heads,
                self.dropout_rate,
                topk=self.unit_spsh_mil_topk,
                residual_scale=self.unit_spsh_mil_residual_scale,
                gate_init=self.unit_spsh_mil_gate_init,
            )
        self.class_feature_residual_adapter = None
        if self.use_class_feature_residual_adapter:
            self.class_feature_residual_adapter = ClassFeatureResidualLogitAdapter(
                self.hidden_dim,
                self.dropout_rate,
                adapter_dim=int(model_config.get("class_feature_residual_adapter_dim", 128) or 128),
                gate_init=float(model_config.get("class_feature_residual_gate_init", -4.0)),
                max_scale=float(model_config.get("class_feature_residual_max_scale", 0.50)),
            )
        self.paralinguistic_style_adapter = None
        if self.use_paralinguistic_style_adapter:
            self.paralinguistic_style_adapter = ParalinguisticStyleQueryAdapter(
                self.hidden_dim,
                num_heads,
                self.dropout_rate,
                adapter_dim=int(model_config.get("paralinguistic_style_adapter_dim", 128) or 128),
                gate_init=float(model_config.get("paralinguistic_style_gate_init", -2.0)),
                max_scale=float(model_config.get("paralinguistic_style_max_scale", 0.50)),
            )
        self.hqa_spsh_interactor = None
        if self.use_hqa_spsh_interactor:
            self.hqa_spsh_interactor = HierarchicalSPShQueryInteractor(
                self.hidden_dim,
                self.num_classes,
                self.hqa_heads,
                self.dropout_rate,
                depth=self.hqa_depth,
                num_private_queries=self.hqa_num_private_queries,
                num_shared_queries=self.hqa_num_shared_queries,
                num_global_queries=self.hqa_num_global_queries,
                router_gate=self.hqa_router_gate,
                feature_modulation=self.hqa_feature_modulation,
                feature_modulation_scale=self.hqa_feature_modulation_scale,
                mamba_mixer=self.hqa_mamba_mixer,
            )
        self.text_sp_sh_reconstructor = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.audio_sp_sh_reconstructor = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.shared_alignment = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 4),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        if self.use_reconstruction_loss:
            self.reconstruction_mask_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
            self.reconstruction_position_embedding = nn.Embedding(self.reconstruction_max_positions, self.hidden_dim)
            self.reconstruction_decoder = FingerprintReconstructionDecoder(
                self.hidden_dim,
                num_heads,
                self.dropout_rate,
                self.reconstruction_decoder_layers,
            )
        else:
            self.reconstruction_decoder = None

    def set_embedding_augmentation_epoch(
        self,
        epoch_index: int,
        total_epochs: int = None,
        num_train_samples: int = None,
    ) -> None:
        super().set_embedding_augmentation_epoch(
            epoch_index,
            total_epochs=total_epochs,
            num_train_samples=num_train_samples,
        )
        self.sp_sh_current_epoch = max(int(epoch_index), 0)

    def _scheduled_factor_value(self, start_epoch: int, ramp_epochs: int) -> float:
        epoch = max(int(getattr(self, "sp_sh_current_epoch", 0)), 0)
        start_epoch = max(int(start_epoch), 0)
        ramp_epochs = max(int(ramp_epochs), 0)
        if epoch < start_epoch:
            return 0.0
        if ramp_epochs <= 0:
            return 1.0
        return min(1.0, max(0.0, float(epoch - start_epoch + 1) / float(ramp_epochs)))

    def _scheduled_factor_tensor(
        self,
        reference: torch.Tensor,
        start_epoch: int,
        ramp_epochs: int,
    ) -> torch.Tensor:
        return reference.new_tensor(self._scheduled_factor_value(start_epoch, ramp_epochs))

    def _sp_sh_residual_factor(self, reference: torch.Tensor) -> torch.Tensor:
        return self._scheduled_factor_tensor(
            reference,
            self.sp_sh_residual_start_epoch,
            self.sp_sh_residual_ramp_epochs,
        )

    def _sp_sh_aux_factor(self, reference: torch.Tensor) -> torch.Tensor:
        return self._scheduled_factor_tensor(
            reference,
            self.sp_sh_aux_start_epoch,
            self.sp_sh_aux_ramp_epochs,
        )

    def _decoupled_effective_scale(self, reference: torch.Tensor) -> torch.Tensor:
        if self.learnable_decoupled_logit_gate:
            gate = torch.sigmoid(self.decoupled_logit_gate.to(device=reference.device, dtype=reference.dtype))
            return reference.new_tensor(self.decoupled_logit_gate_max) * gate
        return reference.new_tensor(self.decoupled_logit_scale)

    def _combine_base_and_decoupled_logits(
        self,
        base_logits: torch.Tensor,
        logits_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        logits_dict["final_logit_source"] = self.final_logit_source
        if self.final_logit_source in {"base", "static_query", "modal_pool"}:
            logits_dict["decoupled_logit_effective_scale"] = base_logits.new_tensor(0.0)
            return _finite_tensor(base_logits, limit=1e4)
        if self.final_logit_source == "decoupled":
            logits_dict["decoupled_logit_effective_scale"] = base_logits.new_tensor(1.0)
            return _finite_tensor(logits_dict["modality_decoupled_logits"], limit=1e4)
        if self.final_logit_source in {
            "sidebranch_sp_sh_query",
            "sidebranch_sp_sh_query_plus_base_residual",
            "hqa_spsh",
        }:
            logits_dict["decoupled_logit_effective_scale"] = base_logits.new_tensor(0.0)
            return _finite_tensor(base_logits, limit=1e4)
        residual_factor = self._sp_sh_residual_factor(base_logits)
        if self.samplewise_decoupled_logit_gate and "decoupled_rep" in logits_dict:
            gate_logits = self.decoupled_sample_gate(_finite_tensor(logits_dict["decoupled_rep"]))
            sample_scale = base_logits.new_tensor(self.decoupled_logit_gate_max) * torch.sigmoid(gate_logits)
            sample_scale = residual_factor * sample_scale
            logits_dict["decoupled_logit_effective_scale"] = sample_scale.detach().mean()
            logits_dict["decoupled_logit_sample_scale"] = sample_scale.detach().squeeze(-1)
            scale = sample_scale
        else:
            scale = residual_factor * self._decoupled_effective_scale(base_logits)
            logits_dict["decoupled_logit_effective_scale"] = scale.detach()
        logits_dict["sp_sh_residual_schedule_factor"] = residual_factor.detach()
        return _finite_tensor(
            base_logits + scale * logits_dict["modality_decoupled_logits"],
            limit=1e4,
        )

    def _unit_availability_mask(
        self,
        text_attention_mask: torch.Tensor,
        audio_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        unit_mask = text_attention_mask.bool().any(dim=-1) | audio_attention_mask.bool().any(dim=-1)
        if not unit_mask.any(dim=1).all():
            unit_mask = unit_mask.clone()
            unit_mask[~unit_mask.any(dim=1), 0] = True
        return unit_mask

    def _build_decoupled_features(
        self,
        logits_dict: Dict[str, torch.Tensor],
        unit_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        text_units = logits_dict["text_prompt_reps"]
        audio_units = logits_dict["audio_prompt_reps"]
        text_sp, text_sh = self.text_decoupler(text_units, unit_mask)
        audio_sp, audio_sh = self.audio_decoupler(audio_units, unit_mask)

        shared_pair = torch.cat(
            [text_sh, audio_sh, torch.abs(text_sh - audio_sh), text_sh * audio_sh],
            dim=-1,
        )
        shared_context = self.shared_alignment(_finite_tensor(shared_pair))
        pooled_query = logits_dict["class_reps"].mean(dim=1)
        fusion_input = torch.cat(
            [pooled_query, text_sp, audio_sp, text_sh, audio_sh, shared_context],
            dim=-1,
        )
        decoupled_rep = self.decoupled_fusion(_finite_tensor(fusion_input))
        decoupled_logits = self.decoupled_classifier(decoupled_rep)
        text_specific_logits = self.text_specific_classifier(_finite_tensor(text_sp))
        audio_specific_logits = self.audio_specific_classifier(_finite_tensor(audio_sp))
        shared_evidence_logits = self.shared_evidence_classifier(_finite_tensor(shared_context))
        private_evidence_logits = 0.5 * (text_specific_logits + audio_specific_logits)
        feature_stack = torch.stack([text_sp, audio_sp, text_sh, audio_sh], dim=1)
        feature_mask = torch.ones(feature_stack.shape[:2], device=feature_stack.device, dtype=torch.bool)
        return {
            "text_specific": text_sp,
            "audio_specific": audio_sp,
            "text_shared": text_sh,
            "audio_shared": audio_sh,
            "shared_context": shared_context,
            "decoupled_rep": decoupled_rep,
            "text_specific_logits": _finite_tensor(text_specific_logits, limit=1e4),
            "audio_specific_logits": _finite_tensor(audio_specific_logits, limit=1e4),
            "private_evidence_logits": _finite_tensor(private_evidence_logits, limit=1e4),
            "shared_evidence_logits": _finite_tensor(shared_evidence_logits, limit=1e4),
            "modality_decoupled_features": _finite_tensor(feature_stack),
            "modality_decoupled_mask": feature_mask,
            "modality_decoupled_logits": _finite_tensor(decoupled_logits, limit=1e4),
        }

    def _build_sidebranch_sp_sh_query_logits(
        self,
        logits_dict: Dict[str, torch.Tensor],
        decoupled: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if (
            not self.use_sidebranch_sp_sh_query_residual
            or self.sp_sh_query_adaption is None
            or "queries" not in logits_dict
            or "modality_decoupled_features" not in decoupled
        ):
            return None, {}
        queries = logits_dict["queries"]
        memory = decoupled["modality_decoupled_features"]
        memory_mask = decoupled.get("modality_decoupled_mask")
        if memory_mask is None:
            memory_mask = torch.ones(memory.shape[:2], device=memory.device, dtype=torch.bool)
        token_indices = {
            "text_sp": 0,
            "audio_sp": 1,
            "text_sh": 2,
            "audio_sh": 3,
        }

        def make_update_plan(source_plan: str):
            source_plan = str(source_plan).lower()
            if source_plan in {"shared", "shared_only", "sh"}:
                return [("shared", ["text_sh", "audio_sh"])]
            if source_plan in {"specific", "specific_only", "sp"}:
                return [("specific", ["text_sp", "audio_sp"])]
            if source_plan in {"sh_then_sp", "shared_then_specific", "shared_specific"}:
                return [
                    ("shared", ["text_sh", "audio_sh"]),
                    ("specific", ["text_sp", "audio_sp"]),
                ]
            if source_plan in {"sp_then_sh", "specific_then_shared", "specific_shared"}:
                return [
                    ("specific", ["text_sp", "audio_sp"]),
                    ("shared", ["text_sh", "audio_sh"]),
                ]
            return [("all", ["text_sp", "audio_sp", "text_sh", "audio_sh"])]

        def run_update_plan(adapter: nn.Module, start_queries: torch.Tensor, update_plan, diag_prefix: str):
            updated_queries = start_queries
            path_contexts = []
            path_gates = []
            for stage_name, token_names in update_plan:
                indices = torch.tensor(
                    [token_indices[name] for name in token_names],
                    device=memory.device,
                    dtype=torch.long,
                )
                stage_memory = memory.index_select(1, indices)
                stage_mask = memory_mask.bool().index_select(1, indices)
                updated_queries, context, gate = adapter(updated_queries, stage_memory, stage_mask)
                path_contexts.append(context)
                path_gates.append(gate)
                diagnostics_prefix = f"sidebranch_sp_sh_query_{diag_prefix}_{stage_name}"
                logits_dict[f"{diagnostics_prefix}_context_norm"] = context.detach().norm(dim=-1).mean()
                if isinstance(adapter, DirectSPShQueryAdapter):
                    logits_dict[f"{diagnostics_prefix}_delta_norm"] = gate.detach().norm(dim=-1).mean()
                else:
                    logits_dict[f"{diagnostics_prefix}_gate"] = gate.detach().mean()
            return updated_queries, path_contexts, path_gates

        if getattr(self.sp_sh_query_adaption, "is_dual_path_sp_sh_query_adapter", False):
            gated_source = str(
                getattr(
                    self,
                    "sidebranch_sp_sh_query_gated_source",
                    getattr(self, "sidebranch_sp_sh_query_update_source", "all"),
                )
            ).lower()
            direct_source = str(
                getattr(self, "sidebranch_sp_sh_query_direct_source", "sh_then_sp")
            ).lower()
            gated_queries, gated_contexts, gated_gates = run_update_plan(
                self.sp_sh_query_adaption.gated_adapter,
                queries,
                make_update_plan(gated_source),
                "gated",
            )
            direct_queries, direct_contexts, direct_gates = run_update_plan(
                self.sp_sh_query_adaption.direct_adapter,
                queries,
                make_update_plan(direct_source),
                "direct",
            )
            gated_context = torch.stack(gated_contexts, dim=0).mean(dim=0)
            direct_context = torch.stack(direct_contexts, dim=0).mean(dim=0)
            updated, path_gate = self.sp_sh_query_adaption.combine_paths(
                queries,
                gated_queries,
                gated_context,
                direct_queries,
                direct_context,
            )
            contexts = [gated_context, direct_context]
            gates = [path_gate]
            logits_dict["sidebranch_sp_sh_query_gated_source"] = gated_context.new_tensor(
                float(len(make_update_plan(gated_source)))
            )
            logits_dict["sidebranch_sp_sh_query_direct_source"] = direct_context.new_tensor(
                float(len(make_update_plan(direct_source)))
            )
            logits_dict["sidebranch_sp_sh_query_direct_path_gate"] = path_gate.detach().mean()
            logits_dict["sidebranch_sp_sh_query_gated_path_gate"] = (1.0 - path_gate.detach()).mean()
            logits_dict["sidebranch_sp_sh_query_gated_internal_gate"] = torch.stack(
                [gate.detach().mean() for gate in gated_gates]
            ).mean()
            logits_dict["sidebranch_sp_sh_query_direct_delta_norm"] = torch.stack(
                [gate.detach().norm(dim=-1).mean() for gate in direct_gates]
            ).mean()
        else:
            source_plan = str(getattr(self, "sidebranch_sp_sh_query_update_source", "all")).lower()
            updated, contexts, gates = run_update_plan(
                self.sp_sh_query_adaption,
                queries,
                make_update_plan(source_plan),
                "single",
            )
        class_reps, _ = self._pool_class_queries(updated)
        query_logits = self.class_scorer(self._finite(class_reps)).squeeze(-1)
        logit_scale = torch.exp(self.logit_scale.clamp(max=math.log(self.max_logit_scale)))
        query_logits = self._finite(logit_scale * self._finite(query_logits) + self.class_logit_bias, limit=1e4)
        diagnostics = {
            "sidebranch_sp_sh_query_gate": torch.stack([gate.detach().mean() for gate in gates]).mean(),
            "sidebranch_sp_sh_query_context_norm": torch.stack(
                [context.detach().norm(dim=-1).mean() for context in contexts]
            ).mean(),
        }
        return query_logits, diagnostics

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        text_attention_mask: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        sample_ids=None,
        augment_embeddings: bool = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        base_logits, logits_dict = super().forward(
            text_features,
            audio_features,
            text_attention_mask,
            audio_attention_mask,
            labels=labels,
            sample_ids=sample_ids,
            augment_embeddings=augment_embeddings,
        )
        unit_mask = self._unit_availability_mask(text_attention_mask, audio_attention_mask)
        decoupled = self._build_decoupled_features(logits_dict, unit_mask)
        logits_dict.update(decoupled)
        logits_dict["base_ref_logits"] = base_logits
        logits = self._combine_base_and_decoupled_logits(base_logits, logits_dict)
        unit_mil_logits, unit_mil_diag = self._build_unit_spsh_mil_logits(base_logits, logits_dict, unit_mask)
        if unit_mil_logits is not None:
            logits = self._finite(unit_mil_logits, limit=1e4)
            logits_dict.update(unit_mil_diag)
            logits_dict["final_logit_source"] = "unit_spsh_mil"
        return logits, logits_dict

    def _representation_orthogonality_loss(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        features = F.normalize(_finite_tensor(features), dim=-1)
        sim = torch.matmul(features, features.transpose(1, 2))
        target = features.new_tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ]
        )
        pair_mask = mask.bool().unsqueeze(2) & mask.bool().unsqueeze(1)
        loss = (sim - target.unsqueeze(0)).pow(2) * pair_mask.to(dtype=sim.dtype)
        return loss.sum() / pair_mask.to(dtype=sim.dtype).sum().clamp_min(1.0)

    def _tcr_diversity_loss(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = _finite_tensor(tokens)
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(1)
        if tokens.size(1) == 0:
            return tokens.new_tensor(0.0)
        if self.tcr_diversity_center_tokens:
            tokens = tokens - tokens.mean(dim=1, keepdim=True)
        if self.tcr_diversity_normalize_tokens:
            tokens = F.normalize(tokens, dim=-1)
        batch_size, num_tokens, hidden_dim = tokens.shape
        scale = tokens.new_tensor(
            float(hidden_dim) / (max(self.tcr_diversity_epsilon, 1e-6) ** 2 * max(num_tokens, 1))
        )
        gram = torch.matmul(tokens, tokens.transpose(1, 2))
        eye = torch.eye(num_tokens, device=tokens.device, dtype=tokens.dtype).unsqueeze(0)
        sign, logabsdet = torch.linalg.slogdet((eye + scale * gram).float())
        logabsdet = torch.where(sign > 0, logabsdet, torch.zeros_like(logabsdet))
        return _finite_tensor((-0.5 * logabsdet).to(dtype=tokens.dtype)).mean()

    def _token_diversity_regularizer(self, logits_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        scope = self.tcr_diversity_scope
        if scope in {"", "legacy", "rol", "orthogonality"}:
            return self._representation_orthogonality_loss(
                logits_dict["modality_decoupled_features"],
                logits_dict["modality_decoupled_mask"],
            )
        if scope in {"specific", "sp", "sp_only", "sponly"}:
            tokens = torch.stack([logits_dict["text_specific"], logits_dict["audio_specific"]], dim=1)
            return self._tcr_diversity_loss(tokens)
        if scope in {"query", "q", "q_only", "qonly"}:
            return self._tcr_diversity_loss(logits_dict["queries"])
        if scope in {"specific_query", "sp_query", "spq"}:
            sp_tokens = torch.stack([logits_dict["text_specific"], logits_dict["audio_specific"]], dim=1)
            query_tokens = logits_dict["queries"]
            return self._tcr_diversity_loss(torch.cat([sp_tokens, query_tokens], dim=1))
        raise ValueError(
            "training.tcr_diversity_scope must be one of: legacy, specific, query, specific_query"
        )

    def _pairwise_distance(self, features: torch.Tensor) -> torch.Tensor:
        features = F.normalize(_finite_tensor(features), dim=-1)
        return torch.cdist(features, features, p=2.0)

    def _cosine_square_loss(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        first = F.normalize(_finite_tensor(first), dim=-1)
        second = F.normalize(_finite_tensor(second), dim=-1)
        return F.cosine_similarity(first, second, dim=-1).pow(2).mean()

    def _sp_sh_shared_consistency_loss(self, logits_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        text_shared = F.normalize(_finite_tensor(logits_dict["text_shared"]), dim=-1)
        audio_shared = F.normalize(_finite_tensor(logits_dict["audio_shared"]), dim=-1)
        similarity = F.cosine_similarity(text_shared, audio_shared, dim=-1)
        return _finite_tensor(1.0 - similarity).mean()

    def _sp_sh_shared_contrastive_loss(
        self,
        logits_dict: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.sp_sh_shared_contrastive_loss_weight <= 0.0 or labels.numel() <= 1:
            zero = logits_dict["text_shared"].new_tensor(0.0)
            return zero, {
                "valid_anchor_ratio": zero,
                "avg_pos_per_anchor": zero,
            }
        shared_views = torch.cat(
            [logits_dict["text_shared"], logits_dict["audio_shared"]],
            dim=0,
        )
        view_labels = labels.repeat(2)
        loss, diagnostics = supervised_contrastive_loss(
            _finite_tensor(shared_views),
            view_labels,
            temperature=self.sp_sh_shared_contrastive_temperature,
            class_balanced=self.sp_sh_shared_contrastive_class_balanced,
            exclude_same_sample=False,
        )
        return _finite_tensor(loss), diagnostics

    def _sp_sh_disentangle_loss(self, logits_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        text_sp = logits_dict["text_specific"]
        audio_sp = logits_dict["audio_specific"]
        text_sh = logits_dict["text_shared"]
        audio_sh = logits_dict["audio_shared"]
        loss = (
            self._cosine_square_loss(text_sp, text_sh)
            + self._cosine_square_loss(audio_sp, audio_sh)
            + self._cosine_square_loss(text_sp, audio_sp)
        ) / 3.0
        return _finite_tensor(loss)

    def _weighted_prompt_pool(
        self,
        reps: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> torch.Tensor:
        reps = _finite_tensor(reps)
        if weights is None:
            return reps.mean(dim=1)
        weights = _finite_tensor(weights.to(device=reps.device, dtype=reps.dtype)).clamp_min(0.0)
        if weights.dim() > 2:
            weights = weights.squeeze(-1)
        if weights.size(1) != reps.size(1):
            return reps.mean(dim=1)
        denom = weights.sum(dim=1, keepdim=True)
        missing = denom.squeeze(1) <= 1e-8
        if missing.any():
            weights = weights.clone()
            weights[missing] = 1.0
            denom = weights.sum(dim=1, keepdim=True)
        return _finite_tensor((reps * weights.unsqueeze(-1)).sum(dim=1) / denom.clamp_min(1e-8))

    def _sp_sh_reconstruction_loss(self, logits_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.sp_sh_reconstruction_loss_weight <= 0.0:
            zero = logits_dict["modality_decoupled_features"].new_tensor(0.0)
            return zero, zero, zero
        unit_weights = logits_dict.get("unit_weights")
        text_target = self._weighted_prompt_pool(logits_dict["text_prompt_reps"].detach(), unit_weights)
        audio_target = self._weighted_prompt_pool(logits_dict["audio_prompt_reps"].detach(), unit_weights)
        text_source = torch.cat([logits_dict["text_specific"], logits_dict["text_shared"]], dim=-1)
        audio_source = torch.cat([logits_dict["audio_specific"], logits_dict["audio_shared"]], dim=-1)
        text_pred = self.text_sp_sh_reconstructor(_finite_tensor(text_source))
        audio_pred = self.audio_sp_sh_reconstructor(_finite_tensor(audio_source))
        if self.sp_sh_reconstruction_normalize_targets:
            text_pred = F.layer_norm(text_pred, (text_pred.size(-1),))
            audio_pred = F.layer_norm(audio_pred, (audio_pred.size(-1),))
            text_target = F.layer_norm(text_target, (text_target.size(-1),))
            audio_target = F.layer_norm(audio_target, (audio_target.size(-1),))
        text_loss = F.mse_loss(_finite_tensor(text_pred), _finite_tensor(text_target))
        audio_loss = F.mse_loss(_finite_tensor(audio_pred), _finite_tensor(audio_target))
        total = 0.5 * (text_loss + audio_loss)
        return _finite_tensor(total), _finite_tensor(text_loss), _finite_tensor(audio_loss)

    def _knowledge_discrepancy_loss(
        self,
        logits_dict: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if labels.numel() < 3:
            return logits_dict["modality_decoupled_features"].new_tensor(0.0)

        text_sp = logits_dict["text_specific"]
        audio_sp = logits_dict["audio_specific"]
        text_sh = logits_dict["text_shared"]
        audio_sh = logits_dict["audio_shared"]
        specific = torch.cat([text_sp, audio_sp], dim=-1)
        shared = torch.cat([text_sh, audio_sh], dim=-1)
        combined = torch.cat([specific, shared], dim=-1)

        dist_sp = self._pairwise_distance(specific).detach()
        dist_sh = self._pairwise_distance(shared).detach()
        dist_combined = self._pairwise_distance(combined)
        same = labels[:, None].eq(labels[None, :])
        self_mask = torch.eye(labels.numel(), device=labels.device, dtype=torch.bool)
        positive_mask = same & ~self_mask
        negative_mask = ~same
        valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
        if not valid.any():
            return combined.new_tensor(0.0)

        large = dist_combined.new_tensor(1e4)
        pos_combined = dist_combined.masked_fill(~positive_mask, -large).max(dim=1).values
        pos_sp = dist_sp.masked_fill(~positive_mask, -large).max(dim=1).values
        pos_sh = dist_sh.masked_fill(~positive_mask, -large).max(dim=1).values
        neg_combined = dist_combined.masked_fill(~negative_mask, large).min(dim=1).values
        neg_sp = dist_sp.masked_fill(~negative_mask, large).min(dim=1).values
        neg_sh = dist_sh.masked_fill(~negative_mask, large).min(dim=1).values

        eps = dist_combined.new_tensor(1e-6)
        pos_ratio = pos_combined / (pos_combined + pos_sp + pos_sh + eps)
        neg_ratio = neg_combined / (neg_combined + neg_sp + neg_sh + eps)
        loss = pos_ratio.abs() + (neg_ratio - 1.0).abs()
        return _finite_tensor(loss[valid]).mean()

    def _build_reconstruction_source(self, logits_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        source_parts = []
        if self.reconstruction_source in {"queries", "query", "queries_decoupled", "all"}:
            queries = logits_dict.get("queries")
            if queries is not None:
                source_parts.append(_finite_tensor(queries))
        if self.reconstruction_source in {"decoupled", "fp", "fingerprint", "queries_decoupled", "all"}:
            decoupled = logits_dict.get("modality_decoupled_features")
            if decoupled is not None:
                source_parts.append(_finite_tensor(decoupled))
        if self.reconstruction_source in {"class", "class_reps", "all"}:
            class_reps = logits_dict.get("class_reps")
            if class_reps is not None:
                source_parts.append(_finite_tensor(class_reps))
        if not source_parts:
            fallback = logits_dict.get("queries")
            if fallback is None:
                return None, None
            source_parts.append(_finite_tensor(fallback))
        source = torch.cat(source_parts, dim=1)
        source_mask = torch.ones(source.shape[:2], device=source.device, dtype=torch.bool)
        return _finite_tensor(source), source_mask

    def _build_reconstruction_queries(
        self,
        target: torch.Tensor,
        num_units: int,
        seq_len: int,
        modality_idx: int,
    ) -> torch.Tensor:
        batch_size = target.size(0)
        device = target.device
        unit_ids = torch.arange(num_units, device=device).clamp(max=self.max_evidence_units - 1)
        position_ids = torch.arange(seq_len, device=device).clamp(max=self.reconstruction_max_positions - 1)
        unit_embed = self.unit_embedding(unit_ids).view(1, num_units, 1, self.hidden_dim)
        pos_embed = self.reconstruction_position_embedding(position_ids).view(1, 1, seq_len, self.hidden_dim)
        modality_embed = self.modality_embedding[modality_idx].view(1, 1, 1, self.hidden_dim)
        query = self.reconstruction_mask_token.view(1, 1, 1, self.hidden_dim) + unit_embed + pos_embed + modality_embed
        return _finite_tensor(query.expand(batch_size, -1, -1, -1).reshape(batch_size, num_units * seq_len, self.hidden_dim))

    def _masked_reconstruction_mse(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = mask.bool()
        prediction = _finite_tensor(prediction)
        target = _finite_tensor(target)
        if getattr(self, "reconstruction_normalize_targets", False):
            prediction = F.layer_norm(prediction, (prediction.size(-1),))
            target = F.layer_norm(target, (target.size(-1),))
        if getattr(self, "reconstruction_loss_type", "mse") in {"smooth_l1", "huber"}:
            token_loss = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)
        else:
            token_loss = (prediction - target).pow(2).mean(dim=-1)
        weights = mask.to(dtype=token_loss.dtype)
        return (token_loss * weights).sum() / weights.sum().clamp_min(1.0)

    def _limit_reconstruction_tokens(
        self,
        query: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_tokens = int(getattr(self, "reconstruction_max_tokens_per_modality", 0) or 0)
        if max_tokens <= 0 or target.size(1) <= max_tokens:
            return query, target, mask
        positions = torch.linspace(
            0,
            target.size(1) - 1,
            steps=max_tokens,
            device=target.device,
        ).round().long()
        return (
            query.index_select(1, positions),
            target.index_select(1, positions),
            mask.index_select(1, positions),
        )

    def _canonical_reconstruction_unit_weights(
        self,
        logits_dict: Dict[str, torch.Tensor],
        num_units: int,
        reference: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        weights = logits_dict.get("unit_weights", logits_dict.get("prompt_weights"))
        if weights is None or not torch.is_tensor(weights):
            return None
        weights = _finite_tensor(weights.to(device=reference.device, dtype=reference.dtype)).clamp_min(0.0)
        if weights.dim() == 2:
            if weights.size(-1) != num_units:
                return None
            weights = weights.unsqueeze(1)
        elif weights.dim() == 3:
            if weights.size(-1) != num_units and weights.size(1) == num_units:
                weights = weights.transpose(1, 2)
            if weights.size(-1) != num_units:
                return None
        else:
            return None
        return weights

    def _prepare_reconstruction_targets(
        self,
        logits_dict: Dict[str, torch.Tensor],
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        int,
        int,
        int,
    ]:
        text_units = logits_dict.get("text_prompt_reps")
        audio_units = logits_dict.get("audio_prompt_reps")
        if text_units is None or audio_units is None:
            return None, None, None, None, 0, 0, 0
        text_units = _finite_tensor(text_units.detach())
        audio_units = _finite_tensor(audio_units.detach())
        if text_units.dim() != 3 or audio_units.dim() != 3:
            return None, None, None, None, 0, 0, 0
        num_units = min(text_units.size(1), audio_units.size(1))
        if num_units <= 0:
            return None, None, None, None, 0, 0, 0
        text_units = text_units[:, :num_units, :]
        audio_units = audio_units[:, :num_units, :]
        mode = str(getattr(self, "reconstruction_target_mode", "full_tokens")).lower()
        weights = self._canonical_reconstruction_unit_weights(logits_dict, num_units, text_units)

        if mode in {"emotion_pooled", "class_pooled", "pooled"}:
            if weights is None:
                weights = text_units.new_ones(text_units.size(0), self.num_classes, num_units)
            elif weights.size(1) == 1 and self.num_classes > 1:
                weights = weights.expand(-1, self.num_classes, -1)
            elif weights.size(1) != self.num_classes:
                weights = weights[:, :1, :].expand(-1, self.num_classes, -1)
            denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            text_target = torch.einsum("bcn,bnd->bcd", weights, text_units) / denom
            audio_target = torch.einsum("bcn,bnd->bcd", weights, audio_units) / denom
            mask = weights.sum(dim=-1) > 1e-8
            if not mask.any(dim=1).all():
                mask = mask.clone()
                missing = ~mask.any(dim=1)
                mask[missing, 0] = True
            return (
                _finite_tensor(text_target),
                _finite_tensor(audio_target),
                mask,
                mask,
                int(text_target.size(1)),
                1,
                1,
            )

        if mode in {"masked_salient_units", "salient_units", "salient"}:
            if weights is None:
                salience = text_units.new_ones(text_units.size(0), num_units)
            elif weights.dim() == 3:
                salience = weights.max(dim=1).values
            else:
                salience = weights.squeeze(1)
            topk = max(1, min(int(getattr(self, "reconstruction_salient_topk", 2) or 2), num_units))
            indices = salience.topk(k=topk, dim=-1).indices
            gather_index = indices.unsqueeze(-1).expand(-1, -1, text_units.size(-1))
            text_target = text_units.gather(1, gather_index)
            audio_target = audio_units.gather(1, gather_index)
            mask = torch.ones(text_target.shape[:2], device=text_target.device, dtype=torch.bool)
            return (
                _finite_tensor(text_target),
                _finite_tensor(audio_target),
                mask,
                mask,
                int(topk),
                1,
                1,
            )

        mask = torch.ones(text_units.shape[:2], device=text_units.device, dtype=torch.bool)
        return text_units, audio_units, mask, mask, int(num_units), 1, 1

    def _reconstruction_loss(self, logits_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.reconstruction_decoder is None:
            base = logits_dict.get("queries")
            if base is None:
                base = logits_dict.get("modality_decoupled_features")
            zero = self.class_queries.new_tensor(0.0) if base is None else base.new_tensor(0.0)
            return zero, zero, zero
        text_target = logits_dict.get("reconstruction_text_target")
        audio_target = logits_dict.get("reconstruction_audio_target")
        text_mask = logits_dict.get("reconstruction_text_mask")
        audio_mask = logits_dict.get("reconstruction_audio_mask")
        target_mode = str(getattr(self, "reconstruction_target_mode", "full_tokens")).lower()
        num_units = 0
        text_seq_len = 0
        audio_seq_len = 0
        if (
            target_mode not in {"full_tokens", "full", "tokens"}
            or text_target is None
            or audio_target is None
            or text_mask is None
            or audio_mask is None
        ):
            (
                text_target,
                audio_target,
                text_mask,
                audio_mask,
                num_units,
                text_seq_len,
                audio_seq_len,
            ) = self._prepare_reconstruction_targets(logits_dict)
        if text_target is None or audio_target is None or text_mask is None or audio_mask is None:
            base = logits_dict.get("queries")
            zero = self.class_queries.new_tensor(0.0) if base is None else base.new_tensor(0.0)
            return zero, zero, zero

        source, source_mask = self._build_reconstruction_source(logits_dict)
        if source is None:
            zero = text_target.new_tensor(0.0)
            return zero, zero, zero
        if num_units <= 0:
            unit_weights = logits_dict.get("unit_weights", logits_dict.get("prompt_weights"))
            if unit_weights is not None and torch.is_tensor(unit_weights):
                num_units = int(unit_weights.size(-1))
            else:
                num_units = int(text_target.size(1))
        text_seq_len = max(1, int(text_seq_len) if text_seq_len else text_target.size(1) // max(num_units, 1))
        audio_seq_len = max(1, int(audio_seq_len) if audio_seq_len else audio_target.size(1) // max(num_units, 1))
        text_query = self._build_reconstruction_queries(text_target, num_units, text_seq_len, 0)
        audio_query = self._build_reconstruction_queries(audio_target, num_units, audio_seq_len, 1)
        text_query, text_target, text_mask = self._limit_reconstruction_tokens(text_query, text_target, text_mask)
        audio_query, audio_target, audio_mask = self._limit_reconstruction_tokens(audio_query, audio_target, audio_mask)
        text_pred = self.reconstruction_decoder(text_query, source, source_mask)
        audio_pred = self.reconstruction_decoder(audio_query, source, source_mask)
        text_loss = self._masked_reconstruction_mse(text_pred, text_target, text_mask)
        audio_loss = self._masked_reconstruction_mse(audio_pred, audio_target, audio_mask)
        total = 0.5 * (text_loss + audio_loss)
        return _finite_tensor(total), _finite_tensor(text_loss), _finite_tensor(audio_loss)

    def calculate_losses(
        self,
        logits: torch.Tensor,
        logits_dict: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.use_refqformer_loss:
            return self._calculate_refqformer_losses(logits, logits_dict, labels)

        losses = super().calculate_losses(logits, logits_dict, labels)
        decoupled_cls_loss = self._classification_loss(logits_dict["modality_decoupled_logits"], labels)
        rol_loss = self._token_diversity_regularizer(logits_dict)
        kdl_loss = logits.new_tensor(0.0)
        if self.training and self.kdl_loss_weight > 0:
            kdl_loss = self._knowledge_discrepancy_loss(logits_dict, labels)
        reconstruction_loss, reconstruction_text_loss, reconstruction_audio_loss = self._reconstruction_loss(logits_dict)
        text_private_loss = self._classification_loss(logits_dict["text_specific_logits"], labels)
        audio_private_loss = self._classification_loss(logits_dict["audio_specific_logits"], labels)
        private_evidence_loss = 0.5 * (text_private_loss + audio_private_loss)
        shared_evidence_loss = self._classification_loss(logits_dict["shared_evidence_logits"], labels)
        sp_sh_consistency_loss = self._sp_sh_shared_consistency_loss(logits_dict)
        sp_sh_disentangle_loss = self._sp_sh_disentangle_loss(logits_dict)
        sp_sh_shared_contrastive_loss, sp_sh_shared_contrastive_diag = self._sp_sh_shared_contrastive_loss(
            logits_dict,
            labels,
        )
        sp_sh_query_cls_loss = logits.new_tensor(0.0)
        if self.sp_sh_query_cls_loss_weight > 0.0 and "query" in logits_dict:
            sp_sh_query_cls_loss = self._classification_loss(logits_dict["query"], labels)
        sidebranch_sp_sh_query_loss = logits.new_tensor(0.0)
        if self.sidebranch_sp_sh_query_loss_weight > 0.0 and "sidebranch_sp_sh_query_logits" in logits_dict:
            sidebranch_sp_sh_query_loss = self._classification_loss(
                logits_dict["sidebranch_sp_sh_query_logits"],
                labels,
            )
        unit_spsh_mil_loss = logits.new_tensor(0.0)
        if self.unit_spsh_mil_loss_weight > 0.0 and "unit_spsh_mil_logits" in logits_dict:
            unit_spsh_mil_loss = self._classification_loss(
                logits_dict["unit_spsh_mil_logits"],
                labels,
            )
        hqa_aux_cls_loss = logits.new_tensor(0.0)
        if self.hqa_aux_cls_loss_weight > 0.0 and "hqa_aux_logits" in logits_dict:
            hqa_aux_cls_loss = self._classification_loss(logits_dict["hqa_aux_logits"], labels)
        hqa_shared_alignment_loss = logits_dict.get("hqa_shared_alignment_loss", logits.new_tensor(0.0))
        hqa_private_orth_loss = logits_dict.get("hqa_private_orth_loss", logits.new_tensor(0.0))
        sp_sh_reconstruction_loss, sp_sh_reconstruction_text_loss, sp_sh_reconstruction_audio_loss = self._sp_sh_reconstruction_loss(logits_dict)

        base_ref_total_loss = losses["total_loss"]
        losses["base_ref_total_loss"] = base_ref_total_loss
        losses["decoupled_classifier_loss"] = decoupled_cls_loss
        losses["rol_loss"] = rol_loss
        if self.tcr_diversity_scope not in {"", "legacy", "rol", "orthogonality"}:
            losses["tcr_diversity_loss"] = rol_loss
        losses["kdl_loss"] = kdl_loss
        losses["reconstruction_loss"] = reconstruction_loss
        losses["reconstruction_text_loss"] = reconstruction_text_loss
        losses["reconstruction_audio_loss"] = reconstruction_audio_loss
        losses["sp_sh_private_loss"] = private_evidence_loss
        losses["sp_sh_text_private_loss"] = text_private_loss
        losses["sp_sh_audio_private_loss"] = audio_private_loss
        losses["sp_sh_shared_loss"] = shared_evidence_loss
        losses["sp_sh_consistency_loss"] = sp_sh_consistency_loss
        losses["sp_sh_disentangle_loss"] = sp_sh_disentangle_loss
        losses["sp_sh_shared_contrastive_loss"] = sp_sh_shared_contrastive_loss
        for diag_name, diag_value in sp_sh_shared_contrastive_diag.items():
            losses[f"sp_sh_shared_contrastive_{diag_name}"] = diag_value.detach()
        losses["sp_sh_query_cls_loss"] = sp_sh_query_cls_loss
        losses["sidebranch_sp_sh_query_loss"] = sidebranch_sp_sh_query_loss
        losses["unit_spsh_mil_loss"] = unit_spsh_mil_loss
        losses["hqa_aux_cls_loss"] = hqa_aux_cls_loss
        losses["hqa_shared_alignment_loss"] = hqa_shared_alignment_loss
        losses["hqa_private_orth_loss"] = hqa_private_orth_loss
        losses["sp_sh_reconstruction_loss"] = sp_sh_reconstruction_loss
        losses["sp_sh_reconstruction_text_loss"] = sp_sh_reconstruction_text_loss
        losses["sp_sh_reconstruction_audio_loss"] = sp_sh_reconstruction_audio_loss
        sp_sh_aux_factor = self._sp_sh_aux_factor(logits)
        losses["sp_sh_aux_schedule_factor"] = sp_sh_aux_factor.detach()
        scheduled_decoupled_cls_loss = decoupled_cls_loss * sp_sh_aux_factor
        scheduled_rol_loss = rol_loss * sp_sh_aux_factor
        scheduled_kdl_loss = kdl_loss * sp_sh_aux_factor
        scheduled_private_evidence_loss = private_evidence_loss * sp_sh_aux_factor
        scheduled_shared_evidence_loss = shared_evidence_loss * sp_sh_aux_factor
        scheduled_sp_sh_consistency_loss = sp_sh_consistency_loss * sp_sh_aux_factor
        scheduled_sp_sh_disentangle_loss = sp_sh_disentangle_loss * sp_sh_aux_factor
        scheduled_sp_sh_shared_contrastive_loss = sp_sh_shared_contrastive_loss * sp_sh_aux_factor
        scheduled_sp_sh_query_cls_loss = sp_sh_query_cls_loss * sp_sh_aux_factor
        scheduled_sidebranch_sp_sh_query_loss = sidebranch_sp_sh_query_loss * sp_sh_aux_factor
        scheduled_unit_spsh_mil_loss = unit_spsh_mil_loss * sp_sh_aux_factor
        scheduled_hqa_aux_cls_loss = hqa_aux_cls_loss * sp_sh_aux_factor
        scheduled_hqa_shared_alignment_loss = hqa_shared_alignment_loss * sp_sh_aux_factor
        scheduled_hqa_private_orth_loss = hqa_private_orth_loss * sp_sh_aux_factor
        scheduled_sp_sh_reconstruction_loss = sp_sh_reconstruction_loss * sp_sh_aux_factor
        if self.decoupled_only_loss:
            losses["total_loss"] = (
                scheduled_decoupled_cls_loss
                + self.rol_loss_weight * scheduled_rol_loss
                + self.kdl_loss_weight * scheduled_kdl_loss
                + self.reconstruction_loss_weight * reconstruction_loss
                + self.sp_sh_private_loss_weight * scheduled_private_evidence_loss
                + self.sp_sh_shared_loss_weight * scheduled_shared_evidence_loss
                + self.sp_sh_consistency_loss_weight * scheduled_sp_sh_consistency_loss
                + self.sp_sh_disentangle_loss_weight * scheduled_sp_sh_disentangle_loss
                + self.sp_sh_shared_contrastive_loss_weight * scheduled_sp_sh_shared_contrastive_loss
                + self.sp_sh_query_cls_loss_weight * scheduled_sp_sh_query_cls_loss
                + self.sidebranch_sp_sh_query_loss_weight * scheduled_sidebranch_sp_sh_query_loss
                + self.unit_spsh_mil_loss_weight * scheduled_unit_spsh_mil_loss
                + self.hqa_aux_cls_loss_weight * scheduled_hqa_aux_cls_loss
                + self.hqa_shared_alignment_weight * scheduled_hqa_shared_alignment_loss
                + self.hqa_private_orth_weight * scheduled_hqa_private_orth_loss
                + self.sp_sh_reconstruction_loss_weight * scheduled_sp_sh_reconstruction_loss
            )
            losses["decoupled_only_loss_applied"] = logits.new_tensor(1.0)
            return losses
        if self.use_learnable_loss_weighting:
            component_losses = {
                "main": losses.get("main_loss"),
                "aux": losses.get("aux_loss"),
                "layer": losses.get("layer_loss"),
                "global": losses.get("global_loss"),
                "light": losses.get("light_loss"),
                "rank": losses.get("rank_loss"),
                "query_contrastive": losses.get("query_contrastive_loss"),
                "alignment": losses.get("alignment_loss"),
                "diversity": losses.get("diversity_loss"),
                "cross_sample": losses.get("cross_sample_loss_applied"),
                "teacher_leading": losses.get("teacher_leading_loss"),
                "decoupled_classifier": scheduled_decoupled_cls_loss,
                "rol": scheduled_rol_loss,
                "kdl": scheduled_kdl_loss if self.kdl_loss_weight > 0 else None,
                "sp_sh_private": scheduled_private_evidence_loss if self.sp_sh_private_loss_weight > 0 else None,
                "sp_sh_shared": scheduled_shared_evidence_loss if self.sp_sh_shared_loss_weight > 0 else None,
                "sp_sh_consistency": scheduled_sp_sh_consistency_loss if self.sp_sh_consistency_loss_weight > 0 else None,
                "sp_sh_disentangle": scheduled_sp_sh_disentangle_loss if self.sp_sh_disentangle_loss_weight > 0 else None,
                "sp_sh_shared_contrastive": scheduled_sp_sh_shared_contrastive_loss if self.sp_sh_shared_contrastive_loss_weight > 0 else None,
                "sp_sh_query_cls": scheduled_sp_sh_query_cls_loss if self.sp_sh_query_cls_loss_weight > 0 else None,
                "sidebranch_sp_sh_query": scheduled_sidebranch_sp_sh_query_loss if self.sidebranch_sp_sh_query_loss_weight > 0 else None,
                "unit_spsh_mil": scheduled_unit_spsh_mil_loss if self.unit_spsh_mil_loss_weight > 0 else None,
                "hqa_aux_cls": scheduled_hqa_aux_cls_loss if self.hqa_aux_cls_loss_weight > 0 else None,
                "hqa_shared_alignment": (
                    scheduled_hqa_shared_alignment_loss if self.hqa_shared_alignment_weight > 0 else None
                ),
                "hqa_private_orth": scheduled_hqa_private_orth_loss if self.hqa_private_orth_weight > 0 else None,
                "sp_sh_reconstruction": scheduled_sp_sh_reconstruction_loss if self.sp_sh_reconstruction_loss_weight > 0 else None,
                "reconstruction": reconstruction_loss if self.reconstruction_loss_weight > 0 else None,
            }
            if self.learnable_loss_component_mode in {"bounded", "bounded_sum", "bounded_fixed"}:
                main_loss = component_losses.get("main")
                total_loss = main_loss if torch.is_tensor(main_loss) else logits.new_tensor(0.0)
                scale_logits = self.learnable_loss_scale_logits.to(device=logits.device, dtype=logits.dtype)
                scale_min = logits.new_tensor(self.learnable_loss_scale_min)
                scale_max = logits.new_tensor(self.learnable_loss_scale_max)
                active_count = 0
                for index, name in enumerate(self.learnable_loss_names):
                    if name == "main":
                        losses["learnable_loss_weight_main"] = logits.new_tensor(1.0)
                        losses["learnable_loss_scale_main"] = logits.new_tensor(1.0)
                        continue
                    component_loss = component_losses.get(name)
                    if component_loss is None or not torch.is_tensor(component_loss):
                        continue
                    base_weight = float(self.learnable_loss_all_default_weights.get(name, 0.0))
                    if base_weight <= 0.0:
                        continue
                    scale = scale_min + (scale_max - scale_min) * torch.sigmoid(scale_logits[index])
                    effective_weight = component_loss.new_tensor(base_weight) * scale
                    total_loss = total_loss + effective_weight * component_loss
                    losses[f"learnable_loss_weight_{name}"] = effective_weight.detach()
                    losses[f"learnable_loss_scale_{name}"] = scale.detach()
                    active_count += 1
                if active_count == 0 and not torch.is_tensor(main_loss):
                    total_loss = (
                        base_ref_total_loss
                        + self.decoupled_classifier_weight * scheduled_decoupled_cls_loss
                        + self.rol_loss_weight * scheduled_rol_loss
                        + self.kdl_loss_weight * scheduled_kdl_loss
                        + self.reconstruction_loss_weight * reconstruction_loss
                        + self.hqa_aux_cls_loss_weight * scheduled_hqa_aux_cls_loss
                        + self.hqa_shared_alignment_weight * scheduled_hqa_shared_alignment_loss
                        + self.hqa_private_orth_weight * scheduled_hqa_private_orth_loss
                        + self.sp_sh_reconstruction_loss_weight * scheduled_sp_sh_reconstruction_loss
                    )
                losses["total_loss"] = total_loss
            else:
                log_vars = self.learnable_loss_log_vars.to(device=logits.device, dtype=logits.dtype)
                log_var_min, log_var_max = self.learnable_loss_log_var_bounds
                if self.learnable_loss_component_mode == "all":
                    total_loss = logits.new_tensor(0.0)
                    active_count = 0
                    for index, name in enumerate(self.learnable_loss_names):
                        component_loss = component_losses.get(name)
                        if component_loss is None or not torch.is_tensor(component_loss):
                            continue
                        log_var = log_vars[index].clamp(min=log_var_min, max=log_var_max)
                        precision = torch.exp(-log_var)
                        total_loss = total_loss + precision * component_loss + log_var
                        losses[f"learnable_loss_weight_{name}"] = precision.detach()
                        losses[f"learnable_loss_log_var_{name}"] = log_var.detach()
                        active_count += 1
                    if active_count == 0:
                        total_loss = (
                            base_ref_total_loss
                            + self.decoupled_classifier_weight * scheduled_decoupled_cls_loss
                            + self.rol_loss_weight * scheduled_rol_loss
                            + self.kdl_loss_weight * scheduled_kdl_loss
                            + self.hqa_aux_cls_loss_weight * scheduled_hqa_aux_cls_loss
                            + self.hqa_shared_alignment_weight * scheduled_hqa_shared_alignment_loss
                            + self.hqa_private_orth_weight * scheduled_hqa_private_orth_loss
                        )
                    losses["total_loss"] = total_loss
                else:
                    components = [
                        ("base_ref", base_ref_total_loss, 0),
                        ("decoupled_classifier", scheduled_decoupled_cls_loss, 1),
                        ("rol", scheduled_rol_loss, 2),
                    ]
                    if self.kdl_loss_weight > 0:
                        components.append(("kdl", scheduled_kdl_loss, 3))
                    total_loss = logits.new_tensor(0.0)
                    for name, component_loss, index in components:
                        log_var = log_vars[index].clamp(min=log_var_min, max=log_var_max)
                        precision = torch.exp(-log_var)
                        total_loss = total_loss + precision * component_loss + log_var
                        losses[f"learnable_loss_weight_{name}"] = precision.detach()
                        losses[f"learnable_loss_log_var_{name}"] = log_var.detach()
                    losses["total_loss"] = total_loss
            if (
                self.reconstruction_loss_weight > 0
                and not (
                    self.learnable_loss_component_mode in {"all", "bounded", "bounded_sum", "bounded_fixed"}
                    and "reconstruction" in self.learnable_loss_names
                )
            ):
                losses["total_loss"] = losses["total_loss"] + self.reconstruction_loss_weight * reconstruction_loss
        else:
            losses["total_loss"] = (
                base_ref_total_loss
                + self.decoupled_classifier_weight * scheduled_decoupled_cls_loss
                + self.rol_loss_weight * scheduled_rol_loss
                + self.kdl_loss_weight * scheduled_kdl_loss
                + self.reconstruction_loss_weight * reconstruction_loss
                + self.sp_sh_private_loss_weight * scheduled_private_evidence_loss
                + self.sp_sh_shared_loss_weight * scheduled_shared_evidence_loss
                + self.sp_sh_consistency_loss_weight * scheduled_sp_sh_consistency_loss
                + self.sp_sh_disentangle_loss_weight * scheduled_sp_sh_disentangle_loss
                + self.sp_sh_shared_contrastive_loss_weight * scheduled_sp_sh_shared_contrastive_loss
                + self.sp_sh_query_cls_loss_weight * scheduled_sp_sh_query_cls_loss
                + self.sidebranch_sp_sh_query_loss_weight * scheduled_sidebranch_sp_sh_query_loss
                + self.unit_spsh_mil_loss_weight * scheduled_unit_spsh_mil_loss
                + self.hqa_aux_cls_loss_weight * scheduled_hqa_aux_cls_loss
                + self.hqa_shared_alignment_weight * scheduled_hqa_shared_alignment_loss
                + self.hqa_private_orth_weight * scheduled_hqa_private_orth_loss
                + self.sp_sh_reconstruction_loss_weight * scheduled_sp_sh_reconstruction_loss
            )
        return losses


class TokenPrependedModalityDecoupledQueryModel(ModalityDecoupledQueryModel):
    """Picture-aligned variant: prepend shared/specific tokens before QA layers.

    The first MDReID-inspired version extracts shared/specific representations
    after the RefFormer backbone. The diagram instead inserts `[SP, SH]` tokens
    into each modality before the transformer/query-adaption stack. This class
    follows that placement while preserving the existing cached feature format:

        audio: [audio_tokens, SP_AUDIO, SH_AUDIO]
        text:  [text_tokens, SP_TEXT, SH_TEXT]

    The project stores multiple evidence units per sample. By default one SP
    and one SH token are appended to every valid unit; experiments can expand
    each type to K latent slots with ``sp_sh_tokens_per_type``. Padded units
    keep the new token mask off and do not become artificial evidence.
    """

    def __init__(self, config: Dict[str, object]):
        super().__init__(config)
        model_config = config.get("model", {})
        training_config = config.get("training", {})
        input_dim = int(model_config.get("input_dim", 768))
        text_input_dim = int(model_config.get("text_input_dim", input_dim))
        audio_input_dim = int(model_config.get("audio_input_dim", input_dim))
        init_std = float(model_config.get("sp_sh_token_init_std", 0.02))
        self.sp_sh_token_scale = float(model_config.get("sp_sh_token_scale", 1.0))
        self.sp_sh_tokens_per_type = int(model_config.get("sp_sh_tokens_per_type", 1) or 1)
        if self.sp_sh_tokens_per_type <= 0:
            raise ValueError("model.sp_sh_tokens_per_type must be positive")
        self.sp_sh_tail_token_count = 2 * self.sp_sh_tokens_per_type
        self.isolate_sp_sh_from_main = bool(model_config.get("isolate_sp_sh_from_main", False))
        self.isolate_sp_sh_from_query_attention = bool(model_config.get("isolate_sp_sh_from_query_attention", False))
        self.sp_sh_side_branch_only = bool(model_config.get("sp_sh_side_branch_only", False))
        self.use_sp_sh_query_update = bool(model_config.get("use_sp_sh_query_update", False))
        self.sp_sh_query_update_mode = str(model_config.get("sp_sh_query_update_mode", "per_layer")).lower()
        self.sp_sh_query_update_source = str(model_config.get("sp_sh_query_update_source", "all")).lower()
        self.sp_sh_query_update_adapter = str(
            model_config.get(
                "sp_sh_query_update_adapter",
                model_config.get("sp_sh_query_update_type", "gated"),
            )
        ).lower()
        self.sp_sh_query_update_scale = float(model_config.get("sp_sh_query_update_scale", 0.10))
        self.sp_sh_query_update_gated_scale = float(
            model_config.get("sp_sh_query_update_gated_scale", self.sp_sh_query_update_scale)
        )
        self.sp_sh_query_update_direct_scale = float(
            model_config.get("sp_sh_query_update_direct_scale", self.sp_sh_query_update_scale)
        )
        self.sp_sh_query_update_mixture_gate_init = float(
            model_config.get("sp_sh_query_update_mixture_gate_init", -0.5)
        )
        self.sp_sh_query_update_path_gate_mode = str(
            model_config.get("sp_sh_query_update_path_gate_mode", "learned")
        ).lower()
        fixed_direct_weight = model_config.get("sp_sh_query_update_fixed_direct_weight", None)
        self.sp_sh_query_update_fixed_direct_weight = (
            None if fixed_direct_weight is None else float(fixed_direct_weight)
        )
        self.sp_sh_query_update_ffn_ratio = float(model_config.get("sp_sh_query_update_ffn_ratio", 2.0))
        self.sp_sh_query_update_num_experts = int(model_config.get("sp_sh_query_update_num_experts", 3) or 3)
        self.sp_sh_query_update_mixer_layers = int(model_config.get("sp_sh_query_update_mixer_layers", 1) or 1)
        self.sp_sh_query_update_router_gate_init = float(
            model_config.get(
                "sp_sh_query_update_router_gate_init",
                model_config.get("sp_sh_query_update_gate_init", -2.5),
            )
        )
        self.sp_sh_query_update_start_epoch = int(
            training_config.get(
                "sp_sh_query_update_start_epoch",
                model_config.get("sp_sh_query_update_start_epoch", 0),
            )
            or 0
        )
        self.sp_sh_query_update_ramp_epochs = int(
            training_config.get(
                "sp_sh_query_update_ramp_epochs",
                model_config.get("sp_sh_query_update_ramp_epochs", 0),
            )
            or 0
        )
        self.learnable_sp_sh_query_gate = bool(model_config.get("learnable_sp_sh_query_gate", False))
        self.sp_sh_query_gate_max = float(model_config.get("sp_sh_query_gate_max", self.sp_sh_query_update_scale))
        self.sp_sh_query_gate_init = float(model_config.get("sp_sh_query_gate_init", -6.0))
        if self.learnable_sp_sh_query_gate:
            self.sp_sh_query_gate_logit = nn.Parameter(
                torch.tensor(self.sp_sh_query_gate_init, dtype=torch.float32)
            )
        self.use_sidebranch_sp_sh_query_residual = bool(
            model_config.get("use_sidebranch_sp_sh_query_residual", False)
        )
        self.sidebranch_sp_sh_query_residual_scale = float(
            model_config.get("sidebranch_sp_sh_query_residual_scale", 0.02)
        )
        self.sidebranch_sp_sh_query_base_residual_scale = float(
            model_config.get("sidebranch_sp_sh_query_base_residual_scale", 0.20)
        )
        self.sidebranch_sp_sh_query_residual_start_epoch = int(
            training_config.get(
                "sidebranch_sp_sh_query_residual_start_epoch",
                model_config.get("sidebranch_sp_sh_query_residual_start_epoch", 0),
            )
            or 0
        )
        self.sidebranch_sp_sh_query_residual_ramp_epochs = int(
            training_config.get(
                "sidebranch_sp_sh_query_residual_ramp_epochs",
                model_config.get("sidebranch_sp_sh_query_residual_ramp_epochs", 0),
            )
            or 0
        )
        self.samplewise_sidebranch_sp_sh_query_gate = bool(
            model_config.get("samplewise_sidebranch_sp_sh_query_gate", False)
        )
        self.sidebranch_sp_sh_query_update_source = str(
            model_config.get(
                "sidebranch_sp_sh_query_update_source",
                model_config.get("sp_sh_query_update_source", "all"),
            )
        ).lower()
        self.sidebranch_sp_sh_query_gated_source = str(
            model_config.get(
                "sidebranch_sp_sh_query_gated_source",
                self.sidebranch_sp_sh_query_update_source,
            )
        ).lower()
        self.sidebranch_sp_sh_query_direct_source = str(
            model_config.get("sidebranch_sp_sh_query_direct_source", "sh_then_sp")
        ).lower()
        self.sidebranch_sp_sh_query_gate_max = float(
            model_config.get(
                "sidebranch_sp_sh_query_gate_max",
                self.sidebranch_sp_sh_query_residual_scale,
            )
        )
        self.sidebranch_sp_sh_query_router_mode = str(
            model_config.get(
                "sidebranch_sp_sh_query_router_mode",
                model_config.get("sidebranch_sp_sh_query_gate_mode", "feature"),
            )
        ).lower()
        self.sidebranch_sp_sh_query_delta_mode = str(
            model_config.get("sidebranch_sp_sh_query_delta_mode", "side_minus_current")
        ).lower()
        sidebranch_gate_hidden = max(16, self.hidden_dim // 2)
        self.sidebranch_sp_sh_query_sample_gate = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, sidebranch_gate_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(sidebranch_gate_hidden, 1),
        )
        sidebranch_gate_init = float(model_config.get("sidebranch_sp_sh_query_gate_init", 0.0))
        nn.init.zeros_(self.sidebranch_sp_sh_query_sample_gate[-1].weight)
        nn.init.constant_(self.sidebranch_sp_sh_query_sample_gate[-1].bias, sidebranch_gate_init)
        self.use_sidebranch_confidence_router = self.sidebranch_sp_sh_query_router_mode in {
            "confidence",
            "confidence_router",
            "r_spaq",
            "routed_spsh",
            "routed_spsh_affective_query",
        }
        router_hidden = int(
            model_config.get("sidebranch_sp_sh_query_router_hidden_dim", sidebranch_gate_hidden)
            or sidebranch_gate_hidden
        )
        router_input_dim = self.hidden_dim * 2 + self.num_classes * 2 + 4
        self.sidebranch_sp_sh_query_confidence_gate = nn.Sequential(
            nn.LayerNorm(router_input_dim),
            nn.Linear(router_input_dim, router_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(router_hidden, 1),
        )
        nn.init.zeros_(self.sidebranch_sp_sh_query_confidence_gate[-1].weight)
        nn.init.constant_(self.sidebranch_sp_sh_query_confidence_gate[-1].bias, sidebranch_gate_init)
        self.sp_sh_query_adaption = None
        if self.use_sp_sh_query_update and self.sp_sh_query_update_scale > 0:
            num_heads = int(model_config.get("num_heads", 4))
            if self.sp_sh_query_update_adapter in {
                "dual",
                "dual_path",
                "mixture",
                "adapter_moe",
                "gated_direct_moe",
            }:
                self.sp_sh_query_adaption = DualPathSPShQueryAdapter(
                    self.hidden_dim,
                    num_heads,
                    self.dropout_rate,
                    gated_scale=self.sp_sh_query_update_gated_scale,
                    direct_scale=self.sp_sh_query_update_direct_scale,
                    ffn_ratio=self.sp_sh_query_update_ffn_ratio,
                    gate_init=self.sp_sh_query_update_mixture_gate_init,
                    gate_mode=self.sp_sh_query_update_path_gate_mode,
                    fixed_direct_weight=self.sp_sh_query_update_fixed_direct_weight,
                )
            elif self.sp_sh_query_update_adapter in {"visual_moe", "prompt_moe", "token_moe"}:
                self.sp_sh_query_adaption = VisualPromptMoESPShQueryAdapter(
                    self.hidden_dim,
                    num_heads,
                    self.dropout_rate,
                    scale=self.sp_sh_query_update_scale,
                    num_experts=self.sp_sh_query_update_num_experts,
                    ffn_ratio=self.sp_sh_query_update_ffn_ratio,
                    mixer_layers=self.sp_sh_query_update_mixer_layers,
                )
            elif self.sp_sh_query_update_adapter in {
                "routed",
                "routed_direct",
                "qguard",
                "qguard_direct",
                "confidence_direct",
            }:
                self.sp_sh_query_adaption = RoutedSPShQueryAdapter(
                    self.hidden_dim,
                    num_heads,
                    self.dropout_rate,
                    scale=self.sp_sh_query_update_scale,
                    gate_init=self.sp_sh_query_update_router_gate_init,
                    ffn_ratio=self.sp_sh_query_update_ffn_ratio,
                )
            elif self.sp_sh_query_update_adapter in {
                "identity",
                "identity_direct",
                "zero_init",
                "zero_init_direct",
                "late_identity",
                "late_residual",
            }:
                self.sp_sh_query_adaption = IdentityInitSPShQueryAdapter(
                    self.hidden_dim,
                    num_heads,
                    self.dropout_rate,
                    scale=self.sp_sh_query_update_scale,
                    ffn_ratio=self.sp_sh_query_update_ffn_ratio,
                )
            elif self.sp_sh_query_update_adapter in {"direct", "direct_adapter", "nogate", "no_gate"}:
                self.sp_sh_query_adaption = DirectSPShQueryAdapter(
                    self.hidden_dim,
                    num_heads,
                    self.dropout_rate,
                    scale=self.sp_sh_query_update_scale,
                    ffn_ratio=self.sp_sh_query_update_ffn_ratio,
                )
            else:
                self.sp_sh_query_adaption = QueryAdaptionModule(
                    self.hidden_dim,
                    num_heads,
                    self.dropout_rate,
                    self.sp_sh_query_update_scale,
                )
        if "reconstruction_exclude_tail_tokens" not in model_config:
            self.reconstruction_exclude_tail_tokens = (
                0 if self.sp_sh_side_branch_only else self.sp_sh_tail_token_count
            )
        self.return_final_modality_tokens = True
        self.text_sp_sh_tokens = nn.Parameter(
            torch.randn(self.sp_sh_tail_token_count, text_input_dim) * init_std
        )
        self.audio_sp_sh_tokens = nn.Parameter(
            torch.randn(self.sp_sh_tail_token_count, audio_input_dim) * init_std
        )
        self.sp_sh_content_conditioned_tokens = bool(
            model_config.get("sp_sh_content_conditioned_tokens", False)
        )
        self.sp_sh_content_scale = float(model_config.get("sp_sh_content_scale", 0.50))
        self.sp_sh_content_gate = nn.Parameter(
            torch.tensor(float(model_config.get("sp_sh_content_gate_init", -1.0)), dtype=torch.float32)
        )
        text_content_hidden = int(model_config.get("text_sp_sh_content_hidden_dim", model_config.get("sp_sh_content_hidden_dim", text_input_dim * 2)))
        audio_content_hidden = int(model_config.get("audio_sp_sh_content_hidden_dim", model_config.get("sp_sh_content_hidden_dim", audio_input_dim * 2)))
        self.text_sp_sh_content_projector = nn.Sequential(
            nn.LayerNorm(text_input_dim),
            nn.Linear(text_input_dim, text_content_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(text_content_hidden, text_input_dim * self.sp_sh_tail_token_count),
        )
        self.audio_sp_sh_content_projector = nn.Sequential(
            nn.LayerNorm(audio_input_dim),
            nn.Linear(audio_input_dim, audio_content_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(audio_content_hidden, audio_input_dim * self.sp_sh_tail_token_count),
        )
        self.use_acoustic_style_query_adapter = bool(
            model_config.get("use_acoustic_style_query_adapter", False)
        )
        self.acoustic_style_query_adapter = None
        if self.use_acoustic_style_query_adapter:
            num_heads = int(model_config.get("num_heads", 4))
            self.acoustic_style_query_adapter = AcousticStyleQueryAdapter(
                audio_input_dim,
                self.hidden_dim,
                num_heads,
                self.dropout_rate,
                adapter_dim=int(model_config.get("acoustic_style_adapter_dim", 128) or 128),
                gate_init=float(model_config.get("acoustic_style_gate_init", -1.0)),
                max_scale=float(model_config.get("acoustic_style_max_scale", 0.50)),
            )
        modal_pool_hidden = int(model_config.get("modal_pool_hidden_dim", self.hidden_dim) or self.hidden_dim)
        self.modal_pool_classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, modal_pool_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(modal_pool_hidden, self.num_classes),
        )

    def _main_evidence_token_mask(
        self,
        mask: torch.Tensor,
        num_units: int,
        seq_len: int,
        modality: str,
    ) -> torch.Tensor:
        tail_count = self.sp_sh_tail_token_count
        if not self.isolate_sp_sh_from_main or seq_len < tail_count:
            return mask
        batch_size = mask.size(0)
        grouped = mask.reshape(batch_size, num_units, seq_len).clone()
        grouped[:, :, -tail_count:] = False
        main_mask = grouped.reshape(batch_size, num_units * seq_len)
        missing_rows = ~main_mask.any(dim=1)
        if missing_rows.any():
            main_mask = main_mask.clone()
            main_mask[missing_rows, 0] = True
        return main_mask

    def _append_sp_sh_tokens(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor,
        tokens: torch.Tensor,
        content_projector: nn.Module = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_units, _, input_dim = features.shape
        if tokens.size(-1) != input_dim:
            raise ValueError(
                f"SP/SH token dim {tokens.size(-1)} does not match feature dim {input_dim}"
            )
        unit_valid = attention_mask.bool().any(dim=-1, keepdim=True)
        base_tokens = tokens.to(device=features.device, dtype=features.dtype)
        tail_count = self.sp_sh_tail_token_count
        if base_tokens.size(0) != tail_count:
            raise ValueError(
                f"Expected {tail_count} SP/SH tokens, got {base_tokens.size(0)}"
            )
        token_features = base_tokens.view(1, 1, tail_count, input_dim).expand(
            batch_size, num_units, -1, -1
        )
        if self.sp_sh_content_conditioned_tokens and content_projector is not None:
            valid = attention_mask.bool()
            token_count = valid.to(dtype=features.dtype).sum(dim=2, keepdim=True).clamp_min(1.0)
            unit_pool = (features * valid.unsqueeze(-1).to(dtype=features.dtype)).sum(dim=2) / token_count
            content_delta = content_projector(_finite_tensor(unit_pool))
            content_delta = content_delta.view(batch_size, num_units, tail_count, input_dim)
            content_scale = features.new_tensor(self.sp_sh_content_scale) * torch.sigmoid(
                self.sp_sh_content_gate.to(device=features.device, dtype=features.dtype)
            )
            token_features = token_features + content_scale * _finite_tensor(content_delta, limit=10.0)
        token_features = token_features * features.new_tensor(self.sp_sh_token_scale)
        token_mask = unit_valid.expand(batch_size, num_units, tail_count)
        features = torch.cat([features, token_features], dim=2)
        attention_mask = torch.cat([attention_mask.bool(), token_mask], dim=2)
        return features, attention_mask

    def _pool_valid_unit_tokens(self, unit_tokens: torch.Tensor, unit_mask: torch.Tensor) -> torch.Tensor:
        unit_mask = unit_mask.bool()
        if not unit_mask.any(dim=1).all():
            unit_mask = unit_mask.clone()
            unit_mask[~unit_mask.any(dim=1), 0] = True
        weights = unit_mask.to(dtype=unit_tokens.dtype).unsqueeze(-1)
        return (unit_tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _stage_allows_sp_sh_query_update(self, stage: str) -> bool:
        mode = self.sp_sh_query_update_mode
        stage_name = str(stage).lower()
        is_final = stage_name == "final"
        layer_number = None
        if stage_name.startswith("l") and stage_name[1:].isdigit():
            layer_number = int(stage_name[1:])
        if mode in {"all", "both"}:
            return True
        if mode in {"final", "final_only"}:
            return is_final
        if mode in {"late2", "last2", "last_two", "last2_with_final"}:
            last_layer = int(getattr(self, "num_query_layers", 0) or 0)
            return is_final or (layer_number is not None and layer_number >= max(1, last_layer))
        if mode in {"late2_layers", "last_two_layers", "last2_layers"}:
            last_layer = int(getattr(self, "num_query_layers", 0) or 0)
            return layer_number is not None and layer_number >= max(1, last_layer - 1)
        if mode in {"late1", "last1"}:
            last_layer = int(getattr(self, "num_query_layers", 0) or 0)
            return layer_number is not None and layer_number >= max(1, last_layer)
        return not is_final

    def _sp_sh_query_update_factor(self, reference: torch.Tensor) -> torch.Tensor:
        return self._scheduled_factor_tensor(
            reference,
            self.sp_sh_query_update_start_epoch,
            self.sp_sh_query_update_ramp_epochs,
        )

    def _sp_sh_query_update(
        self,
        queries: torch.Tensor,
        text_tokens: torch.Tensor,
        audio_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        num_units: int,
        text_seq_len: int,
        audio_seq_len: int,
        stage: str,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if (
            self.sp_sh_query_adaption is None
            or self.sp_sh_side_branch_only
            or text_seq_len < self.sp_sh_tail_token_count
            or audio_seq_len < self.sp_sh_tail_token_count
            or not self._stage_allows_sp_sh_query_update(stage)
        ):
            return queries, {}
        batch_size = queries.size(0)
        hidden_dim = queries.size(-1)
        text_group = text_tokens.reshape(batch_size, num_units, text_seq_len, hidden_dim)
        audio_group = audio_tokens.reshape(batch_size, num_units, audio_seq_len, hidden_dim)
        text_mask_group = text_mask.reshape(batch_size, num_units, text_seq_len).bool()
        audio_mask_group = audio_mask.reshape(batch_size, num_units, audio_seq_len).bool()
        text_valid = text_mask_group.any(dim=-1)
        audio_valid = audio_mask_group.any(dim=-1)
        slots = self.sp_sh_tokens_per_type
        text_tail_start = text_seq_len - self.sp_sh_tail_token_count
        audio_tail_start = audio_seq_len - self.sp_sh_tail_token_count
        token_parts = {
            "text_sp": text_group[:, :, text_tail_start : text_tail_start + slots, :],
            "text_sh": text_group[:, :, text_tail_start + slots :, :],
            "audio_sp": audio_group[:, :, audio_tail_start : audio_tail_start + slots, :],
            "audio_sh": audio_group[:, :, audio_tail_start + slots :, :],
        }
        mask_parts = {
            "text_sp": text_valid.unsqueeze(-1).expand(-1, -1, slots),
            "text_sh": text_valid.unsqueeze(-1).expand(-1, -1, slots),
            "audio_sp": audio_valid.unsqueeze(-1).expand(-1, -1, slots),
            "audio_sh": audio_valid.unsqueeze(-1).expand(-1, -1, slots),
        }

        def build_memory(names):
            memory = torch.cat([token_parts[name] for name in names], dim=2).reshape(
                batch_size,
                num_units * slots * len(names),
                hidden_dim,
            )
            memory_mask = torch.cat([mask_parts[name] for name in names], dim=2).reshape(
                batch_size,
                num_units * slots * len(names),
            )
            missing_rows = ~memory_mask.any(dim=1)
            if missing_rows.any():
                memory = memory.clone()
                memory_mask = memory_mask.clone()
                memory[missing_rows, 0, :] = 0.0
                memory_mask[missing_rows, 0] = True
            return memory, memory_mask

        def adapt_once(base_queries, names):
            memory, memory_mask = build_memory(names)
            schedule_factor_once = self._sp_sh_query_update_factor(base_queries)
            if self.sp_sh_query_update_adapter in {
                "direct",
                "direct_adapter",
                "nogate",
                "no_gate",
                "identity",
                "identity_direct",
                "zero_init",
                "zero_init_direct",
                "late_identity",
                "late_residual",
            }:
                proposed_once, context_once, delta_once = self.sp_sh_query_adaption(
                    base_queries,
                    memory,
                    memory_mask,
                )
                update_factor_once = schedule_factor_once
                updated_once = _finite_tensor(base_queries + update_factor_once * (proposed_once - base_queries))
                delta_norm_once = delta_once.detach().norm(dim=-1, keepdim=True)
                return updated_once, context_once, delta_norm_once, update_factor_once

            updated_once, context_once, gate_once = self.sp_sh_query_adaption(
                base_queries,
                memory,
                memory_mask,
            )
            external_gate_once = schedule_factor_once
            if self.learnable_sp_sh_query_gate:
                external_gate_once = schedule_factor_once * base_queries.new_tensor(self.sp_sh_query_gate_max) * torch.sigmoid(
                    self.sp_sh_query_gate_logit.to(device=base_queries.device, dtype=base_queries.dtype)
                )
            updated_once = _finite_tensor(base_queries + external_gate_once * (updated_once - base_queries))
            return updated_once, context_once, gate_once, external_gate_once

        source = getattr(self, "sp_sh_query_update_source", "all")
        if source in {"shared", "shared_only", "sh"}:
            update_plan = [("shared", ["text_sh", "audio_sh"])]
        elif source in {"specific", "specific_only", "sp"}:
            update_plan = [("specific", ["text_sp", "audio_sp"])]
        elif source in {"sh_then_sp", "shared_then_specific", "shared_specific"}:
            update_plan = [
                ("shared", ["text_sh", "audio_sh"]),
                ("specific", ["text_sp", "audio_sp"]),
            ]
        elif source in {"sp_then_sh", "specific_then_shared", "specific_shared"}:
            update_plan = [
                ("specific", ["text_sp", "audio_sp"]),
                ("shared", ["text_sh", "audio_sh"]),
            ]
        else:
            update_plan = [("all", ["text_sp", "text_sh", "audio_sp", "audio_sh"])]

        updated = queries
        contexts = []
        gates = []
        external_gates = []
        for _, names in update_plan:
            updated, context, gate, external_gate = adapt_once(updated, names)
            contexts.append(context)
            gates.append(gate)
            external_gates.append(external_gate.reshape(()))

        prefix = f"sp_sh_query_{stage}"
        context_norm = torch.stack([ctx.detach().norm(dim=-1).mean() for ctx in contexts]).mean()
        gate_mean = torch.stack([g.detach().mean() for g in gates]).mean()
        external_gate_mean = torch.stack([g.detach() for g in external_gates]).mean()
        update_norm_ratio = (updated.detach() - queries.detach()).norm(dim=-1).mean() / queries.detach().norm(dim=-1).mean().clamp_min(1e-6)
        return updated, {
            f"{prefix}_gate": gate_mean,
            f"{prefix}_external_gate": external_gate_mean,
            f"{prefix}_context_norm": context_norm,
            f"{prefix}_update_norm_ratio": update_norm_ratio,
        }

    def _extract_final_sp_sh_tokens(
        self,
        logits_dict: Dict[str, torch.Tensor],
        text_unit_mask: torch.Tensor,
        audio_unit_mask: torch.Tensor,
        num_units: int,
        text_seq_len: int,
        audio_seq_len: int,
    ) -> Dict[str, torch.Tensor]:
        batch_size = text_unit_mask.size(0)
        final_text = logits_dict["final_text_tokens"]
        final_audio = logits_dict["final_audio_tokens"]
        hidden_dim = final_text.size(-1)
        final_text = final_text.view(batch_size, num_units, text_seq_len, hidden_dim)
        final_audio = final_audio.view(batch_size, num_units, audio_seq_len, hidden_dim)

        slots = self.sp_sh_tokens_per_type
        text_tail_start = text_seq_len - self.sp_sh_tail_token_count
        audio_tail_start = audio_seq_len - self.sp_sh_tail_token_count
        text_sp_token_units = final_text[:, :, text_tail_start : text_tail_start + slots, :]
        text_sh_token_units = final_text[:, :, text_tail_start + slots :, :]
        audio_sp_token_units = final_audio[:, :, audio_tail_start : audio_tail_start + slots, :]
        audio_sh_token_units = final_audio[:, :, audio_tail_start + slots :, :]

        # Preserve the existing [B, U, D] downstream contract while QA attends
        # to every latent slot independently.
        text_sp_units = text_sp_token_units.mean(dim=2)
        text_sh_units = text_sh_token_units.mean(dim=2)
        audio_sp_units = audio_sp_token_units.mean(dim=2)
        audio_sh_units = audio_sh_token_units.mean(dim=2)

        return {
            "text_specific": self._pool_valid_unit_tokens(text_sp_units, text_unit_mask),
            "text_shared": self._pool_valid_unit_tokens(text_sh_units, text_unit_mask),
            "audio_specific": self._pool_valid_unit_tokens(audio_sp_units, audio_unit_mask),
            "audio_shared": self._pool_valid_unit_tokens(audio_sh_units, audio_unit_mask),
            "text_specific_units": _finite_tensor(text_sp_units),
            "text_shared_units": _finite_tensor(text_sh_units),
            "audio_specific_units": _finite_tensor(audio_sp_units),
            "audio_shared_units": _finite_tensor(audio_sh_units),
            "text_specific_token_units": _finite_tensor(text_sp_token_units),
            "text_shared_token_units": _finite_tensor(text_sh_token_units),
            "audio_specific_token_units": _finite_tensor(audio_sp_token_units),
            "audio_shared_token_units": _finite_tensor(audio_sh_token_units),
            "text_unit_mask": text_unit_mask.bool(),
            "audio_unit_mask": audio_unit_mask.bool(),
        }

    def _build_modal_pool_logits(
        self,
        logits_dict: Dict[str, torch.Tensor],
        unit_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        text_pool = _masked_mean_sequence(logits_dict["text_prompt_reps"], unit_mask)
        audio_pool = _masked_mean_sequence(logits_dict["audio_prompt_reps"], unit_mask)
        modal_rep = torch.cat([text_pool, audio_pool], dim=-1)
        modal_logits = _finite_tensor(self.modal_pool_classifier(_finite_tensor(modal_rep)), limit=1e4)
        return modal_logits, {
            "modal_pool_logits": modal_logits.detach(),
            "modal_pool_text_norm": text_pool.detach().norm(dim=-1).mean(),
            "modal_pool_audio_norm": audio_pool.detach().norm(dim=-1).mean(),
        }

    def _build_decoupled_features_from_tokens(
        self,
        logits_dict: Dict[str, torch.Tensor],
        token_parts: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        text_sp = token_parts["text_specific"]
        audio_sp = token_parts["audio_specific"]
        text_sh = token_parts["text_shared"]
        audio_sh = token_parts["audio_shared"]
        shared_pair = torch.cat(
            [text_sh, audio_sh, torch.abs(text_sh - audio_sh), text_sh * audio_sh],
            dim=-1,
        )
        shared_context = self.shared_alignment(_finite_tensor(shared_pair))
        pooled_query = logits_dict["class_reps"].mean(dim=1)
        fusion_input = torch.cat(
            [pooled_query, text_sp, audio_sp, text_sh, audio_sh, shared_context],
            dim=-1,
        )
        decoupled_rep = self.decoupled_fusion(_finite_tensor(fusion_input))
        decoupled_logits = self.decoupled_classifier(decoupled_rep)
        text_specific_logits = self.text_specific_classifier(_finite_tensor(text_sp))
        audio_specific_logits = self.audio_specific_classifier(_finite_tensor(audio_sp))
        shared_evidence_logits = self.shared_evidence_classifier(_finite_tensor(shared_context))
        private_evidence_logits = 0.5 * (text_specific_logits + audio_specific_logits)
        feature_stack = torch.stack([text_sp, audio_sp, text_sh, audio_sh], dim=1)
        feature_mask = torch.ones(feature_stack.shape[:2], device=feature_stack.device, dtype=torch.bool)
        return {
            "text_specific": text_sp,
            "audio_specific": audio_sp,
            "text_shared": text_sh,
            "audio_shared": audio_sh,
            "shared_context": shared_context,
            "decoupled_rep": decoupled_rep,
            "text_specific_logits": _finite_tensor(text_specific_logits, limit=1e4),
            "audio_specific_logits": _finite_tensor(audio_specific_logits, limit=1e4),
            "private_evidence_logits": _finite_tensor(private_evidence_logits, limit=1e4),
            "shared_evidence_logits": _finite_tensor(shared_evidence_logits, limit=1e4),
            "modality_decoupled_features": _finite_tensor(feature_stack),
            "modality_decoupled_mask": feature_mask,
            "modality_decoupled_logits": _finite_tensor(decoupled_logits, limit=1e4),
        }

    def _sidebranch_confidence_router_input(
        self,
        logits_dict: Dict[str, torch.Tensor],
        base_logits: torch.Tensor,
        side_query_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        decoupled_rep = logits_dict.get("decoupled_rep")
        class_reps = logits_dict.get("class_reps")
        if class_reps is not None:
            query_rep = _finite_tensor(class_reps).mean(dim=1)
        elif decoupled_rep is not None:
            query_rep = _finite_tensor(decoupled_rep)
        else:
            query_rep = base_logits.new_zeros(base_logits.size(0), self.hidden_dim)
        if decoupled_rep is None:
            decoupled_rep = query_rep
        decoupled_rep = _finite_tensor(decoupled_rep)

        base_probs = torch.softmax(_finite_tensor(base_logits, limit=1e4), dim=-1)
        side_probs = torch.softmax(_finite_tensor(side_query_logits, limit=1e4), dim=-1)
        sorted_probs = base_probs.sort(dim=-1, descending=True).values
        if sorted_probs.size(-1) > 1:
            margin = sorted_probs[:, :1] - sorted_probs[:, 1:2]
        else:
            margin = sorted_probs[:, :1]
        entropy = -(base_probs * (base_probs.clamp_min(1e-8)).log()).sum(dim=-1, keepdim=True)
        entropy = entropy / base_logits.new_tensor(math.log(max(int(base_probs.size(-1)), 2)))
        agreement = base_probs.argmax(dim=-1).eq(side_probs.argmax(dim=-1)).to(dtype=base_probs.dtype).unsqueeze(-1)
        delta_norm = (side_query_logits - base_logits).norm(dim=-1, keepdim=True)
        delta_norm = delta_norm / base_logits.new_tensor(math.sqrt(max(int(base_logits.size(-1)), 1)))
        confidence_features = torch.cat(
            [base_probs, side_probs, margin, entropy, agreement, delta_norm],
            dim=-1,
        )
        router_input = torch.cat([query_rep, decoupled_rep, confidence_features], dim=-1)
        diagnostics = {
            "sidebranch_sp_sh_query_base_margin": margin.detach().mean(),
            "sidebranch_sp_sh_query_base_entropy": entropy.detach().mean(),
            "sidebranch_sp_sh_query_base_side_agreement": agreement.detach().mean(),
            "sidebranch_sp_sh_query_logit_delta_norm": delta_norm.detach().mean(),
        }
        return _finite_tensor(router_input), diagnostics

    def _sidebranch_query_delta(
        self,
        current_logits: torch.Tensor,
        base_logits: torch.Tensor,
        side_query_logits: torch.Tensor,
    ) -> torch.Tensor:
        mode = self.sidebranch_sp_sh_query_delta_mode
        if mode in {"side", "side_logits", "raw_side"}:
            return side_query_logits
        if mode in {"side_minus_base", "query_minus_base"}:
            return side_query_logits - base_logits
        if mode in {"side_minus_base_detach", "query_minus_base_detach", "spsh_minus_base_detach"}:
            return side_query_logits - base_logits.detach()
        if mode in {"side_minus_current_detach", "query_minus_current_detach"}:
            return side_query_logits - current_logits.detach()
        return side_query_logits - current_logits

    def _build_unit_spsh_mil_logits(
        self,
        base_logits: torch.Tensor,
        logits_dict: Dict[str, torch.Tensor],
        unit_mask: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        if (
            self.unit_spsh_mil_head is None
            or "text_prompt_reps" not in logits_dict
            or "audio_prompt_reps" not in logits_dict
            or "class_reps" not in logits_dict
        ):
            return None, {}
        fused_class_reps, mil_logits, diagnostics = self.unit_spsh_mil_head(
            logits_dict["text_prompt_reps"],
            logits_dict["audio_prompt_reps"],
            logits_dict["class_reps"],
            unit_mask,
            base_logits,
        )
        feature_logits = self.class_scorer(self._finite(fused_class_reps)).squeeze(-1)
        logit_scale = torch.exp(self.logit_scale.clamp(max=math.log(self.max_logit_scale)))
        feature_logits = self._finite(
            logit_scale * self._finite(feature_logits) + self.class_logit_bias,
            limit=1e4,
        )
        blend = feature_logits.new_tensor(self.unit_spsh_mil_logit_blend).clamp(min=0.0, max=1.0)
        mil_margin = _finite_tensor(mil_logits, limit=1e4)
        mil_margin = mil_margin - mil_margin.mean(dim=-1, keepdim=True)
        raw_final_logits = self._finite(feature_logits + blend * mil_margin, limit=1e4)
        output_mode = str(getattr(self, "unit_spsh_mil_output_mode", "replace")).lower()
        if output_mode in {"residual", "base_residual", "base_plus_residual", "delta"}:
            residual_scale = raw_final_logits.new_tensor(
                getattr(self, "unit_spsh_mil_logit_residual_scale", 1.0)
            ).clamp(min=0.0, max=1.0)
            unit_delta = raw_final_logits - base_logits.detach()
            final_logits = self._finite(base_logits + residual_scale * unit_delta, limit=1e4)
        elif output_mode in {"margin", "margin_residual", "base_plus_margin"}:
            residual_scale = raw_final_logits.new_tensor(
                getattr(self, "unit_spsh_mil_logit_residual_scale", 1.0)
            ).clamp(min=0.0, max=1.0)
            final_logits = self._finite(base_logits + residual_scale * mil_margin, limit=1e4)
        else:
            residual_scale = raw_final_logits.new_tensor(1.0)
            final_logits = raw_final_logits
        output = {
            "unit_spsh_mil_logits": _finite_tensor(mil_logits, limit=1e4),
            "unit_spsh_mil_feature_logits": _finite_tensor(feature_logits, limit=1e4),
            "unit_spsh_mil_final_logits": _finite_tensor(final_logits, limit=1e4),
            "unit_spsh_mil_class_reps": _finite_tensor(fused_class_reps),
            "unit_spsh_mil_logit_blend": blend.detach(),
            "unit_spsh_mil_output_mode": final_logits.new_tensor(
                1.0 if output_mode in {"residual", "base_residual", "base_plus_residual", "delta"} else 0.0
            ),
            "unit_spsh_mil_logit_residual_scale": residual_scale.detach(),
        }
        output.update(diagnostics)
        return final_logits, output

    def _build_hqa_spsh_logits(
        self,
        base_logits: torch.Tensor,
        logits_dict: Dict[str, torch.Tensor],
        token_parts: Dict[str, torch.Tensor],
        unit_mask: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        if (
            self.hqa_spsh_interactor is None
            or "text_prompt_reps" not in logits_dict
            or "audio_prompt_reps" not in logits_dict
            or "class_reps" not in logits_dict
            or "text_specific_units" not in token_parts
            or "audio_specific_units" not in token_parts
        ):
            return None, {}
        output = self.hqa_spsh_interactor(
            logits_dict["text_prompt_reps"],
            logits_dict["audio_prompt_reps"],
            token_parts["text_specific_units"],
            token_parts["text_shared_units"],
            token_parts["audio_specific_units"],
            token_parts["audio_shared_units"],
            logits_dict["class_reps"],
            unit_mask,
            text_unit_mask=token_parts.get("text_unit_mask"),
            audio_unit_mask=token_parts.get("audio_unit_mask"),
        )
        mlp_logits = _finite_tensor(output["hqa_logits"], limit=1e4)
        class_reps = _finite_tensor(output.get("hqa_class_reps", logits_dict["class_reps"]))
        feature_logits = self.class_scorer(self._finite(class_reps)).squeeze(-1)
        logit_scale = torch.exp(self.logit_scale.clamp(max=math.log(self.max_logit_scale)))
        feature_logits = self._finite(
            logit_scale * self._finite(feature_logits) + self.class_logit_bias,
            limit=1e4,
        )
        classifier_mode = str(getattr(self, "hqa_classifier_mode", "mlp")).lower()
        if classifier_mode in {"class", "class_scorer", "feature", "feature_scorer"}:
            hqa_logits = feature_logits
            mode_value = 1.0
        elif classifier_mode in {"blend", "class_mlp_blend", "feature_mlp_blend"}:
            blend = mlp_logits.new_tensor(getattr(self, "hqa_logit_blend", 0.50)).clamp(min=0.0, max=1.0)
            hqa_logits = (1.0 - blend) * feature_logits + blend * mlp_logits
            mode_value = 2.0
        elif classifier_mode in {"base_residual", "base_plus_delta", "residual"}:
            blend = mlp_logits.new_tensor(getattr(self, "hqa_logit_blend", 0.50)).clamp(min=0.0, max=1.0)
            hqa_logits = base_logits + blend * (mlp_logits - base_logits.detach())
            mode_value = 3.0
        else:
            hqa_logits = mlp_logits
            mode_value = 0.0
        hqa_logits = _finite_tensor(hqa_logits, limit=1e4)
        diagnostics = {
            key: value
            for key, value in output.items()
            if key != "hqa_logits"
        }
        diagnostics["hqa_mlp_logits"] = mlp_logits
        diagnostics["hqa_feature_logits"] = feature_logits
        diagnostics["hqa_logits"] = hqa_logits
        diagnostics["hqa_classifier_mode"] = hqa_logits.new_tensor(mode_value)
        diagnostics["hqa_base_delta_norm"] = (hqa_logits - base_logits.detach()).norm(dim=-1).detach().mean()
        return hqa_logits, diagnostics

    def _build_class_feature_residual_logits(
        self,
        logits: torch.Tensor,
        logits_dict: Dict[str, torch.Tensor],
        unit_mask: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        if (
            self.class_feature_residual_adapter is None
            or "class_reps" not in logits_dict
            or "text_prompt_reps" not in logits_dict
            or "audio_prompt_reps" not in logits_dict
        ):
            return None, {}
        text_pool = _masked_mean_sequence(logits_dict["text_prompt_reps"], unit_mask)
        audio_pool = _masked_mean_sequence(logits_dict["audio_prompt_reps"], unit_mask)
        global_rep = logits_dict.get("global_rep")
        if global_rep is None:
            global_rep = 0.5 * (text_pool + audio_pool)
        adapter_logits, diagnostics = self.class_feature_residual_adapter(
            logits,
            logits_dict["class_reps"],
            text_pool,
            audio_pool,
            global_rep,
        )
        diagnostics["class_feature_residual_logits"] = adapter_logits.detach()
        return adapter_logits, diagnostics

    def _build_paralinguistic_style_logits(
        self,
        logits: torch.Tensor,
        logits_dict: Dict[str, torch.Tensor],
        unit_mask: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        if (
            self.paralinguistic_style_adapter is None
            or "class_reps" not in logits_dict
            or "text_prompt_reps" not in logits_dict
            or "audio_prompt_reps" not in logits_dict
        ):
            return None, {}
        adapter_logits, diagnostics = self.paralinguistic_style_adapter(
            logits,
            logits_dict["class_reps"],
            logits_dict["text_prompt_reps"],
            logits_dict["audio_prompt_reps"],
            unit_mask,
        )
        diagnostics["paralinguistic_style_logits"] = adapter_logits.detach()
        return adapter_logits, diagnostics

    def _build_acoustic_style_logits(
        self,
        logits: torch.Tensor,
        logits_dict: Dict[str, torch.Tensor],
        audio_features: torch.Tensor,
        audio_attention_mask: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        if self.acoustic_style_query_adapter is None or "class_reps" not in logits_dict:
            return None, {}
        adapter_logits, diagnostics = self.acoustic_style_query_adapter(
            logits,
            logits_dict["class_reps"],
            audio_features,
            audio_attention_mask,
        )
        diagnostics["acoustic_style_logits"] = adapter_logits.detach()
        return adapter_logits, diagnostics

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        text_attention_mask: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        sample_ids=None,
        augment_embeddings: bool = None,
        return_loss_items: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raw_audio_features = audio_features
        raw_audio_attention_mask = audio_attention_mask
        if self.sp_sh_side_branch_only:
            base_logits, logits_dict = RefFormerAudioTextEmotionModel.forward(
                self,
                text_features,
                audio_features,
                text_attention_mask,
                audio_attention_mask,
                labels=labels,
                sample_ids=sample_ids,
                augment_embeddings=augment_embeddings,
            )
            unit_mask = self._unit_availability_mask(text_attention_mask, audio_attention_mask)
            with torch.random.fork_rng(devices=self._fork_rng_devices(base_logits), enabled=self.training):
                decoupled = self._build_decoupled_features(logits_dict, unit_mask)
            logits_dict.update(decoupled)
            logits_dict["base_ref_logits"] = base_logits
            logits = self._combine_base_and_decoupled_logits(base_logits, logits_dict)
            class_feature_logits, class_feature_diag = self._build_class_feature_residual_logits(
                logits,
                logits_dict,
                unit_mask,
            )
            if self.final_logit_source == "modal_pool":
                logits, modal_pool_diag = self._build_modal_pool_logits(logits_dict, unit_mask)
                logits_dict.update(modal_pool_diag)
                logits_dict["final_logit_source"] = "modal_pool"
            if class_feature_logits is not None:
                logits = self._finite(class_feature_logits, limit=1e4)
                logits_dict.update(class_feature_diag)
                logits_dict["final_logit_source"] = "class_feature_residual"
            if self.final_logit_source == "hqa_spsh":
                raise RuntimeError("final_logit_source=hqa_spsh requires token-prepended SP/SH mode, not side-branch-only mode")
            unit_mil_logits, unit_mil_diag = self._build_unit_spsh_mil_logits(base_logits, logits_dict, unit_mask)
            if unit_mil_logits is not None:
                logits = self._finite(unit_mil_logits, limit=1e4)
                logits_dict.update(unit_mil_diag)
                logits_dict["final_logit_source"] = "unit_spsh_mil"
            side_query_logits, side_query_diag = self._build_sidebranch_sp_sh_query_logits(logits_dict, decoupled)
            if side_query_logits is not None:
                residual_factor = self._scheduled_factor_tensor(
                    logits,
                    self.sidebranch_sp_sh_query_residual_start_epoch,
                    self.sidebranch_sp_sh_query_residual_ramp_epochs,
                )
                if self.final_logit_source == "sidebranch_sp_sh_query":
                    residual_scale = logits.new_tensor(1.0)
                    logits = self._finite(side_query_logits, limit=1e4)
                    logits_dict["final_logit_source"] = self.final_logit_source
                elif self.final_logit_source == "sidebranch_sp_sh_query_plus_base_residual":
                    residual_scale = logits.new_tensor(1.0)
                    base_residual_scale = logits.new_tensor(
                        self.sidebranch_sp_sh_query_base_residual_scale
                    )
                    logits = self._finite(
                        side_query_logits + base_residual_scale * (base_logits - side_query_logits),
                        limit=1e4,
                    )
                    logits_dict["final_logit_source"] = self.final_logit_source
                    logits_dict["sidebranch_sp_sh_query_base_residual_scale"] = (
                        base_residual_scale.detach()
                    )
                elif self.samplewise_sidebranch_sp_sh_query_gate:
                    if self.use_sidebranch_confidence_router:
                        router_input, router_diag = self._sidebranch_confidence_router_input(
                            logits_dict,
                            base_logits,
                            side_query_logits,
                        )
                        gate_logits = self.sidebranch_sp_sh_query_confidence_gate(router_input)
                        logits_dict.update(router_diag)
                        logits_dict["sidebranch_sp_sh_query_router_mode"] = logits.new_tensor(1.0)
                    else:
                        gate_logits = self.sidebranch_sp_sh_query_sample_gate(
                            _finite_tensor(logits_dict["decoupled_rep"])
                        )
                        logits_dict["sidebranch_sp_sh_query_router_mode"] = logits.new_tensor(0.0)
                    residual_scale = logits.new_tensor(self.sidebranch_sp_sh_query_gate_max) * torch.sigmoid(
                        gate_logits
                    )
                    residual_scale = residual_factor * residual_scale
                    logits_dict["sidebranch_sp_sh_query_sample_scale"] = residual_scale.detach().squeeze(-1)
                    logits_dict["sidebranch_sp_sh_query_gate_max"] = logits.new_tensor(
                        self.sidebranch_sp_sh_query_gate_max
                    )
                    query_delta = self._sidebranch_query_delta(logits, base_logits, side_query_logits)
                    logits = self._finite(logits + residual_scale * query_delta, limit=1e4)
                else:
                    residual_scale = residual_factor * logits.new_tensor(self.sidebranch_sp_sh_query_residual_scale)
                    query_delta = self._sidebranch_query_delta(logits, base_logits, side_query_logits)
                    logits = self._finite(logits + residual_scale * query_delta, limit=1e4)
                logits_dict["sidebranch_sp_sh_query_logits"] = side_query_logits
                logits_dict["sidebranch_sp_sh_query_residual_scale"] = residual_scale.detach().mean()
                logits_dict["sidebranch_sp_sh_query_residual_schedule_factor"] = residual_factor.detach()
                logits_dict["sidebranch_sp_sh_query_delta_mode"] = logits.new_tensor(
                    1.0 if self.sidebranch_sp_sh_query_delta_mode in {
                        "side_minus_base_detach",
                        "query_minus_base_detach",
                        "spsh_minus_base_detach",
                    } else 0.0
                )
                logits_dict.update(side_query_diag)
            elif self.final_logit_source in {
                "sidebranch_sp_sh_query",
                "sidebranch_sp_sh_query_plus_base_residual",
            }:
                raise RuntimeError(
                    f"final_logit_source={self.final_logit_source} requires "
                    "use_sidebranch_sp_sh_query_residual=true and use_sp_sh_query_update=true"
                )
            style_logits, style_diag = self._build_paralinguistic_style_logits(logits, logits_dict, unit_mask)
            if style_logits is not None:
                logits = self._finite(style_logits, limit=1e4)
                logits_dict.update(style_diag)
                logits_dict["final_logit_source"] = "paralinguistic_style"
            acoustic_logits, acoustic_diag = self._build_acoustic_style_logits(
                logits,
                logits_dict,
                raw_audio_features,
                raw_audio_attention_mask,
            )
            if acoustic_logits is not None:
                logits = self._finite(acoustic_logits, limit=1e4)
                logits_dict.update(acoustic_diag)
                logits_dict["final_logit_source"] = "acoustic_style"
            logits_dict["sp_sh_side_branch_only"] = logits.new_tensor(1.0)
            return self._format_forward_output(logits, logits_dict, return_loss_items)

        batch_size = text_features.size(0)
        num_units = text_features.size(1)
        text_unit_mask = text_attention_mask.bool().any(dim=-1)
        audio_unit_mask = audio_attention_mask.bool().any(dim=-1)
        text_features, text_attention_mask = self._append_sp_sh_tokens(
            text_features,
            text_attention_mask,
            self.text_sp_sh_tokens,
            self.text_sp_sh_content_projector,
        )
        audio_features, audio_attention_mask = self._append_sp_sh_tokens(
            audio_features,
            audio_attention_mask,
            self.audio_sp_sh_tokens,
            self.audio_sp_sh_content_projector,
        )
        text_seq_len = text_features.size(2)
        audio_seq_len = audio_features.size(2)
        base_logits, logits_dict = RefFormerAudioTextEmotionModel.forward(
            self,
            text_features,
            audio_features,
            text_attention_mask,
            audio_attention_mask,
            labels=labels,
            sample_ids=sample_ids,
            augment_embeddings=augment_embeddings,
        )
        token_parts = self._extract_final_sp_sh_tokens(
            logits_dict,
            text_unit_mask,
            audio_unit_mask,
            num_units,
            text_seq_len,
            audio_seq_len,
        )
        decoupled = self._build_decoupled_features_from_tokens(logits_dict, token_parts)
        logits_dict.update(decoupled)
        logits_dict["base_ref_logits"] = base_logits
        logits = self._combine_base_and_decoupled_logits(base_logits, logits_dict)
        unit_mask = text_unit_mask | audio_unit_mask
        if self.final_logit_source == "modal_pool":
            logits, modal_pool_diag = self._build_modal_pool_logits(logits_dict, unit_mask)
            logits_dict.update(modal_pool_diag)
            logits_dict["final_logit_source"] = "modal_pool"
        class_feature_logits, class_feature_diag = self._build_class_feature_residual_logits(
            logits,
            logits_dict,
            unit_mask,
        )
        if class_feature_logits is not None:
            logits = self._finite(class_feature_logits, limit=1e4)
            logits_dict.update(class_feature_diag)
            logits_dict["final_logit_source"] = "class_feature_residual"
        hqa_logits, hqa_diag = self._build_hqa_spsh_logits(base_logits, logits_dict, token_parts, unit_mask)
        if hqa_logits is not None:
            logits_dict.update(hqa_diag)
            if self.final_logit_source == "hqa_spsh":
                logits = self._finite(hqa_logits, limit=1e4)
                logits_dict["final_logit_source"] = "hqa_spsh"
        elif self.final_logit_source == "hqa_spsh":
            raise RuntimeError("final_logit_source=hqa_spsh requires use_hqa_spsh_interactor=true")
        style_logits, style_diag = self._build_paralinguistic_style_logits(logits, logits_dict, unit_mask)
        if style_logits is not None:
            logits = self._finite(style_logits, limit=1e4)
            logits_dict.update(style_diag)
            logits_dict["final_logit_source"] = "paralinguistic_style"
        acoustic_logits, acoustic_diag = self._build_acoustic_style_logits(
            logits,
            logits_dict,
            raw_audio_features,
            raw_audio_attention_mask,
        )
        if acoustic_logits is not None:
            logits = self._finite(acoustic_logits, limit=1e4)
            logits_dict.update(acoustic_diag)
            logits_dict["final_logit_source"] = "acoustic_style"
        logits_dict["sp_sh_token_scale"] = logits.new_tensor(self.sp_sh_token_scale)
        logits_dict["sp_sh_tokens_per_type"] = logits.new_tensor(
            float(self.sp_sh_tokens_per_type)
        )
        if self.sp_sh_content_conditioned_tokens:
            content_gate = torch.sigmoid(
                self.sp_sh_content_gate.to(device=logits.device, dtype=logits.dtype)
            )
            logits_dict["sp_sh_content_gate"] = content_gate.detach()
            logits_dict["sp_sh_content_effective_scale"] = (
                logits.new_tensor(self.sp_sh_content_scale) * content_gate.detach()
            )
        return self._format_forward_output(logits, logits_dict, return_loss_items)
