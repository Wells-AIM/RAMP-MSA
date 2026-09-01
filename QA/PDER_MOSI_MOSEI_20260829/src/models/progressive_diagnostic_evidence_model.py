"""Progressive Diagnostic Evidence Reasoning (PDER) model.

This model is a paper-clean alternative to the token-prepended SP/SH variant.
It keeps the validated RefFormer warm-start backbone and changes only the full
stage reasoning path:

    projected text/audio evidence
        -> evidence-only Consensus/Complementary Registers
        -> 2 x general diagnostic query layers
        -> Consensus refinement + QA3 consolidation
        -> Complementary refinement + QA4 consolidation
        -> class-conditioned diagnosis

Important design choices
------------------------
1. Evidence registers are extracted *before* any query-to-token feedback. They
   never live inside the main text/audio token sequence, so diagnostic queries
   cannot see them during general evidence acquisition (no early leakage).
2. Consensus and Complementary refiners are separate modules with independent
   parameters and an explicit C -> P order.
3. The existing RefQFormer loss remains compatible through aliases:
      text_shared/audio_shared     == consensus registers
      text_specific/audio_specific == complementary registers
4. The warm checkpoint remains partially load-compatible because the original
   projections, class queries, four QA layers, scorers and calibrators retain
   their parameter names.

The code intentionally does not claim that the complementary registers are
strictly modality-private. The training constraint makes them complementary to
consensus within each modality, which is the mathematically supported claim.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unified_referential_evidence_query_model import RefFormerAudioTextEmotionModel


def _finite_tensor(tensor: torch.Tensor, limit: float = 50.0) -> torch.Tensor:
    if tensor is None:
        return tensor
    return torch.nan_to_num(tensor, nan=0.0, posinf=limit, neginf=-limit).clamp(-limit, limit)


class UnitEvidenceRegisterExtractor(nn.Module):
    """Extract one or more consensus/complementary slots from each evidence unit.

    The extractor only reads one modality's evidence tokens. It is run on the
    projected evidence *before* diagnostic queries modify the token stream.
    Therefore its outputs are evidence-derived registers rather than query
    projections.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        slots_per_type: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.slots_per_type = max(1, int(slots_per_type))

        self.consensus_seed = nn.Parameter(
            torch.randn(self.slots_per_type, hidden_dim) * 0.02
        )
        self.complement_seed = nn.Parameter(
            torch.randn(self.slots_per_type, hidden_dim) * 0.02
        )

        # Separate operators are deliberate: consensus and complementarity
        # have different semantic roles and should not be forced through the
        # same projection parameters.
        self.consensus_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.complement_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.consensus_norm1 = nn.LayerNorm(hidden_dim)
        self.consensus_norm2 = nn.LayerNorm(hidden_dim)
        self.complement_norm1 = nn.LayerNorm(hidden_dim)
        self.complement_norm2 = nn.LayerNorm(hidden_dim)
        self.consensus_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.complement_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def _extract(
        self,
        seed: torch.Tensor,
        attn: nn.MultiheadAttention,
        norm1: nn.LayerNorm,
        norm2: nn.LayerNorm,
        ffn: nn.Module,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_units = memory.size(0)
        q = seed.unsqueeze(0).expand(batch_units, -1, -1)
        context, weights = attn(
            q,
            _finite_tensor(memory),
            _finite_tensor(memory),
            key_padding_mask=~memory_mask.bool(),
            need_weights=True,
            average_attn_weights=False,
        )
        context = _finite_tensor(context)
        slots = _finite_tensor(norm1(q + self.dropout(context)))
        slots = _finite_tensor(norm2(slots + ffn(slots)))
        # [BU, heads, slots, L] -> [BU, slots, L]
        weights = _finite_tensor(weights.detach()).mean(dim=1)
        return slots, weights

    def forward(
        self,
        flat_tokens: torch.Tensor,
        flat_mask: torch.Tensor,
        num_units: int,
        seq_len: int,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, _, hidden_dim = flat_tokens.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"register hidden dim mismatch: got {hidden_dim}, expected {self.hidden_dim}"
            )
        grouped_tokens = flat_tokens.reshape(batch_size, num_units, seq_len, hidden_dim)
        grouped_mask = flat_mask.reshape(batch_size, num_units, seq_len).bool()
        unit_valid = grouped_mask.any(dim=-1)

        memory = grouped_tokens.reshape(batch_size * num_units, seq_len, hidden_dim).clone()
        memory_mask = grouped_mask.reshape(batch_size * num_units, seq_len).clone()
        missing = ~memory_mask.any(dim=1)
        if missing.any():
            memory[missing, 0, :] = 0.0
            memory_mask[missing, 0] = True

        c_slots, c_attn = self._extract(
            self.consensus_seed,
            self.consensus_attn,
            self.consensus_norm1,
            self.consensus_norm2,
            self.consensus_ffn,
            memory,
            memory_mask,
        )
        p_slots, p_attn = self._extract(
            self.complement_seed,
            self.complement_attn,
            self.complement_norm1,
            self.complement_norm2,
            self.complement_ffn,
            memory,
            memory_mask,
        )

        c_slots = c_slots.reshape(batch_size, num_units, self.slots_per_type, hidden_dim)
        p_slots = p_slots.reshape(batch_size, num_units, self.slots_per_type, hidden_dim)
        valid_scale = unit_valid.to(dtype=c_slots.dtype).unsqueeze(-1).unsqueeze(-1)
        c_slots = _finite_tensor(c_slots * valid_scale)
        p_slots = _finite_tensor(p_slots * valid_scale)
        c_units = c_slots.mean(dim=2)
        p_units = p_slots.mean(dim=2)

        diagnostics: Dict[str, torch.Tensor] = {
            "unit_valid_ratio": unit_valid.float().mean().detach(),
            "consensus_norm": c_units.detach().norm(dim=-1).mean(),
            "complement_norm": p_units.detach().norm(dim=-1).mean(),
        }
        if return_attention:
            diagnostics["consensus_token_attention"] = c_attn.reshape(
                batch_size, num_units, self.slots_per_type, seq_len
            )
            diagnostics["complement_token_attention"] = p_attn.reshape(
                batch_size, num_units, self.slots_per_type, seq_len
            )
        return c_units, p_units, unit_valid, diagnostics


class DiagnosticEvidenceRefiner(nn.Module):
    """Gated diagnostic-query refinement from one structured evidence memory."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        max_scale: float,
        gate_init: float = -1.5,
        schedule_entire_refiner: bool = False,
    ) -> None:
        super().__init__()
        self.max_scale = float(max_scale)
        self.schedule_entire_refiner = bool(schedule_entire_refiner)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, float(gate_init))
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        schedule_factor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        queries = _finite_tensor(queries)
        memory = _finite_tensor(memory).masked_fill(~memory_mask.bool().unsqueeze(-1), 0.0)
        context, attention = self.cross_attn(
            queries,
            memory,
            memory,
            key_padding_mask=~memory_mask.bool(),
            need_weights=True,
            average_attn_weights=False,
        )
        context = _finite_tensor(context)
        gate_input = torch.cat(
            [queries, context, torch.abs(queries - context), queries * context], dim=-1
        )
        gate = torch.sigmoid(self.gate(_finite_tensor(gate_input)))
        schedule = schedule_factor.to(device=queries.device, dtype=queries.dtype)
        if self.schedule_entire_refiner:
            raw_delta = queries.new_tensor(self.max_scale) * gate * self.dropout(context)
            refined = self.norm(queries + raw_delta)
            delta = schedule * (refined - queries)
            updated = _finite_tensor(queries + delta)
            scale = queries.new_tensor(self.max_scale) * schedule
        else:
            scale = queries.new_tensor(self.max_scale) * schedule
            delta = scale * gate * self.dropout(context)
            updated = _finite_tensor(self.norm(queries + delta))
        diagnostics = {
            "gate_mean": gate.detach().mean(),
            "effective_scale": scale.detach().reshape(()),
            "context_norm": context.detach().norm(dim=-1).mean(),
            "update_norm_ratio": delta.detach().norm(dim=-1).mean()
            / queries.detach().norm(dim=-1).mean().clamp_min(1e-6),
            # [B, heads, Q, M] -> [B, Q, M]
            "attention": _finite_tensor(attention.detach()).mean(dim=1),
        }
        return updated, diagnostics


class ProgressiveDiagnosticEvidenceReasoningModel(RefFormerAudioTextEmotionModel):
    """General -> Consensus -> Complementarity diagnostic evidence reasoning.

    Four inherited QA layers are assigned explicit roles by default:
      QA1, QA2 : general multimodal evidence acquisition
      C -> QA3 : consensus grounding and consolidation
      P -> QA4 : complementary refinement and consolidation

    The model is designed as a drop-in full-stage replacement for the existing
    warm-to-full pipeline.
    """

    def __init__(self, config: Dict[str, object]):
        super().__init__(config)
        model_cfg = config.get("model", {})
        train_cfg = config.get("training", {})
        num_heads = int(model_cfg.get("num_heads", 4))

        self.pder_general_layers = int(model_cfg.get("pder_general_layers", 2))
        if self.pder_general_layers < 1:
            raise ValueError("model.pder_general_layers must be >= 1")
        if self.num_query_layers < self.pder_general_layers + 2:
            raise ValueError(
                "PDER needs at least general_layers + 2 QA layers; "
                f"got num_query_layers={self.num_query_layers}, "
                f"general_layers={self.pder_general_layers}"
            )

        self.pder_register_slots_per_type = int(
            model_cfg.get("pder_register_slots_per_type", 1)
        )
        self.pder_return_attention_maps = bool(
            model_cfg.get("pder_return_attention_maps", False)
        )
        self.pder_consensus_scale = float(model_cfg.get("pder_consensus_scale", 0.08))
        self.pder_complement_scale = float(model_cfg.get("pder_complement_scale", 0.06))
        self.pder_probe_affinity_temperature = float(
            model_cfg.get("pder_probe_affinity_temperature", 0.20)
        )
        self.pder_schedule_entire_refiner = bool(
            model_cfg.get("pder_schedule_entire_refiner", False)
        )
        self.pder_allow_late_referential_update = bool(
            model_cfg.get("pder_allow_late_referential_update", False)
        )
        self.pder_evidence_registration_timing = str(
            model_cfg.get("pder_evidence_registration_timing", "pre")
        ).strip().lower()
        valid_evidence_timings = {"pre", "postg", "interleaved"}
        if self.pder_evidence_registration_timing not in valid_evidence_timings:
            raise ValueError(
                "model.pder_evidence_registration_timing must be one of "
                f"{sorted(valid_evidence_timings)}, got "
                f"{self.pder_evidence_registration_timing!r}"
            )
        self.pder_refinement_start_epoch = int(
            train_cfg.get(
                "pder_refinement_start_epoch",
                model_cfg.get("pder_refinement_start_epoch", 3),
            )
        )
        self.pder_refinement_ramp_epochs = int(
            train_cfg.get(
                "pder_refinement_ramp_epochs",
                model_cfg.get("pder_refinement_ramp_epochs", 5),
            )
        )
        # Persist progress in checkpoints so standalone test uses the same
        # refinement strength as the selected training epoch.
        self.register_buffer("pder_schedule_progress", torch.tensor(0.0), persistent=True)

        self.text_evidence_registers = UnitEvidenceRegisterExtractor(
            self.hidden_dim,
            num_heads,
            self.dropout_rate,
            slots_per_type=self.pder_register_slots_per_type,
        )
        self.audio_evidence_registers = UnitEvidenceRegisterExtractor(
            self.hidden_dim,
            num_heads,
            self.dropout_rate,
            slots_per_type=self.pder_register_slots_per_type,
        )
        self.consensus_refiner = DiagnosticEvidenceRefiner(
            self.hidden_dim,
            num_heads,
            self.dropout_rate,
            max_scale=self.pder_consensus_scale,
            gate_init=float(model_cfg.get("pder_consensus_gate_init", -1.5)),
            schedule_entire_refiner=self.pder_schedule_entire_refiner,
        )
        self.complement_refiner = DiagnosticEvidenceRefiner(
            self.hidden_dim,
            num_heads,
            self.dropout_rate,
            max_scale=self.pder_complement_scale,
            gate_init=float(model_cfg.get("pder_complement_gate_init", -1.5)),
            schedule_entire_refiner=self.pder_schedule_entire_refiner,
        )

    def _schedule_value(self, epoch_index: int) -> float:
        start = max(0, int(self.pder_refinement_start_epoch))
        ramp = max(0, int(self.pder_refinement_ramp_epochs))
        epoch = max(0, int(epoch_index))
        if epoch < start:
            return 0.0
        if ramp <= 0:
            return 1.0
        return min(1.0, max(0.0, float(epoch - start + 1) / float(ramp)))

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
        self.pder_schedule_progress.fill_(self._schedule_value(epoch_index))

    @staticmethod
    def _safe_memory(
        memory: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = mask.bool()
        missing = ~mask.any(dim=1)
        if missing.any():
            memory = memory.clone()
            mask = mask.clone()
            memory[missing, 0, :] = 0.0
            mask[missing, 0] = True
        return memory, mask

    def _build_register_memories(
        self,
        text_tokens: torch.Tensor,
        audio_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        num_units: int,
        text_seq_len: int,
        audio_seq_len: int,
    ):
        text_c, text_p, text_unit_mask, text_diag = self.text_evidence_registers(
            text_tokens,
            text_mask,
            num_units,
            text_seq_len,
            return_attention=self.pder_return_attention_maps,
        )
        audio_c, audio_p, audio_unit_mask, audio_diag = self.audio_evidence_registers(
            audio_tokens,
            audio_mask,
            num_units,
            audio_seq_len,
            return_attention=self.pder_return_attention_maps,
        )
        consensus_memory = torch.cat([text_c, audio_c], dim=1)
        consensus_mask = torch.cat([text_unit_mask, audio_unit_mask], dim=1)
        complement_memory = torch.cat([text_p, audio_p], dim=1)
        complement_mask = torch.cat([text_unit_mask, audio_unit_mask], dim=1)
        consensus_memory, consensus_mask = self._safe_memory(consensus_memory, consensus_mask)
        complement_memory, complement_mask = self._safe_memory(complement_memory, complement_mask)
        return (
            text_c,
            text_p,
            audio_c,
            audio_p,
            text_unit_mask,
            audio_unit_mask,
            consensus_memory,
            consensus_mask,
            complement_memory,
            complement_mask,
            text_diag,
            audio_diag,
        )

    def _layer_step(
        self,
        layer_index: int,
        queries: torch.Tensor,
        text_tokens: torch.Tensor,
        audio_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        num_units: int,
        text_seq_len: int,
        audio_seq_len: int,
        allow_referential_update: bool,
    ):
        text_unit_reps, audio_unit_reps, unit_weights, class_unit_context, unit_query_context = self._build_unit_context(
            text_tokens,
            audio_tokens,
            text_mask,
            audio_mask,
            num_units,
            text_seq_len,
            audio_seq_len,
        )
        if allow_referential_update and (not self.disable_query_adaption):
            unit_evidence = self._compose_unit_evidence(text_unit_reps, audio_unit_reps)
            queries, _, _ = self._referential_query_update(queries, unit_evidence, self._safe_unit_mask_from_unit_masks(text_unit_reps, audio_unit_reps, text_mask, audio_mask, num_units, text_seq_len, audio_seq_len))
        queries, text_tokens, audio_tokens = self.query_layers[layer_index](
            queries,
            text_tokens,
            audio_tokens,
            text_mask,
            audio_mask,
            self._finite(unit_query_context),
            None,
            None,
            None,
        )
        return (
            self._finite(queries),
            self._finite(text_tokens),
            self._finite(audio_tokens),
            self._finite(text_unit_reps),
            self._finite(audio_unit_reps),
            unit_weights,
            self._finite(class_unit_context),
        )

    def _safe_unit_mask_from_unit_masks(
        self,
        text_unit_reps: torch.Tensor,
        audio_unit_reps: torch.Tensor,
        text_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        num_units: int,
        text_seq_len: int,
        audio_seq_len: int,
    ) -> torch.Tensor:
        del text_unit_reps, audio_unit_reps  # only kept to make call sites explicit
        text_unit_mask = text_mask.reshape(text_mask.size(0), num_units, text_seq_len).any(dim=-1)
        audio_unit_mask = audio_mask.reshape(audio_mask.size(0), num_units, audio_seq_len).any(dim=-1)
        unit_mask = text_unit_mask | audio_unit_mask
        missing = ~unit_mask.any(dim=1)
        if missing.any():
            unit_mask = unit_mask.clone()
            unit_mask[missing, 0] = True
        return unit_mask

    def _probe_unit_affinity(
        self,
        queries: torch.Tensor,
        unit_evidence: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> torch.Tensor:
        grouped = queries.reshape(
            queries.size(0), self.num_classes, self.queries_per_class, self.hidden_dim
        )
        q = F.normalize(_finite_tensor(grouped), dim=-1)
        e = F.normalize(_finite_tensor(unit_evidence), dim=-1)
        temperature = max(float(self.pder_probe_affinity_temperature), 1e-4)
        scores = torch.einsum("bckh,buh->bcku", q, e) / temperature
        scores = scores.masked_fill(~unit_mask[:, None, None, :].bool(), -1e4)
        return torch.softmax(scores, dim=-1).detach()

    def _forward_impl(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        text_attention_mask: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        sample_ids=None,
        augment_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del labels, sample_ids  # PDER v1 intentionally excludes cross-sample branches.
        if self.use_cross_sample:
            raise RuntimeError("PDER v1 does not support cross_sample; set cross_sample.use_cross_sample=false")

        batch_size = text_features.size(0)
        num_units = text_features.size(1)
        text_seq_len = text_features.size(2)
        audio_seq_len = audio_features.size(2)

        text_tokens, text_mask = self._flatten_tokens(
            text_features, text_attention_mask, self.text_projection, 0
        )
        audio_tokens, audio_mask = self._flatten_tokens(
            audio_features, audio_attention_mask, self.audio_projection, 1
        )
        embedding_aug_stats: Dict[str, torch.Tensor] = {}
        if augment_embeddings:
            text_tokens, audio_tokens, embedding_aug_stats = self._apply_embedding_augmentation(
                text_tokens, audio_tokens, text_mask, audio_mask
            )

        evidence_registration_timing = self.pder_evidence_registration_timing

        def build_evidence_snapshot(current_text_tokens, current_audio_tokens):
            return self._build_register_memories(
                current_text_tokens,
                current_audio_tokens,
                text_mask,
                audio_mask,
                num_units,
                text_seq_len,
                audio_seq_len,
            )

        def make_unit_mask(current_text_unit_mask, current_audio_unit_mask):
            current_unit_mask = current_text_unit_mask | current_audio_unit_mask
            missing_rows = ~current_unit_mask.any(dim=1)
            if missing_rows.any():
                current_unit_mask = current_unit_mask.clone()
                current_unit_mask[missing_rows, 0] = True
            return current_unit_mask

        register_state = (None,) * 12
        unit_mask = None
        if evidence_registration_timing == "pre":
            register_state = build_evidence_snapshot(text_tokens, audio_tokens)
            unit_mask = make_unit_mask(register_state[4], register_state[5])

        (
            text_consensus,
            text_complement,
            audio_consensus,
            audio_complement,
            text_unit_mask,
            audio_unit_mask,
            consensus_memory,
            consensus_memory_mask,
            complement_memory,
            complement_memory_mask,
            text_register_diag,
            audio_register_diag,
        ) = register_state

        queries, q0_diagnostics = self._initialize_queries(
            batch_size,
            text_tokens,
            text_mask,
            audio_tokens,
            audio_mask,
        )
        initial_queries = queries.detach()

        layer_logits = []
        layer_logits_by_number: Dict[int, torch.Tensor] = {}
        class_unit_context = None
        text_unit_reps = None
        audio_unit_reps = None
        unit_weights = None

        # Stage G: general evidence acquisition. Registers are not visible here.
        for layer_index in range(self.pder_general_layers):
            (
                queries,
                text_tokens,
                audio_tokens,
                text_unit_reps,
                audio_unit_reps,
                unit_weights,
                class_unit_context,
            ) = self._layer_step(
                layer_index,
                queries,
                text_tokens,
                audio_tokens,
                text_mask,
                audio_mask,
                num_units,
                text_seq_len,
                audio_seq_len,
                allow_referential_update=self.referential_update_mode in {"per_layer", "all"},
            )
            class_reps_i, _ = self._pool_class_queries(queries)
            logits_i, _, _, _ = self._score_queries(class_reps_i, class_unit_context)
            layer_number = layer_index + 1
            layer_logits.append(logits_i)
            layer_logits_by_number[layer_number] = logits_i

        general_queries = queries
        general_class_reps, _ = self._pool_class_queries(general_queries)
        general_logits, _, _, _ = self._score_queries(general_class_reps, class_unit_context)

        if evidence_registration_timing in {"postg", "interleaved"}:
            (
                text_consensus,
                text_complement,
                audio_consensus,
                audio_complement,
                text_unit_mask,
                audio_unit_mask,
                consensus_memory,
                consensus_memory_mask,
                complement_memory,
                complement_memory_mask,
                text_register_diag,
                audio_register_diag,
            ) = build_evidence_snapshot(text_tokens, audio_tokens)
            unit_mask = make_unit_mask(text_unit_mask, audio_unit_mask)

        # Stage C: cross-modal consensus grounds the diagnostic hypotheses.
        schedule_factor = self.pder_schedule_progress.to(
            device=queries.device, dtype=queries.dtype
        )
        queries, consensus_diag = self.consensus_refiner(
            queries, consensus_memory, consensus_memory_mask, schedule_factor
        )
        consensus_layer_index = self.pder_general_layers
        (
            queries,
            text_tokens,
            audio_tokens,
            text_unit_reps,
            audio_unit_reps,
            unit_weights,
            class_unit_context,
        ) = self._layer_step(
            consensus_layer_index,
            queries,
            text_tokens,
            audio_tokens,
            text_mask,
            audio_mask,
            num_units,
            text_seq_len,
            audio_seq_len,
            allow_referential_update=(
                self.pder_allow_late_referential_update
                and self.referential_update_mode in {"per_layer", "all"}
            ),
        )
        consensus_queries = queries
        consensus_class_reps, _ = self._pool_class_queries(consensus_queries)
        consensus_logits, _, _, _ = self._score_queries(
            consensus_class_reps, class_unit_context
        )
        layer_logits.append(consensus_logits)
        layer_logits_by_number[consensus_layer_index + 1] = consensus_logits

        if evidence_registration_timing == "interleaved":
            (
                _postc_text_consensus,
                text_complement,
                _postc_audio_consensus,
                audio_complement,
                postc_text_unit_mask,
                postc_audio_unit_mask,
                _postc_consensus_memory,
                _postc_consensus_memory_mask,
                complement_memory,
                complement_memory_mask,
                _postc_text_register_diag,
                _postc_audio_register_diag,
            ) = build_evidence_snapshot(text_tokens, audio_tokens)
            if not torch.equal(postc_text_unit_mask, text_unit_mask) or not torch.equal(
                postc_audio_unit_mask, audio_unit_mask
            ):
                raise RuntimeError("Evidence masks changed between post-G and post-C snapshots")

        # Stage P: complementary evidence corrects/refines the grounded state.
        queries, complement_diag = self.complement_refiner(
            queries, complement_memory, complement_memory_mask, schedule_factor
        )
        complement_layer_index = self.pder_general_layers + 1
        (
            queries,
            text_tokens,
            audio_tokens,
            text_unit_reps,
            audio_unit_reps,
            unit_weights,
            class_unit_context,
        ) = self._layer_step(
            complement_layer_index,
            queries,
            text_tokens,
            audio_tokens,
            text_mask,
            audio_mask,
            num_units,
            text_seq_len,
            audio_seq_len,
            allow_referential_update=(
                self.pder_allow_late_referential_update
                and self.referential_update_mode in {"per_layer", "all"}
            ),
        )

        # Any extra QA layers are post-refinement consolidation. Current configs
        # use exactly four layers, so this branch is normally empty.
        for layer_index in range(complement_layer_index + 1, self.num_query_layers):
            (
                queries,
                text_tokens,
                audio_tokens,
                text_unit_reps,
                audio_unit_reps,
                unit_weights,
                class_unit_context,
            ) = self._layer_step(
                layer_index,
                queries,
                text_tokens,
                audio_tokens,
                text_mask,
                audio_mask,
                num_units,
                text_seq_len,
                audio_seq_len,
                allow_referential_update=False,
            )

        final_class_reps, query_weights = self._pool_class_queries(queries)
        final_query_logits, base_logits, evidence_logits, evidence_strength = self._score_queries(
            final_class_reps, class_unit_context
        )
        layer_logits.append(final_query_logits)
        layer_logits_by_number[complement_layer_index + 1] = final_query_logits
        layer_logits_tensor = torch.stack(layer_logits, dim=0)
        query_logits = (
            (1.0 - self.layer_logit_weight) * final_query_logits
            + self.layer_logit_weight * layer_logits_tensor.mean(dim=0)
        )
        logit_scale = torch.exp(
            self.logit_scale.clamp(max=math.log(self.max_logit_scale))
        )
        logits = self._finite(
            logit_scale * self._finite(query_logits) + self.class_logit_bias,
            limit=1e4,
        )

        final_unit_evidence = self._compose_unit_evidence(
            self._finite(text_unit_reps), self._finite(audio_unit_reps)
        )
        probe_unit_affinity = self._probe_unit_affinity(
            queries, final_unit_evidence, unit_mask
        )
        query_delta = queries.detach() - initial_queries
        query_update_norm_ratio = query_delta.norm(dim=-1).mean() / initial_queries.norm(
            dim=-1
        ).mean().clamp_min(1e-6)
        if self.num_classes > 1:
            query_class_cosine = F.cosine_similarity(
                final_class_reps[:, 0, :], final_class_reps[:, 1, :], dim=-1
            ).mean()
        else:
            query_class_cosine = final_class_reps.new_tensor(1.0)

        logits_dict: Dict[str, torch.Tensor] = {
            "query": logits,
            "base_query": base_logits,
            "evidence_query": evidence_logits,
            "score_evidence_strength": evidence_strength,
            "class_reps": final_class_reps,
            "queries": queries,
            "final_queries": queries,
            "general_queries": general_queries,
            "consensus_queries": consensus_queries,
            "query_weights": query_weights,
            "num_queries_per_class": queries.new_tensor(float(self.queries_per_class)),
            "query_update_norm_ratio": query_update_norm_ratio.detach(),
            "query_class_cosine": query_class_cosine.detach(),
            "layer_logits": layer_logits_tensor,
            # RefQFormerLoss-compatible stage supervision:
            # Q2 = general state, Q3 = consensus-grounded state.
            "aux_logits_q2": general_logits,
            "aux_logits_q3": consensus_logits,
            # Existing loss aliases: shared == consensus, specific == complement.
            "text_shared": text_consensus,
            "audio_shared": audio_consensus,
            "text_specific": text_complement,
            "audio_specific": audio_complement,
            "text_unit_mask": text_unit_mask,
            "audio_unit_mask": audio_unit_mask,
            "text_prompt_reps": text_unit_reps,
            "audio_prompt_reps": audio_unit_reps,
            "unit_weights": unit_weights,
            "prompt_weights": unit_weights,
            "probe_unit_affinity": probe_unit_affinity,
            "pder_schedule_progress": schedule_factor.detach(),
            "pder_evidence_timing_code": queries.new_tensor(
                float({"pre": 0, "postg": 1, "interleaved": 2}[evidence_registration_timing])
            ).detach(),
            "pder_consensus_gate_mean": consensus_diag["gate_mean"],
            "pder_consensus_update_norm_ratio": consensus_diag["update_norm_ratio"],
            "pder_consensus_effective_scale": consensus_diag["effective_scale"],
            "pder_complement_gate_mean": complement_diag["gate_mean"],
            "pder_complement_update_norm_ratio": complement_diag["update_norm_ratio"],
            "pder_complement_effective_scale": complement_diag["effective_scale"],
            "consensus_refiner_attention": consensus_diag["attention"],
            "complement_refiner_attention": complement_diag["attention"],
            "active_query_layer_mask": queries.new_ones(self.num_query_layers),
            "active_query_layer_count": queries.new_tensor(float(self.num_query_layers)),
        }
        for prefix, diag in (("text_register", text_register_diag), ("audio_register", audio_register_diag)):
            for key, value in diag.items():
                logits_dict[f"pder_{prefix}_{key}"] = value
        if q0_diagnostics:
            logits_dict.update(q0_diagnostics)
        if embedding_aug_stats:
            logits_dict.update(embedding_aug_stats)
        return logits, logits_dict
