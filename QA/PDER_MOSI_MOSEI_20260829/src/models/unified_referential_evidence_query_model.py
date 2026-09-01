"""Referential evidence-unit text+audio QA model.

This variant keeps the unified 12-evidence-unit input schema and the same
label-space-auto training protocol, but strengthens the query/evidence
interaction with three dataset-agnostic blocks:
- class queries first attend evidence-unit memory before reading raw tokens;
- text evidence lightly stimulates matching audio evidence tokens;
- final text/audio/query/global branches are fused by a self gate.
"""
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cross_sample import (
    CrossSampleProjector,
    build_batch_positive_refs,
    supervised_contrastive_loss,
    supervised_contrastive_loss_with_memory,
    weighted_supervised_contrastive_loss_with_memory,
    ccr_margin_retrieval_loss_with_memory,
    reliable_logit_distillation_loss,
    build_query_relation_graph,
    relation_logit_consistency_loss,
    BatchRelationEncoder,
    build_batch_relation_nodes,
    relation_auxiliary_ce_loss,
    relation_aware_loss_weights,
    boundary_logit_margin_loss,
)
from src.models.unified_evidence_query_model import UnifiedEvidenceTextAudioQueryModel


def _finite_tensor(tensor: torch.Tensor, limit: float = 50.0) -> torch.Tensor:
    if tensor is None:
        return tensor
    return torch.nan_to_num(tensor, nan=0.0, posinf=limit, neginf=-limit).clamp(min=-limit, max=limit)


class QueryAdaptionModule(nn.Module):
    """RefFormer-style Query Adaption Module for audio-text emotion recognition.

    The module updates class-conditioned emotion queries by letting them read
    the current text/audio evidence-unit memory. Across stacked layers, the
    trainable queries move from random initialization toward sample-aware,
    task-relevant emotion queries.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, scale: float):
        super().__init__()
        self.scale = float(scale)
        self.query_to_evidence = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.update_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        queries: torch.Tensor,
        evidence_memory: torch.Tensor,
        evidence_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.scale <= 0:
            zeros = queries.new_zeros(queries.shape)
            return queries, zeros, zeros

        queries = _finite_tensor(queries)
        memory = _finite_tensor(evidence_memory).masked_fill(~evidence_mask.bool().unsqueeze(-1), 0.0)
        reference, _ = self.query_to_evidence(
            queries,
            memory,
            memory,
            key_padding_mask=~evidence_mask.bool(),
            need_weights=False,
        )
        reference = _finite_tensor(reference)
        gate_input = torch.cat(
            [queries, reference, torch.abs(queries - reference), queries * reference],
            dim=-1,
        )
        gate = self.update_gate(_finite_tensor(gate_input))
        updated = _finite_tensor(self.norm(queries + self.scale * gate * reference))
        return updated, reference, gate


class CrossSampleQueryRelationAdapter(nn.Module):
    """Small gated query updater driven by cross-sample relation context."""

    def __init__(self, hidden_dim: int, dropout: float, gate_init: float = -4.0, max_scale: float = 0.05):
        super().__init__()
        self.max_scale = float(max_scale)
        self.node_projector = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.query_update = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def build_node(self, q_rep: torch.Tensor, text_rep: torch.Tensor, audio_rep: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.node_projector(torch.cat([q_rep, text_rep, audio_rep], dim=-1)), dim=-1)

    def forward(self, queries: torch.Tensor, relation_context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        context = relation_context[:, None, :].expand(-1, queries.size(1), -1)
        update_input = torch.cat([queries, context, torch.abs(queries - context), queries * context], dim=-1)
        update = self.query_update(_finite_tensor(update_input))
        gate = self.max_scale * torch.sigmoid(self.gate_logit)
        return _finite_tensor(queries + gate * update), gate.detach()


class CrossSampleBatchRelationAuxAdapter(nn.Module):
    """Train-only BatchFormer-style auxiliary query branch."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, gate_init: float = -3.0, max_scale: float = 0.10):
        super().__init__()
        self.max_scale = float(max_scale)
        self.node_projector = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3), nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.encoder = BatchRelationEncoder(hidden_dim, num_heads=num_heads, dropout=dropout)
        self.query_update = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4), nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim)
        )
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, queries: torch.Tensor, q_rep: torch.Tensor, text_rep: torch.Tensor, audio_rep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        node = build_batch_relation_nodes(q_rep, text_rep, audio_rep, self.node_projector)
        relation_node, diagnostics = self.encoder(node)
        context = relation_node[:, None, :].expand(-1, queries.size(1), -1)
        update_input = torch.cat([queries, context, torch.abs(queries - context), queries * context], dim=-1)
        update = self.query_update(_finite_tensor(update_input))
        gate = self.max_scale * torch.sigmoid(self.gate_logit)
        delta = gate * update
        aux_queries = _finite_tensor(queries + delta)
        query_norm = queries.detach().norm(dim=-1).mean().clamp_min(1e-8)
        diagnostics = dict(diagnostics)
        diagnostics["relation_gate"] = gate.detach()
        diagnostics["relation_delta_norm"] = delta.detach().norm(dim=-1).mean() / query_norm
        return aux_queries, node, relation_node, diagnostics


class UnifiedReferentialEvidenceQueryModel(UnifiedEvidenceTextAudioQueryModel):
    """A single fixed referential QA network for MELD, DAIC-WOZ, and EATD."""

    def __init__(self, config: Dict[str, object]):
        super().__init__(config)
        model_config = config["model"]
        training_config = config.get("training", {})
        num_heads = int(model_config.get("num_heads", 4))

        raw_active_query_layers = model_config.get("active_query_layers")
        if raw_active_query_layers is None:
            active_query_layers = list(range(1, self.num_query_layers + 1))
        elif isinstance(raw_active_query_layers, (list, tuple, set)):
            active_query_layers = sorted({int(layer) for layer in raw_active_query_layers})
        else:
            raise ValueError("model.active_query_layers must be a list of 1-based layer indices")
        invalid_query_layers = [
            layer
            for layer in active_query_layers
            if layer < 1 or layer > self.num_query_layers
        ]
        if invalid_query_layers:
            raise ValueError(
                "model.active_query_layers contains indices outside "
                f"[1, {self.num_query_layers}]: {invalid_query_layers}"
            )
        self.active_query_layers = tuple(active_query_layers)
        self.active_query_layer_set = frozenset(active_query_layers)

        self.referential_query_scale = float(model_config.get("referential_query_scale", 0.35))
        self.referential_update_mode = str(model_config.get("referential_update_mode", "per_layer")).lower()
        self.text_guided_audio_scale = float(model_config.get("text_guided_audio_scale", 0.20))
        self.text_guided_audio_mode = str(model_config.get("text_guided_audio_mode", "per_layer")).lower()
        self.self_gated_fusion_scale = float(model_config.get("self_gated_fusion_scale", 0.30))
        self.teacher_leading_loss_weight = float(training_config.get("teacher_leading_loss_weight", 0.04))
        self.teacher_temperature = float(training_config.get("teacher_temperature", 2.0))
        augmentation_config = config.get("embedding_augmentation", {})
        self.use_embedding_augmentation = bool(augmentation_config.get("use_embedding_augmentation", False))
        self.embedding_augmentation_clean_augmented_views = bool(
            augmentation_config.get("use_clean_augmented_views", True)
        )
        self.embedding_augmented_loss_weight = float(augmentation_config.get("augmented_loss_weight", 0.25))
        self.audio_mask_prob = float(augmentation_config.get("audio_mask_prob", 0.05))
        self.audio_noise_std = float(augmentation_config.get("audio_noise_std", 0.01))
        self.text_mask_prob = float(augmentation_config.get("text_mask_prob", 0.03))
        self.text_noise_std = float(augmentation_config.get("text_noise_std", 0.005))
        self.use_modality_dropout = bool(augmentation_config.get("use_modality_dropout", False))
        self.modality_dropout_prob = float(augmentation_config.get("modality_dropout_prob", 0.05))
        self.use_embedding_consistency_loss = bool(augmentation_config.get("use_consistency_loss", False))
        self.embedding_consistency_loss_weight = float(augmentation_config.get("consistency_loss_weight", 0.05))
        self.embedding_consistency_temperature = float(augmentation_config.get("consistency_temperature", 2.0))
        curriculum_config = config.get("embedding_augmentation_curriculum", {})
        self.use_embedding_augmentation_curriculum = bool(curriculum_config.get("enabled", False) and self.use_embedding_augmentation)
        self.embedding_aug_reference_sample_count = float(curriculum_config.get("size_reference_samples", 1000.0) or 1000.0)
        self.embedding_aug_warmup_epochs = int(curriculum_config.get("warmup_epochs", 1) or 0)
        self.embedding_aug_ramp_epochs = int(curriculum_config.get("ramp_epochs", 2) or 0)
        self.embedding_aug_total_epochs = int(curriculum_config.get("total_epochs", 1) or 1)
        self.embedding_aug_train_samples = int(curriculum_config.get("num_train_samples", 0) or 0)
        self.embedding_aug_current_epoch = 0
        self.embedding_aug_size_factor = 1.0
        self.embedding_aug_epoch_factor = 1.0
        self.embedding_aug_factor = 1.0
        self._refresh_embedding_augmentation_factors(self.embedding_aug_total_epochs, self.embedding_aug_train_samples)
        cross_sample_config = config.get("cross_sample", {})
        self.use_cross_sample = bool(cross_sample_config.get("use_cross_sample", False))
        self.cross_sample_contrastive_layers = {
            int(layer) for layer in cross_sample_config.get("contrastive_layers", [1, 2, 3, 4])
        }
        self.cross_sample_ref_layers = {
            int(layer) for layer in cross_sample_config.get("ref_layers", [3, 4])
        }
        self.cross_sample_top_k = int(cross_sample_config.get("top_k", 4))
        self.cross_sample_temperature = float(cross_sample_config.get("temperature", 0.07))
        self.cross_sample_lambda = float(cross_sample_config.get("lambda_cs", 1.0))
        self.cross_sample_gradient_guard = str(cross_sample_config.get("gradient_guard", "none") or "none").lower()
        self.cross_sample_grad_guard_min_cos = float(cross_sample_config.get("gradient_guard_min_cos", 0.0))
        raw_layer_weights = cross_sample_config.get(
            "layer_weights",
            {"1": 0.001, "2": 0.005, "3": 0.01, "4": 0.01},
        )
        self.cross_sample_layer_weights = {
            int(layer): float(weight) for layer, weight in raw_layer_weights.items()
        }
        self.eval_use_cross_sample_ref = bool(cross_sample_config.get("eval_use_cross_sample_ref", False))
        self.cross_sample_warmup_epochs = int(cross_sample_config.get("warmup_epochs", 0) or 0)
        self.cross_sample_ramp_epochs = int(
            cross_sample_config.get("ramp_epochs", self.cross_sample_warmup_epochs) or 0
        )
        self.cross_sample_decay_start_epoch = int(cross_sample_config.get("decay_start_epoch", -1) or -1)
        self.cross_sample_decay_epochs = int(cross_sample_config.get("decay_epochs", 0) or 0)
        self.cross_sample_active_until_epoch = int(cross_sample_config.get("active_until_epoch", -1) or -1)
        self.cross_sample_current_epoch = 0
        self.cross_sample_class_balanced = bool(cross_sample_config.get("class_balanced", False))
        self.cross_sample_exclude_same_sample = bool(cross_sample_config.get("exclude_same_sample", False))
        self.cross_sample_ref_sim_threshold = float(cross_sample_config.get("ref_sim_threshold", -1.0))
        self.cross_sample_detach_refs = bool(cross_sample_config.get("detach_refs", False))
        self.cross_sample_ref_dropout = float(cross_sample_config.get("ref_dropout", 0.0) or 0.0)
        self.cross_sample_ref_max_scale = float(cross_sample_config.get("ref_max_scale", 1.0))
        self.cross_sample_ref_gate_init = float(cross_sample_config.get("ref_gate_init", 20.0))
        self.cross_sample_memory_bank_size = int(cross_sample_config.get("memory_bank_size", 0) or 0)
        self.cross_sample_use_memory_bank = bool(
            cross_sample_config.get("use_memory_bank", self.cross_sample_memory_bank_size > 0)
        )
        self.cross_sample_positive_top_k = int(cross_sample_config.get("positive_top_k", 0) or 0)
        raw_positive_sim_threshold = cross_sample_config.get("positive_sim_threshold", None)
        self.cross_sample_positive_sim_threshold = (
            None if raw_positive_sim_threshold is None else float(raw_positive_sim_threshold)
        )
        self.cross_sample_mode = str(cross_sample_config.get("mode", "standard")).lower()
        self.cross_sample_positive_filter_quantile = float(
            cross_sample_config.get("positive_filter_quantile", 0.75)
        )
        self.cross_sample_positive_min_filter_sim = float(
            cross_sample_config.get("positive_min_filter_sim", 0.15)
        )
        self.cross_sample_positive_weight_temperature = float(
            cross_sample_config.get("positive_weight_temperature", 0.20)
        )
        self.cross_sample_margin = float(cross_sample_config.get("margin", 1.0))
        raw_modality_weights = cross_sample_config.get(
            "modality_weights", {"text": 0.4, "audio": 0.4, "fused": 0.2}
        )
        self.cross_sample_modality_weights = {
            str(name).lower(): float(weight) for name, weight in raw_modality_weights.items()
        }
        self.cross_sample_drop_ambiguous_negatives = str(
            cross_sample_config.get("drop_ambiguous_negatives", "false")
        ).lower()
        self.cross_sample_negative_filter_sim_threshold = float(
            cross_sample_config.get("negative_filter_sim_threshold", 0.65)
        )
        self.cross_sample_anchor_confidence_gate = bool(
            cross_sample_config.get("anchor_confidence_gate", False)
        )
        self.cross_sample_neighbor_label_prob_threshold = float(
            cross_sample_config.get("neighbor_label_prob_threshold", 0.35)
        )
        self.cross_sample_binary_neighbor_label_prob_threshold = float(
            cross_sample_config.get("binary_neighbor_label_prob_threshold", 0.60)
        )
        self.cross_sample_teacher_true_prob = float(cross_sample_config.get("teacher_true_prob", 0.90))
        self.cross_sample_logit_distill_temperature = float(
            cross_sample_config.get("teacher_temperature", 2.0)
        )
        raw_filter_weights = cross_sample_config.get("filter_weights", {"text": 0.7, "audio": 0.3})
        self.cross_sample_filter_weights = {
            str(name).lower(): float(weight) for name, weight in raw_filter_weights.items()
        }
        self.cross_sample_quality_gate_min_valid_anchor_ratio = float(
            cross_sample_config.get("quality_gate_min_valid_anchor_ratio", 0.85)
        )
        self.cross_sample_quality_gate_min_pos_per_anchor = float(
            cross_sample_config.get("quality_gate_min_pos_per_anchor", 1.25)
        )
        self.cross_sample_quality_gate_target_pos_per_anchor = float(
            cross_sample_config.get("quality_gate_target_pos_per_anchor", 3.0)
        )
        self.cross_sample_quality_gate_min_filter_sim = float(
            cross_sample_config.get("quality_gate_min_filter_sim", 0.25)
        )
        self.cross_sample_quality_gate_target_filter_sim = float(
            cross_sample_config.get("quality_gate_target_filter_sim", 0.70)
        )
        self.cross_sample_quality_gate_min_anchor_gate = float(
            cross_sample_config.get("quality_gate_min_anchor_gate", 0.20)
        )
        self.cross_sample_relation_layer = int(
            cross_sample_config.get("relation_layer", cross_sample_config.get("contrastive_layers", [self.num_query_layers])[-1])
        )
        self.cross_sample_relation_top_k = int(cross_sample_config.get("relation_top_k", 4))
        self.cross_sample_relation_min_sim = float(cross_sample_config.get("relation_min_sim", 0.10))
        self.cross_sample_relation_same_label_bias = float(cross_sample_config.get("relation_same_label_bias", 0.05))
        self.cross_sample_relation_dropout = float(cross_sample_config.get("relation_dropout", 0.0) or 0.0)
        self.cross_sample_relation_gate_init = float(cross_sample_config.get("relation_gate_init", -4.0))
        self.cross_sample_relation_max_scale = float(cross_sample_config.get("relation_max_scale", 0.05))
        self.cross_sample_relation_loss_temperature = float(cross_sample_config.get("relation_loss_temperature", 2.0))
        self.cross_sample_aux_gate_init = float(cross_sample_config.get("aux_gate_init", cross_sample_config.get("relation_gate_init", -3.0)))
        self.cross_sample_aux_max_scale = float(cross_sample_config.get("aux_max_scale", cross_sample_config.get("relation_max_scale", 0.10)))
        self.cross_sample_reweight_min = float(cross_sample_config.get("relation_weight_min", 0.90))
        self.cross_sample_reweight_max = float(cross_sample_config.get("relation_weight_max", 1.15))
        self.cross_sample_reweight_risk_threshold = float(cross_sample_config.get("relation_risk_sim_threshold", 0.35))
        self.cross_sample_reweight_safe_margin = float(cross_sample_config.get("relation_safe_margin", 0.15))
        self.cross_sample_reweight_temperature = float(cross_sample_config.get("relation_reweight_temperature", 0.10))
        self.cross_sample_reweight_normalize = bool(cross_sample_config.get("relation_reweight_normalize", True))
        self.cross_sample_boundary_min_sim = float(cross_sample_config.get("boundary_min_sim", 0.85))
        self.cross_sample_boundary_margin = float(cross_sample_config.get("boundary_margin", 0.35))
        self.cross_sample_boundary_temperature = float(cross_sample_config.get("boundary_temperature", 0.05))
        self.cross_sample_boundary_confidence_low = float(cross_sample_config.get("boundary_confidence_low", 0.50))
        self.cross_sample_boundary_confidence_high = float(cross_sample_config.get("boundary_confidence_high", 0.85))
        self.cross_sample_boundary_max_active_ratio = float(cross_sample_config.get("boundary_max_active_ratio", 1.0))
        self.cross_sample_mixup_alpha = float(cross_sample_config.get("mixup_alpha", 0.2))
        self.cross_sample_mixup_min_lambda = float(cross_sample_config.get("mixup_min_lambda", 0.05))
        self.cross_sample_rank_mixup_margin = float(cross_sample_config.get("rank_mixup_margin", 0.03))
        self.cross_sample_rank_mixup_pair_margin = float(cross_sample_config.get("rank_mixup_pair_margin", 0.05))
        self.cross_sample_rank_mixup_ce_weight = float(cross_sample_config.get("rank_mixup_ce_weight", 0.10))
        self.cross_sample_rank_mixup_target_min_anchor_mass = float(
            cross_sample_config.get("rank_mixup_target_min_anchor_mass", 0.80)
        )
        raw_relation_weights = cross_sample_config.get("relation_weights", {"node": 0.5, "text": 0.3, "audio": 0.2})
        self.cross_sample_relation_weights = {
            str(name).lower(): float(weight) for name, weight in raw_relation_weights.items()
        }

        self.query_adaption = QueryAdaptionModule(
            hidden_dim=self.hidden_dim,
            num_heads=num_heads,
            dropout=self.dropout_rate,
            scale=self.referential_query_scale,
        )
        self.disable_query_adaption = bool(model_config.get("disable_query_adaption", False))
        self.freeze_query_updates = bool(
            model_config.get(
                "freeze_query_updates",
                model_config.get("static_queries", False),
            )
        )
        self.text_to_audio_bridge = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
        )
        self.audio_stimulation_gate = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 4),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.audio_stimulation_norm = nn.LayerNorm(self.hidden_dim)

        self.self_fusion_gate = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 4),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, 4),
        )
        nn.init.zeros_(self.self_fusion_gate[4].weight)
        with torch.no_grad():
            self.self_fusion_gate[4].bias.copy_(torch.tensor([2.0, -1.0, -1.0, -2.0]))

        # Keep baseline parameter initialization identical; extra CS modules are appended last.
        if self.use_cross_sample:
            self.cross_sample_projectors = None
            self.cross_sample_ref_gate_logit = None
            self.cross_sample_relation_adapter = None
            self.cross_sample_batch_relation_aux_adapter = None
            if self.cross_sample_mode == "batch_relation_aux":
                self.cross_sample_batch_relation_aux_adapter = CrossSampleBatchRelationAuxAdapter(
                    self.hidden_dim,
                    num_heads,
                    self.cross_sample_relation_dropout if self.cross_sample_relation_dropout > 0 else self.dropout_rate,
                    gate_init=self.cross_sample_aux_gate_init,
                    max_scale=self.cross_sample_aux_max_scale,
                )
            if self.cross_sample_mode == "query_relation_adapter":
                self.cross_sample_relation_adapter = CrossSampleQueryRelationAdapter(
                    self.hidden_dim,
                    self.dropout_rate,
                    gate_init=self.cross_sample_relation_gate_init,
                    max_scale=self.cross_sample_relation_max_scale,
                )
            if self.cross_sample_mode not in {"reliable_logit_distill", "query_relation_adapter", "batch_relation_aux", "relation_loss_reweight", "boundary_logit_margin", "batch_mixup_aux", "rank_mixup_aux"}:
                self.cross_sample_projectors = nn.ModuleList(
                    [CrossSampleProjector(self.hidden_dim, self.dropout_rate) for _ in range(self.num_query_layers)]
                )
                self.cross_sample_ref_gate_logit = nn.Parameter(
                    torch.tensor(float(self.cross_sample_ref_gate_init))
                )
            if self.cross_sample_use_memory_bank and self.cross_sample_memory_bank_size > 0:
                for layer_number in range(1, self.num_query_layers + 1):
                    self.register_buffer(
                        f"cross_sample_memory_z_l{layer_number}",
                        torch.empty(0, self.hidden_dim),
                        persistent=False,
                    )
                    self.register_buffer(
                        f"cross_sample_memory_labels_l{layer_number}",
                        torch.empty(0, dtype=torch.long),
                        persistent=False,
                    )
                    self.register_buffer(
                        f"cross_sample_memory_sample_ids_l{layer_number}",
                        torch.empty(0, dtype=torch.long),
                        persistent=False,
                    )
                    self.register_buffer(
                        f"cross_sample_memory_filter_text_l{layer_number}",
                        torch.empty(0, self.hidden_dim),
                        persistent=False,
                    )
                    self.register_buffer(
                        f"cross_sample_memory_filter_audio_l{layer_number}",
                        torch.empty(0, self.hidden_dim),
                        persistent=False,
                    )
                    self.register_buffer(
                        f"cross_sample_memory_logits_l{layer_number}",
                        torch.empty(0, self.num_classes),
                        persistent=False,
                    )

    def _safe_unit_mask(self, text_attention_mask: torch.Tensor, audio_attention_mask: torch.Tensor) -> torch.Tensor:
        unit_mask = text_attention_mask.bool().any(dim=-1) | audio_attention_mask.bool().any(dim=-1)
        if not unit_mask.any(dim=1).all():
            unit_mask = unit_mask.clone()
            missing_rows = ~unit_mask.any(dim=1)
            unit_mask[missing_rows, 0] = True
        return unit_mask

    def _main_evidence_token_mask(
        self,
        mask: torch.Tensor,
        num_units: int,
        seq_len: int,
        modality: str,
    ) -> torch.Tensor:
        return mask

    def _finite(self, tensor: torch.Tensor, limit: float = 50.0) -> torch.Tensor:
        return _finite_tensor(tensor, limit=limit)

    @staticmethod
    def _clamped_prob(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _embedding_augmentation_size_factor(self) -> torch.Tensor:
        if not self.use_embedding_augmentation_curriculum:
            return self.class_queries.new_tensor(1.0)
        if self.embedding_aug_train_samples <= 0:
            return self.class_queries.new_tensor(1.0)
        factor = math.sqrt(max(float(self.embedding_aug_train_samples), 0.0) / max(self.embedding_aug_reference_sample_count, 1.0))
        return self.class_queries.new_tensor(min(1.0, max(0.0, factor)))

    def _embedding_augmentation_epoch_factor(self) -> torch.Tensor:
        if not self.use_embedding_augmentation_curriculum:
            return self.class_queries.new_tensor(1.0)
        if self.embedding_aug_warmup_epochs <= 0:
            return self.class_queries.new_tensor(1.0)
        if self.embedding_aug_current_epoch < self.embedding_aug_warmup_epochs:
            return self.class_queries.new_tensor(0.0)
        if self.embedding_aug_ramp_epochs <= 0:
            return self.class_queries.new_tensor(1.0)
        ramp_step = self.embedding_aug_current_epoch - self.embedding_aug_warmup_epochs + 1
        factor = float(ramp_step) / float(max(self.embedding_aug_ramp_epochs, 1))
        return self.class_queries.new_tensor(min(1.0, max(0.0, factor)))

    def _embedding_augmentation_factor(self) -> torch.Tensor:
        return self._embedding_augmentation_size_factor() * self._embedding_augmentation_epoch_factor()

    def _refresh_embedding_augmentation_factors(self, total_epochs: int = None, num_train_samples: int = None) -> None:
        if total_epochs is not None:
            self.embedding_aug_total_epochs = max(int(total_epochs), 1)
        if num_train_samples is not None and num_train_samples > 0:
            self.embedding_aug_train_samples = int(num_train_samples)
        if self.use_embedding_augmentation_curriculum:
            self.embedding_aug_size_factor = float(self._embedding_augmentation_size_factor().detach().cpu())
            self.embedding_aug_epoch_factor = float(self._embedding_augmentation_epoch_factor().detach().cpu())
            self.embedding_aug_factor = float(self.embedding_aug_size_factor * self.embedding_aug_epoch_factor)
            return
        self.embedding_aug_size_factor = 1.0
        self.embedding_aug_epoch_factor = 1.0
        self.embedding_aug_factor = 1.0

    def set_embedding_augmentation_epoch(
        self,
        epoch_index: int,
        total_epochs: int = None,
        num_train_samples: int = None,
    ) -> None:
        self.embedding_aug_current_epoch = max(int(epoch_index), 0)
        self._refresh_embedding_augmentation_factors(total_epochs=total_epochs, num_train_samples=num_train_samples)

    def _apply_token_mask_and_noise(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        mask_prob: float,
        noise_std: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = mask.bool()
        dropped = torch.zeros_like(valid)
        mask_prob = self._clamped_prob(mask_prob)
        if mask_prob > 0.0:
            dropped = (torch.rand(valid.shape, device=tokens.device) < mask_prob) & valid
            tokens = tokens.masked_fill(dropped.unsqueeze(-1), 0.0)
        if noise_std > 0.0:
            noise = torch.randn_like(tokens) * float(noise_std)
            tokens = tokens + noise * valid.to(dtype=tokens.dtype).unsqueeze(-1)
        return self._finite(tokens), dropped

    def _apply_modality_dropout(
        self,
        text_tokens: torch.Tensor,
        audio_tokens: torch.Tensor,
        drop_prob: float = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = text_tokens.size(0)
        drop_text = torch.zeros(batch_size, device=text_tokens.device, dtype=torch.bool)
        drop_audio = torch.zeros(batch_size, device=audio_tokens.device, dtype=torch.bool)
        if not self.use_modality_dropout:
            return text_tokens, audio_tokens, drop_text, drop_audio
        if drop_prob is None:
            drop_prob = self.modality_dropout_prob
        drop_prob = self._clamped_prob(drop_prob)
        if drop_prob <= 0.0:
            return text_tokens, audio_tokens, drop_text, drop_audio

        drop_text = torch.rand(batch_size, device=text_tokens.device) < drop_prob
        drop_audio = torch.rand(batch_size, device=audio_tokens.device) < drop_prob
        both_dropped = drop_text & drop_audio
        if both_dropped.any():
            both_indices = both_dropped.nonzero(as_tuple=False).flatten()
            keep_text = torch.rand(both_indices.numel(), device=text_tokens.device) < 0.5
            drop_text[both_indices[keep_text]] = False
            drop_audio[both_indices[~keep_text]] = False
        if drop_text.any():
            text_tokens = text_tokens.masked_fill(drop_text.view(batch_size, 1, 1), 0.0)
        if drop_audio.any():
            audio_tokens = audio_tokens.masked_fill(drop_audio.view(batch_size, 1, 1), 0.0)
        return self._finite(text_tokens), self._finite(audio_tokens), drop_text, drop_audio

    def _apply_embedding_augmentation(
        self,
        text_tokens: torch.Tensor,
        audio_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        audio_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if not (self.training and self.use_embedding_augmentation):
            return text_tokens, audio_tokens, {}

        factor = float(self._embedding_augmentation_factor().detach().cpu())
        text_mask_prob = self.text_mask_prob * factor
        text_noise_std = self.text_noise_std * factor
        audio_mask_prob = self.audio_mask_prob * factor
        audio_noise_std = self.audio_noise_std * factor
        modality_dropout_prob = self.modality_dropout_prob * factor

        text_tokens, text_dropped = self._apply_token_mask_and_noise(
            text_tokens,
            text_mask,
            text_mask_prob,
            text_noise_std,
        )
        audio_tokens, audio_dropped = self._apply_token_mask_and_noise(
            audio_tokens,
            audio_mask,
            audio_mask_prob,
            audio_noise_std,
        )
        text_tokens, audio_tokens, modality_text_dropped, modality_audio_dropped = self._apply_modality_dropout(
            text_tokens,
            audio_tokens,
            drop_prob=modality_dropout_prob,
        )

        text_valid = text_mask.bool().sum().clamp_min(1)
        audio_valid = audio_mask.bool().sum().clamp_min(1)
        stats = {
            "embedding_aug_active": text_tokens.new_tensor(1.0 if factor > 0.0 else 0.0),
            "embedding_aug_size_factor": text_tokens.new_tensor(float(self.embedding_aug_size_factor)),
            "embedding_aug_epoch_factor": text_tokens.new_tensor(float(self.embedding_aug_epoch_factor)),
            "embedding_aug_factor": text_tokens.new_tensor(float(self.embedding_aug_factor)),
            "embedding_aug_text_mask_prob": text_tokens.new_tensor(self._clamped_prob(text_mask_prob)),
            "embedding_aug_audio_mask_prob": text_tokens.new_tensor(self._clamped_prob(audio_mask_prob)),
            "embedding_aug_text_noise_std": text_tokens.new_tensor(float(text_noise_std)),
            "embedding_aug_audio_noise_std": text_tokens.new_tensor(float(audio_noise_std)),
            "embedding_aug_text_mask_ratio": text_dropped.float().sum() / text_valid,
            "embedding_aug_audio_mask_ratio": audio_dropped.float().sum() / audio_valid,
            "embedding_aug_text_modality_drop_ratio": modality_text_dropped.float().mean(),
            "embedding_aug_audio_modality_drop_ratio": modality_audio_dropped.float().mean(),
        }
        return text_tokens, audio_tokens, stats

    def _compose_unit_evidence(self, text_unit_reps: torch.Tensor, audio_unit_reps: torch.Tensor) -> torch.Tensor:
        unit_pair = torch.cat(
            [
                text_unit_reps,
                audio_unit_reps,
                torch.abs(text_unit_reps - audio_unit_reps),
                text_unit_reps * audio_unit_reps,
            ],
            dim=-1,
        )
        return self._finite(self.unit_evidence(self._finite(unit_pair)))

    def _masked_unit_pool(self, unit_reps: torch.Tensor, unit_mask: torch.Tensor) -> torch.Tensor:
        weights = unit_mask.float().unsqueeze(-1)
        return self._finite((unit_reps * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0))

    def set_cross_sample_epoch(self, epoch_index: int) -> None:
        self.cross_sample_current_epoch = max(int(epoch_index), 0)

    def _cross_sample_loss_factor(self) -> torch.Tensor:
        if (
            self.cross_sample_active_until_epoch >= 0
            and self.cross_sample_current_epoch >= self.cross_sample_active_until_epoch
        ):
            return self.class_queries.new_tensor(0.0)

        if self.cross_sample_warmup_epochs <= 0:
            factor = 1.0
        elif self.cross_sample_current_epoch < self.cross_sample_warmup_epochs:
            factor = 0.0
        elif self.cross_sample_ramp_epochs <= 0:
            factor = 1.0
        else:
            ramp_step = self.cross_sample_current_epoch - self.cross_sample_warmup_epochs + 1
            factor = min(1.0, max(0.0, float(ramp_step) / float(self.cross_sample_ramp_epochs)))

        if self.cross_sample_decay_start_epoch >= 0 and self.cross_sample_current_epoch >= self.cross_sample_decay_start_epoch:
            if self.cross_sample_decay_epochs <= 0:
                factor = 0.0
            else:
                decay_step = self.cross_sample_current_epoch - self.cross_sample_decay_start_epoch + 1
                decay = 1.0 - min(1.0, max(0.0, float(decay_step) / float(self.cross_sample_decay_epochs)))
                factor *= decay
        return self.class_queries.new_tensor(factor)

    def _apply_cross_sample_ref_controls(
        self,
        ref_text: torch.Tensor,
        ref_audio: torch.Tensor,
        ref_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        diagnostics = {}
        if ref_mask is None:
            return ref_text, ref_audio, ref_mask, diagnostics
        if self.training and self.cross_sample_ref_dropout > 0 and ref_mask.numel() > 0:
            keep_mask = torch.rand(ref_mask.shape, device=ref_mask.device) >= self.cross_sample_ref_dropout
            ref_mask = ref_mask & keep_mask
            ref_text = ref_text * ref_mask.unsqueeze(-1).to(dtype=ref_text.dtype)
            ref_audio = ref_audio * ref_mask.unsqueeze(-1).to(dtype=ref_audio.dtype)
        if ref_mask.numel() > 0:
            diagnostics["ref_coverage_after_dropout"] = ref_mask.any(dim=1).float().mean()
        else:
            diagnostics["ref_coverage_after_dropout"] = ref_text.new_tensor(0.0)
        if self.cross_sample_ref_gate_logit is None:
            return ref_text, ref_audio, ref_mask, diagnostics
        ref_scale = self.cross_sample_ref_max_scale * torch.sigmoid(self.cross_sample_ref_gate_logit)
        diagnostics["ref_gate_scale"] = ref_scale.detach()
        return ref_scale * ref_text, ref_scale * ref_audio, ref_mask, diagnostics

    def _build_cross_sample_embedding(
        self,
        layer_index: int,
        queries: torch.Tensor,
        text_unit_reps: torch.Tensor,
        audio_unit_reps: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        class_reps, _ = self._pool_class_queries(queries)
        q_rep = class_reps.mean(dim=1)
        text_rep = self._masked_unit_pool(text_unit_reps, unit_mask)
        audio_rep = self._masked_unit_pool(audio_unit_reps, unit_mask)
        if self.cross_sample_projectors is None:
            z = self._finite(F.normalize(q_rep + text_rep + audio_rep, dim=-1))
        else:
            z = self.cross_sample_projectors[layer_index](self._finite(q_rep), self._finite(text_rep), self._finite(audio_rep))
        return z, text_rep, audio_rep

    @staticmethod
    def _fork_rng_devices(reference: torch.Tensor):
        if reference is not None and reference.is_cuda:
            return [reference.device.index or 0]
        return []

    def _cross_sample_anchor_weights(
        self,
        logits: torch.Tensor,
        logits_dict: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if not self.cross_sample_anchor_confidence_gate:
            return None
        branches = [logits]
        for branch_name in ("text", "audio"):
            branch_logits = logits_dict.get(branch_name)
            if branch_logits is not None:
                branches.append(branch_logits)
        label_index = labels.view(-1, 1)
        confidences = []
        for branch_logits in branches:
            probs = torch.softmax(self._finite(branch_logits.detach(), limit=1e4), dim=-1)
            confidences.append(probs.gather(1, label_index).squeeze(1))
        confidence = torch.stack(confidences, dim=0).mean(dim=0)
        chance = 1.0 / max(float(self.num_classes), 1.0)
        full_confidence = 0.50
        if full_confidence <= chance + 1e-6:
            return confidence.ge(chance).to(dtype=logits.dtype)
        return ((confidence - chance) / (full_confidence - chance)).clamp(0.0, 1.0).to(dtype=logits.dtype)


    def _sample_weighted_classification_loss(self, logits: torch.Tensor, labels: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
        class_weights = self.class_weights.to(logits.device) if self.class_weights.numel() > 0 else None
        loss_logits = logits
        if self.logit_adjustment_tau != 0 and self.class_log_prior.numel() == self.num_classes:
            loss_logits = logits + self.logit_adjustment_tau * self.class_log_prior.to(logits.device)
        per_sample = F.cross_entropy(loss_logits, labels, weight=class_weights, label_smoothing=self.label_smoothing, reduction="none")
        if self.focal_gamma > 0:
            true_probs = F.softmax(loss_logits, dim=-1).gather(1, labels.view(-1, 1)).squeeze(1)
            per_sample = per_sample * (1.0 - true_probs.clamp_min(1e-6)).pow(self.focal_gamma)
        sample_weights = self._finite(sample_weights.to(device=logits.device, dtype=logits.dtype)).clamp_min(0.0)
        if class_weights is not None and class_weights.numel() == self.num_classes:
            denom_weights = class_weights.gather(0, labels).to(device=logits.device, dtype=logits.dtype)
            denominator = (sample_weights * denom_weights).sum().clamp_min(1e-8)
        else:
            denominator = sample_weights.sum().clamp_min(1e-8)
        return (per_sample * sample_weights).sum() / denominator

    def _use_cross_sample_ambiguous_negative_drop(self) -> bool:
        mode = self.cross_sample_drop_ambiguous_negatives
        if mode in {"1", "true", "yes", "on"}:
            return True
        if mode == "auto":
            return self.num_classes > 2
        return False

    def _cross_sample_metric_gate(
        self,
        value: torch.Tensor,
        minimum: float,
        target: float,
    ) -> torch.Tensor:
        value = self._finite(value if torch.is_tensor(value) else self.class_queries.new_tensor(float(value)))
        minimum_value = value.new_tensor(float(minimum))
        target_value = value.new_tensor(float(target))
        span = (target_value - minimum_value).clamp_min(1e-6)
        return ((value - minimum_value) / span).clamp(0.0, 1.0)

    def _cross_sample_quality_gate(
        self,
        diagnostics: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        base = self.class_queries.new_tensor(0.0)
        valid_ratio = diagnostics.get("valid_anchor_ratio", base)
        pos_per_anchor = diagnostics.get("avg_pos_per_anchor", base)
        filter_sim = diagnostics.get("avg_filter_sim", base)
        anchor_gate = diagnostics.get("avg_anchor_gate", self.class_queries.new_tensor(1.0))
        valid_gate = self._cross_sample_metric_gate(
            valid_ratio,
            self.cross_sample_quality_gate_min_valid_anchor_ratio,
            1.0,
        )
        density_gate = self._cross_sample_metric_gate(
            pos_per_anchor,
            self.cross_sample_quality_gate_min_pos_per_anchor,
            self.cross_sample_quality_gate_target_pos_per_anchor,
        )
        sim_gate = self._cross_sample_metric_gate(
            filter_sim,
            self.cross_sample_quality_gate_min_filter_sim,
            self.cross_sample_quality_gate_target_filter_sim,
        )
        confidence_gate = self._cross_sample_metric_gate(
            anchor_gate,
            self.cross_sample_quality_gate_min_anchor_gate,
            1.0,
        )
        return self._finite(valid_gate * density_gate * sim_gate * confidence_gate).clamp(0.0, 1.0)


    def _memory_sample_ids(self, sample_ids, labels: torch.Tensor) -> torch.Tensor:
        batch_size = labels.numel()
        if torch.is_tensor(sample_ids) and sample_ids.numel() == batch_size:
            return sample_ids.detach().view(-1).to(device=labels.device, dtype=torch.long)
        if sample_ids is not None:
            try:
                if len(sample_ids) == batch_size:
                    return torch.tensor(
                        [hash(str(item)) & 0x7FFFFFFFFFFFFFFF for item in sample_ids],
                        device=labels.device,
                        dtype=torch.long,
                    )
            except TypeError:
                pass
        return torch.full((batch_size,), -1, device=labels.device, dtype=torch.long)

    def _get_cross_sample_memory(self, layer_number: int):
        if not (self.cross_sample_use_memory_bank and self.cross_sample_memory_bank_size > 0):
            return None, None, None, None, None
        return (
            getattr(self, f"cross_sample_memory_z_l{layer_number}", None),
            getattr(self, f"cross_sample_memory_labels_l{layer_number}"),
            getattr(self, f"cross_sample_memory_sample_ids_l{layer_number}"),
            getattr(self, f"cross_sample_memory_filter_text_l{layer_number}", None),
            getattr(self, f"cross_sample_memory_filter_audio_l{layer_number}", None),
        )

    def _get_cross_sample_logit_memory(self, layer_number: int):
        if not (self.cross_sample_use_memory_bank and self.cross_sample_memory_bank_size > 0):
            return None, None, None, None, None
        return (
            getattr(self, f"cross_sample_memory_logits_l{layer_number}", None),
            getattr(self, f"cross_sample_memory_labels_l{layer_number}"),
            getattr(self, f"cross_sample_memory_sample_ids_l{layer_number}"),
            getattr(self, f"cross_sample_memory_filter_text_l{layer_number}", None),
            getattr(self, f"cross_sample_memory_filter_audio_l{layer_number}", None),
        )

    @torch.no_grad()
    def _enqueue_cross_sample_memory(
        self,
        layer_number: int,
        z: torch.Tensor = None,
        labels: torch.Tensor = None,
        sample_ids=None,
        filter_text: torch.Tensor = None,
        filter_audio: torch.Tensor = None,
        logits: torch.Tensor = None,
    ) -> None:
        if not (self.cross_sample_use_memory_bank and self.cross_sample_memory_bank_size > 0):
            return
        if labels is None:
            return
        base_tensor = logits if logits is not None else z
        if base_tensor is None:
            return
        base_tensor = base_tensor.detach()
        device = base_tensor.device
        dtype = base_tensor.dtype
        labels = labels.detach().view(-1).to(dtype=torch.long, device=device)
        batch_size = labels.numel()
        sample_id_tensor = self._memory_sample_ids(sample_ids, labels)
        old_z, old_labels, old_sample_ids, old_filter_text, old_filter_audio = self._get_cross_sample_memory(layer_number)
        if old_labels is None:
            return
        old_labels = old_labels.to(device=device, dtype=torch.long)
        old_sample_ids = old_sample_ids.to(device=device, dtype=torch.long)
        if filter_text is None:
            filter_text = base_tensor.new_zeros(batch_size, self.hidden_dim)
        if filter_audio is None:
            filter_audio = base_tensor.new_zeros(batch_size, self.hidden_dim)
        filter_text = filter_text.detach().to(device=device, dtype=dtype).view(batch_size, -1)
        filter_audio = filter_audio.detach().to(device=device, dtype=dtype).view(batch_size, -1)
        if old_filter_text is None:
            old_filter_text = base_tensor.new_empty(0, filter_text.size(-1))
        if old_filter_audio is None:
            old_filter_audio = base_tensor.new_empty(0, filter_audio.size(-1))
        old_filter_text = old_filter_text.to(device=device, dtype=dtype)
        old_filter_audio = old_filter_audio.to(device=device, dtype=dtype)
        new_labels = torch.cat([old_labels, labels], dim=0)[-self.cross_sample_memory_bank_size :]
        new_sample_ids = torch.cat([old_sample_ids, sample_id_tensor], dim=0)[-self.cross_sample_memory_bank_size :]
        new_filter_text = torch.cat([old_filter_text, filter_text], dim=0)[-self.cross_sample_memory_bank_size :]
        new_filter_audio = torch.cat([old_filter_audio, filter_audio], dim=0)[-self.cross_sample_memory_bank_size :]
        if z is not None:
            z = F.normalize(z.detach().to(device=device, dtype=dtype), dim=-1)
            if old_z is None:
                old_z = z.new_empty(0, z.size(-1))
            old_z = old_z.to(device=device, dtype=dtype)
            setattr(self, f"cross_sample_memory_z_l{layer_number}", torch.cat([old_z, z], dim=0)[-self.cross_sample_memory_bank_size :])
        if logits is not None and hasattr(self, f"cross_sample_memory_logits_l{layer_number}"):
            logits = logits.detach().to(device=device, dtype=dtype).view(batch_size, -1)
            old_logits = getattr(self, f"cross_sample_memory_logits_l{layer_number}").to(device=device, dtype=dtype)
            setattr(self, f"cross_sample_memory_logits_l{layer_number}", torch.cat([old_logits, logits], dim=0)[-self.cross_sample_memory_bank_size :])
        setattr(self, f"cross_sample_memory_labels_l{layer_number}", new_labels)
        setattr(self, f"cross_sample_memory_sample_ids_l{layer_number}", new_sample_ids)
        setattr(self, f"cross_sample_memory_filter_text_l{layer_number}", new_filter_text)
        setattr(self, f"cross_sample_memory_filter_audio_l{layer_number}", new_filter_audio)

    def _referential_query_update(
        self,
        queries: torch.Tensor,
        unit_evidence: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.query_adaption(queries, unit_evidence, unit_mask)

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
        return queries, {}

    def _reconstruction_target_mask(
        self,
        mask: torch.Tensor,
        num_units: int,
        seq_len: int,
        exclude_tail_tokens: int = 0,
    ) -> torch.Tensor:
        if exclude_tail_tokens <= 0 or seq_len <= 0:
            return mask.bool()
        grouped = mask.reshape(mask.size(0), num_units, seq_len).bool().clone()
        tail = min(int(exclude_tail_tokens), seq_len)
        grouped[:, :, seq_len - tail :] = False
        flat_mask = grouped.reshape(mask.size(0), num_units * seq_len)
        missing_rows = ~flat_mask.any(dim=1)
        if missing_rows.any():
            flat_mask = flat_mask.clone()
            flat_mask[missing_rows, 0] = True
        return flat_mask

    def _apply_text_guided_audio_stimulation(
        self,
        audio_tokens: torch.Tensor,
        text_unit_reps: torch.Tensor,
        audio_unit_reps: torch.Tensor,
        unit_mask: torch.Tensor,
        num_units: int,
        audio_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.text_guided_audio_scale <= 0:
            return audio_tokens, audio_tokens.new_zeros(audio_tokens.size(0), num_units)
        batch_size = audio_tokens.size(0)
        audio_grouped = audio_tokens.reshape(batch_size, num_units, audio_seq_len, self.hidden_dim)
        gate_input = torch.cat(
            [
                text_unit_reps,
                audio_unit_reps,
                torch.abs(text_unit_reps - audio_unit_reps),
                text_unit_reps * audio_unit_reps,
            ],
            dim=-1,
        )
        gate = self.audio_stimulation_gate(gate_input) * unit_mask.float().unsqueeze(-1)
        text_drive = self.text_to_audio_bridge(text_unit_reps)
        stimulated = audio_grouped + self.text_guided_audio_scale * gate[:, :, None, :] * text_drive[:, :, None, :]
        stimulated = self._finite(self.audio_stimulation_norm(self._finite(stimulated)))
        return self._finite(stimulated.reshape(batch_size, num_units * audio_seq_len, self.hidden_dim)), gate.squeeze(-1)

    def _self_gated_logits(
        self,
        query_logits: torch.Tensor,
        global_logits: torch.Tensor,
        text_logits: torch.Tensor,
        audio_logits: torch.Tensor,
        class_reps: torch.Tensor,
        class_unit_context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_features = torch.cat(
            [
                class_reps,
                class_unit_context,
                torch.abs(class_reps - class_unit_context),
                class_reps * class_unit_context,
            ],
            dim=-1,
        )
        gate_weights = torch.softmax(self.self_fusion_gate(gate_features), dim=-1)
        branch_logits = torch.stack([query_logits, text_logits, audio_logits, global_logits], dim=-1)
        gated_logits = torch.sum(self._finite(branch_logits) * gate_weights, dim=-1)
        return self._finite(gated_logits), gate_weights

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
        batch_size = text_features.size(0)
        num_units = text_features.size(1)
        text_seq_len = text_features.size(2)
        audio_seq_len = audio_features.size(2)
        unit_mask = self._safe_unit_mask(text_attention_mask, audio_attention_mask)

        text_tokens, text_mask = self._flatten_tokens(text_features, text_attention_mask, self.text_projection, 0)
        audio_tokens, audio_mask = self._flatten_tokens(audio_features, audio_attention_mask, self.audio_projection, 1)
        main_text_mask = self._main_evidence_token_mask(text_mask, num_units, text_seq_len, "text")
        main_audio_mask = self._main_evidence_token_mask(audio_mask, num_units, audio_seq_len, "audio")
        layer_text_mask = main_text_mask if getattr(self, "isolate_sp_sh_from_query_attention", False) else text_mask
        layer_audio_mask = main_audio_mask if getattr(self, "isolate_sp_sh_from_query_attention", False) else audio_mask
        embedding_aug_stats = {}
        if augment_embeddings:
            text_tokens, audio_tokens, embedding_aug_stats = self._apply_embedding_augmentation(
                text_tokens,
                audio_tokens,
                text_mask,
                audio_mask,
            )
        initial_text_tokens = text_tokens
        initial_audio_tokens = audio_tokens
        cross_sample_filter_text = None
        cross_sample_filter_audio = None
        if self.use_cross_sample:
            with torch.random.fork_rng(devices=self._fork_rng_devices(text_features), enabled=self.training):
                with torch.no_grad():
                    initial_text_unit_reps, initial_audio_unit_reps, _, _, _ = self._build_unit_context(
                        initial_text_tokens,
                        initial_audio_tokens,
                        main_text_mask,
                        main_audio_mask,
                        num_units,
                        text_seq_len,
                        audio_seq_len,
                    )
                    cross_sample_filter_text = self._masked_unit_pool(initial_text_unit_reps, unit_mask).detach()
                    cross_sample_filter_audio = self._masked_unit_pool(initial_audio_unit_reps, unit_mask).detach()

        queries, q0_diagnostics = self._initialize_queries(
            batch_size,
            text_tokens,
            main_text_mask,
            audio_tokens,
            main_audio_mask,
        )
        initial_queries_for_diag = queries.detach()

        layer_logits = []
        layer_logits_by_number = {}
        referential_context = None
        referential_gate = None
        audio_stimulation_gate = None
        text_unit_reps = None
        audio_unit_reps = None
        unit_weights = None
        class_unit_context = None
        cross_sample_embeddings = {}
        cross_sample_text_embeddings = {}
        cross_sample_audio_embeddings = {}
        cross_sample_ref_masks = {}
        cross_sample_diagnostics = {}
        sp_sh_query_diagnostics = {}
        cross_sample_relation_weights = None
        cross_sample_relation_node = None
        cross_sample_relation_logits = None
        cross_sample_relation_query_logits = None
        allow_cross_sample_ref = (
            self.cross_sample_mode not in {"reliable_logit_distill", "query_relation_adapter", "batch_relation_aux", "relation_loss_reweight", "boundary_logit_margin", "batch_mixup_aux", "rank_mixup_aux"}
            and labels is not None
            and (self.training or self.eval_use_cross_sample_ref)
        )

        for layer_index, layer in enumerate(self.query_layers):
            layer_number = layer_index + 1
            if layer_number not in self.active_query_layer_set:
                continue
            text_unit_reps, audio_unit_reps, unit_weights, class_unit_context, unit_query_context = self._build_unit_context(
                text_tokens,
                audio_tokens,
                main_text_mask,
                main_audio_mask,
                num_units,
                text_seq_len,
                audio_seq_len,
            )
            unit_evidence = self._compose_unit_evidence(text_unit_reps, audio_unit_reps)
            if self.text_guided_audio_mode in {"per_layer", "all"}:
                audio_tokens, audio_stimulation_gate = self._apply_text_guided_audio_stimulation(
                    audio_tokens,
                    text_unit_reps,
                    audio_unit_reps,
                    unit_mask,
                    num_units,
                    audio_seq_len,
                )
            else:
                audio_stimulation_gate = audio_tokens.new_zeros(batch_size, num_units)
            if (not self.disable_query_adaption) and self.referential_update_mode in {"per_layer", "all"}:
                queries, referential_context, referential_gate = self._referential_query_update(
                    queries,
                    unit_evidence,
                unit_mask,
            )
            if not self.disable_query_adaption:
                queries, sp_sh_diag = self._sp_sh_query_update(
                    queries,
                    text_tokens,
                    audio_tokens,
                    text_mask,
                    audio_mask,
                    num_units,
                    text_seq_len,
                    audio_seq_len,
                    f"l{layer_number}",
                )
                if sp_sh_diag:
                    sp_sh_query_diagnostics.update(sp_sh_diag)
            ref_text = None
            ref_audio = None
            ref_mask = None
            if self.use_cross_sample:
                with torch.random.fork_rng(devices=self._fork_rng_devices(text_features), enabled=self.training):
                    z, text_pool, audio_pool = self._build_cross_sample_embedding(
                        layer_index,
                        queries,
                        text_unit_reps,
                        audio_unit_reps,
                        unit_mask,
                    )
                if self.cross_sample_mode != "reliable_logit_distill":
                    cross_sample_embeddings[str(layer_number)] = z
                cross_sample_text_embeddings[str(layer_number)] = self._finite(text_pool)
                cross_sample_audio_embeddings[str(layer_number)] = self._finite(audio_pool)
                if (
                    self.cross_sample_mode == "query_relation_adapter"
                    and self.training
                    and labels is not None
                    and layer_number == self.cross_sample_relation_layer
                    and self.cross_sample_relation_adapter is not None
                ):
                    class_reps_for_relation, _ = self._pool_class_queries(queries)
                    q_rep_for_relation = class_reps_for_relation.mean(dim=1)
                    relation_node = self.cross_sample_relation_adapter.build_node(
                        self._finite(q_rep_for_relation), self._finite(text_pool), self._finite(audio_pool)
                    )
                    memory_node, memory_labels, memory_sample_ids, memory_filter_text, memory_filter_audio = self._get_cross_sample_memory(layer_number)
                    relation_context, relation_weights, relation_diagnostics = build_query_relation_graph(
                        relation_node,
                        self._finite(text_pool),
                        self._finite(audio_pool),
                        labels=labels,
                        memory_node=memory_node,
                        memory_text_key=memory_filter_text,
                        memory_audio_key=memory_filter_audio,
                        memory_labels=memory_labels,
                        sample_ids=sample_ids,
                        memory_sample_ids=memory_sample_ids,
                        exclude_same_sample=self.cross_sample_exclude_same_sample,
                        top_k=self.cross_sample_relation_top_k,
                        min_sim=self.cross_sample_relation_min_sim,
                        same_label_bias=self.cross_sample_relation_same_label_bias,
                        node_weight=self.cross_sample_relation_weights.get("node", 0.5),
                        text_weight=self.cross_sample_relation_weights.get("text", 0.3),
                        audio_weight=self.cross_sample_relation_weights.get("audio", 0.2),
                    )
                    if self.cross_sample_relation_dropout > 0:
                        keep = torch.rand(relation_context.size(0), 1, device=relation_context.device) >= self.cross_sample_relation_dropout
                        relation_context = relation_context * keep.to(dtype=relation_context.dtype)
                        relation_weights = relation_weights * keep.to(dtype=relation_weights.dtype)
                    queries, relation_gate = self.cross_sample_relation_adapter(queries, relation_context)
                    cross_sample_relation_node = relation_node
                    cross_sample_relation_weights = relation_weights
                    for diag_name, diag_value in relation_diagnostics.items():
                        cross_sample_diagnostics[f"relation_{diag_name}"] = diag_value
                    cross_sample_diagnostics["relation_avg_relation_gate"] = relation_gate
                if layer_number in self.cross_sample_ref_layers and allow_cross_sample_ref:
                    ref_text, ref_audio, ref_mask, ref_diagnostics = build_batch_positive_refs(
                        z,
                        labels,
                        text_pool,
                        audio_pool,
                        self.cross_sample_top_k,
                        sample_ids=sample_ids,
                        sim_threshold=self.cross_sample_ref_sim_threshold,
                        detach_refs=self.cross_sample_detach_refs,
                        exclude_same_sample=self.cross_sample_exclude_same_sample,
                    )
                    ref_text, ref_audio, ref_mask, control_diagnostics = self._apply_cross_sample_ref_controls(
                        ref_text,
                        ref_audio,
                        ref_mask,
                    )
                    cross_sample_ref_masks[str(layer_number)] = ref_mask
                    for diag_name, diag_value in {**ref_diagnostics, **control_diagnostics}.items():
                        cross_sample_diagnostics[f"l{layer_number}_{diag_name}"] = diag_value
            pre_layer_queries = queries
            queries, text_tokens, audio_tokens = layer(
                queries,
                text_tokens,
                audio_tokens,
                layer_text_mask,
                layer_audio_mask,
                self._finite(unit_query_context),
                ref_text,
                ref_audio,
                ref_mask,
            )
            queries = self._finite(pre_layer_queries if self.freeze_query_updates else queries)
            text_tokens = self._finite(text_tokens)
            audio_tokens = self._finite(audio_tokens)
            class_reps, _ = self._pool_class_queries(queries)
            layer_combined_logits, _, _, _ = self._score_queries(class_reps, class_unit_context)
            layer_logits.append(layer_combined_logits)
            layer_logits_by_number[layer_number] = layer_combined_logits

        text_unit_reps, audio_unit_reps, unit_weights, class_unit_context, _ = self._build_unit_context(
            text_tokens,
            audio_tokens,
            main_text_mask,
            main_audio_mask,
            num_units,
            text_seq_len,
            audio_seq_len,
        )
        text_unit_reps = self._finite(text_unit_reps)
        audio_unit_reps = self._finite(audio_unit_reps)
        class_unit_context = self._finite(class_unit_context)
        final_unit_evidence = self._compose_unit_evidence(text_unit_reps, audio_unit_reps)
        if (not self.disable_query_adaption) and self.referential_update_mode in {"final", "final_only", "per_layer", "all"}:
            queries, referential_context, referential_gate = self._referential_query_update(
                queries,
                final_unit_evidence,
                unit_mask,
            )
        if not self.disable_query_adaption:
            queries, sp_sh_diag = self._sp_sh_query_update(
                queries,
                text_tokens,
                audio_tokens,
                text_mask,
                audio_mask,
                num_units,
                text_seq_len,
                audio_seq_len,
                "final",
            )
            if sp_sh_diag:
                sp_sh_query_diagnostics.update(sp_sh_diag)

        class_reps, query_weights = self._pool_class_queries(queries)
        query_delta = queries.detach() - initial_queries_for_diag
        query_update_norm_ratio = query_delta.norm(dim=-1).mean() / initial_queries_for_diag.norm(dim=-1).mean().clamp_min(1e-6)
        if self.num_classes > 1:
            query_class_cosine = F.cosine_similarity(class_reps[:, 0, :], class_reps[:, 1, :], dim=-1).mean()
        else:
            query_class_cosine = class_reps.new_tensor(1.0)
        if query_weights.numel() > 0 and query_weights.size(-1) > 1:
            query_pool_entropy = -(
                query_weights.clamp_min(1e-8) * query_weights.clamp_min(1e-8).log()
            ).sum(dim=-1).mean() / query_weights.new_tensor(math.log(query_weights.size(-1)))
        else:
            query_pool_entropy = query_weights.new_tensor(0.0)
        referential_gate_mean = (
            referential_gate.detach().mean()
            if torch.is_tensor(referential_gate)
            else queries.new_tensor(0.0)
        )
        final_query_logits, base_logits, evidence_logits, score_evidence_strength = self._score_queries(
            class_reps,
            class_unit_context,
        )
        layer_logits_tensor = torch.stack(layer_logits, dim=0) if layer_logits else final_query_logits.unsqueeze(0)
        query_logits = (
            (1.0 - self.layer_logit_weight) * final_query_logits
            + self.layer_logit_weight * layer_logits_tensor.mean(dim=0)
        )
        text_logits = self._auxiliary_logits(queries, text_tokens, main_text_mask, self.text_aux_query, self.text_aux_scorer)
        audio_logits = self._auxiliary_logits(queries, audio_tokens, main_audio_mask, self.audio_aux_query, self.audio_aux_scorer)

        if (
            self.use_cross_sample
            and self.cross_sample_mode == "batch_relation_aux"
            and self.training
            and labels is not None
            and self.cross_sample_batch_relation_aux_adapter is not None
        ):
            q_rep_for_relation = class_reps.mean(dim=1)
            text_pool_for_relation = self._masked_unit_pool(text_unit_reps, unit_mask)
            audio_pool_for_relation = self._masked_unit_pool(audio_unit_reps, unit_mask)
            aux_queries, relation_node, _, relation_diagnostics = self.cross_sample_batch_relation_aux_adapter(
                queries,
                self._finite(q_rep_for_relation),
                self._finite(text_pool_for_relation),
                self._finite(audio_pool_for_relation),
            )
            aux_class_reps, _ = self._pool_class_queries(aux_queries)
            cross_sample_relation_query_logits, _, _, _ = self._score_queries(aux_class_reps, class_unit_context)
            cross_sample_relation_node = relation_node
            for diag_name, diag_value in relation_diagnostics.items():
                cross_sample_diagnostics[f"batch_relation_{diag_name}"] = diag_value

        text_global = self._masked_global_mean(text_tokens, main_text_mask)
        audio_global = self._masked_global_mean(audio_tokens, main_audio_mask)
        global_input = torch.cat(
            [text_global, audio_global, torch.abs(text_global - audio_global), text_global * audio_global],
            dim=-1,
        )
        global_rep = self.global_fusion(global_input)
        global_logits = self.global_scorer(global_rep)

        light_logits = None
        light_query_weights = None
        if self.use_parallel_light_path:
            light_logits, _, _, light_query_weights = self._run_parallel_light_path(
                initial_text_tokens,
                initial_audio_tokens,
                main_text_mask,
                main_audio_mask,
                num_units,
                text_seq_len,
                audio_seq_len,
            )

        evidence_gate = None
        meta_delta = None
        if self.final_fusion_mode == "global_gated_residual":
            mixture_weights = torch.softmax(self.score_mixture_logits, dim=0)
            logits, evidence_gate = self._global_gated_residual_logits(
                query_logits,
                global_logits,
                global_rep,
                class_reps,
                class_unit_context,
            )
        elif self.final_fusion_mode == "query_gated_global_residual":
            mixture_weights = torch.softmax(self.score_mixture_logits, dim=0)
            _, evidence_gate = self._global_gated_residual_logits(
                query_logits,
                global_logits,
                global_rep,
                class_reps,
                class_unit_context,
            )
            logits = query_logits + self.qa_gate_scale * evidence_gate * (global_logits - query_logits)
        elif self.final_fusion_mode == "dual_position_late":
            mixture_weights = torch.softmax(self.dual_mixture_logits, dim=0)
            stable_logits = light_logits if light_logits is not None else query_logits
            branch_logits = torch.stack([query_logits, stable_logits, global_logits], dim=1)
            logits = torch.einsum("k,bkc->bc", mixture_weights, branch_logits)
        elif self.final_fusion_mode == "meta_calibrated":
            mixture_weights = torch.softmax(self.score_mixture_logits, dim=0)
            logits, meta_delta = self._meta_calibrated_logits(
                query_logits,
                global_logits,
                base_logits,
                evidence_logits,
                unit_weights,
            )
        elif self.score_mode == "late_mixture":
            mixture_weights = torch.softmax(self.score_mixture_logits, dim=0)
            branch_logits = torch.stack([base_logits, evidence_logits, global_logits], dim=1)
            logits = torch.einsum("k,bkc->bc", mixture_weights, branch_logits)
        else:
            mixture_weights = torch.softmax(self.score_mixture_logits, dim=0)
            logits = query_logits + self.global_logit_scale * global_logits

        self_gated_logits, self_gate_weights = self._self_gated_logits(
            query_logits,
            global_logits,
            text_logits,
            audio_logits,
            class_reps,
            class_unit_context,
        )
        logits = logits + self.self_gated_fusion_scale * (self_gated_logits - query_logits)

        logit_scale = torch.exp(self.logit_scale.clamp(max=math.log(self.max_logit_scale)))
        logits = self._finite(logit_scale * self._finite(logits) + self.class_logit_bias, limit=1e4)
        if cross_sample_relation_query_logits is not None:
            cross_sample_relation_logits = self._finite(
                logit_scale * self._finite(cross_sample_relation_query_logits) + self.class_logit_bias,
                limit=1e4,
            )

        logits_dict = {
            "query": self._finite(logit_scale * self._finite(query_logits) + self.class_logit_bias, limit=1e4),
            "base_query": base_logits,
            "evidence_query": evidence_logits,
            "self_gated": self_gated_logits,
            "self_gate_weights": self_gate_weights,
            "referential_context": referential_context,
            "referential_gate": referential_gate,
            "audio_stimulation_gate": audio_stimulation_gate,
            "score_evidence_strength": score_evidence_strength,
            "score_mixture_weights": mixture_weights,
            "dual_mixture_weights": torch.softmax(self.dual_mixture_logits, dim=0),
            "evidence_gate": evidence_gate,
            "meta_delta": meta_delta,
            "global": global_logits,
            "light": light_logits if light_logits is not None else query_logits,
            "text": text_logits,
            "audio": audio_logits,
            "global_rep": global_rep,
            "class_reps": class_reps,
            "queries": queries,
            "query_weights": query_weights,
            "num_queries_per_class": queries.new_tensor(float(self.queries_per_class)),
            "query_pool_entropy": query_pool_entropy.detach(),
            "query_update_norm_ratio": query_update_norm_ratio.detach(),
            "query_class_cosine": query_class_cosine.detach(),
            "query_adaption_disabled": queries.new_tensor(1.0 if self.disable_query_adaption else 0.0),
            "query_updates_frozen": queries.new_tensor(1.0 if self.freeze_query_updates else 0.0),
            "referential_gate_mean": referential_gate_mean.detach(),
            "light_query_weights": light_query_weights,
            "text_prompt_reps": text_unit_reps,
            "audio_prompt_reps": audio_unit_reps,
            "unit_weights": unit_weights,
            "prompt_weights": unit_weights,
            "layer_logits": layer_logits_tensor,
            "active_query_layer_mask": queries.new_tensor(
                [
                    1.0 if layer_number in self.active_query_layer_set else 0.0
                    for layer_number in range(1, self.num_query_layers + 1)
                ]
            ),
            "active_query_layer_count": queries.new_tensor(float(len(self.active_query_layers))),
        }
        if 2 in layer_logits_by_number:
            logits_dict["aux_logits_q2"] = layer_logits_by_number[2]
        if 3 in layer_logits_by_number:
            logits_dict["aux_logits_q3"] = layer_logits_by_number[3]
        logits_dict["final_queries"] = queries
        if getattr(self, "use_reconstruction_loss", False):
            exclude_tail = int(getattr(self, "reconstruction_exclude_tail_tokens", 0) or 0)
            logits_dict["reconstruction_text_target"] = initial_text_tokens.detach()
            logits_dict["reconstruction_audio_target"] = initial_audio_tokens.detach()
            logits_dict["reconstruction_text_mask"] = self._reconstruction_target_mask(
                text_mask,
                num_units,
                text_seq_len,
                exclude_tail,
            )
            logits_dict["reconstruction_audio_mask"] = self._reconstruction_target_mask(
                audio_mask,
                num_units,
                audio_seq_len,
                exclude_tail,
            )
        if getattr(self, "return_final_modality_tokens", False):
            logits_dict["final_text_tokens"] = self._finite(text_tokens)
            logits_dict["final_audio_tokens"] = self._finite(audio_tokens)
            logits_dict["final_text_mask"] = text_mask
            logits_dict["final_audio_mask"] = audio_mask
        if sp_sh_query_diagnostics:
            logits_dict.update(sp_sh_query_diagnostics)
        if q0_diagnostics:
            logits_dict.update(q0_diagnostics)
        if embedding_aug_stats:
            logits_dict.update(embedding_aug_stats)
        if self.use_cross_sample:
            logits_dict["cross_sample_embeddings"] = cross_sample_embeddings
            logits_dict["cross_sample_text_embeddings"] = cross_sample_text_embeddings
            logits_dict["cross_sample_audio_embeddings"] = cross_sample_audio_embeddings
            logits_dict["cross_sample_ref_masks"] = cross_sample_ref_masks
            logits_dict["cross_sample_diagnostics"] = cross_sample_diagnostics
            logits_dict["cross_sample_sample_ids"] = sample_ids
            logits_dict["cross_sample_filter_text"] = cross_sample_filter_text
            logits_dict["cross_sample_filter_audio"] = cross_sample_filter_audio
            logits_dict["cross_sample_relation_weights"] = cross_sample_relation_weights
            logits_dict["cross_sample_relation_node"] = cross_sample_relation_node
            logits_dict["cross_sample_relation_logits"] = cross_sample_relation_logits
        return logits, logits_dict

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
        if augment_embeddings is not None:
            logits, logits_dict = self._forward_impl(
                text_features,
                audio_features,
                text_attention_mask,
                audio_attention_mask,
                labels=labels,
                sample_ids=sample_ids,
                augment_embeddings=bool(augment_embeddings),
            )
            return self._format_forward_output(logits, logits_dict, return_loss_items)

        if not (self.training and self.use_embedding_augmentation):
            logits, logits_dict = self._forward_impl(
                text_features,
                audio_features,
                text_attention_mask,
                audio_attention_mask,
                labels=labels,
                sample_ids=sample_ids,
                augment_embeddings=False,
            )
            return self._format_forward_output(logits, logits_dict, return_loss_items)

        if not self.embedding_augmentation_clean_augmented_views:
            logits, logits_dict = self._forward_impl(
                text_features,
                audio_features,
                text_attention_mask,
                audio_attention_mask,
                labels=labels,
                sample_ids=sample_ids,
                augment_embeddings=True,
            )
            return self._format_forward_output(logits, logits_dict, return_loss_items)

        clean_logits, clean_dict = self._forward_impl(
            text_features,
            audio_features,
            text_attention_mask,
            audio_attention_mask,
            labels=labels,
            sample_ids=sample_ids,
            augment_embeddings=False,
        )
        augmented_logits, augmented_dict = self._forward_impl(
            text_features,
            audio_features,
            text_attention_mask,
            audio_attention_mask,
            labels=labels,
            sample_ids=sample_ids,
            augment_embeddings=True,
        )
        clean_dict["augmented_logits"] = augmented_logits
        for key, value in augmented_dict.items():
            if key.startswith("embedding_aug_"):
                clean_dict[key] = value
        return self._format_forward_output(clean_logits, clean_dict, return_loss_items)

    def _add_embedding_augmentation_losses(
        self,
        losses: Dict[str, torch.Tensor],
        logits: torch.Tensor,
        logits_dict: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        augmented_logits = logits_dict.get("augmented_logits")
        if not (self.training and augmented_logits is not None):
            return losses

        augmented_logits = self._finite(augmented_logits, limit=1e4)
        augmented_loss = self._classification_loss(augmented_logits, labels)
        losses["embedding_augmented_loss"] = augmented_loss
        effective_factor = float(self.embedding_aug_size_factor * self.embedding_aug_epoch_factor)
        losses["embedding_aug_factor"] = logits.new_tensor(effective_factor)
        losses["embedding_aug_size_factor"] = logits.new_tensor(float(self.embedding_aug_size_factor))
        losses["embedding_aug_epoch_factor"] = logits.new_tensor(float(self.embedding_aug_epoch_factor))
        augmented_loss_weight = float(self.embedding_augmented_loss_weight) * effective_factor
        losses["embedding_augmented_loss_weight_effective"] = logits.new_tensor(augmented_loss_weight)
        if augmented_loss_weight > 0:
            losses["total_loss"] = losses["total_loss"] + augmented_loss_weight * augmented_loss

        consistency_loss = logits.new_tensor(0.0)
        if self.use_embedding_consistency_loss and self.embedding_consistency_loss_weight > 0:
            temperature = max(self.embedding_consistency_temperature, 1e-3)
            clean_probs = F.softmax(logits.detach() / temperature, dim=-1)
            augmented_log_probs = F.log_softmax(augmented_logits / temperature, dim=-1)
            consistency_loss = F.kl_div(
                augmented_log_probs,
                clean_probs,
                reduction="batchmean",
            ) * (temperature ** 2)
            consistency_weight = float(self.embedding_consistency_loss_weight) * effective_factor
            losses["embedding_consistency_loss_weight_effective"] = logits.new_tensor(consistency_weight)
            if consistency_weight > 0:
                losses["total_loss"] = losses["total_loss"] + consistency_weight * consistency_loss
        losses["embedding_consistency_loss"] = consistency_loss
        return losses

    def calculate_losses(
        self,
        logits: torch.Tensor,
        logits_dict: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.use_refqformer_loss:
            losses = self._calculate_refqformer_losses(logits, logits_dict, labels)
            # Keep clean/augmented-view training active for RefQFormerLoss too.
            # The previous early return silently discarded augmented_logits.
            losses = self._add_embedding_augmentation_losses(losses, logits, logits_dict, labels)
            return losses

        losses = super().calculate_losses(logits, logits_dict, labels)
        losses = self._add_embedding_augmentation_losses(losses, logits, logits_dict, labels)
        if self.training and self.use_cross_sample and self.cross_sample_mode in {"batch_mixup_aux", "rank_mixup_aux"}:
            return losses
        if self.training and self.use_cross_sample:
            cross_sample_total = logits.new_tensor(0.0)
            cross_sample_embeddings = logits_dict.get("cross_sample_embeddings", {})
            cross_sample_text_embeddings = logits_dict.get("cross_sample_text_embeddings", {})
            cross_sample_audio_embeddings = logits_dict.get("cross_sample_audio_embeddings", {})
            sample_ids = logits_dict.get("cross_sample_sample_ids")
            cross_sample_filter_text = logits_dict.get("cross_sample_filter_text")
            cross_sample_filter_audio = logits_dict.get("cross_sample_filter_audio")
            anchor_weights = self._cross_sample_anchor_weights(logits, logits_dict, labels)
            for layer_number in sorted(self.cross_sample_contrastive_layers):
                z = cross_sample_embeddings.get(str(layer_number))
                text_z = cross_sample_text_embeddings.get(str(layer_number))
                audio_z = cross_sample_audio_embeddings.get(str(layer_number))
                if z is None and self.cross_sample_mode != "reliable_logit_distill":
                    continue
                if z is None:
                    z = logits
                memory_z, memory_labels, memory_sample_ids, memory_filter_text, memory_filter_audio = self._get_cross_sample_memory(layer_number)
                if self.cross_sample_mode == "batch_relation_aux":
                    if layer_number != self.cross_sample_relation_layer:
                        continue
                    relation_logits = logits_dict.get("cross_sample_relation_logits")
                    layer_loss, diagnostics = relation_auxiliary_ce_loss(
                        relation_logits,
                        labels,
                        class_weights=self.class_weights,
                        label_smoothing=self.label_smoothing,
                        logit_adjustment_tau=self.logit_adjustment_tau,
                        class_log_prior=self.class_log_prior,
                    )
                    for diag_name, diag_value in logits_dict.get("cross_sample_diagnostics", {}).items():
                        if diag_name.startswith("batch_relation_"):
                            diagnostics[diag_name.replace("batch_relation_", "", 1)] = diag_value
                elif self.cross_sample_mode == "relation_loss_reweight":
                    if layer_number != self.cross_sample_relation_layer:
                        continue
                    if text_z is None or audio_z is None:
                        text_z = z
                        audio_z = z
                    weights, diagnostics = relation_aware_loss_weights(
                        z.detach(),
                        text_z.detach(),
                        audio_z.detach(),
                        labels,
                        sample_ids=sample_ids,
                        memory_node=memory_z,
                        memory_text_key=memory_filter_text,
                        memory_audio_key=memory_filter_audio,
                        memory_labels=memory_labels,
                        memory_sample_ids=memory_sample_ids,
                        exclude_same_sample=self.cross_sample_exclude_same_sample,
                        min_weight=self.cross_sample_reweight_min,
                        max_weight=self.cross_sample_reweight_max,
                        risk_sim_threshold=self.cross_sample_reweight_risk_threshold,
                        safe_margin=self.cross_sample_reweight_safe_margin,
                        temperature=self.cross_sample_reweight_temperature,
                        node_weight=self.cross_sample_relation_weights.get("node", 0.5),
                        text_weight=self.cross_sample_relation_weights.get("text", 0.3),
                        audio_weight=self.cross_sample_relation_weights.get("audio", 0.2),
                        normalize_mean=self.cross_sample_reweight_normalize,
                    )
                    weighted_main_loss = self._sample_weighted_classification_loss(logits, labels, weights)
                    layer_loss = weighted_main_loss - losses["main_loss"]
                    diagnostics["weighted_main_loss"] = weighted_main_loss.detach()
                    diagnostics["main_loss_delta"] = layer_loss.detach()
                elif self.cross_sample_mode == "boundary_logit_margin":
                    if layer_number != self.cross_sample_relation_layer:
                        continue
                    if cross_sample_filter_text is None or cross_sample_filter_audio is None:
                        diagnostics = {
                            "valid_anchor_ratio": logits.new_tensor(0.0),
                            "avg_hard_negative_sim": logits.new_tensor(0.0),
                            "avg_confidence_gate": logits.new_tensor(0.0),
                            "avg_boundary_weight": logits.new_tensor(0.0),
                            "avg_margin_violation": logits.new_tensor(0.0),
                            "active_margin_ratio": logits.new_tensor(0.0),
                            "memory_size": logits.new_tensor(0.0),
                        }
                        layer_loss = logits.new_tensor(0.0)
                    else:
                        class_weights = self.class_weights if self.class_weights.numel() > 0 else None
                        layer_loss, diagnostics = boundary_logit_margin_loss(
                            logits,
                            labels,
                            cross_sample_filter_text,
                            cross_sample_filter_audio,
                            sample_ids=sample_ids,
                            min_sim=self.cross_sample_boundary_min_sim,
                            margin=self.cross_sample_boundary_margin,
                            temperature=self.cross_sample_boundary_temperature,
                            confidence_low=self.cross_sample_boundary_confidence_low,
                            confidence_high=self.cross_sample_boundary_confidence_high,
                            max_active_ratio=self.cross_sample_boundary_max_active_ratio,
                            text_weight=self.cross_sample_filter_weights.get("text", 0.7),
                            audio_weight=self.cross_sample_filter_weights.get("audio", 0.3),
                            class_weights=class_weights,
                        )
                elif self.cross_sample_mode == "query_relation_adapter":
                    if layer_number != self.cross_sample_relation_layer:
                        continue
                    relation_weights = logits_dict.get("cross_sample_relation_weights")
                    relation_node = logits_dict.get("cross_sample_relation_node")
                    memory_logits, memory_labels, memory_sample_ids, memory_filter_text, memory_filter_audio = self._get_cross_sample_logit_memory(layer_number)
                    layer_loss, diagnostics = relation_logit_consistency_loss(
                        logits,
                        relation_weights,
                        memory_logits=memory_logits,
                        temperature=self.cross_sample_relation_loss_temperature,
                    )
                    for diag_name, diag_value in logits_dict.get("cross_sample_diagnostics", {}).items():
                        if diag_name.startswith("relation_"):
                            diagnostics[diag_name.replace("relation_", "", 1)] = diag_value
                    if relation_node is not None:
                        z = relation_node
                elif self.cross_sample_mode == "reliable_logit_distill":
                    memory_logits, memory_labels, memory_sample_ids, memory_filter_text, memory_filter_audio = self._get_cross_sample_logit_memory(layer_number)
                    if cross_sample_filter_text is None or cross_sample_filter_audio is None:
                        diagnostics = {
                            "valid_anchor_count": logits.new_tensor(0.0),
                            "valid_anchor_ratio": logits.new_tensor(0.0),
                            "avg_neighbors": logits.new_tensor(0.0),
                            "avg_filter_sim": logits.new_tensor(0.0),
                            "avg_neighbor_label_prob": logits.new_tensor(0.0),
                            "avg_anchor_gate": logits.new_tensor(0.0),
                            "teacher_true_prob": logits.new_tensor(0.0),
                            "memory_size": logits.new_tensor(0.0),
                        }
                        layer_loss = logits.new_tensor(0.0)
                    else:
                        layer_loss, diagnostics = reliable_logit_distillation_loss(
                            logits,
                            labels,
                            cross_sample_filter_text,
                            cross_sample_filter_audio,
                            memory_logits=memory_logits,
                            memory_labels=memory_labels,
                            memory_filter_text=memory_filter_text,
                            memory_filter_audio=memory_filter_audio,
                            sample_ids=sample_ids,
                            memory_sample_ids=memory_sample_ids,
                            exclude_same_sample=self.cross_sample_exclude_same_sample,
                            positive_top_k=self.cross_sample_positive_top_k,
                            positive_filter_quantile=self.cross_sample_positive_filter_quantile,
                            positive_min_filter_sim=self.cross_sample_positive_min_filter_sim,
                            positive_weight_temperature=self.cross_sample_positive_weight_temperature,
                            neighbor_label_prob_threshold=self.cross_sample_neighbor_label_prob_threshold,
                            binary_neighbor_label_prob_threshold=self.cross_sample_binary_neighbor_label_prob_threshold,
                            teacher_temperature=self.cross_sample_logit_distill_temperature,
                            teacher_true_prob=self.cross_sample_teacher_true_prob,
                            text_weight=self.cross_sample_filter_weights.get("text", 0.7),
                            audio_weight=self.cross_sample_filter_weights.get("audio", 0.3),
                        )
                elif self.cross_sample_mode in {"ccr_margin", "ccr_positive"}:
                    layer_loss, diagnostics = ccr_margin_retrieval_loss_with_memory(
                        z,
                        labels,
                        margin=self.cross_sample_margin,
                        memory_anchor=memory_z,
                        memory_labels=memory_labels,
                        sample_ids=sample_ids,
                        memory_sample_ids=memory_sample_ids,
                        exclude_same_sample=self.cross_sample_exclude_same_sample,
                        anchor_weights=anchor_weights,
                        positive_top_k=self.cross_sample_positive_top_k,
                        positive_only=self.cross_sample_mode == "ccr_positive",
                    )
                elif self.cross_sample_mode in {"ccr_modality", "ccr_modality_positive"}:
                    text_z = cross_sample_text_embeddings.get(str(layer_number))
                    audio_z = cross_sample_audio_embeddings.get(str(layer_number))
                    layer_loss = z.new_tensor(0.0)
                    diagnostics = {}
                    parts = []
                    positive_only = self.cross_sample_mode == "ccr_modality_positive"
                    if text_z is not None and self.cross_sample_modality_weights.get("text", 0.0) > 0.0:
                        text_loss, text_diag = ccr_margin_retrieval_loss_with_memory(
                            text_z,
                            labels,
                            margin=self.cross_sample_margin,
                            sample_ids=sample_ids,
                            exclude_same_sample=self.cross_sample_exclude_same_sample,
                            anchor_weights=anchor_weights,
                            positive_top_k=self.cross_sample_positive_top_k,
                            positive_only=positive_only,
                        )
                        parts.append(("text", text_loss, text_diag))
                    if audio_z is not None and self.cross_sample_modality_weights.get("audio", 0.0) > 0.0:
                        audio_loss, audio_diag = ccr_margin_retrieval_loss_with_memory(
                            audio_z,
                            labels,
                            margin=self.cross_sample_margin,
                            sample_ids=sample_ids,
                            exclude_same_sample=self.cross_sample_exclude_same_sample,
                            anchor_weights=anchor_weights,
                            positive_top_k=self.cross_sample_positive_top_k,
                            positive_only=positive_only,
                        )
                        parts.append(("audio", audio_loss, audio_diag))
                    if self.cross_sample_modality_weights.get("fused", 0.0) > 0.0:
                        fused_loss, fused_diag = ccr_margin_retrieval_loss_with_memory(
                            z,
                            labels,
                            margin=self.cross_sample_margin,
                            memory_anchor=memory_z,
                            memory_labels=memory_labels,
                            sample_ids=sample_ids,
                            memory_sample_ids=memory_sample_ids,
                            exclude_same_sample=self.cross_sample_exclude_same_sample,
                            anchor_weights=anchor_weights,
                            positive_top_k=self.cross_sample_positive_top_k,
                            positive_only=positive_only,
                        )
                        parts.append(("fused", fused_loss, fused_diag))
                    if parts:
                        part_weights = self.cross_sample_modality_weights
                        weighted_losses = [part_weights.get(name, 0.0) * loss for name, loss, _ in parts]
                        layer_loss = torch.stack(weighted_losses).sum()
                        for name, _, diag in parts:
                            for diag_name, diag_value in diag.items():
                                diagnostics[f"{name}_{diag_name}"] = diag_value
                        diagnostics["valid_anchor_count"] = torch.stack([diag["valid_anchor_count"] for _, _, diag in parts]).mean()
                        diagnostics["valid_anchor_ratio"] = torch.stack([diag["valid_anchor_ratio"] for _, _, diag in parts]).mean()
                        diagnostics["avg_pos_sim"] = torch.stack([diag["avg_pos_sim"] for _, _, diag in parts]).mean()
                        diagnostics["avg_neg_sim"] = torch.stack([diag["avg_neg_sim"] for _, _, diag in parts]).mean()
                        diagnostics["margin_active_ratio"] = torch.stack([diag["margin_active_ratio"] for _, _, diag in parts]).mean()
                        diagnostics["avg_anchor_gate"] = torch.stack([diag.get("avg_anchor_gate", z.new_tensor(1.0)) for _, _, diag in parts]).mean()
                        diagnostics["memory_size"] = torch.stack([diag.get("memory_size", z.new_tensor(0.0)) for _, _, diag in parts]).mean()
                elif self.cross_sample_mode in {"reliable_weighted", "quality_gated_reliable_weighted"}:
                    if cross_sample_filter_text is None or cross_sample_filter_audio is None:
                        diagnostics = {
                            "valid_anchor_count": z.new_tensor(0.0),
                            "valid_anchor_ratio": z.new_tensor(0.0),
                            "avg_pos_per_anchor": z.new_tensor(0.0),
                            "avg_filter_sim": z.new_tensor(0.0),
                            "avg_anchor_gate": z.new_tensor(0.0),
                            "quality_gate": z.new_tensor(0.0),
                            "dropped_negative_ratio": z.new_tensor(0.0),
                            "memory_size": z.new_tensor(0.0),
                        }
                        layer_loss = z.new_tensor(0.0)
                    else:
                        layer_loss, diagnostics = weighted_supervised_contrastive_loss_with_memory(
                            z,
                            labels,
                            self.cross_sample_temperature,
                            cross_sample_filter_text,
                            cross_sample_filter_audio,
                            memory_z=memory_z,
                            memory_labels=memory_labels,
                            memory_filter_text=memory_filter_text,
                            memory_filter_audio=memory_filter_audio,
                            sample_ids=sample_ids,
                            memory_sample_ids=memory_sample_ids,
                            class_balanced=self.cross_sample_class_balanced,
                            exclude_same_sample=self.cross_sample_exclude_same_sample,
                            positive_top_k=self.cross_sample_positive_top_k,
                            positive_filter_quantile=self.cross_sample_positive_filter_quantile,
                            positive_min_filter_sim=self.cross_sample_positive_min_filter_sim,
                            positive_weight_temperature=self.cross_sample_positive_weight_temperature,
                            drop_ambiguous_negatives=self._use_cross_sample_ambiguous_negative_drop(),
                            negative_filter_sim_threshold=self.cross_sample_negative_filter_sim_threshold,
                            anchor_weights=anchor_weights,
                        )
                        if self.cross_sample_mode == "quality_gated_reliable_weighted":
                            quality_gate = self._cross_sample_quality_gate(diagnostics)
                            diagnostics["quality_gate"] = quality_gate.detach()
                            layer_loss = quality_gate * layer_loss
                elif memory_z is not None and memory_z.numel() > 0:
                    layer_loss, diagnostics = supervised_contrastive_loss_with_memory(
                        z,
                        labels,
                        self.cross_sample_temperature,
                        memory_z=memory_z,
                        memory_labels=memory_labels,
                        sample_ids=sample_ids,
                        memory_sample_ids=memory_sample_ids,
                        class_balanced=self.cross_sample_class_balanced,
                        exclude_same_sample=self.cross_sample_exclude_same_sample,
                        positive_top_k=self.cross_sample_positive_top_k,
                        positive_sim_threshold=self.cross_sample_positive_sim_threshold,
                    )
                else:
                    layer_loss, diagnostics = supervised_contrastive_loss(
                        z,
                        labels,
                        self.cross_sample_temperature,
                        sample_ids=sample_ids,
                        class_balanced=self.cross_sample_class_balanced,
                        exclude_same_sample=self.cross_sample_exclude_same_sample,
                        positive_top_k=self.cross_sample_positive_top_k,
                        positive_sim_threshold=self.cross_sample_positive_sim_threshold,
                    )
                    diagnostics["memory_size"] = z.new_tensor(0.0)
                self._enqueue_cross_sample_memory(
                    layer_number,
                    None if self.cross_sample_mode == "reliable_logit_distill" else z,
                    labels,
                    sample_ids=sample_ids,
                    filter_text=cross_sample_filter_text,
                    filter_audio=cross_sample_filter_audio,
                    logits=logits if self.cross_sample_mode in {"reliable_logit_distill", "query_relation_adapter"} else None,
                )
                layer_weight = self.cross_sample_layer_weights.get(layer_number, 0.0)
                cross_sample_total = cross_sample_total + layer_weight * layer_loss
                losses[f"cross_sample_loss_l{layer_number}"] = layer_loss
                for diag_name, diag_value in diagnostics.items():
                    losses[f"cross_sample_{diag_name}_l{layer_number}"] = diag_value
            for diag_name, diag_value in logits_dict.get("cross_sample_diagnostics", {}).items():
                losses[f"cross_sample_{diag_name}"] = diag_value
            loss_factor = self._cross_sample_loss_factor()
            losses["cross_sample_loss"] = cross_sample_total
            losses["cross_sample_loss_factor"] = loss_factor
            losses["cross_sample_loss_applied"] = loss_factor * cross_sample_total
            losses["total_loss"] = losses["total_loss"] + self.cross_sample_lambda * loss_factor * cross_sample_total
        teacher_loss = logits.new_tensor(0.0)
        if self.teacher_leading_loss_weight > 0:
            temperature = max(self.teacher_temperature, 1e-3)
            teacher_probs = F.softmax(logits_dict["text"].detach() / temperature, dim=-1)
            audio_log_probs = F.log_softmax(logits_dict["audio"] / temperature, dim=-1)
            teacher_loss = F.kl_div(audio_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
            losses["total_loss"] = losses["total_loss"] + self.teacher_leading_loss_weight * teacher_loss
        losses["teacher_leading_loss"] = teacher_loss
        return losses


class RefFormerAudioTextEmotionModel(UnifiedReferentialEvidenceQueryModel):
    """Named RefFormer-style QAM model for audio + text emotion recognition."""

    pass
