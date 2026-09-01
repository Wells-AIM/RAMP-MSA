"""Unified evidence-unit text+audio QA model.

The same module is intended to run EATD, MELD, and DAIC. Dataset-specific
preprocessing only decides what one evidence unit means:
- EATD: one elicitation prompt;
- MELD: one conversational context slot;
- DAIC: one interview window.
"""
import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.refqformer_loss import RefQFormerLoss, build_refqformer_loss_outputs


class ResidualLinearClassifier(nn.Module):
    """Single linear classifier with a gated MLP correction branch."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float, gate_init: float = -1.0):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.residual = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + torch.sigmoid(self.gate_logit) * self.residual(x)


class CrossQueryLayer(nn.Module):
    """One iterative layer where class queries read evidence units and raw tokens."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        update_tokens: bool = True,
        use_evidence_context: bool = False,
        use_residual_bridge: bool = False,
        bridge_scale: float = 0.25,
        bridge_init: float = -2.5,
    ):
        super().__init__()
        self.update_tokens = update_tokens
        self.use_evidence_context = use_evidence_context
        self.use_residual_bridge = use_residual_bridge
        self.bridge_scale = float(bridge_scale)
        self.bridge_logit = nn.Parameter(torch.tensor(float(bridge_init)))
        self.query_self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.query_to_text = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.query_to_audio = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.query_norm1 = nn.LayerNorm(hidden_dim)
        self.query_norm2 = nn.LayerNorm(hidden_dim)
        self.query_norm3 = nn.LayerNorm(hidden_dim)
        self.query_norm4 = nn.LayerNorm(hidden_dim)
        self.bridge_norm = nn.LayerNorm(hidden_dim)
        self.query_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),
        )
        self.evidence_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.cross_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5),
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.text_evidence_to_audio = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.audio_evidence_to_text = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.bridge_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5),
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.Sigmoid(),
        )
        self.bridge_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 7),
            nn.Linear(hidden_dim * 7, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        nn.init.zeros_(self.bridge_fusion[4].weight)
        nn.init.zeros_(self.bridge_fusion[4].bias)
        self.query_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        if update_tokens:
            self.text_to_query = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            self.audio_to_query = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            self.text_token_norm = nn.LayerNorm(hidden_dim)
            self.audio_token_norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _finite(tensor: torch.Tensor, limit: float = 50.0) -> torch.Tensor:
        if tensor is None:
            return tensor
        return torch.nan_to_num(tensor, nan=0.0, posinf=limit, neginf=-limit).clamp(min=-limit, max=limit)

    def forward(
        self,
        queries: torch.Tensor,
        text_tokens: torch.Tensor,
        audio_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        evidence_context: torch.Tensor = None,
        ref_text: torch.Tensor = None,
        ref_audio: torch.Tensor = None,
        ref_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_kv = text_tokens
        audio_kv = audio_tokens
        text_kv_mask = text_mask
        audio_kv_mask = audio_mask
        if ref_text is not None and ref_audio is not None:
            if ref_mask is None:
                ref_mask = torch.ones(ref_text.shape[:2], device=ref_text.device, dtype=torch.bool)
            ref_mask = ref_mask.bool()
            ref_text = self._finite(ref_text)
            ref_audio = self._finite(ref_audio)
            text_kv = torch.cat([text_tokens, ref_text], dim=1)
            audio_kv = torch.cat([audio_tokens, ref_audio], dim=1)
            text_kv_mask = torch.cat([text_mask.bool(), ref_mask], dim=1)
            audio_kv_mask = torch.cat([audio_mask.bool(), ref_mask], dim=1)

        queries = self._finite(queries)
        text_tokens = self._finite(text_tokens)
        audio_tokens = self._finite(audio_tokens)
        text_kv = self._finite(text_kv)
        audio_kv = self._finite(audio_kv)

        q_self, _ = self.query_self_attn(queries, queries, queries, need_weights=False)
        q_self = self._finite(q_self)
        queries = self._finite(self.query_norm1(queries + self.dropout(q_self)))

        if self.use_evidence_context and evidence_context is not None:
            evidence_context = self._finite(evidence_context)
            evidence_gate = self.evidence_gate(torch.cat([queries, evidence_context], dim=-1))
            queries = self._finite(self.query_norm2(queries + self.dropout(evidence_gate * evidence_context)))

        text_context, _ = self.query_to_text(
            queries,
            text_kv,
            text_kv,
            key_padding_mask=~text_kv_mask.bool(),
            need_weights=False,
        )
        audio_context, _ = self.query_to_audio(
            queries,
            audio_kv,
            audio_kv,
            key_padding_mask=~audio_kv_mask.bool(),
            need_weights=False,
        )
        text_context = self._finite(text_context)
        audio_context = self._finite(audio_context)
        if self.use_evidence_context:
            interaction = torch.cat(
                [
                    queries,
                    text_context,
                    audio_context,
                    torch.abs(text_context - audio_context),
                    text_context * audio_context,
                ],
                dim=-1,
            )
            queries = self._finite(self.query_norm3(queries + self.cross_fusion(interaction)))

            if self.use_residual_bridge:
                text_followup, _ = self.text_evidence_to_audio(
                    text_context,
                    audio_kv,
                    audio_kv,
                    key_padding_mask=~audio_kv_mask.bool(),
                    need_weights=False,
                )
                audio_followup, _ = self.audio_evidence_to_text(
                    audio_context,
                    text_kv,
                    text_kv,
                    key_padding_mask=~text_kv_mask.bool(),
                    need_weights=False,
                )
                text_followup = self._finite(text_followup)
                audio_followup = self._finite(audio_followup)
                bridge_gate = self.bridge_gate(interaction)
                bridge_input = torch.cat(
                    [
                        queries,
                        text_context,
                        audio_context,
                        text_followup,
                        audio_followup,
                        torch.abs(text_followup - audio_followup),
                        text_followup * audio_followup,
                    ],
                    dim=-1,
                )
                bridge_strength = self.bridge_scale * torch.sigmoid(self.bridge_logit)
                bridge_update = self._finite(self.bridge_fusion(bridge_input))
                queries = self._finite(self.bridge_norm(queries + bridge_strength * bridge_gate * bridge_update))

            queries = self._finite(self.query_norm4(queries + self.query_ffn(queries)))
        else:
            gates = self.query_gate(torch.cat([queries, text_context, audio_context], dim=-1))
            cross_update = self._finite(gates[..., :1] * text_context + gates[..., 1:] * audio_context)
            queries = self._finite(self.query_norm2(queries + self.dropout(cross_update)))
            queries = self._finite(self.query_norm3(queries + self.query_ffn(queries)))

        if self.update_tokens:
            text_update, _ = self.text_to_query(text_tokens, queries, queries, need_weights=False)
            audio_update, _ = self.audio_to_query(audio_tokens, queries, queries, need_weights=False)
            text_update = self._finite(text_update)
            audio_update = self._finite(audio_update)
            text_tokens = self._finite(self.text_token_norm(text_tokens + self.dropout(text_update)))
            audio_tokens = self._finite(self.audio_token_norm(audio_tokens + self.dropout(audio_update)))

        if not self.use_evidence_context:
            queries = self._finite(self.query_norm4(queries))
        return self._finite(queries), self._finite(text_tokens), self._finite(audio_tokens)


class UnifiedEvidenceTextAudioQueryModel(nn.Module):
    """Use class-conditioned queries to read text/audio evidence units.

    The network structure is independent of the dataset. It accepts a tensor of
    evidence units and performs the same operations for every benchmark:
    projection -> class query/text/audio cross attention -> unit attention ->
    class scoring.
    """

    def __init__(self, config: Dict[str, object]):
        super().__init__()
        model_config = config["model"]
        training_config = config.get("training", {})
        loss_config = config.get("loss", {})

        self.num_classes = int(model_config.get("output", {}).get("num_classes", 2))
        self.input_dim = int(model_config.get("input_dim", 768))
        self.text_input_dim = int(model_config.get("text_input_dim", self.input_dim))
        self.audio_input_dim = int(model_config.get("audio_input_dim", self.input_dim))
        self.hidden_dim = int(model_config.get("hidden_dim", 384))
        data_config = config.get("data", {})
        self.max_evidence_units = int(
            model_config.get(
                "max_evidence_units",
                data_config.get("max_evidence_units", data_config.get("num_prompts", 8)),
            )
        )
        self.queries_per_class = int(
            model_config.get("num_queries_per_class", model_config.get("queries_per_class", 4))
        )
        if self.queries_per_class <= 0:
            raise ValueError("model.num_queries_per_class / queries_per_class must be positive")
        self.num_queries_per_class = self.queries_per_class
        self.query_pooling_mode = str(model_config.get("query_pooling_mode", "attention")).lower()
        if self.query_pooling_mode not in {"attention", "attn", "learned", "mean", "avg"}:
            raise ValueError("model.query_pooling_mode must be one of: attention, mean")
        self.num_query_layers = int(model_config.get("num_query_layers", 2))
        self.dropout_rate = float(model_config.get("dropout", 0.3))
        self.token_dropout_rate = float(model_config.get("token_dropout", 0.05))
        self.aux_loss_weight = float(training_config.get("aux_loss_weight", 0.15))
        self.alignment_loss_weight = float(training_config.get("alignment_loss_weight", 0.02))
        self.diversity_loss_weight = float(training_config.get("diversity_loss_weight", 0.01))
        self.layer_loss_weight = float(training_config.get("layer_loss_weight", 0.0))
        self.global_loss_weight = float(training_config.get("global_loss_weight", 0.0))
        self.light_loss_weight = float(training_config.get("light_loss_weight", 0.0))
        self.rank_loss_weight = float(training_config.get("rank_loss_weight", 0.0))
        self.rank_margin = float(training_config.get("rank_margin", 0.2))
        self.query_contrastive_loss_weight = float(training_config.get("query_contrastive_loss_weight", 0.0))
        self.query_contrastive_temperature = float(training_config.get("query_contrastive_temperature", 0.10))
        self.label_smoothing = float(training_config.get("label_smoothing", 0.0))
        self.focal_gamma = float(training_config.get("focal_gamma", 0.0))
        self.use_refqformer_loss = str(loss_config.get("name", "")).lower() == "refqformer_loss"
        self.refqformer_loss = RefQFormerLoss(config) if self.use_refqformer_loss else None
        self.logit_adjustment_tau = float(training_config.get("logit_adjustment_tau", 0.0))
        self.global_logit_scale = float(model_config.get("global_logit_scale", 1.0))
        self.final_fusion_mode = str(model_config.get("final_fusion_mode", "add"))
        self.use_parallel_light_path = bool(model_config.get("use_parallel_light_path", False))
        self.parallel_light_layers = int(model_config.get("parallel_light_layers", max(1, min(2, self.num_query_layers))))
        self.parallel_light_update_tokens = bool(model_config.get("parallel_light_update_tokens", False))
        dual_mixture_init = model_config.get("dual_mixture_init", [3.0, -4.0, -4.0])
        if not isinstance(dual_mixture_init, list) or len(dual_mixture_init) != 3:
            dual_mixture_init = [3.0, -4.0, -4.0]
        self.dual_mixture_logits = nn.Parameter(torch.tensor(dual_mixture_init, dtype=torch.float))
        self.qa_gate_scale = float(model_config.get("qa_gate_scale", 1.0))
        self.qa_gate_init = float(model_config.get("qa_gate_init", -1.5))
        self.meta_calibrator_scale = float(model_config.get("meta_calibrator_scale", 0.5))
        self.layer_logit_weight = float(model_config.get("layer_logit_weight", 0.0))
        self.score_mode = str(model_config.get("score_mode", "evidence"))
        self.classifier_head_type = str(model_config.get("classifier_head_type", "mlp") or "mlp").lower()
        if self.classifier_head_type not in {"mlp", "linear", "ln_linear", "residual_mlp"}:
            self.classifier_head_type = "mlp"
        self.classifier_hidden_ratio = float(model_config.get("classifier_hidden_ratio", 0.5) or 0.5)
        self.classifier_dropout_scale = float(model_config.get("classifier_dropout_scale", 1.0) or 1.0)
        self.classifier_residual_gate_init = float(model_config.get("classifier_residual_gate_init", -1.0) or -1.0)
        self.unit_context_mode = str(model_config.get("unit_context_mode", "pair"))
        self.use_embedding_conditioned_q0 = bool(model_config.get("use_embedding_conditioned_q0", False))
        self.q0_condition_mode = str(model_config.get("q0_condition_mode", "shared")).lower()
        self.q0_condition_scale = float(model_config.get("q0_condition_scale", 0.25))
        self.q0_condition_gate_init = float(model_config.get("q0_condition_gate_init", -1.5))
        self.q0_condition_dropout = float(model_config.get("q0_condition_dropout", self.dropout_rate))
        self.q0_condition_zero_init = bool(model_config.get("q0_condition_zero_init", False))
        self.q0_condition_prototype_path = model_config.get("q0_condition_prototype_path")
        self.use_q0_class_prototypes = (
            self.use_embedding_conditioned_q0
            and self.q0_condition_mode
            in {"class_prototype", "prototype", "prototype_class", "static_class_prototype", "class_prototype_static"}
        )
        self.use_q0_static_class_prototypes = (
            self.use_embedding_conditioned_q0
            and self.q0_condition_mode in {"static_class_prototype", "class_prototype_static"}
        )
        self.score_evidence_scale = float(model_config.get("score_evidence_scale", 0.25))
        self.score_evidence_logit = nn.Parameter(
            torch.tensor(float(model_config.get("score_evidence_init", -2.5)))
        )
        mixture_init = model_config.get("score_mixture_init", [0.0, 0.0, 0.0])
        if not isinstance(mixture_init, list) or len(mixture_init) != 3:
            mixture_init = [0.0, 0.0, 0.0]
        self.score_mixture_logits = nn.Parameter(torch.tensor(mixture_init, dtype=torch.float))
        default_trainable_prefixes = [
            "dual_mixture_logits",
            "global_evidence_gate",
            "global_fusion",
            "global_scorer",
            "light_",
            "meta_calibrator",
            "q0_condition_",
            "class_logit_bias",
            "logit_scale",
        ]
        trainable_prefixes = model_config.get("freeze_trainable_prefixes", default_trainable_prefixes)
        if not isinstance(trainable_prefixes, list) or not trainable_prefixes:
            trainable_prefixes = default_trainable_prefixes
        self.freeze_trainable_prefixes = tuple(str(prefix) for prefix in trainable_prefixes)

        num_heads = int(model_config.get("num_heads", 6))
        update_tokens = bool(model_config.get("update_tokens", True))
        use_evidence_context = bool(model_config.get("use_evidence_context", False))
        use_residual_bridge = bool(model_config.get("use_residual_bridge", False))
        bridge_scale = float(model_config.get("bridge_scale", 0.25))
        bridge_init = float(model_config.get("bridge_init", -2.5))

        self.text_projection = self._make_projection(self.text_input_dim)
        self.audio_projection = self._make_projection(self.audio_input_dim)
        self.unit_embedding = nn.Embedding(self.max_evidence_units, self.hidden_dim)
        self.modality_embedding = nn.Parameter(torch.randn(2, 1, self.hidden_dim) * 0.02)
        self.class_embedding = nn.Parameter(torch.randn(self.num_classes, self.hidden_dim) * 0.02)
        self.class_queries = nn.Parameter(
            torch.randn(self.num_classes, self.queries_per_class, self.hidden_dim) * 0.02
        )
        if self.use_embedding_conditioned_q0:
            q0_rng_state = torch.random.get_rng_state()
            torch.manual_seed(int(model_config.get("q0_condition_seed", 7919)))
            try:
                self.q0_condition_text_scorer = nn.Sequential(
                    nn.LayerNorm(self.hidden_dim),
                    nn.Linear(self.hidden_dim, 1),
                )
                self.q0_condition_fusion = nn.Sequential(
                    nn.LayerNorm(self.hidden_dim * 2),
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.q0_condition_dropout),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
                self.q0_condition_norm = nn.LayerNorm(self.hidden_dim)
                self.q0_condition_gate = nn.Parameter(torch.tensor(self.q0_condition_gate_init, dtype=torch.float))
                if self.use_q0_class_prototypes:
                    self.q0_condition_prototype_fusion = nn.Sequential(
                        nn.LayerNorm(self.hidden_dim * 2),
                        nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                        nn.GELU(),
                        nn.Dropout(self.q0_condition_dropout),
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                    )
                    self.q0_condition_class_fusion = nn.Sequential(
                        nn.LayerNorm(self.hidden_dim * 4),
                        nn.Linear(self.hidden_dim * 4, self.hidden_dim),
                        nn.GELU(),
                        nn.Dropout(self.q0_condition_dropout),
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                    )
                    self.q0_condition_class_norm = nn.LayerNorm(self.hidden_dim)
                if self.q0_condition_zero_init:
                    self._zero_last_linear(self.q0_condition_fusion)
                    if self.use_q0_class_prototypes:
                        self._zero_last_linear(self.q0_condition_prototype_fusion)
                        self._zero_last_linear(self.q0_condition_class_fusion)
            finally:
                torch.random.set_rng_state(q0_rng_state)

        if self.use_parallel_light_path:
            self.light_class_queries = nn.Parameter(
                torch.randn(self.num_classes, self.queries_per_class, self.hidden_dim) * 0.02
            )
            self.light_query_layers = nn.ModuleList(
                [
                    CrossQueryLayer(
                        self.hidden_dim,
                        num_heads,
                        self.dropout_rate,
                        self.parallel_light_update_tokens,
                        False,
                        False,
                    )
                    for _ in range(self.parallel_light_layers)
                ]
            )
            self.light_query_pool = nn.Linear(self.hidden_dim, 1)
            self.light_class_scorer = self._make_scalar_classifier()
            self.light_class_unit_fusion = nn.Sequential(
                nn.LayerNorm(self.hidden_dim * 2),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
            )

        self.query_layers = nn.ModuleList(
            [
                CrossQueryLayer(
                    self.hidden_dim,
                    num_heads,
                    self.dropout_rate,
                    update_tokens,
                    use_evidence_context,
                    use_residual_bridge,
                    bridge_scale,
                    bridge_init,
                )
                for _ in range(self.num_query_layers)
            ]
        )
        self.query_pool = nn.Linear(self.hidden_dim, 1)
        self.class_scorer = self._make_scalar_classifier()
        self.class_logit_bias = nn.Parameter(torch.zeros(self.num_classes))
        prior_bias = model_config.get("class_logit_bias_init")
        if isinstance(prior_bias, list) and len(prior_bias) == self.num_classes:
            with torch.no_grad():
                self.class_logit_bias.copy_(torch.tensor(prior_bias, dtype=torch.float))
        self.logit_scale = nn.Parameter(torch.tensor(float(model_config.get("logit_scale_init", 0.0))))
        self.max_logit_scale = float(model_config.get("max_logit_scale", 4.0))

        self.text_aux_query = nn.MultiheadAttention(self.hidden_dim, num_heads, dropout=self.dropout_rate, batch_first=True)
        self.audio_aux_query = nn.MultiheadAttention(self.hidden_dim, num_heads, dropout=self.dropout_rate, batch_first=True)
        self.text_aux_scorer = self._make_aux_scorer()
        self.audio_aux_scorer = self._make_aux_scorer()
        self.unit_evidence = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 4),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.class_unit_scorer = self._make_vector_classifier(hidden_layers=False)
        self.class_unit_fusion = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
        )
        self.global_fusion = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 4),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
        )
        self.global_scorer = self._make_vector_classifier(hidden_layers=True)
        self.global_evidence_gate = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 5),
            nn.Linear(self.hidden_dim * 5, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.global_evidence_gate[4].weight)
        nn.init.zeros_(self.global_evidence_gate[4].bias)
        self.global_evidence_gate_bias = nn.Parameter(torch.tensor(self.qa_gate_init, dtype=torch.float))
        meta_feature_dim = self.num_classes * (self.max_evidence_units + 4)
        meta_hidden_dim = int(model_config.get("meta_calibrator_hidden_dim", min(256, max(32, meta_feature_dim * 2))))
        self.meta_calibrator = nn.Sequential(
            nn.LayerNorm(meta_feature_dim),
            nn.Linear(meta_feature_dim, meta_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(meta_hidden_dim, self.num_classes),
        )
        nn.init.zeros_(self.meta_calibrator[4].weight)
        nn.init.zeros_(self.meta_calibrator[4].bias)
        self.register_buffer("class_weights", torch.empty(0), persistent=False)
        self.register_buffer("class_log_prior", torch.empty(0), persistent=False)
        text_prototypes, audio_prototypes = self._load_q0_class_prototypes()
        self.register_buffer("q0_condition_text_prototypes", text_prototypes, persistent=False)
        self.register_buffer("q0_condition_audio_prototypes", audio_prototypes, persistent=False)

    def _finite(self, tensor: torch.Tensor, limit: float = 50.0) -> torch.Tensor:
        if tensor is None:
            return tensor
        return torch.nan_to_num(tensor, nan=0.0, posinf=limit, neginf=-limit).clamp(min=-limit, max=limit)

    def _make_projection(self, input_dim: int = None) -> nn.Module:
        if input_dim is None:
            input_dim = self.input_dim
        return nn.Sequential(
            nn.Linear(int(input_dim), self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
        )

    @staticmethod
    def _zero_last_linear(module: nn.Module) -> None:
        for layer in reversed(list(module.modules())):
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                return

    def _load_q0_class_prototypes(self) -> Tuple[torch.Tensor, torch.Tensor]:
        empty = torch.empty(0, dtype=torch.float32)
        if not self.use_q0_class_prototypes:
            return empty, empty
        if not self.q0_condition_prototype_path:
            raise ValueError("q0_condition_mode=class_prototype requires model.q0_condition_prototype_path")
        prototype_path = Path(str(self.q0_condition_prototype_path))
        if not prototype_path.exists():
            raise FileNotFoundError(f"Q0 prototype file not found: {prototype_path}")
        payload = torch.load(prototype_path, map_location="cpu")
        text_prototypes = payload.get("text_prototypes")
        audio_prototypes = payload.get("audio_prototypes")
        if text_prototypes is None or audio_prototypes is None:
            raise KeyError(f"Q0 prototype file must contain text_prototypes and audio_prototypes: {prototype_path}")
        text_prototypes = torch.as_tensor(text_prototypes, dtype=torch.float32)
        audio_prototypes = torch.as_tensor(audio_prototypes, dtype=torch.float32)
        expected_text = (self.num_classes, self.text_input_dim)
        expected_audio = (self.num_classes, self.audio_input_dim)
        if tuple(text_prototypes.shape) != expected_text or tuple(audio_prototypes.shape) != expected_audio:
            raise ValueError(
                "Q0 prototype shape mismatch: "
                f"text={tuple(text_prototypes.shape)}, audio={tuple(audio_prototypes.shape)}, "
                f"expected_text={expected_text}, expected_audio={expected_audio}"
            )
        return text_prototypes, audio_prototypes

    def _project_q0_raw_prototypes(self, prototypes: torch.Tensor, projection: nn.Module, modality_idx: int) -> torch.Tensor:
        projected = prototypes.to(next(projection.parameters()).device)
        for layer in projection:
            if isinstance(layer, nn.Dropout):
                continue
            projected = layer(projected)
        modality_embed = self.modality_embedding[modality_idx].view(1, self.hidden_dim)
        return self._finite(projected + modality_embed)

    def _classifier_hidden_dim(self) -> int:
        ratio = min(1.0, max(0.125, float(self.classifier_hidden_ratio)))
        return max(8, int(round(self.hidden_dim * ratio)))

    def _classifier_dropout(self) -> float:
        return min(0.8, max(0.0, self.dropout_rate * float(self.classifier_dropout_scale)))

    def _make_scalar_classifier(self) -> nn.Module:
        if self.classifier_head_type == "linear":
            return nn.Linear(self.hidden_dim, 1)
        if self.classifier_head_type == "ln_linear":
            return nn.Sequential(nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, 1))
        if self.classifier_head_type == "residual_mlp":
            return ResidualLinearClassifier(
                self.hidden_dim,
                1,
                self._classifier_hidden_dim(),
                self._classifier_dropout(),
                gate_init=self.classifier_residual_gate_init,
            )
        return nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, 1),
        )

    def _make_vector_classifier(self, hidden_layers: bool = True) -> nn.Module:
        if self.classifier_head_type == "linear":
            return nn.Linear(self.hidden_dim, self.num_classes)
        if self.classifier_head_type == "ln_linear":
            return nn.Sequential(nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.num_classes))
        if self.classifier_head_type == "residual_mlp":
            return ResidualLinearClassifier(
                self.hidden_dim,
                self.num_classes,
                self._classifier_hidden_dim(),
                self._classifier_dropout(),
                gate_init=self.classifier_residual_gate_init,
            )
        if not hidden_layers:
            return nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.num_classes),
            )
        return nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, self.num_classes),
        )

    def _make_aux_scorer(self) -> nn.Module:
        return self._make_scalar_classifier()

    def set_class_weights(self, weights: torch.Tensor) -> None:
        self.class_weights = weights.detach().float()

    def set_class_counts(self, counts: torch.Tensor) -> None:
        counts = counts.detach().float().clamp_min(1.0)
        prior = counts / counts.sum().clamp_min(1.0)
        self.class_log_prior = prior.log()

    def _format_forward_output(
        self,
        logits: torch.Tensor,
        logits_dict: Dict[str, torch.Tensor],
        return_loss_items: bool = False,
    ):
        if return_loss_items:
            return build_refqformer_loss_outputs(logits, logits_dict)
        return logits, logits_dict

    def _calculate_refqformer_losses(
        self,
        logits: torch.Tensor,
        logits_dict: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.refqformer_loss is None:
            raise RuntimeError("RefQFormerLoss requested but criterion was not initialized")
        outputs = build_refqformer_loss_outputs(logits, logits_dict)
        if (
            getattr(self.refqformer_loss, "use_reconstruction_loss", False)
            and getattr(self.refqformer_loss, "lambda_recon", 0.0) > 0.0
            and hasattr(self, "_reconstruction_loss")
        ):
            reconstruction_loss, reconstruction_text_loss, reconstruction_audio_loss = self._reconstruction_loss(
                logits_dict
            )
            outputs["loss_recon"] = reconstruction_loss
            outputs["reconstruction_loss"] = reconstruction_loss
            outputs["loss_recon_text"] = reconstruction_text_loss
            outputs["reconstruction_text_loss"] = reconstruction_text_loss
            outputs["loss_recon_audio"] = reconstruction_audio_loss
            outputs["reconstruction_audio_loss"] = reconstruction_audio_loss
        class_weights = self.class_weights if self.class_weights.numel() > 0 else None
        _, losses = self.refqformer_loss(outputs, labels, class_weights=class_weights)
        return losses

    def freeze_pretrained_base(self) -> int:
        trainable_count = 0
        for name, param in self.named_parameters():
            param.requires_grad = name.startswith(self.freeze_trainable_prefixes)
            if param.requires_grad:
                trainable_count += param.numel()
        return trainable_count

    def _flatten_tokens(self, features: torch.Tensor, attention_mask: torch.Tensor, projection: nn.Module, modality_idx: int):
        batch_size, num_units, seq_len, _ = features.shape
        features = self._finite(features, limit=1e4)
        projected = self._finite(projection(features))
        unit_ids = torch.arange(num_units, device=features.device).clamp(max=self.max_evidence_units - 1)
        unit_embed = self.unit_embedding(unit_ids).view(1, num_units, 1, self.hidden_dim)
        modality_embed = self.modality_embedding[modality_idx].view(1, 1, 1, self.hidden_dim)
        projected = self._finite(projected + unit_embed + modality_embed)
        if self.training and self.token_dropout_rate > 0:
            projected = self._finite(F.dropout(projected, p=self.token_dropout_rate, training=True))
        flat_tokens = projected.reshape(batch_size, num_units * seq_len, self.hidden_dim)
        flat_mask = attention_mask.reshape(batch_size, num_units * seq_len).bool()
        missing_rows = ~flat_mask.any(dim=1)
        if missing_rows.any():
            flat_tokens = flat_tokens.clone()
            flat_mask = flat_mask.clone()
            flat_tokens[missing_rows, 0, :] = 0.0
            flat_mask[missing_rows, 0] = True
        return self._finite(flat_tokens), flat_mask

    def _masked_unit_mean(self, tokens: torch.Tensor, mask: torch.Tensor, num_units: int, seq_len: int) -> torch.Tensor:
        batch_size = tokens.size(0)
        unit_tokens = tokens.reshape(batch_size, num_units, seq_len, self.hidden_dim)
        unit_mask = mask.reshape(batch_size, num_units, seq_len).float().unsqueeze(-1)
        return self._finite((unit_tokens * unit_mask).sum(dim=2) / unit_mask.sum(dim=2).clamp_min(1.0))

    def _masked_global_mean(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.float().unsqueeze(-1)
        return self._finite((tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0))

    def _masked_chunk_mean(self, tokens: torch.Tensor, mask: torch.Tensor, num_chunks: int) -> torch.Tensor:
        batch_size, seq_len, _ = tokens.shape
        num_chunks = max(1, int(num_chunks))
        chunk_ids = torch.arange(seq_len, device=tokens.device) * num_chunks // max(seq_len, 1)
        chunk_ids = chunk_ids.clamp(max=num_chunks - 1)
        chunk_mask = F.one_hot(chunk_ids, num_classes=num_chunks).to(dtype=tokens.dtype)
        weights = mask.to(dtype=tokens.dtype).unsqueeze(-1) * chunk_mask.unsqueeze(0)
        denom = weights.sum(dim=1).clamp_min(1.0)
        pooled = torch.einsum("blh,blk->bkh", tokens, weights) / denom.unsqueeze(-1)
        return self._finite(pooled)

    def _build_embedding_conditioned_q0_delta(
        self,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        audio_tokens: torch.Tensor,
        audio_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.use_embedding_conditioned_q0:
            return text_tokens.new_zeros(text_tokens.size(0), self.hidden_dim), {}
        audio_pool = self._masked_global_mean(audio_tokens, audio_mask)
        text_slots = self._masked_chunk_mean(text_tokens, text_mask, 2)
        text_slot_logits = self.q0_condition_text_scorer(text_slots).squeeze(-1)
        text_slot_weights = torch.softmax(self._finite(text_slot_logits), dim=-1)
        text_pool = torch.sum(text_slots * text_slot_weights.unsqueeze(-1), dim=1)
        fused_slots = torch.cat([audio_pool, text_pool], dim=-1)
        q0_delta = self.q0_condition_norm(self.q0_condition_fusion(self._finite(fused_slots)))
        q0_gate = self.q0_condition_scale * torch.sigmoid(self.q0_condition_gate)
        q0_delta = self._finite(q0_gate * q0_delta)
        diagnostics = {
            "q0_condition_gate": q0_gate.detach(),
            "q0_condition_delta_norm": q0_delta.detach().norm(dim=-1).mean(),
            "q0_condition_text_slot_weights": text_slot_weights.detach(),
        }
        return q0_delta, diagnostics

    def _build_class_prototype_q0_delta(
        self,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        audio_tokens: torch.Tensor,
        audio_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.use_q0_class_prototypes:
            return text_tokens.new_zeros(text_tokens.size(0), self.num_classes, self.hidden_dim), {}

        text_proto = self._project_q0_raw_prototypes(
            self.q0_condition_text_prototypes,
            self.text_projection,
            0,
        )
        audio_proto = self._project_q0_raw_prototypes(
            self.q0_condition_audio_prototypes,
            self.audio_projection,
            1,
        )
        proto_hidden = self.q0_condition_norm(
            self.q0_condition_prototype_fusion(self._finite(torch.cat([audio_proto, text_proto], dim=-1)))
        )
        q0_gate = self.q0_condition_scale * torch.sigmoid(self.q0_condition_gate)
        if self.use_q0_static_class_prototypes:
            class_delta = self.q0_condition_class_norm(proto_hidden)
            class_delta = class_delta[None, :, :].expand(text_tokens.size(0), -1, -1)
            class_delta = self._finite(q0_gate * class_delta)
            diagnostics = {
                "q0_condition_gate": q0_gate.detach(),
                "q0_condition_delta_norm": class_delta.detach().norm(dim=-1).mean(),
            }
            return class_delta, diagnostics

        audio_pool = self._masked_global_mean(audio_tokens, audio_mask)
        text_slots = self._masked_chunk_mean(text_tokens, text_mask, 2)
        text_slot_logits = self.q0_condition_text_scorer(text_slots).squeeze(-1)
        text_slot_weights = torch.softmax(self._finite(text_slot_logits), dim=-1)
        text_pool = torch.sum(text_slots * text_slot_weights.unsqueeze(-1), dim=1)
        sample_hidden = self.q0_condition_norm(
            self.q0_condition_fusion(self._finite(torch.cat([audio_pool, text_pool], dim=-1)))
        )
        sample_by_class = sample_hidden[:, None, :].expand(-1, self.num_classes, -1)
        proto_by_batch = proto_hidden[None, :, :].expand(text_tokens.size(0), -1, -1)
        class_features = torch.cat(
            [
                sample_by_class,
                proto_by_batch,
                torch.abs(sample_by_class - proto_by_batch),
                sample_by_class * proto_by_batch,
            ],
            dim=-1,
        )
        class_delta = self.q0_condition_class_norm(self.q0_condition_class_fusion(self._finite(class_features)))
        class_delta = self._finite(q0_gate * class_delta)
        proto_similarity = F.cosine_similarity(sample_by_class.detach(), proto_by_batch.detach(), dim=-1)
        diagnostics = {
            "q0_condition_gate": q0_gate.detach(),
            "q0_condition_delta_norm": class_delta.detach().norm(dim=-1).mean(),
            "q0_condition_text_slot_weights": text_slot_weights.detach(),
            "q0_condition_proto_similarity": proto_similarity,
        }
        return class_delta, diagnostics

    def _initialize_queries(
        self,
        batch_size: int,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        audio_tokens: torch.Tensor,
        audio_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        base_queries = self.class_queries + self.class_embedding[:, None, :]
        queries = base_queries.reshape(
            1,
            self.num_classes * self.queries_per_class,
            self.hidden_dim,
        ).expand(batch_size, -1, -1)
        if self.use_embedding_conditioned_q0:
            if self.use_q0_class_prototypes:
                class_delta, q0_diagnostics = self._build_class_prototype_q0_delta(
                    text_tokens,
                    text_mask,
                    audio_tokens,
                    audio_mask,
                )
                grouped = queries.reshape(batch_size, self.num_classes, self.queries_per_class, self.hidden_dim)
                queries = self._finite(grouped + class_delta[:, :, None, :]).reshape(
                    batch_size,
                    self.num_classes * self.queries_per_class,
                    self.hidden_dim,
                )
            else:
                q0_delta, q0_diagnostics = self._build_embedding_conditioned_q0_delta(
                    text_tokens,
                    text_mask,
                    audio_tokens,
                    audio_mask,
                )
                queries = self._finite(queries + q0_delta[:, None, :])
        else:
            q0_diagnostics = {}
        return self._finite(queries), q0_diagnostics

    def _pool_class_queries(
        self,
        queries: torch.Tensor,
        pool_layer: nn.Module = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = queries.size(0)
        grouped = queries.reshape(batch_size, self.num_classes, self.queries_per_class, self.hidden_dim)
        if self.query_pooling_mode in {"mean", "avg"}:
            reps = grouped.mean(dim=2)
            weights = queries.new_full(
                (batch_size, self.num_classes, self.queries_per_class),
                1.0 / float(self.queries_per_class),
            )
            return reps, weights
        if pool_layer is None:
            pool_layer = self.query_pool
        weights = torch.softmax(pool_layer(grouped), dim=2)
        reps = torch.sum(grouped * weights, dim=2)
        return reps, weights.squeeze(-1)

    def _build_unit_context(
        self,
        text_tokens: torch.Tensor,
        audio_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        num_units: int,
        text_seq_len: int,
        audio_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        text_unit_reps = self._masked_unit_mean(text_tokens, text_mask, num_units, text_seq_len)
        audio_unit_reps = self._masked_unit_mean(audio_tokens, audio_mask, num_units, audio_seq_len)
        if self.unit_context_mode == "legacy_mean":
            unit_evidence = 0.5 * (text_unit_reps + audio_unit_reps)
        else:
            unit_pair = torch.cat(
                [
                    text_unit_reps,
                    audio_unit_reps,
                    torch.abs(text_unit_reps - audio_unit_reps),
                    text_unit_reps * audio_unit_reps,
                ],
                dim=-1,
            )
            unit_evidence = self._finite(self.unit_evidence(self._finite(unit_pair)))
        unit_scores = self._finite(self.class_unit_scorer(unit_evidence).transpose(1, 2))
        unit_weights = torch.softmax(unit_scores, dim=-1)
        class_unit_context = self._finite(torch.einsum("bcu,buh->bch", unit_weights, unit_evidence))
        unit_query_context = (
            class_unit_context[:, :, None, :]
            .expand(-1, -1, self.queries_per_class, -1)
            .reshape(text_tokens.size(0), self.num_classes * self.queries_per_class, self.hidden_dim)
        )
        return (
            self._finite(text_unit_reps),
            self._finite(audio_unit_reps),
            unit_weights,
            self._finite(class_unit_context),
            self._finite(unit_query_context),
        )

    def _score_with_units(self, class_reps: torch.Tensor, class_unit_context: torch.Tensor) -> torch.Tensor:
        fused = class_reps + self.class_unit_fusion(torch.cat([class_reps, class_unit_context], dim=-1))
        return self.class_scorer(fused).squeeze(-1)

    def _score_queries(
        self,
        class_reps: torch.Tensor,
        class_unit_context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        class_reps = self._finite(class_reps)
        class_unit_context = self._finite(class_unit_context)
        base_logits = self._finite(self.class_scorer(class_reps).squeeze(-1), limit=1e4)
        evidence_logits = self._finite(self._score_with_units(class_reps, class_unit_context), limit=1e4)
        if self.score_mode == "residual_evidence":
            strength = self.score_evidence_scale * torch.sigmoid(self.score_evidence_logit)
            combined_logits = self._finite(base_logits + strength * (evidence_logits - base_logits), limit=1e4)
        elif self.score_mode == "base":
            strength = base_logits.new_tensor(0.0)
            combined_logits = base_logits
        else:
            strength = base_logits.new_tensor(1.0)
            combined_logits = evidence_logits
        return self._finite(combined_logits, limit=1e4), base_logits, evidence_logits, strength

    def _score_light_queries(self, class_reps: torch.Tensor, class_unit_context: torch.Tensor) -> torch.Tensor:
        fused = class_reps + self.light_class_unit_fusion(torch.cat([class_reps, class_unit_context], dim=-1))
        return self.light_class_scorer(fused).squeeze(-1)

    def _run_parallel_light_path(
        self,
        text_tokens: torch.Tensor,
        audio_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        num_units: int,
        text_seq_len: int,
        audio_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = text_tokens.size(0)
        light_queries = self.light_class_queries + self.class_embedding[:, None, :]
        light_queries = light_queries.reshape(
            1, self.num_classes * self.queries_per_class, self.hidden_dim
        ).expand(batch_size, -1, -1)
        light_text_tokens = text_tokens
        light_audio_tokens = audio_tokens

        for layer in self.light_query_layers:
            light_queries, light_text_tokens, light_audio_tokens = layer(
                light_queries,
                light_text_tokens,
                light_audio_tokens,
                text_mask,
                audio_mask,
                None,
            )

        _, _, _, light_unit_context, _ = self._build_unit_context(
            light_text_tokens,
            light_audio_tokens,
            text_mask,
            audio_mask,
            num_units,
            text_seq_len,
            audio_seq_len,
        )
        light_class_reps, light_query_weights = self._pool_class_queries(light_queries, self.light_query_pool)
        light_logits = self._score_light_queries(light_class_reps, light_unit_context)
        return light_logits, light_class_reps, light_unit_context, light_query_weights

    def _meta_calibrated_logits(
        self,
        query_logits: torch.Tensor,
        global_logits: torch.Tensor,
        base_logits: torch.Tensor,
        evidence_logits: torch.Tensor,
        unit_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if unit_weights.size(-1) < self.max_evidence_units:
            pad = self.max_evidence_units - unit_weights.size(-1)
            unit_features = F.pad(unit_weights, (0, pad))
        else:
            unit_features = unit_weights[..., : self.max_evidence_units]
        meta_features = torch.cat(
            [
                query_logits,
                global_logits,
                base_logits,
                evidence_logits,
                unit_features.reshape(unit_features.size(0), -1),
            ],
            dim=-1,
        )
        meta_delta = self.meta_calibrator(meta_features)
        return query_logits + self.meta_calibrator_scale * meta_delta, meta_delta

    def _global_gated_residual_logits(
        self,
        query_logits: torch.Tensor,
        global_logits: torch.Tensor,
        global_rep: torch.Tensor,
        class_reps: torch.Tensor,
        class_unit_context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        global_context = global_rep[:, None, :].expand(-1, self.num_classes, -1)
        gate_features = torch.cat(
            [
                global_context,
                class_reps,
                class_unit_context,
                torch.abs(class_reps - class_unit_context),
                class_reps * class_unit_context,
            ],
            dim=-1,
        )
        evidence_gate = torch.sigmoid(
            self.global_evidence_gate(gate_features).squeeze(-1) + self.global_evidence_gate_bias
        )
        logits = global_logits + self.qa_gate_scale * evidence_gate * (query_logits - global_logits)
        return logits, evidence_gate

    def _auxiliary_logits(
        self,
        class_queries: torch.Tensor,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        attention: nn.MultiheadAttention,
        scorer: nn.Module,
    ) -> torch.Tensor:
        attended, _ = attention(
            self._finite(class_queries),
            self._finite(tokens),
            self._finite(tokens),
            key_padding_mask=~mask.bool(),
            need_weights=False,
        )
        reps, _ = self._pool_class_queries(self._finite(attended))
        return self._finite(scorer(self._finite(reps)).squeeze(-1), limit=1e4)

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        text_attention_mask: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        return_loss_items: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = text_features.size(0)
        num_units = text_features.size(1)
        text_seq_len = text_features.size(2)
        audio_seq_len = audio_features.size(2)
        text_tokens, text_mask = self._flatten_tokens(text_features, text_attention_mask, self.text_projection, 0)
        audio_tokens, audio_mask = self._flatten_tokens(audio_features, audio_attention_mask, self.audio_projection, 1)
        initial_text_tokens = text_tokens
        initial_audio_tokens = audio_tokens

        queries, q0_diagnostics = self._initialize_queries(
            batch_size,
            text_tokens,
            text_mask,
            audio_tokens,
            audio_mask,
        )
        layer_logits = []
        text_unit_reps = None
        audio_unit_reps = None
        unit_weights = None
        class_unit_context = None
        for layer_index, layer in enumerate(self.query_layers):
            text_unit_reps, audio_unit_reps, unit_weights, class_unit_context, unit_query_context = self._build_unit_context(
                text_tokens,
                audio_tokens,
                text_mask,
                audio_mask,
                num_units,
                text_seq_len,
                audio_seq_len,
            )
            queries, text_tokens, audio_tokens = layer(
                queries,
                text_tokens,
                audio_tokens,
                text_mask,
                audio_mask,
                unit_query_context,
            )
            class_reps, _ = self._pool_class_queries(queries)
            layer_combined_logits, _, _, _ = self._score_queries(class_reps, class_unit_context)
            layer_logits.append(layer_combined_logits)

        text_unit_reps, audio_unit_reps, unit_weights, class_unit_context, _ = self._build_unit_context(
            text_tokens,
            audio_tokens,
            text_mask,
            audio_mask,
            num_units,
            text_seq_len,
            audio_seq_len,
        )
        class_reps, query_weights = self._pool_class_queries(queries)
        final_query_logits, base_logits, evidence_logits, score_evidence_strength = self._score_queries(
            class_reps,
            class_unit_context,
        )
        layer_logits_tensor = torch.stack(layer_logits, dim=0) if layer_logits else final_query_logits.unsqueeze(0)
        query_logits = (
            (1.0 - self.layer_logit_weight) * final_query_logits
            + self.layer_logit_weight * layer_logits_tensor.mean(dim=0)
        )
        text_logits = self._auxiliary_logits(queries, text_tokens, text_mask, self.text_aux_query, self.text_aux_scorer)
        audio_logits = self._auxiliary_logits(queries, audio_tokens, audio_mask, self.audio_aux_query, self.audio_aux_scorer)

        text_global = self._masked_global_mean(text_tokens, text_mask)
        audio_global = self._masked_global_mean(audio_tokens, audio_mask)
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
                text_mask,
                audio_mask,
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

        logit_scale = torch.exp(self.logit_scale.clamp(max=math.log(self.max_logit_scale)))
        logits = logit_scale * logits + self.class_logit_bias

        logits_dict = {
            "query": logit_scale * query_logits + self.class_logit_bias,
            "base_query": base_logits,
            "evidence_query": evidence_logits,
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
            "light_query_weights": light_query_weights,
            "text_prompt_reps": text_unit_reps,
            "audio_prompt_reps": audio_unit_reps,
            "unit_weights": unit_weights,
            "prompt_weights": unit_weights,
            "layer_logits": layer_logits_tensor,
        }
        if layer_logits_tensor.size(0) >= 2:
            logits_dict["aux_logits_q2"] = layer_logits_tensor[1]
        if layer_logits_tensor.size(0) >= 3:
            logits_dict["aux_logits_q3"] = layer_logits_tensor[2]
        logits_dict["final_queries"] = queries
        if q0_diagnostics:
            logits_dict.update(q0_diagnostics)
        return self._format_forward_output(logits, logits_dict, return_loss_items)

    def _diversity_loss(self, queries: torch.Tensor) -> torch.Tensor:
        grouped = queries.reshape(queries.size(0), self.num_classes, self.queries_per_class, self.hidden_dim)
        normalized = F.normalize(grouped, dim=-1)
        similarity = torch.matmul(normalized, normalized.transpose(-1, -2))
        identity = torch.eye(self.queries_per_class, device=queries.device).view(1, 1, self.queries_per_class, self.queries_per_class)
        return (similarity - identity).pow(2).mean()

    def _classification_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        class_weights = self.class_weights if self.class_weights.numel() > 0 else None
        loss_logits = logits
        if self.logit_adjustment_tau != 0 and self.class_log_prior.numel() == self.num_classes:
            loss_logits = logits + self.logit_adjustment_tau * self.class_log_prior.to(logits.device)
        if self.focal_gamma <= 0:
            return F.cross_entropy(
                loss_logits,
                labels,
                weight=class_weights,
                label_smoothing=self.label_smoothing,
            )

        ce_loss = F.cross_entropy(
            loss_logits,
            labels,
            weight=class_weights,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        true_probs = F.softmax(loss_logits, dim=-1).gather(1, labels.view(-1, 1)).squeeze(1)
        focal_weight = (1.0 - true_probs.clamp_min(1e-6)).pow(self.focal_gamma)
        return (focal_weight * ce_loss).mean()

    def _binary_rank_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.num_classes != 2:
            return logits.new_tensor(0.0)
        margin = logits[:, 1] - logits[:, 0]
        pos_margin = margin[labels == 1]
        neg_margin = margin[labels == 0]
        if pos_margin.numel() == 0 or neg_margin.numel() == 0:
            return logits.new_tensor(0.0)
        return F.softplus(neg_margin[:, None] - pos_margin[None, :] + self.rank_margin).mean()

    def _query_contrastive_loss(self, class_reps: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if class_reps is None or class_reps.size(0) < 2:
            return labels.new_tensor(0.0, dtype=torch.float32)
        batch_size = class_reps.size(0)
        labels = labels.long()
        if (labels < 0).any() or (labels >= self.num_classes).any():
            return class_reps.new_tensor(0.0)

        batch_indices = torch.arange(batch_size, device=class_reps.device)
        true_class_reps = class_reps[batch_indices, labels]
        z = F.normalize(self._finite(true_class_reps), dim=-1)
        temperature = max(float(self.query_contrastive_temperature), 1e-4)
        similarity = torch.matmul(z, z.transpose(0, 1)) / temperature

        self_mask = torch.eye(batch_size, device=class_reps.device, dtype=torch.bool)
        positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
        valid = positive_mask.any(dim=1)
        if not valid.any():
            return class_reps.new_tensor(0.0)

        similarity = self._finite(similarity, limit=50.0).masked_fill(self_mask, -1e4)
        log_prob = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
        positive_count = positive_mask.float().sum(dim=1).clamp_min(1.0)
        per_anchor_loss = -(log_prob * positive_mask.float()).sum(dim=1) / positive_count
        return self._finite(per_anchor_loss[valid]).mean()

    def calculate_losses(
        self,
        logits: torch.Tensor,
        logits_dict: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.use_refqformer_loss:
            return self._calculate_refqformer_losses(logits, logits_dict, labels)

        main_loss = self._classification_loss(logits, labels)
        text_loss = self._classification_loss(logits_dict["text"], labels)
        audio_loss = self._classification_loss(logits_dict["audio"], labels)
        aux_loss = 0.5 * (text_loss + audio_loss)
        global_loss = logits.new_tensor(0.0)
        if self.global_loss_weight > 0:
            global_loss = self._classification_loss(logits_dict["global"], labels)

        light_loss = logits.new_tensor(0.0)
        if self.light_loss_weight > 0 and "light" in logits_dict:
            light_loss = self._classification_loss(logits_dict["light"], labels)

        rank_loss = logits.new_tensor(0.0)
        if self.rank_loss_weight > 0:
            rank_loss = self._binary_rank_loss(logits, labels)

        query_contrastive_loss = logits.new_tensor(0.0)
        if self.training and self.query_contrastive_loss_weight > 0:
            query_contrastive_loss = self._query_contrastive_loss(logits_dict.get("class_reps"), labels)

        layer_loss = logits.new_tensor(0.0)
        if self.layer_loss_weight > 0 and "layer_logits" in logits_dict:
            layer_loss = torch.stack(
                [self._classification_loss(layer_logits, labels) for layer_logits in logits_dict["layer_logits"]]
            ).mean()

        alignment_loss = logits.new_tensor(0.0)
        if self.alignment_loss_weight > 0:
            alignment_loss = 1.0 - F.cosine_similarity(
                logits_dict["text_prompt_reps"], logits_dict["audio_prompt_reps"], dim=-1
            ).mean()

        diversity_loss = logits.new_tensor(0.0)
        if self.diversity_loss_weight > 0:
            diversity_loss = self._diversity_loss(logits_dict["queries"])

        total_loss = (
            main_loss
            + self.aux_loss_weight * aux_loss
            + self.layer_loss_weight * layer_loss
            + self.global_loss_weight * global_loss
            + self.light_loss_weight * light_loss
            + self.rank_loss_weight * rank_loss
            + self.query_contrastive_loss_weight * query_contrastive_loss
            + self.alignment_loss_weight * alignment_loss
            + self.diversity_loss_weight * diversity_loss
        )
        return {
            "total_loss": total_loss,
            "main_loss": main_loss,
            "aux_loss": aux_loss,
            "layer_loss": layer_loss,
            "global_loss": global_loss,
            "light_loss": light_loss,
            "rank_loss": rank_loss,
            "query_contrastive_loss": query_contrastive_loss,
            "alignment_loss": alignment_loss,
            "diversity_loss": diversity_loss,
        }
