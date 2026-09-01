from __future__ import annotations

import copy
import math
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .coalition import CoalitionFusion
from .encoders import ModalityResampler, TemporalEncoder
from .losses import task_loss_per_sample
from .memory import ProceduralMemory, RetrievalOutput
from .mobius import MobiusInteraction


class PredictionHead(nn.Module):
    def __init__(self, d_model: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden = max(d_model // 2, 32)
        # Keep the prediction head deterministic. The paired memory regret
        # compares base and memory-corrected losses on the same sample; a
        # stochastic head would contaminate that comparison with dropout noise.
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RAMPModel(nn.Module):
    """RAMP: Regret-Aware Multimodal Procedural Memory.

    Core idea:
      1) evaluate a shared fusion function on all T/A/V coalitions;
      2) Möbius-decompose the result into 7 interaction coordinates;
      3) use the interaction state to retrieve a 7-D procedural fusion policy;
      4) apply the policy as a residual reweighting of interaction components.
    """

    POLICY_DIM = 7

    def __init__(self, text_dim: int, audio_dim: int, vision_dim: int, cfg):
        super().__init__()
        self.cfg = cfg
        self.task_type = str(cfg.task.type)
        self.num_classes = int(cfg.task.get("num_classes", 4))
        d_model = int(cfg.model.d_model)
        self.d_model = d_model
        nhead = int(cfg.model.nhead)
        dropout = float(cfg.model.dropout)

        self.text_encoder = TemporalEncoder(
            text_dim, d_model, nhead=nhead,
            num_layers=int(cfg.model.get("temporal_layers", 1)),
            ff_mult=int(cfg.model.get("ff_mult", 4)), dropout=dropout,
            max_tokens=int(cfg.model.get("max_text_tokens", 64)),
        )
        self.audio_encoder = TemporalEncoder(
            audio_dim, d_model, nhead=nhead,
            num_layers=int(cfg.model.get("temporal_layers", 1)),
            ff_mult=int(cfg.model.get("ff_mult", 4)), dropout=dropout,
            max_tokens=int(cfg.model.get("max_audio_tokens", 96)),
        )
        self.vision_encoder = TemporalEncoder(
            vision_dim, d_model, nhead=nhead,
            num_layers=int(cfg.model.get("temporal_layers", 1)),
            ff_mult=int(cfg.model.get("ff_mult", 4)), dropout=dropout,
            max_tokens=int(cfg.model.get("max_vision_tokens", 96)),
        )
        num_latents = int(cfg.model.get("resampler_tokens", 4))
        self.text_resampler = ModalityResampler(d_model, num_latents, nhead=nhead, ff_mult=int(cfg.model.get("ff_mult", 4)), dropout=dropout)
        self.audio_resampler = ModalityResampler(d_model, num_latents, nhead=nhead, ff_mult=int(cfg.model.get("ff_mult", 4)), dropout=dropout)
        self.vision_resampler = ModalityResampler(d_model, num_latents, nhead=nhead, ff_mult=int(cfg.model.get("ff_mult", 4)), dropout=dropout)
        self.coalition_fusion = CoalitionFusion(
            d_model=d_model,
            num_modality_tokens=num_latents,
            nhead=nhead,
            num_layers=int(cfg.model.get("fusion_layers", 2)),
            ff_mult=int(cfg.model.get("ff_mult", 4)),
            dropout=dropout,
        )
        state_dim = int(cfg.model.get("interaction_dim", 128))
        self.mobius = MobiusInteraction(d_model, state_dim, dropout=dropout)
        self.base_norm = nn.LayerNorm(d_model)

        key_dim = int(cfg.memory.key_dim)
        query_in = state_dim + d_model
        self.query_fast = nn.Sequential(
            nn.Linear(query_in, key_dim),
            nn.GELU(),
            nn.LayerNorm(key_dim),
            nn.Linear(key_dim, key_dim),
        )
        self.query_slow = copy.deepcopy(self.query_fast)
        for p in self.query_slow.parameters():
            p.requires_grad = False

        out_dim = 1 if self.task_type == "regression" else self.num_classes
        self.head = PredictionHead(d_model, out_dim, dropout=dropout)

        self.memory_enabled = bool(cfg.memory.get("enabled", True))
        self.memory = ProceduralMemory(
            num_regimes=int(cfg.memory.num_regimes),
            slots_per_regime=int(cfg.memory.slots_per_regime),
            key_dim=key_dim,
            policy_dim=self.POLICY_DIM,
            top_regimes=int(cfg.memory.get("top_regimes", 2)),
            top_slots=int(cfg.memory.get("top_slots", 16)),
            route_temperature=float(cfg.memory.get("route_temperature", 0.12)),
            slot_temperature=float(cfg.memory.get("slot_temperature", 0.08)),
            utility_momentum=float(cfg.memory.get("utility_momentum", 0.95)),
            stats_momentum=float(cfg.memory.get("stats_momentum", 0.95)),
            merge_similarity=float(cfg.memory.get("merge_similarity", 0.90)),
            merge_momentum=float(cfg.memory.get("merge_momentum", 0.20)),
            redundancy_weight=float(cfg.memory.get("redundancy_weight", 0.25)),
            min_updates=int(cfg.memory.get("min_updates", 1)),
            max_updates=int(cfg.memory.get("max_updates", 4)),
            plasticity_bias=float(cfg.memory.get("plasticity_bias", -0.5)),
            plasticity_hardness=float(cfg.memory.get("plasticity_hardness", 1.0)),
            plasticity_novelty=float(cfg.memory.get("plasticity_novelty", 2.0)),
            plasticity_bad_gain=float(cfg.memory.get("plasticity_bad_gain", 1.0)),
        )

        # Scalar trust gate: it cannot itself choose a 7-D policy, so the memory
        # remains responsible for procedural action selection.
        self.memory_gate = nn.Sequential(
            nn.Linear(state_dim + d_model + 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        init_strength = float(cfg.memory.get("initial_policy_strength", 0.25))
        raw = math.log(math.exp(max(init_strength, 1e-4)) - 1.0)
        self.raw_policy_strength = nn.Parameter(torch.tensor(raw, dtype=torch.float32))

    def _encode_modalities(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        ht, _, mt = self.text_encoder(batch["text"], batch["text_mask"])
        ha, _, ma = self.audio_encoder(batch["audio"], batch["audio_mask"])
        hv, _, mv = self.vision_encoder(batch["vision"], batch["vision_mask"])
        zt = self.text_resampler(ht, mt)
        za = self.audio_resampler(ha, ma)
        zv = self.vision_resampler(hv, mv)
        modal_vecs = torch.stack([zt, za, zv], dim=1)
        availability = batch.get("availability")
        if availability is not None:
            if availability.ndim != 2 or availability.shape != modal_vecs.shape[:2]:
                raise ValueError(
                    "availability must have shape [batch, 3], got "
                    f"{tuple(availability.shape)}"
                )
            # A missing stream is zeroed after its temporal encoder/resampler as
            # well as at the raw input. This blocks positional embeddings,
            # biases, and learned resampler queries from acting as a surrogate
            # observation for an unavailable modality.
            modal_vecs = modal_vecs * availability.to(modal_vecs.dtype).unsqueeze(-1).unsqueeze(-1)
        return modal_vecs

    def _queries(self, h_base: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_in = torch.cat([state, self.base_norm(h_base)], dim=-1)
        fast = F.normalize(self.query_fast(q_in), p=2, dim=-1, eps=1e-8)
        with torch.no_grad():
            slow = F.normalize(self.query_slow(q_in.detach()), p=2, dim=-1, eps=1e-8)
        return fast, slow

    def forward(self, batch: Dict[str, torch.Tensor], use_memory: bool = True) -> Dict[str, torch.Tensor]:
        modal_vecs = self._encode_modalities(batch)
        coalition_states = self.coalition_fusion(modal_vecs)
        components, interaction_state, policy_directions = self.mobius(coalition_states)
        h_base = coalition_states[:, 6]
        base_logits = self.head(h_base)
        fast_q, slow_q = self._queries(h_base, interaction_state)

        memory_active = self.memory_enabled and use_memory and self.memory.ready()
        retrieval: RetrievalOutput = self.memory.retrieve(fast_q) if memory_active else self.memory.retrieve(fast_q.detach())

        if memory_active:
            # Normalize entropy by its maximum possible value for a stable gate input.
            max_ent = math.log(max(int(self.memory.valid.sum().item()), 2))
            ent = (retrieval.entropy / max_ent).unsqueeze(-1)
            sim = retrieval.max_similarity.unsqueeze(-1)
            gate_in = torch.cat([interaction_state, self.base_norm(h_base), ent, sim], dim=-1)
            gate = torch.sigmoid(self.memory_gate(gate_in))
            strength = F.softplus(self.raw_policy_strength)
            alpha = gate * strength * retrieval.policy
            delta = torch.einsum("bp,bpd->bd", alpha, policy_directions)
            h_final = h_base + delta
        else:
            gate = torch.zeros(h_base.size(0), 1, device=h_base.device, dtype=h_base.dtype)
            alpha = torch.zeros(h_base.size(0), self.POLICY_DIM, device=h_base.device, dtype=h_base.dtype)
            h_final = h_base

        final_logits = self.head(h_final) if memory_active else base_logits
        return {
            "base_logits": base_logits,
            "final_logits": final_logits,
            "h_base": h_base,
            "h_final": h_final,
            "coalition_states": coalition_states,
            "components": components,
            "interaction_state": interaction_state,
            "policy_directions": policy_directions,
            "fast_key": fast_q,
            "slow_key": slow_q,
            "retrieved_policy": retrieval.policy,
            "route_probs": retrieval.route_probs,
            "slot_attn": retrieval.slot_attn,
            "retrieval_entropy": retrieval.entropy,
            "max_similarity": retrieval.max_similarity,
            "memory_gate": gate,
            "applied_policy": alpha,
            "memory_active": torch.tensor(memory_active, device=h_base.device),
        }

    def oracle_policy(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        view_dropout: float = 0.10,
        smooth_l1_beta: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute a sample-wise loss-reducing local fusion policy.

        Only gradients w.r.t. seven temporary interaction coefficients are taken;
        no parameter gradient is accumulated and no second-order graph is built.
        """
        h = outputs["h_base"].detach().float()
        dirs = outputs["policy_directions"].detach().float()
        labels_det = labels.detach()

        def one_view(direction_view: torch.Tensor) -> torch.Tensor:
            with torch.enable_grad():
                alpha = torch.zeros(
                    h.size(0), self.POLICY_DIM, device=h.device, dtype=h.dtype, requires_grad=True
                )
                h_oracle = h + torch.einsum("bp,bpd->bd", alpha, direction_view)
                logits = self.head(h_oracle)
                losses = task_loss_per_sample(logits, labels_det, self.task_type, smooth_l1_beta=smooth_l1_beta)
                grad = torch.autograd.grad(losses.sum(), alpha, create_graph=False, retain_graph=False)[0]
            policy = F.normalize(-grad.detach(), p=2, dim=-1, eps=1e-8)
            return policy

        if view_dropout > 0:
            d1 = F.dropout(dirs, p=view_dropout, training=True)
            d2 = F.dropout(dirs, p=view_dropout, training=True)
            p1 = one_view(d1)
            p2 = one_view(d2)
            policy = F.normalize(p1 + p2, p=2, dim=-1, eps=1e-8)
            stability = ((p1 * p2).sum(dim=-1) + 1.0) * 0.5
            stability = stability.clamp(0.0, 1.0)
        else:
            policy = one_view(dirs)
            stability = torch.ones(h.size(0), device=h.device)
        return policy, stability

    @torch.no_grad()
    def update_slow_writer(self, momentum: float = 0.99) -> None:
        m = float(momentum)
        for fast, slow in zip(self.query_fast.parameters(), self.query_slow.parameters()):
            slow.data.mul_(m).add_(fast.data, alpha=1.0 - m)
