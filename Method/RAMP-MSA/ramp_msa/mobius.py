from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .coalition import COALITION_NAMES


class MobiusInteraction(nn.Module):
    """Exact 3-modality Möbius inversion over a shared coalition set function.

    Component order: T, A, V, TA, TV, AV, TAV.
    The components sum exactly to h_TAV (up to floating-point error).
    """

    def __init__(self, d_model: int, state_dim: int, dropout: float = 0.1):
        super().__init__()
        self.component_norm = nn.LayerNorm(d_model)
        self.state_mlp = nn.Sequential(
            nn.Linear(7 * d_model + 7, 2 * state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * state_dim, state_dim),
            nn.LayerNorm(state_dim),
        )

    @staticmethod
    def decompose(h: torch.Tensor) -> torch.Tensor:
        # h: [B,7,D], in T,A,V,TA,TV,AV,TAV order.
        h_t, h_a, h_v, h_ta, h_tv, h_av, h_tav = [h[:, i] for i in range(7)]
        i_t = h_t
        i_a = h_a
        i_v = h_v
        i_ta = h_ta - h_t - h_a
        i_tv = h_tv - h_t - h_v
        i_av = h_av - h_a - h_v
        i_tav = h_tav - h_ta - h_tv - h_av + h_t + h_a + h_v
        return torch.stack([i_t, i_a, i_v, i_ta, i_tv, i_av, i_tav], dim=1)

    def forward(self, coalition_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        comps = self.decompose(coalition_states)
        normed = self.component_norm(comps)
        magnitudes = comps.norm(dim=-1)
        mag_features = torch.log1p(magnitudes)
        state = self.state_mlp(torch.cat([normed.flatten(1), mag_features], dim=-1))

        h_base = coalition_states[:, 6]
        base_norm = h_base.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        # Stable semantic action coordinates: each interaction component becomes
        # a unit direction scaled to an equal share of the current fused norm.
        directions = F.normalize(comps, p=2, dim=-1, eps=1e-6)
        directions = directions * (base_norm / math.sqrt(7.0)).unsqueeze(1)
        return comps, state, directions
