from __future__ import annotations

import math

import torch
import torch.nn as nn

from .utils import sinusoidal_position_encoding


class AttentivePool(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D], mask: [B, L] True for valid.
        logits = self.score(x).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * mask.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.einsum("bl,bld->bd", weights, x)


class TemporalEncoder(nn.Module):
    """Lightweight sequence encoder for pre-extracted modality features."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int = 4,
        num_layers: int = 1,
        ff_mult: int = 4,
        dropout: float = 0.1,
        max_tokens: int | None = None,
    ):
        super().__init__()
        self.max_tokens = max_tokens
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.pool = AttentivePool(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

    def _subsample(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.max_tokens is None or x.size(1) <= self.max_tokens:
            return x, mask
        # Uniform temporal sampling keeps the package lightweight on unaligned
        # MMSA features (audio can be 500 frames, vision 375 frames).
        idx = torch.linspace(0, x.size(1) - 1, steps=self.max_tokens, device=x.device).round().long()
        return x.index_select(1, idx), mask.index_select(1, idx)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, mask = self._subsample(x, mask)
        # Guarantee at least one unmasked token for numerically safe attention.
        bad = ~mask.any(dim=1)
        if bad.any():
            mask = mask.clone()
            mask[bad, 0] = True
        h = self.proj(x)
        pe = sinusoidal_position_encoding(h.size(1), h.size(2), h.device, h.dtype)
        h = self.dropout(h + pe.unsqueeze(0))
        h = self.encoder(h, src_key_padding_mask=~mask)
        h = self.norm(h)
        pooled = self.pool(h, mask)
        return h, pooled, mask


class ModalityResampler(nn.Module):
    """Compress a variable-length modality sequence into a few latent tokens.

    This keeps the shared 7-coalition fusion expressive without replicating
    hundreds of raw temporal tokens seven times.
    """

    def __init__(self, d_model: int, num_tokens: int = 4, nhead: int = 4, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.queries = nn.Parameter(torch.randn(1, self.num_tokens, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b = seq.size(0)
        q = self.queries.expand(b, -1, -1)
        attn, _ = self.attn(q, seq, seq, key_padding_mask=~mask, need_weights=False)
        q = self.norm1(q + self.dropout(attn))
        q = self.norm2(q + self.dropout(self.ffn(q)))
        return q
