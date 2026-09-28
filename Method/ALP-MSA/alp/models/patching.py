"""
Affective Latent Patching (ALP): turn a variable-length frame sequence into K
latent tokens whose temporal extent follows the *affective event structure* of
the modality instead of a fixed frame grid.

Tokenizers (all return (tokens, key_mask, aux)):
  almt     original ALMT compression: K learnable tokens prepended to the
           (unmasked, zero-padded) frame sequence -- faithful baseline.
  query    same as `almt` but padding frames are masked out.
  frame    no compression: masked frame-level features (key_mask returned).
  uniform  fixed-length patches: frames are split into K equal-duration
           patches of the *valid* part of the sequence and mean-pooled.
  dynamic  ALP (ours): an event-boundary score b_t decides how much "affective
           mass" each frame carries; the cumulative mass is re-normalised to
           [0, K] so that regions with many emotion changes are split into
           short patches while stationary regions are merged into long ones.
           With constant b_t the tokenizer reduces exactly to `uniform`.
"""

import math
import torch
import torch.nn.functional as F
from torch import nn
from einops import repeat

from .layers import (Attention, FeedForward, PreNormAttention, PreNormForward, Transformer,
                     TransformerEncoder)


class FrameEncoder(nn.Module):
    """Linear projection + learned positions + masked transformer over frames."""

    def __init__(self, in_dim, dim, max_len, depth, heads, mlp_dim, dropout=0., pos_std=0.02):
        super().__init__()
        self.proj = nn.Linear(in_dim, dim)
        self.pos = nn.Parameter(torch.randn(1, max_len, dim) * pos_std)   # ALMT uses std 1
        self.encoder = TransformerEncoder(dim, depth, heads, 64, mlp_dim, dropout)

    def forward(self, x, mask):
        h = self.proj(x) + self.pos[:, :x.size(1)]
        return self.encoder(h, mask=mask)


class AffectivePatcher(nn.Module):
    def __init__(self, mode, in_dim, dim, num_tokens, max_len, depth=1, heads=8, mlp_dim=128,
                 patch_depth=1, tau=0.25, b_min=0.05, use_raw_cue=True, dropout=0., frame_pos_std=0.02):
        super().__init__()
        assert mode in ('almt', 'query', 'frame', 'uniform', 'dynamic', 'uniq', 'dynq'), mode
        self.mode, self.K, self.tau, self.b_min = mode, num_tokens, tau, b_min
        self.use_raw_cue = use_raw_cue
        self.is_dyn = mode in ('dynamic', 'dynq')      # learned event boundaries
        self.is_query = mode in ('uniq', 'dynq')       # tokens = learnable queries with patch receptive fields

        if mode == 'almt':
            self.net = nn.Sequential(
                nn.Linear(in_dim, dim),
                Transformer(num_frames=max_len, token_len=num_tokens, save_hidden=False,
                            dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim))
            return
        if mode == 'query':
            self.proj = nn.Linear(in_dim, dim)
            self.net = Transformer(num_frames=max_len, token_len=num_tokens, save_hidden=False,
                                   dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim)
            return

        self.frame_enc = FrameEncoder(in_dim, dim, max_len, depth, heads, mlp_dim, dropout, frame_pos_std)
        if mode == 'frame':
            return

        # patch-level modules (uniform / dynamic / uniq / dynq)
        self.patch_pos = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
        self.desc_proj = nn.Linear(3, dim)          # duration, temporal centre, boundary strength
        self.patch_enc = TransformerEncoder(dim, patch_depth, heads, 64, mlp_dim, dropout)

        if self.is_query:
            # event queries: K learnable queries, query k attends to the frames with bias
            # bias_scale * log w(t, k) (+ salience), i.e. its receptive field is its (event) patch
            self.queries = nn.Parameter(torch.randn(1, num_tokens, dim))   # std 1 as ALMT token pos-emb: stable early token space
            self.q_attn = PreNormAttention(dim, Attention(dim, heads=heads, dim_head=64, dropout=dropout))
            self.q_ff = PreNormForward(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            self.bias_scale = nn.Parameter(torch.tensor(1.0))

        if self.is_dyn:
            self.to_q = nn.Linear(dim, dim, bias=False)
            self.to_k = nn.Linear(dim, dim, bias=False)
            self.cos_scale = nn.Parameter(torch.tensor(1.0))
            self.boundary_bias = nn.Parameter(torch.tensor(0.0))
            if use_raw_cue:
                # raw affective change cue |x_t - x_{t-1}| (prosody / facial motion)
                self.raw_cue = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 1))
            self.salience = nn.Linear(dim, 1)

    # ------------------------------------------------------------------ #
    def _boundaries(self, x, h, m):
        """Return boundary strength b in [b_min, 1], shape (B, L); padding -> 0."""
        B, L, _ = h.shape
        if not self.is_dyn:
            return m.clone()
        q, k = self.to_q(h), self.to_k(h)
        cos = F.cosine_similarity(q[:, 1:], k[:, :-1], dim=-1)           # (B, L-1)
        p = ((1 - cos) / 2).clamp(1e-4, 1 - 1e-4)                         # H-Net style change prob.
        logit = self.cos_scale * torch.log(p / (1 - p)) + self.boundary_bias
        if self.use_raw_cue:
            dx = (x[:, 1:] - x[:, :-1]).abs()
            logit = logit + self.raw_cue(dx).squeeze(-1)
        b = self.b_min + (1 - self.b_min) * torch.sigmoid(logit)
        b = torch.cat([torch.ones(B, 1, device=h.device, dtype=h.dtype), b], 1)  # first frame opens an event
        return b * m

    def _pool(self, h, m, b, salience):
        """Soft assignment of frames to K content-adaptive patches."""
        K = self.K
        U = torch.cumsum(b, 1)
        total = U[:, -1:].clamp(min=1e-6)
        centre = (U - b / 2) / total * K                                    # (B, L) in (0, K)
        grid = torch.arange(K, device=h.device, dtype=h.dtype) + 0.5       # (K,)
        logits_w = -(centre.unsqueeze(-1) - grid) ** 2 / self.tau          # (B, L, K)
        w = torch.softmax(logits_w, -1) * m.unsqueeze(-1)                  # frame -> patch assignment

        if salience is not None:
            s = salience.masked_fill(m == 0, -1e4)
            s = torch.exp(s - s.max(1, keepdim=True).values) * m          # (B, L)
            a = w * s.unsqueeze(-1)
        else:
            a = w
        denom = a.sum(1).clamp(min=1e-6)                                   # (B, K)
        tokens = torch.einsum('blk,bld->bkd', a, h) / denom.unsqueeze(-1)

        # patch descriptors
        n_valid = m.sum(1, keepdim=True).clamp(min=1.0)
        w_sum = w.sum(1).clamp(min=1e-6)                                   # (B, K)
        t_rel = (torch.arange(h.size(1), device=h.device, dtype=h.dtype)[None] / n_valid) * m
        duration = w_sum / n_valid
        t_centre = torch.einsum('blk,bl->bk', w, t_rel) / w_sum
        strength = torch.einsum('blk,bl->bk', w, b) / w_sum
        desc = torch.stack([duration, t_centre, strength], -1)
        return tokens, desc, w, a / denom.unsqueeze(1)

    @staticmethod
    def _event_loss(h, m, w, a_norm):
        """Event coherence: fraction of frame variance NOT explained by the frame's patch.

        Frames are reconstructed from the (salience-weighted) mean of the patch they are
        assigned to.  h is detached, so the loss only shapes boundaries / salience: patches
        should cover internally homogeneous affective events (learned change-point detection).
        """
        hd = h.detach()
        q = torch.einsum('blk,bld->bkd', a_norm, hd)
        h_hat = torch.einsum('blk,bkd->bld', w, q)
        err = ((hd - h_hat) ** 2).mean(-1)
        n = m.sum(1, keepdim=True).clamp(min=1.0)
        mu = (hd * m.unsqueeze(-1)).sum(1, keepdim=True) / n.unsqueeze(-1)
        var = ((hd - mu) ** 2).mean(-1)
        return ((err * m).sum(1) / ((var * m).sum(1) + 1e-6)).mean()

    # ------------------------------------------------------------------ #
    def forward(self, x, mask):
        """x: (B, L, in_dim); mask: (B, L) bool, True = valid frame."""
        if self.mode == 'almt':
            return self.net(x)[:, :self.K], None, {}
        if self.mode == 'query':
            return self.net(self.proj(x), mask=mask)[:, :self.K], None, {}

        h = self.frame_enc(x, mask)
        if self.mode == 'frame':
            return h, mask, {}

        m = mask.to(h.dtype)
        b = self._boundaries(x, h, m)
        sal = self.salience(h).squeeze(-1) if self.is_dyn else None
        tokens, desc, w, a_norm = self._pool(h, m, b, sal)
        if self.is_query:
            bias = self.bias_scale * torch.log(w.clamp(min=1e-4)).transpose(1, 2)   # (B, K, L)
            if sal is not None:
                bias = bias + sal.unsqueeze(1)
            q0 = self.queries + self.patch_pos + self.desc_proj(desc)
            tokens = q0 + self.q_attn(q0, h, h, mask=mask, bias=bias.unsqueeze(1))
            tokens = tokens + self.q_ff(tokens)
        else:
            tokens = tokens + self.patch_pos + self.desc_proj(desc)
        tokens = self.patch_enc(tokens)
        aux = {'boundary': b, 'assign': w, 'desc': desc}
        if self.is_dyn:
            aux['event_loss'] = self._event_loss(h, m, w, a_norm)
        return tokens, None, aux
