"""
Transformer building blocks adapted from ALMT (Zhang et al., EMNLP 2023,
https://github.com/Haoyu-ha/ALMT, MIT License).

Changes w.r.t. the original `almt_layer.py`:
  * every attention op accepts an optional key-padding mask
    (B, N_k) bool, True = valid position);
  * the hyper-modality layer accepts separate masks for audio / vision keys,
    so that frame-level (uncompressed) A/V sequences can be used as keys.
With mask=None all modules behave exactly like the original ALMT code.
"""

import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat


def _sdpa(q, k, v, mask=None, bias=None):
    """softmax(q k^T / sqrt(d) + bias, keys masked) v via the memory-efficient SDPA kernel.
    q, k, v: (b, h, n, d); mask: (b, j) bool, True = keep; bias: (b, h, i, j) additive or None."""
    attn_mask = None
    if bias is not None:
        attn_mask = bias
        if mask is not None:
            attn_mask = attn_mask.masked_fill(~mask[:, None, None, :], float('-inf'))
    elif mask is not None:
        attn_mask = mask[:, None, None, :]
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)


class PreNormForward(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNormAttention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, mask=None):
        return self.fn(self.norm_q(q), self.norm_k(k), self.norm_v(v), mask=mask)


class PreNormAHL(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, h_t, h_a, h_v, h_hyper, mask_a=None, mask_v=None, times=None):
        return self.fn(self.norm1(h_t), self.norm2(h_a), self.norm3(h_v), self.norm4(h_hyper),
                       mask_a=mask_a, mask_v=mask_v, times=times)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v, mask=None):
        h = self.heads
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        out = rearrange(_sdpa(q, k, v, mask), 'b h n d -> b n (h d)')
        return self.to_out(out)


class HhyperLearningLayer(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k_ta = nn.Linear(dim, inner_dim, bias=False)
        self.to_k_tv = nn.Linear(dim, inner_dim, bias=False)
        self.to_v_ta = nn.Linear(dim, inner_dim, bias=False)
        self.to_v_tv = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=True),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        # event-synchronous fusion: per-head relative-time penalty gamma*|t_q - t_k|, only used
        # when `times` is given.  gamma = GAMMA_SCALE * raw so it can move on the O(1-10) scale.
        self.gamma_a = nn.Parameter(torch.zeros(heads))
        self.gamma_v = nn.Parameter(torch.zeros(heads))

    GAMMA_SCALE = 10.0

    def _time_bias(self, gamma, t_q, t_k):
        # gamma: (h,), t_q: (b, i), t_k: (b, j) in [0, 1] -> (b, h, i, j)
        g = self.GAMMA_SCALE * gamma
        return -g[None, :, None, None] * (t_q[:, None, :, None] - t_k[:, None, None, :]).abs()

    def forward(self, h_t, h_a, h_v, h_hyper, mask_a=None, mask_v=None, times=None):
        h = self.heads
        q = self.to_q(h_t)
        k_ta, k_tv = self.to_k_ta(h_a), self.to_k_tv(h_v)
        v_ta, v_tv = self.to_v_ta(h_a), self.to_v_tv(h_v)
        q, k_ta, k_tv, v_ta, v_tv = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                                        (q, k_ta, k_tv, v_ta, v_tv))

        bias_a = bias_v = None
        if times is not None:
            t_l, t_a, t_v = times
            bias_a = self._time_bias(self.gamma_a, t_l, t_a).to(q.dtype)
            bias_v = self._time_bias(self.gamma_v, t_l, t_v).to(q.dtype)
        out_ta = rearrange(_sdpa(q, k_ta, v_ta, mask_a, bias_a), 'b h n d -> b n (h d)')
        out_tv = rearrange(_sdpa(q, k_tv, v_tv, mask_v, bias_v), 'b h n d -> b n (h d)')

        return h_hyper + self.to_out(out_ta + out_tv)


class HhyperLearningEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            PreNormAHL(dim, HhyperLearningLayer(dim, heads=heads, dim_head=dim_head, dropout=dropout))
            for _ in range(depth)
        ])

    def forward(self, h_t_list, h_a, h_v, h_hyper, mask_a=None, mask_v=None, times=None):
        for i, layer in enumerate(self.layers):
            h_hyper = layer(h_t_list[i], h_a, h_v, h_hyper, mask_a=mask_a, mask_v=mask_v, times=times)
        return h_hyper


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNormAttention(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNormForward(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]) for _ in range(depth)
        ])

    def forward(self, x, save_hidden=False, mask=None):
        hidden_list = [x]
        for attn, ff in self.layers:
            x = attn(x, x, x, mask=mask) + x
            x = ff(x) + x
            hidden_list.append(x)
        return hidden_list if save_hidden else x


class CrossTransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNormAttention(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNormForward(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]) for _ in range(depth)
        ])

    def forward(self, source_x, target_x, source_mask=None):
        for attn, ff in self.layers:
            target_x = attn(target_x, source_x, source_x, mask=source_mask) + target_x
            target_x = ff(target_x) + target_x
        return target_x


class Transformer(nn.Module):
    """ALMT token-compression transformer: prepends `token_len` learnable tokens."""

    def __init__(self, *, num_frames, token_len, save_hidden, dim, depth, heads, mlp_dim,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.token_len = token_len
        self.save_hidden = save_hidden
        if token_len is not None:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames + token_len, dim))
            self.extra_token = nn.Parameter(torch.zeros(1, token_len, dim))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
            self.extra_token = None
        self.dropout = nn.Dropout(emb_dropout)
        self.encoder = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x, mask=None):
        b, n, _ = x.shape
        if self.token_len is not None:
            extra_token = repeat(self.extra_token, '1 n d -> b n d', b=b)
            x = torch.cat((extra_token, x), dim=1)
            x = x + self.pos_embedding[:, :n + self.token_len]
            if mask is not None:
                mask = torch.cat([torch.ones(b, self.token_len, dtype=torch.bool, device=x.device), mask], 1)
        else:
            x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        return self.encoder(x, self.save_hidden, mask=mask)


class CrossTransformer(nn.Module):
    def __init__(self, *, source_num_frames, tgt_num_frames, dim, depth, heads, mlp_dim,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.pos_embedding_s = nn.Parameter(torch.randn(1, source_num_frames + 1, dim))
        self.pos_embedding_t = nn.Parameter(torch.randn(1, tgt_num_frames + 1, dim))
        self.extra_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.CrossTransformerEncoder = CrossTransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, source_x, target_x):
        b, n_s, _ = source_x.shape
        _, n_t, _ = target_x.shape
        extra_token = repeat(self.extra_token, '1 1 d -> b 1 d', b=b)
        source_x = torch.cat((extra_token, source_x), dim=1) + self.pos_embedding_s[:, :n_s + 1]
        target_x = torch.cat((extra_token, target_x), dim=1) + self.pos_embedding_t[:, :n_t + 1]
        return self.CrossTransformerEncoder(self.dropout(source_x), self.dropout(target_x))
