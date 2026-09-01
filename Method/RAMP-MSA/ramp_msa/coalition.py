from __future__ import annotations

import torch
import torch.nn as nn


COALITION_NAMES = ["T", "A", "V", "TA", "TV", "AV", "TAV"]
COALITION_MASKS = torch.tensor(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=torch.bool,
)


class CoalitionFusion(nn.Module):
    """Shared set-fusion network evaluated on all 7 non-empty T/A/V coalitions.

    Input is a small bank of latent tokens per modality: [B, 3, K, D].
    Sharing exactly the same fusion function across subsets makes the seven
    outputs a coherent set function for Möbius decomposition.
    """

    def __init__(
        self,
        d_model: int,
        num_modality_tokens: int = 4,
        nhead: int = 4,
        num_layers: int = 2,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_modality_tokens = int(num_modality_tokens)
        self.modality_embedding = nn.Parameter(torch.randn(3, 1, d_model) * 0.02)
        self.latent_embedding = nn.Parameter(torch.randn(1, self.num_modality_tokens, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
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
        self.register_buffer("coalition_masks", COALITION_MASKS.clone(), persistent=False)

    def forward(self, modality_tokens: torch.Tensor) -> torch.Tensor:
        # modality_tokens: [B,3,K,D]
        if modality_tokens.ndim != 4:
            raise ValueError(f"Expected [B,3,K,D], got {tuple(modality_tokens.shape)}")
        b, nmod, k, d = modality_tokens.shape
        if nmod != 3 or k != self.num_modality_tokens:
            raise ValueError(f"Expected 3 modalities and K={self.num_modality_tokens}, got {tuple(modality_tokens.shape)}")
        c = self.coalition_masks.size(0)

        tokens = modality_tokens + self.modality_embedding[None, :, :, :] + self.latent_embedding[None, :, :, :]
        tokens = tokens[:, None].expand(b, c, 3, k, d).reshape(b, c, 3 * k, d)
        cls = self.cls_token.expand(b * c, 1, d).reshape(b, c, 1, d)
        tokens = torch.cat([cls, tokens], dim=2).reshape(b * c, 1 + 3 * k, d)

        active = self.coalition_masks[None, :, :, None].expand(b, c, 3, k).reshape(b, c, 3 * k)
        pad_mask = torch.cat(
            [torch.zeros(b, c, 1, dtype=torch.bool, device=active.device), ~active], dim=-1
        ).reshape(b * c, 1 + 3 * k)

        out = self.encoder(tokens, src_key_padding_mask=pad_mask)
        return self.norm(out[:, 0]).reshape(b, c, d)
