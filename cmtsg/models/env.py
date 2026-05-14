from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class GAFDepthwisePointwiseEncoder(nn.Module):
    def __init__(self, n_vars: int, gaf_size: int, hidden_channels: int | None = None) -> None:
        super().__init__()
        hidden_channels = hidden_channels or n_vars
        self.n_vars = n_vars
        self.gaf_size = gaf_size
        self.depthwise = nn.Sequential(
            nn.Conv2d(n_vars, n_vars, kernel_size=3, padding=1, groups=n_vars),
            nn.GELU(),
            nn.Conv2d(n_vars, n_vars, kernel_size=3, padding=1, groups=n_vars),
            nn.GELU(),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(n_vars, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, n_vars, kernel_size=1),
        )

    @property
    def output_dim(self) -> int:
        return self.n_vars * self.gaf_size

    def forward(self, gaf: torch.Tensor) -> torch.Tensor:
        if gaf.ndim != 4:
            raise ValueError(f"Expected GAF shape [B,K,M,M], got {tuple(gaf.shape)}")
        h = self.depthwise(gaf)
        h = self.pointwise(h)
        h = h.mean(dim=-1)
        return h.reshape(h.shape[0], -1)


class EnvironmentRouter(nn.Module):
    def __init__(
        self,
        n_vars: int,
        gaf_size: int,
        text_dim: int,
        env_dim: int,
        hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = GAFDepthwisePointwiseEncoder(n_vars, gaf_size, hidden_channels)
        self.key = nn.Linear(self.encoder.output_dim, text_dim)
        self.value = nn.Linear(self.encoder.output_dim, env_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        self.key_norm = nn.LayerNorm(text_dim)

    def forward(self, text_emb: torch.Tensor, anchor_gaf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        env_raw = self.encoder(anchor_gaf)
        keys = self.key_norm(self.key(env_raw))
        values = self.value(env_raw)
        query = self.text_norm(text_emb)
        scores = query @ keys.t() / math.sqrt(query.shape[-1])
        alpha = F.softmax(scores, dim=-1)
        env_mix = alpha @ values
        return env_mix, alpha, env_raw
