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
        n_env: int = 12,
        env_source: str = "gaf",
        routing: str = "text",
        hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        if env_source not in {"gaf", "learned"}:
            raise ValueError(f"Unsupported env_source: {env_source}")
        if routing not in {"text", "uniform", "learned_query"}:
            raise ValueError(f"Unsupported routing: {routing}")
        self.n_env = n_env
        self.env_source = env_source
        self.routing = routing
        self.output_dim = n_vars * gaf_size
        self.encoder = GAFDepthwisePointwiseEncoder(n_vars, gaf_size, hidden_channels) if env_source == "gaf" else None
        self.learned_env_raw = nn.Parameter(torch.randn(n_env, self.output_dim) * 0.02) if env_source == "learned" else None
        self.learned_query = nn.Parameter(torch.randn(1, text_dim) * 0.02) if routing == "learned_query" else None
        self.key = nn.Linear(self.output_dim, text_dim)
        self.value = nn.Linear(self.output_dim, env_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        self.key_norm = nn.LayerNorm(text_dim)

    def forward(self, text_emb: torch.Tensor, anchor_gaf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.env_source == "gaf":
            if self.encoder is None:
                raise RuntimeError("GAF environment source requires an encoder")
            env_raw = self.encoder(anchor_gaf)
        else:
            if self.learned_env_raw is None:
                raise RuntimeError("Learned environment source requires learned_env_raw")
            env_raw = self.learned_env_raw
        keys = self.key_norm(self.key(env_raw))
        values = self.value(env_raw)
        if self.routing == "uniform":
            alpha = torch.full(
                (text_emb.shape[0], env_raw.shape[0]),
                1.0 / float(env_raw.shape[0]),
                device=text_emb.device,
                dtype=text_emb.dtype,
            )
        else:
            if self.routing == "learned_query":
                if self.learned_query is None:
                    raise RuntimeError("learned_query routing requires learned_query")
                query = self.learned_query.expand(text_emb.shape[0], -1)
            else:
                query = text_emb
            query = self.text_norm(query)
            scores = query @ keys.t() / math.sqrt(query.shape[-1])
            alpha = F.softmax(scores, dim=-1)
        env_mix = alpha @ values
        return env_mix, alpha, env_raw
