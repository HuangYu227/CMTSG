from __future__ import annotations

import torch
from torch import nn

from cmtsg.models.dit import TimeSeriesDiT
from cmtsg.models.env import EnvironmentRouter


class CMTSGModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        n_vars: int,
        gaf_size: int,
        text_dim: int = 64,
        env_dim: int = 64,
        hidden_size: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        patch_size: int = 1,
        dropout: float = 0.0,
        n_env: int = 12,
        use_text_condition: bool = True,
        use_env_condition: bool = True,
        env_source: str = "gaf",
        routing: str = "text",
    ) -> None:
        super().__init__()
        self.n_env = n_env
        self.text_dim = text_dim
        self.env_dim = env_dim
        self.use_text_condition = use_text_condition
        self.use_env_condition = use_env_condition
        self.router = EnvironmentRouter(
            n_vars,
            gaf_size,
            text_dim,
            env_dim,
            n_env=n_env,
            env_source=env_source,
            routing=routing,
        )
        self.dit = TimeSeriesDiT(
            seq_len=seq_len,
            n_vars=n_vars,
            text_dim=text_dim,
            env_dim=env_dim,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, text_emb: torch.Tensor, anchor_gaf: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        router_text = text_emb if self.use_text_condition else torch.zeros_like(text_emb)
        text_for_dit = text_emb if self.use_text_condition else torch.zeros_like(text_emb)
        if self.use_env_condition:
            env_mix, alpha, env_raw = self.router(router_text, anchor_gaf)
        else:
            env_mix = torch.zeros(text_emb.shape[0], self.env_dim, device=text_emb.device, dtype=text_emb.dtype)
            alpha = torch.full(
                (text_emb.shape[0], self.n_env),
                1.0 / float(self.n_env),
                device=text_emb.device,
                dtype=text_emb.dtype,
            )
            env_raw = torch.zeros(self.n_env, self.router.output_dim, device=text_emb.device, dtype=text_emb.dtype)
        pred = self.dit(x, t, text_for_dit, env_mix)
        entropy = -(alpha * alpha.clamp_min(1e-8).log()).sum(dim=-1).mean()
        aux = {
            "alpha": alpha,
            "env_raw": env_raw,
            "route_entropy": entropy,
            "route_max": alpha.max(dim=-1).values.mean(),
        }
        return pred, aux
