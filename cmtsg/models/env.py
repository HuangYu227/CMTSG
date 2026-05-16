from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class SpatioTemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, expansion: int = 4) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same-size depthwise convolution")
        hidden = channels * expansion
        self.depthwise = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=channels),
            nn.BatchNorm2d(channels),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x + residual


class GAFDepthwisePointwiseEncoder(nn.Module):
    def __init__(
        self,
        n_vars: int,
        gaf_size: int,
        hidden_channels: int | None = None,
        num_blocks: int = 4,
        kernel_size: int = 7,
        expansion: int = 4,
        pool_size: int = 4,
    ) -> None:
        super().__init__()
        if num_blocks < 3 or num_blocks > 6:
            raise ValueError("num_blocks must be in [3, 6]")
        hidden_channels = hidden_channels or 64
        self.n_vars = n_vars
        self.gaf_size = gaf_size
        self.hidden_channels = hidden_channels
        self.pool_size = pool_size
        self.stem = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                SpatioTemporalBlock(hidden_channels, kernel_size=kernel_size, expansion=expansion)
                for _ in range(num_blocks)
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))

    @property
    def output_dim(self) -> int:
        return self.hidden_channels * self.pool_size * self.pool_size

    def forward(self, gaf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if gaf.ndim != 4:
            raise ValueError(f"Expected GAF shape [B,K,M,M], got {tuple(gaf.shape)}")
        if gaf.shape[1] != self.n_vars:
            raise ValueError(f"Expected {self.n_vars} GAF variable channels, got {gaf.shape[1]}")
        bsz, n_vars, height, width = gaf.shape
        x = gaf.reshape(bsz * n_vars, 1, height, width)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        tokens = x.flatten(start_dim=1).reshape(bsz, n_vars, -1)
        return tokens, tokens.mean(dim=1)


class GADFRelationSlotEncoder(nn.Module):
    def __init__(
        self,
        n_vars: int,
        gaf_size: int,
        env_dim: int,
        n_env: int = 12,
        hidden_channels: int | None = None,
        num_blocks: int = 4,
        kernel_size: int = 7,
        expansion: int = 4,
        pool_size: int = 4,
    ) -> None:
        super().__init__()
        self.local_encoder = GAFDepthwisePointwiseEncoder(
            n_vars,
            gaf_size,
            hidden_channels,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            expansion=expansion,
            pool_size=pool_size,
        )
        self.n_env = n_env
        self.env_dim = env_dim
        output_dim = self.local_encoder.output_dim
        heads = 4 if output_dim % 4 == 0 else 1
        self.variable_mixer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=output_dim,
                nhead=heads,
                dim_feedforward=output_dim * 2,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=1,
        )
        self.slot_queries = nn.Parameter(torch.randn(n_env, output_dim) * 0.02)
        self.slot_pool = nn.MultiheadAttention(output_dim, heads, batch_first=True)
        self.slot_norm = nn.LayerNorm(output_dim)
        self.slot_value = nn.Linear(output_dim, env_dim)
        self.out_norm = nn.LayerNorm(env_dim)

    @property
    def output_dim(self) -> int:
        return self.local_encoder.output_dim

    def forward(self, gaf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        env_tokens_raw, _ = self.local_encoder(gaf)
        env_tokens = self.variable_mixer(env_tokens_raw)
        queries = self.slot_queries.to(device=env_tokens.device, dtype=env_tokens.dtype)
        queries = queries.unsqueeze(0).expand(env_tokens.shape[0], -1, -1)
        pooled, _ = self.slot_pool(queries, env_tokens, env_tokens, need_weights=False)
        env_slots = self.out_norm(self.slot_value(self.slot_norm(pooled)))
        env_raw = env_tokens.mean(dim=1)
        return env_slots, env_raw, env_tokens


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
        env_slot_mode: str = "dynamic_gaf",
        hidden_channels: int | None = None,
        num_blocks: int = 4,
        kernel_size: int = 7,
        expansion: int = 4,
        pool_size: int = 4,
    ) -> None:
        super().__init__()
        if env_source not in {"gaf", "learned"}:
            raise ValueError(f"Unsupported env_source: {env_source}")
        if routing not in {"text", "uniform", "learned_query"}:
            raise ValueError(f"Unsupported routing: {routing}")
        if env_source == "gaf" and env_slot_mode != "dynamic_gaf":
            raise ValueError(f"GAF environments require env_slot_mode='dynamic_gaf', got {env_slot_mode}")
        self.n_env = n_env
        self.n_vars = n_vars
        self.env_dim = env_dim
        self.env_source = env_source
        self.routing = routing
        self.env_slot_mode = env_slot_mode
        self.encoder = (
            GADFRelationSlotEncoder(
                n_vars,
                gaf_size,
                env_dim,
                n_env=n_env,
                hidden_channels=hidden_channels,
                num_blocks=num_blocks,
                kernel_size=kernel_size,
                expansion=expansion,
                pool_size=pool_size,
            )
            if env_source == "gaf"
            else None
        )
        self.output_dim = self.encoder.output_dim if self.encoder is not None else n_vars * gaf_size
        self.learned_env_raw = nn.Parameter(torch.randn(n_env, self.output_dim) * 0.02) if env_source == "learned" else None
        self.learned_query = nn.Parameter(torch.randn(1, text_dim) * 0.02) if routing == "learned_query" else None
        self.text_slot_projector = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, n_env * env_dim),
        )
        self.query = nn.Linear(text_dim, env_dim)
        self.query_norm = nn.LayerNorm(env_dim)
        self.slot_norm = nn.LayerNorm(env_dim)
        self.value = nn.Linear(self.output_dim, env_dim)
        self.text_norm = nn.LayerNorm(text_dim)

    def _route_slots(self, text_emb: torch.Tensor, env_slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.routing == "uniform":
            query = self.query_norm(self.query(self.text_norm(text_emb)))
            alpha = torch.full(
                (text_emb.shape[0], self.n_env),
                1.0 / float(self.n_env),
                device=text_emb.device,
                dtype=text_emb.dtype,
            )
            return alpha, query, self.slot_norm(env_slots)

        if self.routing == "learned_query":
            if self.learned_query is None:
                raise RuntimeError("learned_query routing requires learned_query")
            query_text = self.learned_query.expand(text_emb.shape[0], -1)
        else:
            query_text = text_emb
        query = self.query_norm(self.query(self.text_norm(query_text)))
        slot_keys = self.slot_norm(env_slots)
        scores = (slot_keys * query[:, None, :]).sum(dim=-1) / math.sqrt(slot_keys.shape[-1])
        alpha = F.softmax(scores, dim=-1)
        return alpha, query, slot_keys

    def _slot_aux_losses(self, env_slots: torch.Tensor, alpha: torch.Tensor, query: torch.Tensor, env_mix: torch.Tensor) -> dict[str, torch.Tensor]:
        normalized_slots = F.normalize(env_slots, dim=-1)
        slot_cos = normalized_slots @ normalized_slots.transpose(1, 2)
        if self.n_env > 1:
            off_diag_mask = ~torch.eye(self.n_env, device=env_slots.device, dtype=torch.bool)
            off_diag = slot_cos[:, off_diag_mask]
            slot_diversity_loss = off_diag.pow(2).mean()
            slot_cosine_mean = off_diag.abs().mean()
        else:
            slot_diversity_loss = env_slots.new_zeros(())
            slot_cosine_mean = env_slots.new_zeros(())

        route_entropy = -(alpha * alpha.clamp_min(1e-8).log()).sum(dim=-1).mean()
        max_entropy = math.log(float(self.n_env)) if self.n_env > 1 else 1.0
        route_entropy_loss = (1.0 - route_entropy / max_entropy).clamp_min(0.0)

        text_slot_align_loss = 1.0 - F.cosine_similarity(query, env_mix, dim=-1).mean()
        return {
            "slot_diversity_loss": slot_diversity_loss,
            "route_entropy_loss": route_entropy_loss,
            "text_slot_align_loss": text_slot_align_loss,
            "slot_cosine_mean": slot_cosine_mean.detach(),
        }

    def _text_env_slots(self, text_emb: torch.Tensor) -> torch.Tensor:
        env_slots = self.text_slot_projector(text_emb).reshape(text_emb.shape[0], self.n_env, self.env_dim)
        return self.slot_norm(env_slots)

    def encode_gaf_slots(self, gaf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.encoder is None:
            raise RuntimeError("GAF environment source requires an encoder")
        env_slots, env_raw, env_tokens = self.encoder(gaf)
        return self.slot_norm(env_slots), env_raw, env_tokens

    def forward(self, text_emb: torch.Tensor, gaf: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self.env_source == "gaf":
            text_env_slots = self._text_env_slots(text_emb)
            if gaf is None:
                env_slots = text_env_slots
                env_raw = text_emb.new_zeros(text_emb.shape[0], self.output_dim)
                alpha, query, _ = self._route_slots(text_emb, env_slots)
                env_mix = torch.einsum("bn,bnd->bd", alpha, env_slots)
                router_aux = self._slot_aux_losses(env_slots, alpha, query, env_mix)
                router_aux["text_env_slot_loss"] = env_mix.new_zeros(())
                return env_mix, alpha, env_raw, env_slots, router_aux
            env_slots, env_raw, _ = self.encode_gaf_slots(gaf)
            alpha, query, _ = self._route_slots(text_emb, env_slots)
            env_mix = torch.einsum("bn,bnd->bd", alpha, env_slots)
            router_aux = self._slot_aux_losses(env_slots, alpha, query, env_mix)
            text_slot_cos = F.cosine_similarity(text_env_slots, env_slots.detach(), dim=-1)
            router_aux["text_env_slot_loss"] = 1.0 - text_slot_cos.mean()
            return env_mix, alpha, env_raw, env_slots, router_aux

        if self.learned_env_raw is None:
            raise RuntimeError("Learned environment source requires learned_env_raw")
        env_raw = self.learned_env_raw
        values = self.slot_norm(self.value(env_raw))
        env_slots = values.unsqueeze(0).expand(text_emb.shape[0], -1, -1)
        alpha, query, _ = self._route_slots(text_emb, env_slots)
        env_mix = torch.einsum("bn,bnd->bd", alpha, env_slots)
        router_aux = self._slot_aux_losses(env_slots, alpha, query, env_mix)
        router_aux["text_env_slot_loss"] = env_mix.new_zeros(())
        return env_mix, alpha, env_raw, env_slots, router_aux
