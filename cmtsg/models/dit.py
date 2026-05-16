from __future__ import annotations

import math

import torch
from torch import nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    while shift.ndim < x.ndim:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    return x * (1.0 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t))


class PatchEmbed1D(nn.Module):
    def __init__(self, n_vars: int, patch_size: int, hidden_size: int) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, hidden_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        bsz, length, n_vars = x.shape
        if n_vars != self.n_vars:
            raise ValueError(f"Expected {self.n_vars} variables, got {n_vars}")
        pad_len = (self.patch_size - length % self.patch_size) % self.patch_size
        if pad_len:
            x = torch.cat([x, torch.zeros(bsz, pad_len, n_vars, device=x.device, dtype=x.dtype)], dim=1)
        patches = x.reshape(bsz, x.shape[1] // self.patch_size, self.patch_size, n_vars)
        patches = patches.permute(0, 3, 1, 2).contiguous()
        return self.proj(patches), pad_len


class FactorizedDiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_temporal = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.temporal_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm_variable = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.variable_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 12 * hidden_size))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, env_context: torch.Tensor) -> torch.Tensor:
        (
            shift_t,
            scale_t,
            gate_t,
            shift_v,
            scale_v,
            gate_v,
            shift_c,
            scale_c,
            gate_c,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(cond).chunk(12, dim=1)
        bsz, n_vars, n_tokens, hidden = x.shape

        temporal_in = modulate(self.norm_temporal(x), shift_t, scale_t).reshape(bsz * n_vars, n_tokens, hidden)
        temporal_out, _ = self.temporal_attn(temporal_in, temporal_in, temporal_in, need_weights=False)
        temporal_out = temporal_out.reshape(bsz, n_vars, n_tokens, hidden)
        x = x + gate_t[:, None, None, :] * temporal_out

        variable_in = modulate(self.norm_variable(x), shift_v, scale_v).permute(0, 2, 1, 3).reshape(bsz * n_tokens, n_vars, hidden)
        variable_out, _ = self.variable_attn(variable_in, variable_in, variable_in, need_weights=False)
        variable_out = variable_out.reshape(bsz, n_tokens, n_vars, hidden).permute(0, 2, 1, 3)
        x = x + gate_v[:, None, None, :] * variable_out

        cross_in = modulate(self.norm_cross(x), shift_c, scale_c).reshape(bsz, n_vars * n_tokens, hidden)
        cross_out, _ = self.cross_attn(cross_in, env_context, env_context, need_weights=False)
        cross_out = cross_out.reshape(bsz, n_vars, n_tokens, hidden)
        x = x + gate_c[:, None, None, :] * cross_out

        mlp_in = modulate(self.norm_mlp(x), shift_mlp, scale_mlp)
        x = x + gate_mlp[:, None, None, :] * self.mlp(mlp_in)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


class TimeSeriesDiT(nn.Module):
    def __init__(
        self,
        seq_len: int,
        n_vars: int,
        text_dim: int,
        env_dim: int,
        hidden_size: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        patch_size: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_vars = n_vars
        self.patch_size = patch_size
        self.x_embedder = PatchEmbed1D(n_vars, patch_size, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.condition_mlp = nn.Sequential(
            nn.Linear(text_dim + env_dim + min(text_dim, env_dim), hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.env_token_proj = nn.Linear(env_dim, hidden_size)
        max_tokens = math.ceil(seq_len / patch_size)
        self.time_pos_embed = nn.Parameter(torch.zeros(1, 1, max_tokens, hidden_size))
        self.var_pos_embed = nn.Parameter(torch.zeros(1, n_vars, 1, hidden_size))
        self.blocks = nn.ModuleList(
            [FactorizedDiTBlock(hidden_size, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size)
        nn.init.normal_(self.time_pos_embed, std=0.02)
        nn.init.normal_(self.var_pos_embed, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_emb: torch.Tensor,
        env_mix: torch.Tensor,
        env_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens, pad_len = self.x_embedder(x)
        tokens = tokens + self.time_pos_embed[:, :, : tokens.shape[2]] + self.var_pos_embed
        prod_dim = min(text_emb.shape[-1], env_mix.shape[-1])
        cond_in = torch.cat([text_emb, env_mix, text_emb[:, :prod_dim] * env_mix[:, :prod_dim]], dim=-1)
        cond = self.t_embedder(t) + self.condition_mlp(cond_in)
        if env_tokens is None:
            env_tokens = env_mix[:, None, :]
        env_context = self.env_token_proj(env_tokens)
        for block in self.blocks:
            tokens = block(tokens, cond, env_context)
        patches = self.final_layer(tokens, cond)
        out = patches.permute(0, 2, 3, 1).contiguous().reshape(x.shape[0], -1, self.n_vars)
        if pad_len:
            out = out[:, :-pad_len]
        return out[:, : self.seq_len]
