from __future__ import annotations

import math

import torch
from torch import nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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
        self.proj = nn.Linear(n_vars * patch_size, hidden_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        bsz, length, n_vars = x.shape
        if n_vars != self.n_vars:
            raise ValueError(f"Expected {self.n_vars} variables, got {n_vars}")
        pad_len = (self.patch_size - length % self.patch_size) % self.patch_size
        if pad_len:
            x = torch.cat([x, torch.zeros(bsz, pad_len, n_vars, device=x.device, dtype=x.dtype)], dim=1)
        patches = x.reshape(bsz, x.shape[1] // self.patch_size, self.patch_size * n_vars)
        return self.proj(patches), pad_len


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=1)
        attn_in = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out
        mlp_in = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_in)
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
        max_tokens = math.ceil(seq_len / patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, hidden_size))
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size * n_vars)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, t: torch.Tensor, text_emb: torch.Tensor, env_mix: torch.Tensor) -> torch.Tensor:
        tokens, pad_len = self.x_embedder(x)
        tokens = tokens + self.pos_embed[:, : tokens.shape[1]]
        prod_dim = min(text_emb.shape[-1], env_mix.shape[-1])
        cond_in = torch.cat([text_emb, env_mix, text_emb[:, :prod_dim] * env_mix[:, :prod_dim]], dim=-1)
        cond = self.t_embedder(t) + self.condition_mlp(cond_in)
        for block in self.blocks:
            tokens = block(tokens, cond)
        patches = self.final_layer(tokens, cond)
        out = patches.reshape(x.shape[0], -1, self.patch_size, self.n_vars).reshape(x.shape[0], -1, self.n_vars)
        if pad_len:
            out = out[:, :-pad_len]
        return out[:, : self.seq_len]
