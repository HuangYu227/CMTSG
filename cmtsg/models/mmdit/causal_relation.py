from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from cmtsg.models.dit import FinalLayer, PatchEmbed1D, TimestepEmbedder, modulate
from cmtsg.models.grounding import CausalSemanticGrounding


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.weight = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) * self.weight * self.scale


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.to_shift_scale = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim * 2))
        nn.init.zeros_(self.to_shift_scale[-1].weight)
        nn.init.zeros_(self.to_shift_scale[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.to_shift_scale(cond).chunk(2, dim=-1)
        return modulate(self.norm(x), shift, scale)


class TriModalJointAttention(nn.Module):
    """
    MMDiT-style joint attention with modality-specific qkv and output projections.

    The three streams are packed only after each modality has projected its own q/k/v.
    This preserves the MMDiT design choice that text, relation slots, and series
    latents do not share attention parameters before entering the joint attention map.
    """

    def __init__(
        self,
        dim_modalities: tuple[int, int, int],
        num_heads: int,
        dim_head: int | None = None,
        dropout: float = 0.0,
        qk_rmsnorm: bool = True,
        softclamp: bool = True,
        softclamp_value: float = 50.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head or dim_modalities[0] // num_heads
        self.inner_dim = self.num_heads * self.dim_head
        self.scale = self.dim_head**-0.5
        self.softclamp = softclamp
        self.softclamp_value = softclamp_value
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.ModuleList([nn.Linear(dim, self.inner_dim * 3, bias=False) for dim in dim_modalities])
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, dim, bias=False) for dim in dim_modalities])
        self.qk_rmsnorm = qk_rmsnorm
        self.q_norms = nn.ModuleList([MultiHeadRMSNorm(self.dim_head, num_heads) for _ in dim_modalities])
        self.k_norms = nn.ModuleList([MultiHeadRMSNorm(self.dim_head, num_heads) for _ in dim_modalities])
        self.relation_pair_bias = nn.Sequential(
            nn.LayerNorm(dim_modalities[2]),
            nn.Linear(dim_modalities[2], num_heads * len(dim_modalities) * len(dim_modalities)),
        )
        nn.init.zeros_(self.relation_pair_bias[-1].weight)
        nn.init.zeros_(self.relation_pair_bias[-1].bias)

    def _project(self, x: torch.Tensor, proj: nn.Linear, q_norm: nn.Module, k_norm: nn.Module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, n_tokens, _ = x.shape
        qkv = proj(x).reshape(bsz, n_tokens, 3, self.num_heads, self.dim_head)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.unbind(dim=0)
        if self.qk_rmsnorm:
            q = q_norm(q)
            k = k_norm(k)
        return q, k, v

    def forward(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        masks: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if masks is None:
            masks = (None, None, None)
        lengths = [x.shape[1] for x in inputs]
        qkv = [
            self._project(x, proj, q_norm, k_norm)
            for x, proj, q_norm, k_norm in zip(inputs, self.to_qkv, self.q_norms, self.k_norms)
        ]
        q = torch.cat([item[0] for item in qkv], dim=2)
        k = torch.cat([item[1] for item in qkv], dim=2)
        v = torch.cat([item[2] for item in qkv], dim=2)

        attn_logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        pair_bias = self.relation_pair_bias(inputs[2].mean(dim=1))
        pair_bias = pair_bias.reshape(inputs[0].shape[0], self.num_heads, len(inputs), len(inputs))
        q_start = 0
        for q_idx, q_len in enumerate(lengths):
            k_start = 0
            for k_idx, k_len in enumerate(lengths):
                attn_logits[:, :, q_start : q_start + q_len, k_start : k_start + k_len] = (
                    attn_logits[:, :, q_start : q_start + q_len, k_start : k_start + k_len]
                    + pair_bias[:, :, q_idx, k_idx][:, :, None, None]
                )
                k_start += k_len
            q_start += q_len
        if self.softclamp:
            attn_logits = (attn_logits / self.softclamp_value).tanh() * self.softclamp_value
        packed_masks = []
        for x, mask in zip(inputs, masks):
            if mask is None:
                packed_masks.append(torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
            else:
                packed_masks.append(mask)
        key_mask = torch.cat(packed_masks, dim=1)
        attn_logits = attn_logits.masked_fill(~key_mask[:, None, None, :], -torch.finfo(attn_logits.dtype).max)
        attn = attn_logits.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(inputs[0].shape[0], sum(lengths), self.inner_dim)
        outs = out.split(lengths, dim=1)
        return tuple(proj(chunk) for chunk, proj in zip(outs, self.to_out))  # type: ignore[return-value]


class FrequencyLagPositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, n_vars: int, patch_size: int, hidden_size: int, num_bands: int = 16) -> None:
        super().__init__()
        self.max_tokens = math.ceil(seq_len / patch_size)
        self.num_bands = num_bands
        self.time_pos_embed = nn.Parameter(torch.zeros(1, 1, self.max_tokens, hidden_size))
        self.var_pos_embed = nn.Parameter(torch.zeros(1, n_vars, 1, hidden_size))
        self.freq_lag_proj = nn.Sequential(
            nn.Linear(num_bands * 4 + 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.normal_(self.time_pos_embed, std=0.02)
        nn.init.normal_(self.var_pos_embed, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        n_tokens = tokens.shape[2]
        device = tokens.device
        dtype = tokens.dtype
        pos = torch.linspace(0.0, 1.0, n_tokens, device=device, dtype=dtype)
        centered = pos - 0.5
        bands = 2.0 ** torch.arange(self.num_bands, device=device, dtype=dtype)
        args = 2.0 * math.pi * pos[:, None] * bands[None]
        lag_args = 2.0 * math.pi * centered.abs()[:, None] * bands[None]
        features = torch.cat(
            [
                pos[:, None],
                centered.abs()[:, None],
                torch.sin(args),
                torch.cos(args),
                torch.sin(lag_args),
                torch.cos(lag_args),
            ],
            dim=-1,
        )
        spectral_pos = self.freq_lag_proj(features).reshape(1, 1, n_tokens, -1)
        return tokens + self.time_pos_embed[:, :, :n_tokens] + self.var_pos_embed + spectral_pos


class SeriesStructureBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_temporal = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.temporal_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm_variable = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.variable_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.to_gates = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6))
        nn.init.zeros_(self.to_gates[-1].weight)
        nn.init.zeros_(self.to_gates[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_t, scale_t, gate_t, shift_v, scale_v, gate_v = self.to_gates(cond).chunk(6, dim=-1)
        bsz, n_vars, n_tokens, hidden = x.shape
        temporal_in = modulate(self.norm_temporal(x), shift_t, scale_t).reshape(bsz * n_vars, n_tokens, hidden)
        temporal_out, _ = self.temporal_attn(temporal_in, temporal_in, temporal_in, need_weights=False)
        x = x + gate_t[:, None, None, :] * temporal_out.reshape(bsz, n_vars, n_tokens, hidden)

        variable_in = modulate(self.norm_variable(x), shift_v, scale_v).permute(0, 2, 1, 3).reshape(bsz * n_tokens, n_vars, hidden)
        variable_out, _ = self.variable_attn(variable_in, variable_in, variable_in, need_weights=False)
        variable_out = variable_out.reshape(bsz, n_tokens, n_vars, hidden).permute(0, 2, 1, 3)
        return x + gate_v[:, None, None, :] * variable_out


class SemanticTokenAdapter(nn.Module):
    def __init__(self, text_dim: int, hidden_size: int, num_tokens: int) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.token_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_size * num_tokens),
        )
        self.atom_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_size),
        )
        self.token_type = nn.Parameter(torch.randn(1, num_tokens, hidden_size) * 0.02)

    def forward(self, text_emb: torch.Tensor) -> torch.Tensor:
        if text_emb.ndim == 2:
            tokens = self.token_proj(text_emb).reshape(text_emb.shape[0], self.num_tokens, -1)
        elif text_emb.ndim == 3:
            tokens = self.atom_proj(text_emb)
        else:
            raise ValueError(f"Expected text embedding [B,D] or atom embeddings [B,C,D], got {tuple(text_emb.shape)}")
        token_type = self.token_type.to(device=tokens.device, dtype=tokens.dtype)
        if tokens.shape[1] <= token_type.shape[1]:
            token_type = token_type[:, : tokens.shape[1]]
        else:
            token_type = F.interpolate(
                token_type.transpose(1, 2),
                size=tokens.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return tokens + token_type


class SeriesRegisterResampler(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_registers: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_registers = num_registers
        self.register_tokens = nn.Parameter(torch.randn(1, num_registers, hidden_size) * 0.02)
        self.norm_register = AdaptiveLayerNorm(hidden_size, hidden_size)
        self.norm_series = AdaptiveLayerNorm(hidden_size, hidden_size)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm_ff = AdaptiveLayerNorm(hidden_size, hidden_size)
        self.ff = FeedForward(hidden_size, mlp_ratio, dropout)
        self.to_gates = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 2))
        nn.init.zeros_(self.to_gates[-1].weight)
        nn.init.zeros_(self.to_gates[-1].bias)

    def forward(self, series_tokens: torch.Tensor, cond: torch.Tensor, registers: torch.Tensor | None = None) -> torch.Tensor:
        bsz, n_vars, n_tokens, hidden = series_tokens.shape
        series_flat = series_tokens.reshape(bsz, n_vars * n_tokens, hidden)
        if registers is None:
            registers = self.register_tokens.to(device=series_tokens.device, dtype=series_tokens.dtype).expand(bsz, -1, -1)
        gate_attn, gate_ff = self.to_gates(cond).chunk(2, dim=-1)
        query = self.norm_register(registers, cond)
        memory = self.norm_series(series_flat, cond)
        attn_out, _ = self.cross_attn(query, memory, memory, need_weights=False)
        registers = registers + gate_attn[:, None, :] * attn_out
        registers = registers + gate_ff[:, None, :] * self.ff(self.norm_ff(registers, cond))
        return registers


class SeriesContextFeedback(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_series = AdaptiveLayerNorm(hidden_size, hidden_size)
        self.norm_context = AdaptiveLayerNorm(hidden_size, hidden_size)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm_ff = AdaptiveLayerNorm(hidden_size, hidden_size)
        self.ff = FeedForward(hidden_size, mlp_ratio, dropout)
        self.to_gates = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 2))
        nn.init.zeros_(self.to_gates[-1].weight)
        nn.init.zeros_(self.to_gates[-1].bias)

    def forward(self, series_tokens: torch.Tensor, context_tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        bsz, n_vars, n_tokens, hidden = series_tokens.shape
        series_flat = series_tokens.reshape(bsz, n_vars * n_tokens, hidden)
        gate_attn, gate_ff = self.to_gates(cond).chunk(2, dim=-1)
        query = self.norm_series(series_flat, cond)
        context = self.norm_context(context_tokens, cond)
        attn_out, _ = self.cross_attn(query, context, context, need_weights=False)
        series_flat = series_flat + gate_attn[:, None, :] * attn_out
        series_flat = series_flat + gate_ff[:, None, :] * self.ff(self.norm_ff(series_flat, cond))
        return series_flat.reshape(bsz, n_vars, n_tokens, hidden)


class TriModalMMDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        dim_head: int | None = None,
        qk_rmsnorm: bool = True,
        softclamp: bool = True,
        series_register_tokens: int = 64,
    ) -> None:
        super().__init__()
        self.series_structure = SeriesStructureBlock(hidden_size, num_heads, dropout)
        self.series_resampler = SeriesRegisterResampler(
            hidden_size,
            num_heads,
            num_registers=series_register_tokens,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.attn_norms = nn.ModuleList([AdaptiveLayerNorm(hidden_size, hidden_size) for _ in range(3)])
        self.joint_attn = TriModalJointAttention(
            (hidden_size, hidden_size, hidden_size),
            num_heads=num_heads,
            dim_head=dim_head,
            dropout=dropout,
            qk_rmsnorm=qk_rmsnorm,
            softclamp=softclamp,
        )
        self.ff_norms = nn.ModuleList([AdaptiveLayerNorm(hidden_size, hidden_size) for _ in range(3)])
        self.ff = nn.ModuleList([FeedForward(hidden_size, mlp_ratio, dropout) for _ in range(3)])
        self.attn_gates = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 3))
        self.ff_gates = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 3))
        nn.init.zeros_(self.attn_gates[-1].weight)
        nn.init.zeros_(self.attn_gates[-1].bias)
        nn.init.zeros_(self.ff_gates[-1].weight)
        nn.init.zeros_(self.ff_gates[-1].bias)

    def forward(
        self,
        series_tokens: torch.Tensor,
        series_registers: torch.Tensor | None,
        semantic_tokens: torch.Tensor,
        relation_tokens: torch.Tensor,
        cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        series_tokens = self.series_structure(series_tokens, cond)
        series_registers = self.series_resampler(series_tokens, cond, series_registers)
        streams = (series_registers, semantic_tokens, relation_tokens)

        attn_in = tuple(norm(tokens, cond) for tokens, norm in zip(streams, self.attn_norms))
        attn_out = self.joint_attn(attn_in)  # type: ignore[arg-type]
        attn_gates = self.attn_gates(cond).chunk(3, dim=-1)
        streams = tuple(tokens + gate[:, None, :] * out for tokens, gate, out in zip(streams, attn_gates, attn_out))

        ff_in = tuple(norm(tokens, cond) for tokens, norm in zip(streams, self.ff_norms))
        ff_out = tuple(ff(tokens) for tokens, ff in zip(ff_in, self.ff))
        ff_gates = self.ff_gates(cond).chunk(3, dim=-1)
        streams = tuple(tokens + gate[:, None, :] * out for tokens, gate, out in zip(streams, ff_gates, ff_out))
        series_registers, semantic_tokens, relation_tokens = streams
        return series_tokens, series_registers, semantic_tokens, relation_tokens


class CausalRelationMMDiTDenoiser(nn.Module):
    def __init__(
        self,
        seq_len: int,
        n_vars: int,
        text_dim: int,
        env_dim: int,
        hidden_size: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        patch_size: int = 1,
        dropout: float = 0.0,
        text_tokens: int = 4,
        series_register_tokens: int = 64,
        dim_head: int | None = None,
        qk_rmsnorm: bool = True,
        softclamp: bool = True,
        use_semantic_grounding: bool = True,
        grounding_num_heads: int = 4,
        grounding_sinkhorn_iters: int = 24,
        grounding_ot_temperature: float = 0.07,
        grounding_mask_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_vars = n_vars
        self.patch_size = patch_size
        self.use_semantic_grounding = use_semantic_grounding
        self.x_embedder = PatchEmbed1D(n_vars, patch_size, hidden_size)
        self.pos_embedder = FrequencyLagPositionalEncoding(seq_len, n_vars, patch_size, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.condition_mlp = nn.Sequential(
            nn.Linear(text_dim + env_dim + min(text_dim, env_dim), hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.semantic_adapter = SemanticTokenAdapter(text_dim, hidden_size, text_tokens)
        self.relation_proj = nn.Sequential(nn.LayerNorm(env_dim), nn.Linear(env_dim, hidden_size))
        if use_semantic_grounding:
            self.semantic_grounding = CausalSemanticGrounding(
                dim=hidden_size,
                num_heads=grounding_num_heads,
                sinkhorn_iters=grounding_sinkhorn_iters,
                ot_temperature=grounding_ot_temperature,
                mask_temperature=grounding_mask_temperature,
            )
        else:
            self.semantic_grounding = None
        self.blocks = nn.ModuleList(
            [
                TriModalMMDiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    dim_head=dim_head,
                    qk_rmsnorm=qk_rmsnorm,
                    softclamp=softclamp,
                    series_register_tokens=series_register_tokens,
                )
                for _ in range(depth)
            ]
        )
        self.context_feedback = SeriesContextFeedback(hidden_size, num_heads, mlp_ratio, dropout)
        self.final_layer = FinalLayer(hidden_size, patch_size)
        self.series_summary_norm = nn.LayerNorm(hidden_size)
        self.semantic_summary_norm = nn.LayerNorm(hidden_size)
        self.relation_summary_norm = nn.LayerNorm(hidden_size)

    def _condition(self, t: torch.Tensor, text_emb: torch.Tensor, env_mix: torch.Tensor) -> torch.Tensor:
        prod_dim = min(text_emb.shape[-1], env_mix.shape[-1])
        cond_in = torch.cat([text_emb, env_mix, text_emb[:, :prod_dim] * env_mix[:, :prod_dim]], dim=-1)
        return self.t_embedder(t) + self.condition_mlp(cond_in)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_emb: torch.Tensor,
        env_mix: torch.Tensor,
        env_tokens: torch.Tensor,
        semantic_text_emb: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        series_tokens, pad_len = self.x_embedder(x)
        series_tokens = self.pos_embedder(series_tokens)
        cond = self._condition(t, text_emb, env_mix)
        semantic_tokens = self.semantic_adapter(text_emb if semantic_text_emb is None else semantic_text_emb)
        relation_tokens = self.relation_proj(env_tokens)
        series_registers = None
        grounding_outputs: dict[str, torch.Tensor] = {}
        grounding_losses: dict[str, torch.Tensor] = {}

        if self.semantic_grounding is not None:
            grounding_features = series_tokens.permute(0, 3, 1, 2).contiguous()
            grounding_outputs, grounding_losses = self.semantic_grounding(
                semantic_tokens,
                relation_tokens,
                grounding_features,
            )
            semantic_tokens = grounding_outputs["semantic_atoms"]
            relation_tokens = grounding_outputs["relation_slots"]

        for block in self.blocks:
            series_tokens, series_registers, semantic_tokens, relation_tokens = block(
                series_tokens,
                series_registers,
                semantic_tokens,
                relation_tokens,
                cond,
            )

        if series_registers is None:
            raise RuntimeError("CausalRelationMMDiTDenoiser produced no series registers")
        context_tokens = torch.cat([series_registers, semantic_tokens, relation_tokens], dim=1)
        series_tokens = self.context_feedback(series_tokens, context_tokens, cond)
        patches = self.final_layer(series_tokens, cond)
        out = patches.permute(0, 2, 3, 1).contiguous().reshape(x.shape[0], -1, self.n_vars)
        if pad_len:
            out = out[:, :-pad_len]
        out = out[:, : self.seq_len]
        if not return_aux:
            return out
        aux = {
            "series_summary": self.series_summary_norm(series_registers.mean(dim=1)),
            "semantic_summary": self.semantic_summary_norm(semantic_tokens.mean(dim=1)),
            "relation_summary": self.relation_summary_norm(relation_tokens.mean(dim=1)),
        }
        if grounding_losses:
            aux.update(
                {
                    "grounding_loss_ot": grounding_losses["loss_ot"],
                    "grounding_loss_mask": grounding_losses["loss_ground"],
                    "grounding_loss_cycle": grounding_losses["loss_cycle"],
                    "grounding_transport_entropy": grounding_losses["transport_entropy"],
                    "grounding_transport_row_error": grounding_losses["transport_row_error"],
                    "grounding_transport_col_error": grounding_losses["transport_col_error"],
                }
            )
        return out, aux
