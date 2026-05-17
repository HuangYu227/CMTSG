from __future__ import annotations

import torch
from torch import nn

from cmtsg.models.dit import TimeSeriesDiT
from cmtsg.models.env import EnvironmentRouter
from cmtsg.models.mmdit import CausalRelationMMDiTDenoiser


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
        env_hidden_channels: int | None = None,
        env_num_blocks: int = 4,
        env_kernel_size: int = 7,
        env_expansion: int = 4,
        env_pool_size: int = 4,
        env_slot_mode: str = "dynamic_gaf",
        architecture: str = "causal_relation_mmdit",
        text_tokens: int = 4,
        series_register_tokens: int = 64,
        mmdit_dim_head: int | None = None,
        qk_rmsnorm: bool = True,
        softclamp: bool = True,
        use_semantic_grounding: bool = True,
        grounding_num_heads: int = 4,
        grounding_sinkhorn_iters: int = 24,
        grounding_ot_temperature: float = 0.07,
        grounding_mask_temperature: float = 1.0,
        grounding_ot_weight: float = 0.01,
        grounding_mask_weight: float = 0.01,
        grounding_cycle_weight: float = 0.01,
        slot_diversity_weight: float = 0.01,
        route_entropy_weight: float = 0.001,
        text_slot_align_weight: float = 0.01,
        text_env_slot_weight: float = 0.02,
        text_drop_prob: float = 0.0,
        env_drop_prob: float = 0.0,
        semantic_drop_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_env = n_env
        self.text_dim = text_dim
        self.env_dim = env_dim
        self.use_text_condition = use_text_condition
        self.use_env_condition = use_env_condition
        self.architecture = architecture
        self.slot_diversity_weight = float(slot_diversity_weight)
        self.route_entropy_weight = float(route_entropy_weight)
        self.text_slot_align_weight = float(text_slot_align_weight)
        self.text_env_slot_weight = float(text_env_slot_weight)
        self.grounding_ot_weight = float(grounding_ot_weight)
        self.grounding_mask_weight = float(grounding_mask_weight)
        self.grounding_cycle_weight = float(grounding_cycle_weight)
        self.route_entropy_scale = 1.0
        self.text_drop_prob = float(text_drop_prob)
        self.env_drop_prob = float(env_drop_prob)
        self.semantic_drop_prob = float(semantic_drop_prob)
        self.null_text_emb = nn.Parameter(torch.randn(1, text_dim) * 0.02)
        self.null_env_mix = nn.Parameter(torch.randn(1, env_dim) * 0.02)
        self.router = EnvironmentRouter(
            n_vars,
            gaf_size,
            text_dim,
            env_dim,
            n_env=n_env,
            env_source=env_source,
            routing=routing,
            env_slot_mode=env_slot_mode,
            hidden_channels=env_hidden_channels,
            num_blocks=env_num_blocks,
            kernel_size=env_kernel_size,
            expansion=env_expansion,
            pool_size=env_pool_size,
        )
        if architecture == "causal_relation_mmdit":
            self.dit = CausalRelationMMDiTDenoiser(
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
                text_tokens=text_tokens,
                series_register_tokens=series_register_tokens,
                dim_head=mmdit_dim_head,
                qk_rmsnorm=qk_rmsnorm,
                softclamp=softclamp,
                use_semantic_grounding=use_semantic_grounding,
                grounding_num_heads=grounding_num_heads,
                grounding_sinkhorn_iters=grounding_sinkhorn_iters,
                grounding_ot_temperature=grounding_ot_temperature,
                grounding_mask_temperature=grounding_mask_temperature,
            )
        elif architecture in {"factorized_dit", "dit"}:
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
        else:
            raise ValueError(f"Unsupported model architecture: {architecture}")

    def set_route_entropy_scale(self, scale: float) -> None:
        self.route_entropy_scale = float(scale)

    def relation_slots_from_gaf(self, gaf: torch.Tensor) -> torch.Tensor:
        env_slots, _, _ = self.router.encode_gaf_slots(gaf)
        return env_slots

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_emb: torch.Tensor,
        gaf: torch.Tensor | None = None,
        semantic_atoms: torch.Tensor | None = None,
        *,
        force_drop_text: bool = False,
        force_drop_env: bool = False,
        force_drop_semantic: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = text_emb.shape[0]
        device = text_emb.device
        dtype = text_emb.dtype
        null_text = self.null_text_emb.to(device=device, dtype=dtype).expand(batch_size, -1)
        null_env = self.null_env_mix.to(device=device, dtype=dtype).expand(batch_size, -1)
        text_for_dit = text_emb if self.use_text_condition else null_text
        if force_drop_text:
            text_for_dit = null_text
            drop_text = torch.ones(batch_size, device=device, dtype=torch.bool)
        elif self.training and self.use_text_condition and self.text_drop_prob > 0:
            drop_text = torch.rand(batch_size, device=device) < self.text_drop_prob
            text_for_dit = torch.where(drop_text[:, None], null_text, text_for_dit)
        else:
            drop_text = torch.zeros(batch_size, device=device, dtype=torch.bool)
        semantic_text = semantic_atoms if semantic_atoms is not None else text_for_dit
        null_semantic = (
            null_text[:, None, :].expand(-1, semantic_text.shape[1], -1)
            if semantic_text.ndim == 3
            else null_text
        )
        if force_drop_semantic:
            semantic_text = null_semantic
            drop_semantic = torch.ones(batch_size, device=device, dtype=torch.bool)
        elif self.training and self.semantic_drop_prob > 0:
            drop_semantic = torch.rand(batch_size, device=device) < self.semantic_drop_prob
            if semantic_text.ndim == 3:
                semantic_text = torch.where(drop_semantic[:, None, None], null_semantic, semantic_text)
            else:
                semantic_text = torch.where(drop_semantic[:, None], null_semantic, semantic_text)
        else:
            drop_semantic = torch.zeros(batch_size, device=device, dtype=torch.bool)
        router_text = text_for_dit
        if self.use_env_condition:
            env_mix, alpha, env_raw, env_slots, router_aux = self.router(router_text, gaf)
            if force_drop_env:
                env_mix = null_env
                env_slots = null_env[:, None, :].expand(batch_size, self.n_env, -1)
                drop_env = torch.ones(batch_size, device=device, dtype=torch.bool)
            elif self.training and self.env_drop_prob > 0:
                drop_env = torch.rand(batch_size, device=device) < self.env_drop_prob
                env_mix = torch.where(drop_env[:, None], null_env, env_mix)
                env_slots = torch.where(drop_env[:, None, None], null_env[:, None, :], env_slots)
            else:
                drop_env = torch.zeros(batch_size, device=device, dtype=torch.bool)
        else:
            env_mix = null_env
            env_slots = null_env[:, None, :].expand(batch_size, self.n_env, -1)
            drop_env = torch.ones(batch_size, device=device, dtype=torch.bool)
            if self.router.env_source == "gaf":
                alpha = torch.full(
                    (batch_size, self.n_env),
                    1.0 / float(self.n_env),
                    device=device,
                    dtype=dtype,
                )
                env_raw = torch.zeros(batch_size, self.router.output_dim, device=device, dtype=dtype)
            else:
                alpha = torch.full(
                    (batch_size, self.n_env),
                    1.0 / float(self.n_env),
                    device=device,
                    dtype=dtype,
                )
                env_raw = torch.zeros(self.n_env, self.router.output_dim, device=device, dtype=dtype)
            zero = env_mix.new_zeros(())
            router_aux = {
                "slot_diversity_loss": zero,
                "route_entropy_loss": zero,
                "text_slot_align_loss": zero,
                "text_env_slot_loss": zero,
                "slot_cosine_mean": zero,
            }
        slot_aux_loss = (
            self.slot_diversity_weight * router_aux["slot_diversity_loss"]
            + self.route_entropy_weight * self.route_entropy_scale * router_aux["route_entropy_loss"]
            + self.text_slot_align_weight * router_aux["text_slot_align_loss"]
            + self.text_env_slot_weight * router_aux["text_env_slot_loss"]
        )
        denoiser_aux: dict[str, torch.Tensor] = {}
        if self.architecture == "causal_relation_mmdit":
            pred, denoiser_aux = self.dit(
                x,
                t,
                text_for_dit,
                env_mix,
                env_slots,
                semantic_text_emb=semantic_text,
                return_aux=True,
            )
        else:
            pred = self.dit(x, t, text_for_dit, env_mix, env_slots)
        grounding_loss_ot = denoiser_aux.get("grounding_loss_ot", env_mix.new_zeros(()))
        grounding_loss_mask = denoiser_aux.get("grounding_loss_mask", env_mix.new_zeros(()))
        grounding_loss_cycle = denoiser_aux.get("grounding_loss_cycle", env_mix.new_zeros(()))
        grounding_aux_loss = (
            self.grounding_ot_weight * grounding_loss_ot
            + self.grounding_mask_weight * grounding_loss_mask
            + self.grounding_cycle_weight * grounding_loss_cycle
        )
        entropy = -(alpha * alpha.clamp_min(1e-8).log()).sum(dim=-1).mean()
        aux = {
            "alpha": alpha,
            "env_raw": env_raw,
            "env_mix": env_mix,
            "env_slots": env_slots,
            "slot_aux_loss": slot_aux_loss,
            "slot_diversity_loss": router_aux["slot_diversity_loss"],
            "route_entropy_loss": router_aux["route_entropy_loss"],
            "text_slot_align_loss": router_aux["text_slot_align_loss"],
            "text_env_slot_loss": router_aux["text_env_slot_loss"],
            "slot_cosine_mean": router_aux["slot_cosine_mean"],
            "route_entropy_scale": torch.tensor(self.route_entropy_scale, device=device, dtype=dtype),
            "route_entropy": entropy,
            "route_max": alpha.max(dim=-1).values.mean(),
            "text_drop_rate": drop_text.float().mean(),
            "env_drop_rate": drop_env.float().mean(),
            "semantic_drop_rate": drop_semantic.float().mean(),
            "grounding_aux_loss": grounding_aux_loss,
            "grounding_loss_ot": grounding_loss_ot,
            "grounding_loss_mask": grounding_loss_mask,
            "grounding_loss_cycle": grounding_loss_cycle,
            "grounding_transport_entropy": denoiser_aux.get("grounding_transport_entropy", env_mix.new_zeros(())),
            "grounding_transport_row_error": denoiser_aux.get("grounding_transport_row_error", env_mix.new_zeros(())),
            "grounding_transport_col_error": denoiser_aux.get("grounding_transport_col_error", env_mix.new_zeros(())),
        }
        aux.update(denoiser_aux)
        return pred, aux
