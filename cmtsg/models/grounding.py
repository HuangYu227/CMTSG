from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GroundingConfig:
    dim: int = 64
    num_heads: int = 4
    sinkhorn_iters: int = 32
    ot_temperature: float = 0.07
    mask_temperature: float = 1.0
    eps: float = 1e-8


def _valid_group_count(channels: int, preferred: int = 8) -> int:
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class CausalSemanticGrounding(nn.Module):
    """
    Causal Semantic-Relation Grounding (CSRG).

    Inputs:
        semantic_atoms: [B, C, D]
            Text-derived causal atoms such as trend, volatility, periodicity,
            co-movement, lag, and regime tokens.
        relation_slots: [B, R, D]
            GADF-derived relation/environment slots.
        spatial_temporal_features: [B, D, K, N]
            Sequence feature map preserving variable axis K and temporal patch
            axis N.

    Outputs:
        A dictionary containing updated atoms, updated slots, OT transport,
        spatial masks, and reconstructed atoms; plus a loss dictionary with OT,
        grounding KL, and cycle losses.

    The module deliberately avoids nn.MultiheadAttention. Cross-modal matching
    is implemented with explicit multi-head bilinear energies and torch.einsum.
    """

    def __init__(
        self,
        dim: int = 64,
        num_heads: int = 4,
        sinkhorn_iters: int = 32,
        ot_temperature: float = 0.07,
        mask_temperature: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError(f"num_heads must divide dim, got dim={dim}, num_heads={num_heads}")
        if sinkhorn_iters <= 0:
            raise ValueError("sinkhorn_iters must be positive")
        if ot_temperature <= 0.0 or mask_temperature <= 0.0:
            raise ValueError("temperatures must be positive")

        self.cfg = GroundingConfig(
            dim=dim,
            num_heads=num_heads,
            sinkhorn_iters=sinkhorn_iters,
            ot_temperature=ot_temperature,
            mask_temperature=mask_temperature,
            eps=eps,
        )
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scale = self.dim_head**-0.5

        self.atom_norm = nn.LayerNorm(dim)
        self.slot_norm = nn.LayerNorm(dim)

        self.atom_to_q = nn.Linear(dim, dim, bias=False)
        self.slot_to_k = nn.Linear(dim, dim, bias=False)
        self.atom_metric = nn.Linear(dim, dim, bias=False)
        self.slot_metric = nn.Linear(dim, dim, bias=False)

        # One bilinear metric per head. This is richer than cosine attention but
        # still yields a transparent energy tensor [B, H, C, R].
        self.head_bilinear = nn.Parameter(torch.empty(num_heads, self.dim_head, self.dim_head))
        self.head_logits = nn.Parameter(torch.zeros(num_heads))

        # Cross-modal residual updates. No atom/slot concatenation is used:
        # the OT plan pulls relation context into atom space and atom context
        # into slot space through explicit einsum barycenters.
        self.atom_context_proj = nn.Linear(dim, dim)
        self.slot_context_proj = nn.Linear(dim, dim)
        self.atom_gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.Sigmoid())
        self.slot_gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.Sigmoid())
        self.atom_out_norm = nn.LayerNorm(dim)
        self.slot_out_norm = nn.LayerNorm(dim)

        # Spatial-temporal mask generator. Tokens query a depthwise-enhanced
        # feature field; masks are produced by token-to-field einsum.
        groups = _valid_group_count(dim)
        self.feature_norm = nn.GroupNorm(groups, dim)
        self.feature_field = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
        )
        self.atom_to_mask_q = nn.Linear(dim, dim, bias=False)
        self.slot_to_mask_q = nn.Linear(dim, dim, bias=False)
        self.atom_mask_head_logits = nn.Parameter(torch.zeros(num_heads))
        self.slot_mask_head_logits = nn.Parameter(torch.zeros(num_heads))

        # Inverse head for triadic cycle consistency:
        # relation slots -> semantic atom space.
        self.inverse_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.head_bilinear)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Gates begin near identity-preserving residual routing.
        nn.init.zeros_(self.atom_gate[1].weight)
        nn.init.zeros_(self.atom_gate[1].bias)
        nn.init.zeros_(self.slot_gate[1].weight)
        nn.init.zeros_(self.slot_gate[1].bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return x.reshape(batch, tokens, self.num_heads, self.dim_head)

    def _pairwise_scores(
        self,
        semantic_atoms: torch.Tensor,
        relation_slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        atom = self.atom_norm(semantic_atoms)
        slot = self.slot_norm(relation_slots)

        atom_q = F.normalize(self._split_heads(self.atom_to_q(atom)), dim=-1, eps=self.cfg.eps)
        slot_k = F.normalize(self._split_heads(self.slot_to_k(slot)), dim=-1, eps=self.cfg.eps)

        # [B,C,H,d] x [H,d,e] x [B,R,H,e] -> [B,H,C,R]
        head_scores = torch.einsum("bchd,hde,brhe->bhcr", atom_q, self.head_bilinear, slot_k)
        head_weights = self.head_logits.softmax(dim=0)
        bilinear_scores = torch.einsum("h,bhcr->bcr", head_weights, head_scores) * self.scale

        atom_metric = F.normalize(self.atom_metric(atom), dim=-1, eps=self.cfg.eps)
        slot_metric = F.normalize(self.slot_metric(slot), dim=-1, eps=self.cfg.eps)
        cosine_sim = torch.einsum("bcd,brd->bcr", atom_metric, slot_metric)

        scores = bilinear_scores + cosine_sim
        scores = torch.nan_to_num(scores, nan=0.0, posinf=20.0, neginf=-20.0)
        scores = (scores / 20.0).tanh() * 20.0
        return scores, cosine_sim

    def _log_sinkhorn(self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, num_atoms, num_slots = scores.shape
        log_alpha = scores / self.cfg.ot_temperature
        log_alpha = torch.nan_to_num(log_alpha, nan=0.0, posinf=50.0, neginf=-50.0)

        log_mu = torch.full(
            (batch, num_atoms),
            -math.log(float(num_atoms)),
            device=scores.device,
            dtype=scores.dtype,
        )
        log_nu = torch.full(
            (batch, num_slots),
            -math.log(float(num_slots)),
            device=scores.device,
            dtype=scores.dtype,
        )

        log_u = torch.zeros_like(log_mu)
        log_v = torch.zeros_like(log_nu)
        for _ in range(self.cfg.sinkhorn_iters):
            log_u = log_mu - torch.logsumexp(log_alpha + log_v[:, None, :], dim=-1)
            log_v = log_nu - torch.logsumexp(log_alpha + log_u[:, :, None], dim=-2)

        log_plan = log_alpha + log_u[:, :, None] + log_v[:, None, :]
        plan = log_plan.exp().clamp_min(self.cfg.eps)
        plan = plan / plan.sum(dim=(1, 2), keepdim=True).clamp_min(self.cfg.eps)
        return plan, log_plan

    def _normalized_grounding(self, transport_plan: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        atom_to_slot = transport_plan / transport_plan.sum(dim=-1, keepdim=True).clamp_min(self.cfg.eps)
        slot_to_atom = transport_plan / transport_plan.sum(dim=1, keepdim=True).clamp_min(self.cfg.eps)
        return atom_to_slot, slot_to_atom

    def _spatial_softmax(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, tokens, height, width = logits.shape
        flat = logits.reshape(batch, tokens, height * width) / self.cfg.mask_temperature
        log_probs = F.log_softmax(flat, dim=-1).reshape(batch, tokens, height, width)
        probs = log_probs.exp()
        return probs, log_probs

    def _make_masks(
        self,
        semantic_atoms: torch.Tensor,
        relation_slots: torch.Tensor,
        spatial_temporal_features: torch.Tensor,
        slot_to_atom: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        field = self.feature_field(self.feature_norm(spatial_temporal_features))
        batch, _, num_vars, num_patches = field.shape
        field = field.reshape(batch, self.num_heads, self.dim_head, num_vars, num_patches)

        atom_q = F.normalize(self._split_heads(self.atom_to_mask_q(self.atom_norm(semantic_atoms))), dim=-1, eps=self.cfg.eps)
        slot_q = F.normalize(self._split_heads(self.slot_to_mask_q(self.slot_norm(relation_slots))), dim=-1, eps=self.cfg.eps)

        # Token-to-field logits:
        # atoms [B,C,H,d] x field [B,H,d,K,N] -> [B,C,H,K,N]
        atom_logits_h = torch.einsum("bchd,bhdkn->bchkn", atom_q, field) * self.scale
        slot_logits_h = torch.einsum("brhd,bhdkn->brhkn", slot_q, field) * self.scale
        atom_weights = self.atom_mask_head_logits.softmax(dim=0)
        slot_weights = self.slot_mask_head_logits.softmax(dim=0)
        atom_logits = torch.einsum("h,bchkn->bckn", atom_weights, atom_logits_h)
        slot_logits = torch.einsum("h,brhkn->brkn", slot_weights, slot_logits_h)

        text_mask, text_log_mask = self._spatial_softmax(atom_logits)
        slot_mask, slot_log_mask = self._spatial_softmax(slot_logits)

        # Blueprint aggregation: M_text_to_slot = G^T @ M_text.
        # slot_to_atom is column-normalized over atoms for each slot, so the
        # weighted sum remains a valid spatial distribution after renormalizing.
        text_to_slot = torch.einsum("bcr,bckn->brkn", slot_to_atom, text_mask)
        text_to_slot = text_to_slot / text_to_slot.sum(dim=(-2, -1), keepdim=True).clamp_min(self.cfg.eps)
        text_to_slot_log = text_to_slot.clamp_min(self.cfg.eps).log()
        return slot_mask, slot_log_mask, text_mask, text_log_mask, text_to_slot, text_to_slot_log

    def forward(
        self,
        semantic_atoms: torch.Tensor,
        relation_slots: torch.Tensor,
        spatial_temporal_features: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if semantic_atoms.ndim != 3:
            raise ValueError(f"semantic_atoms must be [B,C,D], got {tuple(semantic_atoms.shape)}")
        if relation_slots.ndim != 3:
            raise ValueError(f"relation_slots must be [B,R,D], got {tuple(relation_slots.shape)}")
        if spatial_temporal_features.ndim != 4:
            raise ValueError(f"spatial_temporal_features must be [B,D,K,N], got {tuple(spatial_temporal_features.shape)}")
        if semantic_atoms.shape[0] != relation_slots.shape[0] or semantic_atoms.shape[0] != spatial_temporal_features.shape[0]:
            raise ValueError("All inputs must share the same batch size")
        if semantic_atoms.shape[-1] != self.dim or relation_slots.shape[-1] != self.dim:
            raise ValueError(f"semantic_atoms/relation_slots last dim must be {self.dim}")
        if spatial_temporal_features.shape[1] != self.dim:
            raise ValueError(f"spatial_temporal_features channel dim must be {self.dim}")

        original_atoms = semantic_atoms

        scores, cosine_sim = self._pairwise_scores(semantic_atoms, relation_slots)
        transport_plan, log_transport_plan = self._log_sinkhorn(scores)
        atom_to_slot, slot_to_atom = self._normalized_grounding(transport_plan)

        relation_context_for_atoms = torch.einsum("bcr,brd->bcd", atom_to_slot, relation_slots)
        atom_context_for_slots = torch.einsum("bcr,bcd->brd", slot_to_atom, semantic_atoms)

        atom_delta = self.atom_context_proj(relation_context_for_atoms)
        slot_delta = self.slot_context_proj(atom_context_for_slots)
        atom_gate = self.atom_gate(semantic_atoms)
        slot_gate = self.slot_gate(relation_slots)
        updated_atoms = self.atom_out_norm(semantic_atoms + atom_gate * atom_delta)
        updated_slots = self.slot_out_norm(relation_slots + slot_gate * slot_delta)

        slot_mask, slot_log_mask, text_mask, text_log_mask, text_to_slot, text_to_slot_log = self._make_masks(
            semantic_atoms,
            relation_slots,
            spatial_temporal_features,
            slot_to_atom,
        )

        atom_pullback = torch.einsum("bcr,brd->bcd", atom_to_slot, updated_slots)
        reconstructed_atoms = self.inverse_head(atom_pullback)

        # Loss 1: OT alignment. The transport plan has total mass 1 per sample,
        # so this scale is stable across different C and R.
        loss_ot = -(transport_plan * cosine_sim).sum(dim=(1, 2)).mean()

        # Loss 2: spatial-temporal grounding KL, averaged over samples and slots.
        kl_map = slot_mask * (slot_log_mask - text_to_slot_log)
        loss_ground = kl_map.sum(dim=(-2, -1)).mean()

        # Loss 3: triadic cycle consistency in semantic atom space.
        cycle_cos = F.cosine_similarity(original_atoms, reconstructed_atoms, dim=-1, eps=self.cfg.eps)
        loss_cycle = 1.0 - cycle_cos.mean()

        row_target = 1.0 / float(semantic_atoms.shape[1])
        col_target = 1.0 / float(relation_slots.shape[1])
        losses = {
            "loss_ot": loss_ot,
            "loss_ground": loss_ground,
            "loss_cycle": loss_cycle,
            "transport_entropy": -(transport_plan * log_transport_plan).sum(dim=(1, 2)).mean().detach(),
            "transport_row_error": (transport_plan.sum(dim=-1) - row_target).abs().mean().detach(),
            "transport_col_error": (transport_plan.sum(dim=1) - col_target).abs().mean().detach(),
        }

        outputs = {
            "semantic_atoms": updated_atoms,
            "relation_slots": updated_slots,
            "transport_plan": transport_plan,
            "grounding_matrix": slot_to_atom,
            "atom_to_slot": atom_to_slot,
            "slot_to_atom": slot_to_atom,
            "slot_masks": slot_mask,
            "text_masks": text_mask,
            "text_to_slot_masks": text_to_slot,
            "reconstructed_atoms": reconstructed_atoms,
        }
        return outputs, losses


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains non-finite values")


if __name__ == "__main__":
    torch.manual_seed(7)
    batch, num_atoms, num_slots, dim, num_vars, num_patches = 2, 6, 12, 64, 7, 16

    semantic_atoms = torch.randn(batch, num_atoms, dim)
    relation_slots = torch.randn(batch, num_slots, dim)
    spatial_temporal_features = torch.randn(batch, dim, num_vars, num_patches)

    module = CausalSemanticGrounding(
        dim=dim,
        num_heads=4,
        sinkhorn_iters=24,
        ot_temperature=0.07,
        mask_temperature=1.0,
    )
    module.train()

    outputs, losses = module(semantic_atoms, relation_slots, spatial_temporal_features)

    assert outputs["semantic_atoms"].shape == (batch, num_atoms, dim)
    assert outputs["relation_slots"].shape == (batch, num_slots, dim)
    assert outputs["transport_plan"].shape == (batch, num_atoms, num_slots)
    assert outputs["grounding_matrix"].shape == (batch, num_atoms, num_slots)
    assert outputs["slot_masks"].shape == (batch, num_slots, num_vars, num_patches)
    assert outputs["text_masks"].shape == (batch, num_atoms, num_vars, num_patches)
    assert outputs["text_to_slot_masks"].shape == (batch, num_slots, num_vars, num_patches)
    assert outputs["reconstructed_atoms"].shape == (batch, num_atoms, dim)

    for key, value in outputs.items():
        if torch.is_tensor(value) and value.dtype.is_floating_point:
            _assert_finite(key, value)
    for key, value in losses.items():
        _assert_finite(key, value)

    print("CausalSemanticGrounding self-test passed.")
    for key in ("loss_ot", "loss_ground", "loss_cycle", "transport_row_error", "transport_col_error"):
        print(f"{key}: {float(losses[key].detach()):.6f}")
