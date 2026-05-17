"""
CMTSG 4-Loss Framework
======================

L1  Flow Matching     — 锚定: CausalRelationMMDiTDenoiser
    生成质量的基础保证。MSE(pred_v, target_v)。

L2  Causal Grounding  — 锚定: CausalSemanticGrounding
    语义原子（文本因果）与关系槽（GADF环境）的最优传输对齐。
    子项: OT alignment + spatial mask KL + triadic cycle consistency。

L3  Spectral Structure — 锚定: GADFRelationalSpectralLoss
    频域结构保持。保证生成序列的 GADF 频谱与真实序列一致。

L4  Causal Consistency — 锚定: EnvironmentRouter + CausalRelationMMDiTDenoiser
    环境发现的因果闭环。
    子项: triad contrastive (3模态摘要对齐) +
          cycle relation (生成→GADF→环境 的闭环一致性) +
          slot diversity (环境槽去相关)。

设计原则:
- 每个损失锚定一个核心模块，不跨模块重复
- 模块内部的子项由该模块自己管理
- 外部只需提供 4 个权重: λ_ground, λ_spectral, λ_triad, λ_cycle_relation
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def flow_matching_loss(pred_v: torch.Tensor, target_v: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_v, target_v)


def causal_grounding_loss(
    grounding_losses: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_ot = grounding_losses.get("loss_ot", torch.zeros((), device=device))
    loss_mask = grounding_losses.get("loss_ground", torch.zeros((), device=device))
    loss_cycle = grounding_losses.get("loss_cycle", torch.zeros((), device=device))
    total = loss_ot + loss_mask + loss_cycle
    return total, {"ground_ot": loss_ot, "ground_mask": loss_mask, "ground_cycle": loss_cycle}


def spectral_structure_loss(
    x0_pred: torch.Tensor,
    x_start: torch.Tensor,
    t: torch.Tensor,
    gadf_loss: nn.Module,
    warmup_power: float,
    spectral_weight_per_sample: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    loss_per_sample = gadf_loss(x0_pred, x_start)
    if spectral_weight_per_sample is not None:
        weight = spectral_weight_per_sample.detach()
    else:
        weight = (1.0 - t).pow(warmup_power).detach()
    return (loss_per_sample * weight).mean(), weight.mean()


def causal_consistency_loss(
    aux: dict[str, torch.Tensor],
    env_slots: torch.Tensor | None,
    pred_relation_slots: torch.Tensor | None,
    temperature: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # --- Sub-loss 1: Triad Contrastive ---
    # 对齐三模态摘要: series ↔ semantic ↔ relation
    series_s = aux.get("series_summary")
    semantic_s = aux.get("semantic_summary")
    relation_s = aux.get("relation_summary")
    if series_s is not None and semantic_s is not None and relation_s is not None and series_s.shape[0] >= 2:
        def _nce(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            a = F.normalize(a, dim=-1)
            b = F.normalize(b, dim=-1)
            logits = a @ b.T / temperature
            labels = torch.arange(a.shape[0], device=a.device)
            return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
        loss_triad = (_nce(series_s, semantic_s) + _nce(series_s, relation_s) + _nce(semantic_s, relation_s)) / 3.0
    else:
        loss_triad = torch.zeros((), device=device)

    # --- Sub-loss 2: Cycle Relation ---
    # 闭环: 生成 x0 → GADF → env_slots' 应与原始 env_slots 一致
    if pred_relation_slots is not None and env_slots is not None:
        loss_cycle_rel = 1.0 - F.cosine_similarity(pred_relation_slots, env_slots.detach(), dim=-1).mean()
    else:
        loss_cycle_rel = torch.zeros((), device=device)

    # --- Sub-loss 3: Slot Diversity ---
    # 环境槽去相关: off-diagonal cosine → 0
    if env_slots is not None and env_slots.shape[1] > 1:
        normed = F.normalize(env_slots, dim=-1)
        cos_mat = normed @ normed.transpose(1, 2)
        off_diag = cos_mat[:, ~torch.eye(env_slots.shape[1], device=device, dtype=torch.bool)]
        loss_diversity = off_diag.pow(2).mean()
    else:
        loss_diversity = torch.zeros((), device=device)

    total = loss_triad + loss_cycle_rel + loss_diversity
    return total, {
        "consistency_triad": loss_triad,
        "consistency_cycle_rel": loss_cycle_rel,
        "consistency_slot_diversity": loss_diversity,
    }


def compute_all_losses(
    aux: dict[str, torch.Tensor],
    x0_pred: torch.Tensor,
    x_start: torch.Tensor,
    t: torch.Tensor,
    pred_relation_slots: torch.Tensor | None,
    gadf_loss: nn.Module,
    spectral_warmup_power: float,
    contrastive_temperature: float,
    lambda_ground: float,
    lambda_spectral: float,
    lambda_triad: float,
    lambda_cycle_relation: float,
    spectral_weight_per_sample: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Central loss computation. Returns (total_loss, metrics_dict).

    L = L1_flow
      + λ_g  * L2_ground
      + λ_s  * L3_spectral
      + L4_consistency  (triad + cycle_rel + diversity)
    """
    device = x0_pred.device

    # L1: Flow Matching
    loss_l1 = aux["loss_flow"]

    # L2: Causal Grounding
    grounding_raw = {
        "loss_ot": aux.get("grounding_loss_ot", torch.zeros((), device=device)),
        "loss_ground": aux.get("grounding_loss_mask", torch.zeros((), device=device)),
        "loss_cycle": aux.get("grounding_loss_cycle", torch.zeros((), device=device)),
    }
    loss_l2, ground_metrics = causal_grounding_loss(grounding_raw, device)

    # L3: Spectral Structure
    loss_l3, spectral_weight = spectral_structure_loss(
        x0_pred, x_start, t, gadf_loss, spectral_warmup_power,
        spectral_weight_per_sample=spectral_weight_per_sample,
    )

    # L4: Causal Consistency
    env_slots = aux.get("env_slots")
    loss_l4, consistency_metrics = causal_consistency_loss(
        aux, env_slots, pred_relation_slots, contrastive_temperature, device,
    )

    # Total
    total = loss_l1 + lambda_ground * loss_l2 + lambda_spectral * loss_l3 + loss_l4

    metrics = {
        "loss": total.detach(),
        "loss_l1_flow": loss_l1.detach(),
        "loss_l2_ground": loss_l2.detach(),
        "loss_l3_spectral": loss_l3.detach(),
        "loss_l4_consistency": loss_l4.detach(),
        "spectral_weight": spectral_weight.detach(),
        **{k: v.detach() for k, v in ground_metrics.items()},
        **{k: v.detach() for k, v in consistency_metrics.items()},
        # Telemetry
        "route_entropy": aux.get("route_entropy", torch.zeros((), device=device)).detach(),
        "route_max": aux.get("route_max", torch.zeros((), device=device)).detach(),
        "text_drop_rate": aux.get("text_drop_rate", torch.zeros((), device=device)).detach(),
        "env_drop_rate": aux.get("env_drop_rate", torch.zeros((), device=device)).detach(),
        "semantic_drop_rate": aux.get("semantic_drop_rate", torch.zeros((), device=device)).detach(),
        "debug_env_slots_shape": str(tuple(env_slots.shape)) if env_slots is not None else "None",
        "debug_env_mix_shape": str(tuple(aux["env_mix"].shape)),
        # OT diagnostics
        "grounding_transport_entropy": aux.get("grounding_transport_entropy", torch.zeros((), device=device)).detach(),
        "grounding_transport_row_error": aux.get("grounding_transport_row_error", torch.zeros((), device=device)).detach(),
        "grounding_transport_col_error": aux.get("grounding_transport_col_error", torch.zeros((), device=device)).detach(),
    }
    return total, metrics
