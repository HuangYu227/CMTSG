from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cmtsg.config import get_nested, load_config
from cmtsg.data import load_ts
from cmtsg.dataset import CMTSGDataset
from cmtsg.logging_utils import append_csv, append_jsonl
from cmtsg.metrics import fid_raw, flat_kl, jftsd_text_proxy, mdd, mmd_rbf
from cmtsg.models import CMTSGModel
from cmtsg.models.diffusion import GaussianDiffusion, RectifiedFlow
from cmtsg.semantic_metrics import compute_cttp_metrics
from cmtsg.utils import ensure_dir, resolve_path, save_json, set_seed


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _make_model(cfg: dict, seq_len: int, n_vars: int, gaf_size: int) -> GaussianDiffusion | RectifiedFlow:
    model_cfg = cfg.get("model", {})
    text_dim = int(cfg.get("text_emb_dim", 64))
    env_dim = int(model_cfg.get("env_dim", text_dim))
    n_env = int(model_cfg.get("n_env", cfg.get("n_env", 12)))
    model = CMTSGModel(
        seq_len=seq_len,
        n_vars=n_vars,
        gaf_size=gaf_size,
        text_dim=text_dim,
        env_dim=env_dim,
        hidden_size=int(model_cfg.get("hidden_size", 128)),
        depth=int(model_cfg.get("depth", 4)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
        patch_size=int(model_cfg.get("patch_size", 1)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        n_env=n_env,
        use_text_condition=bool(model_cfg.get("use_text_condition", True)),
        use_env_condition=bool(model_cfg.get("use_env_condition", True)),
        env_source=str(model_cfg.get("env_source", "gaf")),
        routing=str(model_cfg.get("routing", "text")),
        env_hidden_channels=int(model_cfg.get("env_hidden_channels", 64)),
        env_num_blocks=int(model_cfg.get("env_num_blocks", 4)),
        env_kernel_size=int(model_cfg.get("env_kernel_size", 7)),
        env_expansion=int(model_cfg.get("env_expansion", 4)),
        env_pool_size=int(model_cfg.get("env_pool_size", 4)),
        env_slot_mode=str(model_cfg.get("env_slot_mode", "dynamic_gaf")),
        architecture=str(model_cfg.get("architecture", "causal_relation_mmdit")),
        text_tokens=int(model_cfg.get("text_tokens", 4)),
        series_register_tokens=int(model_cfg.get("series_register_tokens", 64)),
        mmdit_dim_head=(int(model_cfg["mmdit_dim_head"]) if "mmdit_dim_head" in model_cfg else None),
        qk_rmsnorm=bool(model_cfg.get("qk_rmsnorm", True)),
        softclamp=bool(model_cfg.get("softclamp", True)),
        use_semantic_grounding=bool(model_cfg.get("use_semantic_grounding", True)),
        grounding_num_heads=int(model_cfg.get("grounding_num_heads", 4)),
        grounding_sinkhorn_iters=int(model_cfg.get("grounding_sinkhorn_iters", 24)),
        grounding_ot_temperature=float(model_cfg.get("grounding_ot_temperature", 0.07)),
        grounding_mask_temperature=float(model_cfg.get("grounding_mask_temperature", 1.0)),
        grounding_ot_weight=float(model_cfg.get("grounding_ot_weight", 0.01)),
        grounding_mask_weight=float(model_cfg.get("grounding_mask_weight", 0.01)),
        grounding_cycle_weight=float(model_cfg.get("grounding_cycle_weight", 0.01)),
        slot_diversity_weight=float(model_cfg.get("slot_diversity_weight", 0.01)),
        route_entropy_weight=float(model_cfg.get("route_entropy_weight", 0.001)),
        text_slot_align_weight=float(model_cfg.get("text_slot_align_weight", 0.01)),
        text_env_slot_weight=float(model_cfg.get("text_env_slot_weight", 0.02)),
        text_drop_prob=float(model_cfg.get("text_drop_prob", 0.0)),
        env_drop_prob=float(model_cfg.get("env_drop_prob", 0.0)),
        semantic_drop_prob=float(model_cfg.get("semantic_drop_prob", 0.0)),
    )
    diff_cfg = cfg.get("diffusion", {})
    objective = str(diff_cfg.get("objective", "rectified_flow")).lower()
    if objective in {"rectified_flow", "flow", "flow_matching"}:
        sampling_cfg = cfg.get("sampling", {})
        return RectifiedFlow(
            model=model,
            num_steps=int(diff_cfg.get("num_steps", 100)),
            lambda_spectral=float(diff_cfg.get("lambda_spectral", 0.05)),
            spectral_warmup_power=float(diff_cfg.get("spectral_warmup_power", 1.0)),
            spectral_mode=str(diff_cfg.get("spectral_mode", "abs")),
            spectral_high_freq_gamma=float(diff_cfg.get("spectral_high_freq_gamma", 1.0)),
            spectral_dc_weight=float(diff_cfg.get("spectral_dc_weight", 0.05)),
            lambda_cycle_relation=float(diff_cfg.get("lambda_cycle_relation", 0.01)),
            lambda_triad_contrastive=float(diff_cfg.get("lambda_triad_contrastive", 0.01)),
            contrastive_temperature=float(diff_cfg.get("contrastive_temperature", 0.07)),
            solver=str(sampling_cfg.get("solver", diff_cfg.get("solver", "heun"))),
            guidance_text=float(sampling_cfg.get("guidance_text", 2.0)),
            guidance_relation=float(sampling_cfg.get("guidance_relation", 1.5)),
            guidance_joint=float(sampling_cfg.get("guidance_joint", 1.0)),
        )
    if objective in {"ddpm", "gaussian_diffusion"}:
        return GaussianDiffusion(
            model=model,
            num_steps=int(diff_cfg.get("num_steps", 100)),
            beta_start=float(diff_cfg.get("beta_start", 0.0001)),
            beta_end=float(diff_cfg.get("beta_end", 0.02)),
            schedule=str(diff_cfg.get("schedule", "quad")),
            lambda_spectral=float(diff_cfg.get("lambda_spectral", 0.05)),
            spectral_warmup_power=float(diff_cfg.get("spectral_warmup_power", 1.0)),
            spectral_mode=str(diff_cfg.get("spectral_mode", "abs")),
            spectral_high_freq_gamma=float(diff_cfg.get("spectral_high_freq_gamma", 1.0)),
            spectral_dc_weight=float(diff_cfg.get("spectral_dc_weight", 0.05)),
        )
    raise ValueError(f"Unsupported diffusion objective: {objective}")


def _save_checkpoint(path: Path, diffusion: GaussianDiffusion, optimizer: torch.optim.Optimizer, epoch: int, cfg: dict, stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": diffusion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "stats": stats,
        },
        path,
    )


def _default_best_metric_specs() -> list[dict[str, str]]:
    return [
        {"name": "fid_cttp", "mode": "min"},
        {"name": "jftsd_cttp", "mode": "min"},
        {"name": "cttp", "mode": "max"},
    ]


def _is_better(value: float, best: float | None, mode: str) -> bool:
    if best is None:
        return True
    if mode == "min":
        return value < best
    if mode == "max":
        return value > best
    raise ValueError(f"Unsupported best metric mode: {mode}")


def _optional_to_device(batch: dict, key: str, device: torch.device) -> torch.Tensor | None:
    value = batch.get(key)
    if value is None:
        return None
    return value.to(device)


def _shape_str(value: torch.Tensor | None) -> str:
    if value is None:
        return "None"
    return "x".join(str(dim) for dim in value.shape)


def _module_grad_norm(module: torch.nn.Module | None) -> float:
    if module is None:
        return 0.0
    total = 0.0
    found = False
    for param in module.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total += float(grad.pow(2).sum().cpu())
        found = True
    return float(total**0.5) if found else 0.0


def _debug_modules(diffusion: GaussianDiffusion | RectifiedFlow) -> tuple[torch.nn.Module | None, torch.nn.Module | None, torch.nn.Module | None]:
    model = diffusion.model
    router = getattr(model, "router", None)
    dit = getattr(model, "dit", None)
    gaf_encoder = getattr(router, "encoder", None) if router is not None else None
    grounding = getattr(dit, "semantic_grounding", None) if dit is not None else None
    return gaf_encoder, grounding, dit


def train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = _device(args.device)

    data_root = resolve_path(args.data_root or cfg["data_root"])
    processed_root = resolve_path(args.processed_root or cfg["processed_root"])
    output_root = ensure_dir(args.output_root or cfg["output_root"])

    train_ts = load_ts(data_root / "train_ts.npy")
    seq_len, n_vars = train_ts.shape[1], train_ts.shape[2]
    mean = train_ts.mean(axis=(0, 1), keepdims=True)
    std = train_ts.std(axis=(0, 1), keepdims=True) + 1e-6
    gaf_max_size = int(cfg.get("gaf_max_size", 384))
    gaf_size = min(seq_len, gaf_max_size)

    train_ds = CMTSGDataset(data_root, processed_root, "train", mean=mean, std=std, train=True, gaf_max_size=gaf_max_size)
    valid_ds = CMTSGDataset(data_root, processed_root, "valid", mean=mean, std=std, train=False, gaf_max_size=gaf_max_size)
    batch_size = int(args.batch_size or get_nested(cfg, "training.batch_size", 128))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(get_nested(cfg, "training.num_workers", 0)),
        drop_last=True,
    )
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    diffusion = _make_model(cfg, seq_len, n_vars, gaf_size).to(device)
    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr=float(args.lr or get_nested(cfg, "training.lr", 2e-4)),
        weight_decay=float(get_nested(cfg, "training.weight_decay", 0.01)),
    )
    grad_clip = float(get_nested(cfg, "training.grad_clip", 1.0))
    epochs = int(args.epochs or get_nested(cfg, "training.epochs", 50))
    best_val = float("inf")
    stats = {
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
        "seq_len": seq_len,
        "n_vars": n_vars,
        "gaf_size": gaf_size,
        "gaf_max_size": gaf_max_size,
        "model_ablation": {
            "architecture": str(cfg.get("model", {}).get("architecture", "causal_relation_mmdit")),
            "objective": str(cfg.get("diffusion", {}).get("objective", "rectified_flow")),
            "use_text_condition": bool(cfg.get("model", {}).get("use_text_condition", True)),
            "use_env_condition": bool(cfg.get("model", {}).get("use_env_condition", True)),
            "env_source": str(cfg.get("model", {}).get("env_source", "gaf")),
            "routing": str(cfg.get("model", {}).get("routing", "text")),
            "env_slot_mode": str(cfg.get("model", {}).get("env_slot_mode", "dynamic_gaf")),
            "n_env": int(cfg.get("model", {}).get("n_env", cfg.get("n_env", 12))),
            "env_dim": int(cfg.get("model", {}).get("env_dim", cfg.get("text_emb_dim", 64))),
            "use_semantic_grounding": bool(cfg.get("model", {}).get("use_semantic_grounding", True)),
            "grounding_ot_weight": float(cfg.get("model", {}).get("grounding_ot_weight", 0.01)),
            "grounding_mask_weight": float(cfg.get("model", {}).get("grounding_mask_weight", 0.01)),
            "grounding_cycle_weight": float(cfg.get("model", {}).get("grounding_cycle_weight", 0.01)),
        },
    }
    save_json(stats, output_root / "train_stats.json")
    log_dir = ensure_dir(output_root / "logs")
    eval_cfg = cfg.get("evaluation", {})
    sample_every = int(args.sample_every if args.sample_every is not None else eval_cfg.get("sample_every", 0))
    sample_count = int(args.sample_count or eval_cfg.get("sample_count", 128))
    best_metric_specs = eval_cfg.get("best_metrics") or _default_best_metric_specs()
    best_sample_metrics: dict[str, float] = {}
    route_entropy_warmup_epochs = float(cfg.get("model", {}).get("route_entropy_warmup_epochs", 10.0))

    for epoch in range(1, epochs + 1):
        if route_entropy_warmup_epochs > 0:
            route_entropy_scale = max(0.0, 1.0 - float(epoch - 1) / route_entropy_warmup_epochs)
        else:
            route_entropy_scale = 0.0
        diffusion.model.set_route_entropy_scale(route_entropy_scale)
        epoch_start = time.time()
        diffusion.train()
        train_losses = []
        train_loss_diff = []
        train_loss_spectral = []
        train_loss_slot_aux = []
        train_loss_grounding_aux = []
        train_loss_cycle_relation = []
        train_loss_triad_contrastive = []
        train_grounding_ot = []
        train_grounding_mask = []
        train_grounding_cycle = []
        train_spectral_weight = []
        train_route_entropy = []
        train_route_entropy_loss = []
        train_route_entropy_scale = []
        train_route_max = []
        train_slot_diversity = []
        train_text_slot_align = []
        train_text_env_slot = []
        train_slot_cosine = []
        train_text_drop = []
        train_env_drop = []
        train_semantic_drop = []
        train_gaf_encoder_grad = []
        train_grounding_grad = []
        train_mmdit_grad = []
        debug_shapes: dict[str, str] = {}
        for batch in train_loader:
            x = batch["x"].to(device)
            text_emb = batch["text_emb"].to(device)
            gaf = batch["gaf"].to(device)
            semantic_atoms = _optional_to_device(batch, "semantic_atoms", device)
            loss, metrics = diffusion.training_loss(x, text_emb, gaf, semantic_atoms=semantic_atoms)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gaf_encoder, grounding, mmdit = _debug_modules(diffusion)
            train_gaf_encoder_grad.append(_module_grad_norm(gaf_encoder))
            train_grounding_grad.append(_module_grad_norm(grounding))
            train_mmdit_grad.append(_module_grad_norm(mmdit))
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(diffusion.parameters(), grad_clip)
            optimizer.step()
            if not debug_shapes:
                debug_shapes = {
                    "debug_x_shape": _shape_str(x),
                    "debug_gaf_shape": _shape_str(gaf),
                    "debug_text_emb_shape": _shape_str(text_emb),
                    "debug_semantic_atoms_shape": _shape_str(semantic_atoms),
                    "debug_env_slots_shape": str(metrics.get("debug_env_slots_shape", "unknown")),
                    "debug_env_mix_shape": str(metrics.get("debug_env_mix_shape", "unknown")),
                }
            train_losses.append(float(loss.detach().cpu()))
            train_loss_diff.append(float(metrics["loss_diff"].detach().cpu()))
            train_loss_spectral.append(float(metrics["loss_spectral"].detach().cpu()))
            train_loss_slot_aux.append(float(metrics["loss_slot_aux"].detach().cpu()))
            train_loss_grounding_aux.append(float(metrics.get("loss_grounding_aux", torch.tensor(0.0, device=device)).detach().cpu()))
            train_loss_cycle_relation.append(float(metrics.get("loss_cycle_relation", torch.tensor(0.0, device=device)).detach().cpu()))
            train_loss_triad_contrastive.append(float(metrics.get("loss_triad_contrastive", torch.tensor(0.0, device=device)).detach().cpu()))
            train_grounding_ot.append(float(metrics.get("grounding_loss_ot", torch.tensor(0.0, device=device)).detach().cpu()))
            train_grounding_mask.append(float(metrics.get("grounding_loss_mask", torch.tensor(0.0, device=device)).detach().cpu()))
            train_grounding_cycle.append(float(metrics.get("grounding_loss_cycle", torch.tensor(0.0, device=device)).detach().cpu()))
            train_spectral_weight.append(float(metrics["spectral_weight"].detach().cpu()))
            train_route_entropy.append(float(metrics["route_entropy"].detach().cpu()))
            train_route_entropy_loss.append(float(metrics["route_entropy_loss"].detach().cpu()))
            train_route_entropy_scale.append(float(metrics["route_entropy_scale"].detach().cpu()))
            train_route_max.append(float(metrics["route_max"].detach().cpu()))
            train_slot_diversity.append(float(metrics["slot_diversity_loss"].detach().cpu()))
            train_text_slot_align.append(float(metrics["text_slot_align_loss"].detach().cpu()))
            train_text_env_slot.append(float(metrics["text_env_slot_loss"].detach().cpu()))
            train_slot_cosine.append(float(metrics["slot_cosine_mean"].detach().cpu()))
            train_text_drop.append(float(metrics["text_drop_rate"].detach().cpu()))
            train_env_drop.append(float(metrics["env_drop_rate"].detach().cpu()))
            train_semantic_drop.append(float(metrics.get("semantic_drop_rate", torch.tensor(0.0, device=device)).detach().cpu()))

        diffusion.eval()
        val_losses = []
        val_loss_diff = []
        val_loss_spectral = []
        val_loss_slot_aux = []
        val_loss_grounding_aux = []
        val_loss_cycle_relation = []
        val_loss_triad_contrastive = []
        val_grounding_ot = []
        val_grounding_mask = []
        val_grounding_cycle = []
        val_spectral_weight = []
        val_route_entropy = []
        val_route_entropy_loss = []
        val_route_entropy_scale = []
        val_route_max = []
        val_slot_diversity = []
        val_text_slot_align = []
        val_text_env_slot = []
        val_slot_cosine = []
        val_text_drop = []
        val_env_drop = []
        val_semantic_drop = []
        with torch.no_grad():
            for batch in valid_loader:
                x = batch["x"].to(device)
                text_emb = batch["text_emb"].to(device)
                gaf = batch["gaf"].to(device)
                semantic_atoms = _optional_to_device(batch, "semantic_atoms", device)
                loss, metrics = diffusion.training_loss(x, text_emb, gaf, semantic_atoms=semantic_atoms)
                val_losses.append(float(loss.detach().cpu()))
                val_loss_diff.append(float(metrics["loss_diff"].detach().cpu()))
                val_loss_spectral.append(float(metrics["loss_spectral"].detach().cpu()))
                val_loss_slot_aux.append(float(metrics["loss_slot_aux"].detach().cpu()))
                val_loss_grounding_aux.append(float(metrics.get("loss_grounding_aux", torch.tensor(0.0, device=device)).detach().cpu()))
                val_loss_cycle_relation.append(float(metrics.get("loss_cycle_relation", torch.tensor(0.0, device=device)).detach().cpu()))
                val_loss_triad_contrastive.append(float(metrics.get("loss_triad_contrastive", torch.tensor(0.0, device=device)).detach().cpu()))
                val_grounding_ot.append(float(metrics.get("grounding_loss_ot", torch.tensor(0.0, device=device)).detach().cpu()))
                val_grounding_mask.append(float(metrics.get("grounding_loss_mask", torch.tensor(0.0, device=device)).detach().cpu()))
                val_grounding_cycle.append(float(metrics.get("grounding_loss_cycle", torch.tensor(0.0, device=device)).detach().cpu()))
                val_spectral_weight.append(float(metrics["spectral_weight"].detach().cpu()))
                val_route_entropy.append(float(metrics["route_entropy"].detach().cpu()))
                val_route_entropy_loss.append(float(metrics["route_entropy_loss"].detach().cpu()))
                val_route_entropy_scale.append(float(metrics["route_entropy_scale"].detach().cpu()))
                val_route_max.append(float(metrics["route_max"].detach().cpu()))
                val_slot_diversity.append(float(metrics["slot_diversity_loss"].detach().cpu()))
                val_text_slot_align.append(float(metrics["text_slot_align_loss"].detach().cpu()))
                val_text_env_slot.append(float(metrics["text_env_slot_loss"].detach().cpu()))
                val_slot_cosine.append(float(metrics["slot_cosine_mean"].detach().cpu()))
                val_text_drop.append(float(metrics["text_drop_rate"].detach().cpu()))
                val_env_drop.append(float(metrics["env_drop_rate"].detach().cpu()))
                val_semantic_drop.append(float(metrics.get("semantic_drop_rate", torch.tensor(0.0, device=device)).detach().cpu()))
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_loss_diff": float(np.mean(train_loss_diff)),
            "train_loss_spectral": float(np.mean(train_loss_spectral)),
            "train_loss_slot_aux": float(np.mean(train_loss_slot_aux)),
            "train_loss_grounding_aux": float(np.mean(train_loss_grounding_aux)),
            "train_loss_cycle_relation": float(np.mean(train_loss_cycle_relation)),
            "train_loss_triad_contrastive": float(np.mean(train_loss_triad_contrastive)),
            "train_grounding_loss_ot": float(np.mean(train_grounding_ot)),
            "train_grounding_loss_mask": float(np.mean(train_grounding_mask)),
            "train_grounding_loss_cycle": float(np.mean(train_grounding_cycle)),
            "train_spectral_weight": float(np.mean(train_spectral_weight)),
            "val_loss_diff": float(np.mean(val_loss_diff)),
            "val_loss_spectral": float(np.mean(val_loss_spectral)),
            "val_loss_slot_aux": float(np.mean(val_loss_slot_aux)),
            "val_loss_grounding_aux": float(np.mean(val_loss_grounding_aux)),
            "val_loss_cycle_relation": float(np.mean(val_loss_cycle_relation)),
            "val_loss_triad_contrastive": float(np.mean(val_loss_triad_contrastive)),
            "val_grounding_loss_ot": float(np.mean(val_grounding_ot)),
            "val_grounding_loss_mask": float(np.mean(val_grounding_mask)),
            "val_grounding_loss_cycle": float(np.mean(val_grounding_cycle)),
            "val_spectral_weight": float(np.mean(val_spectral_weight)),
            "train_route_entropy": float(np.mean(train_route_entropy)),
            "train_route_entropy_loss": float(np.mean(train_route_entropy_loss)),
            "train_route_entropy_scale": float(np.mean(train_route_entropy_scale)),
            "train_route_max": float(np.mean(train_route_max)),
            "train_slot_diversity_loss": float(np.mean(train_slot_diversity)),
            "train_text_slot_align_loss": float(np.mean(train_text_slot_align)),
            "train_text_env_slot_loss": float(np.mean(train_text_env_slot)),
            "train_slot_cosine_mean": float(np.mean(train_slot_cosine)),
            "val_route_entropy": float(np.mean(val_route_entropy)),
            "val_route_entropy_loss": float(np.mean(val_route_entropy_loss)),
            "val_route_entropy_scale": float(np.mean(val_route_entropy_scale)),
            "val_route_max": float(np.mean(val_route_max)),
            "val_slot_diversity_loss": float(np.mean(val_slot_diversity)),
            "val_text_slot_align_loss": float(np.mean(val_text_slot_align)),
            "val_text_env_slot_loss": float(np.mean(val_text_env_slot)),
            "val_slot_cosine_mean": float(np.mean(val_slot_cosine)),
            "train_text_drop_rate": float(np.mean(train_text_drop)),
            "train_env_drop_rate": float(np.mean(train_env_drop)),
            "train_semantic_drop_rate": float(np.mean(train_semantic_drop)),
            "val_text_drop_rate": float(np.mean(val_text_drop)),
            "val_env_drop_rate": float(np.mean(val_env_drop)),
            "val_semantic_drop_rate": float(np.mean(val_semantic_drop)),
            "train_gaf_encoder_grad_norm": float(np.mean(train_gaf_encoder_grad)),
            "train_grounding_grad_norm": float(np.mean(train_grounding_grad)),
            "train_mmdit_grad_norm": float(np.mean(train_mmdit_grad)),
            **debug_shapes,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": float(time.time() - epoch_start),
        }
        print(
            f"epoch={epoch:04d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"val_diff={row['val_loss_diff']:.6f} val_spec={row['val_loss_spectral']:.6f} "
            f"val_slot={row['val_loss_slot_aux']:.6f} val_ground={row['val_loss_grounding_aux']:.6f} "
            f"route_max={row['val_route_max']:.4f} "
            f"grad_gaf={row['train_gaf_encoder_grad_norm']:.3e} "
            f"grad_ground={row['train_grounding_grad_norm']:.3e} "
            f"grad_mmdit={row['train_mmdit_grad_norm']:.3e}"
        )

        _save_checkpoint(output_root / "checkpoints" / "last.pt", diffusion, optimizer, epoch, cfg, stats)
        if val_loss < best_val:
            best_val = val_loss
            _save_checkpoint(output_root / "checkpoints" / "best.pt", diffusion, optimizer, epoch, cfg, stats)
        if epoch % int(get_nested(cfg, "training.save_every", 10)) == 0:
            _save_checkpoint(output_root / "checkpoints" / f"epoch_{epoch:04d}.pt", diffusion, optimizer, epoch, cfg, stats)
        if sample_every > 0 and epoch % sample_every == 0:
            sample_metrics = sample_and_score(
                diffusion,
                valid_ds,
                output_root,
                device,
                mean,
                std,
                sample_count,
                cfg,
                tag=f"epoch_{epoch:04d}",
            )
            row.update({f"sample_{key}": value for key, value in sample_metrics.items() if isinstance(value, (int, float))})
            for spec in best_metric_specs:
                metric_name = str(spec["name"])
                metric_mode = str(spec.get("mode", "min"))
                metric_value = sample_metrics.get(metric_name)
                if not isinstance(metric_value, (int, float)):
                    continue
                previous_best = best_sample_metrics.get(metric_name)
                if _is_better(float(metric_value), previous_best, metric_mode):
                    best_sample_metrics[metric_name] = float(metric_value)
                    metric_stats = {
                        **stats,
                        "best_metric_name": metric_name,
                        "best_metric_mode": metric_mode,
                        "best_metric_value": float(metric_value),
                        "best_metric_epoch": epoch,
                    }
                    _save_checkpoint(
                        output_root / "checkpoints" / f"best_{metric_name}.pt",
                        diffusion,
                        optimizer,
                        epoch,
                        cfg,
                        metric_stats,
                    )
                    save_json(best_sample_metrics, output_root / "checkpoints" / "best_sample_metrics.json")

        append_csv(log_dir / "epoch_metrics.csv", row)
        append_jsonl(log_dir / "epoch_metrics.jsonl", row)

    if args.sample_after_train:
        sample_and_score(
            diffusion,
            valid_ds,
            output_root,
            device,
            mean,
            std,
            sample_count,
            cfg,
            tag="final",
        )


@torch.no_grad()
def sample_and_score(
    diffusion: GaussianDiffusion,
    dataset: CMTSGDataset,
    output_root: Path,
    device: torch.device,
    mean: np.ndarray,
    std: np.ndarray,
    sample_count: int,
    cfg: dict,
    tag: str = "valid",
) -> dict[str, float | str]:
    diffusion.eval()
    count = min(sample_count, len(dataset))
    batch = [dataset[i] for i in range(count)]
    text_emb = torch.stack([item["text_emb"] for item in batch], dim=0).to(device)
    semantic_atoms = (
        torch.stack([item["semantic_atoms"] for item in batch], dim=0).to(device)
        if batch and "semantic_atoms" in batch[0]
        else None
    )
    eval_cfg = cfg.get("evaluation", {})
    if bool(eval_cfg.get("use_gaf_condition", True)):
        gaf = torch.stack([item["gaf"] for item in batch], dim=0).to(device)
    else:
        gaf = None
    gen_norm = diffusion.sample((count, dataset.ts.shape[1], dataset.ts.shape[2]), text_emb, gaf, semantic_atoms=semantic_atoms)
    gen = gen_norm.cpu().numpy() * std + mean
    real = dataset.ts[:count]
    text_np = text_emb.detach().cpu().numpy()
    sample_dir = ensure_dir(output_root / "samples" / tag)
    np.save(sample_dir / "valid_samples.npy", gen.astype(np.float32))
    scores: dict[str, float | str] = {
        "mdd": mdd(real, gen),
        "flat_kl": flat_kl(real, gen),
        "mmd_rbf": mmd_rbf(real, gen),
        "fid_raw_proxy": fid_raw(real, gen),
        "jftsd_text_proxy": jftsd_text_proxy(real, gen, text_np),
    }
    require_cttp = bool(eval_cfg.get("require_cttp", False))
    cttp_root = eval_cfg.get("cttp_root")
    verbalts_root = eval_cfg.get("verbalts_root")
    if cttp_root and verbalts_root:
        captions = [str(dataset.caps[i, 0]) for i in range(count)]
        try:
            cttp_scores = compute_cttp_metrics(
                real,
                gen,
                captions,
                verbalts_root=verbalts_root,
                cttp_root=cttp_root,
                device=str(device),
                batch_size=int(eval_cfg.get("cttp_batch_size", 128)),
            )
            scores.update(cttp_scores)
            scores["cttp_status"] = "ok"
        except Exception as exc:
            scores["cttp_status"] = f"failed: {type(exc).__name__}: {exc}"
            if require_cttp:
                save_json(scores, sample_dir / "metrics_failed.json")
                raise RuntimeError(
                    "CTTP metrics are required but failed. "
                    f"Set evaluation.require_cttp=false only for debugging. Cause: {type(exc).__name__}: {exc}"
                ) from exc
    else:
        scores["cttp_status"] = "missing_config: set evaluation.verbalts_root and evaluation.cttp_root"
        if require_cttp:
            save_json(scores, sample_dir / "metrics_failed.json")
            raise RuntimeError("CTTP metrics are required but evaluation.verbalts_root/evaluation.cttp_root is missing.")
    save_json(scores, sample_dir / "metrics.json")
    append_jsonl(output_root / "logs" / "sample_metrics.jsonl", {"tag": tag, **scores})
    print(f"sample_metrics={scores}")
    return scores


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CMTSG DiT diffusion model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--processed-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--sample-after-train", action="store_true")
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--sample-every", type=int, default=None)
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
