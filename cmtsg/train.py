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
from cmtsg.env_bank import build_anchor_gaf
from cmtsg.logging_utils import append_csv, append_jsonl
from cmtsg.metrics import fid_raw, flat_kl, jftsd_text_proxy, mdd, mmd_rbf
from cmtsg.models import CMTSGModel
from cmtsg.models.diffusion import GaussianDiffusion
from cmtsg.semantic_metrics import compute_cttp_metrics
from cmtsg.utils import ensure_dir, resolve_path, save_json, set_seed


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _make_model(cfg: dict, seq_len: int, n_vars: int, gaf_size: int) -> GaussianDiffusion:
    model_cfg = cfg.get("model", {})
    model = CMTSGModel(
        seq_len=seq_len,
        n_vars=n_vars,
        gaf_size=gaf_size,
        text_dim=int(cfg.get("text_emb_dim", 64)),
        env_dim=int(cfg.get("text_emb_dim", 64)),
        hidden_size=int(model_cfg.get("hidden_size", 128)),
        depth=int(model_cfg.get("depth", 4)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
        patch_size=int(model_cfg.get("patch_size", 1)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        n_env=int(cfg.get("n_env", 12)),
        use_text_condition=bool(model_cfg.get("use_text_condition", True)),
        use_env_condition=bool(model_cfg.get("use_env_condition", True)),
        env_source=str(model_cfg.get("env_source", "gaf")),
        routing=str(model_cfg.get("routing", "text")),
    )
    diff_cfg = cfg.get("diffusion", {})
    return GaussianDiffusion(
        model=model,
        num_steps=int(diff_cfg.get("num_steps", 50)),
        beta_start=float(diff_cfg.get("beta_start", 0.0001)),
        beta_end=float(diff_cfg.get("beta_end", 0.5)),
        schedule=str(diff_cfg.get("schedule", "quad")),
    )


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
    norm_train_ts = (train_ts - mean) / std

    n_env = int(cfg.get("n_env", 12))
    env_seed = int(cfg.get("env_seed", 42))
    gaf_max_size = int(cfg.get("gaf_max_size", 64))
    anchor_indices, anchor_gaf_np = build_anchor_gaf(
        norm_train_ts,
        n_env=n_env,
        seed=env_seed,
        max_size=gaf_max_size,
        output_json=output_root / "env_anchor_indices.json",
    )
    gaf_size = int(anchor_gaf_np.shape[-1])
    anchor_gaf = torch.from_numpy(anchor_gaf_np).to(device)

    train_ds = CMTSGDataset(data_root, processed_root, "train", mean=mean, std=std, train=True)
    valid_ds = CMTSGDataset(data_root, processed_root, "valid", mean=mean, std=std, train=False)
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
        "anchor_indices": anchor_indices.tolist(),
        "model_ablation": {
            "use_text_condition": bool(cfg.get("model", {}).get("use_text_condition", True)),
            "use_env_condition": bool(cfg.get("model", {}).get("use_env_condition", True)),
            "env_source": str(cfg.get("model", {}).get("env_source", "gaf")),
            "routing": str(cfg.get("model", {}).get("routing", "text")),
        },
    }
    save_json(stats, output_root / "train_stats.json")
    log_dir = ensure_dir(output_root / "logs")
    eval_cfg = cfg.get("evaluation", {})
    sample_every = int(args.sample_every if args.sample_every is not None else eval_cfg.get("sample_every", 0))
    sample_count = int(args.sample_count or eval_cfg.get("sample_count", 128))
    best_metric_specs = eval_cfg.get("best_metrics") or _default_best_metric_specs()
    best_sample_metrics: dict[str, float] = {}

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        diffusion.train()
        train_losses = []
        train_route_entropy = []
        train_route_max = []
        for batch in train_loader:
            x = batch["x"].to(device)
            text_emb = batch["text_emb"].to(device)
            loss, metrics = diffusion.training_loss(x, text_emb, anchor_gaf)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(diffusion.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            train_route_entropy.append(float(metrics["route_entropy"].detach().cpu()))
            train_route_max.append(float(metrics["route_max"].detach().cpu()))

        diffusion.eval()
        val_losses = []
        val_route_entropy = []
        val_route_max = []
        with torch.no_grad():
            for batch in valid_loader:
                x = batch["x"].to(device)
                text_emb = batch["text_emb"].to(device)
                loss, metrics = diffusion.training_loss(x, text_emb, anchor_gaf)
                val_losses.append(float(loss.detach().cpu()))
                val_route_entropy.append(float(metrics["route_entropy"].detach().cpu()))
                val_route_max.append(float(metrics["route_max"].detach().cpu()))
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_route_entropy": float(np.mean(train_route_entropy)),
            "train_route_max": float(np.mean(train_route_max)),
            "val_route_entropy": float(np.mean(val_route_entropy)),
            "val_route_max": float(np.mean(val_route_max)),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": float(time.time() - epoch_start),
        }
        print(
            f"epoch={epoch:04d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"route_max={row['val_route_max']:.4f}"
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
                anchor_gaf,
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
            anchor_gaf,
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
    anchor_gaf: torch.Tensor,
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
    gen_norm = diffusion.sample((count, dataset.ts.shape[1], dataset.ts.shape[2]), text_emb, anchor_gaf)
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
    eval_cfg = cfg.get("evaluation", {})
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
