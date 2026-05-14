from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cmtsg.config import get_nested, load_config
from cmtsg.data import load_ts
from cmtsg.dataset import CMTSGDataset
from cmtsg.env_bank import build_anchor_gaf
from cmtsg.metrics import flat_kl, mdd, mmd_rbf
from cmtsg.models import CMTSGModel
from cmtsg.models.diffusion import GaussianDiffusion
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
    }
    save_json(stats, output_root / "train_stats.json")

    for epoch in range(1, epochs + 1):
        diffusion.train()
        train_losses = []
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

        diffusion.eval()
        val_losses = []
        with torch.no_grad():
            for batch in valid_loader:
                x = batch["x"].to(device)
                text_emb = batch["text_emb"].to(device)
                loss, _ = diffusion.training_loss(x, text_emb, anchor_gaf)
                val_losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        print(f"epoch={epoch:04d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        _save_checkpoint(output_root / "checkpoints" / "last.pt", diffusion, optimizer, epoch, cfg, stats)
        if val_loss < best_val:
            best_val = val_loss
            _save_checkpoint(output_root / "checkpoints" / "best.pt", diffusion, optimizer, epoch, cfg, stats)
        if epoch % int(get_nested(cfg, "training.save_every", 10)) == 0:
            _save_checkpoint(output_root / "checkpoints" / f"epoch_{epoch:04d}.pt", diffusion, optimizer, epoch, cfg, stats)

    if args.sample_after_train:
        sample_and_score(diffusion, valid_ds, anchor_gaf, output_root, device, mean, std, args.sample_count)


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
) -> None:
    diffusion.eval()
    count = min(sample_count, len(dataset))
    batch = [dataset[i] for i in range(count)]
    text_emb = torch.stack([item["text_emb"] for item in batch], dim=0).to(device)
    gen_norm = diffusion.sample((count, dataset.ts.shape[1], dataset.ts.shape[2]), text_emb, anchor_gaf)
    gen = gen_norm.cpu().numpy() * std + mean
    real = dataset.ts[:count]
    sample_dir = ensure_dir(output_root / "samples")
    np.save(sample_dir / "valid_samples.npy", gen.astype(np.float32))
    scores = {"mdd": mdd(real, gen), "flat_kl": flat_kl(real, gen), "mmd_rbf": mmd_rbf(real, gen)}
    save_json(scores, sample_dir / "metrics.json")
    print(f"sample_metrics={scores}")


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
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
