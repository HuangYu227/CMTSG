from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from cmtsg.config import load_config
from cmtsg.data import load_ts
from cmtsg.dataset import CMTSGDataset
from cmtsg.metrics import t2s_metric_suite
from cmtsg.train import _make_model
from cmtsg.utils import ensure_dir, resolve_path, save_json, set_seed


def _load_checkpoint(path: str | Path, device: torch.device) -> dict:
    checkpoint = torch.load(resolve_path(path), map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Expected CMTSG checkpoint with a 'model' key: {path}")
    return checkpoint


def _append_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _aggregate(samples: list[np.ndarray], mode: str) -> np.ndarray:
    stacked = np.stack(samples, axis=0)
    if mode == "median":
        return np.median(stacked, axis=0)
    if mode == "mean":
        return stacked.mean(axis=0)
    raise ValueError(f"Unsupported aggregation: {mode}")


def _build_gaf_indices(total: int, mode: str, seed: int) -> np.ndarray | None:
    if mode in {"real", "none"}:
        return None
    rng = np.random.default_rng(seed)
    if mode == "shuffle":
        indices = rng.permutation(total)
        if total > 1:
            fixed = indices == np.arange(total)
            if fixed.any():
                indices[fixed] = np.roll(indices, 1)[fixed]
        return indices.astype(np.int64)
    if mode == "random":
        indices = rng.integers(0, total, size=total, dtype=np.int64)
        if total > 1:
            fixed = indices == np.arange(total)
            indices[fixed] = (indices[fixed] + 1) % total
        return indices
    raise ValueError(f"Unsupported gaf_mode: {mode}")


@torch.no_grad()
def generate(
    diffusion,
    dataset: CMTSGDataset,
    device: torch.device,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    n_samples: int,
    aggregation: str,
    sampler: str | None,
    gaf_mode: str,
    gaf_seed: int,
    max_eval_samples: int | None,
) -> np.ndarray:
    diffusion.eval()
    total = len(dataset) if max_eval_samples is None else min(len(dataset), max_eval_samples)
    gaf_indices = _build_gaf_indices(total, gaf_mode, gaf_seed)
    preds: list[np.ndarray] = []
    for start in tqdm(range(0, total, batch_size), desc="generate"):
        end = min(start + batch_size, total)
        items = [dataset[idx] for idx in range(start, end)]
        text_emb = torch.stack([item["text_emb"] for item in items], dim=0).to(device)
        semantic_atoms = (
            torch.stack([item["semantic_atoms"] for item in items], dim=0).to(device)
            if items and "semantic_atoms" in items[0]
            else None
        )
        if gaf_mode == "none":
            gaf = None
        elif gaf_mode == "real":
            gaf = torch.stack([item["gaf"] for item in items], dim=0).to(device)
        else:
            if gaf_indices is None:
                raise RuntimeError(f"Internal GAF index error for mode={gaf_mode}")
            gaf_items = [dataset[int(gaf_indices[idx])] for idx in range(start, end)]
            gaf = torch.stack([item["gaf"] for item in gaf_items], dim=0).to(device)
        sample_preds = []
        for _ in range(n_samples):
            gen_norm = diffusion.sample(
                (end - start, dataset.ts.shape[1], dataset.ts.shape[2]),
                text_emb,
                gaf,
                sampler=sampler,
                semantic_atoms=semantic_atoms,
            )
            sample_preds.append(gen_norm.cpu().numpy() * std + mean)
        preds.append(_aggregate(sample_preds, aggregation).astype(np.float32))
    return np.concatenate(preds, axis=0)


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = _load_checkpoint(args.checkpoint, device)
    cfg = load_config(args.config) if args.config else checkpoint["config"]
    data_root = resolve_path(args.data_root or cfg["data_root"])
    processed_root = resolve_path(args.processed_root or cfg["processed_root"])
    output_root = ensure_dir(args.output_root or cfg["output_root"])
    eval_root = ensure_dir(output_root / "t2s_protocol" / args.split)

    train_ts = load_ts(data_root / "train_ts.npy")
    seq_len, n_vars = train_ts.shape[1], train_ts.shape[2]
    stats = checkpoint.get("stats") or {}
    mean = np.asarray(stats.get("mean"), dtype=np.float32) if "mean" in stats else train_ts.mean(axis=(0, 1), keepdims=True)
    std = np.asarray(stats.get("std"), dtype=np.float32) if "std" in stats else train_ts.std(axis=(0, 1), keepdims=True) + 1e-6
    gaf_max_size = int(cfg.get("gaf_max_size", 384))
    diffusion = _make_model(cfg, seq_len, n_vars, min(seq_len, gaf_max_size)).to(device)
    diffusion.load_state_dict(checkpoint["model"])

    dataset = CMTSGDataset(
        data_root=data_root,
        processed_root=processed_root,
        split=args.split,
        mean=mean,
        std=std,
        train=False,
        gaf_max_size=gaf_max_size,
    )
    objective = str(cfg.get("diffusion", {}).get("objective", "rectified_flow")).lower()
    sampler = args.sampler or ("heun" if objective in {"rectified_flow", "flow", "flow_matching"} else "ddim")
    gaf_mode = args.gaf_mode or ("real" if args.use_gaf_condition else "none")
    gen = generate(
        diffusion=diffusion,
        dataset=dataset,
        device=device,
        mean=mean,
        std=std,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        aggregation=args.aggregation,
        sampler=sampler,
        gaf_mode=gaf_mode,
        gaf_seed=args.gaf_seed,
        max_eval_samples=args.max_eval_samples,
    )
    real = dataset.ts[: gen.shape[0]]
    scores = t2s_metric_suite(real, gen)
    tag = args.tag or f"{cfg.get('dataset', data_root.name)}_{args.split}"
    pred_path = eval_root / f"{tag}_pred.npy"
    np.save(pred_path, gen.astype(np.float32))
    row: dict[str, object] = {
        "tag": tag,
        "dataset": str(cfg.get("dataset", data_root.name)),
        "split": args.split,
        "checkpoint": str(resolve_path(args.checkpoint)),
        "num_eval": int(gen.shape[0]),
        "seq_len": int(gen.shape[1]),
        "n_vars": int(gen.shape[2]),
        "sampler": sampler,
        "n_samples": int(args.n_samples),
        "aggregation": args.aggregation,
        "use_gaf_condition": bool(gaf_mode != "none"),
        "gaf_mode": gaf_mode,
        "gaf_seed": int(args.gaf_seed),
        "pred_path": str(pred_path),
        **scores,
    }
    save_json(row, eval_root / f"{tag}_metrics.json")
    _append_csv(eval_root / "metrics.csv", row)
    if args.summary_csv:
        _append_csv(resolve_path(args.summary_csv), row)
    print(json.dumps(row, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate CMTSG with T2S full multimodal protocol metrics.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--processed-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--aggregation", choices=["median", "mean"], default="median")
    parser.add_argument("--sampler", choices=["ddim", "ddpm", "euler", "heun"], default=None)
    parser.add_argument("--use-gaf-condition", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gaf-mode", choices=["real", "none", "shuffle", "random"], default=None)
    parser.add_argument("--gaf-seed", type=int, default=123)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--summary-csv", default=None)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
