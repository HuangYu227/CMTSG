from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from cmtsg.config import load_config
from cmtsg.data import load_text_caps, load_ts
from cmtsg.imaging import gasf_multivariate
from cmtsg.semantic_metrics import CTTPMetricEvaluator
from cmtsg.train import _make_model
from cmtsg.utils import ensure_dir, resolve_path, save_json, set_seed


def _load_checkpoint(path: str | Path, device: torch.device) -> dict:
    checkpoint = torch.load(resolve_path(path), map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Expected a CMTSG checkpoint with a 'model' key: {path}")
    return checkpoint


def _select_indices(n_items: int, n_options: int, policy: str, seed: int) -> np.ndarray:
    if policy == "first":
        return np.zeros(n_items, dtype=np.int64)
    elif policy == "random":
        rng = random.Random(seed)
        return np.array([rng.randint(0, n_options - 1) for _ in range(n_items)], dtype=np.int64)
    else:
        try:
            fixed = int(policy)
        except ValueError as exc:
            raise ValueError(f"Unsupported caption policy: {policy}") from exc
        if fixed < 0 or fixed >= n_options:
            raise ValueError(f"Caption index {fixed} out of range for {n_options} options")
        return np.full(n_items, fixed, dtype=np.int64)


def _select_caption_column(caps: np.ndarray, policy: str, seed: int, indices: np.ndarray | None = None) -> list[str]:
    if caps.ndim == 1:
        caps = caps[:, None]
    if indices is None:
        indices = _select_indices(caps.shape[0], caps.shape[1], policy, seed)
    if indices.shape[0] != caps.shape[0]:
        raise ValueError(f"Caption index count mismatch: {indices.shape[0]} vs {caps.shape[0]}")
    return [str(caps[i, indices[i]]) for i in range(caps.shape[0])]


def _load_metric_captions(
    data_root: Path,
    processed_root: Path,
    split: str,
    source: str,
    policy: str,
    seed: int,
    indices: np.ndarray | None = None,
) -> list[str]:
    if source == "original":
        return _select_caption_column(load_text_caps(data_root / f"{split}_text_caps.npy"), policy, seed, indices)
    if source == "causal":
        path = processed_root / f"{split}_causal_text.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        return _select_caption_column(np.load(path, allow_pickle=True).astype(str), policy, seed, indices)
    raise ValueError(f"Unsupported caption source: {source}")


def _load_condition_embeddings(processed_root: Path, split: str, policy: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    path = processed_root / f"{split}_text_emb.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    emb = np.load(path).astype(np.float32)
    if emb.ndim == 2:
        emb = emb[:, None, :]
    indices = _select_indices(emb.shape[0], emb.shape[1], policy, seed)
    return emb[np.arange(emb.shape[0]), indices], indices


def _load_semantic_atoms(processed_root: Path, split: str, base_embeddings: np.ndarray, max_eval_samples: int | None) -> np.ndarray | None:
    path = processed_root / f"{split}_semantic_atoms.npy"
    if path.exists():
        atoms = np.load(path).astype(np.float32)
        if atoms.ndim == 2:
            atoms = atoms[:, None, :]
        if atoms.ndim != 3:
            raise ValueError(f"Expected semantic atoms [N,C,D], got {atoms.shape}")
    elif base_embeddings.ndim == 3 and base_embeddings.shape[1] > 1:
        atoms = base_embeddings.astype(np.float32)
    else:
        return None
    if max_eval_samples is not None:
        atoms = atoms[:max_eval_samples]
    return atoms


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
def _generate_median(
    diffusion,
    eval_ts: np.ndarray,
    text_emb: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    seq_len: int,
    n_vars: int,
    gaf_max_size: int,
    batch_size: int,
    n_samples: int,
    sampler: str,
    device: torch.device,
    gaf_mode: str,
    gaf_seed: int,
    semantic_atoms: np.ndarray | None,
) -> np.ndarray:
    preds = []
    gaf_indices = _build_gaf_indices(text_emb.shape[0], gaf_mode, gaf_seed)
    for start in tqdm(range(0, text_emb.shape[0], batch_size), desc="generate"):
        end = start + batch_size
        emb = torch.from_numpy(text_emb[start:end]).to(device)
        sem = torch.from_numpy(semantic_atoms[start:end]).to(device) if semantic_atoms is not None else None
        if gaf_mode == "none":
            gaf = None
        elif gaf_mode == "real":
            gaf_np = np.stack([gasf_multivariate(sample, max_size=gaf_max_size) for sample in eval_ts[start:end]], axis=0)
            gaf = torch.from_numpy(gaf_np).to(device)
        else:
            if gaf_indices is None:
                raise RuntimeError(f"Internal GAF index error for mode={gaf_mode}")
            gaf_np = np.stack(
                [gasf_multivariate(eval_ts[int(gaf_indices[idx])], max_size=gaf_max_size) for idx in range(start, end)],
                axis=0,
            )
            gaf = torch.from_numpy(gaf_np).to(device)
        sample_preds = []
        for _ in range(n_samples):
            gen_norm = diffusion.sample((emb.shape[0], seq_len, n_vars), emb, gaf, sampler=sampler, semantic_atoms=sem)
            sample_preds.append(gen_norm.cpu().numpy() * std + mean)
        preds.append(np.median(np.stack(sample_preds, axis=0), axis=0))
    return np.concatenate(preds, axis=0).astype(np.float32)


def _append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = _load_checkpoint(args.checkpoint, device)
    cfg = load_config(args.config) if args.config else checkpoint["config"]

    data_root = resolve_path(args.data_root or cfg["data_root"])
    processed_root = resolve_path(args.processed_root or cfg["processed_root"])
    output_root = ensure_dir(args.output_root) if args.output_root else resolve_path(cfg["output_root"])
    eval_root = ensure_dir(output_root / "verbalts_protocol")

    train_ts = load_ts(data_root / "train_ts.npy")
    eval_ts = load_ts(data_root / f"{args.split}_ts.npy")
    if args.max_eval_samples is not None:
        eval_ts = eval_ts[: args.max_eval_samples]

    stats = checkpoint.get("stats") or {}
    mean = np.array(stats.get("mean"), dtype=np.float32) if "mean" in stats else train_ts.mean(axis=(0, 1), keepdims=True)
    std = np.array(stats.get("std"), dtype=np.float32) if "std" in stats else train_ts.std(axis=(0, 1), keepdims=True) + 1e-6
    seq_len, n_vars = int(train_ts.shape[1]), int(train_ts.shape[2])
    gaf_max_size = int(cfg.get("gaf_max_size", 384))
    gaf_size = min(seq_len, gaf_max_size)

    diffusion = _make_model(cfg, seq_len, n_vars, gaf_size).to(device)
    diffusion.load_state_dict(checkpoint["model"])
    diffusion.eval()

    emb_path = processed_root / f"{args.split}_text_emb.npy"
    all_condition_embeddings = np.load(emb_path).astype(np.float32)
    if all_condition_embeddings.ndim == 2:
        all_condition_embeddings = all_condition_embeddings[:, None, :]
    text_emb, condition_indices = _load_condition_embeddings(processed_root, args.split, args.condition_policy, args.condition_seed)
    if args.max_eval_samples is not None:
        text_emb = text_emb[: args.max_eval_samples]
        condition_indices = condition_indices[: args.max_eval_samples]
        all_condition_embeddings = all_condition_embeddings[: args.max_eval_samples]
    semantic_atoms = _load_semantic_atoms(processed_root, args.split, all_condition_embeddings, args.max_eval_samples)
    objective = str(cfg.get("diffusion", {}).get("objective", "rectified_flow")).lower()
    sampler = args.sampler or ("heun" if objective in {"rectified_flow", "flow", "flow_matching"} else "ddim")
    gaf_mode = args.gaf_mode or ("none" if args.text_only_sampling else "real")
    gen = _generate_median(
        diffusion,
        eval_ts,
        text_emb,
        mean,
        std,
        seq_len,
        n_vars,
        gaf_max_size,
        args.batch_size,
        args.n_samples,
        sampler,
        device,
        gaf_mode,
        args.gaf_seed,
        semantic_atoms,
    )

    if args.save_pred:
        np.save(eval_root / f"{args.tag}_pred.npy", gen)

    reference_caption_policy = args.reference_caption_policy or (
        "random" if args.metric_caption_policy == "condition" else args.metric_caption_policy
    )
    train_captions = _load_metric_captions(
        data_root,
        processed_root,
        "train",
        args.metric_caption_source,
        reference_caption_policy,
        args.metric_caption_seed,
    )
    eval_caption_indices = condition_indices if args.metric_caption_policy == "condition" else None
    eval_caption_policy = "first" if args.metric_caption_policy == "condition" else args.metric_caption_policy
    eval_captions = _load_metric_captions(
        data_root,
        processed_root,
        args.split,
        args.metric_caption_source,
        eval_caption_policy,
        args.metric_caption_seed + 1,
        eval_caption_indices,
    )
    if args.max_eval_samples is not None:
        eval_captions = eval_captions[: args.max_eval_samples]

    cache_folder = args.cache_folder
    if cache_folder is None:
        cache_folder = eval_root / f"cache_{args.metric_caption_source}_{args.metric_caption_policy}"

    evaluator = CTTPMetricEvaluator(args.verbalts_root, args.cttp_root, str(device))
    scores = evaluator.compute_verbalts_protocol(
        train_ts=train_ts,
        train_captions=train_captions,
        gen_ts=gen,
        eval_captions=eval_captions,
        cache_folder=cache_folder,
        batch_size=args.cttp_batch_size,
    )
    row = {
        "tag": args.tag,
        "checkpoint": str(resolve_path(args.checkpoint)),
        "split": args.split,
        "sampler": sampler,
        "text_only_sampling": bool(args.text_only_sampling),
        "gaf_mode": gaf_mode,
        "gaf_seed": int(args.gaf_seed),
        "n_samples": args.n_samples,
        "batch_size": args.batch_size,
        "num_eval": int(gen.shape[0]),
        "condition_policy": args.condition_policy,
        "metric_caption_source": args.metric_caption_source,
        "metric_caption_policy": args.metric_caption_policy,
        "reference_caption_policy": reference_caption_policy,
        "cache_folder": str(resolve_path(cache_folder)),
        **scores,
    }
    save_json(row, eval_root / f"{args.tag}_metrics.json")
    _append_csv(eval_root / "metrics.csv", row)
    print(json.dumps(row, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate CMTSG checkpoints with the VerbalTS FID/JFTSD/CTTP protocol.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--processed-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--verbalts-root", default="../VerbalTS")
    parser.add_argument("--cttp-root", required=True)
    parser.add_argument("--cache-folder", default=None)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--tag", default="eval")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cttp-batch-size", type=int, default=128)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--sampler", choices=["ddim", "ddpm", "euler", "heun"], default=None)
    parser.add_argument("--text-only-sampling", action="store_true", help="Do not condition generation on evaluation time-series GADF.")
    parser.add_argument("--gaf-mode", choices=["real", "none", "shuffle", "random"], default=None)
    parser.add_argument("--gaf-seed", type=int, default=123)
    parser.add_argument("--condition-policy", default="first", help="first, random, or a numeric cached embedding index.")
    parser.add_argument("--condition-seed", type=int, default=42)
    parser.add_argument("--metric-caption-source", choices=["original", "causal"], default="original")
    parser.add_argument("--reference-caption-policy", default=None, help="Policy for train-reference captions. Defaults to metric-caption-policy.")
    parser.add_argument("--metric-caption-policy", default="condition", help="condition, first, random, or a numeric caption index.")
    parser.add_argument("--metric-caption-seed", type=int, default=42)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--save-pred", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
