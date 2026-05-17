from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

from cmtsg.data import load_ts
from cmtsg.imaging import gasf_multivariate
from cmtsg.utils import ensure_dir, resolve_path


def precompute_split(
    data_root: str | Path,
    split: str,
    max_size: int = 384,
    overwrite: bool = False,
    chunk_size: int = 1024,
) -> dict[str, object]:
    data_root = resolve_path(data_root)
    ts_path = data_root / f"{split}_ts.npy"
    out_path = data_root / f"{split}_gaf.npy"
    if not ts_path.exists():
        raise FileNotFoundError(ts_path)

    ts = load_ts(ts_path)
    if ts.ndim != 3:
        raise ValueError(f"Expected {ts_path} shape [N,L,K], got {ts.shape}")
    n_samples, seq_len, n_vars = ts.shape
    gaf_size = min(int(seq_len), int(max_size))
    expected_shape = (n_samples, n_vars, gaf_size, gaf_size)

    if out_path.exists() and not overwrite:
        cached = np.load(out_path, mmap_mode="r")
        if tuple(cached.shape) == expected_shape and cached.dtype == np.float32:
            del cached
            return {
                "split": split,
                "status": "exists",
                "path": str(out_path),
                "shape": list(expected_shape),
                "dtype": "float32",
            }
        shape = tuple(cached.shape)
        dtype = cached.dtype
        del cached
        raise ValueError(
            f"Existing GAF cache has wrong metadata: {out_path} shape={shape}, "
            f"dtype={dtype}, expected shape={expected_shape}, dtype=float32. "
            "Pass --overwrite to regenerate it."
        )

    ensure_dir(data_root)
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=expected_shape)
    for start in tqdm(range(0, n_samples, chunk_size), desc=f"{data_root.name}:{split}:gaf"):
        end = min(start + chunk_size, n_samples)
        for idx in range(start, end):
            out[idx] = gasf_multivariate(ts[idx], max_size=max_size)
        out.flush()

    meta = {
        "split": split,
        "status": "created",
        "path": str(out_path),
        "source_ts": str(ts_path),
        "shape": list(expected_shape),
        "dtype": "float32",
        "max_size": int(max_size),
        "seq_len": int(seq_len),
        "n_vars": int(n_vars),
        "samples": int(n_samples),
    }
    meta_path = data_root / f"{split}_gaf_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    del out
    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute CMTSG GADF caches as {split}_gaf.npy files.")
    parser.add_argument("--data-root", required=True, help="Dataset directory containing train_ts.npy etc.")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--max-size", type=int, default=384)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for split in args.splits:
        meta = precompute_split(
            data_root=args.data_root,
            split=split,
            max_size=args.max_size,
            overwrite=args.overwrite,
            chunk_size=args.chunk_size,
        )
        print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
