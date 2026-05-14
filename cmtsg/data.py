from __future__ import annotations

from pathlib import Path

import numpy as np

from cmtsg.utils import resolve_path


SPLITS = ("train", "valid", "test")


def load_ts(path: str | Path) -> np.ndarray:
    arr = np.load(resolve_path(path), allow_pickle=False)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Time series must have shape [N,L,K], got {arr.shape}")
    return arr.astype(np.float32)


def load_text_caps(path: str | Path) -> np.ndarray:
    arr = np.load(resolve_path(path), allow_pickle=True)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Text captions must have shape [N,C], got {arr.shape}")
    return arr.astype(str)


def split_paths(data_root: str | Path, split: str) -> tuple[Path, Path]:
    if split not in SPLITS:
        raise ValueError(f"Invalid split: {split}")
    root = resolve_path(data_root)
    return root / f"{split}_ts.npy", root / f"{split}_text_caps.npy"


def validate_split(data_root: str | Path, split: str) -> tuple[int, int, int, int]:
    ts_path, caps_path = split_paths(data_root, split)
    if not ts_path.exists():
        raise FileNotFoundError(ts_path)
    if not caps_path.exists():
        raise FileNotFoundError(caps_path)
    ts = load_ts(ts_path)
    caps = load_text_caps(caps_path)
    if ts.shape[0] != caps.shape[0]:
        raise ValueError(f"Sample count mismatch: {ts.shape[0]} vs {caps.shape[0]}")
    return int(ts.shape[0]), int(ts.shape[1]), int(ts.shape[2]), int(caps.shape[1])
