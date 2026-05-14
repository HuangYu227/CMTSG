from __future__ import annotations

from pathlib import Path

import numpy as np

from cmtsg.imaging import gasf_multivariate
from cmtsg.utils import resolve_path, save_json


def select_anchor_indices(n_samples: int, n_env: int, seed: int) -> np.ndarray:
    if n_samples <= 0:
        raise ValueError("Cannot select anchors from an empty dataset")
    rng = np.random.default_rng(seed)
    count = min(n_env, n_samples)
    return np.sort(rng.choice(n_samples, size=count, replace=False)).astype(np.int64)


def build_anchor_gaf(
    train_ts: np.ndarray,
    n_env: int = 12,
    seed: int = 42,
    max_size: int = 64,
    output_json: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    indices = select_anchor_indices(train_ts.shape[0], n_env, seed)
    gaf = np.stack([gasf_multivariate(train_ts[int(idx)], max_size=max_size) for idx in indices], axis=0)
    if output_json is not None:
        save_json({"seed": seed, "indices": indices.tolist()}, resolve_path(output_json))
    return indices, gaf.astype(np.float32)
