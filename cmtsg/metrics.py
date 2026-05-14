from __future__ import annotations

import numpy as np


def flat_kl(real: np.ndarray, gen: np.ndarray, bins: int = 80) -> float:
    real_values = np.asarray(real, dtype=np.float64).ravel()
    gen_values = np.asarray(gen, dtype=np.float64).ravel()
    lo = min(real_values.min(), gen_values.min())
    hi = max(real_values.max(), gen_values.max())
    p, edges = np.histogram(real_values, bins=bins, range=(lo, hi), density=True)
    q, _ = np.histogram(gen_values, bins=edges, density=True)
    p = p + 1e-8
    q = q + 1e-8
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def mdd(real: np.ndarray, gen: np.ndarray) -> float:
    real_mean = np.asarray(real).mean(axis=0)
    gen_mean = np.asarray(gen).mean(axis=0)
    return float(np.mean(np.abs(real_mean - gen_mean)))


def mmd_rbf(real: np.ndarray, gen: np.ndarray, gamma: float | None = None, max_samples: int = 512) -> float:
    x = np.asarray(real, dtype=np.float64).reshape(real.shape[0], -1)
    y = np.asarray(gen, dtype=np.float64).reshape(gen.shape[0], -1)
    if x.shape[0] > max_samples:
        x = x[:max_samples]
    if y.shape[0] > max_samples:
        y = y[:max_samples]
    if gamma is None:
        gamma = 1.0 / max(1, x.shape[1])
    xx = np.exp(-gamma * ((x[:, None] - x[None]) ** 2).sum(axis=-1)).mean()
    yy = np.exp(-gamma * ((y[:, None] - y[None]) ** 2).sum(axis=-1)).mean()
    xy = np.exp(-gamma * ((x[:, None] - y[None]) ** 2).sum(axis=-1)).mean()
    return float(xx + yy - 2.0 * xy)
