from __future__ import annotations

import numpy as np


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    mu1 = np.atleast_1d(mu1).astype(np.float64)
    mu2 = np.atleast_1d(mu2).astype(np.float64)
    sigma1 = np.atleast_2d(sigma1).astype(np.float64)
    sigma2 = np.atleast_2d(sigma2).astype(np.float64)
    if sigma1.shape != sigma2.shape:
        raise ValueError(f"Covariance shape mismatch: {sigma1.shape} vs {sigma2.shape}")
    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps
    diff = mu1 - mu2
    try:
        from scipy import linalg

        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        if isinstance(covmean, tuple):
            covmean = covmean[0]
    except Exception:
        eigvals, eigvecs = np.linalg.eig(sigma1.dot(sigma2))
        covmean = eigvecs.dot(np.diag(np.sqrt(np.clip(eigvals.real, 0.0, None)))).dot(np.linalg.inv(eigvecs))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    value = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)
    return float(max(value, 0.0))


def _mean_cov(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got {features.shape}")
    return features.mean(axis=0), np.cov(features, rowvar=False)


def fid_from_features(real_features: np.ndarray, gen_features: np.ndarray) -> float:
    mu_r, cov_r = _mean_cov(real_features)
    mu_g, cov_g = _mean_cov(gen_features)
    return frechet_distance(mu_r, cov_r, mu_g, cov_g)


def fid_raw(real: np.ndarray, gen: np.ndarray, max_dim: int = 256) -> float:
    real_features = np.asarray(real, dtype=np.float64).reshape(real.shape[0], -1)
    gen_features = np.asarray(gen, dtype=np.float64).reshape(gen.shape[0], -1)
    if real_features.shape[1] > max_dim:
        rng = np.random.default_rng(123)
        proj = rng.normal(0.0, 1.0 / np.sqrt(real_features.shape[1]), size=(real_features.shape[1], max_dim))
        real_features = real_features @ proj
        gen_features = gen_features @ proj
    return fid_from_features(real_features, gen_features)


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


def jftsd_text_proxy(real: np.ndarray, gen: np.ndarray, text_emb: np.ndarray, max_dim: int = 256) -> float:
    real_x = np.asarray(real, dtype=np.float64).reshape(real.shape[0], -1)
    gen_x = np.asarray(gen, dtype=np.float64).reshape(gen.shape[0], -1)
    text = np.asarray(text_emb, dtype=np.float64)
    if real_x.shape[1] > max_dim:
        rng = np.random.default_rng(456)
        proj = rng.normal(0.0, 1.0 / np.sqrt(real_x.shape[1]), size=(real_x.shape[1], max_dim))
        real_x = real_x @ proj
        gen_x = gen_x @ proj
    return fid_from_features(np.concatenate([real_x, text], axis=1), np.concatenate([gen_x, text], axis=1))
