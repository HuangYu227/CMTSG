from __future__ import annotations

import numpy as np


def _as_3d_pair(real: np.ndarray, gen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    real_arr = np.asarray(real, dtype=np.float64)
    gen_arr = np.asarray(gen, dtype=np.float64)
    if real_arr.ndim == 2:
        real_arr = real_arr[..., None]
    if gen_arr.ndim == 2:
        gen_arr = gen_arr[..., None]
    if real_arr.ndim != 3 or gen_arr.ndim != 3:
        raise ValueError(f"Expected [N,L,K] arrays, got {real_arr.shape} and {gen_arr.shape}")
    if real_arr.shape != gen_arr.shape:
        raise ValueError(f"Shape mismatch: {real_arr.shape} vs {gen_arr.shape}")
    return real_arr, gen_arr


def mse(real: np.ndarray, gen: np.ndarray) -> float:
    real_arr, gen_arr = _as_3d_pair(real, gen)
    return float(np.mean((real_arr - gen_arr) ** 2))


def wape(real: np.ndarray, gen: np.ndarray, eps: float = 1e-8) -> float:
    real_arr, gen_arr = _as_3d_pair(real, gen)
    numerator = np.abs(real_arr - gen_arr).sum()
    denominator = np.abs(real_arr).sum()
    return float(numerator / max(float(denominator), eps))


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
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Inputs contain non-finite values")
    if hi <= lo:
        return 0.0
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


def mdd_histogram(real: np.ndarray, gen: np.ndarray, bins: int = 50, eps: float = 1e-8) -> float:
    real_arr, gen_arr = _as_3d_pair(real, gen)
    _, seq_len, n_vars = real_arr.shape
    distances = []
    for t_idx in range(seq_len):
        for var_idx in range(n_vars):
            real_values = real_arr[:, t_idx, var_idx]
            gen_values = gen_arr[:, t_idx, var_idx]
            lo = min(real_values.min(), gen_values.min())
            hi = max(real_values.max(), gen_values.max())
            if hi <= lo:
                distances.append(0.0)
                continue
            p, edges = np.histogram(real_values, bins=bins, range=(lo, hi), density=False)
            q, _ = np.histogram(gen_values, bins=edges, density=False)
            p = p.astype(np.float64) + eps
            q = q.astype(np.float64) + eps
            p = p / p.sum()
            q = q / q.sum()
            distances.append(np.mean(np.abs(p - q)))
    return float(np.mean(distances)) if distances else 0.0


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


def _acf(arr: np.ndarray, max_lag: int, eps: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr - arr.mean(axis=(0, 1), keepdims=True)
    var = arr.var(axis=(0, 1)) + eps
    values = []
    for lag in range(max_lag):
        if lag == 0:
            cov = np.mean(arr * arr, axis=(0, 1))
        else:
            cov = np.mean(arr[:, lag:, :] * arr[:, :-lag, :], axis=(0, 1))
        values.append(cov / var)
    return np.stack(values, axis=0)


def acf_error(real: np.ndarray, gen: np.ndarray, max_lag: int = 64, eps: float = 1e-8) -> float:
    real_arr, gen_arr = _as_3d_pair(real, gen)
    max_lag = max(1, min(int(max_lag), real_arr.shape[1]))
    diff = _acf(real_arr, max_lag, eps) - _acf(gen_arr, max_lag, eps)
    per_var = np.sqrt(np.sum(diff**2, axis=0))
    return float(np.nan_to_num(per_var, nan=0.0, posinf=1e6, neginf=1e6).mean())


def correlation_error(real: np.ndarray, gen: np.ndarray, eps: float = 1e-8) -> float:
    real_arr, gen_arr = _as_3d_pair(real, gen)
    n_vars = real_arr.shape[2]
    if n_vars <= 1:
        return 0.0
    real_flat = real_arr.reshape(-1, n_vars)
    gen_flat = gen_arr.reshape(-1, n_vars)
    real_std = real_flat.std(axis=0)
    gen_std = gen_flat.std(axis=0)
    valid = (real_std > eps) & (gen_std > eps)
    if valid.sum() <= 1:
        return 0.0
    real_corr = np.corrcoef(real_flat[:, valid], rowvar=False)
    gen_corr = np.corrcoef(gen_flat[:, valid], rowvar=False)
    mask = ~np.eye(real_corr.shape[0], dtype=bool)
    diff = np.abs(np.nan_to_num(real_corr - gen_corr, nan=0.0, posinf=0.0, neginf=0.0))
    return float(diff[mask].mean()) if mask.any() else 0.0


def t2s_metric_suite(real: np.ndarray, gen: np.ndarray) -> dict[str, float]:
    return {
        "MSE": mse(real, gen),
        "WAPE": wape(real, gen),
        "MDD": mdd_histogram(real, gen),
        "KL": flat_kl(real, gen),
        "MMD": mmd_rbf(real, gen),
        "ACF Error": acf_error(real, gen),
        "Correlation Error": correlation_error(real, gen),
    }


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
