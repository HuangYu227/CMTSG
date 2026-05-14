from __future__ import annotations

from typing import Literal

import numpy as np


GafKind = Literal["GASF", "GADF"]


def paa_1d(series: np.ndarray, output_size: int) -> np.ndarray:
    values = np.asarray(series, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"PAA expects 1D input, got {values.shape}")
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    if len(values) == output_size:
        return values.astype(np.float32, copy=False)
    if len(values) < output_size:
        raise ValueError("PAA output_size cannot exceed input length")

    edges = np.linspace(0, len(values), output_size + 1)
    out = np.empty(output_size, dtype=np.float32)
    for i in range(output_size):
        start = int(np.floor(edges[i]))
        end = int(np.floor(edges[i + 1]))
        end = max(end, start + 1)
        out[i] = values[start:end].mean()
    return out


def minmax_scale_1d(series: np.ndarray) -> np.ndarray:
    values = np.asarray(series, dtype=np.float32)
    span = float(values.max() - values.min())
    if span < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - values.min()) / span).astype(np.float32)


def gramian_angular_field(series: np.ndarray, kind: GafKind = "GASF") -> np.ndarray:
    cos_values = np.asarray(series, dtype=np.float32)
    if cos_values.ndim != 1:
        raise ValueError(f"GAF expects 1D input, got {cos_values.shape}")
    if np.any(cos_values < -1.0) or np.any(cos_values > 1.0):
        raise ValueError("GAF input must be scaled to [0, 1] or [-1, 1]")
    cos_values = np.clip(cos_values, -1.0, 1.0)
    sin_values = np.sqrt(np.clip(1.0 - cos_values**2, 0.0, 1.0))
    if kind == "GASF":
        gaf = np.outer(cos_values, cos_values) - np.outer(sin_values, sin_values)
    elif kind == "GADF":
        gaf = np.outer(sin_values, cos_values) - np.outer(cos_values, sin_values)
    else:
        raise ValueError(f"Unknown GAF kind: {kind}")
    return ((gaf + 1.0) * 0.5).astype(np.float32)


def gasf_multivariate(sample: np.ndarray, max_size: int = 64) -> np.ndarray:
    values = np.asarray(sample, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"Expected sample shape [L,K], got {values.shape}")
    length, n_vars = values.shape
    size = min(length, max_size)
    images = []
    for var_idx in range(n_vars):
        series = minmax_scale_1d(values[:, var_idx])
        if length > size:
            series = paa_1d(series, size)
        images.append(gramian_angular_field(series, "GASF"))
    return np.stack(images, axis=0).astype(np.float32)


def chart_stats(sample: np.ndarray) -> dict[str, object]:
    values = np.asarray(sample, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    start = values[0]
    end = values[-1]
    slope = end - start
    return {
        "length": int(values.shape[0]),
        "variables": int(values.shape[1]),
        "global_min": float(np.nanmin(values)),
        "global_max": float(np.nanmax(values)),
        "mean": np.nanmean(values, axis=0).round(6).tolist(),
        "std": np.nanstd(values, axis=0).round(6).tolist(),
        "start": start.round(6).tolist(),
        "end": end.round(6).tolist(),
        "end_minus_start": slope.round(6).tolist(),
    }
