from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmtsg.imaging import gasf_multivariate, gramian_angular_field, paa_1d


def test_paa_keeps_means() -> None:
    values = np.arange(8, dtype=np.float32)
    out = paa_1d(values, 4)
    assert np.allclose(out, [0.5, 2.5, 4.5, 6.5])


def test_gasf_range_and_shape() -> None:
    sample = np.stack([np.linspace(0, 1, 12), np.linspace(1, 0, 12)], axis=1)
    gaf = gasf_multivariate(sample, max_size=6)
    assert gaf.shape == (2, 6, 6)
    assert gaf.min() >= 0.0
    assert gaf.max() <= 1.0


def test_formula_is_normalized_to_zero_one() -> None:
    values = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    gaf = gramian_angular_field(values, "GASF")
    raw = np.outer(values, values) - np.outer(np.sqrt(1 - values**2), np.sqrt(1 - values**2))
    assert np.allclose(gaf, (raw + 1.0) * 0.5)


def test_gadf_uses_absolute_intensity() -> None:
    values = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    gadf = gramian_angular_field(values, "GADF")
    sin_values = np.sqrt(1 - values**2)
    raw = np.outer(sin_values, values) - np.outer(values, sin_values)
    assert np.allclose(gadf, np.abs(raw))
    assert gadf.min() >= 0.0
    assert gadf.max() <= 1.0


if __name__ == "__main__":
    test_paa_keeps_means()
    test_gasf_range_and_shape()
    test_formula_is_normalized_to_zero_one()
    test_gadf_uses_absolute_intensity()
    print("test_imaging.py: OK")
