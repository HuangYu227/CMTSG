from __future__ import annotations

import csv
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _write_tiny_t2s_zip(path: Path, n_rows: int = 12, horizon: int = 24, emb_dim: int = 128) -> None:
    rows = []
    for idx in range(n_rows):
        series = [float(idx + step) for step in range(horizon)]
        embedding = [float(idx) / 100.0 + float(dim) / 1000.0 for dim in range(emb_dim)]
        rows.append(
            {
                "SampleID": str(idx),
                "SampleNumID": str(idx),
                "TimeInterval": str(horizon),
                "Text": f"sample {idx}",
                "TextEmbedding": "[" + " ".join(str(value) for value in embedding) + "]",
                "OT": str(series),
            }
        )
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "embedding_cleaned_traffic_24.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with zipfile.ZipFile(path, "w") as zf:
            zf.write(csv_path, "Three Levels Data/TSFragment-600K/embedding_cleaned_traffic_24.csv")


def main() -> None:
    try:
        import numpy as np
    except Exception as exc:
        print(f"SKIP: NumPy is not installed ({exc})")
        return

    from cmtsg.metrics import acf_error, correlation_error, mse, t2s_metric_suite, wape
    from cmtsg.preprocess.convert_t2s_three_levels import T2SSource, convert_one
    from cmtsg.preprocess.precompute_gaf import precompute_split

    real = np.arange(4 * 24, dtype=np.float32).reshape(4, 24, 1)
    gen = real.copy()
    assert mse(real, gen) == 0.0
    assert wape(real, gen) == 0.0
    assert acf_error(real, gen) == 0.0
    assert correlation_error(real, gen) == 0.0
    scores = t2s_metric_suite(real, gen)
    for key in ("MSE", "WAPE", "MDD", "KL", "MMD", "ACF Error", "Correlation Error"):
        assert key in scores
        assert np.isfinite(scores[key]), key

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        zip_path = tmp_path / "three_levels.zip"
        _write_tiny_t2s_zip(zip_path)
        source = T2SSource(zip_path)
        try:
            meta = convert_one(
                source=source,
                dataset="traffic",
                horizon=24,
                data_root=tmp_path / "datasets",
                processed_root=tmp_path / "processed",
                seed=123,
                t2s_train_proportion=0.99,
                valid_ratio=0.01,
            )
        finally:
            source.close()
        assert meta["text_emb_dim"] == 128
        train_ts = np.load(tmp_path / "datasets" / "traffic_24" / "train_ts.npy")
        valid_ts = np.load(tmp_path / "datasets" / "traffic_24" / "valid_ts.npy")
        test_ts = np.load(tmp_path / "datasets" / "traffic_24" / "test_ts.npy")
        train_emb = np.load(tmp_path / "processed" / "traffic_24" / "train_text_emb.npy")
        assert train_ts.ndim == 3 and train_ts.shape[1:] == (24, 1)
        assert valid_ts.ndim == 3 and test_ts.ndim == 3
        assert train_emb.ndim == 3 and train_emb.shape[1:] == (1, 128)
        gaf_meta = precompute_split(tmp_path / "datasets" / "traffic_24", "train", max_size=384)
        assert gaf_meta["status"] == "created"
        train_gaf = np.load(tmp_path / "datasets" / "traffic_24" / "train_gaf.npy", mmap_mode="r")
        assert train_gaf.shape == (train_ts.shape[0], 1, 24, 24)
        del train_gaf
        cached_meta = precompute_split(tmp_path / "datasets" / "traffic_24", "train", max_size=384)
        assert cached_meta["status"] == "exists"
    print("test_t2s_protocol.py: OK")


if __name__ == "__main__":
    main()
