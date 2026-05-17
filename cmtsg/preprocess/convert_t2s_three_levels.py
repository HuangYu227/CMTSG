from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cmtsg.utils import ensure_dir, resolve_path


DATASETS = {
    "traffic": "traffic",
    "airquality": "airquality",
    "ettm1": "ETTm1",
}
HORIZONS = (24, 48, 96)


@dataclass(frozen=True)
class T2SRows:
    series: np.ndarray
    captions: np.ndarray
    embeddings: np.ndarray


class T2SSource:
    def __init__(self, source: str | Path) -> None:
        self.source = resolve_path(source)
        self._zip: zipfile.ZipFile | None = None
        if self.source.is_file() and self.source.suffix.lower() == ".zip":
            self._zip = zipfile.ZipFile(self.source)
            self._names = set(self._zip.namelist())
        elif self.source.is_dir():
            self._names = None
        else:
            raise FileNotFoundError(f"Expected a T2S data zip or directory: {self.source}")

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()

    def read_text(self, relative_path: str) -> str:
        candidates = [
            relative_path.replace("\\", "/"),
            f"Three Levels Data/{relative_path}".replace("\\", "/"),
        ]
        if self._zip is not None:
            for candidate in candidates:
                if candidate in self._names:
                    return self._zip.read(candidate).decode("utf-8", errors="replace")
            raise FileNotFoundError(f"{relative_path} not found in {self.source}")
        for candidate in candidates:
            path = self.source / candidate
            if path.exists():
                return path.read_text(encoding="utf-8", errors="replace")
        raise FileNotFoundError(f"{relative_path} not found under {self.source}")


def _parse_list(value: str) -> list[float]:
    value = str(value).strip()
    try:
        parsed = ast.literal_eval(value)
        return [float(item) for item in parsed]
    except Exception:
        cleaned = value.replace("[", " ").replace("]", " ").replace(",", " ")
        return [float(item) for item in cleaned.split()]


def _read_csv_rows(source: T2SSource, dataset: str, horizon: int) -> T2SRows:
    filename = f"TSFragment-600K/embedding_cleaned_{DATASETS[dataset]}_{horizon}.csv"
    text = source.read_text(filename)
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    reader = csv.DictReader(io.StringIO(text))
    series: list[list[float]] = []
    captions: list[str] = []
    embeddings: list[list[float]] = []
    for row_idx, row in enumerate(reader):
        try:
            ts_values = _parse_list(row["OT"])
            emb_values = _parse_list(row["TextEmbedding"])
        except KeyError as exc:
            raise ValueError(f"Missing required column {exc} in {filename}") from exc
        if len(ts_values) != horizon:
            raise ValueError(f"{filename} row {row_idx} has OT length {len(ts_values)}, expected {horizon}")
        series.append(ts_values)
        captions.append(str(row.get("Text", "")))
        embeddings.append(emb_values)
    if not series:
        raise ValueError(f"No rows found in {filename}")
    emb_dim = len(embeddings[0])
    if emb_dim <= 0:
        raise ValueError(f"Empty TextEmbedding in {filename}")
    if any(len(item) != emb_dim for item in embeddings):
        raise ValueError(f"Inconsistent TextEmbedding dimensions in {filename}")
    return T2SRows(
        series=np.asarray(series, dtype=np.float32)[:, :, None],
        captions=np.asarray(captions, dtype=object)[:, None],
        embeddings=np.asarray(embeddings, dtype=np.float32)[:, None, :],
    )


def _split_indices(
    n_items: int,
    seed: int,
    t2s_train_proportion: float,
    valid_ratio: float,
) -> dict[str, np.ndarray]:
    if n_items < 3:
        raise ValueError(f"Need at least 3 rows for train/valid/test split, got {n_items}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_items)
    regular_train_num = int(math.ceil(n_items * t2s_train_proportion))
    regular_train_num = min(max(regular_train_num, 2), n_items - 1)
    train_valid = perm[:regular_train_num]
    test = perm[regular_train_num:]
    if test.size == 0:
        test = train_valid[-1:]
        train_valid = train_valid[:-1]
    valid_count = max(1, int(round(n_items * valid_ratio)))
    valid_count = min(valid_count, train_valid.size - 1)
    valid = train_valid[-valid_count:]
    train = train_valid[:-valid_count]
    return {"train": train, "valid": valid, "test": test}


def _write_split(
    rows: T2SRows,
    indices: dict[str, np.ndarray],
    data_dir: Path,
    processed_dir: Path,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    for split, split_indices in indices.items():
        np.save(data_dir / f"{split}_ts.npy", rows.series[split_indices].astype(np.float32))
        np.save(data_dir / f"{split}_text_caps.npy", rows.captions[split_indices])
        np.save(processed_dir / f"{split}_text_emb.npy", rows.embeddings[split_indices].astype(np.float32))


def convert_one(
    source: T2SSource,
    dataset: str,
    horizon: int,
    data_root: Path,
    processed_root: Path,
    seed: int,
    t2s_train_proportion: float,
    valid_ratio: float,
) -> dict[str, object]:
    rows = _read_csv_rows(source, dataset, horizon)
    indices = _split_indices(rows.series.shape[0], seed, t2s_train_proportion, valid_ratio)
    name = f"{dataset}_{horizon}"
    data_dir = ensure_dir(data_root / name)
    processed_dir = ensure_dir(processed_root / name)
    _write_split(rows, indices, data_dir, processed_dir)
    meta = {
        "dataset": dataset,
        "horizon": horizon,
        "source": "TSFragment-600K",
        "samples": int(rows.series.shape[0]),
        "seq_len": int(rows.series.shape[1]),
        "n_vars": int(rows.series.shape[2]),
        "text_emb_dim": int(rows.embeddings.shape[-1]),
        "splits": {split: int(split_indices.size) for split, split_indices in indices.items()},
        "seed": int(seed),
        "t2s_train_proportion": float(t2s_train_proportion),
        "valid_ratio": float(valid_ratio),
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (processed_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert T2S Three Levels Data CSV files to CMTSG npy datasets.")
    parser.add_argument("--source", required=True, help="Path to Three Levels Data.zip or extracted Three Levels Data directory.")
    parser.add_argument("--data-root", default="datasets/t2s")
    parser.add_argument("--processed-root", default="processed/t2s")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS), choices=list(HORIZONS))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--t2s-train-proportion", type=float, default=0.99)
    parser.add_argument("--valid-ratio", type=float, default=0.01)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source = T2SSource(args.source)
    try:
        for dataset in args.datasets:
            for horizon in args.horizons:
                meta = convert_one(
                    source=source,
                    dataset=dataset,
                    horizon=horizon,
                    data_root=resolve_path(args.data_root),
                    processed_root=resolve_path(args.processed_root),
                    seed=args.seed,
                    t2s_train_proportion=args.t2s_train_proportion,
                    valid_ratio=args.valid_ratio,
                )
                print(
                    f"{dataset}_{horizon}: samples={meta['samples']} "
                    f"splits={meta['splits']} text_emb_dim={meta['text_emb_dim']}"
                )
    finally:
        source.close()


if __name__ == "__main__":
    main()
