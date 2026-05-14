from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from cmtsg.data import SPLITS, validate_split
from cmtsg.utils import ensure_dir, normalize_dataset_name, resolve_path


REQUIRED_SUFFIXES = ("ts.npy", "text_caps.npy", "attrs_idx.npy")


def copy_dataset(source: Path, target: Path, overwrite: bool = False) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        for suffix in REQUIRED_SUFFIXES:
            name = f"{split}_{suffix}"
            src = source / name
            dst = target / name
            if not src.exists():
                if suffix == "attrs_idx.npy":
                    continue
                raise FileNotFoundError(src)
            if dst.exists() and not overwrite:
                continue
            shutil.copy2(src, dst)
    meta = source / "meta.json"
    if meta.exists() and (overwrite or not (target / "meta.json").exists()):
        shutil.copy2(meta, target / "meta.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy VerbalTS-style npy dataset into CMTSG datasets/.")
    parser.add_argument("--dataset", required=True, choices=["weather", "synth-m"])
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset = normalize_dataset_name(args.dataset)
    source = resolve_path(args.source)
    target = ensure_dir(args.target or f"datasets/{dataset}")
    copy_dataset(source, target, overwrite=args.overwrite)
    for split in SPLITS:
        n, length, n_vars, n_caps = validate_split(target, split)
        print(f"{split}: samples={n} length={length} vars={n_vars} captions={n_caps}")
    print(f"Dataset ready: {target}")


if __name__ == "__main__":
    main()
