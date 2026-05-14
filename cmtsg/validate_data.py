from __future__ import annotations

import argparse

from cmtsg.data import SPLITS, validate_split
from cmtsg.utils import normalize_dataset_name, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate VerbalTS-style CMTSG dataset files.")
    parser.add_argument("--dataset", required=True, choices=["weather", "synth-m"])
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    dataset = normalize_dataset_name(args.dataset)
    data_root = resolve_path(args.data_root or f"datasets/{dataset}")
    for split in SPLITS:
        n, length, n_vars, n_caps = validate_split(data_root, split)
        print(f"{split}: samples={n} length={length} vars={n_vars} captions={n_caps}")


if __name__ == "__main__":
    main()
