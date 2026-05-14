from __future__ import annotations

import argparse

from cmtsg.semantic_metrics import CTTPMetricEvaluator, discover_cttp_files
from cmtsg.utils import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate VerbalTS CTTP checkpoint/config loading.")
    parser.add_argument("--verbalts-root", required=True)
    parser.add_argument("--cttp-root", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config, checkpoint = discover_cttp_files(args.cttp_root)
    print(f"CTTP config: {config}")
    print(f"CTTP checkpoint: {checkpoint}")
    evaluator = CTTPMetricEvaluator(resolve_path(args.verbalts_root), resolve_path(args.cttp_root), args.device)
    print(f"Loaded CTTP on device: {evaluator.device}")


if __name__ == "__main__":
    main()
