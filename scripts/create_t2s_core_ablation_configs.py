from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config: {path}")
    return data


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def _with_output_root(cfg: dict, family: str, stem: str) -> dict:
    out = deepcopy(cfg)
    out["output_root"] = f"runs/t2s_ablation/{family}/{stem}"
    return out


def make_no_grounding(cfg: dict, stem: str) -> dict:
    out = _with_output_root(cfg, "no_grounding", stem)
    model = out.setdefault("model", {})
    model["use_semantic_grounding"] = False
    model["grounding_ot_weight"] = 0.0
    model["grounding_mask_weight"] = 0.0
    model["grounding_cycle_weight"] = 0.0
    return out


def make_no_spectral(cfg: dict, stem: str) -> dict:
    out = _with_output_root(cfg, "no_spectral", stem)
    diffusion = out.setdefault("diffusion", {})
    diffusion["lambda_spectral"] = 0.0
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create T2S core ablation configs from full CMTSG configs.")
    parser.add_argument("--config-dir", default="configs/t2s")
    parser.add_argument("--output-dir", default="configs/t2s_ablation")
    parser.add_argument("--patterns", nargs="+", default=["*.yaml"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_dir = Path(args.config_dir)
    output_dir = Path(args.output_dir)
    configs: list[Path] = []
    for pattern in args.patterns:
        configs.extend(sorted(config_dir.glob(pattern)))
    if not configs:
        raise FileNotFoundError(f"No configs matched {args.patterns} under {config_dir}")
    for path in configs:
        cfg = _load_yaml(path)
        stem = path.stem
        _write_yaml(output_dir / "no_grounding" / f"{stem}.yaml", make_no_grounding(cfg, stem))
        _write_yaml(output_dir / "no_spectral" / f"{stem}.yaml", make_no_spectral(cfg, stem))
        print(f"created ablation configs for {stem}")


if __name__ == "__main__":
    main()
