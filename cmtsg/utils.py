from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
"""
normalize_dataset_name鲁棒性不够,只能支持特定的几个文件名
"""

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_dir(path: str | Path) -> Path:
    path = resolve_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_json(path: str | Path) -> Any:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(obj: Any, path: str | Path) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def normalize_dataset_name(name: str) -> str:
    lowered = name.lower().replace("_", "-")
    if lowered in {"weather", "weather datasets"}:
        return "weather"
    if lowered in {"synth-m", "synth-multi", "synthetic-m", "synthetic_m"}:
        return "synth-m"
    raise ValueError(f"Unsupported dataset: {name}")
