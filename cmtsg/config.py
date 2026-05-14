from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from cmtsg.utils import resolve_path


def load_config(path: str | Path) -> dict[str, Any]:
    path = resolve_path(path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return cfg


def get_nested(cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    value: Any = cfg
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value
