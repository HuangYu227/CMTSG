from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from cmtsg.utils import resolve_path


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def append_csv(path: str | Path, row: dict[str, Any]) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
        expanded = False
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
                expanded = True
        rows.append({key: row.get(key, "") for key in fieldnames})
        if expanded:
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            return
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writerow({key: row.get(key, "") for key in fieldnames})
        return

    fieldnames = list(row.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})
