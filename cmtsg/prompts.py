from __future__ import annotations

from pathlib import Path

from cmtsg.utils import PROJECT_ROOT, normalize_dataset_name


def _extract_block(text: str, marker: str) -> str:
    marker_pos = text.find(marker)
    if marker_pos < 0:
        raise ValueError(f"Prompt marker not found: {marker}")
    start = text.rfind("System:", 0, marker_pos)
    if start < 0:
        start = marker_pos
    next_start = text.find("\nSystem:", marker_pos + len(marker))
    if next_start < 0:
        next_start = len(text)
    return text[start:next_start].strip()


def load_prompt_template(dataset: str, path: str | Path | None = None) -> str:
    dataset = normalize_dataset_name(dataset)
    path = Path(path) if path is not None else PROJECT_ROOT / "prompts.txt"
    text = path.read_text(encoding="utf-8")
    if dataset == "weather":
        return _extract_block(text, "The dataset is Weather")
    if dataset == "synth-m":
        return _extract_block(text, "The dataset is Synth-M")
    raise ValueError(f"Unsupported dataset: {dataset}")


def fill_prompt(template: str, text_info: str, chart_stats: str = "") -> str:
    return template.replace("{text_info}", text_info).replace("{chart_stats}", chart_stats)
