from __future__ import annotations

from pathlib import Path

from cmtsg.utils import PROJECT_ROOT, normalize_dataset_name

"""
load_prompt_template 鲁棒性不够，只有两个文件夹。

"""
def _extract_block(text: str, marker: str) -> str:
    """
    marker_pos是用来结尾的一段字符串，提示词也就是system到这个marker_pos的一段文本。
    """
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
    """
    用来返回提取到的提示词文本，提取的依据是dataset和path。
    """
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


def compact_generation_condition_prompt(dataset: str, text_info: str, chart_stats: str = "") -> str:
    """
    这个提示词不够鲁邦，只有两个dataset的提示词。
    """
    dataset = normalize_dataset_name(dataset)
    if dataset == "weather":
        return (
            "You are a meteorological causal reasoning expert for text-to-time-series generation.\n"
            "Use the forecast text as the primary source and the chart only as support.\n"
            "Return valid JSON only, with exactly one key: generation_condition.\n"
            "The value must be one concise English sentence describing the weather-driven causal "
            "conditions for generating the multivariate time series.\n\n"
            f"[Original Weather Text Description]\n{text_info}\n\n"
            f"[Optional Chart Statistics]\n{chart_stats}\n\n"
            'Return format: {"generation_condition": "..."}'
        )
    return (
        "You are a causal mechanism extractor for synthetic multivariate time-series generation.\n"
        "Use the text as the primary source and the chart only as visual verification.\n"
        "Return valid JSON only, with exactly one key: generation_condition.\n"
        "The value must be one concise English sentence describing the causal generation mechanism.\n\n"
        f"[Original Text Description]\n{text_info}\n\n"
        f"[Optional Chart Statistics]\n{chart_stats}\n\n"
        'Return format: {"generation_condition": "..."}'
    )
