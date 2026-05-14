from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

from cmtsg.charts import render_line_chart
from cmtsg.data import load_text_caps, load_ts, split_paths
from cmtsg.imaging import chart_stats
from cmtsg.prompts import compact_generation_condition_prompt, fill_prompt, load_prompt_template
from cmtsg.utils import ensure_dir, normalize_dataset_name, resolve_path


def _json_from_text(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _write_failed_raw(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _generation_condition(obj: dict[str, Any], fallback: str) -> str:
    value = obj.get("generation_condition")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback.strip()


def _mock_json(dataset: str, caption: str) -> dict[str, Any]:
    sentence = caption.strip().split(".")[0].strip()
    if not sentence:
        sentence = "The time series follows the described generation condition"
    return {
        "dataset": "Weather" if dataset == "weather" else "Synth-M",
        "chart_consistency": {"is_consistent_with_text": "unclear", "evidence": "mock output"},
        "generation_condition": sentence + ".",
    }


def _generation_condition_from_raw(raw: str) -> str | None:
    marker = '"generation_condition"'
    pos = raw.find(marker)
    if pos < 0:
        return None
    colon = raw.find(":", pos + len(marker))
    if colon < 0:
        return None
    first_quote = raw.find('"', colon + 1)
    if first_quote < 0:
        return None
    end = first_quote + 1
    escaped = False
    while end < len(raw):
        ch = raw[end]
        if ch == '"' and not escaped:
            return raw[first_quote + 1 : end].strip()
        escaped = ch == "\\" and not escaped
        if ch != "\\":
            escaped = False
        end += 1
    return None


class QwenVLRunner:
    def __init__(self, model_path: str | Path, device: str = "auto") -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.model_path = str(resolve_path(model_path))
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map=device,
            local_files_only=True,
        )

    def generate(self, prompt: str, image_path: Path, max_new_tokens: int = 512) -> str:
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
        ]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def generate_batch(self, prompts: list[str], image_paths: list[Path], max_new_tokens: int = 256) -> list[str]:
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
            for prompt, image_path in zip(prompts, image_paths, strict=True)
        ]
        texts = [
            self.processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
            for message in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
        trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
        ]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def run(args: argparse.Namespace) -> None:
    dataset = normalize_dataset_name(args.dataset)
    data_root = resolve_path(args.data_root or f"datasets/{dataset}")
    processed_root = ensure_dir(args.processed_root or f"processed/{dataset}")
    split_dir = ensure_dir(processed_root / args.split)
    chart_dir = ensure_dir(split_dir / "line_chart_pngs")

    ts_path, caps_path = split_paths(data_root, args.split)
    ts = load_ts(ts_path)
    caps = load_text_caps(caps_path)
    limit = min(args.limit or ts.shape[0], ts.shape[0])
    template = load_prompt_template(dataset, args.prompts)

    if args.caption_policy == "all":
        cap_indices = list(range(caps.shape[1]))
    elif args.caption_policy == "first":
        cap_indices = [0]
    else:
        rng = np.random.default_rng(args.caption_seed)
        cap_indices = [None]

    out_caps = caps.shape[1] if args.caption_policy == "all" else 1
    json_path = processed_root / f"{args.split}_causal_json.npy"
    text_path = processed_root / f"{args.split}_causal_text.npy"
    if args.resume and json_path.exists() and text_path.exists():
        causal_json = np.load(json_path, allow_pickle=True)
        causal_text = np.load(text_path, allow_pickle=True)
        if causal_json.shape != (limit, out_caps) or causal_text.shape != (limit, out_caps):
            raise ValueError(
                f"Cannot resume with shape mismatch: json={causal_json.shape}, text={causal_text.shape}, "
                f"expected={(limit, out_caps)}"
            )
    else:
        causal_json = np.full((limit, out_caps), "", dtype=object)
        causal_text = np.full((limit, out_caps), "", dtype=object)
    runner = None if args.mock else QwenVLRunner(args.qwen_path, args.device_map)
    failed_raw_path = processed_root / f"{args.split}_qwen_failed_raw.jsonl"

    pending: list[tuple[int, int, str, Path, str]] = []
    for idx in tqdm(range(limit), desc=f"{dataset}:{args.split}:prepare"):
        image_path = chart_dir / f"{idx}.png"
        if args.render_charts or not image_path.exists():
            render_line_chart(ts[idx], image_path, title=f"{dataset} {args.split} #{idx}")
        stats_text = json.dumps(chart_stats(ts[idx]), ensure_ascii=False)
        selected_cap_indices = cap_indices
        if args.caption_policy == "random":
            selected_cap_indices = [int(rng.integers(0, caps.shape[1]))]
        for out_idx, cap_idx in enumerate(selected_cap_indices):
            if cap_idx is None:
                raise RuntimeError("Internal caption policy error")
            if args.resume and str(causal_text[idx, out_idx]).strip():
                continue
            caption = str(caps[idx, cap_idx])
            if args.mock:
                obj = _mock_json(dataset, caption)
                causal_json[idx, out_idx] = json.dumps(obj, ensure_ascii=False)
                causal_text[idx, out_idx] = _generation_condition(obj, caption)
            else:
                if args.compact:
                    prompt = compact_generation_condition_prompt(dataset, caption, stats_text)
                else:
                    prompt = fill_prompt(template, caption, stats_text)
                pending.append((idx, out_idx, prompt, image_path, caption))

    if not args.mock:
        for start in tqdm(range(0, len(pending), args.batch_size), desc=f"{dataset}:{args.split}:qwen"):
            batch = pending[start : start + args.batch_size]
            prompts = [item[2] for item in batch]
            image_paths = [item[3] for item in batch]
            captions = [item[4] for item in batch]
            raws = runner.generate_batch(prompts, image_paths, args.max_new_tokens)
            for (idx, out_idx, _, _, _), caption, raw in zip(batch, captions, raws, strict=True):
                try:
                    obj = _json_from_text(raw)
                except Exception as exc:
                    fallback_condition = _generation_condition_from_raw(raw)
                    if args.allow_partial_json and fallback_condition:
                        obj = {
                            "dataset": "Weather" if dataset == "weather" else "Synth-M",
                            "generation_condition": fallback_condition,
                            "parse_warning": f"partial_json_fallback: {type(exc).__name__}: {exc}",
                        }
                        causal_json[idx, out_idx] = json.dumps(obj, ensure_ascii=False)
                        causal_text[idx, out_idx] = fallback_condition
                        continue
                    if args.fallback_to_caption:
                        fallback = caption.strip()
                        obj = {
                            "dataset": "Weather" if dataset == "weather" else "Synth-M",
                            "generation_condition": fallback,
                            "parse_warning": f"caption_fallback: {type(exc).__name__}: {exc}",
                        }
                        _write_failed_raw(
                            failed_raw_path,
                            {
                                "idx": idx,
                                "caption_index": out_idx,
                                "caption": caption,
                                "raw": raw,
                                "fallback": fallback,
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                        )
                        causal_json[idx, out_idx] = json.dumps(obj, ensure_ascii=False)
                        causal_text[idx, out_idx] = fallback
                        continue
                    _write_failed_raw(
                        failed_raw_path,
                        {
                            "idx": idx,
                            "caption_index": out_idx,
                            "caption": caption,
                            "raw": raw,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                    preview = raw[:500].replace("\n", "\\n")
                    raise ValueError(
                        f"Qwen output is not valid JSON for idx={idx}, caption_index={out_idx}. "
                        f"Raw output was saved to {failed_raw_path}. Preview: {preview!r}"
                    ) from exc
                causal_json[idx, out_idx] = json.dumps(obj, ensure_ascii=False)
                causal_text[idx, out_idx] = _generation_condition(obj, caption)
            if args.save_every_batches > 0 and ((start // args.batch_size) + 1) % args.save_every_batches == 0:
                np.save(json_path, causal_json)
                np.save(text_path, causal_text)

    np.save(json_path, causal_json)
    np.save(text_path, causal_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract causal generation text with Qwen2.5-VL.")
    parser.add_argument("--dataset", required=True, choices=["weather", "synth-m"])
    parser.add_argument("--split", required=True, choices=["train", "valid", "test"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--processed-root", default=None)
    parser.add_argument("--prompts", default="prompts.txt")
    parser.add_argument("--qwen-path", default="pretrained/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--caption-policy", choices=["first", "all", "random"], default="first")
    parser.add_argument("--caption-seed", type=int, default=42)
    parser.add_argument("--allow-partial-json", action="store_true")
    parser.add_argument("--fallback-to-caption", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every-batches", type=int, default=20)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--render-charts", action="store_true", default=True)
    parser.add_argument("--mock", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
