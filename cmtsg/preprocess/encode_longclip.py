from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

from cmtsg.utils import ensure_dir, normalize_dataset_name, resolve_path


def _load_texts(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.load(path, allow_pickle=True)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Expected causal text shape [N,C], got {arr.shape}")
    return arr.astype(str)


def _projection(in_dim: int, out_dim: int, path: Path) -> np.ndarray:
    if path.exists():
        matrix = np.load(path)
        if matrix.shape != (in_dim, out_dim):
            raise ValueError(f"Projection shape mismatch: {matrix.shape} vs {(in_dim, out_dim)}")
        return matrix.astype(np.float32)
    rng = np.random.default_rng(42)
    matrix = rng.normal(0.0, 1.0 / np.sqrt(in_dim), size=(in_dim, out_dim)).astype(np.float32)
    np.save(path, matrix)
    return matrix


class LongCLIPRunner:
    def __init__(self, model_path: str | Path, device: str) -> None:
        import torch
        from transformers import AutoTokenizer, CLIPTextConfig, CLIPTextModelWithProjection

        self.torch = torch
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_path = str(resolve_path(model_path))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        if "longclip" in self.model_path.lower():
            config = CLIPTextConfig.from_pretrained(self.model_path, local_files_only=True)
            config.max_position_embeddings = max(getattr(config, "max_position_embeddings", 77), 248)
            self.model = CLIPTextModelWithProjection.from_pretrained(
                self.model_path, config=config, local_files_only=True
            )
        else:
            self.model = CLIPTextModelWithProjection.from_pretrained(self.model_path, local_files_only=True)
        self.model.to(self.device).eval()
        self.max_length = getattr(self.model.config, "max_position_embeddings", 77)

    @property
    def output_dim(self) -> int:
        return int(getattr(self.model.config, "projection_dim", 768))

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        outputs = []
        with self.torch.no_grad():
            for start in tqdm(range(0, len(texts), batch_size), desc="longclip"):
                batch = texts[start : start + batch_size]
                tokens = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                out = self.model(**tokens)
                emb = out.text_embeds
                emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                outputs.append(emb.cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0)


def run(args: argparse.Namespace) -> None:
    dataset = normalize_dataset_name(args.dataset)
    processed_root = ensure_dir(args.processed_root or f"processed/{dataset}")
    texts = _load_texts(processed_root / f"{args.split}_causal_text.npy")
    limit = min(args.limit or texts.shape[0], texts.shape[0])
    texts = texts[:limit]
    flat = texts.reshape(-1).tolist()

    if args.mock:
        rng = np.random.default_rng(123)
        emb = rng.normal(size=(len(flat), args.output_dim)).astype(np.float32)
        emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True).clip(min=1e-6)
    else:
        runner = LongCLIPRunner(args.longclip_path, args.device)
        raw = runner.encode(flat, args.batch_size)
        if raw.shape[1] == args.output_dim:
            emb = raw
        else:
            proj_path = processed_root / f"longclip_projection_{raw.shape[1]}x{args.output_dim}.npy"
            proj = _projection(raw.shape[1], args.output_dim, proj_path)
            emb = raw @ proj
            emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True).clip(min=1e-6)

    emb = emb.reshape(texts.shape[0], texts.shape[1], args.output_dim).astype(np.float32)
    np.save(processed_root / f"{args.split}_text_emb.npy", emb)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encode causal_text.npy with LongCLIP.")
    parser.add_argument("--dataset", required=True, choices=["weather", "synth-m"])
    parser.add_argument("--split", required=True, choices=["train", "valid", "test"])
    parser.add_argument("--processed-root", default=None)
    parser.add_argument("--longclip-path", default="pretrained/LongCLIP")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mock", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
