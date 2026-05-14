from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None
    Dataset = object

from cmtsg.data import load_text_caps, load_ts, split_paths
from cmtsg.utils import resolve_path


class CMTSGDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        processed_root: str | Path,
        split: str,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        train: bool = False,
    ) -> None:
        if torch is None:
            raise RuntimeError("CMTSGDataset requires PyTorch")
        ts_path, caps_path = split_paths(data_root, split)
        self.ts = load_ts(ts_path)
        self.caps = load_text_caps(caps_path)
        self.split = split
        self.train = train
        self.processed_root = resolve_path(processed_root)
        emb_path = self.processed_root / f"{split}_text_emb.npy"
        if not emb_path.exists():
            raise FileNotFoundError(
                f"Missing text embeddings: {emb_path}. Run cmtsg.preprocess.encode_longclip first."
            )
        self.text_emb = np.load(emb_path).astype(np.float32)
        if self.text_emb.ndim == 2:
            self.text_emb = self.text_emb[:, None, :]
        if self.text_emb.shape[0] != self.ts.shape[0]:
            raise ValueError(f"Embedding count mismatch: {self.text_emb.shape} vs {self.ts.shape}")
        if self.text_emb.shape[1] != self.caps.shape[1]:
            raise ValueError(f"Caption count mismatch: {self.text_emb.shape} vs {self.caps.shape}")

        self.mean = mean if mean is not None else self.ts.mean(axis=(0, 1), keepdims=True)
        self.std = std if std is not None else self.ts.std(axis=(0, 1), keepdims=True) + 1e-6
        self.ts_norm = (self.ts - self.mean) / self.std

    def __len__(self) -> int:
        return int(self.ts.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.train:
            cap_idx = np.random.randint(0, self.text_emb.shape[1])
        else:
            cap_idx = 0
        return {
            "x": torch.from_numpy(self.ts_norm[idx]).float(),
            "text_emb": torch.from_numpy(self.text_emb[idx, cap_idx]).float(),
            "caption_index": torch.tensor(cap_idx, dtype=torch.long),
        }
