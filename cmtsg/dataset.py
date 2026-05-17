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
from cmtsg.imaging import gasf_multivariate
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
        gaf_max_size: int = 384,
    ) -> None:
        if torch is None:
            raise RuntimeError("CMTSGDataset requires PyTorch")
        self.data_root = resolve_path(data_root)
        ts_path, caps_path = split_paths(data_root, split)
        self.ts = load_ts(ts_path)
        self.caps = load_text_caps(caps_path)
        self.split = split
        self.train = train
        self.gaf_max_size = int(gaf_max_size)
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
        if self.text_emb.shape[1] not in (1, self.caps.shape[1]):
            raise ValueError(f"Caption count mismatch: {self.text_emb.shape} vs {self.caps.shape}")
        self.semantic_atoms = self._load_semantic_atoms()

        self.mean = mean if mean is not None else self.ts.mean(axis=(0, 1), keepdims=True)
        self.std = std if std is not None else self.ts.std(axis=(0, 1), keepdims=True) + 1e-6
        self.ts_norm = (self.ts - self.mean) / self.std
        self.gaf_size = min(int(self.ts.shape[1]), self.gaf_max_size)
        self.gaf_cache = self._load_gaf_cache()

    def __len__(self) -> int:
        return int(self.ts.shape[0])

    def _load_semantic_atoms(self) -> np.ndarray | None:
        """
        Optional token-level causal atom embeddings.

        Preferred file convention:
            processed_root/{split}_semantic_atoms.npy -> [N,C,D] or [N,D]

        Backward-compatible fallback:
            if split_text_emb.npy already contains multiple embeddings per sample
            ([N,C,D], C > 1), use the whole set as semantic atoms instead of
            throwing away token-level information through random caption choice.
        """
        atom_path = self.processed_root / f"{self.split}_semantic_atoms.npy"
        if atom_path.exists():
            atoms = np.load(atom_path).astype(np.float32)
            if atoms.ndim == 2:
                atoms = atoms[:, None, :]
            if atoms.ndim != 3:
                raise ValueError(f"Expected semantic atoms [N,C,D], got {atoms.shape}")
            if atoms.shape[0] != self.ts.shape[0]:
                raise ValueError(f"Semantic atom count mismatch: {atoms.shape} vs {self.ts.shape}")
            return atoms
        if self.text_emb.shape[1] > 1:
            return self.text_emb
        return None

    def _load_gaf_cache(self) -> np.ndarray | None:
        cache_path = self.data_root / f"{self.split}_gaf.npy"
        if not cache_path.exists():
            return None
        gaf = np.load(cache_path, mmap_mode="r")
        expected = (self.ts.shape[0], self.ts.shape[2], self.gaf_size, self.gaf_size)
        if tuple(gaf.shape) != expected:
            raise ValueError(f"GAF cache shape mismatch: {cache_path} has {gaf.shape}, expected {expected}")
        if gaf.dtype != np.float32:
            raise ValueError(f"GAF cache must be float32: {cache_path} has dtype={gaf.dtype}")
        return gaf

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.train:
            cap_idx = np.random.randint(0, self.text_emb.shape[1])
        else:
            cap_idx = 0
        if self.gaf_cache is None:
            gaf = gasf_multivariate(self.ts[idx], max_size=self.gaf_max_size)
        else:
            gaf = np.array(self.gaf_cache[idx], dtype=np.float32, copy=True)
        item = {
            "x": torch.from_numpy(self.ts_norm[idx]).float(),
            "gaf": torch.from_numpy(gaf).float(),
            "text_emb": torch.from_numpy(self.text_emb[idx, cap_idx]).float(),
            "caption_index": torch.tensor(cap_idx, dtype=torch.long),
        }
        if self.semantic_atoms is not None:
            item["semantic_atoms"] = torch.from_numpy(self.semantic_atoms[idx]).float()
        return item
