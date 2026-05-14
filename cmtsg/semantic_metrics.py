from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from cmtsg.metrics import fid_from_features
from cmtsg.utils import resolve_path


def _first_existing(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    for name in names:
        matches = list(root.rglob(name))
        if matches:
            return matches[0]
    return None


def discover_cttp_files(cttp_root: str | Path) -> tuple[Path, Path]:
    root = resolve_path(cttp_root)
    if not root.exists():
        raise FileNotFoundError(root)
    config = _first_existing(root, ("model_configs.yaml", "model_config.yaml", "config.yaml"))
    checkpoint = _first_existing(root, ("clip_model_best.pth", "model_best.pth", "best.pth"))
    if config is None:
        raise FileNotFoundError(f"Cannot find CTTP model config under {root}")
    if checkpoint is None:
        raise FileNotFoundError(f"Cannot find CTTP checkpoint under {root}")
    return config, checkpoint


class CTTPMetricEvaluator:
    def __init__(self, verbalts_root: str | Path, cttp_root: str | Path, device: str = "auto") -> None:
        import torch
        import yaml

        self.torch = torch
        verbalts_root = resolve_path(verbalts_root)
        if str(verbalts_root) not in sys.path:
            sys.path.insert(0, str(verbalts_root))
        from models.cttp.cttp_model import CTTP

        config_path, checkpoint_path = discover_cttp_files(cttp_root)
        configs = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if device != "auto":
            configs["device"] = device
        self.model = CTTP(configs)
        state = torch.load(checkpoint_path, map_location=self.model.device)
        self.model.load_state_dict(state)
        self.model = self.model.to(self.model.device).eval()
        self.config_path = str(config_path)
        self.checkpoint_path = str(checkpoint_path)

    @property
    def device(self):
        return self.model.device

    def encode_ts(self, ts: np.ndarray, batch_size: int = 128) -> np.ndarray:
        outs = []
        with self.torch.no_grad():
            for start in range(0, ts.shape[0], batch_size):
                batch = self.torch.as_tensor(ts[start : start + batch_size], device=self.device).float()
                lengths = self.torch.full((batch.shape[0],), batch.shape[1], device=self.device, dtype=self.torch.int32)
                emb = self.model.get_ts_coemb(batch, lengths)
                outs.append(emb.detach().cpu().numpy())
        return np.concatenate(outs, axis=0)

    def encode_text(self, captions: list[str], batch_size: int = 128) -> np.ndarray:
        outs = []
        with self.torch.no_grad():
            for start in range(0, len(captions), batch_size):
                emb = self.model.get_text_coemb(captions[start : start + batch_size], None)
                outs.append(emb.detach().cpu().numpy())
        return np.concatenate(outs, axis=0)

    def compute(
        self,
        real_ts: np.ndarray,
        gen_ts: np.ndarray,
        captions: list[str],
        batch_size: int = 128,
    ) -> dict[str, float | str]:
        real_emb = self.encode_ts(real_ts, batch_size)
        gen_emb = self.encode_ts(gen_ts, batch_size)
        text_emb = self.encode_text(captions, batch_size)
        cttp = float(np.trace(gen_emb @ text_emb.T) / max(1, gen_emb.shape[0]))
        fid = fid_from_features(real_emb, gen_emb)
        jftsd = fid_from_features(
            np.concatenate([real_emb, text_emb], axis=1),
            np.concatenate([gen_emb, text_emb], axis=1),
        )
        return {
            "cttp": cttp,
            "fid_cttp": fid,
            "jftsd_cttp": jftsd,
            "cttp_config": self.config_path,
            "cttp_checkpoint": self.checkpoint_path,
        }


def compute_cttp_metrics(
    real_ts: np.ndarray,
    gen_ts: np.ndarray,
    captions: list[str],
    verbalts_root: str | Path,
    cttp_root: str | Path,
    device: str = "auto",
    batch_size: int = 128,
) -> dict[str, float | str]:
    evaluator = CTTPMetricEvaluator(verbalts_root, cttp_root, device)
    return evaluator.compute(real_ts, gen_ts, captions, batch_size)
