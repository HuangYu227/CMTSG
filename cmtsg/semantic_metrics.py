from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from cmtsg.metrics import fid_from_features
from cmtsg.utils import resolve_path


def calculate_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    """VerbalTS-compatible Frechet distance implementation."""
    from scipy import linalg

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    if mu1.shape != mu2.shape:
        raise ValueError(f"Mean shape mismatch: {mu1.shape} vs {mu2.shape}")
    if sigma1.shape != sigma2.shape:
        raise ValueError(f"Covariance shape mismatch: {sigma1.shape} vs {sigma2.shape}")
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


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

    def compute_reference_stats(
        self,
        train_ts: np.ndarray,
        train_captions: list[str],
        cache_folder: str | Path | None = None,
        batch_size: int = 128,
    ) -> dict[str, np.ndarray]:
        cache_paths = None
        if cache_folder is not None:
            cache_root = resolve_path(cache_folder)
            cache_root.mkdir(parents=True, exist_ok=True)
            cache_paths = {
                "ts_mean": cache_root / "fid_mean.npy",
                "ts_cov": cache_root / "fid_cov.npy",
                "joint_mean": cache_root / "jftsd_mean.npy",
                "joint_cov": cache_root / "jftsd_cov.npy",
            }
            if all(path.exists() for path in cache_paths.values()):
                return {key: np.load(path) for key, path in cache_paths.items()}

        train_ts_emb = self.encode_ts(train_ts, batch_size)
        train_text_emb = self.encode_text(train_captions, batch_size)
        train_joint_emb = np.concatenate([train_ts_emb, train_text_emb], axis=1)
        stats = {
            "ts_mean": np.mean(train_ts_emb, axis=0),
            "ts_cov": np.cov(train_ts_emb, rowvar=False),
            "joint_mean": np.mean(train_joint_emb, axis=0),
            "joint_cov": np.cov(train_joint_emb, rowvar=False),
        }
        if cache_paths is not None:
            for key, path in cache_paths.items():
                np.save(path, stats[key])
        return stats

    def compute_verbalts_protocol(
        self,
        train_ts: np.ndarray,
        train_captions: list[str],
        gen_ts: np.ndarray,
        eval_captions: list[str],
        cache_folder: str | Path | None = None,
        batch_size: int = 128,
    ) -> dict[str, float | str]:
        stats = self.compute_reference_stats(train_ts, train_captions, cache_folder, batch_size)
        gen_emb = self.encode_ts(gen_ts, batch_size)
        eval_text_emb = self.encode_text(eval_captions, batch_size)
        joint_emb = np.concatenate([gen_emb, eval_text_emb], axis=1)
        cttp = float(np.trace(gen_emb @ eval_text_emb.T) / max(1, gen_emb.shape[0]))
        fid = calculate_frechet_distance(stats["ts_mean"], stats["ts_cov"], np.mean(gen_emb, axis=0), np.cov(gen_emb, rowvar=False))
        jftsd = calculate_frechet_distance(
            stats["joint_mean"],
            stats["joint_cov"],
            np.mean(joint_emb, axis=0),
            np.cov(joint_emb, rowvar=False),
        )
        return {
            "cttp": cttp,
            "fid": fid,
            "jftsd": jftsd,
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
