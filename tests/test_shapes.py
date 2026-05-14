from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    try:
        import torch
    except Exception as exc:
        print(f"SKIP: PyTorch is not installed ({exc})")
        return

    from cmtsg.env_bank import build_anchor_gaf
    from cmtsg.models import CMTSGModel
    from cmtsg.models.diffusion import GaussianDiffusion

    train_ts = np.random.randn(20, 36, 5).astype(np.float32)
    _, anchor_gaf_np = build_anchor_gaf(train_ts, n_env=12, seed=42, max_size=64)
    anchor_gaf = torch.from_numpy(anchor_gaf_np)
    model = CMTSGModel(seq_len=36, n_vars=5, gaf_size=36, text_dim=64, hidden_size=64, depth=2, num_heads=4)
    x = torch.randn(4, 36, 5)
    t = torch.randint(0, 50, (4,))
    text = torch.randn(4, 64)
    pred, aux = model(x, t, text, anchor_gaf)
    assert pred.shape == x.shape
    assert aux["alpha"].shape == (4, 12)
    assert torch.allclose(aux["alpha"].sum(dim=-1), torch.ones(4), atol=1e-5)
    diffusion = GaussianDiffusion(model, num_steps=10)
    loss, metrics = diffusion.training_loss(x, text, anchor_gaf)
    assert loss.ndim == 0
    assert "route_entropy" in metrics
    print("test_shapes.py: OK")


if __name__ == "__main__":
    main()
