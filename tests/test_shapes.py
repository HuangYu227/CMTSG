from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    try:
        import numpy as np
    except Exception as exc:
        print(f"SKIP: NumPy is not installed ({exc})")
        return

    try:
        import torch
    except Exception as exc:
        print(f"SKIP: PyTorch is not installed ({exc})")
        return

    from cmtsg.imaging import gasf_multivariate
    from cmtsg.models.env import EnvironmentRouter
    from cmtsg.models import CMTSGModel
    from cmtsg.models.diffusion import GaussianDiffusion, RectifiedFlow

    train_ts = np.random.randn(20, 36, 5).astype(np.float32)
    gaf_np = np.stack([gasf_multivariate(train_ts[i], max_size=384) for i in range(4)], axis=0)
    gaf = torch.from_numpy(gaf_np)
    model = CMTSGModel(
        seq_len=36,
        n_vars=5,
        gaf_size=36,
        text_dim=64,
        hidden_size=64,
        depth=2,
        num_heads=4,
        architecture="causal_relation_mmdit",
        text_tokens=4,
        series_register_tokens=8,
    )
    x = torch.randn(4, 36, 5)
    t = torch.randint(0, 50, (4,))
    text = torch.randn(4, 64)
    pred, aux = model(x, t, text, gaf)
    assert pred.shape == x.shape
    assert aux["alpha"].shape == (4, model.n_env)
    assert aux["env_slots"].shape == (4, model.n_env, model.env_dim)
    assert aux["series_summary"].shape == (4, 64)
    assert aux["semantic_summary"].shape == (4, 64)
    assert aux["relation_summary"].shape == (4, 64)
    assert torch.allclose(aux["alpha"].sum(dim=-1), torch.ones(4), atol=1e-5)
    assert model.router.output_dim == 64 * 4 * 4
    pred_cfg, _ = model(x, t, text, gaf, force_drop_text=True, force_drop_env=True, force_drop_semantic=True)
    assert pred_cfg.shape == x.shape

    flow = RectifiedFlow(model, num_steps=4, lambda_spectral=0.01, lambda_cycle_relation=0.01, lambda_triad_contrastive=0.01)
    flow_loss, flow_metrics = flow.training_loss(x, text, gaf)
    assert flow_loss.ndim == 0
    assert torch.isfinite(flow_loss)
    assert "loss_flow" in flow_metrics
    assert "loss_cycle_relation" in flow_metrics
    assert "loss_triad_contrastive" in flow_metrics
    flow_sample = flow.sample((4, 36, 5), text, gaf=None, sampler="heun", guidance_text=1.0, guidance_relation=1.0, guidance_joint=1.0)
    assert flow_sample.shape == x.shape

    baseline_model = CMTSGModel(
        seq_len=36,
        n_vars=5,
        gaf_size=36,
        text_dim=64,
        hidden_size=64,
        depth=2,
        num_heads=4,
        architecture="factorized_dit",
    )
    diffusion = GaussianDiffusion(model, num_steps=10)
    loss, metrics = diffusion.training_loss(x, text, gaf)
    assert loss.ndim == 0
    assert "route_entropy" in metrics
    assert "loss_slot_aux" in metrics
    assert "slot_cosine_mean" in metrics
    assert "text_env_slot_loss" in metrics
    assert "route_entropy_scale" in metrics
    text_only = diffusion.sample((4, 36, 5), text, gaf=None, sampler="ddim")
    assert text_only.shape == x.shape
    baseline_diffusion = GaussianDiffusion(baseline_model, num_steps=10)
    baseline_loss, _ = baseline_diffusion.training_loss(x, text, gaf)
    assert baseline_loss.ndim == 0

    router = EnvironmentRouter(n_vars=5, gaf_size=36, text_dim=64, env_dim=64, n_env=12, routing="text")
    env_mix_a, alpha_a, _, slots_a, _ = router(text, gaf)
    text_alt = text.clone()
    text_alt[:, 0] = text_alt[:, 0] + 0.25
    env_mix_b, alpha_b, _, slots_b, _ = router(text_alt, gaf)
    assert env_mix_a.shape == (4, 64)
    assert alpha_a.shape == (4, 12)
    assert slots_a.shape == (4, 12, 64)
    assert torch.allclose(alpha_a.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(slots_a, slots_b, atol=1e-5)
    assert not torch.allclose(alpha_a, alpha_b)
    env_mix_text, alpha_text, _, slots_text, aux_text = router(text, None)
    assert env_mix_text.shape == (4, 64)
    assert alpha_text.shape == (4, 12)
    assert slots_text.shape == (4, 12, 64)
    assert aux_text["text_env_slot_loss"].ndim == 0

    shifted_gaf = torch.roll(gaf, shifts=1, dims=-1)
    _, _, _, slots_shifted, _ = router(text, shifted_gaf)
    assert not torch.allclose(slots_a, slots_shifted)

    uniform_router = EnvironmentRouter(n_vars=5, gaf_size=36, text_dim=64, env_dim=64, n_env=12, routing="uniform")
    _, alpha_uniform, _, slots_uniform, _ = uniform_router(text, gaf)
    assert slots_uniform.shape == (4, 12, 64)
    assert torch.allclose(alpha_uniform, torch.full_like(alpha_uniform, 1.0 / 12.0), atol=1e-6)
    print("test_shapes.py: OK")


if __name__ == "__main__":
    main()
