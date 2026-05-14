from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def make_beta_schedule(num_steps: int, beta_start: float, beta_end: float, schedule: str = "quad") -> torch.Tensor:
    if schedule == "quad":
        betas = np.linspace(beta_start**0.5, beta_end**0.5, num_steps, dtype=np.float64) ** 2
    elif schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_steps, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported beta schedule: {schedule}")
    return torch.from_numpy(betas.astype(np.float32))


def extract(values: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    out = values.gather(0, t)
    return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        num_steps: int = 50,
        beta_start: float = 0.0001,
        beta_end: float = 0.5,
        schedule: str = "quad",
    ) -> None:
        super().__init__()
        self.model = model
        self.num_steps = num_steps
        betas = make_beta_schedule(num_steps, beta_start, beta_end, schedule)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]])
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        posterior_var = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        self.register_buffer("posterior_variance", posterior_var.clamp_min(1e-20))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return extract(self.sqrt_alpha_bars, t, x_start.shape) * x_start + extract(
            self.sqrt_one_minus_alpha_bars, t, x_start.shape
        ) * noise

    def training_loss(self, x_start: torch.Tensor, text_emb: torch.Tensor, anchor_gaf: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        t = torch.randint(0, self.num_steps, (x_start.shape[0],), device=x_start.device)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        pred, aux = self.model(x_noisy, t, text_emb, anchor_gaf)
        loss = F.mse_loss(pred, noise)
        metrics = {"loss": loss, "route_entropy": aux["route_entropy"], "route_max": aux["route_max"]}
        return loss, metrics

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, text_emb: torch.Tensor, anchor_gaf: torch.Tensor) -> torch.Tensor:
        pred_noise, _ = self.model(x, t, text_emb, anchor_gaf)
        beta_t = extract(self.betas, t, x.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alpha_bars, t, x.shape)
        sqrt_recip_alpha = torch.rsqrt(extract(self.alphas, t, x.shape))
        mean = sqrt_recip_alpha * (x - beta_t * pred_noise / sqrt_one_minus)
        var = extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        nonzero = (t != 0).float().reshape(x.shape[0], *((1,) * (x.ndim - 1)))
        return mean + nonzero * torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, shape: tuple[int, int, int], text_emb: torch.Tensor, anchor_gaf: torch.Tensor) -> torch.Tensor:
        x = torch.randn(shape, device=text_emb.device)
        for step in reversed(range(self.num_steps)):
            t = torch.full((shape[0],), step, device=text_emb.device, dtype=torch.long)
            x = self.p_sample(x, t, text_emb, anchor_gaf)
        return x
