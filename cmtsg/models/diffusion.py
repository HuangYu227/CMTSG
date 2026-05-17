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


def triad_contrastive_loss(
    series_summary: torch.Tensor | None,
    semantic_summary: torch.Tensor | None,
    relation_summary: torch.Tensor | None,
    temperature: float = 0.07,
) -> torch.Tensor:
    if series_summary is None or semantic_summary is None or relation_summary is None:
        fallback = series_summary if series_summary is not None else semantic_summary if semantic_summary is not None else relation_summary
        if fallback is None:
            raise ValueError("At least one tensor is required to infer device for a zero contrastive loss")
        return fallback.new_zeros(())
    if series_summary.shape[0] < 2:
        return series_summary.new_zeros(())

    def pair_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        logits = a @ b.transpose(0, 1) / temperature
        labels = torch.arange(a.shape[0], device=a.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))

    return (
        pair_loss(series_summary, semantic_summary)
        + pair_loss(series_summary, relation_summary)
        + pair_loss(semantic_summary, relation_summary)
    ) / 3.0


class GADFRelationalSpectralLoss(nn.Module):
    def __init__(
        self,
        eps: float = 1e-6,
        mode: str = "abs",
        high_freq_gamma: float = 1.0,
        dc_weight: float = 0.05,
        detach_target: bool = True,
    ) -> None:
        super().__init__()
        if mode not in {"abs", "sim"}:
            raise ValueError(f"mode must be 'abs' or 'sim', got {mode}")
        self.eps = eps
        self.mode = mode
        self.high_freq_gamma = high_freq_gamma
        self.dc_weight = dc_weight
        self.detach_target = detach_target

    def compute_gadf_field(self, x: torch.Tensor, ref_min: torch.Tensor, ref_max: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x_norm = (x - ref_min) / (ref_max - ref_min + self.eps)
        x_norm = x_norm.clamp(self.eps, 1.0 - self.eps)
        phi = torch.acos(x_norm)
        gadf = torch.sin(phi.unsqueeze(-1) - phi.unsqueeze(-2)).abs()
        if self.mode == "sim":
            gadf = 1.0 - gadf
        return gadf

    def radial_frequency_weight(self, h: int, w_rfft: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        fy = torch.fft.fftfreq(h, device=device, dtype=dtype).abs()
        fx = torch.fft.rfftfreq((w_rfft - 1) * 2, device=device, dtype=dtype).abs()
        radius = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
        radius = radius / radius.max().clamp_min(self.eps)
        weight = 1.0 + self.high_freq_gamma * radius
        weight[0, 0] = self.dc_weight
        return weight[None, None, :, :]

    def log_normalized_magnitude(self, field: torch.Tensor) -> torch.Tensor:
        fft = torch.fft.rfft2(field, dim=(-2, -1), norm="ortho")
        mag = torch.abs(fft)
        scale = mag.mean(dim=(-2, -1), keepdim=True).detach().clamp_min(self.eps)
        return torch.log1p(mag / scale)

    def forward(self, x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        x_true_t = x_true.transpose(1, 2)
        ref_min = x_true_t.amin(dim=-1, keepdim=True).detach()
        ref_max = x_true_t.amax(dim=-1, keepdim=True).detach()
        gadf_pred = self.compute_gadf_field(x_pred, ref_min, ref_max)
        gadf_true = self.compute_gadf_field(x_true, ref_min, ref_max)
        if self.detach_target:
            gadf_true = gadf_true.detach()
        spec_pred = self.log_normalized_magnitude(gadf_pred)
        spec_true = self.log_normalized_magnitude(gadf_true)
        weight = self.radial_frequency_weight(
            h=spec_pred.shape[-2],
            w_rfft=spec_pred.shape[-1],
            device=spec_pred.device,
            dtype=spec_pred.dtype,
        )
        loss_map = F.smooth_l1_loss(spec_pred * weight, spec_true * weight, reduction="none")
        return loss_map.mean(dim=(1, 2, 3))


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        num_steps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = "quad",
        lambda_spectral: float = 0.05,
        spectral_warmup_power: float = 1.0,
        spectral_mode: str = "abs",
        spectral_high_freq_gamma: float = 1.0,
        spectral_dc_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_steps = num_steps
        self.lambda_spectral = float(lambda_spectral)
        self.spectral_warmup_power = float(spectral_warmup_power)
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
        self.gadf_spectral_loss = GADFRelationalSpectralLoss(
            mode=spectral_mode,
            high_freq_gamma=spectral_high_freq_gamma,
            dc_weight=spectral_dc_weight,
        )

    @staticmethod
    def _validate_condition_batch(
        batch_size: int,
        text_emb: torch.Tensor,
        gaf: torch.Tensor | None = None,
        semantic_atoms: torch.Tensor | None = None,
    ) -> None:
        if text_emb.shape[0] != batch_size:
            raise ValueError(f"text_emb batch mismatch: expected {batch_size}, got {text_emb.shape[0]}")
        if gaf is not None and gaf.shape[0] != batch_size:
            raise ValueError(f"gaf batch mismatch: expected {batch_size}, got {gaf.shape[0]}")
        if semantic_atoms is not None and semantic_atoms.shape[0] != batch_size:
            raise ValueError(f"semantic_atoms batch mismatch: expected {batch_size}, got {semantic_atoms.shape[0]}")

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return extract(self.sqrt_alpha_bars, t, x_start.shape) * x_start + extract(
            self.sqrt_one_minus_alpha_bars, t, x_start.shape
        ) * noise

    def training_loss(
        self,
        x_start: torch.Tensor,
        text_emb: torch.Tensor,
        gaf: torch.Tensor | None = None,
        semantic_atoms: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_condition_batch(x_start.shape[0], text_emb, gaf, semantic_atoms)
        t = torch.randint(0, self.num_steps, (x_start.shape[0],), device=x_start.device)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        pred_noise, aux = self.model(x_noisy, t, text_emb, gaf, semantic_atoms=semantic_atoms)
        loss_diff = F.mse_loss(pred_noise, noise)
        if self.lambda_spectral > 0.0:
            sqrt_ab = extract(self.sqrt_alpha_bars, t, x_start.shape)
            sqrt_omab = extract(self.sqrt_one_minus_alpha_bars, t, x_start.shape)
            x0_pred = (x_noisy - sqrt_omab * pred_noise) / sqrt_ab
            x0_pred = x0_pred.clamp(-5.0, 5.0)
            loss_spectral_per_sample = self.gadf_spectral_loss(x0_pred, x_start)
            alpha_bar_t = extract(self.alpha_bars, t, x_start.shape).reshape(x_start.shape[0])
            spectral_weight = alpha_bar_t.pow(self.spectral_warmup_power).detach()
            loss_spectral = (loss_spectral_per_sample * spectral_weight).mean()
        else:
            loss_spectral = loss_diff.new_zeros(())
            spectral_weight = loss_diff.new_zeros(())
        slot_aux_loss = aux.get("slot_aux_loss", loss_diff.new_zeros(()))
        grounding_aux_loss = aux.get("grounding_aux_loss", loss_diff.new_zeros(()))
        loss = loss_diff + self.lambda_spectral * loss_spectral + slot_aux_loss + grounding_aux_loss
        metrics = {
            "loss": loss.detach(),
            "loss_diff": loss_diff.detach(),
            "loss_spectral": loss_spectral.detach(),
            "loss_slot_aux": slot_aux_loss.detach(),
            "loss_grounding_aux": grounding_aux_loss.detach(),
            "grounding_loss_ot": aux.get("grounding_loss_ot", loss_diff.new_zeros(())).detach(),
            "grounding_loss_mask": aux.get("grounding_loss_mask", loss_diff.new_zeros(())).detach(),
            "grounding_loss_cycle": aux.get("grounding_loss_cycle", loss_diff.new_zeros(())).detach(),
            "grounding_transport_entropy": aux.get("grounding_transport_entropy", loss_diff.new_zeros(())).detach(),
            "grounding_transport_row_error": aux.get("grounding_transport_row_error", loss_diff.new_zeros(())).detach(),
            "grounding_transport_col_error": aux.get("grounding_transport_col_error", loss_diff.new_zeros(())).detach(),
            "slot_diversity_loss": aux.get("slot_diversity_loss", loss_diff.new_zeros(())).detach(),
            "route_entropy_loss": aux.get("route_entropy_loss", loss_diff.new_zeros(())).detach(),
            "text_slot_align_loss": aux.get("text_slot_align_loss", loss_diff.new_zeros(())).detach(),
            "text_env_slot_loss": aux.get("text_env_slot_loss", loss_diff.new_zeros(())).detach(),
            "slot_cosine_mean": aux.get("slot_cosine_mean", loss_diff.new_zeros(())).detach(),
            "route_entropy_scale": aux.get("route_entropy_scale", loss_diff.new_zeros(())).detach(),
            "spectral_weight": spectral_weight.mean().detach(),
            "route_entropy": aux["route_entropy"].detach(),
            "route_max": aux["route_max"].detach(),
            "text_drop_rate": aux.get("text_drop_rate", torch.tensor(0.0, device=x_start.device)).detach(),
            "env_drop_rate": aux.get("env_drop_rate", torch.tensor(0.0, device=x_start.device)).detach(),
            "semantic_drop_rate": aux.get("semantic_drop_rate", torch.tensor(0.0, device=x_start.device)).detach(),
            "debug_env_slots_shape": str(tuple(aux["env_slots"].shape)),
            "debug_env_mix_shape": str(tuple(aux["env_mix"].shape)),
        }
        return loss, metrics

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_emb: torch.Tensor,
        gaf: torch.Tensor | None = None,
        semantic_atoms: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred_noise, _ = self.model(x, t, text_emb, gaf, semantic_atoms=semantic_atoms)
        beta_t = extract(self.betas, t, x.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alpha_bars, t, x.shape)
        sqrt_recip_alpha = torch.rsqrt(extract(self.alphas, t, x.shape))
        mean = sqrt_recip_alpha * (x - beta_t * pred_noise / sqrt_one_minus)
        var = extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        nonzero = (t != 0).float().reshape(x.shape[0], *((1,) * (x.ndim - 1)))
        return mean + nonzero * torch.sqrt(var) * noise

    @torch.no_grad()
    def ddim_sample_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_emb: torch.Tensor,
        gaf: torch.Tensor | None = None,
        semantic_atoms: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred_noise, _ = self.model(x, t, text_emb, gaf, semantic_atoms=semantic_atoms)
        alpha_bar_t = extract(self.alpha_bars, t, x.shape)
        prev_t = (t - 1).clamp_min(0)
        alpha_bar_prev = extract(self.alpha_bars, prev_t, x.shape)
        alpha_bar_prev = torch.where(
            (t == 0).reshape(x.shape[0], *((1,) * (x.ndim - 1))),
            torch.ones_like(alpha_bar_prev),
            alpha_bar_prev,
        )
        pred_x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
        x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + torch.sqrt(1.0 - alpha_bar_prev) * pred_noise
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, int, int],
        text_emb: torch.Tensor,
        gaf: torch.Tensor | None = None,
        sampler: str = "ddpm",
        semantic_atoms: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sampler not in {"ddpm", "ddim"}:
            raise ValueError(f"Unsupported sampler: {sampler}")
        self._validate_condition_batch(shape[0], text_emb, gaf, semantic_atoms)
        x = torch.randn(shape, device=text_emb.device)
        for step in reversed(range(self.num_steps)):
            t = torch.full((shape[0],), step, device=text_emb.device, dtype=torch.long)
            if sampler == "ddpm":
                x = self.p_sample(x, t, text_emb, gaf, semantic_atoms)
            else:
                x = self.ddim_sample_step(x, t, text_emb, gaf, semantic_atoms)
        return x


class RectifiedFlow(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        num_steps: int = 100,
        lambda_spectral: float = 0.05,
        spectral_warmup_power: float = 1.0,
        spectral_mode: str = "abs",
        spectral_high_freq_gamma: float = 1.0,
        spectral_dc_weight: float = 0.05,
        lambda_cycle_relation: float = 0.01,
        lambda_triad_contrastive: float = 0.01,
        contrastive_temperature: float = 0.07,
        t_eps: float = 1e-4,
        solver: str = "heun",
        guidance_text: float = 2.0,
        guidance_relation: float = 1.5,
        guidance_joint: float = 1.0,
    ) -> None:
        super().__init__()
        if solver not in {"euler", "heun"}:
            raise ValueError(f"Unsupported rectified-flow solver: {solver}")
        self.model = model
        self.num_steps = int(num_steps)
        self.lambda_spectral = float(lambda_spectral)
        self.spectral_warmup_power = float(spectral_warmup_power)
        self.lambda_cycle_relation = float(lambda_cycle_relation)
        self.lambda_triad_contrastive = float(lambda_triad_contrastive)
        self.contrastive_temperature = float(contrastive_temperature)
        self.t_eps = float(t_eps)
        self.solver = solver
        self.guidance_text = float(guidance_text)
        self.guidance_relation = float(guidance_relation)
        self.guidance_joint = float(guidance_joint)
        self.gadf_spectral_loss = GADFRelationalSpectralLoss(
            mode=spectral_mode,
            high_freq_gamma=spectral_high_freq_gamma,
            dc_weight=spectral_dc_weight,
        )

    @staticmethod
    def _validate_condition_batch(
        batch_size: int,
        text_emb: torch.Tensor,
        gaf: torch.Tensor | None = None,
        semantic_atoms: torch.Tensor | None = None,
    ) -> None:
        if text_emb.shape[0] != batch_size:
            raise ValueError(f"text_emb batch mismatch: expected {batch_size}, got {text_emb.shape[0]}")
        if gaf is not None and gaf.shape[0] != batch_size:
            raise ValueError(f"gaf batch mismatch: expected {batch_size}, got {gaf.shape[0]}")
        if semantic_atoms is not None and semantic_atoms.shape[0] != batch_size:
            raise ValueError(f"semantic_atoms batch mismatch: expected {batch_size}, got {semantic_atoms.shape[0]}")

    def _model_time(self, t: torch.Tensor) -> torch.Tensor:
        return t * float(self.num_steps)

    def _cycle_relation_loss(self, x0_pred: torch.Tensor, x_start: torch.Tensor, target_slots: torch.Tensor | None) -> torch.Tensor:
        if self.lambda_cycle_relation <= 0.0 or target_slots is None or not hasattr(self.model, "relation_slots_from_gaf"):
            return x0_pred.new_zeros(())
        x_true_t = x_start.transpose(1, 2)
        ref_min = x_true_t.amin(dim=-1, keepdim=True).detach()
        ref_max = x_true_t.amax(dim=-1, keepdim=True).detach()
        pred_gaf = self.gadf_spectral_loss.compute_gadf_field(x0_pred, ref_min, ref_max)
        pred_slots = self.model.relation_slots_from_gaf(pred_gaf)
        return 1.0 - F.cosine_similarity(pred_slots, target_slots.detach(), dim=-1).mean()

    def training_loss(
        self,
        x_start: torch.Tensor,
        text_emb: torch.Tensor,
        gaf: torch.Tensor | None = None,
        semantic_atoms: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_condition_batch(x_start.shape[0], text_emb, gaf, semantic_atoms)
        bsz = x_start.shape[0]
        t = torch.rand(bsz, device=x_start.device)
        t = t * (1.0 - 2.0 * self.t_eps) + self.t_eps
        t_view = t.reshape(bsz, *((1,) * (x_start.ndim - 1)))
        noise = torch.randn_like(x_start)
        x_t = (1.0 - t_view) * x_start + t_view * noise
        target_v = noise - x_start
        pred_v, aux = self.model(
            x_t,
            self._model_time(t),
            text_emb,
            gaf,
            semantic_atoms=semantic_atoms,
        )
        loss_flow = F.mse_loss(pred_v, target_v)

        x0_pred = (x_t - t_view * pred_v).clamp(-5.0, 5.0)
        if self.lambda_spectral > 0.0:
            loss_spectral_per_sample = self.gadf_spectral_loss(x0_pred, x_start)
            spectral_weight = (1.0 - t).pow(self.spectral_warmup_power).detach()
            loss_spectral = (loss_spectral_per_sample * spectral_weight).mean()
        else:
            loss_spectral = loss_flow.new_zeros(())
            spectral_weight = loss_flow.new_zeros(())

        slot_aux_loss = aux.get("slot_aux_loss", loss_flow.new_zeros(()))
        grounding_aux_loss = aux.get("grounding_aux_loss", loss_flow.new_zeros(()))
        loss_cycle_relation = self._cycle_relation_loss(x0_pred, x_start, aux.get("env_slots") if gaf is not None else None)
        if {"series_summary", "semantic_summary", "relation_summary"}.issubset(aux):
            loss_triad = triad_contrastive_loss(
                aux.get("series_summary"),
                aux.get("semantic_summary"),
                aux.get("relation_summary"),
                temperature=self.contrastive_temperature,
            )
        else:
            loss_triad = loss_flow.new_zeros(())
        loss = (
            loss_flow
            + self.lambda_spectral * loss_spectral
            + slot_aux_loss
            + grounding_aux_loss
            + self.lambda_cycle_relation * loss_cycle_relation
            + self.lambda_triad_contrastive * loss_triad
        )
        metrics = {
            "loss": loss.detach(),
            "loss_flow": loss_flow.detach(),
            "loss_diff": loss_flow.detach(),
            "loss_spectral": loss_spectral.detach(),
            "loss_slot_aux": slot_aux_loss.detach(),
            "loss_grounding_aux": grounding_aux_loss.detach(),
            "loss_cycle_relation": loss_cycle_relation.detach(),
            "loss_triad_contrastive": loss_triad.detach(),
            "grounding_loss_ot": aux.get("grounding_loss_ot", loss_flow.new_zeros(())).detach(),
            "grounding_loss_mask": aux.get("grounding_loss_mask", loss_flow.new_zeros(())).detach(),
            "grounding_loss_cycle": aux.get("grounding_loss_cycle", loss_flow.new_zeros(())).detach(),
            "grounding_transport_entropy": aux.get("grounding_transport_entropy", loss_flow.new_zeros(())).detach(),
            "grounding_transport_row_error": aux.get("grounding_transport_row_error", loss_flow.new_zeros(())).detach(),
            "grounding_transport_col_error": aux.get("grounding_transport_col_error", loss_flow.new_zeros(())).detach(),
            "slot_diversity_loss": aux.get("slot_diversity_loss", loss_flow.new_zeros(())).detach(),
            "route_entropy_loss": aux.get("route_entropy_loss", loss_flow.new_zeros(())).detach(),
            "text_slot_align_loss": aux.get("text_slot_align_loss", loss_flow.new_zeros(())).detach(),
            "text_env_slot_loss": aux.get("text_env_slot_loss", loss_flow.new_zeros(())).detach(),
            "slot_cosine_mean": aux.get("slot_cosine_mean", loss_flow.new_zeros(())).detach(),
            "route_entropy_scale": aux.get("route_entropy_scale", loss_flow.new_zeros(())).detach(),
            "spectral_weight": spectral_weight.mean().detach(),
            "route_entropy": aux["route_entropy"].detach(),
            "route_max": aux["route_max"].detach(),
            "text_drop_rate": aux.get("text_drop_rate", torch.tensor(0.0, device=x_start.device)).detach(),
            "env_drop_rate": aux.get("env_drop_rate", torch.tensor(0.0, device=x_start.device)).detach(),
            "semantic_drop_rate": aux.get("semantic_drop_rate", torch.tensor(0.0, device=x_start.device)).detach(),
            "debug_env_slots_shape": str(tuple(aux["env_slots"].shape)),
            "debug_env_mix_shape": str(tuple(aux["env_mix"].shape)),
        }
        return loss, metrics

    @torch.no_grad()
    def _predict_velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_emb: torch.Tensor,
        gaf: torch.Tensor | None,
        semantic_atoms: torch.Tensor | None,
        *,
        force_drop_text: bool,
        force_drop_env: bool,
        force_drop_semantic: bool,
    ) -> torch.Tensor:
        pred, _ = self.model(
            x,
            self._model_time(t),
            text_emb,
            gaf,
            semantic_atoms=semantic_atoms,
            force_drop_text=force_drop_text,
            force_drop_env=force_drop_env,
            force_drop_semantic=force_drop_semantic,
        )
        return pred

    @torch.no_grad()
    def _guided_velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_emb: torch.Tensor,
        gaf: torch.Tensor | None,
        semantic_atoms: torch.Tensor | None,
        guidance_text: float,
        guidance_relation: float,
        guidance_joint: float,
    ) -> torch.Tensor:
        uncond = self._predict_velocity(
            x,
            t,
            text_emb,
            gaf,
            semantic_atoms,
            force_drop_text=True,
            force_drop_env=True,
            force_drop_semantic=True,
        )
        text_only = self._predict_velocity(
            x,
            t,
            text_emb,
            gaf,
            semantic_atoms,
            force_drop_text=False,
            force_drop_env=True,
            force_drop_semantic=False,
        )
        relation_only = self._predict_velocity(
            x,
            t,
            text_emb,
            gaf,
            semantic_atoms,
            force_drop_text=True,
            force_drop_env=False,
            force_drop_semantic=True,
        )
        full = self._predict_velocity(
            x,
            t,
            text_emb,
            gaf,
            semantic_atoms,
            force_drop_text=False,
            force_drop_env=False,
            force_drop_semantic=False,
        )
        interaction = full - text_only - relation_only + uncond
        return (
            uncond
            + guidance_text * (text_only - uncond)
            + guidance_relation * (relation_only - uncond)
            + guidance_joint * interaction
        )

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, int, int],
        text_emb: torch.Tensor,
        gaf: torch.Tensor | None = None,
        sampler: str | None = None,
        semantic_atoms: torch.Tensor | None = None,
        guidance_text: float | None = None,
        guidance_relation: float | None = None,
        guidance_joint: float | None = None,
    ) -> torch.Tensor:
        sampler = sampler or self.solver
        if sampler not in {"euler", "heun"}:
            raise ValueError(f"Unsupported rectified-flow sampler: {sampler}")
        self._validate_condition_batch(shape[0], text_emb, gaf, semantic_atoms)
        g_text = self.guidance_text if guidance_text is None else float(guidance_text)
        g_relation = self.guidance_relation if guidance_relation is None else float(guidance_relation)
        g_joint = self.guidance_joint if guidance_joint is None else float(guidance_joint)
        x = torch.randn(shape, device=text_emb.device)
        schedule = torch.linspace(1.0, 0.0, self.num_steps + 1, device=text_emb.device)
        for idx in range(self.num_steps):
            t = schedule[idx].expand(shape[0])
            t_next = schedule[idx + 1].expand(shape[0])
            dt = (t_next - t).reshape(shape[0], *((1,) * (len(shape) - 1)))
            v = self._guided_velocity(x, t, text_emb, gaf, semantic_atoms, g_text, g_relation, g_joint)
            if sampler == "euler":
                x = x + dt * v
            else:
                x_euler = x + dt * v
                v_next = self._guided_velocity(x_euler, t_next, text_emb, gaf, semantic_atoms, g_text, g_relation, g_joint)
                x = x + 0.5 * dt * (v + v_next)
        return x
