"""
transition.py — Minimal Transition Model training utilities for my-app (pixel-space, no VAE, no CFG).

This file is intentionally **self-contained** (no imports from the main TiM repo modules),
so `my-app` can run standalone inside the docker container.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def expand_t_like_x(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape time tensor [B] to broadcastable shape of x: [B, 1, 1, 1]."""
    dims = [1] * (x.ndim - 1)
    return t.view(t.size(0), *dims)


class Transport:
    """
    Minimal transport interface:
    - sample_t: sample t values
    - c_noise: map t to model conditioning domain
    - interpolant: compute alpha_t, sigma_t and their derivatives
    - target: compute F_target for transition objective
    - from_x_t_to_x_r: sampling update
    """
    def __init__(self, sigma_d: float, T_max: float, T_min: float):
        self.sigma_d = float(sigma_d)
        self.T_max = float(T_max)
        self.T_min = float(T_min)

    def sample_t(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def c_noise(self, t: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def interpolant(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # pragma: no cover
        raise NotImplementedError

    def target(
        self,
        x_t: torch.Tensor,
        v_t: torch.Tensor,
        x: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        dF_dv_dt: torch.Tensor,
    ) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def from_x_t_to_x_r(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        F: torch.Tensor,
        s_ratio: float = 0.0,
    ) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class OT_FM(Transport):
    """
    Linear path Flow-Matching transport:
      alpha_t = 1 - t
      sigma_t = t
      x_t = alpha_t x + sigma_t z
    We use the Transition Model target:
      F_target = v_t - (t-r) * dF/dt
    And transition sampling update:
      x_r = x_t - (t-r) * F
    """
    def __init__(self, P_mean: float = -0.4, P_std: float = 1.0, sigma_d: float = 1.0, T_max: float = 1.0, T_min: float = 0.0):
        super().__init__(sigma_d=sigma_d, T_max=T_max, T_min=T_min)
        self.P_mean = float(P_mean)
        self.P_std = float(P_std)

    def interpolant(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha_t = 1 - t
        sigma_t = t
        d_alpha_t = torch.full_like(t, -1.0)
        d_sigma_t = torch.full_like(t, 1.0)
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def sample_t(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        rnd_normal = torch.randn((batch_size,), dtype=dtype, device=device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        t = sigma / (1 + sigma)  # in [0, 1]
        return t

    def c_noise(self, t: torch.Tensor) -> torch.Tensor:
        # keep as-is (bounded)
        return t

    def target(
        self,
        x_t: torch.Tensor,
        v_t: torch.Tensor,
        x: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        dF_dv_dt: torch.Tensor,
    ) -> torch.Tensor:
        return v_t - (t - r) * dF_dv_dt

    def from_x_t_to_x_r(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        F: torch.Tensor,
        s_ratio: float = 0.0,
    ) -> torch.Tensor:
        # deterministic transition update
        x_r = x_t - (t - r) * F
        # optional stochasticity (kept for parity with reference implementation; default off)
        if s_ratio > 0.0:
            # reference-style noise injection
            z = x_t + (1 - t) * F
            epsilon = torch.randn_like(z)
            dt = t - r
            x_r = x_r - s_ratio * z * dt + torch.sqrt(s_ratio * 2 * t * dt) * epsilon
        return x_r


def mean_flat(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(x, dim=list(range(1, x.ndim)))


@dataclass
class LossOutputs:
    loss: torch.Tensor
    denoising_loss: torch.Tensor
    directional_loss: torch.Tensor
    weight_mean: torch.Tensor


class SimpleTransitionSchedule:
    def __init__(
        self,
        transport: Transport,
        diffusion_ratio: float = 0.0,
        consistency_ratio: float = 0.0,
        derivative_type: str = "dde",  # "dde" or "jvp" (we implement only dde in my-app)
        differential_epsilon: float = 0.005,
        weight_t_and_r: bool = True,
        weight_time_type: str = "constant",
        weight_time_tangent: bool = False,
        weight_time_sigmoid: bool = False,
        adaptive_weighting: bool = True,
        use_dir_loss: bool = True,
    ):
        self.transport = transport
        self.diffusion_ratio = diffusion_ratio
        self.consistency_ratio = consistency_ratio
        self.derivative_type = derivative_type
        self.differential_epsilon = differential_epsilon
        self.weight_t_and_r = weight_t_and_r
        self.weight_time_type = weight_time_type
        self.weight_time_tangent = weight_time_tangent
        self.weight_time_sigmoid = weight_time_sigmoid
        self.use_adaptive_weighting = adaptive_weighting
        self.use_dir_loss = use_dir_loss

        if self.derivative_type != "dde":
            raise NotImplementedError("my-app only implements derivative_type='dde' for simplicity.")

    def sample_t_and_r(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, int]:
        t_1 = self.transport.sample_t(batch_size=batch_size, dtype=dtype, device=device)
        t_2 = self.transport.sample_t(batch_size=batch_size, dtype=dtype, device=device)
        t = torch.maximum(t_1, t_2)
        r = torch.minimum(t_1, t_2)
        n_diffusion = round(self.diffusion_ratio * len(t))
        r[:n_diffusion] = t[:n_diffusion]
        n_consistency = round(self.consistency_ratio * len(t))
        if n_consistency != 0:
            r[-n_consistency:] = self.transport.T_min
        return t, r, n_diffusion

    def prepare_input(self, x: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        b = x.shape[0]
        t, r, n_diffusion = self.sample_t_and_r(batch_size=b, dtype=x.dtype, device=x.device)
        t_x, r_x = expand_t_like_x(t, x), expand_t_like_x(r, x)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.transport.interpolant(t_x)
        x_t = alpha_t * x + sigma_t * z
        v_t = d_alpha_t * x + d_sigma_t * z
        return x_t, v_t, t_x, r_x, n_diffusion

    def model_forward(self, model, x_t: torch.Tensor, t_x: torch.Tensor, r_x: torch.Tensor, model_kwargs: Dict, rng_state: Optional[torch.Tensor]):
        t_input = self.transport.c_noise(t_x.flatten())
        r_input = self.transport.c_noise(r_x.flatten())
        # ensure deterministic across finite-difference calls if any randomness exists in the model
        if rng_state is not None and x_t.is_cuda:
            torch.cuda.set_rng_state(rng_state)
        out, _ = model(x_t, t_input, r_input, **model_kwargs)
        # keep numerics stable under bf16 autocast by promoting to fp32 for target/loss math
        return out.float()

    @torch.no_grad()
    def dde_derivative(
        self,
        model,
        x: torch.Tensor,
        z: torch.Tensor,
        t_x: torch.Tensor,
        r_x: torch.Tensor,
        model_kwargs: Dict,
        rng_state: Optional[torch.Tensor],
        n_diffusion: int,
    ) -> torch.Tensor:
        if n_diffusion == x.size(0):
            return torch.zeros_like(x)

        dF = torch.zeros_like(x)
        x_s, z_s, t_s, r_s = x[n_diffusion:], z[n_diffusion:], t_x[n_diffusion:], r_x[n_diffusion:]
        model_kwargs_s = {k: (v[n_diffusion:] if isinstance(v, torch.Tensor) else v) for k, v in model_kwargs.items()}

        eps = self.differential_epsilon

        def xfunc(t_in: torch.Tensor) -> torch.Tensor:
            # clamp to avoid stepping outside domain (important for bounded transports like OT_FM)
            if hasattr(self.transport, "T_min") and hasattr(self.transport, "T_max"):
                t_in = t_in.clamp(self.transport.T_min + 1e-6, self.transport.T_max - 1e-6)
            alpha_t, sigma_t, _, _ = self.transport.interpolant(t_in)
            x_t = alpha_t * x_s + sigma_t * z_s
            return self.model_forward(model, x_t, t_in, r_s, model_kwargs_s, rng_state)

        dF_dt = (xfunc(t_s + eps) - xfunc(t_s - eps)) * (0.5 / eps)
        dF[n_diffusion:] = dF_dt
        return dF

    def time_weighting(self, t_x: torch.Tensor, r_x: torch.Tensor, n_diffusion: int) -> torch.Tensor:
        t_w, r_w = t_x, r_x
        if self.weight_time_tangent:
            t_w, r_w = torch.tan(t_w), torch.tan(r_w)
        elif self.weight_time_sigmoid:
            t_w, r_w = t_w / (1 - t_w), r_w / (1 - r_w)
        if self.weight_t_and_r:
            delta = (t_w - r_w).flatten()
        else:
            delta = t_w.flatten()
        if self.weight_time_type == "constant":
            w = torch.ones_like(delta)
        elif self.weight_time_type == "reciprocal":
            w = 1 / (delta + self.transport.sigma_d)
        elif self.weight_time_type == "sqrt":
            w = 1 / (delta + self.transport.sigma_d).sqrt()
        elif self.weight_time_type == "square":
            w = 1 / (delta + self.transport.sigma_d) ** 2
        elif self.weight_time_type == "Soft-Min-SNR":
            w = 1 / (delta ** 2 + self.transport.sigma_d ** 2)
        else:
            raise NotImplementedError(self.weight_time_type)
        w[:n_diffusion] = 1.0
        return w

    def adaptive_w(self, loss_vec: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        return 1 / (loss_vec.detach() + eps)

    def loss(self, model, x: torch.Tensor, z: torch.Tensor, model_kwargs: Dict) -> LossOutputs:
        x_t, v_t, t_x, r_x, n_diffusion = self.prepare_input(x, z)
        rng_state = torch.cuda.get_rng_state() if x_t.is_cuda else None

        F_pred = self.model_forward(model, x_t, t_x, r_x, model_kwargs, rng_state)
        dF_dv_dt = self.dde_derivative(model, x, z, t_x, r_x, model_kwargs, rng_state, n_diffusion)
        F_target = self.transport.target(x_t, v_t, x, z, t_x, r_x, dF_dv_dt)

        denoising_loss = mean_flat((F_pred - F_target.float()) ** 2)
        denoising_loss = torch.nan_to_num(denoising_loss, nan=0, posinf=1e5, neginf=-1e5)

        directional_loss = torch.zeros_like(denoising_loss)
        if self.use_dir_loss:
            directional_loss = mean_flat(1 - F.cosine_similarity(F_pred, F_target.float(), dim=1))
            directional_loss = torch.nan_to_num(directional_loss, nan=0, posinf=1e5, neginf=-1e5)
            denoising_loss = denoising_loss + directional_loss

        w_time = self.time_weighting(t_x, r_x, n_diffusion)
        if self.use_adaptive_weighting:
            w = w_time * self.adaptive_w(denoising_loss)
        else:
            w = w_time

        loss = (w * denoising_loss).mean()
        return LossOutputs(
            loss=loss,
            denoising_loss=denoising_loss.mean(),
            directional_loss=directional_loss.mean(),
            weight_mean=w.mean(),
        )

    @torch.no_grad()
    def sample(
        self,
        model,
        global_attrs: torch.Tensor,
        z: torch.Tensor,
        num_steps: int = 2,
        sample_type: str = "transition",  # "transition" or "ddiffusion" (for debugging)
    ) -> torch.Tensor:
        """
        Returns a trajectory tensor of shape [num_steps+1, B, C, H, W].
        """
        dtype = z.dtype
        # Keep the same behavior as reference: integrate from T_max down to T_min
        t_steps = torch.linspace(self.transport.T_max, self.transport.T_min, num_steps + 1, dtype=torch.float64, device=z.device)
        x_cur = z.to(torch.float64)
        traj = [z]

        for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
            if sample_type == "transition":
                r_use = t_next
            elif sample_type == "ddiffusion":
                r_use = t_cur
            else:
                raise ValueError(sample_type)

            t_in = torch.full((z.shape[0],), self.transport.c_noise(t_cur).item(), device=z.device, dtype=dtype)
            r_in = torch.full((z.shape[0],), self.transport.c_noise(r_use).item(), device=z.device, dtype=dtype)
            F_pred, _ = model(x_cur.to(dtype), t_in, r_in, global_attrs)
            x_next = self.transport.from_x_t_to_x_r(x_cur, t_cur, t_next, F_pred.to(torch.float64), s_ratio=0.0)
            traj.append(x_next.to(dtype))
            x_cur = x_next

        return torch.stack(traj, dim=0).to(torch.float32)


