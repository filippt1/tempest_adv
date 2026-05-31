"""Conditional Flow Matching scheduler utilities."""

from __future__ import annotations

import torch
from torch import Tensor


class ConditionalFlowMatchingScheduler:
    """Deterministic scheduler for Euler integration of conditional flow velocity."""

    def __init__(self, device: torch.device | None = None):
        self.time_schedule = torch.empty(0, dtype=torch.float32)
        if device is not None:
            self.to(device)

    def to(self, device: torch.device) -> "ConditionalFlowMatchingScheduler":
        self.time_schedule = self.time_schedule.to(device)
        return self

    def set_integration_schedule(self, num_inference_steps: int, device: torch.device) -> None:
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1")

        self.time_schedule = torch.linspace(
            0.0,
            1.0,
            steps=num_inference_steps + 1,
            device=device,
            dtype=torch.float32,
        )

    def step(self, pred_v: Tensor, timestep_idx: int, x_t: Tensor) -> Tensor:
        t_curr = self.time_schedule[timestep_idx]
        t_next = self.time_schedule[timestep_idx + 1]
        dt = t_next - t_curr
        return x_t + pred_v * dt


class DiffusionSchedule:
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02, device: torch.device | str = "cpu"):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_start: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Sample from q(x_t | x_0). t is 1-indexed."""
        t_idx = t - 1
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t_idx].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_idx].view(-1, 1, 1, 1)
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise
