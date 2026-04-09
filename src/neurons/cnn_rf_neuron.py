"""CNN-RF neuron blocks reserved for future DVS experiments.

This module intentionally only provides reusable neuron/block implementations.
It is not wired into current PSD experiment entrypoints.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvRF2d(nn.Module):
    """2D convolution + resonate-and-fire dynamics for frame/event maps."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, rho: float = 0.95, omega: float = 0.2, v_th: float = 1.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.rho = nn.Parameter(torch.full((out_channels, 1, 1), rho))
        self.omega = nn.Parameter(torch.full((out_channels, 1, 1), omega))
        self.v_th = v_th

    def forward(
        self,
        x_t: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Single-step forward for x_t:(B,C,H,W)."""

        cur = self.conv(x_t)
        if state is None:
            vr = torch.zeros_like(cur)
            vi = torch.zeros_like(cur)
        else:
            vr, vi = state
        rho = torch.clamp(self.rho, 0.0, 0.999)
        vr_new = rho * (torch.cos(self.omega) * vr - torch.sin(self.omega) * vi) + cur
        vi_new = rho * (torch.sin(self.omega) * vr + torch.cos(self.omega) * vi)
        spike = (vr_new >= self.v_th).to(vr_new.dtype)
        vr_new = vr_new - spike * self.v_th
        return spike, (vr_new, vi_new)
