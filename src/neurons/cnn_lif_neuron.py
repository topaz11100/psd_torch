"""CNN-LIF neuron blocks reserved for future DVS experiments.

This module intentionally only provides reusable neuron/block implementations.
It is not wired into current PSD experiment entrypoints.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvLIF2d(nn.Module):
    """2D convolution + LIF dynamics for frame/event maps."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, alpha: float = 0.95, v_th: float = 1.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.alpha = alpha
        self.v_th = v_th

    def forward(self, x_t: torch.Tensor, v_prev: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step forward for x_t:(B,C,H,W)."""

        cur = self.conv(x_t)
        if v_prev is None:
            v_prev = torch.zeros_like(cur)
        v = self.alpha * v_prev + cur
        spike = (v >= self.v_th).to(v.dtype)
        v = v - spike * self.v_th
        return spike, v
