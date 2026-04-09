"""Vanilla RF neuron implementation used for baseline models."""

from __future__ import annotations

import torch
from torch import nn


class VanillaRFCell(nn.Module):
    """Minimal resonate-and-fire style cell with learnable resonance."""

    def __init__(self, hidden_size: int, rho: float = 0.95, omega: float = 0.2, v_th: float = 1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.rho = nn.Parameter(torch.full((hidden_size,), rho))
        self.omega = nn.Parameter(torch.full((hidden_size,), omega))
        self.v_th = v_th

    def forward(self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        vr, vi = state
        rho = torch.clamp(self.rho, 0.0, 0.999)
        phase = self.omega
        vr_new = rho * (torch.cos(phase) * vr - torch.sin(phase) * vi) + x_t
        vi_new = rho * (torch.sin(phase) * vr + torch.cos(phase) * vi)
        spike = (vr_new >= self.v_th).to(vr_new.dtype)
        vr_new = vr_new - spike * self.v_th
        return spike, (vr_new, vi_new)
