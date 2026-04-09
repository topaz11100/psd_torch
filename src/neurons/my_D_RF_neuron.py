"""Proposed variable-branch my_D_RF neuron."""

from __future__ import annotations

import torch
from torch import nn


class MyDRFNeuron(nn.Module):
    """Branch-aggregated resonate-and-fire neuron with branch masks."""

    def __init__(self, input_size: int, hidden_size: int, s_min: int = 1, s_max: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size * s_max)
        self.hidden_size = hidden_size
        self.s_min = s_min
        self.s_max = s_max
        self.rho = nn.Parameter(torch.full((hidden_size,), 0.95))
        self.omega = nn.Parameter(torch.full((hidden_size,), 0.2))
        self.v_th = 1.0

    def forward(self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor], active_branches: int | None = None):
        vr, vi = state
        s = int(active_branches or self.s_max)
        s = max(self.s_min, min(self.s_max, s))
        branch = self.input_proj(x_t).view(x_t.size(0), self.hidden_size, self.s_max)[..., :s].mean(dim=-1)
        rho = torch.clamp(self.rho, 0.0, 0.999)
        vr_new = rho * (torch.cos(self.omega) * vr - torch.sin(self.omega) * vi) + branch
        vi_new = rho * (torch.sin(self.omega) * vr + torch.cos(self.omega) * vi)
        spike = (vr_new >= self.v_th).to(vr_new.dtype)
        vr_new = vr_new - spike * self.v_th
        return spike, (vr_new, vi_new)

    def regularization_loss(self) -> torch.Tensor:
        """Branch + resonance regularization."""

        return 1e-4 * (self.input_proj.weight.pow(2).mean() + self.rho.pow(2).mean() + self.omega.pow(2).mean())
