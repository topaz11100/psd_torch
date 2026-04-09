"""Proposed variable-branch my_DH_SNN neuron."""

from __future__ import annotations

import torch
from torch import nn


class MyDHSNNNeuron(nn.Module):
    """Branch-masked dendritic LIF style neuron with configurable branch range."""

    def __init__(self, input_size: int, hidden_size: int, s_min: int = 1, s_max: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size * s_max)
        self.hidden_size = hidden_size
        self.s_min = s_min
        self.s_max = s_max
        self.v_th = 1.0

    def forward(self, x_t: torch.Tensor, state: torch.Tensor, active_branches: int | None = None):
        s = int(active_branches or self.s_max)
        s = max(self.s_min, min(self.s_max, s))
        branch = self.input_proj(x_t).view(x_t.size(0), self.hidden_size, self.s_max)
        branch = branch[..., :s].mean(dim=-1)
        v = 0.9 * state + branch
        spike = (v >= self.v_th).to(v.dtype)
        v = v - spike * self.v_th
        return spike, v

    def regularization_loss(self) -> torch.Tensor:
        """Simple complexity regularizer for branch weights."""

        return 1e-4 * self.input_proj.weight.pow(2).mean()
