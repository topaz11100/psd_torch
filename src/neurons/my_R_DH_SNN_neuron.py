"""Proposed recurrent variable-branch my_R_DH_SNN neuron."""

from __future__ import annotations

import torch
from torch import nn


class MyRDHSNNNeuron(nn.Module):
    """Recurrent extension of branch-masked dendritic neuron."""

    def __init__(self, input_size: int, hidden_size: int, s_min: int = 1, s_max: int = 4):
        super().__init__()
        self.ff = nn.Linear(input_size, hidden_size * s_max)
        self.rec = nn.Linear(hidden_size, hidden_size * s_max, bias=False)
        self.hidden_size = hidden_size
        self.s_min = s_min
        self.s_max = s_max
        self.v_th = 1.0

    def forward(self, x_t: torch.Tensor, state: torch.Tensor, active_branches: int | None = None):
        s = int(active_branches or self.s_max)
        s = max(self.s_min, min(self.s_max, s))
        ff = self.ff(x_t).view(x_t.size(0), self.hidden_size, self.s_max)
        rr = self.rec(state).view(x_t.size(0), self.hidden_size, self.s_max)
        v = 0.9 * state + (ff[..., :s] + rr[..., :s]).mean(dim=-1)
        spike = (v >= self.v_th).to(v.dtype)
        v = v - spike * self.v_th
        return spike, v

    def regularization_loss(self) -> torch.Tensor:
        """Combined regularization for feed-forward and recurrent params."""

        return 1e-4 * (self.ff.weight.pow(2).mean() + self.rec.weight.pow(2).mean())
