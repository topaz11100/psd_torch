"""Vanilla LIF neuron implementation used for baseline models."""

from __future__ import annotations

import torch
from torch import nn


class VanillaLIFCell(nn.Module):
    """Simple subtractive-soft-reset LIF cell with surrogate spike."""

    def __init__(self, hidden_size: int, alpha: float = 0.95, v_th: float = 1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.v_th = v_th

    def forward(self, x_t: torch.Tensor, v_prev: torch.Tensor):
        v = self.alpha * v_prev + x_t
        spike = (v >= self.v_th).to(v.dtype)
        v = v - spike * self.v_th
        return spike, v
