"""Shared low-level helpers for project neuron layers."""

from __future__ import annotations

import numpy as np
import torch


class FastSigmoidSpike(torch.autograd.Function):
    """Simple surrogate spike used by project-defined layers."""

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, slope: float = 10.0) -> torch.Tensor:
        """Run the forward pass."""
        ctx.save_for_backward(input_tensor)
        ctx.slope = float(slope)
        return (input_tensor > 0).to(dtype=input_tensor.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Run the backward pass."""
        (input_tensor,) = ctx.saved_tensors
        slope = ctx.slope
        grad = grad_output / (1.0 + slope * input_tensor.abs()).pow(2)
        return grad, None


surrogate_spike = FastSigmoidSpike.apply


def trim_open_interval(left: float, right: float, *, epsilon: float = 1.0e-4) -> tuple[float, float]:
    """Trim a possibly closed interval into a numerically stable open interval."""

    left = float(left)
    right = float(right)
    if right < left:
        raise ValueError(f'Invalid interval [{left}, {right}].')
    if np.isclose(left, right):
        return left, right
    width = right - left
    delta = min(float(epsilon), 0.25 * width)
    return left + delta, right - delta


def logit(x: torch.Tensor) -> torch.Tensor:
    """Handle ``logit`` for the ``_common`` module."""
    x = torch.clamp(x, min=1.0e-6, max=1.0 - 1.0e-6)
    return torch.log(x) - torch.log1p(-x)


__all__ = [
    'FastSigmoidSpike',
    'logit',
    'surrogate_spike',
    'trim_open_interval',
]
