"""Readout functions for final_membrane, first_spike, and max_rate."""

from __future__ import annotations

import torch


def final_membrane_readout(output_membrane: torch.Tensor) -> torch.Tensor:
    """Use final membrane value as raw class score."""

    return output_membrane[:, -1, :]


def max_rate_readout(output_spikes: torch.Tensor) -> torch.Tensor:
    """Use mean firing rate as raw class score."""

    return output_spikes.mean(dim=1)


def apply_readout(mode: str, output_spikes: torch.Tensor, output_membrane: torch.Tensor) -> torch.Tensor:
    """Apply configured readout mode and return pre-softmax scores."""

    if mode == "final_membrane":
        return final_membrane_readout(output_membrane)
    if mode == "max_rate":
        return max_rate_readout(output_spikes)
    if mode == "first_spike":
        return -output_spikes.argmax(dim=1).to(output_membrane.dtype)
    raise ValueError(f"Unsupported readout mode: {mode}")
