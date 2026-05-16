"""Thin wrapper around the released First-spike timing path.

The official project requirement is to reuse the released time-encoding and loss
logic rather than designing an alternative first-spike objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model._origin_first_spike import load_first_spike_modules


_DEFAULT_LOSS_PARAMS = {
    'loss_mode': 'first_time',
    'FS': {'D': 16, 'A': 200, 'alpha': 0.2},
    'FR': {'alpha': 1.0},
    'treg': {'lambda': 0.01, 'beta': 0.02},
}


@dataclass
class FirstSpikeAnalysis:
    """Structured result of one first-spike output-record analysis.

    ``firing_rate`` keeps the second tensor expected by the released ``LossFn``
    interface. ``scores`` is exposed as a computed convenience view that matches
    the readout specification.
    """

    first_times: torch.Tensor
    firing_rate: torch.Tensor

    @property
    def scores(self) -> torch.Tensor:
        """Handle ``scores`` for the ``FirstSpikeAnalysis`` dataclass."""
        return -self.first_times + self.first_times.min(dim=1, keepdim=True).values


class FirstSpikeLossAdapter(nn.Module):
    """Project-facing wrapper for the released First-spike coding path."""

    def __init__(self, *, num_classes: int, sequence_length: int, device: torch.device | str, loss_params: dict[str, Any] | None = None) -> None:
        """Initialize the instance with the provided configuration."""
        super().__init__()
        loss_params = dict(_DEFAULT_LOSS_PARAMS if loss_params is None else loss_params)
        time_mod, loss_mod = load_first_spike_modules()
        self.device_name = str(device)
        self.num_classes = int(num_classes)
        self.sequence_length = int(sequence_length)
        self.spike_to_time = time_mod.Spike2Time(loss_params, device=self.device_name)
        self.train_loss = loss_mod.LossFn(loss_params, self.num_classes, step=self.sequence_length, mode='train')
        self.eval_loss = loss_mod.LossFn(loss_params, self.num_classes, step=self.sequence_length, mode='test')

    def analyze(self, output_membrane: torch.Tensor, output_spike: torch.Tensor) -> FirstSpikeAnalysis:
        """Convert output spike/membrane records into released first-spike times.

        Parameters are expected in project-standard shape ``(B, T, C)`` and are
        internally transposed to the released code's ``(B, C, T)`` layout.

        The released models record membrane traces in ``output_potentials`` with
        a one-step right shift (`t=0` is zero, `t+1` stores the membrane after
        processing step `t`). The wrapper mirrors that storage convention before
        delegating to the released ``Spike2Time`` implementation.
        """

        if output_membrane.ndim != 3 or output_spike.ndim != 3:
            raise ValueError('first_spike analysis expects output_membrane and output_spike with shape (B,T,C).')
        spikes_bct = output_spike.transpose(1, 2).contiguous()
        membrane_bct = output_membrane.transpose(1, 2).contiguous()
        origin_potentials = torch.zeros_like(membrane_bct)
        origin_potentials[..., 1:] = membrane_bct[..., :-1]
        first_times = self.spike_to_time(spikes_bct, origin_potentials)
        firing_rate = spikes_bct.mean(dim=-1)
        return FirstSpikeAnalysis(first_times=first_times, firing_rate=firing_rate)

    def predictions_from_analysis(self, analysis: FirstSpikeAnalysis) -> torch.Tensor:
        """Handle ``predictions from analysis`` for the ``first_spike_loss`` module."""
        return analysis.first_times.argmin(dim=1)

    def one_hot(self, target: torch.Tensor) -> torch.Tensor:
        """Handle ``one hot`` for the ``first_spike_loss`` module."""
        return F.one_hot(target.to(dtype=torch.long), num_classes=self.num_classes).to(dtype=torch.float32)

    def loss_from_analysis(self, analysis: FirstSpikeAnalysis, target: torch.Tensor, *, training: bool) -> torch.Tensor:
        """Handle ``loss from analysis`` for the ``first_spike_loss`` module."""
        target_one_hot = self.one_hot(target).to(device=analysis.first_times.device)
        loss_module = self.train_loss if training else self.eval_loss
        return loss_module((analysis.first_times, analysis.firing_rate), target_one_hot, loss_mode='first_time')


__all__ = ['FirstSpikeAnalysis', 'FirstSpikeLossAdapter']
