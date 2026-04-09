"""Thin wrapper around released first-spike code path."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ._origin_first_spike import LOSS_MODULE, TIME_ENCODING_MODULE


@dataclass
class FirstSpikeConfig:
    """Hyper-parameters from proposed readout spec."""

    d: int = 16
    a: int = 200
    alpha_fs: float = 0.2
    lambda_treg: float = 0.01
    beta_treg: float = 0.02


class FirstSpikeCriterion(nn.Module):
    """Adapter that exposes unified train/eval APIs for first-spike readout."""

    requires_output_record = True

    def __init__(self, config: FirstSpikeConfig | None = None):
        super().__init__()
        self.config = config or FirstSpikeConfig()
        self.spike2time = TIME_ENCODING_MODULE.Spike2Time()
        self.loss_impl = LOSS_MODULE.LossFn(
            alpha=self.config.alpha_fs,
            beta=self.config.beta_treg,
            lamb=self.config.lambda_treg,
        )

    def forward(self, output_spikes: torch.Tensor, output_membrane: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute released first-spike supervised loss from output records."""

        fst = self.spike2time(output_spikes, output_membrane)
        return self.loss_impl(fst, target)

    def analyze_output_record(self, output_spikes: torch.Tensor, output_membrane: torch.Tensor) -> torch.Tensor:
        """Extract class-wise first spike time tensor for downstream predictions."""

        return self.spike2time(output_spikes, output_membrane)

    def predictions_from_analysis(self, analysis_tensor: torch.Tensor) -> torch.Tensor:
        """Compute argmin first-spike-time predictions."""

        return torch.argmin(analysis_tensor, dim=-1)
