"""Official readout implementations for PSD analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from src.model.first_spike_loss import FirstSpikeAnalysis, FirstSpikeLossAdapter


@dataclass
class ReadoutAnalysis:
    """Generic container used by non-timing readouts."""

    scores: torch.Tensor


class ReadoutBase(nn.Module):
    """Common interface shared by all official readout modes."""

    mode_name: str = 'base'
    requires_output_record: bool = False

    def __init__(self) -> None:
        """Initialize the instance with the provided configuration."""
        super().__init__()

    def output_layer_overrides(self) -> dict[str, Any]:
        """Handle ``output layer overrides`` for the ``readout`` module."""
        return {}

    def analyze_output_record(self, output_membrane: torch.Tensor, output_spike: torch.Tensor) -> ReadoutAnalysis | FirstSpikeAnalysis:
        """Handle ``analyze output record`` for the ``readout`` module."""
        raise NotImplementedError

    def predictions_from_analysis(self, analysis: ReadoutAnalysis | FirstSpikeAnalysis) -> torch.Tensor:
        """Handle ``predictions from analysis`` for the ``readout`` module."""
        raise NotImplementedError

    def loss_from_analysis(self, analysis: ReadoutAnalysis | FirstSpikeAnalysis, target: torch.Tensor, *, training: bool) -> torch.Tensor:
        """Handle ``loss from analysis`` for the ``readout`` module."""
        raise NotImplementedError


class TemporalMembraneReadout(ReadoutBase):
    """Use time-averaged output-membrane logits.

    scores[b, c] = mean_t output_membrane[b, t, c]
    """

    mode_name = 'temporal_membrane'

    def __init__(self, *, skip_initial_steps: int = 0) -> None:
        """Initialize the instance with the provided configuration."""
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.skip_initial_steps = int(skip_initial_steps)
        if self.skip_initial_steps < 0:
            raise ValueError('skip_initial_steps must be non-negative.')

    def output_layer_overrides(self) -> dict[str, Any]:
        """Use a non-spiking, non-resetting output membrane path."""
        return {'emit_spike': False, 'reset_enabled': False}

    def analyze_output_record(self, output_membrane: torch.Tensor, output_spike: torch.Tensor) -> ReadoutAnalysis:
        """Decode class scores from the whole output membrane sequence."""
        del output_spike
        if output_membrane.ndim != 3:
            raise ValueError('temporal_membrane requires output membrane shape (B,T,C).')

        start = int(self.skip_initial_steps)
        time_steps = int(output_membrane.shape[1])
        if start >= time_steps:
            raise ValueError(
                f'temporal_membrane skip_initial_steps={start} is not valid for sequence length {time_steps}.'
            )

        membrane_window = output_membrane[:, start:, :]
        return ReadoutAnalysis(scores=membrane_window.mean(dim=1))

    def predictions_from_analysis(self, analysis: ReadoutAnalysis) -> torch.Tensor:
        """Return argmax class after temporal probability accumulation."""
        return analysis.scores.argmax(dim=1)

    def loss_from_analysis(self, analysis: ReadoutAnalysis, target: torch.Tensor, *, training: bool) -> torch.Tensor:
        """Use the same PyTorch CE path as the existing project and DH-SNN origin code."""
        del training
        return self.loss_fn(analysis.scores, target)


class FinalMembraneReadout(ReadoutBase):
    """Use the final output membrane vector as class logits.

    scores[b, c] = output_membrane[b, -1, c]
    """

    mode_name = 'final_membrane'

    def __init__(self) -> None:
        """Initialize the instance with the provided configuration."""
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def output_layer_overrides(self) -> dict[str, Any]:
        """Use a non-spiking, non-resetting output membrane path."""
        return {'emit_spike': False, 'reset_enabled': False}

    def analyze_output_record(self, output_membrane: torch.Tensor, output_spike: torch.Tensor) -> ReadoutAnalysis:
        """Decode class scores from only the final output membrane timestep."""
        del output_spike
        if output_membrane.ndim != 3:
            raise ValueError('final_membrane requires output membrane shape (B,T,C).')
        if int(output_membrane.shape[1]) < 1:
            raise ValueError('final_membrane requires at least one timestep.')
        return ReadoutAnalysis(scores=output_membrane[:, -1, :])

    def predictions_from_analysis(self, analysis: ReadoutAnalysis) -> torch.Tensor:
        """Return argmax class from final-timestep membrane logits."""
        return analysis.scores.argmax(dim=1)

    def loss_from_analysis(self, analysis: ReadoutAnalysis, target: torch.Tensor, *, training: bool) -> torch.Tensor:
        """Use cross-entropy over final-timestep membrane logits."""
        del training
        return self.loss_fn(analysis.scores, target)


class MaxFireReadout(ReadoutBase):
    """Use output firing count as the class-score vector."""

    mode_name = 'max_fire'

    def __init__(self) -> None:
        """Initialize the instance with the provided configuration."""
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def analyze_output_record(self, output_membrane: torch.Tensor, output_spike: torch.Tensor) -> ReadoutAnalysis:
        """Handle ``analyze output record`` for the ``readout`` module."""
        del output_membrane
        return ReadoutAnalysis(scores=output_spike.sum(dim=1))

    def predictions_from_analysis(self, analysis: ReadoutAnalysis) -> torch.Tensor:
        """Handle ``predictions from analysis`` for the ``readout`` module."""
        return analysis.scores.argmax(dim=1)

    def loss_from_analysis(self, analysis: ReadoutAnalysis, target: torch.Tensor, *, training: bool) -> torch.Tensor:
        """Handle ``loss from analysis`` for the ``readout`` module."""
        del training
        return self.loss_fn(analysis.scores, target)


class FirstSpikeReadout(ReadoutBase):
    """Released first-spike timing path."""

    mode_name = 'first_spike'
    requires_output_record = True

    def __init__(self, *, num_classes: int, sequence_length: int, device: torch.device | str) -> None:
        """Initialize the instance with the provided configuration."""
        super().__init__()
        self.adapter = FirstSpikeLossAdapter(num_classes=num_classes, sequence_length=sequence_length, device=device)

    def analyze_output_record(self, output_membrane: torch.Tensor, output_spike: torch.Tensor) -> FirstSpikeAnalysis:
        """Handle ``analyze output record`` for the ``readout`` module."""
        return self.adapter.analyze(output_membrane, output_spike)

    def predictions_from_analysis(self, analysis: FirstSpikeAnalysis) -> torch.Tensor:
        """Handle ``predictions from analysis`` for the ``readout`` module."""
        return self.adapter.predictions_from_analysis(analysis)

    def loss_from_analysis(self, analysis: FirstSpikeAnalysis, target: torch.Tensor, *, training: bool) -> torch.Tensor:
        """Handle ``loss from analysis`` for the ``readout`` module."""
        return self.adapter.loss_from_analysis(analysis, target, training=training)


class SpikeGRUMaxOverTimeReadout(ReadoutBase):
    """Use max-over-time readout membrane logits for vanilla SpikGRU."""

    mode_name = 'spikegru_max_over_time'

    def __init__(self) -> None:
        """Initialize the instance with the provided configuration."""
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def output_layer_overrides(self) -> dict[str, Any]:
        """SpikGRU owns its non-spiking recurrent readout internally."""
        return {'emit_spike': False, 'reset_enabled': False}

    def analyze_output_record(self, output_membrane: torch.Tensor, output_spike: torch.Tensor) -> ReadoutAnalysis:
        """Return class scores by max-pooling the readout membrane over time."""
        del output_spike
        if output_membrane.ndim != 3:
            raise ValueError('spikegru_max_over_time requires output membrane shape (B,T,C).')
        return ReadoutAnalysis(scores=output_membrane.max(dim=1).values)

    def predictions_from_analysis(self, analysis: ReadoutAnalysis) -> torch.Tensor:
        """Handle predictions from max-over-time logits."""
        return analysis.scores.argmax(dim=1)

    def loss_from_analysis(self, analysis: ReadoutAnalysis, target: torch.Tensor, *, training: bool) -> torch.Tensor:
        """Handle cross-entropy over max-over-time logits."""
        del training
        return self.loss_fn(analysis.scores, target)


_ALLOWED_READOUTS = {'temporal_membrane', 'final_membrane', 'first_spike', 'max_fire', 'max_rate', 'spikegru_max_over_time'}


def canonicalize_readout_mode(mode: str) -> str:
    """Normalize readout mode token to canonical form."""
    token = str(mode).strip().lower()
    if token == 'max_rate':
        return 'max_fire'
    if token in _ALLOWED_READOUTS:
        return token
    raise ValueError(f'Unsupported readout mode: {mode}')


def build_readout(mode: str, *, num_classes: int, sequence_length: int, device: torch.device | str) -> ReadoutBase:
    """Instantiate one official readout mode."""

    token = canonicalize_readout_mode(mode)
    if token == 'temporal_membrane':
        return TemporalMembraneReadout(skip_initial_steps=0)
    if token == 'final_membrane':
        return FinalMembraneReadout()
    if token == 'max_fire':
        return MaxFireReadout()
    if token == 'spikegru_max_over_time':
        return SpikeGRUMaxOverTimeReadout()
    return FirstSpikeReadout(num_classes=num_classes, sequence_length=sequence_length, device=device)


__all__ = [
    'FinalMembraneReadout',
    'FirstSpikeReadout',
    'TemporalMembraneReadout',
    'MaxFireReadout',
    'ReadoutAnalysis',
    'SpikeGRUMaxOverTimeReadout',
    'ReadoutBase',
    'canonicalize_readout_mode',
    'build_readout',
]
