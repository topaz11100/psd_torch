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
        super().__init__()

    def enable_compiled_forward(self, **compile_kwargs: Any) -> tuple[bool, str]:
        """Optionally install readout-local compiled tensor functions."""
        del compile_kwargs
        return False, f'{self.mode_name}_readout_compile_not_applicable'

    def compile_metadata(self) -> dict[str, Any]:
        """Return lightweight readout compile/runtime metadata."""
        return {
            'readout_mode': self.mode_name,
            'readout_compile_applied': False,
            'readout_compile_policy': 'not_applicable',
            'readout_backend': 'eager',
        }

    def output_layer_overrides(self) -> dict[str, Any]:
        return {}

    def analyze_output_record(self, output_membrane: torch.Tensor, output_spike: torch.Tensor) -> ReadoutAnalysis | FirstSpikeAnalysis:
        raise NotImplementedError

    def predictions_from_analysis(self, analysis: ReadoutAnalysis | FirstSpikeAnalysis) -> torch.Tensor:
        raise NotImplementedError

    def loss_from_analysis(self, analysis: ReadoutAnalysis | FirstSpikeAnalysis, target: torch.Tensor, *, training: bool) -> torch.Tensor:
        raise NotImplementedError


class TemporalMembraneReadout(ReadoutBase):
    """Use time-averaged output-membrane logits."""

    mode_name = 'temporal_membrane'

    def __init__(self, *, skip_initial_steps: int = 0) -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.skip_initial_steps = int(skip_initial_steps)
        if self.skip_initial_steps < 0:
            raise ValueError('skip_initial_steps must be non-negative.')

    def output_layer_overrides(self) -> dict[str, Any]:
        return {'emit_spike': False, 'reset_enabled': False}

    def analyze_output_record(self, output_membrane: torch.Tensor, output_spike: torch.Tensor) -> ReadoutAnalysis:
        del output_spike
        if output_membrane.ndim != 3:
            raise ValueError('temporal_membrane requires output membrane shape (B,T,C).')
        start = int(self.skip_initial_steps)
        time_steps = int(output_membrane.shape[1])
        if start >= time_steps:
            raise ValueError(f'temporal_membrane skip_initial_steps={start} is not valid for sequence length {time_steps}.')
        return ReadoutAnalysis(scores=output_membrane[:, start:, :].mean(dim=1))

    def predictions_from_analysis(self, analysis: ReadoutAnalysis) -> torch.Tensor:
        return analysis.scores.argmax(dim=1)

    def loss_from_analysis(self, analysis: ReadoutAnalysis, target: torch.Tensor, *, training: bool) -> torch.Tensor:
        del training
        return self.loss_fn(analysis.scores.float(), target)


class FinalMembraneReadout(ReadoutBase):
    """Use the final output membrane vector as class logits."""

    mode_name = 'final_membrane'

    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def output_layer_overrides(self) -> dict[str, Any]:
        return {'emit_spike': False, 'reset_enabled': False}

    def analyze_output_record(self, output_membrane: torch.Tensor, output_spike: torch.Tensor) -> ReadoutAnalysis:
        del output_spike
        if output_membrane.ndim != 3:
            raise ValueError('final_membrane requires output membrane shape (B,T,C).')
        if int(output_membrane.shape[1]) < 1:
            raise ValueError('final_membrane requires at least one timestep.')
        return ReadoutAnalysis(scores=output_membrane[:, -1, :])

    def predictions_from_analysis(self, analysis: ReadoutAnalysis) -> torch.Tensor:
        return analysis.scores.argmax(dim=1)

    def loss_from_analysis(self, analysis: ReadoutAnalysis, target: torch.Tensor, *, training: bool) -> torch.Tensor:
        del training
        return self.loss_fn(analysis.scores.float(), target)


class MaxFireReadout(ReadoutBase):
    """Use output firing count as the class-score vector."""

    mode_name = 'max_fire'

    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def analyze_output_record(self, output_membrane: torch.Tensor, output_spike: torch.Tensor) -> ReadoutAnalysis:
        del output_membrane
        return ReadoutAnalysis(scores=output_spike.sum(dim=1))

    def predictions_from_analysis(self, analysis: ReadoutAnalysis) -> torch.Tensor:
        return analysis.scores.argmax(dim=1)

    def loss_from_analysis(self, analysis: ReadoutAnalysis, target: torch.Tensor, *, training: bool) -> torch.Tensor:
        del training
        return self.loss_fn(analysis.scores.float(), target)


class FirstSpikeReadout(ReadoutBase):
    """Released first-spike timing semantics with compile-friendly tensor runtime."""

    mode_name = 'first_spike'
    requires_output_record = True

    def __init__(self, *, num_classes: int, sequence_length: int, device: torch.device | str) -> None:
        super().__init__()
        self.adapter = FirstSpikeLossAdapter(num_classes=num_classes, sequence_length=sequence_length, device=device)

    def enable_compiled_forward(self, **compile_kwargs: Any) -> tuple[bool, str]:
        return self.adapter.enable_compiled_forward(**compile_kwargs)

    def compile_metadata(self) -> dict[str, Any]:
        return {
            'readout_mode': self.mode_name,
            'readout_compile_applied': (
                getattr(self.adapter, '_compiled_analyze', None) is not None
                or getattr(self.adapter, '_compiled_train_loss', None) is not None
                or getattr(self.adapter, '_compiled_eval_loss', None) is not None
            ),
            'readout_compile_policy': str(getattr(self.adapter, '_compile_policy', getattr(self.adapter, '_compiled_sequence_policy', 'eager_tensor_origin_equivalent'))),
            'readout_compile_runtime_disabled': bool(getattr(self.adapter, '_compiled_runtime_disabled', getattr(self.adapter, '_sequence_compiled_runtime_disabled', False))),
            'readout_compile_runtime_error': getattr(self.adapter, '_compiled_runtime_error', getattr(self.adapter, '_sequence_compiled_runtime_error', None)),
            'readout_backend': 'compiled_friendly_tensor_first_spike' if (getattr(self.adapter, '_compiled_analyze', None) is not None or getattr(self.adapter, '_compiled_train_loss', None) is not None or getattr(self.adapter, '_compiled_eval_loss', None) is not None) else 'eager_tensor_first_spike',
        }

    def analyze_output_record(self, output_membrane: torch.Tensor, output_spike: torch.Tensor) -> FirstSpikeAnalysis:
        return self.adapter.analyze(output_membrane, output_spike)

    def predictions_from_analysis(self, analysis: FirstSpikeAnalysis) -> torch.Tensor:
        return self.adapter.predictions_from_analysis(analysis)

    def loss_from_analysis(self, analysis: FirstSpikeAnalysis, target: torch.Tensor, *, training: bool) -> torch.Tensor:
        return self.adapter.loss_from_analysis(analysis, target, training=training)


class SpikeGRUMaxOverTimeReadout(ReadoutBase):
    """Use max-over-time readout membrane logits for vanilla SpikGRU."""

    mode_name = 'spikegru_max_over_time'

    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def output_layer_overrides(self) -> dict[str, Any]:
        return {'emit_spike': False, 'reset_enabled': False}

    def analyze_output_record(self, output_membrane: torch.Tensor, output_spike: torch.Tensor) -> ReadoutAnalysis:
        del output_spike
        if output_membrane.ndim != 3:
            raise ValueError('spikegru_max_over_time requires output membrane shape (B,T,C).')
        return ReadoutAnalysis(scores=output_membrane.max(dim=1).values)

    def predictions_from_analysis(self, analysis: ReadoutAnalysis) -> torch.Tensor:
        return analysis.scores.argmax(dim=1)

    def loss_from_analysis(self, analysis: ReadoutAnalysis, target: torch.Tensor, *, training: bool) -> torch.Tensor:
        del training
        return self.loss_fn(analysis.scores.float(), target)


_ALLOWED_READOUTS = {'temporal_membrane', 'final_membrane', 'first_spike', 'max_fire', 'max_rate', 'spikegru_max_over_time'}


def canonicalize_readout_mode(mode: str) -> str:
    token = str(mode).strip().lower()
    if token == 'max_rate':
        return 'max_fire'
    if token in _ALLOWED_READOUTS:
        return token
    raise ValueError(f'Unsupported readout mode: {mode}')


def build_readout(mode: str, *, num_classes: int, sequence_length: int, device: torch.device | str) -> ReadoutBase:
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
