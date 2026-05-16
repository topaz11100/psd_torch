"""D-RF origin-code thin wrapper built on the released BiRFModel path."""

from __future__ import annotations

import torch
from torch import nn

from src.neurons._origin_imports import load_d_rf_module


class _CapturedAct1:
    """Capture membrane traces while delegating spikes to the released act1.

    ``BiRFModel.forward`` computes ``self.act1(y.real - 1.)``. The wrapper uses
    that released forward path directly and only patches ``act1`` long enough to:

    1. expose the raw membrane ``y.real`` for tracing, and
    2. shift the effective threshold from the released ``1.0`` to the runtime
       ``v_threshold`` without editing the author forward body.
    """

    def __init__(self, base_act: object, *, v_threshold: float) -> None:
        self.base_act = base_act
        self.v_threshold = float(v_threshold)
        self.last_raw_membrane: torch.Tensor | None = None
        self.last_threshold_shifted: torch.Tensor | None = None

    def __call__(self, threshold_shifted_from_origin: torch.Tensor) -> torch.Tensor:
        raw_membrane = threshold_shifted_from_origin + 1.0
        shifted = raw_membrane - float(self.v_threshold)
        self.last_raw_membrane = raw_membrane
        self.last_threshold_shifted = shifted
        return self.base_act(shifted)


class DRFLayer(nn.Module):
    """Sequence layer using the released D-RF ``BiRFModel`` dynamics.

    The project-owned portion is limited to an optional feature-dimension
    adapter when ``input_size != output_size``. The D-RF membrane/spike path is
    taken directly from the released ``BiRFModel.forward`` implementation.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        branch: int = 4,
        emit_spike: bool = True,
        reset_enabled: bool = True,
        v_threshold: float = 1.0,
    ) -> None:
        """Initialize the instance with the provided configuration."""
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.branch = int(branch)
        self.emit_spike = bool(emit_spike)
        self.reset_enabled = bool(reset_enabled)
        self.v_threshold = float(v_threshold)
        if self.input_size == self.output_size:
            self.input_adapter = nn.Identity()
        else:
            self.input_adapter = nn.Linear(self.input_size, self.output_size, bias=False)
        origin = load_d_rf_module()
        self.origin = origin.BiRFModel(d_model=self.output_size, d_state=self.branch)

    def _project_input(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """Internal helper that project input into the released D-RF width."""
        projected = self.input_adapter(input_sequence)
        return projected.transpose(1, 2).contiguous()

    def _origin_forward_with_capture(self, projected_bht: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run released ``BiRFModel.forward`` and capture membrane intermediates."""

        original_act1 = self.origin.act1
        captured_act = _CapturedAct1(original_act1, v_threshold=float(self.v_threshold))
        self.origin.act1 = captured_act
        try:
            spike = self.origin(projected_bht)
        finally:
            self.origin.act1 = original_act1
        raw_membrane = captured_act.last_raw_membrane
        shifted_membrane = captured_act.last_threshold_shifted
        if raw_membrane is None or shifted_membrane is None:
            raise RuntimeError('Released BiRFModel.forward did not invoke act1 as expected; membrane capture failed.')
        return raw_membrane, shifted_membrane, spike

    def effective_input_weight(self) -> torch.Tensor | None:
        """Handle ``effective input weight`` for the ``D_RF_neuron`` module."""
        if isinstance(self.input_adapter, nn.Identity):
            return None
        return self.input_adapter.weight

    def forward(self, input_sequence: torch.Tensor, *, return_traces: bool = False) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Run the forward pass."""
        self._last_layer_input = None
        if input_sequence.ndim != 3:
            raise ValueError(f'Expected shape (B,T,C), got {tuple(input_sequence.shape)}')
        projected_bht = self._project_input(input_sequence)
        raw_membrane_bht, shifted_membrane_bht, spike_bht = self._origin_forward_with_capture(projected_bht)
        record_raw_membrane = (not self.emit_spike) and (not self.reset_enabled)
        if not self.emit_spike:
            spike_bht = torch.zeros_like(spike_bht)
        membrane_record = raw_membrane_bht if record_raw_membrane else shifted_membrane_bht
        self._last_layer_input = projected_bht.transpose(1, 2).contiguous() if return_traces else None
        membrane_seq = membrane_record.transpose(1, 2).contiguous() if return_traces else None
        spike_seq = spike_bht.transpose(1, 2).contiguous()
        return membrane_seq, spike_seq

    def filter_stats_vectors(self) -> dict[str, torch.Tensor]:
        """Handle ``filter stats vectors`` for the ``D_RF_neuron`` module."""
        return {}



__all__ = ['DRFLayer']
