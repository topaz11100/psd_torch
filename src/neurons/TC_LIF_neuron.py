"""TC-LIF thin wrapper around the released ``TCLIFNode`` implementation."""

from __future__ import annotations

import math

import torch
from torch import nn

from src.neurons._origin_imports import load_tc_lif_module


class _OriginCompatibleSurrogate:
    """Adapt the project surrogate to the origin callable signature."""

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # noqa: ARG002 - origin compatibility
        """Call the object like a function."""
        from src.neurons._common import surrogate_spike

        return surrogate_spike(x)


class TCLIFLayer(nn.Module):
    """Sequence wrapper that delegates neuron dynamics to the origin node."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        recurrent: bool = False,
        v_threshold: float = 1.0,
        emit_spike: bool = True,
        reset_enabled: bool = True,
        gamma: float = 0.5,
    ) -> None:
        """Initialize the instance with the provided configuration."""
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.recurrent = bool(recurrent)
        self.v_threshold = float(v_threshold)
        self.emit_spike = bool(emit_spike)
        self.reset_enabled = bool(reset_enabled)
        self.gamma = float(gamma)
        self.input_weight = nn.Parameter(torch.empty(self.output_size, self.input_size))
        if self.recurrent:
            self.recurrent_weight = nn.Parameter(torch.empty(self.output_size, self.output_size))
        else:
            self.register_parameter('recurrent_weight', None)
        origin = load_tc_lif_module()
        self.node = origin.TCLIFNode(
            v_threshold=self.v_threshold,
            surrogate_function=_OriginCompatibleSurrogate(),
            gamma=self.gamma,
            hard_reset=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset parameters."""
        nn.init.kaiming_uniform_(self.input_weight, a=math.sqrt(5.0))
        if self.recurrent_weight is not None:
            nn.init.orthogonal_(self.recurrent_weight)
        if hasattr(self.node, 'decay_factor'):
            nn.init.constant_(self.node.decay_factor, 0.0)

    def effective_input_weight(self) -> torch.Tensor:
        """Handle ``effective input weight`` for the ``TC_LIF_neuron`` module."""
        return self.input_weight

    def forward(self, input_sequence: torch.Tensor, *, return_traces: bool = False) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Run the forward pass."""
        self._last_layer_input = None
        if input_sequence.ndim != 3:
            raise ValueError(f'Expected shape (B,T,C), got {tuple(input_sequence.shape)}')
        batch_size, time_steps, _ = input_sequence.shape
        self.node.reset()
        self.node.v_threshold = float(self.v_threshold)
        self.node.gamma = float(self.gamma)
        prev_spike = input_sequence.new_zeros(batch_size, self.output_size)
        record_raw_membrane = (not self.emit_spike) and (not self.reset_enabled)
        mem_steps: list[torch.Tensor] | None = [] if return_traces else None
        layer_input_steps: list[torch.Tensor] | None = [] if return_traces else None
        spike_steps: list[torch.Tensor] = []
        for time_index in range(time_steps):
            current = input_sequence[:, time_index, :] @ self.input_weight.t()
            if self.recurrent_weight is not None:
                current = current + prev_spike @ self.recurrent_weight.t()
            self.node.v_float_to_tensor(current)
            self.node.neuronal_charge(current)
            threshold = torch.as_tensor(self.node.v_threshold, device=current.device, dtype=current.dtype)
            membrane_value = self.node.v
            membrane_signal = membrane_value - threshold
            if record_raw_membrane:
                spike = torch.zeros_like(membrane_signal)
            else:
                raw_spike = self.node.neuronal_fire()
                spike = raw_spike if self.emit_spike else torch.zeros_like(raw_spike)
                if self.reset_enabled:
                    self.node.neuronal_reset(raw_spike)
            if mem_steps is not None and layer_input_steps is not None:
                layer_input_steps.append(current)
                mem_steps.append(membrane_value if record_raw_membrane else membrane_signal)
            spike_steps.append(spike)
            prev_spike = spike
        self._last_layer_input = torch.stack(layer_input_steps, dim=1) if layer_input_steps is not None else None
        return (torch.stack(mem_steps, dim=1) if mem_steps is not None else None), torch.stack(spike_steps, dim=1)

    def filter_stats_vectors(self) -> dict[str, torch.Tensor]:
        """Handle ``filter stats vectors`` for the ``TC_LIF_neuron`` module."""
        return {}



try:
    from src.neurons.spikingjelly_compat import install_spikingjelly_contract as _install_spikingjelly_contract
    _install_spikingjelly_contract(TCLIFLayer)
except Exception:  # pragma: no cover - defensive import fallback
    pass

__all__ = ['TCLIFLayer']
