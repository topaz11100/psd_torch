"""Project-standard vanilla discrete-time LIF layer."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.neurons._common import logit, surrogate_spike, trim_open_interval

def _positive_threshold_init(v_threshold: float, size: int, *, eps: float) -> torch.Tensor:
    value = max(float(v_threshold) - float(eps), float(eps))
    raw = math.log(math.expm1(value))
    return torch.full((int(size),), float(raw), dtype=torch.float32)



class LIFLayer(nn.Module):
    """Dense discrete-time Leaky Integrate-and-Fire layer."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        recurrent: bool = False,
        v_threshold: float = 1.0,
        trainable_threshold: bool = False,
        reset_mode: str = 'soft_reset',
        alpha_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
        input_mask: torch.Tensor | None = None,
        recurrent_mask: torch.Tensor | None = None,
        emit_spike: bool = True,
        reset_enabled: bool = True,
    ) -> None:
        """Initialize the instance with the provided configuration."""
        super().__init__()
        if reset_mode not in {'soft_reset', 'hard_reset'}:
            raise ValueError("reset_mode must be 'soft_reset' or 'hard_reset'.")
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.recurrent = bool(recurrent)
        self.reset_mode = str(reset_mode)
        self.emit_spike = bool(emit_spike)
        self.reset_enabled = bool(reset_enabled)
        self.trainable_threshold = bool(trainable_threshold)
        self.threshold_eps = 1.0e-6

        self.input_weight = nn.Parameter(torch.empty(self.output_size, self.input_size))
        if self.recurrent:
            self.recurrent_weight = nn.Parameter(torch.empty(self.output_size, self.output_size))
        else:
            self.register_parameter('recurrent_weight', None)

        if input_mask is None:
            input_mask = torch.ones(self.output_size, self.input_size, dtype=torch.float32)
        if self.recurrent and recurrent_mask is None:
            recurrent_mask = torch.ones(self.output_size, self.output_size, dtype=torch.float32)
        self.register_buffer('input_mask', input_mask.to(dtype=torch.float32))
        if recurrent_mask is not None:
            self.register_buffer('recurrent_mask', recurrent_mask.to(dtype=torch.float32))
        else:
            self.register_buffer('recurrent_mask', None)

        if alpha_bounds is None:
            lower = torch.zeros(self.output_size, dtype=torch.float32)
            upper = torch.ones(self.output_size, dtype=torch.float32)
        else:
            lower, upper = alpha_bounds
            lower = lower.detach().clone().to(dtype=torch.float32)
            upper = upper.detach().clone().to(dtype=torch.float32)
        self.register_buffer('alpha_lower', lower)
        self.register_buffer('alpha_upper', upper)
        self.alpha_raw = nn.Parameter(torch.empty(self.output_size))

        threshold_init = torch.full((self.output_size,), float(v_threshold), dtype=torch.float32)
        if self.trainable_threshold:
            self.v_threshold_param = nn.Parameter(_positive_threshold_init(v_threshold, self.output_size, eps=self.threshold_eps))
        else:
            self.register_buffer('v_threshold_buffer', threshold_init)
            self.register_parameter('v_threshold_param', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset parameters."""
        nn.init.kaiming_uniform_(self.input_weight, a=math.sqrt(5.0))
        if self.recurrent_weight is not None:
            nn.init.orthogonal_(self.recurrent_weight)
        init_alpha = torch.empty_like(self.alpha_lower)
        for index in range(init_alpha.numel()):
            left, right = trim_open_interval(float(self.alpha_lower[index]), float(self.alpha_upper[index]))
            if right <= left:
                value = 0.5 * (left + right)
            else:
                value = float(torch.empty(1).uniform_(left, right).item())
            init_alpha[index] = value
        with torch.no_grad():
            self.alpha_raw.copy_(logit(init_alpha))

    def effective_alpha(self) -> torch.Tensor:
        """Handle ``effective alpha`` for the ``LIF_neuron`` module."""
        sigma = torch.sigmoid(self.alpha_raw)
        return self.alpha_lower + (self.alpha_upper - self.alpha_lower) * sigma

    def effective_threshold(self) -> torch.Tensor:
        """Handle ``effective threshold`` for the ``LIF_neuron`` module."""
        if self.v_threshold_param is not None:
            return F.softplus(self.v_threshold_param) + float(self.threshold_eps)
        return self.v_threshold_buffer

    def effective_input_weight(self) -> torch.Tensor:
        """Handle ``effective input weight`` for the ``LIF_neuron`` module."""
        return self.input_weight * self.input_mask

    def effective_recurrent_weight(self) -> torch.Tensor | None:
        """Handle ``effective recurrent weight`` for the ``LIF_neuron`` module."""
        if self.recurrent_weight is None:
            return None
        if self.recurrent_mask is None:
            return self.recurrent_weight
        return self.recurrent_weight * self.recurrent_mask

    def _reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Internal helper that reset state."""
        membrane = torch.zeros(batch_size, self.output_size, device=device, dtype=dtype)
        prev_spike = torch.zeros_like(membrane)
        return membrane, prev_spike

    def forward(self, input_sequence: torch.Tensor, *, return_traces: bool = False) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Run the forward pass."""
        self._last_layer_input = None
        if input_sequence.ndim != 3:
            raise ValueError(f'Expected input shape (B,T,C), got {tuple(input_sequence.shape)}')
        batch_size, time_steps, _ = input_sequence.shape
        weight = self.effective_input_weight()
        recurrent_weight = self.effective_recurrent_weight()
        alpha = self.effective_alpha().to(device=input_sequence.device, dtype=input_sequence.dtype)
        threshold = self.effective_threshold().to(device=input_sequence.device, dtype=input_sequence.dtype)
        membrane, prev_spike = self._reset_state(batch_size, input_sequence.device, input_sequence.dtype)
        record_raw_membrane = (not self.emit_spike) and (not self.reset_enabled)
        membrane_steps: list[torch.Tensor] | None = [] if return_traces else None
        layer_input_steps: list[torch.Tensor] | None = [] if return_traces else None
        spike_steps: list[torch.Tensor] = []
        input_current_sequence = torch.matmul(input_sequence, weight.t())

        for time_index in range(time_steps):
            current = input_current_sequence[:, time_index, :]
            if recurrent_weight is not None:
                current = current + prev_spike @ recurrent_weight.t()
            membrane_pre = alpha.unsqueeze(0) * membrane + current
            membrane_signal = membrane_pre - threshold.unsqueeze(0)
            if self.emit_spike:
                spike = surrogate_spike(membrane_signal)
            else:
                spike = torch.zeros_like(membrane_signal)
            if self.reset_enabled:
                if self.reset_mode == 'soft_reset':
                    membrane = membrane_pre - threshold.unsqueeze(0) * spike
                else:
                    membrane = membrane_pre * (1.0 - spike)
            else:
                membrane = membrane_pre
            if return_traces and membrane_steps is not None and layer_input_steps is not None:
                layer_input_steps.append(current)
                membrane_steps.append(membrane_pre if record_raw_membrane else membrane_signal)
            spike_steps.append(spike)
            prev_spike = spike

        membrane_seq = torch.stack(membrane_steps, dim=1) if membrane_steps is not None else None
        self._last_layer_input = torch.stack(layer_input_steps, dim=1) if layer_input_steps is not None else None
        spike_seq = torch.stack(spike_steps, dim=1)
        return membrane_seq, spike_seq

    def filter_stats_vectors(self) -> dict[str, torch.Tensor]:
        """Handle ``filter stats vectors`` for the ``LIF_neuron`` module."""
        return {'alpha': self.effective_alpha().detach()}



try:
    from src.neurons.spikingjelly_compat import install_spikingjelly_contract as _install_spikingjelly_contract
    _install_spikingjelly_contract(LIFLayer)
except Exception:  # pragma: no cover - defensive import fallback
    pass

__all__ = ['LIFLayer']
