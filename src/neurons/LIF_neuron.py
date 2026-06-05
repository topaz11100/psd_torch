"""Project-standard vanilla discrete-time LIF layer."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.neurons._common import logit, sequence_state_dtype, surrogate_spike, to_sequence_state_dtype, trim_open_interval
from src.neurons._compile import compile_callable, disable_compiled_runtime


def _positive_threshold_init(v_threshold: float, size: int, *, eps: float) -> torch.Tensor:
    value = max(float(v_threshold) - float(eps), float(eps))
    raw = math.log(math.expm1(value))
    return torch.full((int(size),), float(raw), dtype=torch.float32)


def _lif_sequence_no_trace(
    input_current_sequence: torch.Tensor,
    membrane: torch.Tensor,
    prev_spike: torch.Tensor,
    recurrent_weight: torch.Tensor | None,
    alpha_view: torch.Tensor,
    threshold_view: torch.Tensor,
    emit_spike: bool,
    reset_enabled: bool,
    hard_reset: bool,
) -> torch.Tensor:
    batch_size, time_steps, output_size = input_current_sequence.shape
    spike_seq = input_current_sequence.new_empty((batch_size, time_steps, output_size))
    for time_index in range(time_steps):
        current = input_current_sequence[:, time_index, :]
        if recurrent_weight is not None:
            current = current + prev_spike @ recurrent_weight.t()
        membrane_pre = alpha_view * membrane + current
        membrane_signal = membrane_pre - threshold_view
        spike = surrogate_spike(membrane_signal) if emit_spike else torch.zeros_like(membrane_signal)
        if reset_enabled:
            membrane = membrane_pre * (1.0 - spike) if hard_reset else membrane_pre - threshold_view * spike
        else:
            membrane = membrane_pre
        spike_seq[:, time_index, :] = spike
        prev_spike = spike
    return spike_seq


def _lif_sequence_with_trace(
    input_current_sequence: torch.Tensor,
    membrane: torch.Tensor,
    prev_spike: torch.Tensor,
    recurrent_weight: torch.Tensor | None,
    alpha_view: torch.Tensor,
    threshold_view: torch.Tensor,
    emit_spike: bool,
    reset_enabled: bool,
    hard_reset: bool,
    record_raw: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, time_steps, output_size = input_current_sequence.shape
    spike_seq = input_current_sequence.new_empty((batch_size, time_steps, output_size))
    membrane_seq = input_current_sequence.new_empty((batch_size, time_steps, output_size))
    layer_input_seq = input_current_sequence.new_empty((batch_size, time_steps, output_size))
    for time_index in range(time_steps):
        current = input_current_sequence[:, time_index, :]
        if recurrent_weight is not None:
            current = current + prev_spike @ recurrent_weight.t()
        membrane_pre = alpha_view * membrane + current
        membrane_signal = membrane_pre - threshold_view
        spike = surrogate_spike(membrane_signal) if emit_spike else torch.zeros_like(membrane_signal)
        if reset_enabled:
            membrane = membrane_pre * (1.0 - spike) if hard_reset else membrane_pre - threshold_view * spike
        else:
            membrane = membrane_pre
        layer_input_seq[:, time_index, :] = current
        membrane_seq[:, time_index, :] = membrane_pre if record_raw else membrane_signal
        spike_seq[:, time_index, :] = spike
        prev_spike = spike
    return membrane_seq, spike_seq, layer_input_seq


class LIFLayer(nn.Module):
    """Dense discrete-time Leaky Integrate-and-Fire layer with compiled sequence regions."""

    compile_granularity = 'sequence'

    def __init__(self, input_size: int, output_size: int, *, recurrent: bool = False, v_threshold: float = 1.0, trainable_threshold: bool = False, reset_mode: str = 'soft_reset', alpha_bounds: tuple[torch.Tensor, torch.Tensor] | None = None, input_mask: torch.Tensor | None = None, recurrent_mask: torch.Tensor | None = None, emit_spike: bool = True, reset_enabled: bool = True, filter_value: float | None = None) -> None:
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
        self.filter_fixed_value = None if filter_value is None else float(filter_value)
        self.threshold_eps = 1e-6
        self._compiled_sequence_no_trace = None
        self._compiled_sequence_with_trace = None
        self._compiled_sequence_policy = 'eager'
        self._sequence_compiled_runtime_disabled = False
        self._sequence_compiled_runtime_error = None
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
        self.register_buffer('recurrent_mask', None if recurrent_mask is None else recurrent_mask.to(dtype=torch.float32))
        if alpha_bounds is None:
            lower = torch.zeros(self.output_size, dtype=torch.float32)
            upper = torch.ones(self.output_size, dtype=torch.float32)
        else:
            lower, upper = alpha_bounds
            lower = lower.detach().clone().to(dtype=torch.float32)
            upper = upper.detach().clone().to(dtype=torch.float32)
        self.register_buffer('alpha_lower', lower)
        self.register_buffer('alpha_upper', upper)
        if self.filter_fixed_value is not None:
            fixed = torch.full((self.output_size,), float(self.filter_fixed_value), dtype=torch.float32)
            if torch.any(fixed < lower) or torch.any(fixed > upper):
                raise ValueError('Fixed LIF filter alpha is outside alpha clip/bound range.')
        self.alpha_raw = nn.Parameter(torch.empty(self.output_size))
        threshold_init = torch.full((self.output_size,), float(v_threshold), dtype=torch.float32)
        if self.trainable_threshold:
            self.v_threshold_param = nn.Parameter(_positive_threshold_init(v_threshold, self.output_size, eps=self.threshold_eps))
        else:
            self.register_buffer('v_threshold_buffer', threshold_init)
            self.register_parameter('v_threshold_param', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.input_weight, a=math.sqrt(5.0))
        if self.recurrent_weight is not None:
            nn.init.orthogonal_(self.recurrent_weight)
        init_alpha = torch.empty_like(self.alpha_lower)
        if self.filter_fixed_value is None:
            for index in range(init_alpha.numel()):
                left, right = trim_open_interval(float(self.alpha_lower[index]), float(self.alpha_upper[index]))
                init_alpha[index] = 0.5 * (left + right) if right <= left else float(torch.empty(1).uniform_(left, right).item())
        else:
            init_alpha.fill_(float(self.filter_fixed_value))
        with torch.no_grad():
            alpha_span = torch.clamp(self.alpha_upper - self.alpha_lower, min=1e-6)
            alpha01 = torch.clamp((init_alpha - self.alpha_lower) / alpha_span, min=1e-6, max=1 - 1e-6)
            self.alpha_raw.copy_(logit(alpha01))
        self.alpha_raw.requires_grad_(self.filter_fixed_value is None)

    def enable_compiled_forward(self, **compile_kwargs: Any) -> tuple[bool, str]:
        no_trace, no_applied, no_policy = compile_callable(_lif_sequence_no_trace, compile_kwargs=compile_kwargs, label='lif_sequence_no_trace')
        with_trace, trace_applied, trace_policy = compile_callable(_lif_sequence_with_trace, compile_kwargs=compile_kwargs, label='lif_sequence_with_trace')
        if no_applied:
            self._compiled_sequence_no_trace = no_trace
        if trace_applied:
            self._compiled_sequence_with_trace = with_trace
        if no_applied or trace_applied:
            self._compiled_sequence_policy = f'no_trace={no_policy};with_trace={trace_policy}'
            self._sequence_compiled_runtime_disabled = False
            self._sequence_compiled_runtime_error = None
        return bool(no_applied or trace_applied), 'sequence_compile[' + f'no_trace={no_policy};with_trace={trace_policy}' + ']'

    def effective_alpha(self) -> torch.Tensor:
        sigma = torch.sigmoid(self.alpha_raw)
        return self.alpha_lower + (self.alpha_upper - self.alpha_lower) * sigma

    def effective_threshold(self) -> torch.Tensor:
        if self.v_threshold_param is not None:
            return F.softplus(self.v_threshold_param) + float(self.threshold_eps)
        return self.v_threshold_buffer

    def effective_input_weight(self) -> torch.Tensor:
        return self.input_weight * self.input_mask

    def effective_recurrent_weight(self) -> torch.Tensor | None:
        if self.recurrent_weight is None:
            return None
        return self.recurrent_weight if self.recurrent_mask is None else self.recurrent_weight * self.recurrent_mask

    def _reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        membrane = torch.zeros(batch_size, self.output_size, device=device, dtype=dtype)
        return membrane, torch.zeros_like(membrane)

    def _run_sequence(
        self,
        input_current_sequence: torch.Tensor,
        membrane: torch.Tensor,
        prev_spike: torch.Tensor,
        recurrent_weight: torch.Tensor | None,
        alpha_view: torch.Tensor,
        threshold_view: torch.Tensor,
        *,
        return_traces: bool,
        record_raw: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        hard_reset = self.reset_mode == 'hard_reset'
        if return_traces:
            fn = self._compiled_sequence_with_trace
            if fn is not None and not bool(self._sequence_compiled_runtime_disabled):
                try:
                    mem_seq, spike_seq, inp_seq = fn(input_current_sequence, membrane, prev_spike, recurrent_weight, alpha_view, threshold_view, self.emit_spike, self.reset_enabled, hard_reset, record_raw)
                    return mem_seq, spike_seq, inp_seq
                except Exception as exc:
                    disable_compiled_runtime(self, label='sequence', exc=exc)
            mem_seq, spike_seq, inp_seq = _lif_sequence_with_trace(input_current_sequence, membrane, prev_spike, recurrent_weight, alpha_view, threshold_view, self.emit_spike, self.reset_enabled, hard_reset, record_raw)
            return mem_seq, spike_seq, inp_seq
        fn = self._compiled_sequence_no_trace
        if fn is not None and not bool(self._sequence_compiled_runtime_disabled):
            try:
                return None, fn(input_current_sequence, membrane, prev_spike, recurrent_weight, alpha_view, threshold_view, self.emit_spike, self.reset_enabled, hard_reset), None
            except Exception as exc:
                disable_compiled_runtime(self, label='sequence', exc=exc)
        return None, _lif_sequence_no_trace(input_current_sequence, membrane, prev_spike, recurrent_weight, alpha_view, threshold_view, self.emit_spike, self.reset_enabled, hard_reset), None

    def forward(self, input_sequence: torch.Tensor, *, return_traces: bool = False) -> tuple[torch.Tensor | None, torch.Tensor]:
        self._last_layer_input = None
        if input_sequence.ndim != 3:
            raise ValueError(f'Expected input shape (B,T,C), got {tuple(input_sequence.shape)}')
        batch_size, _time_steps, _ = input_sequence.shape
        device = input_sequence.device
        dtype = sequence_state_dtype(input_sequence)
        weight = self.effective_input_weight()
        recurrent_weight = self.effective_recurrent_weight()
        alpha_view = self.effective_alpha().to(device=device, dtype=dtype).unsqueeze(0)
        threshold_view = self.effective_threshold().to(device=device, dtype=dtype).unsqueeze(0)
        membrane, prev_spike = self._reset_state(batch_size, device, dtype)
        record_raw = (not self.emit_spike) and (not self.reset_enabled)
        input_current_sequence = to_sequence_state_dtype(torch.matmul(input_sequence, weight.t()), input_sequence)
        mem_seq, spike_seq, inp_seq = self._run_sequence(input_current_sequence, membrane, prev_spike, recurrent_weight, alpha_view, threshold_view, return_traces=return_traces, record_raw=record_raw)
        self._last_layer_input = inp_seq.contiguous() if inp_seq is not None else None
        return (mem_seq.contiguous() if mem_seq is not None else None), spike_seq.contiguous()

    def filter_stats_vectors(self) -> dict[str, torch.Tensor]:
        return {'alpha': self.effective_alpha().detach(), 'v_threshold': self.effective_threshold().detach()}


try:
    from src.neurons.spikingjelly_compat import install_spikingjelly_contract as _install_spikingjelly_contract
    _install_spikingjelly_contract(LIFLayer)
except Exception:
    pass

__all__ = ['LIFLayer', '_positive_threshold_init']
