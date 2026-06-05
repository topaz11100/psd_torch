from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.neurons._common import sequence_state_dtype, surrogate_spike, to_sequence_state_dtype
from src.neurons._compile import compile_callable, disable_compiled_runtime
from src.neurons.LIF_neuron import _positive_threshold_init


def _if_sequence_no_trace(
    input_current_sequence: torch.Tensor,
    membrane: torch.Tensor,
    prev_spike: torch.Tensor,
    recurrent_weight: torch.Tensor | None,
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
        membrane_pre = membrane + current
        signal = membrane_pre - threshold_view
        spike = surrogate_spike(signal) if emit_spike else torch.zeros_like(signal)
        if reset_enabled:
            membrane = membrane_pre * (1.0 - spike) if hard_reset else membrane_pre - threshold_view * spike
        else:
            membrane = membrane_pre
        spike_seq[:, time_index, :] = spike
        prev_spike = spike
    return spike_seq


def _if_sequence_with_trace(
    input_current_sequence: torch.Tensor,
    membrane: torch.Tensor,
    prev_spike: torch.Tensor,
    recurrent_weight: torch.Tensor | None,
    threshold_view: torch.Tensor,
    emit_spike: bool,
    reset_enabled: bool,
    hard_reset: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, time_steps, output_size = input_current_sequence.shape
    spike_seq = input_current_sequence.new_empty((batch_size, time_steps, output_size))
    mem_seq = input_current_sequence.new_empty((batch_size, time_steps, output_size))
    inp_seq = input_current_sequence.new_empty((batch_size, time_steps, output_size))
    for time_index in range(time_steps):
        current = input_current_sequence[:, time_index, :]
        if recurrent_weight is not None:
            current = current + prev_spike @ recurrent_weight.t()
        membrane_pre = membrane + current
        signal = membrane_pre - threshold_view
        spike = surrogate_spike(signal) if emit_spike else torch.zeros_like(signal)
        if reset_enabled:
            membrane = membrane_pre * (1.0 - spike) if hard_reset else membrane_pre - threshold_view * spike
        else:
            membrane = membrane_pre
        inp_seq[:, time_index, :] = current
        mem_seq[:, time_index, :] = signal
        spike_seq[:, time_index, :] = spike
        prev_spike = spike
    return mem_seq, spike_seq, inp_seq


class IFLayer(nn.Module):
    compile_granularity = 'sequence'

    def __init__(self, input_size: int, output_size: int, *, recurrent: bool = False, v_threshold: float = 1.0, trainable_threshold: bool = False, reset_mode: str = 'soft_reset', input_mask: torch.Tensor | None = None, recurrent_mask: torch.Tensor | None = None, emit_spike: bool = True, reset_enabled: bool = True) -> None:
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

    def enable_compiled_forward(self, **compile_kwargs: Any) -> tuple[bool, str]:
        no_trace, no_applied, no_policy = compile_callable(_if_sequence_no_trace, compile_kwargs=compile_kwargs, label='if_sequence_no_trace')
        with_trace, trace_applied, trace_policy = compile_callable(_if_sequence_with_trace, compile_kwargs=compile_kwargs, label='if_sequence_with_trace')
        if no_applied:
            self._compiled_sequence_no_trace = no_trace
        if trace_applied:
            self._compiled_sequence_with_trace = with_trace
        if no_applied or trace_applied:
            self._compiled_sequence_policy = f'no_trace={no_policy};with_trace={trace_policy}'
            self._sequence_compiled_runtime_disabled = False
            self._sequence_compiled_runtime_error = None
        return bool(no_applied or trace_applied), 'sequence_compile[' + f'no_trace={no_policy};with_trace={trace_policy}' + ']'

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
        threshold_view: torch.Tensor,
        *,
        return_traces: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        hard_reset = self.reset_mode == 'hard_reset'
        if return_traces:
            fn = self._compiled_sequence_with_trace
            if fn is not None and not bool(self._sequence_compiled_runtime_disabled):
                try:
                    mem_seq, spike_seq, inp_seq = fn(input_current_sequence, membrane, prev_spike, recurrent_weight, threshold_view, self.emit_spike, self.reset_enabled, hard_reset)
                    return mem_seq, spike_seq, inp_seq
                except Exception as exc:
                    disable_compiled_runtime(self, label='sequence', exc=exc)
            mem_seq, spike_seq, inp_seq = _if_sequence_with_trace(input_current_sequence, membrane, prev_spike, recurrent_weight, threshold_view, self.emit_spike, self.reset_enabled, hard_reset)
            return mem_seq, spike_seq, inp_seq
        fn = self._compiled_sequence_no_trace
        if fn is not None and not bool(self._sequence_compiled_runtime_disabled):
            try:
                return None, fn(input_current_sequence, membrane, prev_spike, recurrent_weight, threshold_view, self.emit_spike, self.reset_enabled, hard_reset), None
            except Exception as exc:
                disable_compiled_runtime(self, label='sequence', exc=exc)
        return None, _if_sequence_no_trace(input_current_sequence, membrane, prev_spike, recurrent_weight, threshold_view, self.emit_spike, self.reset_enabled, hard_reset), None

    def forward(self, input_sequence: torch.Tensor, *, return_traces: bool = False) -> tuple[torch.Tensor | None, torch.Tensor]:
        self._last_layer_input = None
        if input_sequence.ndim != 3:
            raise ValueError(f'Expected input shape (B,T,C), got {tuple(input_sequence.shape)}')
        batch_size, _time_steps, _ = input_sequence.shape
        device = input_sequence.device
        dtype = sequence_state_dtype(input_sequence)
        weight = self.effective_input_weight()
        recurrent_weight = self.effective_recurrent_weight()
        threshold = self.effective_threshold().to(device=device, dtype=dtype).unsqueeze(0)
        membrane, prev_spike = self._reset_state(batch_size, device, dtype)
        input_current_sequence = to_sequence_state_dtype(torch.matmul(input_sequence, weight.t()), input_sequence)
        mem_seq, spike_seq, inp_seq = self._run_sequence(input_current_sequence, membrane, prev_spike, recurrent_weight, threshold, return_traces=return_traces)
        self._last_layer_input = inp_seq.contiguous() if inp_seq is not None else None
        return (mem_seq.contiguous() if mem_seq is not None else None), spike_seq.contiguous()

    def filter_stats_vectors(self) -> dict[str, torch.Tensor]:
        return {'v_threshold': self.effective_threshold().detach()}


try:
    from src.neurons.spikingjelly_compat import install_spikingjelly_contract as _install_spikingjelly_contract
    _install_spikingjelly_contract(IFLayer)
except Exception:
    pass

__all__ = ['IFLayer']
