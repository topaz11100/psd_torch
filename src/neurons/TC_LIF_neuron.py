"""TC-LIF layer aligned with the released ``TCLIFNode`` equations."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from src.neurons._common import sequence_state_dtype, surrogate_spike, to_sequence_state_dtype
from src.neurons._compile import compile_callable, disable_compiled_runtime
from src.neurons._origin_imports import load_tc_lif_module


class _OriginCompatibleSurrogate:
    def __call__(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return surrogate_spike(x)


def _tc_lif_sequence_no_trace(
    projected: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    prev: torch.Tensor,
    recurrent_weight: torch.Tensor | None,
    decay0: torch.Tensor,
    decay1: torch.Tensor,
    threshold: torch.Tensor,
    gamma: torch.Tensor,
    emit_spike: bool,
    reset_enabled: bool,
) -> torch.Tensor:
    batch_size, time_steps, output_size = projected.shape
    spike_seq = projected.new_empty((batch_size, time_steps, output_size))
    for i in range(time_steps):
        current = projected[:, i, :]
        if recurrent_weight is not None:
            current = current + prev @ recurrent_weight.t()
        v1_pre = v1 - decay0 * v2 + current
        v2_pre = v2 + decay1 * v1_pre
        signal = v2_pre - threshold
        raw = surrogate_spike(signal)
        spike = raw if emit_spike else torch.zeros_like(raw)
        if reset_enabled:
            v1 = v1_pre - gamma * raw
            v2 = v2_pre - threshold * raw
        else:
            v1 = v1_pre
            v2 = v2_pre
        spike_seq[:, i, :] = spike
        prev = spike
    return spike_seq


def _tc_lif_sequence_with_trace(
    projected: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    prev: torch.Tensor,
    recurrent_weight: torch.Tensor | None,
    decay0: torch.Tensor,
    decay1: torch.Tensor,
    threshold: torch.Tensor,
    gamma: torch.Tensor,
    emit_spike: bool,
    reset_enabled: bool,
    record_raw: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, time_steps, output_size = projected.shape
    spike_seq = projected.new_empty((batch_size, time_steps, output_size))
    mem_seq = projected.new_empty((batch_size, time_steps, output_size))
    inp_seq = projected.new_empty((batch_size, time_steps, output_size))
    for i in range(time_steps):
        current = projected[:, i, :]
        if recurrent_weight is not None:
            current = current + prev @ recurrent_weight.t()
        v1_pre = v1 - decay0 * v2 + current
        v2_pre = v2 + decay1 * v1_pre
        signal = v2_pre - threshold
        raw = surrogate_spike(signal)
        spike = raw if emit_spike else torch.zeros_like(raw)
        if reset_enabled:
            v1 = v1_pre - gamma * raw
            v2 = v2_pre - threshold * raw
        else:
            v1 = v1_pre
            v2 = v2_pre
        inp_seq[:, i, :] = current
        mem_seq[:, i, :] = v2_pre if record_raw else signal
        spike_seq[:, i, :] = spike
        prev = spike
    return mem_seq, spike_seq, inp_seq


class TCLIFLayer(nn.Module):
    compile_granularity = 'sequence'

    def __init__(self, input_size: int, output_size: int, *, recurrent: bool = False, v_threshold: float = 1.0, emit_spike: bool = True, reset_enabled: bool = True, gamma: float = 0.5) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.recurrent = bool(recurrent)
        self.v_threshold = float(v_threshold)
        self.emit_spike = bool(emit_spike)
        self.reset_enabled = bool(reset_enabled)
        self.gamma = float(gamma)
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
        origin = load_tc_lif_module()
        self.node = origin.TCLIFNode(v_threshold=self.v_threshold, surrogate_function=_OriginCompatibleSurrogate(), gamma=self.gamma, hard_reset=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.input_weight, a=math.sqrt(5.0))
        if self.recurrent_weight is not None:
            nn.init.orthogonal_(self.recurrent_weight)
        if hasattr(self.node, 'decay_factor'):
            nn.init.constant_(self.node.decay_factor, 0.0)

    def enable_compiled_forward(self, **compile_kwargs: Any) -> tuple[bool, str]:
        no_trace, no_applied, no_policy = compile_callable(_tc_lif_sequence_no_trace, compile_kwargs=compile_kwargs, label='tc_lif_sequence_no_trace')
        with_trace, trace_applied, trace_policy = compile_callable(_tc_lif_sequence_with_trace, compile_kwargs=compile_kwargs, label='tc_lif_sequence_with_trace')
        if no_applied:
            self._compiled_sequence_no_trace = no_trace
        if trace_applied:
            self._compiled_sequence_with_trace = with_trace
        if no_applied or trace_applied:
            self._compiled_sequence_policy = f'no_trace={no_policy};with_trace={trace_policy}'
            self._sequence_compiled_runtime_disabled = False
            self._sequence_compiled_runtime_error = None
        return bool(no_applied or trace_applied), 'sequence_compile[' + f'no_trace={no_policy};with_trace={trace_policy}' + ']'

    def effective_input_weight(self) -> torch.Tensor:
        return self.input_weight

    def _decay_pair(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        decay = torch.sigmoid(self.node.decay_factor).to(device=device, dtype=dtype).reshape(-1)
        return decay[0], decay[1]

    def _step_impl(self, current: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor, prev_spike: torch.Tensor, recurrent_weight: torch.Tensor | None, decay0: torch.Tensor, decay1: torch.Tensor, threshold: torch.Tensor, gamma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Eager single-step reference retained for equation-compliance tests."""
        if recurrent_weight is not None:
            current = current + prev_spike @ recurrent_weight.t()
        v1_pre = v1 - decay0 * v2 + current
        v2_pre = v2 + decay1 * v1_pre
        signal = v2_pre - threshold
        raw = surrogate_spike(signal)
        spike = raw if self.emit_spike else torch.zeros_like(raw)
        if self.reset_enabled:
            v1_next = v1_pre - gamma * raw
            v2_next = v2_pre - threshold * raw
        else:
            v1_next = v1_pre
            v2_next = v2_pre
        return v1_next, v2_next, spike, current, v2_pre, signal

    def _run_sequence(self, projected: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor, prev: torch.Tensor, decay0: torch.Tensor, decay1: torch.Tensor, threshold: torch.Tensor, gamma: torch.Tensor, *, return_traces: bool, record_raw: bool) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        if return_traces:
            fn = self._compiled_sequence_with_trace
            if fn is not None and not bool(self._sequence_compiled_runtime_disabled):
                try:
                    mem_seq, spike_seq, inp_seq = fn(projected, v1, v2, prev, self.recurrent_weight, decay0, decay1, threshold, gamma, self.emit_spike, self.reset_enabled, record_raw)
                    return mem_seq, spike_seq, inp_seq
                except Exception as exc:
                    disable_compiled_runtime(self, label='sequence', exc=exc)
            mem_seq, spike_seq, inp_seq = _tc_lif_sequence_with_trace(projected, v1, v2, prev, self.recurrent_weight, decay0, decay1, threshold, gamma, self.emit_spike, self.reset_enabled, record_raw)
            return mem_seq, spike_seq, inp_seq
        fn = self._compiled_sequence_no_trace
        if fn is not None and not bool(self._sequence_compiled_runtime_disabled):
            try:
                return None, fn(projected, v1, v2, prev, self.recurrent_weight, decay0, decay1, threshold, gamma, self.emit_spike, self.reset_enabled), None
            except Exception as exc:
                disable_compiled_runtime(self, label='sequence', exc=exc)
        return None, _tc_lif_sequence_no_trace(projected, v1, v2, prev, self.recurrent_weight, decay0, decay1, threshold, gamma, self.emit_spike, self.reset_enabled), None

    def forward(self, input_sequence: torch.Tensor, *, return_traces: bool = False) -> tuple[torch.Tensor | None, torch.Tensor]:
        self._last_layer_input = None
        if input_sequence.ndim != 3:
            raise ValueError(f'Expected shape (B,T,C), got {tuple(input_sequence.shape)}')
        batch_size, _time_steps, _ = input_sequence.shape
        device = input_sequence.device
        dtype = sequence_state_dtype(input_sequence)
        v1 = torch.zeros(batch_size, self.output_size, device=device, dtype=dtype)
        v2 = torch.zeros_like(v1)
        prev = torch.zeros_like(v1)
        decay0, decay1 = self._decay_pair(device, dtype)
        threshold = torch.as_tensor(self.v_threshold, device=device, dtype=dtype)
        gamma = torch.as_tensor(self.gamma, device=device, dtype=dtype)
        record_raw = (not self.emit_spike) and (not self.reset_enabled)
        projected = to_sequence_state_dtype(input_sequence @ self.input_weight.t(), input_sequence)
        mem_seq, spike_seq, inp_seq = self._run_sequence(projected, v1, v2, prev, decay0, decay1, threshold, gamma, return_traces=return_traces, record_raw=record_raw)
        self._last_layer_input = inp_seq.contiguous() if inp_seq is not None else None
        return (mem_seq.contiguous() if mem_seq is not None else None), spike_seq.contiguous()

    def filter_stats_vectors(self) -> dict[str, torch.Tensor]:
        device = next(self.parameters(), torch.empty((), device='cpu')).device
        return {'v_threshold': torch.as_tensor([float(self.v_threshold)], device=device, dtype=torch.float32)}


try:
    from src.neurons.spikingjelly_compat import install_spikingjelly_contract as _install_spikingjelly_contract
    _install_spikingjelly_contract(TCLIFLayer)
except Exception:
    pass

__all__ = ['TCLIFLayer']
