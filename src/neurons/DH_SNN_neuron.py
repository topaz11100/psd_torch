"""DH-SNN origin-code aligned layer with compileable tensor sequence."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.neurons._common import sequence_state_dtype, surrogate_spike, to_sequence_state_dtype
from src.neurons._compile import compile_callable, disable_compiled_runtime
from src.neurons._origin_imports import load_dh_snn_modules


def _dh_recurrent_drive(reset_spike: torch.Tensor, recurrent_weight: torch.Tensor | None, output_size: int, branch: int) -> torch.Tensor:
    if recurrent_weight is None:
        return reset_spike.new_zeros((reset_spike.shape[0], int(output_size), int(branch)))
    return F.linear(reset_spike.to(device=recurrent_weight.device, dtype=recurrent_weight.dtype), recurrent_weight, None).reshape(-1, int(output_size), int(branch)).to(device=reset_spike.device, dtype=reset_spike.dtype)


def _dh_snn_sequence_no_trace(
    input_dense_sequence: torch.Tensor,
    mem: torch.Tensor,
    d_input: torch.Tensor,
    reset_spike: torch.Tensor,
    recurrent_weight: torch.Tensor | None,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    threshold: torch.Tensor,
    r_m: torch.Tensor,
    output_size: int,
    branch: int,
    recurrent: bool,
    emit_spike: bool,
    reset_enabled: bool,
) -> torch.Tensor:
    batch_size, time_steps, _output_size, _branch = input_dense_sequence.shape
    spikes = input_dense_sequence.new_empty((batch_size, time_steps, int(output_size)))
    for i in range(time_steps):
        recurrent_drive = _dh_recurrent_drive(reset_spike, recurrent_weight, output_size, branch) if recurrent else input_dense_sequence.new_zeros((batch_size, int(output_size), int(branch)))
        dense_out = input_dense_sequence[:, i, :, :] + recurrent_drive if recurrent else input_dense_sequence[:, i, :, :]
        d_input = beta.unsqueeze(0) * d_input + (1.0 - beta).unsqueeze(0) * dense_out
        l_input = d_input.sum(dim=2)
        mem_pre = mem * alpha.unsqueeze(0) + (1.0 - alpha).unsqueeze(0) * r_m * l_input - threshold * reset_spike
        raw = surrogate_spike(mem_pre - threshold)
        out_spike = raw if emit_spike else torch.zeros_like(raw)
        reset_spike = raw if reset_enabled else torch.zeros_like(raw)
        mem = mem_pre
        spikes[:, i, :] = out_spike
    return spikes


def _dh_snn_sequence_with_trace(
    input_dense_sequence: torch.Tensor,
    mem: torch.Tensor,
    d_input: torch.Tensor,
    reset_spike: torch.Tensor,
    recurrent_weight: torch.Tensor | None,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    threshold: torch.Tensor,
    r_m: torch.Tensor,
    output_size: int,
    branch: int,
    recurrent: bool,
    emit_spike: bool,
    reset_enabled: bool,
    record_raw: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, time_steps, _output_size, _branch = input_dense_sequence.shape
    spikes = input_dense_sequence.new_empty((batch_size, time_steps, int(output_size)))
    mem_seq = input_dense_sequence.new_empty((batch_size, time_steps, int(output_size)))
    inp_seq = input_dense_sequence.new_empty((batch_size, time_steps, int(output_size)))
    for i in range(time_steps):
        recurrent_drive = _dh_recurrent_drive(reset_spike, recurrent_weight, output_size, branch) if recurrent else input_dense_sequence.new_zeros((batch_size, int(output_size), int(branch)))
        dense_out = input_dense_sequence[:, i, :, :] + recurrent_drive if recurrent else input_dense_sequence[:, i, :, :]
        d_input = beta.unsqueeze(0) * d_input + (1.0 - beta).unsqueeze(0) * dense_out
        l_input = d_input.sum(dim=2)
        mem_pre = mem * alpha.unsqueeze(0) + (1.0 - alpha).unsqueeze(0) * r_m * l_input - threshold * reset_spike
        signal = mem_pre - threshold
        raw = surrogate_spike(signal)
        out_spike = raw if emit_spike else torch.zeros_like(raw)
        reset_spike = raw if reset_enabled else torch.zeros_like(raw)
        mem = mem_pre
        inp_seq[:, i, :] = l_input
        mem_seq[:, i, :] = mem_pre if record_raw else signal
        spikes[:, i, :] = out_spike
    return mem_seq, spikes, inp_seq


class DHSNNLayer(nn.Module):
    compile_granularity = 'sequence'

    def __init__(self, input_size: int, output_size: int, *, recurrent: bool = False, branch: int = 4, v_threshold: float = 1.0, emit_spike: bool = True, reset_enabled: bool = True) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.recurrent = bool(recurrent)
        self.branch = int(branch)
        self.v_threshold = float(v_threshold)
        self.emit_spike = bool(emit_spike)
        self.reset_enabled = bool(reset_enabled)
        self._compiled_sequence_no_trace = None
        self._compiled_sequence_with_trace = None
        self._compiled_sequence_policy = 'eager'
        self._sequence_compiled_runtime_disabled = False
        self._sequence_compiled_runtime_error = None
        _, dense_mod, rnn_mod = load_dh_snn_modules()
        self._origin_r_m = float(getattr(rnn_mod if self.recurrent else dense_mod, 'R_m', 1.0))
        if self.recurrent:
            self.layer = rnn_mod.spike_rnn_test_denri_wotanh_R(self.input_size, self.output_size, vth=self.v_threshold, branch=self.branch, device='cpu', bias=True)
        else:
            self.layer = dense_mod.spike_dense_test_denri_wotanh_R(self.input_size, self.output_size, vth=self.v_threshold, branch=self.branch, device='cpu', bias=True)

    def enable_compiled_forward(self, **compile_kwargs: Any) -> tuple[bool, str]:
        no_trace, no_applied, no_policy = compile_callable(_dh_snn_sequence_no_trace, compile_kwargs=compile_kwargs, label='dh_snn_sequence_no_trace')
        with_trace, trace_applied, trace_policy = compile_callable(_dh_snn_sequence_with_trace, compile_kwargs=compile_kwargs, label='dh_snn_sequence_with_trace')
        if no_applied:
            self._compiled_sequence_no_trace = no_trace
        if trace_applied:
            self._compiled_sequence_with_trace = with_trace
        if no_applied or trace_applied:
            self._compiled_sequence_policy = f'no_trace={no_policy};with_trace={trace_policy}'
            self._sequence_compiled_runtime_disabled = False
            self._sequence_compiled_runtime_error = None
        return bool(no_applied or trace_applied), 'sequence_compile[' + f'no_trace={no_policy};with_trace={trace_policy}' + ']'

    def _masked_weight_bias(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        weight = self.layer.dense.weight
        bias = self.layer.dense.bias
        mask = getattr(self.layer, 'mask', None)
        if mask is not None:
            weight = weight * mask.to(device=weight.device, dtype=weight.dtype)
        return weight, bias

    def _project_input_sequence(self, input_sequence: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, _ = input_sequence.shape
        weight, bias = self._masked_weight_bias()
        in_weight = weight[:, :self.input_size]
        flat = input_sequence.reshape(batch_size * time_steps, self.input_size).to(device=weight.device, dtype=weight.dtype)
        projected = F.linear(flat, in_weight, bias).reshape(batch_size, time_steps, self.output_size, self.branch)
        return projected.to(device=input_sequence.device, dtype=input_sequence.dtype)

    def _effective_recurrent_weight(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if not self.recurrent:
            return None
        weight, _bias = self._masked_weight_bias()
        rec_weight = weight[:, self.input_size:self.input_size + self.output_size]
        return rec_weight.to(device=device, dtype=dtype)

    def _run_sequence(self, input_dense_sequence: torch.Tensor, mem: torch.Tensor, d_input: torch.Tensor, reset_spike: torch.Tensor, recurrent_weight: torch.Tensor | None, beta: torch.Tensor, alpha: torch.Tensor, threshold: torch.Tensor, r_m: torch.Tensor, *, return_traces: bool, record_raw: bool) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        if return_traces:
            fn = self._compiled_sequence_with_trace
            if fn is not None and not bool(self._sequence_compiled_runtime_disabled):
                try:
                    mem_seq, spike_seq, inp_seq = fn(input_dense_sequence, mem, d_input, reset_spike, recurrent_weight, beta, alpha, threshold, r_m, self.output_size, self.branch, self.recurrent, self.emit_spike, self.reset_enabled, record_raw)
                    return mem_seq, spike_seq, inp_seq
                except Exception as exc:
                    disable_compiled_runtime(self, label='sequence', exc=exc)
            mem_seq, spike_seq, inp_seq = _dh_snn_sequence_with_trace(input_dense_sequence, mem, d_input, reset_spike, recurrent_weight, beta, alpha, threshold, r_m, self.output_size, self.branch, self.recurrent, self.emit_spike, self.reset_enabled, record_raw)
            return mem_seq, spike_seq, inp_seq
        fn = self._compiled_sequence_no_trace
        if fn is not None and not bool(self._sequence_compiled_runtime_disabled):
            try:
                return None, fn(input_dense_sequence, mem, d_input, reset_spike, recurrent_weight, beta, alpha, threshold, r_m, self.output_size, self.branch, self.recurrent, self.emit_spike, self.reset_enabled), None
            except Exception as exc:
                disable_compiled_runtime(self, label='sequence', exc=exc)
        return None, _dh_snn_sequence_no_trace(input_dense_sequence, mem, d_input, reset_spike, recurrent_weight, beta, alpha, threshold, r_m, self.output_size, self.branch, self.recurrent, self.emit_spike, self.reset_enabled), None

    def forward(self, input_sequence: torch.Tensor, *, return_traces: bool = False) -> tuple[torch.Tensor | None, torch.Tensor]:
        self._last_layer_input = None
        if input_sequence.ndim != 3:
            raise ValueError(f'Expected shape (B,T,C), got {tuple(input_sequence.shape)}')
        batch_size, _time_steps, _ = input_sequence.shape
        device = input_sequence.device
        dtype = sequence_state_dtype(input_sequence)
        mem = torch.zeros(batch_size, self.output_size, device=device, dtype=dtype)
        d_input = torch.zeros(batch_size, self.output_size, self.branch, device=device, dtype=dtype)
        reset_spike = torch.zeros(batch_size, self.output_size, device=device, dtype=dtype)
        beta = torch.sigmoid(self.layer.tau_n).to(device=device, dtype=dtype)
        alpha = torch.sigmoid(self.layer.tau_m).to(device=device, dtype=dtype)
        threshold = torch.as_tensor(self.v_threshold, device=device, dtype=dtype)
        r_m = torch.as_tensor(self._origin_r_m, device=device, dtype=dtype)
        record_raw = (not self.emit_spike) and (not self.reset_enabled)
        input_dense_sequence = to_sequence_state_dtype(self._project_input_sequence(input_sequence), input_sequence)
        recurrent_weight = self._effective_recurrent_weight(device=device, dtype=dtype)
        mem_seq, spike_seq, inp_seq = self._run_sequence(input_dense_sequence, mem, d_input, reset_spike, recurrent_weight, beta, alpha, threshold, r_m, return_traces=return_traces, record_raw=record_raw)
        self._last_layer_input = inp_seq.contiguous() if inp_seq is not None else None
        return (mem_seq.contiguous() if mem_seq is not None else None), spike_seq.contiguous()

    def filter_stats_vectors(self) -> dict[str, torch.Tensor]:
        device = next(self.parameters(), torch.empty((), device='cpu')).device
        return {'v_threshold': torch.as_tensor([float(self.v_threshold)], device=device, dtype=torch.float32)}


try:
    from src.neurons.spikingjelly_compat import install_spikingjelly_contract as _install_spikingjelly_contract
    _install_spikingjelly_contract(DHSNNLayer)
except Exception:
    pass

__all__ = ['DHSNNLayer']
