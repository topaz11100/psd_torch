"""Project-standard resonate-and-fire neuron layer."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.neurons._common import logit, sequence_state_dtype, surrogate_spike, to_sequence_state_dtype, trim_open_interval
from src.neurons._compile import compile_callable, disable_compiled_runtime
from src.neurons.LIF_neuron import _positive_threshold_init


def _rf_sequence_no_trace(
    input_current_sequence: torch.Tensor,
    x_post: torch.Tensor,
    y_post: torch.Tensor,
    prev_spike: torch.Tensor,
    recurrent_weight: torch.Tensor | None,
    threshold_view: torch.Tensor,
    rho_view: torch.Tensor,
    cos_view: torch.Tensor,
    sin_view: torch.Tensor,
    beta_x_view: torch.Tensor,
    beta_y_view: torch.Tensor,
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
        x_pre = rho_view * (cos_view * x_post - sin_view * y_post) + beta_x_view * current
        y_pre = rho_view * (sin_view * x_post + cos_view * y_post) + beta_y_view * current
        signal = x_pre - threshold_view
        spike = surrogate_spike(signal) if emit_spike else torch.zeros_like(signal)
        if reset_enabled:
            if hard_reset:
                keep = 1.0 - spike
                x_post = x_pre * keep
                y_post = y_pre * keep
            else:
                x_post = x_pre - threshold_view * spike
                y_post = y_pre
        else:
            x_post = x_pre
            y_post = y_pre
        spike_seq[:, time_index, :] = spike
        prev_spike = spike
    return spike_seq


def _rf_sequence_with_trace(
    input_current_sequence: torch.Tensor,
    x_post: torch.Tensor,
    y_post: torch.Tensor,
    prev_spike: torch.Tensor,
    recurrent_weight: torch.Tensor | None,
    threshold_view: torch.Tensor,
    rho_view: torch.Tensor,
    cos_view: torch.Tensor,
    sin_view: torch.Tensor,
    beta_x_view: torch.Tensor,
    beta_y_view: torch.Tensor,
    emit_spike: bool,
    reset_enabled: bool,
    hard_reset: bool,
    record_raw: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, time_steps, output_size = input_current_sequence.shape
    spike_seq = input_current_sequence.new_empty((batch_size, time_steps, output_size))
    mem_seq = input_current_sequence.new_empty((batch_size, time_steps, output_size))
    inp_seq = input_current_sequence.new_empty((batch_size, time_steps, output_size))
    for time_index in range(time_steps):
        current = input_current_sequence[:, time_index, :]
        if recurrent_weight is not None:
            current = current + prev_spike @ recurrent_weight.t()
        x_pre = rho_view * (cos_view * x_post - sin_view * y_post) + beta_x_view * current
        y_pre = rho_view * (sin_view * x_post + cos_view * y_post) + beta_y_view * current
        signal = x_pre - threshold_view
        spike = surrogate_spike(signal) if emit_spike else torch.zeros_like(signal)
        if reset_enabled:
            if hard_reset:
                keep = 1.0 - spike
                x_post = x_pre * keep
                y_post = y_pre * keep
            else:
                x_post = x_pre - threshold_view * spike
                y_post = y_pre
        else:
            x_post = x_pre
            y_post = y_pre
        inp_seq[:, time_index, :] = current
        mem_seq[:, time_index, :] = x_pre if record_raw else signal
        spike_seq[:, time_index, :] = spike
        prev_spike = spike
    return mem_seq, spike_seq, inp_seq


class RFLayer(nn.Module):
    """Dense discrete-time resonate-and-fire layer.

    The linear subthreshold core is defined directly in the discrete domain as

        z[t+1] = a z[t] + I[t+1],   a = rho * exp(j * phi),

    and implemented with real states ``x = Re(z)`` and ``y = Im(z)``.  ``rho`` is
    the per-sample pole radius and ``phi`` is the pole angle in radians/sample.
    ``pole_radius_constrained=True`` constrains ``rho`` to ``[0, pole_radius_max)``;
    ``False`` makes ``rho`` a positive softplus parameter and allows controlled
    finite-horizon amplification when the trained radius exceeds one.
    """

    compile_granularity = 'sequence'

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        recurrent: bool = False,
        v_threshold: float = 1.0,
        trainable_threshold: bool = False,
        reset_mode: str = 'soft_reset',
        frequency_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
        damping_magnitude_bounds: tuple[float, float] = (0.1, 1.0),
        input_mask: torch.Tensor | None = None,
        recurrent_mask: torch.Tensor | None = None,
        emit_spike: bool = True,
        reset_enabled: bool = True,
        filter_value: float | None = None,
        pole_radius_constrained: bool = True,
        pole_radius_max: float = 0.9999,
    ) -> None:
        super().__init__()
        if reset_mode not in {'soft_reset', 'hard_reset', 'no_reset'}:
            raise ValueError("reset_mode must be 'soft_reset', 'hard_reset', or 'no_reset'.")
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.recurrent = bool(recurrent)
        self.reset_mode = str(reset_mode)
        self.emit_spike = bool(emit_spike)
        self.reset_enabled = bool(reset_enabled) and self.reset_mode != 'no_reset'
        self.trainable_threshold = bool(trainable_threshold)
        self.filter_fixed_value = None if filter_value is None else float(filter_value)
        self.threshold_eps = 1e-6
        self.pole_radius_constrained = bool(pole_radius_constrained)
        self.pole_radius_max = float(pole_radius_max)
        if not math.isfinite(self.pole_radius_max) or self.pole_radius_max <= 0.0:
            raise ValueError('pole_radius_max must be a positive finite number.')
        if self.pole_radius_constrained and self.pole_radius_max >= 1.0:
            raise ValueError('pole_radius_max must be smaller than 1.0 when pole_radius_constrained=True.')
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

        if frequency_bounds is None:
            lower = torch.zeros(self.output_size, dtype=torch.float32)
            upper = torch.full((self.output_size,), 0.5, dtype=torch.float32)
        else:
            lower, upper = frequency_bounds
            lower = lower.detach().clone().to(dtype=torch.float32)
            upper = upper.detach().clone().to(dtype=torch.float32)
        self.register_buffer('freq_lower', lower)
        self.register_buffer('freq_upper', upper)
        if self.filter_fixed_value is not None:
            fixed = torch.full((self.output_size,), float(self.filter_fixed_value), dtype=torch.float32)
            if torch.any(fixed < lower) or torch.any(fixed > upper):
                raise ValueError('Fixed RF filter center frequency is outside frequency clip/bound range.')

        # Backward-compatible init knobs.  They no longer denote a continuous-time
        # damping coefficient; instead they initialize rho via exp(-damping).
        self.damping_lower = float(damping_magnitude_bounds[0])
        self.damping_upper = float(damping_magnitude_bounds[1])
        if not math.isfinite(self.damping_lower) or not math.isfinite(self.damping_upper):
            raise ValueError('damping_magnitude_bounds must be finite.')
        if self.damping_upper <= self.damping_lower:
            raise ValueError('damping_magnitude_bounds must be increasing.')

        self.pole_angle_raw = nn.Parameter(torch.empty(self.output_size))
        self.pole_radius_raw = nn.Parameter(torch.empty(self.output_size))
        threshold_init = torch.full((self.output_size,), float(v_threshold), dtype=torch.float32)
        if self.trainable_threshold:
            self.v_threshold_param = nn.Parameter(_positive_threshold_init(v_threshold, self.output_size, eps=self.threshold_eps))
        else:
            self.register_buffer('v_threshold_buffer', threshold_init)
            self.register_parameter('v_threshold_param', None)
        self.reset_parameters()

    @staticmethod
    def _softplus_inverse(value: torch.Tensor) -> torch.Tensor:
        value = torch.clamp(value, min=1e-6)
        return torch.log(torch.expm1(value))

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.input_weight, a=math.sqrt(5.0))
        if self.recurrent_weight is not None:
            nn.init.orthogonal_(self.recurrent_weight)
        with torch.no_grad():
            freq = torch.empty_like(self.freq_lower)
            radius = torch.empty_like(self.pole_radius_raw)
            dleft, dright = trim_open_interval(self.damping_lower, self.damping_upper)
            radius_left = math.exp(-dright)
            radius_right = math.exp(-dleft)
            if self.pole_radius_constrained:
                radius_right = min(radius_right, self.pole_radius_max - 1e-6)
            radius_left = max(1e-6, min(radius_left, max(radius_right - 1e-6, 1e-6)))
            for index in range(freq.numel()):
                left, right = trim_open_interval(float(self.freq_lower[index]), float(self.freq_upper[index]))
                if self.filter_fixed_value is None:
                    freq[index] = 0.5 * (left + right) if right <= left else float(torch.empty(1).uniform_(left, right).item())
                else:
                    freq[index] = float(self.filter_fixed_value)
                radius[index] = 0.5 * (radius_left + radius_right) if radius_right <= radius_left else float(torch.empty(1).uniform_(radius_left, radius_right).item())
            freq_span = torch.clamp(self.freq_upper - self.freq_lower, min=1e-6)
            self.pole_angle_raw.copy_(logit(torch.clamp((freq - self.freq_lower) / freq_span, min=1e-6, max=1 - 1e-6)))
            if self.pole_radius_constrained:
                normalized_radius = torch.clamp(radius / self.pole_radius_max, min=1e-6, max=1 - 1e-6)
                self.pole_radius_raw.copy_(logit(normalized_radius))
            else:
                self.pole_radius_raw.copy_(self._softplus_inverse(radius))
        self.pole_angle_raw.requires_grad_(self.filter_fixed_value is None)

    def enable_compiled_forward(self, **compile_kwargs: Any) -> tuple[bool, str]:
        no_trace, no_applied, no_policy = compile_callable(_rf_sequence_no_trace, compile_kwargs=compile_kwargs, label='rf_sequence_no_trace')
        with_trace, trace_applied, trace_policy = compile_callable(_rf_sequence_with_trace, compile_kwargs=compile_kwargs, label='rf_sequence_with_trace')
        if no_applied:
            self._compiled_sequence_no_trace = no_trace
        if trace_applied:
            self._compiled_sequence_with_trace = with_trace
        if no_applied or trace_applied:
            self._compiled_sequence_policy = f'no_trace={no_policy};with_trace={trace_policy}'
            self._sequence_compiled_runtime_disabled = False
            self._sequence_compiled_runtime_error = None
        return bool(no_applied or trace_applied), 'sequence_compile[' + f'no_trace={no_policy};with_trace={trace_policy}' + ']'

    def effective_frequency(self) -> torch.Tensor:
        sigma = torch.sigmoid(self.pole_angle_raw)
        return self.freq_lower + (self.freq_upper - self.freq_lower) * sigma

    def effective_pole_angle(self) -> torch.Tensor:
        return 2.0 * math.pi * self.effective_frequency()

    def effective_pole_radius(self) -> torch.Tensor:
        if self.pole_radius_constrained:
            return self.pole_radius_max * torch.sigmoid(self.pole_radius_raw)
        return F.softplus(self.pole_radius_raw)

    def effective_damping_magnitude(self) -> torch.Tensor:
        """Backward-compatible alias: equivalent per-sample damping ``-log(rho)``."""
        return -torch.log(torch.clamp(self.effective_pole_radius(), min=1e-12))

    def effective_b(self) -> torch.Tensor:
        """Backward-compatible alias: ``log(rho)`` in sample units."""
        return torch.log(torch.clamp(self.effective_pole_radius(), min=1e-12))

    def effective_omega(self) -> torch.Tensor:
        """Backward-compatible alias: pole angle in radians/sample."""
        return self.effective_pole_angle()

    def rho(self) -> torch.Tensor:
        return self.effective_pole_radius()

    def f_cyc_per_sample(self) -> torch.Tensor:
        return self.effective_frequency()

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

    def _run_sequence(
        self,
        input_current_sequence: torch.Tensor,
        x_post: torch.Tensor,
        y_post: torch.Tensor,
        prev_spike: torch.Tensor,
        recurrent_weight: torch.Tensor | None,
        threshold_view: torch.Tensor,
        rho_view: torch.Tensor,
        cos_view: torch.Tensor,
        sin_view: torch.Tensor,
        beta_x_view: torch.Tensor,
        beta_y_view: torch.Tensor,
        *,
        return_traces: bool,
        record_raw: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        hard_reset = self.reset_mode == 'hard_reset'
        if return_traces:
            fn = self._compiled_sequence_with_trace
            if fn is not None and not bool(self._sequence_compiled_runtime_disabled):
                try:
                    mem_seq, spike_seq, inp_seq = fn(input_current_sequence, x_post, y_post, prev_spike, recurrent_weight, threshold_view, rho_view, cos_view, sin_view, beta_x_view, beta_y_view, self.emit_spike, self.reset_enabled, hard_reset, record_raw)
                    return mem_seq, spike_seq, inp_seq
                except Exception as exc:
                    disable_compiled_runtime(self, label='sequence', exc=exc)
            mem_seq, spike_seq, inp_seq = _rf_sequence_with_trace(input_current_sequence, x_post, y_post, prev_spike, recurrent_weight, threshold_view, rho_view, cos_view, sin_view, beta_x_view, beta_y_view, self.emit_spike, self.reset_enabled, hard_reset, record_raw)
            return mem_seq, spike_seq, inp_seq
        fn = self._compiled_sequence_no_trace
        if fn is not None and not bool(self._sequence_compiled_runtime_disabled):
            try:
                return None, fn(input_current_sequence, x_post, y_post, prev_spike, recurrent_weight, threshold_view, rho_view, cos_view, sin_view, beta_x_view, beta_y_view, self.emit_spike, self.reset_enabled, hard_reset), None
            except Exception as exc:
                disable_compiled_runtime(self, label='sequence', exc=exc)
        return None, _rf_sequence_no_trace(input_current_sequence, x_post, y_post, prev_spike, recurrent_weight, threshold_view, rho_view, cos_view, sin_view, beta_x_view, beta_y_view, self.emit_spike, self.reset_enabled, hard_reset), None

    def forward(self, input_sequence: torch.Tensor, *, return_traces: bool = False) -> tuple[torch.Tensor | None, torch.Tensor]:
        self._last_layer_input = None
        if input_sequence.ndim != 3:
            raise ValueError(f'Expected input shape (B,T,C), got {tuple(input_sequence.shape)}')
        batch_size, _time_steps, _ = input_sequence.shape
        device = input_sequence.device
        dtype = sequence_state_dtype(input_sequence)
        weight = self.effective_input_weight()
        recurrent_weight = self.effective_recurrent_weight()
        x_post = torch.zeros(batch_size, self.output_size, device=device, dtype=dtype)
        y_post = torch.zeros_like(x_post)
        prev_spike = torch.zeros_like(x_post)
        threshold_view = self.effective_threshold().to(device=device, dtype=dtype).unsqueeze(0)
        record_raw = (not self.emit_spike) and (not self.reset_enabled)
        radius = self.effective_pole_radius().to(device=device, dtype=dtype)
        angle = self.effective_pole_angle().to(device=device, dtype=dtype)
        rho_view = radius.unsqueeze(0)
        cos_view = torch.cos(angle).unsqueeze(0)
        sin_view = torch.sin(angle).unsqueeze(0)
        beta_x_view = torch.ones_like(rho_view)
        beta_y_view = torch.zeros_like(rho_view)
        input_current_sequence = to_sequence_state_dtype(torch.matmul(input_sequence, weight.t()), input_sequence)
        mem_seq, spike_seq, inp_seq = self._run_sequence(input_current_sequence, x_post, y_post, prev_spike, recurrent_weight, threshold_view, rho_view, cos_view, sin_view, beta_x_view, beta_y_view, return_traces=return_traces, record_raw=record_raw)
        self._last_layer_input = inp_seq.contiguous() if inp_seq is not None else None
        return (mem_seq.contiguous() if mem_seq is not None else None), spike_seq.contiguous()

    def filter_stats_vectors(self) -> dict[str, torch.Tensor]:
        radius = self.effective_pole_radius().detach()
        angle = self.effective_pole_angle().detach()
        return {
            'pole_radius': radius,
            'damping': -torch.log(torch.clamp(radius, min=1e-12)),
            'sample_decay_factor': radius,
            'pole_angle': angle,
            'pole_real': radius * torch.cos(angle),
            'pole_imag': radius * torch.sin(angle),
            'center_frequency': self.f_cyc_per_sample().detach(),
            'stability_margin': 1.0 - radius,
            'stability_excess': torch.relu(radius - 1.0),
            'v_threshold': self.effective_threshold().detach(),
        }

try:
    from src.neurons.spikingjelly_compat import install_spikingjelly_contract as _install_spikingjelly_contract
    _install_spikingjelly_contract(RFLayer)
except Exception:
    pass

__all__ = ['RFLayer']
