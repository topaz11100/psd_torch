"""2-D CNN spiking layers used by fixed VGG11/ResNet18 backbones."""

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


def _flatten_time_for_2d(batch_sequence: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Convert ``(B,T,C,H,W)`` to ``(B*T,C,H,W)`` for 2-D modules."""

    if batch_sequence.ndim != 5:
        raise ValueError(f'Expected shape (B,T,C,H,W), got {tuple(batch_sequence.shape)}')
    batch_size, time_steps, channels, height, width = [int(v) for v in batch_sequence.shape]
    flattened = batch_sequence.reshape(batch_size * time_steps, channels, height, width).contiguous(memory_format=torch.channels_last)
    return flattened, batch_size, time_steps


def _restore_time_from_2d(flattened: torch.Tensor, *, batch_size: int, time_steps: int) -> torch.Tensor:
    """Convert ``(B*T,C,H,W)`` back to ``(B,T,C,H,W)``."""

    channels, height, width = [int(v) for v in flattened.shape[1:]]
    return flattened.reshape(int(batch_size), int(time_steps), channels, height, width).contiguous()


def _cnn_lif_sequence_no_trace(
    current_seq: torch.Tensor,
    membrane: torch.Tensor,
    alpha_view: torch.Tensor,
    threshold_view: torch.Tensor,
    emit_spike: bool,
    reset_enabled: bool,
    hard_reset: bool,
) -> torch.Tensor:
    batch_size, time_steps, output_size, height, width = current_seq.shape
    spike_seq = current_seq.new_empty((batch_size, time_steps, output_size, height, width))
    for time_index in range(time_steps):
        current = current_seq[:, time_index, :, :, :]
        membrane_pre = alpha_view * membrane + current
        membrane_signal = membrane_pre - threshold_view
        spike = surrogate_spike(membrane_signal) if emit_spike else torch.zeros_like(membrane_signal)
        if reset_enabled:
            membrane = membrane_pre * (1.0 - spike) if hard_reset else membrane_pre - threshold_view * spike
        else:
            membrane = membrane_pre
        spike_seq[:, time_index, :, :, :] = spike
    return spike_seq


def _cnn_lif_sequence_with_trace(
    current_seq: torch.Tensor,
    membrane: torch.Tensor,
    alpha_view: torch.Tensor,
    threshold_view: torch.Tensor,
    emit_spike: bool,
    reset_enabled: bool,
    hard_reset: bool,
    record_raw: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, time_steps, output_size, height, width = current_seq.shape
    spike_seq = current_seq.new_empty((batch_size, time_steps, output_size, height, width))
    mem_seq = current_seq.new_empty((batch_size, time_steps, output_size, height, width))
    for time_index in range(time_steps):
        current = current_seq[:, time_index, :, :, :]
        membrane_pre = alpha_view * membrane + current
        membrane_signal = membrane_pre - threshold_view
        spike = surrogate_spike(membrane_signal) if emit_spike else torch.zeros_like(membrane_signal)
        if reset_enabled:
            membrane = membrane_pre * (1.0 - spike) if hard_reset else membrane_pre - threshold_view * spike
        else:
            membrane = membrane_pre
        mem_seq[:, time_index, :, :, :] = membrane_pre if record_raw else membrane_signal
        spike_seq[:, time_index, :, :, :] = spike
    return mem_seq, spike_seq


def _cnn_rf_sequence_no_trace(
    current_seq: torch.Tensor,
    x_post: torch.Tensor,
    y_post: torch.Tensor,
    rho_view: torch.Tensor,
    cos_view: torch.Tensor,
    sin_view: torch.Tensor,
    beta_x_view: torch.Tensor,
    beta_y_view: torch.Tensor,
    threshold_view: torch.Tensor,
    emit_spike: bool,
    reset_enabled: bool,
    hard_reset: bool,
) -> torch.Tensor:
    batch_size, time_steps, output_size, height, width = current_seq.shape
    spike_seq = current_seq.new_empty((batch_size, time_steps, output_size, height, width))
    for time_index in range(time_steps):
        current = current_seq[:, time_index, :, :, :]
        x_pre = rho_view * (cos_view * x_post - sin_view * y_post) + beta_x_view * current
        y_pre = rho_view * (sin_view * x_post + cos_view * y_post) + beta_y_view * current
        membrane_signal = x_pre - threshold_view
        spike = surrogate_spike(membrane_signal) if emit_spike else torch.zeros_like(membrane_signal)
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
        spike_seq[:, time_index, :, :, :] = spike
    return spike_seq


def _cnn_rf_sequence_with_trace(
    current_seq: torch.Tensor,
    x_post: torch.Tensor,
    y_post: torch.Tensor,
    rho_view: torch.Tensor,
    cos_view: torch.Tensor,
    sin_view: torch.Tensor,
    beta_x_view: torch.Tensor,
    beta_y_view: torch.Tensor,
    threshold_view: torch.Tensor,
    emit_spike: bool,
    reset_enabled: bool,
    hard_reset: bool,
    record_raw: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, time_steps, output_size, height, width = current_seq.shape
    spike_seq = current_seq.new_empty((batch_size, time_steps, output_size, height, width))
    mem_seq = current_seq.new_empty((batch_size, time_steps, output_size, height, width))
    for time_index in range(time_steps):
        current = current_seq[:, time_index, :, :, :]
        x_pre = rho_view * (cos_view * x_post - sin_view * y_post) + beta_x_view * current
        y_pre = rho_view * (sin_view * x_post + cos_view * y_post) + beta_y_view * current
        membrane_signal = x_pre - threshold_view
        spike = surrogate_spike(membrane_signal) if emit_spike else torch.zeros_like(membrane_signal)
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
        mem_seq[:, time_index, :, :, :] = x_pre if record_raw else membrane_signal
        spike_seq[:, time_index, :, :, :] = spike
    return mem_seq, spike_seq


class CNN2DLIFLayer(nn.Module):
    """Time-distributed Conv2d input coupling followed by LIF dynamics."""

    compile_granularity = 'sequence'

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        v_threshold: float = 1.0,
        trainable_threshold: bool = False,
        reset_mode: str = 'soft_reset',
        alpha_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
        emit_spike: bool = True,
        reset_enabled: bool = True,
        batch_norm: bool = False,
        bias: bool = False,
        filter_value: float | None = None,
    ) -> None:
        super().__init__()
        if reset_mode not in {'soft_reset', 'hard_reset'}:
            raise ValueError("reset_mode must be 'soft_reset' or 'hard_reset'.")
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.trainable_threshold = bool(trainable_threshold)
        self.filter_fixed_value = None if filter_value is None else float(filter_value)
        self.threshold_eps = 1.0e-6
        self.reset_mode = str(reset_mode)
        self.emit_spike = bool(emit_spike)
        self.reset_enabled = bool(reset_enabled)
        self.uses_batch_norm = bool(batch_norm)
        self._compiled_sequence_no_trace = None
        self._compiled_sequence_with_trace = None
        self._compiled_sequence_policy = 'eager'
        self._sequence_compiled_runtime_disabled = False
        self._sequence_compiled_runtime_error = None
        self.conv = nn.Conv2d(
            self.input_size,
            self.output_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=bool(bias),
        )
        self.bn = nn.BatchNorm2d(self.output_size) if self.uses_batch_norm else None
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
                raise ValueError('Fixed CNN-LIF filter alpha is outside alpha clip/bound range.')
        self.alpha_raw = nn.Parameter(torch.empty(self.output_size))
        threshold_init = torch.full((self.output_size,), float(v_threshold), dtype=torch.float32)
        if self.trainable_threshold:
            self.v_threshold_param = nn.Parameter(_positive_threshold_init(v_threshold, self.output_size, eps=self.threshold_eps))
        else:
            self.register_buffer('v_threshold_buffer', threshold_init)
            self.register_parameter('v_threshold_param', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5.0))
        if self.conv.bias is not None:
            fan_in, _fan_out = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.conv.bias, -bound, bound)
        if self.bn is not None:
            self.bn.reset_parameters()
        init_alpha = torch.empty_like(self.alpha_lower)
        if self.filter_fixed_value is None:
            for index in range(init_alpha.numel()):
                left, right = trim_open_interval(float(self.alpha_lower[index]), float(self.alpha_upper[index]))
                init_alpha[index] = 0.5 * (left + right) if right <= left else float(torch.empty(1).uniform_(left, right).item())
        else:
            init_alpha.fill_(float(self.filter_fixed_value))
        with torch.no_grad():
            alpha_span = torch.clamp(self.alpha_upper - self.alpha_lower, min=1.0e-6)
            alpha01 = torch.clamp((init_alpha - self.alpha_lower) / alpha_span, min=1.0e-6, max=1.0 - 1.0e-6)
            self.alpha_raw.copy_(logit(alpha01))
        self.alpha_raw.requires_grad_(self.filter_fixed_value is None)

    def effective_alpha(self) -> torch.Tensor:
        sigma = torch.sigmoid(self.alpha_raw)
        return self.alpha_lower + (self.alpha_upper - self.alpha_lower) * sigma

    def effective_threshold(self) -> torch.Tensor:
        if self.v_threshold_param is not None:
            return F.softplus(self.v_threshold_param) + float(self.threshold_eps)
        return self.v_threshold_buffer

    def effective_input_weight(self) -> torch.Tensor:
        return self.conv.weight.reshape(self.output_size, -1)

    def enable_compiled_forward(self, **compile_kwargs: Any) -> tuple[bool, str]:
        no_trace, no_applied, no_policy = compile_callable(_cnn_lif_sequence_no_trace, compile_kwargs=compile_kwargs, label='cnn_lif_sequence_no_trace')
        with_trace, trace_applied, trace_policy = compile_callable(_cnn_lif_sequence_with_trace, compile_kwargs=compile_kwargs, label='cnn_lif_sequence_with_trace')
        if no_applied:
            self._compiled_sequence_no_trace = no_trace
        if trace_applied:
            self._compiled_sequence_with_trace = with_trace
        if no_applied or trace_applied:
            self._compiled_sequence_policy = f'no_trace={no_policy};with_trace={trace_policy}'
            self._sequence_compiled_runtime_disabled = False
            self._sequence_compiled_runtime_error = None
        return bool(no_applied or trace_applied), 'sequence_compile[' + f'no_trace={no_policy};with_trace={trace_policy}' + ']'

    def _run_sequence(self, current_seq: torch.Tensor, membrane: torch.Tensor, alpha_view: torch.Tensor, threshold_view: torch.Tensor, *, return_traces: bool, record_raw: bool) -> tuple[torch.Tensor | None, torch.Tensor]:
        hard_reset = self.reset_mode == 'hard_reset'
        if return_traces:
            fn = self._compiled_sequence_with_trace
            if fn is not None and not bool(self._sequence_compiled_runtime_disabled):
                try:
                    return fn(current_seq, membrane, alpha_view, threshold_view, self.emit_spike, self.reset_enabled, hard_reset, record_raw)
                except Exception as exc:
                    disable_compiled_runtime(self, label='sequence', exc=exc)
            return _cnn_lif_sequence_with_trace(current_seq, membrane, alpha_view, threshold_view, self.emit_spike, self.reset_enabled, hard_reset, record_raw)
        fn = self._compiled_sequence_no_trace
        if fn is not None and not bool(self._sequence_compiled_runtime_disabled):
            try:
                return None, fn(current_seq, membrane, alpha_view, threshold_view, self.emit_spike, self.reset_enabled, hard_reset)
            except Exception as exc:
                disable_compiled_runtime(self, label='sequence', exc=exc)
        return None, _cnn_lif_sequence_no_trace(current_seq, membrane, alpha_view, threshold_view, self.emit_spike, self.reset_enabled, hard_reset)

    def forward(self, input_sequence: torch.Tensor, *, return_traces: bool = False) -> tuple[torch.Tensor | None, torch.Tensor]:
        self._last_layer_input = None
        flattened, batch_size, time_steps = _flatten_time_for_2d(input_sequence)
        current_flat = self.conv(flattened)
        if self.bn is not None:
            current_flat = self.bn(current_flat)
        current_seq = to_sequence_state_dtype(_restore_time_from_2d(current_flat, batch_size=batch_size, time_steps=time_steps), input_sequence)
        _batch, _time, _channels, height, width = [int(v) for v in current_seq.shape]
        dtype = sequence_state_dtype(input_sequence)
        alpha = self.effective_alpha().to(device=input_sequence.device, dtype=dtype)
        membrane = torch.zeros(batch_size, self.output_size, height, width, device=input_sequence.device, dtype=dtype)
        threshold = self.effective_threshold().to(device=input_sequence.device, dtype=dtype)
        alpha_view = alpha.view(1, -1, 1, 1)
        threshold_view = threshold.view(1, -1, 1, 1)
        record_raw_membrane = (not self.emit_spike) and (not self.reset_enabled)
        mem_seq, spike_seq = self._run_sequence(current_seq, membrane, alpha_view, threshold_view, return_traces=return_traces, record_raw=record_raw_membrane)
        self._last_layer_input = current_seq if return_traces else None
        return (mem_seq.contiguous() if mem_seq is not None else None), spike_seq.contiguous()

    def filter_stats_vectors(self) -> dict[str, torch.Tensor]:
        return {'alpha': self.effective_alpha().detach(), 'v_threshold': self.effective_threshold().detach()}


class CNN2DRFLayer(nn.Module):
    """Time-distributed Conv2d input coupling followed by direct discrete RF dynamics."""

    compile_granularity = 'sequence'

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        v_threshold: float = 1.0,
        trainable_threshold: bool = False,
        reset_mode: str = 'soft_reset',
        frequency_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
        damping_magnitude_bounds: tuple[float, float] = (0.1, 1.0),
        emit_spike: bool = True,
        reset_enabled: bool = True,
        batch_norm: bool = False,
        bias: bool = False,
        filter_value: float | None = None,
        pole_radius_constrained: bool = True,
        pole_radius_max: float = 0.9999,
    ) -> None:
        super().__init__()
        if reset_mode not in {'soft_reset', 'hard_reset', 'no_reset'}:
            raise ValueError("reset_mode must be 'soft_reset', 'hard_reset', or 'no_reset'.")
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.trainable_threshold = bool(trainable_threshold)
        self.filter_fixed_value = None if filter_value is None else float(filter_value)
        self.threshold_eps = 1.0e-6
        self.emit_spike = bool(emit_spike)
        self.reset_mode = str(reset_mode)
        self.reset_enabled = bool(reset_enabled) and self.reset_mode != 'no_reset'
        self.pole_radius_constrained = bool(pole_radius_constrained)
        self.pole_radius_max = float(pole_radius_max)
        if not math.isfinite(self.pole_radius_max) or self.pole_radius_max <= 0.0:
            raise ValueError('pole_radius_max must be a positive finite number.')
        if self.pole_radius_constrained and self.pole_radius_max >= 1.0:
            raise ValueError('pole_radius_max must be smaller than 1.0 when pole_radius_constrained=True.')
        self.uses_batch_norm = bool(batch_norm)
        self._compiled_sequence_no_trace = None
        self._compiled_sequence_with_trace = None
        self._compiled_sequence_policy = 'eager'
        self._sequence_compiled_runtime_disabled = False
        self._sequence_compiled_runtime_error = None
        self.conv = nn.Conv2d(
            self.input_size,
            self.output_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=bool(bias),
        )
        self.bn = nn.BatchNorm2d(self.output_size) if self.uses_batch_norm else None
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
                raise ValueError('Fixed CNN-RF filter center frequency is outside frequency clip/bound range.')
        self.damping_lower = float(damping_magnitude_bounds[0])
        self.damping_upper = float(damping_magnitude_bounds[1])
        self.pole_angle_raw = nn.Parameter(torch.empty(self.output_size))
        self.pole_radius_raw = nn.Parameter(torch.empty(self.output_size))
        threshold_init = torch.full((self.output_size,), float(v_threshold), dtype=torch.float32)
        if self.trainable_threshold:
            self.v_threshold_param = nn.Parameter(_positive_threshold_init(v_threshold, self.output_size, eps=self.threshold_eps))
        else:
            self.register_buffer('v_threshold_buffer', threshold_init)
            self.register_parameter('v_threshold_param', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5.0))
        if self.conv.bias is not None:
            fan_in, _fan_out = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.conv.bias, -bound, bound)
        if self.bn is not None:
            self.bn.reset_parameters()
        with torch.no_grad():
            freq = torch.empty_like(self.freq_lower)
            radius = torch.empty_like(self.pole_radius_raw)
            dleft, dright = trim_open_interval(self.damping_lower, self.damping_upper)
            radius_left = math.exp(-dright)
            radius_right = math.exp(-dleft)
            if self.pole_radius_constrained:
                radius_right = min(radius_right, self.pole_radius_max - 1.0e-6)
            radius_left = max(1.0e-6, min(radius_left, max(radius_right - 1.0e-6, 1.0e-6)))
            for index in range(freq.numel()):
                left, right = trim_open_interval(float(self.freq_lower[index]), float(self.freq_upper[index]))
                if self.filter_fixed_value is None:
                    freq[index] = 0.5 * (left + right) if right <= left else float(torch.empty(1).uniform_(left, right).item())
                else:
                    freq[index] = float(self.filter_fixed_value)
                radius[index] = 0.5 * (radius_left + radius_right) if radius_right <= radius_left else float(torch.empty(1).uniform_(radius_left, radius_right).item())
            freq_span = torch.clamp(self.freq_upper - self.freq_lower, min=1.0e-6)
            freq01 = torch.clamp((freq - self.freq_lower) / freq_span, min=1.0e-6, max=1.0 - 1.0e-6)
            self.pole_angle_raw.copy_(torch.log(freq01) - torch.log1p(-freq01))
            if self.pole_radius_constrained:
                radius01 = torch.clamp(radius / self.pole_radius_max, min=1.0e-6, max=1.0 - 1.0e-6)
                self.pole_radius_raw.copy_(torch.log(radius01) - torch.log1p(-radius01))
            else:
                self.pole_radius_raw.copy_(torch.log(torch.expm1(torch.clamp(radius, min=1.0e-6))))
        self.pole_angle_raw.requires_grad_(self.filter_fixed_value is None)

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
        return -torch.log(torch.clamp(self.effective_pole_radius(), min=1.0e-12))

    def effective_b(self) -> torch.Tensor:
        return torch.log(torch.clamp(self.effective_pole_radius(), min=1.0e-12))

    def effective_omega(self) -> torch.Tensor:
        return self.effective_pole_angle()

    def effective_threshold(self) -> torch.Tensor:
        if self.v_threshold_param is not None:
            return F.softplus(self.v_threshold_param) + float(self.threshold_eps)
        return self.v_threshold_buffer

    def rho(self) -> torch.Tensor:
        return self.effective_pole_radius()

    def f_cyc_per_sample(self) -> torch.Tensor:
        return self.effective_frequency()

    def effective_input_weight(self) -> torch.Tensor:
        return self.conv.weight.reshape(self.output_size, -1)


    def enable_compiled_forward(self, **compile_kwargs: Any) -> tuple[bool, str]:
        no_trace, no_applied, no_policy = compile_callable(_cnn_rf_sequence_no_trace, compile_kwargs=compile_kwargs, label='cnn_rf_sequence_no_trace')
        with_trace, trace_applied, trace_policy = compile_callable(_cnn_rf_sequence_with_trace, compile_kwargs=compile_kwargs, label='cnn_rf_sequence_with_trace')
        if no_applied:
            self._compiled_sequence_no_trace = no_trace
        if trace_applied:
            self._compiled_sequence_with_trace = with_trace
        if no_applied or trace_applied:
            self._compiled_sequence_policy = f'no_trace={no_policy};with_trace={trace_policy}'
            self._sequence_compiled_runtime_disabled = False
            self._sequence_compiled_runtime_error = None
        return bool(no_applied or trace_applied), 'sequence_compile[' + f'no_trace={no_policy};with_trace={trace_policy}' + ']'

    def _run_sequence(self, current_seq: torch.Tensor, x_post: torch.Tensor, y_post: torch.Tensor, rho_view: torch.Tensor, cos_view: torch.Tensor, sin_view: torch.Tensor, beta_x_view: torch.Tensor, beta_y_view: torch.Tensor, threshold_view: torch.Tensor, *, return_traces: bool, record_raw: bool) -> tuple[torch.Tensor | None, torch.Tensor]:
        hard_reset = self.reset_mode == 'hard_reset'
        if return_traces:
            fn = self._compiled_sequence_with_trace
            if fn is not None and not bool(self._sequence_compiled_runtime_disabled):
                try:
                    return fn(current_seq, x_post, y_post, rho_view, cos_view, sin_view, beta_x_view, beta_y_view, threshold_view, self.emit_spike, self.reset_enabled, hard_reset, record_raw)
                except Exception as exc:
                    disable_compiled_runtime(self, label='sequence', exc=exc)
            return _cnn_rf_sequence_with_trace(current_seq, x_post, y_post, rho_view, cos_view, sin_view, beta_x_view, beta_y_view, threshold_view, self.emit_spike, self.reset_enabled, hard_reset, record_raw)
        fn = self._compiled_sequence_no_trace
        if fn is not None and not bool(self._sequence_compiled_runtime_disabled):
            try:
                return None, fn(current_seq, x_post, y_post, rho_view, cos_view, sin_view, beta_x_view, beta_y_view, threshold_view, self.emit_spike, self.reset_enabled, hard_reset)
            except Exception as exc:
                disable_compiled_runtime(self, label='sequence', exc=exc)
        return None, _cnn_rf_sequence_no_trace(current_seq, x_post, y_post, rho_view, cos_view, sin_view, beta_x_view, beta_y_view, threshold_view, self.emit_spike, self.reset_enabled, hard_reset)

    def forward(self, input_sequence: torch.Tensor, *, return_traces: bool = False) -> tuple[torch.Tensor | None, torch.Tensor]:
        self._last_layer_input = None
        flattened, batch_size, time_steps = _flatten_time_for_2d(input_sequence)
        current_flat = self.conv(flattened)
        if self.bn is not None:
            current_flat = self.bn(current_flat)
        current_seq = to_sequence_state_dtype(_restore_time_from_2d(current_flat, batch_size=batch_size, time_steps=time_steps), input_sequence)
        _batch, _time, _channels, height, width = [int(v) for v in current_seq.shape]
        dtype = sequence_state_dtype(input_sequence)
        device = input_sequence.device
        rho = self.effective_pole_radius().to(device=device, dtype=dtype)
        angle = self.effective_pole_angle().to(device=device, dtype=dtype)
        cos_phi = torch.cos(angle)
        sin_phi = torch.sin(angle)
        beta_x = torch.ones_like(rho)
        beta_y = torch.zeros_like(rho)
        x_post = torch.zeros(batch_size, self.output_size, height, width, device=device, dtype=dtype)
        y_post = torch.zeros_like(x_post)
        threshold = self.effective_threshold().to(device=device, dtype=dtype)
        rho_view = rho.view(1, -1, 1, 1)
        cos_view = cos_phi.view(1, -1, 1, 1)
        sin_view = sin_phi.view(1, -1, 1, 1)
        beta_x_view = beta_x.view(1, -1, 1, 1)
        beta_y_view = beta_y.view(1, -1, 1, 1)
        threshold_view = threshold.view(1, -1, 1, 1)
        record_raw_membrane = (not self.emit_spike) and (not self.reset_enabled)
        mem_seq, spike_seq = self._run_sequence(current_seq, x_post, y_post, rho_view, cos_view, sin_view, beta_x_view, beta_y_view, threshold_view, return_traces=return_traces, record_raw=record_raw_membrane)
        self._last_layer_input = current_seq if return_traces else None
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
    _install_spikingjelly_contract(CNN2DLIFLayer)
    _install_spikingjelly_contract(CNN2DRFLayer)
except Exception:  # pragma: no cover - defensive import fallback
    pass

__all__ = ['CNN2DLIFLayer', 'CNN2DRFLayer']
