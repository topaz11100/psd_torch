"""2-D CNN spiking layers used by fixed VGG11/ResNet18 backbones."""

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



def _flatten_time_for_2d(batch_sequence: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Convert ``(B,T,C,H,W)`` to ``(B*T,C,H,W)`` for 2-D modules."""

    if batch_sequence.ndim != 5:
        raise ValueError(f'Expected shape (B,T,C,H,W), got {tuple(batch_sequence.shape)}')
    batch_size, time_steps, channels, height, width = [int(v) for v in batch_sequence.shape]
    flattened = batch_sequence.reshape(batch_size * time_steps, channels, height, width)
    return flattened, batch_size, time_steps


def _restore_time_from_2d(flattened: torch.Tensor, *, batch_size: int, time_steps: int) -> torch.Tensor:
    """Convert ``(B*T,C,H,W)`` back to ``(B,T,C,H,W)``."""

    channels, height, width = [int(v) for v in flattened.shape[1:]]
    return flattened.reshape(int(batch_size), int(time_steps), channels, height, width).contiguous()


class CNN2DLIFLayer(nn.Module):
    """Time-distributed Conv2d input coupling followed by LIF dynamics."""

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
    ) -> None:
        """Initialize the 2-D convolutional LIF layer."""
        super().__init__()
        if reset_mode not in {'soft_reset', 'hard_reset'}:
            raise ValueError("reset_mode must be 'soft_reset' or 'hard_reset'.")
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.trainable_threshold = bool(trainable_threshold)
        self.threshold_eps = 1.0e-6
        self.reset_mode = str(reset_mode)
        self.emit_spike = bool(emit_spike)
        self.reset_enabled = bool(reset_enabled)
        self.uses_batch_norm = bool(batch_norm)

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
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5.0))
        if self.conv.bias is not None:
            fan_in, _fan_out = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.conv.bias, -bound, bound)
        if self.bn is not None:
            self.bn.reset_parameters()
        init_alpha = torch.empty_like(self.alpha_lower)
        for index in range(init_alpha.numel()):
            left, right = trim_open_interval(float(self.alpha_lower[index]), float(self.alpha_upper[index]))
            init_alpha[index] = 0.5 * (left + right) if right <= left else float(torch.empty(1).uniform_(left, right).item())
        with torch.no_grad():
            self.alpha_raw.copy_(logit(init_alpha))

    def effective_alpha(self) -> torch.Tensor:
        """Handle ``effective alpha`` for the ``cnn_lif`` module."""
        sigma = torch.sigmoid(self.alpha_raw)
        return self.alpha_lower + (self.alpha_upper - self.alpha_lower) * sigma

    def effective_threshold(self) -> torch.Tensor:
        """Handle ``effective threshold`` for the ``cnn_lif`` module."""
        if self.v_threshold_param is not None:
            return F.softplus(self.v_threshold_param) + float(self.threshold_eps)
        return self.v_threshold_buffer

    def effective_input_weight(self) -> torch.Tensor:
        """Handle ``effective input weight`` for the ``cnn_lif`` module."""
        return self.conv.weight.reshape(self.output_size, -1)

    def forward(self, input_sequence: torch.Tensor, *, return_traces: bool = False) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Run the forward pass on ``(B,T,C,H,W)`` tensors."""
        self._last_layer_input = None
        flattened, batch_size, time_steps = _flatten_time_for_2d(input_sequence)
        current_flat = self.conv(flattened)
        if self.bn is not None:
            current_flat = self.bn(current_flat)
        current_seq = _restore_time_from_2d(current_flat, batch_size=batch_size, time_steps=time_steps)
        _batch, _time, _channels, height, width = [int(v) for v in current_seq.shape]
        alpha = self.effective_alpha().to(device=input_sequence.device, dtype=input_sequence.dtype)
        membrane = torch.zeros(batch_size, self.output_size, height, width, device=input_sequence.device, dtype=input_sequence.dtype)
        threshold = self.effective_threshold().to(device=input_sequence.device, dtype=input_sequence.dtype)
        alpha_view = alpha.view(1, -1, 1, 1)
        threshold_view = threshold.view(1, -1, 1, 1)
        record_raw_membrane = (not self.emit_spike) and (not self.reset_enabled)
        mem_steps: list[torch.Tensor] | None = [] if return_traces else None
        spike_steps: list[torch.Tensor] = []
        for time_index in range(time_steps):
            membrane_pre = alpha_view * membrane + current_seq[:, time_index, :, :, :]
            membrane_signal = membrane_pre - threshold_view
            spike = surrogate_spike(membrane_signal) if self.emit_spike else torch.zeros_like(membrane_signal)
            if mem_steps is not None:
                mem_steps.append(membrane_pre if record_raw_membrane else membrane_signal)
            spike_steps.append(spike)
            if self.reset_enabled:
                if self.reset_mode == 'soft_reset':
                    membrane = membrane_pre - threshold_view * spike
                else:
                    membrane = membrane_pre * (1.0 - spike)
            else:
                membrane = membrane_pre
        self._last_layer_input = current_seq if return_traces else None
        return (torch.stack(mem_steps, dim=1) if mem_steps is not None else None), torch.stack(spike_steps, dim=1)

    def filter_stats_vectors(self) -> dict[str, torch.Tensor]:
        """Handle ``filter stats vectors`` for the ``cnn_lif`` module."""
        return {'alpha': self.effective_alpha().detach()}



class CNN2DRFLayer(nn.Module):
    """Time-distributed Conv2d input coupling followed by exact-ZOH RF dynamics."""

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
    ) -> None:
        """Initialize the 2-D convolutional RF layer."""
        super().__init__()
        if reset_mode not in {'soft_reset', 'hard_reset', 'no_reset'}:
            raise ValueError("reset_mode must be 'soft_reset', 'hard_reset', or 'no_reset'.")
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.trainable_threshold = bool(trainable_threshold)
        self.threshold_eps = 1.0e-6
        self.emit_spike = bool(emit_spike)
        self.reset_mode = str(reset_mode)
        self.reset_enabled = bool(reset_enabled) and self.reset_mode != 'no_reset'
        self.uses_batch_norm = bool(batch_norm)

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
        self.damping_lower = float(damping_magnitude_bounds[0])
        self.damping_upper = float(damping_magnitude_bounds[1])

        self.freq_raw = nn.Parameter(torch.empty(self.output_size))
        self.damping_raw = nn.Parameter(torch.empty(self.output_size))
        threshold_init = torch.full((self.output_size,), float(v_threshold), dtype=torch.float32)
        if self.trainable_threshold:
            self.v_threshold_param = nn.Parameter(_positive_threshold_init(v_threshold, self.output_size, eps=self.threshold_eps))
        else:
            self.register_buffer('v_threshold_buffer', threshold_init)
            self.register_parameter('v_threshold_param', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset parameters."""
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5.0))
        if self.conv.bias is not None:
            fan_in, _fan_out = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.conv.bias, -bound, bound)
        if self.bn is not None:
            self.bn.reset_parameters()
        with torch.no_grad():
            freq = torch.empty_like(self.freq_lower)
            damping = torch.empty_like(self.damping_raw)
            for index in range(freq.numel()):
                left, right = trim_open_interval(float(self.freq_lower[index]), float(self.freq_upper[index]))
                freq[index] = 0.5 * (left + right) if right <= left else float(torch.empty(1).uniform_(left, right).item())
                dleft, dright = trim_open_interval(self.damping_lower, self.damping_upper)
                damping[index] = 0.5 * (dleft + dright) if dright <= dleft else float(torch.empty(1).uniform_(dleft, dright).item())
            freq01 = torch.clamp(freq / 0.5, min=1.0e-6, max=1.0 - 1.0e-6)
            damp01 = torch.clamp((damping - self.damping_lower) / (self.damping_upper - self.damping_lower), min=1.0e-6, max=1.0 - 1.0e-6)
            self.freq_raw.copy_(torch.log(freq01) - torch.log1p(-freq01))
            self.damping_raw.copy_(torch.log(damp01) - torch.log1p(-damp01))

    def effective_frequency(self) -> torch.Tensor:
        """Handle ``effective frequency`` for the ``cnn_rf`` module."""
        sigma = torch.sigmoid(self.freq_raw)
        return self.freq_lower + (self.freq_upper - self.freq_lower) * sigma

    def effective_damping_magnitude(self) -> torch.Tensor:
        """Handle ``effective damping magnitude`` for the ``cnn_rf`` module."""
        sigma = torch.sigmoid(self.damping_raw)
        return self.damping_lower + (self.damping_upper - self.damping_lower) * sigma

    def effective_b(self) -> torch.Tensor:
        """Handle ``effective b`` for the ``cnn_rf`` module."""
        return -self.effective_damping_magnitude()

    def effective_omega(self) -> torch.Tensor:
        """Handle ``effective omega`` for the ``cnn_rf`` module."""
        return 2.0 * math.pi * self.effective_frequency()

    def effective_threshold(self) -> torch.Tensor:
        """Handle ``effective threshold`` for the ``cnn_rf`` module."""
        if self.v_threshold_param is not None:
            return F.softplus(self.v_threshold_param) + float(self.threshold_eps)
        return self.v_threshold_buffer

    def rho(self) -> torch.Tensor:
        """Handle ``rho`` for the ``cnn_rf`` module."""
        return torch.exp(self.effective_b())

    def f_cyc_per_sample(self) -> torch.Tensor:
        """Handle ``f cyc per sample`` for the ``cnn_rf`` module."""
        return self.effective_frequency()

    def effective_input_weight(self) -> torch.Tensor:
        """Handle ``effective input weight`` for the ``cnn_rf`` module."""
        return self.conv.weight.reshape(self.output_size, -1)

    def forward(self, input_sequence: torch.Tensor, *, return_traces: bool = False) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Run the forward pass on ``(B,T,C,H,W)`` tensors."""
        self._last_layer_input = None
        flattened, batch_size, time_steps = _flatten_time_for_2d(input_sequence)
        current_flat = self.conv(flattened)
        if self.bn is not None:
            current_flat = self.bn(current_flat)
        current_seq = _restore_time_from_2d(current_flat, batch_size=batch_size, time_steps=time_steps)
        _batch, _time, _channels, height, width = [int(v) for v in current_seq.shape]
        dtype = input_sequence.dtype
        device = input_sequence.device
        b = self.effective_b().to(device=device, dtype=dtype)
        omega = self.effective_omega().to(device=device, dtype=dtype)
        rho = torch.exp(b)
        phi = omega
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        complex_den = torch.complex(b, omega)
        alpha_complex = torch.exp(torch.complex(b, omega))
        beta_complex = (alpha_complex - 1.0) / complex_den
        beta_x = beta_complex.real.to(dtype=dtype)
        beta_y = beta_complex.imag.to(dtype=dtype)

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
        mem_steps: list[torch.Tensor] | None = [] if return_traces else None
        spike_steps: list[torch.Tensor] = []
        for time_index in range(time_steps):
            current = current_seq[:, time_index, :, :, :]
            x_pre = rho_view * (cos_view * x_post - sin_view * y_post) + beta_x_view * current
            y_pre = rho_view * (sin_view * x_post + cos_view * y_post) + beta_y_view * current
            membrane_signal = x_pre - threshold_view
            spike = surrogate_spike(membrane_signal) if self.emit_spike else torch.zeros_like(membrane_signal)
            if self.reset_enabled:
                if self.reset_mode == 'soft_reset':
                    x_post = x_pre - threshold_view * spike
                    y_post = y_pre
                else:
                    keep = 1.0 - spike
                    x_post = x_pre * keep
                    y_post = y_pre * keep
            else:
                x_post = x_pre
                y_post = y_pre
            if mem_steps is not None:
                mem_steps.append(x_pre if record_raw_membrane else membrane_signal)
            spike_steps.append(spike)
        self._last_layer_input = current_seq if return_traces else None
        return (torch.stack(mem_steps, dim=1) if mem_steps is not None else None), torch.stack(spike_steps, dim=1)

    def filter_stats_vectors(self) -> dict[str, torch.Tensor]:
        """Handle ``filter stats vectors`` for the ``cnn_rf`` module."""
        return {
            'damping': self.effective_damping_magnitude().detach(),
            'center_frequency': self.f_cyc_per_sample().detach(),
        }



try:
    from src.neurons.spikingjelly_compat import install_spikingjelly_contract as _install_spikingjelly_contract
    _install_spikingjelly_contract(CNN2DLIFLayer)
    _install_spikingjelly_contract(CNN2DRFLayer)
except Exception:  # pragma: no cover - defensive import fallback
    pass

__all__ = ['CNN2DLIFLayer', 'CNN2DRFLayer']
