"""Project-standard vanilla exact-ZOH RF layer."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.neurons._common import trim_open_interval, surrogate_spike

def _positive_threshold_init(v_threshold: float, size: int, *, eps: float) -> torch.Tensor:
    value = max(float(v_threshold) - float(eps), float(eps))
    raw = math.log(math.expm1(value))
    return torch.full((int(size),), float(raw), dtype=torch.float32)



class RFLayer(nn.Module):
    """Dense exact-ZOH Resonate-and-Fire layer with optional recurrent adapter."""

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
    ) -> None:
        """Initialize the instance with the provided configuration."""
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
        nn.init.kaiming_uniform_(self.input_weight, a=math.sqrt(5.0))
        if self.recurrent_weight is not None:
            nn.init.orthogonal_(self.recurrent_weight)
        with torch.no_grad():
            freq = torch.empty_like(self.freq_lower)
            damping = torch.empty_like(self.damping_raw)
            for index in range(freq.numel()):
                left, right = trim_open_interval(float(self.freq_lower[index]), float(self.freq_upper[index]))
                if right <= left:
                    freq[index] = 0.5 * (left + right)
                else:
                    freq[index] = float(torch.empty(1).uniform_(left, right).item())
                dleft, dright = trim_open_interval(self.damping_lower, self.damping_upper)
                if dright <= dleft:
                    damping[index] = 0.5 * (dleft + dright)
                else:
                    damping[index] = float(torch.empty(1).uniform_(dleft, dright).item())
            freq01 = torch.clamp(freq / 0.5, min=1.0e-6, max=1.0 - 1.0e-6)
            damp01 = torch.clamp((damping - self.damping_lower) / (self.damping_upper - self.damping_lower), min=1.0e-6, max=1.0 - 1.0e-6)
            self.freq_raw.copy_(torch.log(freq01) - torch.log1p(-freq01))
            self.damping_raw.copy_(torch.log(damp01) - torch.log1p(-damp01))

    def effective_frequency(self) -> torch.Tensor:
        """Handle ``effective frequency`` for the ``RF_neuron`` module."""
        sigma = torch.sigmoid(self.freq_raw)
        return self.freq_lower + (self.freq_upper - self.freq_lower) * sigma

    def effective_damping_magnitude(self) -> torch.Tensor:
        """Handle ``effective damping magnitude`` for the ``RF_neuron`` module."""
        sigma = torch.sigmoid(self.damping_raw)
        return self.damping_lower + (self.damping_upper - self.damping_lower) * sigma

    def effective_b(self) -> torch.Tensor:
        """Handle ``effective b`` for the ``RF_neuron`` module."""
        return -self.effective_damping_magnitude()

    def effective_omega(self) -> torch.Tensor:
        """Handle ``effective omega`` for the ``RF_neuron`` module."""
        return 2.0 * math.pi * self.effective_frequency()

    def rho(self) -> torch.Tensor:
        """Handle ``rho`` for the ``RF_neuron`` module."""
        return torch.exp(self.effective_b())

    def f_cyc_per_sample(self) -> torch.Tensor:
        """Handle ``f cyc per sample`` for the ``RF_neuron`` module."""
        return self.effective_frequency()

    def effective_threshold(self) -> torch.Tensor:
        """Handle ``effective threshold`` for the ``RF_neuron`` module."""
        if self.v_threshold_param is not None:
            return F.softplus(self.v_threshold_param) + float(self.threshold_eps)
        return self.v_threshold_buffer

    def effective_input_weight(self) -> torch.Tensor:
        """Handle ``effective input weight`` for the ``RF_neuron`` module."""
        return self.input_weight * self.input_mask

    def effective_recurrent_weight(self) -> torch.Tensor | None:
        """Handle ``effective recurrent weight`` for the ``RF_neuron`` module."""
        if self.recurrent_weight is None:
            return None
        if self.recurrent_mask is None:
            return self.recurrent_weight
        return self.recurrent_weight * self.recurrent_mask

    def forward(self, input_sequence: torch.Tensor, *, return_traces: bool = False) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Run the forward pass."""
        self._last_layer_input = None
        if input_sequence.ndim != 3:
            raise ValueError(f'Expected input shape (B,T,C), got {tuple(input_sequence.shape)}')
        batch_size, time_steps, _ = input_sequence.shape
        device = input_sequence.device
        dtype = input_sequence.dtype
        weight = self.effective_input_weight()
        recurrent_weight = self.effective_recurrent_weight()
        x_post = torch.zeros(batch_size, self.output_size, device=device, dtype=dtype)
        y_post = torch.zeros(batch_size, self.output_size, device=device, dtype=dtype)
        prev_spike = torch.zeros(batch_size, self.output_size, device=device, dtype=dtype)
        threshold = self.effective_threshold().to(device=device, dtype=dtype)
        record_raw_membrane = (not self.emit_spike) and (not self.reset_enabled)

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

        membrane_steps: list[torch.Tensor] | None = [] if return_traces else None
        layer_input_steps: list[torch.Tensor] | None = [] if return_traces else None
        spike_steps: list[torch.Tensor] = []
        input_current_sequence = torch.matmul(input_sequence, weight.t())
        for time_index in range(time_steps):
            current = input_current_sequence[:, time_index, :]
            if recurrent_weight is not None:
                current = current + prev_spike @ recurrent_weight.t()

            x_pre = rho.unsqueeze(0) * (cos_phi.unsqueeze(0) * x_post - sin_phi.unsqueeze(0) * y_post) + beta_x.unsqueeze(0) * current
            y_pre = rho.unsqueeze(0) * (sin_phi.unsqueeze(0) * x_post + cos_phi.unsqueeze(0) * y_post) + beta_y.unsqueeze(0) * current
            membrane_signal = x_pre - threshold.unsqueeze(0)
            if self.emit_spike:
                spike = surrogate_spike(membrane_signal)
            else:
                spike = torch.zeros_like(membrane_signal)

            if self.reset_enabled:
                if self.reset_mode == 'soft_reset':
                    x_post = x_pre - threshold.unsqueeze(0) * spike
                    y_post = y_pre
                else:
                    keep = 1.0 - spike
                    x_post = x_pre * keep
                    y_post = y_pre * keep
            else:
                x_post = x_pre
                y_post = y_pre

            if return_traces and membrane_steps is not None and layer_input_steps is not None:
                layer_input_steps.append(current)
                membrane_steps.append(x_pre if record_raw_membrane else membrane_signal)
            spike_steps.append(spike)
            prev_spike = spike

        membrane_seq = torch.stack(membrane_steps, dim=1) if membrane_steps is not None else None
        self._last_layer_input = torch.stack(layer_input_steps, dim=1) if layer_input_steps is not None else None
        spike_seq = torch.stack(spike_steps, dim=1)
        return membrane_seq, spike_seq

    def filter_stats_vectors(self) -> dict[str, torch.Tensor]:
        """Handle ``filter stats vectors`` for the ``RF_neuron`` module."""
        return {
            'damping': self.effective_damping_magnitude().detach(),
            'center_frequency': self.f_cyc_per_sample().detach(),
        }



__all__ = ['RFLayer']
