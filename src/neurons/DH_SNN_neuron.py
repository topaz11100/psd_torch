"""DH-SNN origin-code thin wrapper.

Source files:
- ``Origin/.../s-mnist/SNN_layers/spike_dense.py``
- ``Origin/.../s-mnist/SNN_layers/spike_rnn.py``
"""

from __future__ import annotations

import torch
from torch import nn

from src.neurons._origin_imports import load_dh_snn_modules


class DHSNNLayer(nn.Module):
    """Sequence wrapper around the released DH-SNN dense and recurrent layers."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        recurrent: bool = False,
        branch: int = 4,
        v_threshold: float = 1.0,
        emit_spike: bool = True,
        reset_enabled: bool = True,
    ) -> None:
        """Initialize the instance with the provided configuration."""
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.recurrent = bool(recurrent)
        self.branch = int(branch)
        self.v_threshold = float(v_threshold)
        self.emit_spike = bool(emit_spike)
        self.reset_enabled = bool(reset_enabled)

        _, dense_mod, rnn_mod = load_dh_snn_modules()
        self._origin_r_m = float(getattr(rnn_mod if self.recurrent else dense_mod, 'R_m', 1.0))
        if self.recurrent:
            self.layer = rnn_mod.spike_rnn_test_denri_wotanh_R(
                self.input_size,
                self.output_size,
                vth=self.v_threshold,
                branch=self.branch,
                device='cpu',
                bias=True,
            )
        else:
            self.layer = dense_mod.spike_dense_test_denri_wotanh_R(
                self.input_size,
                self.output_size,
                vth=self.v_threshold,
                branch=self.branch,
                device='cpu',
                bias=True,
            )

    def _prepare_state(self, batch_size: int, device: torch.device) -> None:
        """Internal helper that prepare state."""
        self.layer.device = str(device)
        self.layer.set_neuron_state(batch_size)
        if hasattr(self.layer, 'spike'):
            self.layer.spike = torch.zeros_like(self.layer.spike, device=device)
        if hasattr(self.layer, 'mem'):
            self.layer.mem = torch.zeros_like(self.layer.mem, device=device)
        if hasattr(self.layer, 'd_input'):
            self.layer.d_input = torch.zeros_like(self.layer.d_input, device=device)
        if hasattr(self.layer, 'v_th'):
            self.layer.v_th = torch.full_like(self.layer.v_th, self.v_threshold, device=device)
        if hasattr(self.layer, 'apply_mask'):
            self.layer.apply_mask()

    def _forward_without_spike_or_reset(self, input_sequence: torch.Tensor, *, return_traces: bool = False) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Run the exact origin current/membrane path without output spikes or reset updates."""
        batch_size, time_steps, _ = input_sequence.shape
        device = input_sequence.device
        dtype = input_sequence.dtype
        self._prepare_state(batch_size, device)
        membrane_steps: list[torch.Tensor] | None = [] if return_traces else None
        layer_input_steps: list[torch.Tensor] | None = [] if return_traces else None
        spike_steps: list[torch.Tensor] = []

        beta = torch.sigmoid(self.layer.tau_n).to(device=device, dtype=dtype)
        alpha = torch.sigmoid(self.layer.tau_m).to(device=device, dtype=dtype)
        pad = int(getattr(self.layer, 'pad', 0))
        r_m = float(self._origin_r_m)
        recurrent_size = int(self.output_size) if self.recurrent else 0

        for time_index in range(time_steps):
            input_t = input_sequence[:, time_index, :].to(dtype=dtype)
            padding = torch.zeros(batch_size, pad, device=device, dtype=dtype)
            if recurrent_size > 0:
                recurrent_input = torch.zeros(batch_size, recurrent_size, device=device, dtype=dtype)
                k_input = torch.cat((input_t, recurrent_input, padding), dim=1)
            else:
                k_input = torch.cat((input_t, padding), dim=1)
            dense_out = self.layer.dense(k_input.float()).reshape(-1, self.output_size, self.branch).to(dtype=dtype)
            self.layer.d_input = beta.unsqueeze(0) * self.layer.d_input.to(dtype=dtype) + (1.0 - beta).unsqueeze(0) * dense_out
            l_input = self.layer.d_input.sum(dim=2, keepdim=False)
            self.layer.mem = self.layer.mem.to(dtype=dtype) * alpha.unsqueeze(0) + (1.0 - alpha).unsqueeze(0) * r_m * l_input
            self.layer.spike = torch.zeros_like(self.layer.mem)
            if membrane_steps is not None and layer_input_steps is not None:
                layer_input_steps.append(l_input)
                membrane_steps.append(self.layer.mem)
            spike_steps.append(self.layer.spike)

        self._last_layer_input = torch.stack(layer_input_steps, dim=1) if layer_input_steps is not None else None
        membrane_seq = torch.stack(membrane_steps, dim=1) if membrane_steps is not None else None
        spike_seq = torch.stack(spike_steps, dim=1)
        return membrane_seq, spike_seq

    def forward(self, input_sequence: torch.Tensor, *, return_traces: bool = False) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Run the forward pass."""
        self._last_layer_input = None
        if input_sequence.ndim != 3:
            raise ValueError(f'Expected shape (B,T,C), got {tuple(input_sequence.shape)}')
        if (not self.emit_spike) and (not self.reset_enabled):
            return self._forward_without_spike_or_reset(input_sequence, return_traces=return_traces)
        batch_size, time_steps, _ = input_sequence.shape
        device = input_sequence.device
        self._prepare_state(batch_size, device)
        membrane_steps: list[torch.Tensor] | None = [] if return_traces else None
        layer_input_steps: list[torch.Tensor] | None = [] if return_traces else None
        spike_steps: list[torch.Tensor] = []

        for time_index in range(time_steps):
            input_t = input_sequence[:, time_index, :]
            mem, spike = self.layer(input_t)
            threshold = self.layer.v_th if hasattr(self.layer, 'v_th') else self.v_threshold
            membrane_signal = mem - threshold
            if not self.emit_spike:
                spike = torch.zeros_like(spike)
            if return_traces and membrane_steps is not None and layer_input_steps is not None:
                layer_input_steps.append(self.layer.d_input.sum(dim=2, keepdim=False))
                membrane_steps.append(membrane_signal)
            spike_steps.append(spike)
            if not self.reset_enabled and hasattr(self.layer, 'spike'):
                self.layer.spike = torch.zeros_like(self.layer.spike)

        self._last_layer_input = torch.stack(layer_input_steps, dim=1) if layer_input_steps is not None else None
        membrane_seq = torch.stack(membrane_steps, dim=1) if membrane_steps is not None else None
        spike_seq = torch.stack(spike_steps, dim=1)
        return membrane_seq, spike_seq

    def effective_input_weight(self) -> torch.Tensor:
        """Handle ``effective input weight`` for the ``DH_SNN_neuron`` module."""
        weight = self.layer.dense.weight
        if weight.ndim != 2:
            return weight
        input_dim = self.input_size
        trimmed = weight[:, :input_dim]
        if trimmed.shape[0] == self.output_size * self.branch:
            return trimmed.reshape(self.output_size, self.branch, input_dim).mean(dim=1)
        return trimmed

    def filter_stats_vectors(self) -> dict[str, torch.Tensor]:
        """Handle ``filter stats vectors`` for the ``DH_SNN_neuron`` module."""
        return {}



__all__ = ['DHSNNLayer']
