from __future__ import annotations

import math

import torch

from src.model.model_registry import model_spec_from_config_fields
from src.neurons.RF_neuron import RFLayer


def test_rf_vanilla_config_defaults_to_no_reset() -> None:
    spec = model_spec_from_config_fields(
        neuron_type='rf',
        recurrent=False,
        reset=None,
        v_th=['fixed', 1.0],
        filter='train',
    )
    assert spec.family == 'rf'
    assert spec.reset_mode == 'no_reset'
    assert spec.reset_enabled is False
    assert spec.canonical_token == 'rf_none_fixed'


def test_rf_zoh_input_coefficients_match_exact_complex_integral() -> None:
    b = torch.tensor([-0.1, -0.5, -0.9], dtype=torch.float64)
    omega = torch.tensor([0.02, 0.7, math.pi - 0.03], dtype=torch.float64)
    beta_x, beta_y = RFLayer._zoh_input_coefficients(b, omega)
    exact = (torch.exp(b.to(torch.complex128) + 1j * omega.to(torch.complex128)) - 1.0) / (
        b.to(torch.complex128) + 1j * omega.to(torch.complex128)
    )
    assert torch.allclose(beta_x, exact.real, atol=1e-12, rtol=1e-12)
    assert torch.allclose(beta_y, exact.imag, atol=1e-12, rtol=1e-12)


def test_rf_no_reset_trace_and_no_trace_spikes_match() -> None:
    torch.manual_seed(7)
    layer = RFLayer(3, 5, recurrent=False, reset_mode='no_reset', filter_value=0.25)
    x = torch.randn(2, 11, 3)
    mem_seq, spike_seq_with_trace = layer(x, return_traces=True)
    no_trace_mem, spike_seq_no_trace = layer(x, return_traces=False)
    assert no_trace_mem is None
    assert mem_seq is not None
    assert mem_seq.shape == (2, 11, 5)
    assert spike_seq_with_trace.shape == (2, 11, 5)
    assert torch.allclose(spike_seq_with_trace, spike_seq_no_trace, atol=0.0, rtol=0.0)


def test_rf_fixed_filter_is_non_trainable_and_exactly_fixed() -> None:
    layer = RFLayer(4, 6, recurrent=False, reset_mode='no_reset', filter_value=0.25)
    assert layer.freq_raw.requires_grad is False
    assert torch.allclose(layer.effective_frequency(), torch.full((6,), 0.25), atol=1e-6, rtol=0.0)
