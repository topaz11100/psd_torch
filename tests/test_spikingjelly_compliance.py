"""Regression tests for SpikingJelly API alignment."""

from __future__ import annotations

import pytest

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from src.neurons import _common


@pytest.mark.skipif(torch is None, reason='torch not installed')
def test_surrogate_backend_prefers_spikingjelly_when_available():
    if not _common.SPIKINGJELLY_AVAILABLE:
        pytest.skip('SpikingJelly is not installed in this test image.')
    assert _common.surrogate_backend_name() == 'spikingjelly.activation_based.surrogate.Sigmoid'


@pytest.mark.skipif(torch is None, reason='torch not installed')
def test_vanilla_layers_expose_spikingjelly_reset_contract():
    from src.neurons.IF_neuron import IFLayer
    from src.neurons.LIF_neuron import LIFLayer
    from src.neurons.RF_neuron import RFLayer

    for cls in (IFLayer, LIFLayer, RFLayer):
        layer = cls(3, 4)
        assert getattr(layer, 'step_mode') == 'm'
        assert 'm' in getattr(layer, 'supported_step_mode')
        layer._last_layer_input = torch.ones(1)
        layer.reset_state()
        assert layer._last_layer_input is None


@pytest.mark.skipif(torch is None, reason='torch not installed')
def test_training_reset_uses_spikingjelly_reset_net_first(monkeypatch):
    from src.model import training

    calls = {'sj': 0, 'custom': 0}

    class Dummy(torch.nn.Module):
        def reset_state(self):
            calls['custom'] += 1

    monkeypatch.setattr(training, 'reset_spikingjelly_state', lambda module: calls.__setitem__('sj', calls['sj'] + 1) or True)
    training._reset_stateful_model(Dummy())
    assert calls == {'sj': 1, 'custom': 0}

    monkeypatch.setattr(training, 'reset_spikingjelly_state', lambda module: calls.__setitem__('sj', calls['sj'] + 1) or False)
    training._reset_stateful_model(Dummy())
    assert calls == {'sj': 2, 'custom': 1}
