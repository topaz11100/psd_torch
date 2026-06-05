import pytest

torch = pytest.importorskip('torch')

import src.model_training as mt
import src.neurons._compile as neuron_compile
from src.model.snn_builder import build_snn_classifier


def _fake_compile(fn, **_kwargs):
    return fn


def _assert_model_compile_hook_applies(model, monkeypatch):
    mt._torch_module()
    monkeypatch.setattr(neuron_compile.torch, 'compile', _fake_compile, raising=True)
    monkeypatch.setattr(mt.torch, 'compile', _fake_compile, raising=True)
    compiled, applied, policy, kwargs = mt._maybe_compile_model(model, requested=True, device=torch.device('cpu'))
    assert kwargs['backend'] == 'eager'
    assert applied is True
    assert compiled is model or getattr(compiled, '_orig_mod', None) is model
    assert 'compile' in policy.lower()


def test_spikegru_compile_hook_and_forward_smoke(monkeypatch):
    model = build_snn_classifier(
        model_token='spikegru',
        input_dim=6,
        sequence_length=5,
        num_classes=3,
        hidden_sizes=[8],
        v_th=1.0,
    )
    _assert_model_compile_hook_applies(model, monkeypatch)
    out = model(torch.randn(2, 5, 6), capture_hidden=True)
    assert out.output_record.membrane.shape == (2, 5, 3)
    assert len(out.hidden_records) == 2


def test_spikeformer_alias_compile_hook_and_forward_smoke(monkeypatch):
    model = build_snn_classifier(
        model_token='spikeformer',
        input_dim=8,
        sequence_length=5,
        num_classes=3,
        hidden_sizes=[8],
        v_th=1.0,
    )
    _assert_model_compile_hook_applies(model, monkeypatch)
    out = model(torch.randn(1, 5, 8), capture_hidden=True)
    assert out.output_record.membrane.shape[-1] == 3
    assert len(out.hidden_records) >= 1


@pytest.mark.parametrize('token', ['resnet_lif_soft_fixed', 'vgg11_lif_soft_fixed'])
def test_fixed_cnn_backbones_compile_hook_and_forward_smoke(token: str, monkeypatch):
    model = build_snn_classifier(
        model_token=token,
        input_dim=2 * 16 * 16,
        sequence_length=1,
        num_classes=4,
        hidden_sizes=[8],
        input_shape=(1, 2, 16, 16),
        arch_spec='-',
        v_th=1.0,
    )
    _assert_model_compile_hook_applies(model, monkeypatch)
    out = model(torch.randn(1, 1, 2, 16, 16), capture_hidden=True)
    assert out.output_record.membrane.shape == (1, 1, 4)
    assert len(out.hidden_records) >= 1
