import pytest
from pathlib import Path

torch = pytest.importorskip('torch')

from src.model.snn_builder import build_snn_classifier

_ORIGIN_REQUIRED_PATHS = [
    Path('Origin/neuron_model/TC-LIF/MNIST/spiking_neuron/TCLIF.py'),
    Path('Origin/neuron_model/TS-LIF/SeqSNN/network/snn/TSLIF_base.py'),
    Path('Origin/neuron_model/DH-SNN/s-mnist/SNN_layers/spike_neuron.py'),
    Path('Origin/neuron_model/D-RF/models/layers.py'),
]
if not all(path.exists() for path in _ORIGIN_REQUIRED_PATHS):
    pytestmark = pytest.mark.skip(reason='Origin reference files for tc/ts/dh/d_rf wrappers are unavailable in this environment.')


def _build(token: str):
    model = build_snn_classifier(
        model_token=token,
        input_dim=8,
        sequence_length=16,
        num_classes=3,
        hidden_sizes=[12, 12],
        v_th=1.0,
    )
    x = torch.randn(2, 16, 8)
    out = model(x, capture_hidden=True)
    assert out.output_record.membrane.shape == (2, 16, 3)
    assert out.output_record.spike.shape == (2, 16, 3)
    assert len(out.hidden_records) == 2
    return model


@pytest.mark.parametrize('token', ['tc_lif', 'tc_lif_R', 'ts_lif', 'ts_lif_R', 'dh_snn_2', 'dh_snn_R_2', 'd_rf_2'])
def test_builder_smoke_new_tokens(token):
    _build(token)


@pytest.mark.parametrize('token', ['tc_lif_R', 'ts_lif_R', 'dh_snn_R_2'])
def test_output_layer_is_non_recurrent_for_recurrent_hidden_tokens(token):
    model = _build(token)
    output_layer = model.output_layer
    assert getattr(output_layer, 'recurrent', False) is False
    assert getattr(output_layer, 'recurrent_weight', None) is None


@pytest.mark.parametrize(
    ('token', 'canonical', 'family', 'recurrent', 'branch'),
    [
        ('tc_lif_R', 'tc_lif_R', 'tc_lif', True, None),
        ('dh_snn_R_8', 'dh_snn_R_8', 'dh_snn', True, 8),
        ('d_rf_4', 'd_rf_4', 'd_rf', False, 4),
    ],
)
def test_model_metadata_for_new_families(token, canonical, family, recurrent, branch):
    model = _build(token)
    meta = model.model_metadata()
    assert meta['canonical_model_token'] == canonical
    assert meta['family'] == family
    assert meta['recurrent'] is recurrent
    assert meta['branch'] == branch
    assert float(meta['v_th']) == 1.0


def test_spikegru_forward_smoke():
    model = build_snn_classifier(
        model_token='spikegru',
        input_dim=8,
        sequence_length=5,
        num_classes=3,
        hidden_sizes=[7],
        v_th=1.0,
    )
    x = torch.randn(2, 5, 8)
    out = model(x, capture_hidden=True)
    assert out.output_record.membrane.shape == (2, 5, 3)
    assert out.output_record.spike.shape == (2, 5, 3)
    assert len(out.hidden_records) == 2



def _mask_active_counts_by_row(layer):
    mask = getattr(layer.layer, 'mask', None)
    assert mask is not None
    return mask.detach().sum(dim=1)


@pytest.mark.parametrize('branch', [1, 2, 4, 8])
def test_dh_snn_dense_branch_mask_density(branch):
    model = build_snn_classifier(
        model_token=f'dh_snn_{branch}',
        input_dim=16,
        sequence_length=4,
        num_classes=3,
        hidden_sizes=[8, 8],
        v_th=1.0,
    )
    layer = model.hidden_layers[0]
    mask = layer.layer.mask.detach()
    padded_input = int(layer.layer.input_dim) + int(layer.layer.pad)
    assert mask.shape == (int(layer.output_size) * branch, padded_input)
    expected_active = padded_input // branch
    assert torch.all(_mask_active_counts_by_row(layer) == expected_active)
    assert float(mask.mean()) == pytest.approx(1.0 / branch)


@pytest.mark.parametrize('branch', [2, 4, 8])
def test_dh_snn_recurrent_branch_mask_density(branch):
    model = build_snn_classifier(
        model_token=f'dh_snn_R_{branch}',
        input_dim=16,
        sequence_length=4,
        num_classes=3,
        hidden_sizes=[8, 8],
        v_th=1.0,
    )
    layer = model.hidden_layers[0]
    mask = layer.layer.mask.detach()
    padded_input = int(layer.layer.input_dim) + int(layer.layer.output_dim) + int(layer.layer.pad)
    assert mask.shape == (int(layer.output_size) * branch, padded_input)
    expected_active = padded_input // branch
    assert torch.all(_mask_active_counts_by_row(layer) == expected_active)
    assert float(mask.mean()) == pytest.approx(1.0 / branch)


def test_spikegru_metadata_and_preallocated_trace_contract():
    model = build_snn_classifier(
        model_token='spikegru',
        input_dim=8,
        sequence_length=5,
        num_classes=3,
        hidden_sizes=[7],
        v_th=1.0,
    )
    meta = model.model_metadata()
    assert meta['model_profile'] == 'spikegru'
    assert meta['recurrent_layers'] == 2
    assert meta['gate_count'] == 1
    assert meta['sequence_buffer_mode'] == 'prealloc' if 'sequence_buffer_mode' in meta else True
    x = torch.randn(2, 5, 8)
    out = model(x, capture_hidden=True)
    for record in out.hidden_records:
        assert hasattr(record, 'i_current')
        assert hasattr(record, 'z_gate')
        assert record.membrane.is_contiguous()
        assert record.spike.is_contiguous()
