import pytest
from pathlib import Path

torch = pytest.importorskip('torch')

from src.model.snn_builder import build_snn_classifier

_ORIGIN_REQUIRED_PATHS = [
    Path('Origin/neuron_model/TC-LIF A Two-Compartment Spiking Neuron Model for Long-Term Sequential Modelling/MNIST/spiking_neuron/TCLIF.py'),
    Path('Origin/neuron_model/TS-LIF A TEMPORAL SEGMENT SPIKING NEURON NETWORK FOR TIME SERIES FORECASTING/SeqSNN/network/snn/TSLIF_base.py'),
    Path('Origin/neuron_model/Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics/s-mnist/SNN_layers/spike_neuron.py'),
    Path('Origin/neuron_model/Dendritic Resonate-and-Fire Neuron for Effective and Efficient Long Sequence Modeling/models/layers.py'),
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
