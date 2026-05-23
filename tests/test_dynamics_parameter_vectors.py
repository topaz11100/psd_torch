import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('spikingjelly')

from psd_snn.config.specs import load_experiment_config
from psd_snn.models.mlp.builder import build_mlp_stack_model
from psd_snn.analysis.dynamics.runner import collect_parameter_vectors


def test_clipstructure_rf_metadata():
    cfg = load_experiment_config('tests/fixtures/configs/mlp_rf_clipstructure_recurrent.json')
    m = build_mlp_stack_model(cfg.model)
    vecs = collect_parameter_vectors(m)
    one = next(v for k,v in vecs.items() if 'damping_magnitude' in k)
    assert one['scenario'] == 'clipstructure'
    assert one['lower_bound'] is not None
