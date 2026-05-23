import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('spikingjelly')

from psd_snn.config.specs import ExperimentConfig
from psd_snn.models.mlp.builder import build_mlp_stack_model
from psd_snn.analysis.trace.adapter import TraceAdapter, TraceContext


def test_trace_context_injection_metadata():
    cfg=ExperimentConfig()
    model = build_mlp_stack_model(cfg.model)
    ta = TraceAdapter(model)
    x = torch.randn(2, 8, cfg.model.topology.input_dim)
    ctx = TraceContext(run_id='run1', checkpoint_epoch=1, split='test', scope='test_balanced_global', probe_family='balanced_global', sample_indices=[0,1], labels=[0,1], probe_manifest_id='pm1', exclusion_family='balanced_global')
    _, traces = ta.run_with_trace(x, probe_family='balanced_global', label='na', context=ctx)
    assert traces
    tr = traces[0]
    assert tr.run_id == 'run1'
    assert tr.checkpoint_epoch == 1
    assert tr.split == 'test'
    assert tr.probe_manifest_id == 'pm1'
