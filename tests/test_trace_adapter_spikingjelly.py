import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('spikingjelly')

from psd_snn.config.specs import ModelSpec, TopologySpec, CellSpec, ReadoutSpec
from psd_snn.models.mlp.builder import build_mlp_stack_model
from psd_snn.analysis.trace.adapter import TraceAdapter


def test_trace_adapter_reset_and_bt_layout():
    spec = ModelSpec(topology=TopologySpec(kind='mlp_stack', input_dim=3, hidden_widths=[4], output_dim=2), cell=CellSpec(kind='lif', reset_mode='soft', recurrent=True), readout=ReadoutSpec(kind='final_mem'))
    m = build_mlp_stack_model(spec)
    ad = TraceAdapter(m)
    x = torch.randn(2,5,3)
    logits, traces = ad.run_with_trace(x, probe_family='label_single', label='0')
    assert logits.shape == (2,2)
    assert all(len(t.tensor.shape) >= 3 for t in traces)
