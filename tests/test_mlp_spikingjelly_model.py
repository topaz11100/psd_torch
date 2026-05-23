import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('spikingjelly')

from psd_snn.config.specs import ModelSpec, TopologySpec, CellSpec, ReadoutSpec
from psd_snn.models.mlp.builder import build_mlp_stack_model


def _spec(readout='final_mem', recurrent=False, kind='lif'):
    return ModelSpec(
        topology=TopologySpec(kind='mlp_stack', input_dim=3, hidden_widths=[5], output_dim=2),
        cell=CellSpec(kind=kind, recurrent=recurrent, reset_mode='soft' if kind!='rf' else 'threshold_only'),
        readout=ReadoutSpec(kind=readout),
    )


def test_final_readout_shapes_and_trace_and_logits_consistency():
    x = torch.randn(4,6,3)
    m_if = build_mlp_stack_model(_spec('final_if'))
    logits_if, traces_if = m_if(x, capture_trace=True)
    assert logits_if.shape == (4,2)
    assert any(t.series == 'logits' for t in traces_if)
    m_mem = build_mlp_stack_model(_spec('final_mem'))
    logits_mem, traces_mem = m_mem(x, capture_trace=True)
    assert logits_mem.shape == (4,2)
    out_mem = [t for t in traces_mem if t.layer_name == 'output' and t.series == 'output_membrane_pre'][-1]
    assert out_mem.tensor.shape[0] == 4


def test_recurrent_weight_presence():
    m = build_mlp_stack_model(_spec('final_mem', recurrent=True))
    assert m.blocks[0].recurrent_weight.shape == (5,5)
    m2 = build_mlp_stack_model(_spec('final_mem', recurrent=False))
    assert m2.blocks[0].recurrent_weight is None
