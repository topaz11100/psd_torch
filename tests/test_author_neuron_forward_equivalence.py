import importlib.util
import types
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')

from src.model.snn_builder import SpikGRUCellBlock
from src.neurons.DH_SNN_neuron import DHSNNLayer
from src.neurons.D_RF_neuron import DRFLayer
from src.neurons.TC_LIF_neuron import TCLIFLayer
from src.neurons.TS_LIF_neuron import TSLIFLayer
from src.neurons._common import logit, surrogate_spike
from src.neurons._origin_imports import load_tc_lif_module, load_ts_lif_module


class _ProjectSurrogate:
    def __call__(self, x, *args, **kwargs):
        del args, kwargs
        return surrogate_spike(x)


def _assert_close(actual, expected, *, atol=1.0e-6, rtol=1.0e-6):
    assert torch.allclose(actual, expected, atol=atol, rtol=rtol), (actual - expected).abs().max().item()


def test_tclif_step_forward_matches_checked_in_origin_node_with_same_surrogate():
    torch.manual_seed(1101)
    batch_size, units = 3, 5
    current = torch.randn(batch_size, units)
    v1 = torch.randn(batch_size, units)
    v2 = torch.randn(batch_size, units)
    previous_spike = torch.zeros(batch_size, units)
    threshold = torch.tensor(0.8)
    gamma = torch.tensor(0.3)
    decay0 = torch.tensor(0.25)
    decay1 = torch.tensor(0.65)

    layer = TCLIFLayer(2, units, v_threshold=float(threshold), gamma=float(gamma), recurrent=False)
    project_state = layer._step_impl(current, v1.clone(), v2.clone(), previous_spike, None, decay0, decay1, threshold, gamma)

    origin = load_tc_lif_module().TCLIFNode(
        v_threshold=float(threshold),
        surrogate_function=_ProjectSurrogate(),
        gamma=float(gamma),
        hard_reset=False,
    )
    origin.decay_factor.data = torch.tensor([[torch.logit(decay0), torch.logit(decay1)]])
    origin.names['v1'] = v1.clone()
    origin.names['v2'] = v2.clone()
    origin.v = v2.clone()
    origin_spike = origin(current)

    _assert_close(project_state[2], origin_spike)
    _assert_close(project_state[0], origin.names['v1'])
    _assert_close(project_state[1], origin.names['v2'])


def test_tslif_step_forward_matches_checked_in_origin_node_with_same_surrogate():
    torch.manual_seed(1102)
    batch_size, units = 3, 5
    current = torch.randn(batch_size, units)
    v1 = torch.randn(batch_size, units)
    v2 = torch.randn(batch_size, units)
    previous_spike = torch.zeros(batch_size, units)
    decay = torch.tensor([0.1, 0.2, 0.3, 0.4])
    kk = torch.tensor(0.6)
    yy = torch.tensor(0.5)
    alpha_s = torch.randn(1, units)
    alpha_l = torch.randn(1, units)
    threshold = torch.tensor(0.8)
    gamma = torch.tensor(0.35)

    layer = TSLIFLayer(2, units, v_threshold=float(threshold), gamma=float(gamma), recurrent=False)
    project_state = layer._step_impl(
        current,
        v1.clone(),
        v2.clone(),
        previous_spike,
        None,
        decay,
        kk,
        yy,
        alpha_s,
        alpha_l,
        threshold,
        gamma,
    )

    origin = load_ts_lif_module().TSLIFNode(
        v_threshold=float(threshold),
        surrogate_function=_ProjectSurrogate(),
        gamma=float(gamma),
        hard_reset=False,
    )
    origin.decay_factor.data = decay.clone()
    origin.kk.data = kk.clone().reshape_as(origin.kk)
    origin.yy.data = yy.clone().reshape_as(origin.yy)
    origin.alpha_s = torch.nn.Parameter(alpha_s.clone())
    origin.alpha_l = torch.nn.Parameter(alpha_l.clone())
    origin.names['v1'] = v1.clone()
    origin.names['v2'] = v2.clone()
    origin.v = v2.clone()
    origin.v_s = v1.clone()
    origin_spike = origin(current)

    _assert_close(project_state[2], origin_spike)
    _assert_close(project_state[0], origin.names['v1'])
    _assert_close(project_state[1], origin.names['v2'])


@pytest.mark.parametrize('recurrent', [False, True])
def test_dhsnn_forward_matches_origin_training_protocol_after_mask_application(recurrent):
    torch.manual_seed(1103 + int(recurrent))
    batch_size, time_steps, input_dim, output_dim, branch = 2, 4, 3, 5, 4
    layer = DHSNNLayer(input_dim, output_dim, recurrent=recurrent, branch=branch, v_threshold=0.5)
    origin = layer.layer
    if hasattr(origin, 'apply_mask'):
        origin.apply_mask()
    x = torch.randn(batch_size, time_steps, input_dim)

    project_mem_signal, project_spike = layer(x, return_traces=True)

    origin.mem = torch.zeros(batch_size, output_dim)
    origin.spike = torch.zeros(batch_size, output_dim)
    origin.d_input = torch.zeros(batch_size, output_dim, branch)
    origin.v_th = torch.ones(batch_size, output_dim) * 0.5
    origin_mem = []
    origin_spike = []
    for t in range(time_steps):
        mem_t, spike_t = origin(x[:, t, :])
        origin_mem.append(mem_t.clone())
        origin_spike.append(spike_t.clone())
    origin_mem_signal = torch.stack(origin_mem, dim=1) - 0.5
    origin_spike = torch.stack(origin_spike, dim=1)

    _assert_close(project_mem_signal, origin_mem_signal)
    _assert_close(project_spike, origin_spike)


def test_drf_forward_uses_origin_birfmodel_spike_path_when_threshold_is_origin_default():
    torch.manual_seed(1104)
    batch_size, time_steps, hidden_size, branch = 2, 6, 4, 3
    layer = DRFLayer(hidden_size, hidden_size, branch=branch, v_threshold=1.0)
    x = torch.randn(batch_size, time_steps, hidden_size)
    projected_bht = x.transpose(1, 2).contiguous()

    expected_spike_bht = layer.origin(projected_bht)
    _mem, project_spike_bth = layer(x, return_traces=True)

    _assert_close(project_spike_bth.transpose(1, 2), expected_spike_bht)


def test_spikegru_vanilla_forward_matches_checked_in_origin_single_gate_layer_after_alpha_mapping():
    torch.manual_seed(1105)
    origin_path = Path('Origin/spikegru/SpikGRU+imagemodel/layers.py')
    spec = importlib.util.spec_from_file_location('origin_spikegru_layers_equivalence', origin_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    batch_size, time_steps, input_dim, hidden_size = 2, 5, 3, 4
    origin = module.GRUlayer(types.SimpleNamespace(), input_dim, hidden_size, ann=False, ternact=False, twogates=False)
    block = SpikGRUCellBlock(input_dim, hidden_size, v_th=1.0)
    with torch.no_grad():
        block.input_to_gate.weight.copy_(origin.wz.weight)
        block.input_to_gate.bias.copy_(origin.wz.bias)
        block.input_to_candidate.weight.copy_(origin.wi.weight)
        block.input_to_candidate.bias.copy_(origin.wi.bias)
        block.hidden_to_gate.weight.copy_(origin.uz.weight)
        block.hidden_to_candidate.weight.copy_(origin.ui.weight)
        block.hidden_to_candidate.bias.copy_(origin.ui.bias)
        block.alpha_raw.copy_(logit(origin.alpha.detach()))

    x = torch.randn(batch_size, time_steps, input_dim)
    hidden, current_state, previous_spike = block.initial_state(batch_size, device=x.device, dtype=x.dtype)
    gate_input_sequence = block.input_to_gate(x)
    candidate_input_sequence = block.input_to_candidate(x)
    project_spikes = []
    for t in range(time_steps):
        hidden, current_state, spike_t, _gate_t, previous_spike = block.step(
            gate_input_t=gate_input_sequence[:, t, :],
            candidate_input_t=candidate_input_sequence[:, t, :],
            hidden=hidden,
            current_state=current_state,
            previous_spike=previous_spike,
        )
        project_spikes.append(spike_t)
    project_spikes = torch.stack(project_spikes, dim=1)
    origin_spikes = origin(x)

    _assert_close(project_spikes, origin_spikes)
