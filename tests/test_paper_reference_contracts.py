import pytest

torch = pytest.importorskip('torch')

from src.model.first_spike_loss import FirstSpikeLossAdapter
from src.model.snn_builder import SpikGRUCellBlock, build_snn_classifier
from src.neurons.TC_LIF_neuron import TCLIFLayer
from src.neurons.TS_LIF_neuron import TSLIFLayer
from src.neurons._common import surrogate_spike


def test_spikegru_step_matches_checked_in_origin_single_gate_equation():
    torch.manual_seed(0)
    block = SpikGRUCellBlock(input_dim=3, hidden_size=4, v_th=0.7)
    assert block.hidden_to_candidate.bias is not None
    assert block.hidden_to_gate.bias is None

    gate_input_t = torch.randn(2, 4)
    candidate_input_t = torch.randn(2, 4)
    hidden = torch.randn(2, 4)
    current_state = torch.randn(2, 4)
    previous_spike = torch.randn(2, 4)

    mem, current, spike, gate, returned_prev = block.step(
        gate_input_t=gate_input_t,
        candidate_input_t=candidate_input_t,
        hidden=hidden,
        current_state=current_state,
        previous_spike=previous_spike,
    )

    alpha = torch.sigmoid(block.alpha_raw).unsqueeze(0)
    expected_gate = torch.sigmoid(gate_input_t + block.hidden_to_gate(previous_spike))
    expected_drive = candidate_input_t + block.hidden_to_candidate(previous_spike)
    expected_current = alpha * current_state + expected_drive
    expected_mem = expected_gate * hidden + (1.0 - expected_gate) * expected_current - 0.7 * previous_spike
    expected_spike = surrogate_spike(expected_mem - 0.7)

    assert torch.allclose(gate, expected_gate)
    assert torch.allclose(current, expected_current)
    assert torch.allclose(mem, expected_mem)
    assert torch.allclose(spike, expected_spike)
    assert torch.allclose(returned_prev, expected_spike)


def test_tc_lif_step_matches_origin_tclifnode_equation():
    layer = TCLIFLayer(3, 4, recurrent=False, v_threshold=0.9, gamma=0.25)
    current = torch.randn(2, 4)
    v1 = torch.randn(2, 4)
    v2 = torch.randn(2, 4)
    prev = torch.zeros(2, 4)
    d0 = torch.tensor(0.3)
    d1 = torch.tensor(0.7)
    threshold = torch.tensor(0.9)
    gamma = torch.tensor(0.25)

    v1_next, v2_next, spike, layer_input, mem, signal = layer._step_impl(
        current, v1, v2, prev, None, d0, d1, threshold, gamma
    )

    expected_v1_pre = v1 - d0 * v2 + current
    expected_v2_pre = v2 + d1 * expected_v1_pre
    expected_spike = surrogate_spike(expected_v2_pre - threshold)
    assert torch.allclose(layer_input, current)
    assert torch.allclose(mem, expected_v2_pre)
    assert torch.allclose(signal, expected_v2_pre - threshold)
    assert torch.allclose(spike, expected_spike)
    assert torch.allclose(v1_next, expected_v1_pre - gamma * expected_spike)
    assert torch.allclose(v2_next, expected_v2_pre - threshold * expected_spike)


def test_ts_lif_step_matches_origin_tslifnode_equation():
    layer = TSLIFLayer(3, 4, recurrent=False, v_threshold=0.8, gamma=0.35)
    current = torch.randn(2, 4)
    v1 = torch.randn(2, 4)
    v2 = torch.randn(2, 4)
    prev = torch.zeros(2, 4)
    decay = torch.tensor([0.1, 0.2, 0.3, 0.4])
    kk = torch.tensor(0.6)
    yy = torch.tensor(0.5)
    alpha_s = torch.randn(1, 4)
    alpha_l = torch.randn(1, 4)
    threshold = torch.tensor(0.8)
    gamma = torch.tensor(0.35)

    v1_next, v2_next, spike, layer_input, mem, signal = layer._step_impl(
        current, v1, v2, prev, None, decay, kk, yy, alpha_s, alpha_l, threshold, gamma
    )

    expected_v1_pre = decay[0] * v1 + decay[1] * current - yy * v2
    expected_v2_pre = decay[2] * v2 + decay[3] * current - kk * expected_v1_pre
    expected_s_s = surrogate_spike(expected_v2_pre - threshold)
    expected_s_l = surrogate_spike(expected_v1_pre - threshold)
    expected_spike = alpha_s * expected_s_s + alpha_l * expected_s_l
    assert torch.allclose(layer_input, current)
    assert torch.allclose(mem, expected_v2_pre)
    assert torch.allclose(signal, expected_v2_pre - threshold)
    assert torch.allclose(spike, expected_spike)
    assert torch.allclose(v1_next, expected_v1_pre - expected_s_l * gamma)
    assert torch.allclose(v2_next, expected_v2_pre - expected_s_s * threshold)


def test_spikingssm_author_source_builds_and_exposes_origin_scope_metadata():
    model = build_snn_classifier(
        model_token='spikingssm',
        input_dim=4,
        sequence_length=5,
        num_classes=2,
        hidden_sizes=[8],
        v_th=1.0,
    )
    meta = model.model_metadata()
    assert meta['model_profile'] == 'spikingssm'
    assert meta['source_code_path'] == 'Origin/state_space_sd4/models/spike/ss4d.py'
    assert 'checked-in origin SpikingSSM/SS4D core' in meta['paper_definition_scope']
    assert 'src.models.nn.DropoutNd' in meta['origin_import_shims']
    out = model(torch.randn(1, 5, 4), capture_hidden=True)
    assert out.output_record.membrane.shape == (1, 5, 2)
    assert len(out.hidden_records) == 1
    assert out.hidden_records[0].membrane.shape == (1, 5, 400)


def test_spikformer_metadata_does_not_claim_paper_exact_runtime_under_stubs():
    model = build_snn_classifier(
        model_token='spikformer',
        input_dim=32,
        sequence_length=7,
        num_classes=3,
        hidden_sizes=[8],
        v_th=1.0,
    )
    meta = model.model_metadata()
    if meta['dependency_backend'] == 'fallback_stubs':
        assert meta['structure_variation'] == 'dependency_stub_runtime_smoke_only'
        assert 'not paper-exact' in meta['paper_definition_compliance']
    else:
        assert meta['structure_variation'] == 'none'
        assert meta['paper_definition_compliance'] == 'author_source_with_real_dependencies'


def test_first_spike_loss_uses_released_time_and_loss_modules():
    adapter = FirstSpikeLossAdapter(num_classes=3, sequence_length=6, device='cpu')
    output_membrane = torch.randn(2, 6, 3)
    output_spike = (torch.randn(2, 6, 3) > 0.5).float()
    analysis = adapter.analyze(output_membrane, output_spike)
    assert analysis.first_times.shape == (2, 3)
    assert analysis.firing_rate.shape == (2, 3)
    loss = adapter.loss_from_analysis(analysis, torch.tensor([0, 2]), training=True)
    assert loss.ndim == 0
    assert adapter.train_loss.__class__.__module__ == 'origin_first_spike_loss'
