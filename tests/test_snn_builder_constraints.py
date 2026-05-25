import pytest
import torch

from src.model.constraints import ConstraintConfig
from src.model.snn_builder import build_snn_classifier


def _forward(model):
    x = torch.randn(2, 16, 8)
    out = model(x, capture_hidden=True)
    assert out.output_record.membrane.shape == (2, 16, 3)
    return out


def test_lif_clip_smoke_and_output_unconstrained():
    model = build_snn_classifier(
        model_token='lif_soft_fixed', input_dim=8, sequence_length=16, num_classes=3, hidden_sizes=[8, 8], v_th=1.0,
        constraint_config=ConstraintConfig(mode='clip', alpha_clip_edges=(0.0, 0.5, 1.0), band_neuron_ends=('4', '4'), tear=1),
    )
    _forward(model)
    assert model.hidden_layers[0].alpha_lower is not None
    assert torch.allclose(model.output_layer.alpha_lower, torch.zeros_like(model.output_layer.alpha_lower))
    assert torch.allclose(model.output_layer.alpha_upper, torch.ones_like(model.output_layer.alpha_upper))
    assert model.output_layer.input_mask is not None and torch.allclose(model.output_layer.input_mask, torch.ones_like(model.output_layer.input_mask))


def test_rf_clip_smoke():
    model = build_snn_classifier(
        model_token='rf_soft_fixed', input_dim=8, sequence_length=16, num_classes=3, hidden_sizes=[8, 8], v_th=1.0,
        constraint_config=ConstraintConfig(mode='clip', w_clip_edges=(0.0, 0.25, 0.5), band_neuron_ends=('4', '4'), tear=1),
    )
    _forward(model)
    assert model.hidden_layers[0].freq_lower is not None
    assert model.hidden_layers[0].damping_lower == 0.1 and model.hidden_layers[0].damping_upper == 1.0


def test_structure_and_tear_behavior():
    model = build_snn_classifier(
        model_token='lif_R_soft_fixed', input_dim=8, sequence_length=16, num_classes=3, hidden_sizes=[8, 8, 8], v_th=1.0,
        constraint_config=ConstraintConfig(mode='structure', band_neuron_ends=('4', '4', '4'), tear=2),
    )
    _forward(model)
    # tear=2 => layer0 has no structure mask, but recurrent mask allowed
    assert torch.allclose(model.hidden_layers[0].input_mask, torch.ones_like(model.hidden_layers[0].input_mask))
    assert model.hidden_layers[0].recurrent_mask is not None
    # layer1/2 should get block feedforward mask
    assert not torch.allclose(model.hidden_layers[1].input_mask, torch.ones_like(model.hidden_layers[1].input_mask))
    assert not torch.allclose(model.hidden_layers[2].input_mask, torch.ones_like(model.hidden_layers[2].input_mask))


def test_rf_clipstructure_smoke_and_output_nonrecurrent():
    model = build_snn_classifier(
        model_token='rf_R_soft_fixed', input_dim=8, sequence_length=16, num_classes=3, hidden_sizes=[8, 8], v_th=1.0,
        constraint_config=ConstraintConfig(mode='clipstructure', w_clip_edges=(0.0, 0.25, 0.5), band_neuron_ends=('4', '4'), tear=1),
    )
    _forward(model)
    assert model.hidden_layers[0].freq_lower is not None
    assert model.hidden_layers[1].input_mask is not None
    assert getattr(model.output_layer, 'recurrent', False) is False
    assert getattr(model.output_layer, 'recurrent_weight', None) is None


def test_gradient_blocking_for_structure_masks():
    model = build_snn_classifier(
        model_token='lif_R_soft_fixed', input_dim=8, sequence_length=16, num_classes=3, hidden_sizes=[8, 8], v_th=1.0,
        constraint_config=ConstraintConfig(mode='structure', band_neuron_ends=('4', '4'), tear=1),
    )
    x = torch.randn(2, 16, 8)
    out = model(x, capture_hidden=False)
    loss = out.output_record.membrane.sum()
    loss.backward()
    layer2 = model.hidden_layers[1]
    mask = layer2.input_mask
    grad = layer2.input_weight.grad
    assert grad is not None
    assert torch.allclose(grad[mask == 0], torch.zeros_like(grad[mask == 0]))
    rec_mask = layer2.recurrent_mask
    rec_grad = layer2.recurrent_weight.grad
    assert rec_grad is not None
    assert torch.allclose(rec_grad[rec_mask == 0], torch.zeros_like(rec_grad[rec_mask == 0]))


def test_bounds_are_in_expected_group_ranges():
    lif = build_snn_classifier(
        model_token='lif_soft_fixed', input_dim=8, sequence_length=16, num_classes=3, hidden_sizes=[8, 8], v_th=1.0,
        constraint_config=ConstraintConfig(mode='clip', alpha_clip_edges=(0.0, 0.5, 1.0), band_neuron_ends=('4', '4'), tear=1),
    )
    alpha = lif.hidden_layers[0].effective_alpha().detach()
    assert torch.all(alpha[:4] >= 0.0) and torch.all(alpha[:4] <= 0.5)
    assert torch.all(alpha[4:] >= 0.5) and torch.all(alpha[4:] <= 1.0)

    rf = build_snn_classifier(
        model_token='rf_soft_fixed', input_dim=8, sequence_length=16, num_classes=3, hidden_sizes=[8, 8], v_th=1.0,
        constraint_config=ConstraintConfig(mode='clip', w_clip_edges=(0.0, 0.25, 0.5), band_neuron_ends=('4', '4'), tear=1),
    )
    freq = rf.hidden_layers[0].effective_frequency().detach()
    assert torch.all(freq[:4] >= 0.0) and torch.all(freq[:4] <= 0.25)
    assert torch.all(freq[4:] >= 0.25) and torch.all(freq[4:] <= 0.5)
    assert rf.hidden_layers[0].damping_lower == 0.1 and rf.hidden_layers[0].damping_upper == 1.0


@pytest.mark.parametrize('token,mode', [
    ('tc_lif', 'clip'),
    ('ts_lif', 'clip'),
    ('dh_snn_4', 'structure'),
    ('d_rf_4', 'clipstructure'),
    ('vgg11_lif_soft_fixed', 'clip'),
    ('resnet18_rf_soft_fixed', 'structure'),
    ('spikegru', 'structure'),
])
def test_unsupported_family_error(token, mode):
    with pytest.raises(ValueError):
        build_snn_classifier(
            model_token=token, input_dim=8, sequence_length=16, num_classes=3, hidden_sizes=[8, 8], v_th=1.0,
            constraint_config=ConstraintConfig(mode=mode, alpha_clip_edges=(0.0, 0.5, 1.0), w_clip_edges=(0.0, 0.25, 0.5), band_neuron_ends=('4', '4'), tear=1),
        )


def test_none_backward_compatibility():
    for token in ['lif_soft_fixed', 'rf_soft_fixed']:
        model = build_snn_classifier(model_token=token, input_dim=8, sequence_length=16, num_classes=3, hidden_sizes=[8, 8], v_th=1.0)
        _forward(model)
        model = build_snn_classifier(model_token=token, input_dim=8, sequence_length=16, num_classes=3, hidden_sizes=[8, 8], v_th=1.0, constraint_config=ConstraintConfig(mode='none'))
        _forward(model)


def test_constraint_metadata_and_state_dict_buffers():
    model = build_snn_classifier(
        model_token='lif_R_soft_fixed', input_dim=8, sequence_length=16, num_classes=3, hidden_sizes=[8, 8], v_th=1.0,
        constraint_config=ConstraintConfig(mode='clipstructure', alpha_clip_edges=(0.0, 0.5, 1.0), band_neuron_ends=('4', '4'), tear=1),
    )
    meta = model.model_metadata()['constraint_metadata']
    assert meta['constraint_mode'] == 'clipstructure'
    assert meta['structure_mask'] is True
    assert meta['clip_params'] is True
    assert meta['applies_to_output_layer'] is False
    assert meta['supported_scope'] == 'dense_hidden_layers_only'
    assert meta['tear'] == 1
    state = model.state_dict()
    assert any(k.endswith('alpha_lower') for k in state)
    assert any(k.endswith('alpha_upper') for k in state)
    assert any(k.endswith('input_mask') for k in state)
    assert any(k.endswith('recurrent_mask') for k in state)
