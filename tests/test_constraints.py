import pytest
torch = pytest.importorskip('torch')

from src.model.constraints import (
    ConstraintConfig,
    default_band_neuron_ends,
    group_ids_from_ends,
    layer_constraint_for_hidden_index,
    normalize_constraint_mode,
    parse_band_neuron_ends,
    resolve_constraint_plan,
    validate_lif_clip_edges,
    validate_rf_clip_edges,
)
from src.model.model_registry import canonicalize_model_token


def test_normalize_constraint_mode():
    assert normalize_constraint_mode(' none ') == 'none'
    assert normalize_constraint_mode('clip') == 'clip'
    assert normalize_constraint_mode('Structure') == 'structure'
    assert normalize_constraint_mode('clipstructure') == 'clipstructure'
    assert normalize_constraint_mode('clip_structure') == 'clipstructure'
    with pytest.raises(ValueError):
        normalize_constraint_mode('x')


def test_validate_clip_edges():
    assert validate_lif_clip_edges([0.0, 0.5, 1.0]) == [0.0, 0.5, 1.0]
    for bad in ([-0.1, 0.5], [0.0, 0.5, 1.1], [0.0, 0.5, 0.5], [0.5]):
        with pytest.raises(ValueError):
            validate_lif_clip_edges(bad)
    assert validate_rf_clip_edges([0.0, 0.25, 0.5]) == [0.0, 0.25, 0.5]
    for bad in ([0.0, 0.6], [0.0, 0.25, 0.25], [0.2]):
        with pytest.raises(ValueError):
            validate_rf_clip_edges(bad)


def test_band_ends_and_groups():
    assert default_band_neuron_ends([8], 4) == ['2,4,6']
    with pytest.raises(ValueError):
        default_band_neuron_ends([3], 4)
    parsed = parse_band_neuron_ends([8, 8], ['2,4,6', '2,4,6'], 4)
    assert parsed == [[2, 4, 6], [2, 4, 6]]
    with pytest.raises(ValueError):
        parse_band_neuron_ends([8], ['2,4,6', '2,4,6'], 4)
    with pytest.raises(ValueError):
        parse_band_neuron_ends([8], ['2,4'], 4)
    with pytest.raises(ValueError):
        parse_band_neuron_ends([8], ['2,2,6'], 4)
    with pytest.raises(ValueError):
        parse_band_neuron_ends([8], ['2,4,8'], 4)
    with pytest.raises(ValueError):
        parse_band_neuron_ends([8], ['0,4,6'], 4)
    assert group_ids_from_ends(8, [2, 4, 6], 4) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_layer_masks_and_bounds():
    spec = canonicalize_model_token('lif_R_soft_fixed')
    cfg = ConstraintConfig(mode='clipstructure', alpha_clip_edges=(0.0, 0.5, 1.0), band_neuron_ends=('4', '4'), tear=1)
    plan = resolve_constraint_plan(spec, [8, 8], cfg)
    lc0 = layer_constraint_for_hidden_index(plan, 0, 8, 8, True)
    assert lc0.input_mask is None
    assert lc0.recurrent_mask is not None and tuple(lc0.recurrent_mask.shape) == (8, 8)
    low, high = lc0.lif_alpha_bounds
    assert torch.allclose(low[:4], torch.zeros(4)) and torch.allclose(high[:4], torch.full((4,), 0.5))

    lc1 = layer_constraint_for_hidden_index(plan, 1, 8, 8, True)
    assert tuple(lc1.input_mask.shape) == (8, 8)
    assert tuple(lc1.recurrent_mask.shape) == (8, 8)
    assert float(lc1.input_mask[0, 0]) == 1.0 and float(lc1.input_mask[0, 5]) == 0.0

    rf_spec = canonicalize_model_token('rf_soft_fixed')
    rf_plan = resolve_constraint_plan(rf_spec, [8], ConstraintConfig(mode='clip', w_clip_edges=(0.0, 0.25, 0.5), band_neuron_ends=('4',), tear=1))
    rf_lc = layer_constraint_for_hidden_index(rf_plan, 0, 8, 8, False)
    low, high = rf_lc.rf_frequency_bounds
    assert torch.allclose(low[:4], torch.zeros(4)) and torch.allclose(high[:4], torch.full((4,), 0.25))
