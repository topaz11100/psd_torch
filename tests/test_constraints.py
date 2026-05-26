import pytest

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from src.model.constraints import ConstraintConfig, layer_constraint_for_hidden_index, normalize_scenario_mode, resolve_constraint_plan
from src.model.model_registry import canonicalize_model_token


def test_scenario_mode_has_no_constraint_mode_alias():
    assert normalize_scenario_mode('clip_structure') == 'clipstructure'
    with pytest.raises(ValueError):
        normalize_scenario_mode('constraint_mode')


def test_clip_edges_3d_validation_and_metadata_lif():
    spec = canonicalize_model_token('lif_soft_fixed')
    cfg = ConstraintConfig(
        mode='clip',
        alpha_clip_edges=[[[0.1, 0.3], [0.3, 0.7]], [[0.2, 0.4], [0.4, 0.8]]],
        band_edge=[None, None],
        tear=2,
    )
    plan = resolve_constraint_plan(spec, [6, 6], cfg)
    assert plan.metadata['lif_clip_edges_per_layer'][0][0] == [0.1, 0.3]
    with pytest.raises(ValueError):
        resolve_constraint_plan(spec, [6, 6], ConstraintConfig(mode='clip', alpha_clip_edges=[[[0.3, 0.2]], [[0.2, 0.4]]], band_edge=[None, None]))


def test_rf_clip_edges_3d_validation():
    spec = canonicalize_model_token('rf_soft_fixed')
    cfg = ConstraintConfig(mode='clip', w_clip_edges=[[[0.01, 0.2], [0.2, 0.45]], [[0.05, 0.15], [0.15, 0.35]]], band_edge=[None, None], tear=1)
    plan = resolve_constraint_plan(spec, [6, 6], cfg)
    assert plan.metadata['rf_clip_edges_per_layer'][1][1] == [0.15, 0.35]
    with pytest.raises(ValueError):
        resolve_constraint_plan(spec, [6, 6], ConstraintConfig(mode='clip', w_clip_edges=[[[0.1, 0.6]], [[0.1, 0.2]]], band_edge=[None, None]))


def test_band_edge_assignment_and_validation():
    spec = canonicalize_model_token('lif_soft_fixed')
    cfg = ConstraintConfig(mode='clip', alpha_clip_edges=[[[0.1, 0.3], [0.3, 0.7], [0.7, 0.9]]], band_edge=[[5, 10]], tear=1)
    plan = resolve_constraint_plan(spec, [12], cfg)
    gids = plan.group_ids_per_hidden_layer[0]
    assert gids[:5] == [0] * 5 and gids[5:10] == [1] * 5 and gids[10:] == [2] * 2
    with pytest.raises(ValueError):
        resolve_constraint_plan(spec, [12], ConstraintConfig(mode='clip', alpha_clip_edges=[[[0.1, 0.3], [0.3, 0.7], [0.7, 0.9]]], band_edge=[[10, 5]]))


@pytest.mark.skipif(torch is None, reason='torch not installed')
def test_clipstructure_tear_applies_only_structure_masks_not_clip():
    spec = canonicalize_model_token('lif_R_soft_fixed')
    cfg = ConstraintConfig(
        mode='clipstructure',
        alpha_clip_edges=[[[0.1, 0.3], [0.3, 0.7]], [[0.2, 0.4], [0.4, 0.8]]],
        band_edge=[None, None],
        tear=2,
    )
    plan = resolve_constraint_plan(spec, [8, 8], cfg)
    lc0 = layer_constraint_for_hidden_index(plan, 0, 8, 8, True)
    assert lc0 is not None and lc0.lif_alpha_bounds is not None
    assert lc0.input_mask is None and lc0.recurrent_mask is None
    lc1 = layer_constraint_for_hidden_index(plan, 1, 8, 8, True)
    assert lc1.lif_alpha_bounds is not None
    assert lc1.input_mask is not None and lc1.recurrent_mask is not None


def test_if_mode_constraints():
    spec = canonicalize_model_token('if_soft_fixed')
    with pytest.raises(ValueError):
        resolve_constraint_plan(spec, [8], ConstraintConfig(mode='clip', alpha_clip_edges=[[[0.1, 0.4], [0.4, 0.9]]], band_edge=[None]))
    with pytest.raises(ValueError):
        resolve_constraint_plan(spec, [8], ConstraintConfig(mode='clipstructure', alpha_clip_edges=[[[0.1, 0.4], [0.4, 0.9]]], band_edge=[None]))
    plan = resolve_constraint_plan(spec, [8], ConstraintConfig(mode='structure', band_edge=[None], tear=1))
    assert plan.structure_mask is True
