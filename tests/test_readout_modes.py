import importlib.util

import pytest

HAS_TORCH = importlib.util.find_spec('torch') is not None

if HAS_TORCH:
    import torch
    from src.readout.readout import TemporalMembraneReadout, FinalMembraneReadout, MaxFireReadout, build_readout, canonicalize_readout_mode
    import src.model_training as mt


def test_readout_modes_collection_smoke():
    assert True


@pytest.mark.skipif(not HAS_TORCH, reason='torch not installed')
def test_canonicalize_readout_mode_max_rate_alias():
    assert canonicalize_readout_mode('max_fire') == 'max_fire'
    assert canonicalize_readout_mode('max_rate') == 'max_fire'
    assert mt._canonical_run_readout_mode('max_rate') == 'max_fire'


@pytest.mark.skipif(not HAS_TORCH, reason='torch not installed')
def test_temporal_membrane_uses_mean_logits_not_softmax_sum():
    m = torch.tensor([[[1.0, 0.0], [3.0, 2.0]]])
    s = torch.zeros_like(m)
    r = TemporalMembraneReadout()
    a = r.analyze_output_record(m, s)
    assert torch.allclose(a.scores, torch.tensor([[2.0, 1.0]]))


@pytest.mark.skipif(not HAS_TORCH, reason='torch not installed')
def test_final_membrane_last_timestep():
    m = torch.tensor([[[1.0, 5.0], [3.0, 2.0]]])
    s = torch.zeros_like(m)
    r = FinalMembraneReadout()
    a = r.analyze_output_record(m, s)
    assert torch.allclose(a.scores, torch.tensor([[3.0, 2.0]]))


@pytest.mark.skipif(not HAS_TORCH, reason='torch not installed')
def test_max_fire_uses_count_sum_and_alias():
    m = torch.zeros((1, 3, 2))
    spk = torch.tensor([[[1, 0], [1, 1], [0, 1]]], dtype=torch.float32)
    r = MaxFireReadout()
    a = r.analyze_output_record(m, spk)
    assert torch.allclose(a.scores, torch.tensor([[2.0, 2.0]]))
    assert isinstance(build_readout('max_fire', num_classes=2, sequence_length=3, device='cpu'), MaxFireReadout)
    assert isinstance(build_readout('max_rate', num_classes=2, sequence_length=3, device='cpu'), MaxFireReadout)
