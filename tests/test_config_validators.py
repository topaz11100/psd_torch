from psd_snn.config.specs import ExperimentConfig, validate_config, TopologySpec, ModelSpec, CellSpec, SpectralSpec, ProbeSpec
import pytest

def test_mlp_hidden_required():
    cfg = ExperimentConfig(model=ModelSpec(topology=TopologySpec(kind='mlp_stack', hidden_widths=[])))
    with pytest.raises(ValueError):
        validate_config(cfg)

def test_spectral_and_probe_validation():
    cfg = ExperimentConfig(spectral=SpectralSpec(axis_policy='exact', userbin_reducer='mean', distance_metric='centered_l2'), probe=ProbeSpec(family='label_set'))
    validate_config(cfg)

def test_rf_reset_modes():
    cfg = ExperimentConfig(model=ModelSpec(cell=CellSpec(kind='rf', reset_mode='hard_state')))
    validate_config(cfg)
