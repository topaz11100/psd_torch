import pytest
from psd_snn.config.specs import ExperimentConfig, ModelSpec, ConstraintSpec, ScenarioSpec, validate_config

def test_apply_to_output_error():
    cfg = ExperimentConfig(model=ModelSpec(constraints=ConstraintSpec(scenario=ScenarioSpec(scenario='clip', apply_to_output=True))))
    with pytest.raises(ValueError):
        validate_config(cfg)
