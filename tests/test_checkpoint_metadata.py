from psd_snn.config.specs import ExperimentConfig, ModelSpec, ConstraintSpec, ScenarioSpec, StructureSpec, ClipSpec
from psd_snn.models.checkpoint_metadata import canonical_config_dict, stable_hash

def test_structclip_canonicalized_for_hash():
    c1 = ExperimentConfig(model=ModelSpec(constraints=ConstraintSpec(scenario=ScenarioSpec(scenario='structclip'), structure=StructureSpec(enabled=True), clip=ClipSpec(enabled=True))))
    c2 = ExperimentConfig(model=ModelSpec(constraints=ConstraintSpec(scenario=ScenarioSpec(scenario='clipstructure'), structure=StructureSpec(enabled=True), clip=ClipSpec(enabled=True))))
    assert stable_hash(canonical_config_dict(c1)) == stable_hash(canonical_config_dict(c2))
