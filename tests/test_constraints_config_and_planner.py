from psd_snn.config.specs import load_experiment_config, ExperimentConfig, ModelSpec, TopologySpec, CellSpec, ConstraintSpec, ScenarioSpec, StructureSpec, ClipSpec, validate_config
from psd_snn.models.constraints.planner import build_constraint_plan, hash_constraint_dict
import pytest


def test_structclip_alias_normalized():
    cfg = ExperimentConfig(model=ModelSpec(constraints=ConstraintSpec(scenario=ScenarioSpec(scenario='structclip'), structure=StructureSpec(enabled=True), clip=ClipSpec(enabled=True))))
    validate_config(cfg)
    assert cfg.model.constraints.scenario.scenario == 'clipstructure'


def test_fixed_topology_scenario_error():
    cfg = ExperimentConfig(model=ModelSpec(topology=TopologySpec(kind='vgg'), constraints=ConstraintSpec(scenario=ScenarioSpec(scenario='clip'), clip=ClipSpec(enabled=True))))
    with pytest.raises(ValueError): validate_config(cfg)


def test_planner_masks():
    spec={'scenario':{'scenario':'structure','apply_to_output':False},'structure':{'enabled':True,'group_count':2,'apply_recurrent':True},'clip':{'enabled':False}}
    p=build_constraint_plan([6,6],4,True,spec)
    assert len(p.layers[1].feedforward_mask) == 6 and len(p.layers[1].feedforward_mask[0]) == 6
    assert p.layers[0].feedforward_mask is None
    assert len(p.layers[0].recurrent_mask) == 6 and len(p.layers[0].recurrent_mask[0]) == 6


def test_hash_deterministic():
    a={'scenario':{'scenario':'structure'},'structure':{'enabled':True,'group_count':2},'clip':{'enabled':False}}
    b={'clip':{'enabled':False},'structure':{'group_count':2,'enabled':True},'scenario':{'scenario':'structure'}}
    assert hash_constraint_dict(a)==hash_constraint_dict(b)
