from __future__ import annotations
from psd_snn.config.specs import ModelSpec
from psd_snn.models.cells.factory import build_cell
from psd_snn.models.mlp.blocks import DenseTemporalBlock
from psd_snn.models.mlp.model import MLPStackModel
from psd_snn.models.constraints.planner import build_constraint_plan


def build_mlp_stack_model(spec: ModelSpec):
    t = spec.topology; c = spec.cell
    plan = build_constraint_plan(list(t.hidden_widths), t.input_dim, bool(c.recurrent), {'scenario': {'scenario': spec.constraints.scenario.scenario, 'apply_to_output': spec.constraints.scenario.apply_to_output}, 'structure': spec.constraints.structure.__dict__, 'clip': spec.constraints.clip.__dict__})
    dims = [t.input_dim] + list(t.hidden_widths)
    blocks = []
    for i in range(len(t.hidden_widths)):
        cell = build_cell(c, dims[i + 1], cell_bounds=lp.cell_bounds, group_ids=lp.group_ids, layer_index=i, layer_name=f'hidden_{i}')
        lp = plan.layers[i]
        blocks.append(DenseTemporalBlock(dims[i], dims[i + 1], cell, recurrent=bool(c.recurrent), feedforward_mask=lp.feedforward_mask, recurrent_mask=lp.recurrent_mask, layer_group_ids=lp.group_ids))
    out_cell = build_cell(c, t.output_dim, cell_bounds=None, group_ids=None, layer_index=len(t.hidden_widths), layer_name='output')
    output_block = DenseTemporalBlock(dims[-1], t.output_dim, out_cell, recurrent=False)
    model = MLPStackModel(blocks, output_block, readout_kind=spec.readout.kind)
    model.scenario = plan.scenario
    model.constraint_hash = plan.constraint_hash
    model.constraint_spec = {'scenario': {'scenario': spec.constraints.scenario.scenario, 'apply_to_output': spec.constraints.scenario.apply_to_output}, 'structure': spec.constraints.structure.__dict__, 'clip': spec.constraints.clip.__dict__}
    return model
