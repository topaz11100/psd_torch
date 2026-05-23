from __future__ import annotations
from dataclasses import dataclass
import hashlib, json

@dataclass
class ParameterBounds:
    name: str
    lower: list[float] | float
    upper: list[float] | float
    unit: str
    group_ids: list[int] | None
    source: str
    parameter_role: str
    trainable: bool | None = None

@dataclass
class CellBounds:
    threshold_bounds: ParameterBounds | None = None
    lif_alpha_bounds: ParameterBounds | None = None
    rf_frequency_bounds: ParameterBounds | None = None
    rf_damping_bounds: ParameterBounds | None = None

@dataclass
class LayerConstraintPlan:
    layer_index: int
    layer_name: str
    width: int
    group_ids: list[int] | None
    feedforward_mask: list[list[float]] | None
    recurrent_mask: list[list[float]] | None
    cell_bounds: CellBounds | None
    feedforward_mask_applied: bool
    recurrent_mask_applied: bool
    scenario: str
    constraint_hash: str

@dataclass
class ConstraintPlan:
    scenario: str
    layers: list[LayerConstraintPlan]
    constraint_hash: str

def hash_constraint_dict(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:16]

def _group_ids(width, group_count=1, band_neuron_ends=None):
    if band_neuron_ends:
        prev=0; gids=[0]*width
        for gi,e in enumerate(band_neuron_ends+[width]):
            if e<=prev or e>width: raise ValueError('invalid band_neuron_ends')
            for i in range(prev,e): gids[i]=gi
            prev=e
        return gids
    return [min(group_count-1,(i*group_count)//width) for i in range(width)]

def _mask(tg, sg): return [[1.0 if a==b else 0.0 for b in sg] for a in tg]

def build_constraint_plan(hidden_widths, input_dim, recurrent, constraint_spec):
    sc = constraint_spec['scenario']['scenario'] if isinstance(constraint_spec.get('scenario'), dict) else constraint_spec.get('scenario','none')
    st = constraint_spec.get('structure',{}); cl = constraint_spec.get('clip',{})
    h = hash_constraint_dict(constraint_spec)
    layers=[]; prev_g=None; ig = st.get('input_group_ids')
    for li,w in enumerate(hidden_widths):
        gids = _group_ids(w, st.get('group_count',1), (st.get('band_neuron_ends') or [None]*len(hidden_widths))[li] if st.get('band_neuron_ends') else None)
        ff=None
        if sc in {'structure','clipstructure'} and li>0: ff=_mask(gids, prev_g)
        if sc in {'structure','clipstructure'} and li==0 and st.get('apply_first_layer',False):
            if ig is None: raise ValueError('apply_first_layer=true requires input_group_ids')
            ff=_mask(gids, ig)
        rm = _mask(gids,gids) if recurrent and sc in {'structure','clipstructure'} and st.get('apply_recurrent',True) else None
        cb=None
        if sc in {'clip','clipstructure'}:
            cb=CellBounds(
                threshold_bounds=ParameterBounds('threshold',*(cl['threshold_bounds'][0]),'membrane',gids if sc=='clipstructure' else None,'threshold','firing_threshold') if cl.get('threshold_bounds') else None,
                lif_alpha_bounds=ParameterBounds('lif_alpha',*(cl['lif_alpha_bounds'][0]),'dimensionless',gids if sc=='clipstructure' else None,'clip','lif_membrane_decay') if cl.get('lif_alpha_bounds') else None,
                rf_frequency_bounds=ParameterBounds('rf_frequency',*(cl['rf_frequency_bounds'][0]),'cycle_per_step',gids if sc=='clipstructure' else None,'clip','rf_frequency') if cl.get('rf_frequency_bounds') else None,
                rf_damping_bounds=ParameterBounds('rf_damping',*(cl['rf_damping_bounds'][0]),'per_step',gids if sc=='clipstructure' else None,'clip','rf_damping') if cl.get('rf_damping_bounds') else None,
            )
        layers.append(LayerConstraintPlan(li,f'hidden_{li}',w,gids,ff,rm,cb,ff is not None,rm is not None,sc,h))
        prev_g=gids
    return ConstraintPlan(sc,layers,h)
