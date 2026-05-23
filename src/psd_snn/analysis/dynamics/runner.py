from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DynamicsStats:
    parameter_stats: Dict[str, Any]
    internal_state_stats: Dict[str, Any]

def collect_parameter_vectors(model) -> Dict[str, Any]:
    out = {}
    scenario = getattr(model, 'scenario', 'none')
    constraint_hash = getattr(model, 'constraint_hash', None)
    for name, module in model.named_modules():
        fn = getattr(module, 'analysis_parameter_vectors', None)
        if callable(fn):
            for k, meta in fn().items():
                key = f'{name}.{k}' if name else k
                vals = meta['values']
                out[key] = {
                    'layer_name': name,
                    'parameter_name': meta.get('name', k),
                    'role': meta.get('role', ''),
                    'unit': meta.get('unit', ''),
                    'values': vals,
                    'shape': tuple(vals.shape),
                    'trainable': bool(meta.get('trainable', False)),
                    'lower_bound': meta.get('lower_bound'),
                    'upper_bound': meta.get('upper_bound'),
                    'group_ids': meta.get('group_ids'),
                    'scenario': scenario,
                    'constraint_hash': constraint_hash,
                    'bounds_source': 'clip' if meta.get('lower_bound') is not None else None,
                }
    return out

def analyze_dynamics(parameter_stats: Dict[str, Any], internal_state_stats: Dict[str, Any]) -> DynamicsStats:
    return DynamicsStats(parameter_stats=parameter_stats, internal_state_stats=internal_state_stats)
