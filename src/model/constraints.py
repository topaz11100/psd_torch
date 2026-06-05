from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class ConstraintConfig:
    mode: str = 'none'
    w_clip_edges: Any = None
    alpha_clip_edges: Any = None
    band_edge: Any = None
    band_neuron_ends: Any = None  # legacy source-only alias for band_edge
    tear: int = 1


@dataclass
class LayerConstraint:
    input_mask: Any = None
    recurrent_mask: Any = None
    lif_alpha_bounds: Any = None
    rf_frequency_bounds: Any = None


@dataclass
class ResolvedConstraintPlan:
    config: ConstraintConfig
    enabled: bool
    structure_mask: bool
    clip_params: bool
    num_groups: int
    group_cumulative_ends_per_hidden_layer: list[list[int]] | None
    group_ids_per_hidden_layer: list[list[int]] | None
    metadata: dict[str, Any]


def normalize_scenario_mode(mode: Any) -> str:
    token = str(mode if mode is not None else 'none').strip().lower()
    if token == 'clip_structure':
        return 'clipstructure'
    if token in {'none', 'clip', 'structure', 'clipstructure'}:
        return token
    raise ValueError(f'Unsupported scenario_mode: {mode!r}')


def _normalize_layerwise_edges(edges: Any, *, lower: float, upper: float, name: str, num_layers: int) -> list[list[list[float]]] | None:
    if edges is None:
        return None
    if not isinstance(edges, (tuple, list)):
        raise ValueError(f'{name} must be a layer/group/bounds list.')
    per_layer = list(edges)
    if len(per_layer) != int(num_layers):
        raise ValueError(f'{name} layer count must match hidden layer count ({num_layers}).')
    out: list[list[list[float]]] = []
    for layer in per_layer:
        if not isinstance(layer, (tuple, list)) or len(layer) < 1:
            raise ValueError(f'{name} each layer must contain at least one group bound.')
        layer_out: list[list[float]] = []
        for bounds in layer:
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                raise ValueError(f'{name} each group must be [lower, upper].')
            lo, hi = float(bounds[0]), float(bounds[1])
            if not (lower <= lo < hi <= upper):
                raise ValueError(f'{name} group bound must satisfy {lower} <= lower < upper <= {upper}.')
            layer_out.append([lo, hi])
        out.append(layer_out)
    return out


def _normalize_band_edge(hidden_widths: Sequence[int], band_edge: Any, num_groups_per_layer: list[int]) -> list[list[int]]:
    widths = [int(v) for v in hidden_widths]
    if band_edge is None:
        band_edge = [None for _ in widths]
    if len(band_edge) != len(widths):
        raise ValueError('band_edge entry count must match hidden layer count.')
    out: list[list[int]] = []
    for i, (width, entry, groups) in enumerate(zip(widths, band_edge, num_groups_per_layer)):
        groups = int(groups)
        if groups < 1:
            raise ValueError(f'band_edge[{i}] requires at least one group.')
        if entry is None:
            if width < groups:
                raise ValueError(f'band_edge[{i}] cannot be null when hidden width {width} < groups {groups}.')
            out.append([round(width * g / groups) for g in range(1, groups)])
            continue
        vals = [int(v) for v in entry]
        if len(vals) != groups - 1:
            raise ValueError(f'band_edge[{i}] length must be groups-1={groups-1}.')
        if any(vals[j] >= vals[j + 1] for j in range(len(vals) - 1)):
            raise ValueError('band_edge boundaries must be strictly increasing.')
        if any(v <= 0 or v >= width for v in vals):
            raise ValueError('band_edge boundaries must satisfy 1 <= edge < hidden_width.')
        out.append(vals)
    return out


def _validate_edges(edges: Sequence[Any] | None, *, lower: float, upper: float, name: str) -> list[float]:
    if edges is None:
        raise ValueError(f'{name} is required.')
    values = [float(v) for v in edges]
    if len(values) < 2:
        raise ValueError(f'{name} must contain at least two edge values.')
    for i in range(len(values) - 1):
        if not values[i] < values[i + 1]:
            raise ValueError(f'{name} must be strictly increasing.')
    if values[0] < lower or values[-1] > upper:
        raise ValueError(f'{name} must be within [{lower}, {upper}].')
    return values


def validate_lif_clip_edges(edges: Sequence[Any] | None) -> list[float]:
    return _validate_edges(edges, lower=0.0, upper=1.0, name='alpha_clip_edges')


def validate_rf_clip_edges(edges: Sequence[Any] | None) -> list[float]:
    return _validate_edges(edges, lower=0.0, upper=0.5, name='w_clip_edges')


def default_band_neuron_ends(hidden_widths: Sequence[int], num_groups: int) -> list[str]:
    out: list[str] = []
    for width in [int(v) for v in hidden_widths]:
        ends: list[int] = []
        for g in range(1, int(num_groups)):
            end = round(width * g / int(num_groups))
            end = min(max(end, 1), width - 1)
            ends.append(int(end))
        if len(set(ends)) != len(ends):
            raise ValueError(f'Duplicate default band endpoints for width={width}, num_groups={num_groups}.')
        out.append(','.join(str(v) for v in ends))
    return out


def parse_band_neuron_ends(hidden_widths: Sequence[int], band_neuron_ends: Sequence[str] | None, num_groups: int) -> list[list[int]]:
    widths = [int(v) for v in hidden_widths]
    if band_neuron_ends is None:
        raise ValueError('band_neuron_ends is required for parsing.')
    entries = [str(v).strip() for v in band_neuron_ends]
    if len(entries) != len(widths):
        raise ValueError('band_neuron_ends entry count must match hidden layer count.')
    expected = int(num_groups) - 1
    parsed: list[list[int]] = []
    for width, entry in zip(widths, entries):
        parts = [int(p.strip()) for p in entry.split(',') if p.strip() != '']
        if len(parts) != expected:
            raise ValueError(f'Each band_neuron_ends entry must have {expected} endpoints.')
        for i in range(len(parts) - 1):
            if not parts[i] < parts[i + 1]:
                raise ValueError('band_neuron_ends endpoints must be strictly increasing.')
        for endpoint in parts:
            if endpoint <= 0:
                raise ValueError('band_neuron_ends endpoints must be >= 1.')
            if endpoint >= width:
                raise ValueError('band_neuron_ends endpoints must be < hidden width.')
        parsed.append(parts)
    return parsed


def group_ids_from_ends(width: int, cumulative_ends: Sequence[int], num_groups: int) -> list[int]:
    ids: list[int] = []
    cursor = 0
    for gid, end in enumerate(list(cumulative_ends) + [int(width)]):
        while cursor < int(end):
            ids.append(int(gid))
            cursor += 1
    if len(ids) != int(width) or any(g < 0 or g >= int(num_groups) for g in ids):
        raise ValueError('Failed to build group ids from cumulative ends.')
    return ids


def resolve_constraint_plan(model_spec: Any, hidden_widths: Sequence[int], constraint_config: ConstraintConfig | None) -> ResolvedConstraintPlan:
    config = constraint_config if constraint_config is not None else ConstraintConfig()
    mode = normalize_scenario_mode(config.mode)
    widths = [int(v) for v in hidden_widths]
    if not widths:
        mode = 'none'
    enabled = mode != 'none'
    if not enabled:
        md = {'scenario_mode': 'none', 'structure_mask': False, 'clip_params': False, 'supported_scope': 'dense_hidden_layers_only', 'applies_to_output_layer': False, 'tear': int(config.tear), 'num_groups': 0, 'band_edge': None, 'group_cumulative_ends_per_hidden_layer': None}
        return ResolvedConstraintPlan(config=ConstraintConfig(mode='none', w_clip_edges=config.w_clip_edges, alpha_clip_edges=config.alpha_clip_edges, band_edge=config.band_edge, band_neuron_ends=config.band_neuron_ends, tear=int(config.tear)), enabled=False, structure_mask=False, clip_params=False, num_groups=0, group_cumulative_ends_per_hidden_layer=None, group_ids_per_hidden_layer=None, metadata=md)

    if str(model_spec.family) not in {'lif', 'rf', 'if'}:
        raise ValueError(f'scenario_mode={mode!r} is supported only for dense if/lif/rf families; got family={model_spec.family!r}.')
    if int(getattr(model_spec, 'branch', 0) or 0) not in (0,):
        pass

    structure_mask = mode in {'structure', 'clipstructure'}
    clip_params = mode in {'clip', 'clipstructure'}

    if model_spec.family == 'if' and clip_params:
        raise ValueError('IF family does not support clip/clipstructure because there is no clip target parameter.')
    lif_edges = _normalize_layerwise_edges(config.alpha_clip_edges, lower=0.0, upper=1.0, name='alpha_clip_edges', num_layers=len(widths)) if clip_params and model_spec.family == 'lif' else None
    rf_edges = _normalize_layerwise_edges(config.w_clip_edges, lower=0.0, upper=0.5, name='w_clip_edges', num_layers=len(widths)) if clip_params and model_spec.family == 'rf' else None
    if clip_params and model_spec.family == 'lif' and lif_edges is None:
        if mode == 'clipstructure':
            clip_params = False
            mode = 'structure'
        else:
            raise ValueError('alpha_clip_edges is required for lif clip modes.')
    if clip_params and model_spec.family == 'rf' and rf_edges is None:
        if mode == 'clipstructure':
            clip_params = False
            mode = 'structure'
        else:
            raise ValueError('w_clip_edges is required for rf clip modes.')
    if mode == 'structure' and (config.alpha_clip_edges is not None or config.w_clip_edges is not None):
        raise ValueError('structure mode does not accept clip edges.')

    if config.band_edge is not None and config.band_neuron_ends is not None:
        raise ValueError('Use band_edge as public key; band_neuron_ends is source-only compatibility and cannot be combined.')
    legacy_band = config.band_neuron_ends
    if legacy_band is not None and config.band_edge is None:
        be = [[int(p.strip()) for p in str(e).split(',') if p.strip()] for e in legacy_band]
    else:
        be = config.band_edge
    if mode == 'structure':
        groups = [2 if (be is None or be[i] is None) else len(be[i]) + 1 for i in range(len(widths))]
    elif model_spec.family == 'lif':
        groups = [len(layer) for layer in lif_edges]
    else:
        groups = [len(layer) for layer in rf_edges]
    cumulative = _normalize_band_edge(widths, be, groups)
    gids = [group_ids_from_ends(w, ends, g) for w, ends, g in zip(widths, cumulative, groups)]

    tear = int(config.tear)
    if tear < 1 or tear > len(widths):
        raise ValueError(f'tear must satisfy 1 <= tear <= num_hidden_layers ({len(widths)}).')

    md = {
        'scenario_mode': mode,
        'structure_mask': bool(structure_mask),
        'clip_params': bool(clip_params),
        'supported_scope': 'dense_hidden_layers_only',
        'applies_to_output_layer': False,
        'tear': tear,
        'num_groups_per_hidden_layer': groups,
        'band_edge': [list(v) for v in cumulative],
        'group_cumulative_ends_per_hidden_layer': cumulative,
        'lif_clip_edges_per_layer': lif_edges,
        'rf_clip_edges_per_layer': rf_edges,
        'rf_clip_edges_unit': 'normalized_frequency_cyc_per_sample_nyquist_0p5',
        'lif_clip_edges_unit': 'alpha_unit_interval_0p1',
        'dynamics_init': 'bounded_parameterization',
    }
    normalized_config = ConstraintConfig(mode=mode, w_clip_edges=rf_edges, alpha_clip_edges=lif_edges, band_edge=[list(v) for v in cumulative], tear=tear)
    return ResolvedConstraintPlan(config=normalized_config, enabled=True, structure_mask=structure_mask, clip_params=clip_params, num_groups=int(max(groups) if groups else 0), group_cumulative_ends_per_hidden_layer=cumulative, group_ids_per_hidden_layer=gids, metadata=md)


def layer_constraint_for_hidden_index(plan: ResolvedConstraintPlan, hidden_index: int, input_size: int, output_size: int, recurrent: bool) -> LayerConstraint | None:
    if torch is None:
        raise RuntimeError('torch is required to build layer constraints.')
    if not plan.enabled:
        return None
    gids = plan.group_ids_per_hidden_layer
    if gids is None:
        return None
    current = gids[int(hidden_index)]
    out = LayerConstraint()

    if plan.clip_params:
        if plan.config.alpha_clip_edges is not None:
            edges = plan.config.alpha_clip_edges[int(hidden_index)]
            lows = torch.tensor([edges[g][0] for g in current], dtype=torch.float32)
            highs = torch.tensor([edges[g][1] for g in current], dtype=torch.float32)
            out.lif_alpha_bounds = (lows, highs)
        if plan.config.w_clip_edges is not None:
            edges = plan.config.w_clip_edges[int(hidden_index)]
            lows = torch.tensor([edges[g][0] for g in current], dtype=torch.float32)
            highs = torch.tensor([edges[g][1] for g in current], dtype=torch.float32)
            out.rf_frequency_bounds = (lows, highs)

    structure_active = bool(plan.structure_mask) and (int(hidden_index) + 1 >= int(plan.config.tear))
    if structure_active:
        if int(hidden_index) > 0:
            prev = gids[int(hidden_index) - 1]
            in_mask = torch.zeros((int(output_size), int(input_size)), dtype=torch.float32)
            for o in range(int(output_size)):
                for i in range(int(input_size)):
                    in_mask[o, i] = 1.0 if int(current[o]) == int(prev[i]) else 0.0
            out.input_mask = in_mask
        if bool(recurrent):
            rec_mask = torch.zeros((int(output_size), int(output_size)), dtype=torch.float32)
            for o in range(int(output_size)):
                for i in range(int(output_size)):
                    rec_mask[o, i] = 1.0 if int(current[o]) == int(current[i]) else 0.0
            out.recurrent_mask = rec_mask
    if out.input_mask is None and out.recurrent_mask is None and out.lif_alpha_bounds is None and out.rf_frequency_bounds is None:
        return None
    return out


__all__ = [
    'ConstraintConfig',
    'LayerConstraint',
    'ResolvedConstraintPlan',
    'default_band_neuron_ends',
    'group_ids_from_ends',
    'layer_constraint_for_hidden_index',
    'normalize_scenario_mode',
    'parse_band_neuron_ends',
    'resolve_constraint_plan',
    'validate_lif_clip_edges',
    'validate_rf_clip_edges',
]
