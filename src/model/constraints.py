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
    w_clip_edges: tuple[float, ...] | None = None
    alpha_clip_edges: tuple[float, ...] | None = None
    band_neuron_ends: tuple[str, ...] | None = None
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


def normalize_constraint_mode(mode: Any) -> str:
    token = str(mode if mode is not None else 'none').strip().lower()
    if token == 'clip_structure':
        return 'clipstructure'
    if token in {'none', 'clip', 'structure', 'clipstructure'}:
        return token
    raise ValueError(f'Unsupported constraint_mode: {mode!r}')


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
    mode = normalize_constraint_mode(config.mode)
    widths = [int(v) for v in hidden_widths]
    if not widths:
        mode = 'none'
    enabled = mode != 'none'
    if not enabled:
        md = {'constraint_mode': 'none', 'structure_mask': False, 'clip_params': False, 'supported_scope': 'dense_hidden_layers_only', 'applies_to_output_layer': False, 'tear': int(config.tear), 'num_groups': 0, 'band_neuron_ends': None, 'group_cumulative_ends_per_hidden_layer': None, 'lif_clip_edges': None, 'rf_clip_edges': None, 'rf_clip_edges_unit': 'normalized_frequency_cyc_per_sample_nyquist_0p5', 'lif_clip_edges_unit': 'alpha_unit_interval_0p1', 'dynamics_init': 'bounded_parameterization'}
        return ResolvedConstraintPlan(config=ConstraintConfig(mode='none', w_clip_edges=config.w_clip_edges, alpha_clip_edges=config.alpha_clip_edges, band_neuron_ends=config.band_neuron_ends, tear=int(config.tear)), enabled=False, structure_mask=False, clip_params=False, num_groups=0, group_cumulative_ends_per_hidden_layer=None, group_ids_per_hidden_layer=None, metadata=md)

    if str(model_spec.family) not in {'lif', 'rf'}:
        raise ValueError(f'constraint_mode={mode!r} is supported only for dense lif/rf families; got family={model_spec.family!r}.')
    if int(getattr(model_spec, 'branch', 0) or 0) not in (0,):
        pass

    structure_mask = mode in {'structure', 'clipstructure'}
    clip_params = mode in {'clip', 'clipstructure'}

    lif_edges = validate_lif_clip_edges(config.alpha_clip_edges) if clip_params and model_spec.family == 'lif' else None
    rf_edges = validate_rf_clip_edges(config.w_clip_edges) if clip_params and model_spec.family == 'rf' else None
    if clip_params and model_spec.family == 'lif' and lif_edges is None:
        raise ValueError('alpha_clip_edges is required for lif clip/clipstructure modes.')
    if clip_params and model_spec.family == 'rf' and rf_edges is None:
        raise ValueError('w_clip_edges is required for rf clip/clipstructure modes.')
    if mode == 'structure' and (config.alpha_clip_edges is not None or config.w_clip_edges is not None):
        raise ValueError('structure mode does not accept clip edges.')

    if mode == 'structure':
        if config.band_neuron_ends is not None:
            num_groups = len(str(config.band_neuron_ends[0]).split(',')) + 1
        else:
            num_groups = 2
    elif model_spec.family == 'lif':
        num_groups = len(lif_edges) - 1
    else:
        num_groups = len(rf_edges) - 1

    bands = list(config.band_neuron_ends) if config.band_neuron_ends is not None else default_band_neuron_ends(widths, num_groups)
    cumulative = parse_band_neuron_ends(widths, bands, num_groups)
    gids = [group_ids_from_ends(w, ends, num_groups) for w, ends in zip(widths, cumulative)]

    tear = int(config.tear)
    if tear < 1 or tear > len(widths):
        raise ValueError(f'tear must satisfy 1 <= tear <= num_hidden_layers ({len(widths)}).')

    md = {
        'constraint_mode': mode,
        'structure_mask': bool(structure_mask),
        'clip_params': bool(clip_params),
        'supported_scope': 'dense_hidden_layers_only',
        'applies_to_output_layer': False,
        'tear': tear,
        'num_groups': int(num_groups),
        'band_neuron_ends': bands,
        'group_cumulative_ends_per_hidden_layer': cumulative,
        'lif_clip_edges': lif_edges,
        'rf_clip_edges': rf_edges,
        'rf_clip_edges_unit': 'normalized_frequency_cyc_per_sample_nyquist_0p5',
        'lif_clip_edges_unit': 'alpha_unit_interval_0p1',
        'dynamics_init': 'bounded_parameterization',
    }
    normalized_config = ConstraintConfig(mode=mode, w_clip_edges=None if rf_edges is None else tuple(rf_edges), alpha_clip_edges=None if lif_edges is None else tuple(lif_edges), band_neuron_ends=tuple(bands), tear=tear)
    return ResolvedConstraintPlan(config=normalized_config, enabled=True, structure_mask=structure_mask, clip_params=clip_params, num_groups=int(num_groups), group_cumulative_ends_per_hidden_layer=cumulative, group_ids_per_hidden_layer=gids, metadata=md)


def layer_constraint_for_hidden_index(plan: ResolvedConstraintPlan, hidden_index: int, input_size: int, output_size: int, recurrent: bool) -> LayerConstraint | None:
    if torch is None:
        raise RuntimeError('torch is required to build layer constraints.')
    if not plan.enabled:
        return None
    if int(hidden_index) + 1 < int(plan.config.tear):
        return None
    gids = plan.group_ids_per_hidden_layer
    if gids is None:
        return None
    current = gids[int(hidden_index)]
    out = LayerConstraint()

    if plan.clip_params:
        if plan.config.alpha_clip_edges is not None:
            lows = torch.tensor([plan.config.alpha_clip_edges[g] for g in current], dtype=torch.float32)
            highs = torch.tensor([plan.config.alpha_clip_edges[g + 1] for g in current], dtype=torch.float32)
            out.lif_alpha_bounds = (lows, highs)
        if plan.config.w_clip_edges is not None:
            lows = torch.tensor([plan.config.w_clip_edges[g] for g in current], dtype=torch.float32)
            highs = torch.tensor([plan.config.w_clip_edges[g + 1] for g in current], dtype=torch.float32)
            out.rf_frequency_bounds = (lows, highs)

    if plan.structure_mask:
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
    return out


__all__ = [
    'ConstraintConfig',
    'LayerConstraint',
    'ResolvedConstraintPlan',
    'default_band_neuron_ends',
    'group_ids_from_ends',
    'layer_constraint_for_hidden_index',
    'normalize_constraint_mode',
    'parse_band_neuron_ends',
    'resolve_constraint_plan',
    'validate_lif_clip_edges',
    'validate_rf_clip_edges',
]
