"""Structured model-configuration registry.

Model selection is intentionally explicit: user-facing configs provide separate
fields such as ``neuron_type``, ``branch``, ``reset`` and ``v_th``.  Historical
monolithic tokens (for example ``my_d_rf_8_hard_train`` or
``lif_soft_fixed``) are rejected at the configuration boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any

_RESET_SUFFIX_TO_MODE = {
    'soft': 'soft_reset',
    'hard': 'hard_reset',
    'none': 'no_reset',
}

_THRESHOLD_SUFFIX_TO_TRAINABLE = {
    'fixed': False,
    'train': True,
}

_ALLOWED_THRESHOLD_SUFFIXES = frozenset(_THRESHOLD_SUFFIX_TO_TRAINABLE)
_ALLOWED_RESET_BY_FAMILY: dict[str, frozenset[str]] = {
    'if': frozenset({'soft', 'hard'}),
    'lif': frozenset({'soft', 'hard'}),
    'rf': frozenset({'soft', 'hard', 'none'}),
    'cnn_lif': frozenset({'soft', 'hard'}),
    'cnn_rf': frozenset({'soft', 'hard'}),
    'my_dh_snn': frozenset({'soft', 'hard', 'none'}),
    'my_d_rf': frozenset({'soft', 'hard', 'none'}),
    'my_r_dh_snn': frozenset({'soft', 'hard', 'none'}),
}

_BRANCH_DEFAULTS = {
    'dh_snn': 4,
    'd_rf': 4,
    'my_dh_snn': 8,
    'my_d_rf': 8,
    'my_r_dh_snn': 8,
}

_BRANCH_FAMILIES = frozenset(_BRANCH_DEFAULTS)
_TRAINABLE_THRESHOLD_FAMILIES = frozenset({
    'if',
    'lif',
    'rf',
    'cnn_lif',
    'cnn_rf',
    'my_dh_snn',
    'my_d_rf',
    'my_r_dh_snn',
})

_CANONICAL_FAMILY_ALIASES: dict[str, tuple[str, str | None]] = {
    'if': ('if', None),
    'lif': ('lif', None),
    'rf': ('rf', None),
    'tc': ('tc_lif', None),
    'tc_lif': ('tc_lif', None),
    'tclif': ('tc_lif', None),
    'ts': ('ts_lif', None),
    'ts_lif': ('ts_lif', None),
    'tslif': ('ts_lif', None),
    'dh': ('dh_snn', None),
    'dh_snn': ('dh_snn', None),
    'd_rf': ('d_rf', None),
    'drf': ('d_rf', None),
    'my_dh': ('my_dh_snn', None),
    'my_dh_snn': ('my_dh_snn', None),
    'my_d_rf': ('my_d_rf', None),
    'my_drf': ('my_d_rf', None),
    'my_r_dh': ('my_r_dh_snn', None),
    'my_r_dh_snn': ('my_r_dh_snn', None),
    'my_reverse_dh': ('my_r_dh_snn', None),
    'my_reverse_dh_snn': ('my_r_dh_snn', None),
    'spikegru': ('spikegru', 'spikegru'),
    'spikformer': ('spikformer', 'spikformer'),
    'spikeformer': ('spikformer', 'spikformer'),
    'spikingssm': ('spikingssm', None),
    'spiking_ssm': ('spikingssm', None),
    'vgg': ('cnn_lif', 'vgg11'),
    'vgg11': ('cnn_lif', 'vgg11'),
    'vgg11_lif': ('cnn_lif', 'vgg11'),
    'vgg11_rf': ('cnn_rf', 'vgg11'),
    'resnet': ('cnn_lif', 'resnet18'),
    'resnet18': ('cnn_lif', 'resnet18'),
    'resnet18_lif': ('cnn_lif', 'resnet18'),
    'resnet18_rf': ('cnn_rf', 'resnet18'),
}

_REMOVED_TOKEN_HINT = (
    "Model-token syntax has been removed. Configure models with separate fields, "
    "for example: neuron_type: my_d_rf, branch: 8, reset: hard, v_th: ['train', 1.0]."
)


@dataclass(frozen=True)
class ModelSpec:
    """Resolved model contract built only from structured config fields."""

    raw_token: str
    canonical_token: str
    family: str
    recurrent: bool = False
    branch: int | None = None
    reset_mode: str | None = None
    threshold_suffix: str | None = None
    trainable_threshold: bool | None = None
    backbone: str | None = None
    threshold_value: float = 1.0
    filter_mode: str = 'train'
    filter_value: float | None = None
    rf_pole_radius_constrained: bool = True
    rf_pole_radius_max: float = 0.9999

    @property
    def reset_enabled(self) -> bool | None:
        """Return whether reset is active when that concept exists for the family."""

        if self.reset_mode is None:
            return None
        return self.reset_mode != 'no_reset'


def _is_blank_config_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ''
    if isinstance(value, (list, tuple)):
        return all(_is_blank_config_value(item) for item in value)
    return False


def _normalize_text(value: Any) -> str:
    return str(value).strip().lower().replace('-', '_')


def _reject_removed_token_syntax(raw: Any, normalized: str) -> None:
    """Reject known historical monolithic model-token spellings."""

    parts = [part for part in normalized.split('_') if part]
    suffix_parts = {'soft', 'hard', 'none', 'fixed', 'train'}
    has_suffix_part = any(part in suffix_parts for part in parts)
    has_branch_suffix = re.search(r'_(\d+)(?:_|$)', normalized) is not None
    has_recurrent_suffix = normalized.endswith('_r') and normalized not in {'my_r_dh', 'my_r_dh_snn'}
    if has_suffix_part or has_branch_suffix or has_recurrent_suffix:
        raise ValueError(f"Unsupported neuron_type={raw!r}. {_REMOVED_TOKEN_HINT}")


def _resolve_family_and_backbone(neuron_type: Any) -> tuple[str, str | None]:
    if _is_blank_config_value(neuron_type):
        raise ValueError('model.neuron_type must be provided; legacy model-token fields are not supported.')
    normalized = _normalize_text(neuron_type)
    _reject_removed_token_syntax(neuron_type, normalized)
    if normalized not in _CANONICAL_FAMILY_ALIASES:
        supported = ', '.join(sorted(_CANONICAL_FAMILY_ALIASES))
        raise ValueError(f'Unsupported neuron_type={neuron_type!r}. Supported base neuron_type values: {supported}.')
    return _CANONICAL_FAMILY_ALIASES[normalized]


def _parse_boolish(value: Any, *, default: bool) -> bool:
    if _is_blank_config_value(value):
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if text in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise ValueError(f'Could not parse boolean value {value!r}.')


def _parse_recurrent_for_family(value: Any, *, family: str) -> bool:
    if family == 'spikegru':
        # The architecture is recurrent by definition; no monolithic _R token is
        # exposed in the structured config contract.
        return True
    recurrent = _parse_boolish(value, default=False)
    if not recurrent:
        return False
    if family in {'d_rf', 'my_dh_snn', 'my_d_rf', 'my_r_dh_snn', 'cnn_lif', 'cnn_rf', 'spikformer', 'spikingssm'}:
        raise ValueError(f'neuron_type={family!r} does not support model.recurrent=true; choose a family that exposes recurrent dynamics.')
    return True


def _parse_branch_for_family(value: Any, *, family: str) -> int | None:
    if family not in _BRANCH_FAMILIES:
        if not _is_blank_config_value(value):
            raise ValueError(f'model.branch is only supported for {sorted(_BRANCH_FAMILIES)}; got branch={value!r} for {family!r}.')
        return None
    if _is_blank_config_value(value):
        raise ValueError(
            f'model.branch / --branch is required for neuron_type={family!r}. '
            'Do not encode branch count inside neuron_type.'
        )
    try:
        branch = int(str(value).strip())
    except Exception as exc:
        raise ValueError(f'model.branch for {family!r} must be a positive integer, got {value!r}.') from exc
    if branch <= 0:
        raise ValueError(f'model.branch for {family!r} must be positive, got {branch}.')
    return branch


def parse_v_threshold_setting(value: Any) -> tuple[bool, float]:
    """Parse ``v_th`` as ``[fixed|train, value]`` or as a legacy scalar value."""

    if _is_blank_config_value(value):
        return False, 1.0
    if isinstance(value, (list, tuple)):
        nonblank = [item for item in value if not _is_blank_config_value(item)]
        if not nonblank:
            return False, 1.0
        if len(nonblank) == 1:
            item = nonblank[0]
            if isinstance(item, str) and item.strip().lower() in _THRESHOLD_SUFFIX_TO_TRAINABLE:
                return bool(_THRESHOLD_SUFFIX_TO_TRAINABLE[item.strip().lower()]), 1.0
            return False, float(item)
        mode = str(nonblank[0]).strip().lower()
        if mode not in _THRESHOLD_SUFFIX_TO_TRAINABLE:
            allowed = ', '.join(sorted(_ALLOWED_THRESHOLD_SUFFIXES))
            raise ValueError(f'v_th first element must be one of {{{allowed}}}; got {nonblank[0]!r}.')
        return bool(_THRESHOLD_SUFFIX_TO_TRAINABLE[mode]), float(nonblank[1])
    return False, float(value)


def _resolve_reset_suffix(value: Any, *, family: str) -> str | None:
    allowed = _ALLOWED_RESET_BY_FAMILY.get(family)
    if allowed is None:
        if _is_blank_config_value(value):
            return None
        raise ValueError(f'model.reset is not supported for neuron_type={family!r}.')
    default = 'none' if family == 'rf' else 'soft'
    if _is_blank_config_value(value):
        suffix = default
    else:
        suffix = _normalize_text(value)
    if suffix not in allowed:
        allowed_text = ', '.join(sorted(allowed))
        raise ValueError(f'model.reset for neuron_type={family!r} must be one of {{{allowed_text}}}; got {value!r}.')
    return suffix


def parse_filter_setting(value: Any) -> tuple[str, float | None]:
    """Parse filter initialization/training mode."""

    if _is_blank_config_value(value):
        return 'train', None
    if isinstance(value, (list, tuple)):
        nonblank = [item for item in value if not _is_blank_config_value(item)]
        if not nonblank:
            return 'train', None
        if len(nonblank) == 1:
            mode = str(nonblank[0]).strip().lower()
            if mode in {'train', 'fixed'}:
                return mode, None
            return 'fixed', float(nonblank[0])
        mode = str(nonblank[0]).strip().lower()
        if mode not in {'train', 'fixed'}:
            raise ValueError(f'filter first element must be "train" or "fixed"; got {nonblank[0]!r}.')
        return mode, float(nonblank[1])
    text = str(value).strip().lower()
    if text in {'train', 'fixed'}:
        return text, None
    return 'fixed', float(value)


def parse_rf_pole_radius_setting(*, constrained: Any = None, radius_max: Any = None) -> tuple[bool, float]:
    """Parse vanilla-RF discrete-pole radius constraint settings."""

    is_constrained = _parse_boolish(constrained, default=True)
    max_value = 0.9999 if _is_blank_config_value(radius_max) else float(radius_max)
    if max_value <= 0.0:
        raise ValueError(f'rf_pole_radius_max must be positive, got {max_value}.')
    if is_constrained and max_value >= 1.0:
        raise ValueError(f'rf_pole_radius_max must be < 1.0 when constrained=true, got {max_value}.')
    return is_constrained, max_value


def _structured_canonical_token(spec: ModelSpec) -> str:
    parts: list[str] = [spec.family]
    if spec.backbone:
        parts.append(f'backbone_{spec.backbone}')
    if spec.recurrent:
        parts.append('recurrent')
    if spec.branch is not None:
        parts.append(f'branch_{spec.branch}')
    if spec.reset_mode is not None:
        reset_suffix = next(key for key, value in _RESET_SUFFIX_TO_MODE.items() if value == spec.reset_mode)
        parts.append(f'reset_{reset_suffix}')
    if spec.threshold_suffix is not None:
        parts.append(f'vth_{spec.threshold_suffix}')
    if spec.family == 'rf':
        radius_mode = 'stable' if spec.rf_pole_radius_constrained else 'free_radius'
        parts.append(radius_mode)
    return '__'.join(parts)


def model_spec_from_config_fields(
    *,
    model: Any = None,
    neuron_type: Any = None,
    recurrent: Any = False,
    reset: Any = None,
    v_th: Any = None,
    filter: Any = None,
    branch: Any = None,
    rf_pole_radius_constrained: Any = None,
    rf_pole_radius_max: Any = None,
) -> ModelSpec:
    """Resolve explicit config fields into ``ModelSpec``.

    The ``model`` argument is kept only to produce a clear error for legacy
    callers.  It is not accepted as a user-facing shortcut.
    """

    if not _is_blank_config_value(model):
        raise ValueError(f'Legacy model-token field model={model!r} is no longer supported. {_REMOVED_TOKEN_HINT}')

    family, backbone = _resolve_family_and_backbone(neuron_type)
    is_recurrent = _parse_recurrent_for_family(recurrent, family=family)
    branch_value = _parse_branch_for_family(branch, family=family)
    reset_suffix = _resolve_reset_suffix(reset, family=family)
    reset_mode = None if reset_suffix is None else _RESET_SUFFIX_TO_MODE[reset_suffix]
    trainable_threshold, threshold_value = parse_v_threshold_setting(v_th)
    threshold_suffix = 'train' if trainable_threshold else 'fixed'
    if trainable_threshold and family not in _TRAINABLE_THRESHOLD_FAMILIES:
        raise ValueError(f'v_th: ["train", value] is only supported for soma-threshold families; got neuron_type={family!r}.')
    if family not in _TRAINABLE_THRESHOLD_FAMILIES and _is_blank_config_value(v_th):
        threshold_suffix = None
        trainable_threshold = None
    filter_mode, filter_value = parse_filter_setting(filter)
    radius_constrained, radius_max = parse_rf_pole_radius_setting(
        constrained=rf_pole_radius_constrained,
        radius_max=rf_pole_radius_max,
    )

    spec = ModelSpec(
        raw_token='structured_config',
        canonical_token='structured_config',
        family=family,
        recurrent=is_recurrent,
        branch=branch_value,
        reset_mode=reset_mode,
        threshold_suffix=threshold_suffix,
        trainable_threshold=trainable_threshold,
        backbone=backbone,
        threshold_value=float(threshold_value),
        filter_mode=filter_mode,
        filter_value=filter_value,
        rf_pole_radius_constrained=radius_constrained,
        rf_pole_radius_max=radius_max,
    )
    canonical = _structured_canonical_token(spec)
    return replace(spec, raw_token=canonical, canonical_token=canonical)


def model_spec_from_namespace(args: Any) -> ModelSpec:
    return model_spec_from_config_fields(
        model=getattr(args, 'model', None),
        neuron_type=getattr(args, 'neuron_type', None),
        recurrent=getattr(args, 'recurrent', False),
        reset=getattr(args, 'reset', None),
        v_th=getattr(args, 'v_th', None),
        filter=getattr(args, 'filter', None),
        branch=getattr(args, 'branch', None),
        rf_pole_radius_constrained=getattr(args, 'rf_pole_radius_constrained', None),
        rf_pole_radius_max=getattr(args, 'rf_pole_radius_max', None),
    )


def canonicalize_model_token(token: str) -> ModelSpec:
    """Reject historical monolithic model tokens.

    This function remains importable so old call sites fail with an actionable
    error instead of silently constructing a legacy model contract.
    """

    raise ValueError(f'Unsupported model token {token!r}. {_REMOVED_TOKEN_HINT}')


def canonicalize_model_tokens(tokens: list[str] | tuple[str, ...]) -> list[ModelSpec]:
    """Reject a sequence of historical model tokens."""

    return [canonicalize_model_token(token) for token in tokens]


def model_token_from_config_fields(*args: Any, **kwargs: Any) -> tuple[str, bool, float]:
    """Legacy helper retained only to reject token construction."""

    raise ValueError(_REMOVED_TOKEN_HINT)


__all__ = [
    'ModelSpec',
    'canonicalize_model_token',
    'canonicalize_model_tokens',
    'model_token_from_config_fields',
    'model_spec_from_config_fields',
    'model_spec_from_namespace',
    'parse_v_threshold_setting',
    'parse_filter_setting',
    'parse_rf_pole_radius_setting',
]
