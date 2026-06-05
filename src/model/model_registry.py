"""Canonical model-token parsing for the official experiment CLI."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any


_RESETTABLE_FAMILIES = {
    'if': 'if',
    'lif': 'lif',
    'rf': 'rf',
}

_FIXED_CNN_BACKBONE_ALIASES = {
    'vgg11': 'vgg11',
    'resnet18': 'resnet18',
    'resnet': 'resnet18',
}
_FIXED_CNN_BASES: dict[str, tuple[str, str, str]] = {}
for _alias, _backbone in _FIXED_CNN_BACKBONE_ALIASES.items():
    _FIXED_CNN_BASES[f'{_alias}_lif'] = (_backbone, 'lif', 'cnn_lif')
    _FIXED_CNN_BASES[f'{_alias}_rf'] = (_backbone, 'rf', 'cnn_rf')

_RESET_SUFFIX_TO_MODE = {
    'soft': 'soft_reset',
    'hard': 'hard_reset',
    'none': 'no_reset',
}

_THRESHOLD_SUFFIX_TO_TRAINABLE = {
    'fixed': False,
    'train': True,
}

_ALLOWED_LIF_RESET_SUFFIXES = {'soft', 'hard'}
_ALLOWED_RF_RESET_SUFFIXES = {'soft', 'hard', 'none'}
_ALLOWED_IF_RESET_SUFFIXES = {'soft', 'hard'}
_ALLOWED_CNN_RF_RESET_SUFFIXES = {'soft', 'hard'}
_ALLOWED_THRESHOLD_SUFFIXES = frozenset(_THRESHOLD_SUFFIX_TO_TRAINABLE)


@dataclass(frozen=True)
class ModelSpec:
    """Resolved canonical information for one CLI model token."""

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

    @property
    def reset_enabled(self) -> bool | None:
        """Return whether reset is active when that concept exists for the family."""

        if self.reset_mode is None:
            return None
        return self.reset_mode != 'no_reset'


def _split_trailing_modifiers(token: str) -> tuple[str, bool, str | None, str | None]:
    """Split base token, recurrent flag, reset suffix, and threshold suffix."""

    parts = [part for part in token.split('_') if part != '']
    threshold_suffix = None
    reset_suffix = None
    recurrent = False

    if parts and parts[-1] in _THRESHOLD_SUFFIX_TO_TRAINABLE:
        threshold_suffix = parts.pop()
    if parts and parts[-1] in _RESET_SUFFIX_TO_MODE:
        reset_suffix = parts.pop()
    if parts and parts[-1] == 'r':
        recurrent = True
        parts.pop()

    base = '_'.join(parts)
    return base, recurrent, reset_suffix, threshold_suffix


def _require_reset_suffix(*, family: str, reset_suffix: str | None, token: str) -> str:
    """Require one explicit trailing reset suffix for the resolved family."""

    if family in {'lif', 'cnn_lif'}:
        allowed = _ALLOWED_LIF_RESET_SUFFIXES
    elif family == 'if':
        allowed = _ALLOWED_IF_RESET_SUFFIXES
    elif family == 'rf':
        allowed = _ALLOWED_RF_RESET_SUFFIXES
    elif family == 'cnn_rf':
        allowed = _ALLOWED_CNN_RF_RESET_SUFFIXES
    else:
        if reset_suffix is not None:
            raise ValueError(f'Model token {token!r} does not support reset suffixes.')
        raise ValueError(f'Model token {token!r} requires an explicit family-specific parser.')
    if reset_suffix is None:
        allowed_tokens = ', '.join(sorted(allowed))
        raise ValueError(f'Model token {token!r} must include an explicit reset suffix {{{allowed_tokens}}}.')
    if reset_suffix not in allowed:
        allowed_tokens = ', '.join(sorted(allowed))
        raise ValueError(f'Model token {token!r} only supports reset suffixes {{{allowed_tokens}}}.')
    return reset_suffix


def _require_threshold_suffix(*, threshold_suffix: str | None, token: str) -> str:
    """Require one explicit trailing threshold suffix."""

    if threshold_suffix is None:
        allowed_tokens = ', '.join(sorted(_ALLOWED_THRESHOLD_SUFFIXES))
        raise ValueError(f'Model token {token!r} must include an explicit threshold suffix {{{allowed_tokens}}}.')
    if threshold_suffix not in _ALLOWED_THRESHOLD_SUFFIXES:
        allowed_tokens = ', '.join(sorted(_ALLOWED_THRESHOLD_SUFFIXES))
        raise ValueError(f'Model token {token!r} only supports threshold suffixes {{{allowed_tokens}}}.')
    return threshold_suffix


def _canonicalize_resettable_family(raw: str, normalized: str) -> ModelSpec | None:
    """Canonicalize one LIF/RF token when applicable."""

    base_token, recurrent, reset_suffix, threshold_suffix = _split_trailing_modifiers(normalized)
    if base_token not in _RESETTABLE_FAMILIES:
        return None
    canonical_base = _RESETTABLE_FAMILIES[base_token]
    family = canonical_base
    resolved_reset_suffix = _require_reset_suffix(family=family, reset_suffix=reset_suffix, token=raw)
    resolved_threshold_suffix = _require_threshold_suffix(threshold_suffix=threshold_suffix, token=raw)
    canonical = canonical_base
    if recurrent:
        canonical += '_R'
    canonical += f'_{resolved_reset_suffix}_{resolved_threshold_suffix}'
    return ModelSpec(
        raw_token=raw,
        canonical_token=canonical,
        family=family,
        recurrent=recurrent,
        reset_mode=_RESET_SUFFIX_TO_MODE[resolved_reset_suffix],
        threshold_suffix=resolved_threshold_suffix,
        trainable_threshold=bool(_THRESHOLD_SUFFIX_TO_TRAINABLE[resolved_threshold_suffix]),
    )


def _canonicalize_fixed_cnn_family(raw: str, normalized: str) -> ModelSpec | None:
    """Canonicalize one fixed CNN-backbone token such as vgg11_lif_soft_fixed."""

    base_token, recurrent, reset_suffix, threshold_suffix = _split_trailing_modifiers(normalized)
    if base_token not in _FIXED_CNN_BASES:
        return None
    if recurrent:
        raise ValueError(f'Fixed CNN model token {raw!r} does not support recurrent suffix _R.')
    backbone, neuron_name, family = _FIXED_CNN_BASES[base_token]
    resolved_reset_suffix = _require_reset_suffix(family=family, reset_suffix=reset_suffix, token=raw)
    resolved_threshold_suffix = _require_threshold_suffix(threshold_suffix=threshold_suffix, token=raw)
    canonical = f'{backbone}_{neuron_name}_{resolved_reset_suffix}_{resolved_threshold_suffix}'
    return ModelSpec(
        raw_token=raw,
        canonical_token=canonical,
        family=family,
        recurrent=False,
        reset_mode=_RESET_SUFFIX_TO_MODE[resolved_reset_suffix],
        threshold_suffix=resolved_threshold_suffix,
        trainable_threshold=bool(_THRESHOLD_SUFFIX_TO_TRAINABLE[resolved_threshold_suffix]),
        backbone=backbone,
    )


def _canonicalize_tc_ts_family(raw: str, normalized: str) -> ModelSpec | None:
    alias_to_family = {
        'tc': 'tc_lif',
        'tc_lif': 'tc_lif',
        'tclif': 'tc_lif',
        'ts': 'ts_lif',
        'ts_lif': 'ts_lif',
        'tslif': 'ts_lif',
    }
    patterns = [
        ('_r', True),
        ('', False),
    ]
    for suffix, recurrent in patterns:
        for alias, family in alias_to_family.items():
            if normalized == f'{alias}{suffix}':
                canonical = family + ('_R' if recurrent else '')
                return ModelSpec(raw_token=raw, canonical_token=canonical, family=family, recurrent=recurrent, branch=None)
    return None


def _canonicalize_dh_snn_family(raw: str, normalized: str) -> ModelSpec | None:
    match = re.fullmatch(r'^(dh|dh_snn)(?:_(r))?(?:_(\d+))?$', normalized)
    if match is None:
        return None
    base = match.group(1)
    recurrent = match.group(2) is not None
    branch_text = match.group(3)
    branch = 4 if branch_text is None else int(branch_text)
    if branch <= 0:
        raise ValueError(f'Model token {raw!r} must use a positive branch integer for dh_snn.')
    canonical = f"dh_snn{'_R' if recurrent else ''}_{branch}"
    return ModelSpec(raw_token=raw, canonical_token=canonical, family='dh_snn', recurrent=recurrent, branch=branch)


def _canonicalize_d_rf_family(raw: str, normalized: str) -> ModelSpec | None:
    recurrent = re.fullmatch(r'^d_rf_r(?:_(\d+))?$', normalized)
    if recurrent is not None:
        raise ValueError('d_rf_R is not supported because DRFLayer does not expose true recurrent dynamics.')
    match = re.fullmatch(r'^d_rf(?:_(\d+))?$', normalized)
    if match is None:
        return None
    branch_text = match.group(1)
    branch = 4 if branch_text is None else int(branch_text)
    if branch <= 0:
        raise ValueError(f'Model token {raw!r} must use a positive branch integer for d_rf.')
    canonical = f'd_rf_{branch}'
    return ModelSpec(raw_token=raw, canonical_token=canonical, family='d_rf', recurrent=False, branch=branch)



def _parse_boolish(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    token = str(value).strip().lower()
    if token in {'1', 'true', 'yes', 'y', 'on', 'r', 'recurrent'}:
        return True
    if token in {'0', 'false', 'no', 'n', 'off', 'none', ''}:
        return False
    raise ValueError(f'Cannot parse boolean model field value: {value!r}.')


def parse_v_threshold_setting(value: Any = None) -> tuple[bool, float]:
    """Parse the new config-level v_th pair [fixed|train, initial_value]."""

    if value is None or value == '':
        return False, 1.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return False, float(value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith('['):
            import json
            return parse_v_threshold_setting(json.loads(text))
        parts = [part for part in re.split(r'[\s,]+', text) if part]
    else:
        parts = list(value) if isinstance(value, (list, tuple)) else [value]
    if not parts:
        return False, 1.0
    if len(parts) == 1:
        return False, float(parts[0])
    if len(parts) != 2:
        raise ValueError('v_th must be ["fixed"|"train", initial_value].')
    mode = str(parts[0]).strip().lower()
    if mode not in {'fixed', 'train'}:
        raise ValueError('v_th[0] must be "fixed" or "train".')
    init = float(parts[1])
    if init <= 0.0:
        raise ValueError('v_th initial value must be positive.')
    return mode == 'train', init


def parse_filter_setting(value: Any = None) -> tuple[str, float | None]:
    """Parse config-level filter setting.

    ``"train"`` leaves the neuron filter parameter trainable. A numeric string
    fixes the filter parameter to that value and disables its gradient. Dense/CNN
    LIF interpret this as alpha; RF interprets it as center frequency.
    """

    if value is None or value == '':
        return 'train', None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return 'fixed', float(value)
    token = str(value).strip().lower()
    if token in {'train', 'learn', 'learnable', 'parameter', 'param'}:
        return 'train', None
    if token in {'none', 'null'}:
        return 'train', None
    try:
        return 'fixed', float(token)
    except ValueError as exc:
        raise ValueError('filter must be "train" or a numeric string for a fixed non-trainable filter parameter.') from exc


def _reset_suffix_from_field(reset: Any, *, neuron_type: str) -> str:
    if reset is None or reset == '':
        if neuron_type in {'rf', 'cnn_rf'}:
            return 'none'
        return 'soft'
    token = str(reset).strip().lower().replace('-', '_')
    aliases = {
        'soft': 'soft', 'soft_reset': 'soft',
        'hard': 'hard', 'hard_reset': 'hard',
        'none': 'none', 'no': 'none', 'no_reset': 'none', 'off': 'none',
    }
    if token not in aliases:
        raise ValueError(f'Unsupported reset setting: {reset!r}.')
    return aliases[token]


def _extract_branch_from_neuron_type(token: str, *, base: str, default: int = 4) -> int:
    if token == base:
        return int(default)
    match = re.fullmatch(rf'{re.escape(base)}_(\d+)', token)
    if match is None:
        return int(default)
    value = int(match.group(1))
    if value <= 0:
        raise ValueError(f'{base} branch must be positive.')
    return value


def model_token_from_config_fields(
    *,
    neuron_type: Any,
    recurrent: Any = False,
    reset: Any = None,
    v_th: Any = None,
    branch: Any = None,
) -> tuple[str, bool, float]:
    """Build the canonical legacy token from the new explicit model fields."""

    if neuron_type is None or str(neuron_type).strip() == '':
        raise ValueError('Either model or neuron_type must be provided.')
    nt = str(neuron_type).strip().lower().replace('-', '_')
    rec = _parse_boolish(recurrent, default=False)
    trainable_threshold, threshold_value = parse_v_threshold_setting(v_th)
    threshold_suffix = 'train' if trainable_threshold else 'fixed'
    reset_suffix = _reset_suffix_from_field(reset, neuron_type=nt)

    if nt in {'if', 'lif', 'rf'}:
        token = nt
        if rec:
            token += '_R'
        token += f'_{reset_suffix}_{threshold_suffix}'
        return token, trainable_threshold, threshold_value

    if nt in {'tc', 'tc_lif', 'tclif'}:
        return 'tc_lif' + ('_R' if rec else ''), trainable_threshold, threshold_value
    if nt in {'ts', 'ts_lif', 'tslif'}:
        return 'ts_lif' + ('_R' if rec else ''), trainable_threshold, threshold_value

    if nt in {'dh', 'dh_snn'} or nt.startswith('dh_snn_'):
        b = int(branch) if branch not in (None, '') else _extract_branch_from_neuron_type(nt, base='dh_snn', default=4)
        return f"dh_snn{'_R' if rec else ''}_{b}", trainable_threshold, threshold_value
    if nt in {'d_rf', 'drf'} or nt.startswith('d_rf_'):
        if rec:
            raise ValueError('d_rf does not support recurrent=true.')
        b = int(branch) if branch not in (None, '') else _extract_branch_from_neuron_type(nt, base='d_rf', default=4)
        return f'd_rf_{b}', trainable_threshold, threshold_value

    if nt in {'spikformer', 'spikeformer'}:
        return 'spikformer', trainable_threshold, threshold_value
    if nt == 'spikegru':
        return 'spikegru', trainable_threshold, threshold_value
    if nt in {'spikingssm', 'spiking_ssm'}:
        return 'spikingssm', trainable_threshold, threshold_value

    # Fixed CNN backbones. The neuron suffix is kept explicit so configs do not
    # need the old monolithic model token.
    cnn_aliases = {
        'vgg': ('vgg11', 'lif'),
        'vgg11': ('vgg11', 'lif'),
        'vgg11_lif': ('vgg11', 'lif'),
        'vgg11_rf': ('vgg11', 'rf'),
        'resnet': ('resnet18', 'lif'),
        'resnet18': ('resnet18', 'lif'),
        'resnet18_lif': ('resnet18', 'lif'),
        'resnet18_rf': ('resnet18', 'rf'),
    }
    if nt in cnn_aliases:
        if rec:
            raise ValueError(f'CNN backbone neuron_type={neuron_type!r} does not support recurrent=true.')
        backbone, neuron = cnn_aliases[nt]
        return f'{backbone}_{neuron}_{reset_suffix}_{threshold_suffix}', trainable_threshold, threshold_value

    raise ValueError(f'Unsupported neuron_type: {neuron_type!r}.')


def model_spec_from_config_fields(
    *,
    model: Any = None,
    neuron_type: Any = None,
    recurrent: Any = False,
    reset: Any = None,
    v_th: Any = None,
    filter: Any = None,
    branch: Any = None,
) -> ModelSpec:
    """Resolve either the new explicit fields or one legacy token into ModelSpec."""

    if model not in (None, ''):
        trainable_threshold, threshold_value = parse_v_threshold_setting(v_th)
        spec = canonicalize_model_token(str(model))
        # A legacy token already encodes train/fixed. Only the initial value comes
        # from v_th when supplied as a scalar or pair.
        if v_th is not None:
            spec = replace(spec, trainable_threshold=bool(trainable_threshold), threshold_value=float(threshold_value))
        else:
            spec = replace(spec, threshold_value=float(threshold_value))
    else:
        token, trainable_threshold, threshold_value = model_token_from_config_fields(
            neuron_type=neuron_type,
            recurrent=recurrent,
            reset=reset,
            v_th=v_th,
            branch=branch,
        )
        spec = canonicalize_model_token(token)
        spec = replace(spec, trainable_threshold=bool(trainable_threshold), threshold_value=float(threshold_value))
    filter_mode, filter_value = parse_filter_setting(filter)
    return replace(spec, filter_mode=filter_mode, filter_value=filter_value)


def model_spec_from_namespace(args: Any) -> ModelSpec:
    return model_spec_from_config_fields(
        model=getattr(args, 'model', None),
        neuron_type=getattr(args, 'neuron_type', None),
        recurrent=getattr(args, 'recurrent', False),
        reset=getattr(args, 'reset', None),
        v_th=getattr(args, 'v_th', None),
        filter=getattr(args, 'filter', None),
        branch=getattr(args, 'branch', None),
    )

def canonicalize_model_token(token: str) -> ModelSpec:
    """Parse one CLI model token into the official canonical representation."""

    raw = str(token)
    normalized = raw.strip().lower().replace('-', '_')

    if normalized == 'spikingssm':
        return ModelSpec(raw_token=raw, canonical_token='spikingssm', family='spikingssm')
    if normalized in {'spikformer', 'spikeformer'}:
        return ModelSpec(raw_token=raw, canonical_token='spikformer', family='spikformer', backbone='spikformer')
    if normalized == 'spikegru':
        return ModelSpec(raw_token=raw, canonical_token='spikegru', family='spikegru', recurrent=True, backbone='spikegru')

    fixed_cnn_spec = _canonicalize_fixed_cnn_family(raw, normalized)
    if fixed_cnn_spec is not None:
        return fixed_cnn_spec

    resettable_spec = _canonicalize_resettable_family(raw, normalized)
    if resettable_spec is not None:
        return resettable_spec

    tc_ts_spec = _canonicalize_tc_ts_family(raw, normalized)
    if tc_ts_spec is not None:
        return tc_ts_spec

    dh_spec = _canonicalize_dh_snn_family(raw, normalized)
    if dh_spec is not None:
        return dh_spec

    d_rf_spec = _canonicalize_d_rf_family(raw, normalized)
    if d_rf_spec is not None:
        return d_rf_spec

    if normalized in {'lif', 'lif_r', 'rf', 'rf_r'}:
        raise ValueError(
            f'Model token {raw!r} is an incomplete shorthand. Use the full official token format such as '
            'lif_soft_fixed, lif_R_soft_fixed, rf_soft_fixed, or rf_R_soft_fixed.'
        )

    base_token, recurrent, reset_suffix, threshold_suffix = _split_trailing_modifiers(normalized)
    if reset_suffix is not None or threshold_suffix is not None or recurrent:
        raise ValueError(f'Unsupported official model token: {raw!r}.')

    raise ValueError(f'Unsupported model token: {token}')


def canonicalize_model_tokens(tokens: list[str] | tuple[str, ...]) -> list[ModelSpec]:
    """Canonicalize many model tokens."""

    return [canonicalize_model_token(token) for token in tokens]


__all__ = ['ModelSpec', 'canonicalize_model_token', 'canonicalize_model_tokens', 'model_token_from_config_fields', 'model_spec_from_config_fields', 'model_spec_from_namespace', 'parse_v_threshold_setting', 'parse_filter_setting']
