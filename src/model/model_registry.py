"""Canonical model-token parsing for the official experiment CLI."""

from __future__ import annotations

from dataclasses import dataclass
import re


_RESETTABLE_FAMILIES = {
    'lif': 'lif',
    'rf': 'rf',
}

_FIXED_CNN_BASES: dict[str, tuple[str, str, str]] = {}
for _backbone in ('vgg11', 'resnet18'):
    _FIXED_CNN_BASES[f'{_backbone}_lif'] = (_backbone, 'lif', 'cnn_lif')
    _FIXED_CNN_BASES[f'{_backbone}_rf'] = (_backbone, 'rf', 'cnn_rf')

_SIMPLE_ALIASES = {
    'tc': 'tc_lif',
    'tc_lif': 'tc_lif',
    'tclif': 'tc_lif',
    'ts': 'ts_lif',
    'ts_lif': 'ts_lif',
    'tslif': 'ts_lif',
}

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
_ALLOWED_RF_RESET_SUFFIXES = {'soft', 'hard'}
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

    @property
    def reset_enabled(self) -> bool | None:
        """Return whether reset is active when that concept exists for the family."""

        if self.reset_mode is None:
            return None
        return self.reset_mode != 'no_reset'


_DH_PATTERN = re.compile(r'^(dh_snn|dh)(?:_.+)?$')
_DRF_PATTERN = re.compile(r'^d_rf(?:_(\d+))?$')


def _parse_dh_variant(token: str) -> tuple[bool, int | None] | None:
    """Parse one DH-SNN token with optional recurrent and branch modifiers."""

    match = _DH_PATTERN.fullmatch(token)
    if match is None:
        return None
    prefix = match.group(1)
    if token == prefix:
        return False, None
    suffix = token[len(prefix) + 1 :]
    parts = [part for part in suffix.split('_') if part != '']
    if not parts:
        return False, None

    recurrent = False
    branch: int | None = None
    for part in parts:
        if part == 'r' and not recurrent:
            recurrent = True
            continue
        if part.isdigit() and branch is None:
            branch = int(part)
            continue
        return None
    return recurrent, branch


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


def canonicalize_model_token(token: str) -> ModelSpec:
    """Parse one CLI model token into the official canonical representation."""

    raw = str(token)
    normalized = raw.strip().lower().replace('-', '_')

    if normalized == 'spikingssm':
        return ModelSpec(raw_token=raw, canonical_token='spikingssm', family='spikingssm')
    if normalized == 'spikformer':
        return ModelSpec(raw_token=raw, canonical_token='spikformer', family='spikformer', backbone='spikformer')
    if normalized == 'spikegru':
        return ModelSpec(raw_token=raw, canonical_token='spikegru', family='spikegru', recurrent=True, backbone='spikegru')

    fixed_cnn_spec = _canonicalize_fixed_cnn_family(raw, normalized)
    if fixed_cnn_spec is not None:
        return fixed_cnn_spec

    resettable_spec = _canonicalize_resettable_family(raw, normalized)
    if resettable_spec is not None:
        return resettable_spec

    if normalized in {'lif', 'lif_r', 'rf', 'rf_r'}:
        raise ValueError(
            f'Model token {raw!r} is an incomplete shorthand. Use the full official token format such as '
            'lif_soft_fixed, lif_R_soft_fixed, rf_soft_fixed, or rf_R_soft_fixed.'
        )
    if _parse_dh_variant(normalized) is not None or _DRF_PATTERN.fullmatch(normalized) is not None:
        raise ValueError(
            f'Model token {raw!r} belongs to a reinterpretation-only family and is not an official psd_analysis model token.'
        )
    base_token, recurrent, reset_suffix, threshold_suffix = _split_trailing_modifiers(normalized)
    if reset_suffix is not None or threshold_suffix is not None or recurrent:
        raise ValueError(f'Unsupported official model token: {raw!r}.')
    if normalized in _SIMPLE_ALIASES:
        raise ValueError(
            f'Model token {raw!r} is not an official Spec/ psd_analysis token; '
            'use a full lif/rf token, a fixed CNN token, or one official auxiliary token.'
        )

    raise ValueError(f'Unsupported model token: {token}')

def canonicalize_model_tokens(tokens: list[str] | tuple[str, ...]) -> list[ModelSpec]:
    """Canonicalize many model tokens."""

    return [canonicalize_model_token(token) for token in tokens]


__all__ = ['ModelSpec', 'canonicalize_model_token', 'canonicalize_model_tokens']
