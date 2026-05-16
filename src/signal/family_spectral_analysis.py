"""PSD-first family spectral summaries and curve distances."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from src.signal.psd_utils import (
    aggregate_userbins_torch,
    apply_centering,
    centered_pointwise_l2,
    first_difference_l2,
    exact_periodogram_from_maps,
    exact_spectrogram_from_maps,
    power_to_db,
)


_REDUCERS = ('mean', 'median')
_CENTERINGS = ('raw', 'cen')
_CENTERING_ALIASES = {'raw': 'raw', 'cen': 'cen', 'centered': 'cen'}
_SCALES = ('raw', 'db')
_CURVE_EXTRACTORS = ('psd_exact',)
_CURVE_SPACES = ('exact', 'userbin')
_CURVE_DISTANCE_METRICS = ('centered_l2', 'diff_l2')
_CURVE_DISTANCE_ALIASES = {
    'centered_l2': 'centered_l2',
    'center_l2': 'centered_l2',
    'centered': 'centered_l2',
    'diff_l2': 'diff_l2',
    'difference_l2': 'diff_l2',
    'first_difference_l2': 'diff_l2',
}


def _ensure_maps(signal_maps: torch.Tensor) -> torch.Tensor:
    if not isinstance(signal_maps, torch.Tensor):
        raise TypeError('signal_maps must be a torch.Tensor.')
    if signal_maps.ndim != 3:
        raise ValueError(f'Expected signal_maps with shape (samples, rows, time), got {tuple(signal_maps.shape)}.')
    if not signal_maps.is_floating_point():
        signal_maps = signal_maps.to(dtype=torch.float32)
    return signal_maps


def _ensure_rank3_curve_stack(values: torch.Tensor) -> torch.Tensor:
    if not isinstance(values, torch.Tensor):
        raise TypeError('values must be a torch.Tensor.')
    if values.ndim < 3:
        raise ValueError(f'Expected values with shape (samples, rows, ...), got {tuple(values.shape)}.')
    return values


def _reduce_rows(values: torch.Tensor, reducer: str) -> torch.Tensor:
    reducer = str(reducer).strip().lower()
    if reducer not in _REDUCERS:
        raise ValueError(f'Unsupported reducer: {reducer!r}.')
    curves = _ensure_rank3_curve_stack(values)
    if reducer == 'mean':
        return curves.mean(dim=1)
    return curves.median(dim=1).values


def _normalize_centering(centering: str) -> str:
    token = str(centering).strip().lower()
    normalized = _CENTERING_ALIASES.get(token)
    if normalized is None:
        raise ValueError(f'Unsupported signal centering: {centering!r}.')
    return normalized


def _variant_maps(signal_maps: torch.Tensor, centering: str) -> torch.Tensor:
    token = _normalize_centering(centering)
    return signal_maps if token == 'raw' else apply_centering(signal_maps)


def _apply_curve_scale(curves: torch.Tensor, scale: str) -> torch.Tensor:
    token = str(scale).strip().lower()
    if token not in _SCALES:
        raise ValueError(f'Unsupported curve scale: {scale!r}.')
    return curves if token == 'raw' else power_to_db(curves)


def _normalize_curve_distance_metric(metric: str) -> str:
    token = str(metric).strip().lower().replace('-', '_')
    normalized = _CURVE_DISTANCE_ALIASES.get(token)
    if normalized is None:
        allowed = ', '.join(_CURVE_DISTANCE_METRICS)
        raise ValueError(f'Unsupported curve distance metric: {metric!r}. Allowed values: {allowed}.')
    return normalized


def centered_pointwise_l2_torch(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Torch implementation of the centered pointwise curve-shape distance."""

    if not isinstance(u, torch.Tensor) or not isinstance(v, torch.Tensor):
        raise TypeError('centered_pointwise_l2_torch expects torch.Tensor inputs.')
    u_flat = u.reshape(-1)
    v_flat = v.reshape(-1)
    if u_flat.shape != v_flat.shape:
        raise ValueError(f'Curve shape mismatch: {tuple(u_flat.shape)} vs {tuple(v_flat.shape)}.')
    diff = (u_flat - u_flat.mean()) - (v_flat - v_flat.mean())
    return torch.linalg.vector_norm(diff, ord=2)


def first_difference_l2_torch(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Torch implementation of the unnormalized first-difference L2 distance."""

    if not isinstance(u, torch.Tensor) or not isinstance(v, torch.Tensor):
        raise TypeError('first_difference_l2_torch expects torch.Tensor inputs.')
    u_flat = u.reshape(-1)
    v_flat = v.reshape(-1)
    if u_flat.shape != v_flat.shape:
        raise ValueError(f'Curve shape mismatch: {tuple(u_flat.shape)} vs {tuple(v_flat.shape)}.')
    if int(u_flat.numel()) < 2:
        return u_flat.new_zeros(())
    diff = torch.diff(u_flat) - torch.diff(v_flat)
    return torch.linalg.vector_norm(diff, ord=2)


def curve_pointwise_distance_torch(u: torch.Tensor, v: torch.Tensor, *, metric: str = 'centered_l2') -> torch.Tensor:
    """Dispatch a torch curve distance by metric name."""

    token = _normalize_curve_distance_metric(metric)
    if token == 'centered_l2':
        return centered_pointwise_l2_torch(u, v)
    if token == 'diff_l2':
        return first_difference_l2_torch(u, v)
    raise AssertionError(f'Unhandled curve distance metric: {token!r}.')


def curve_pointwise_distance(u: torch.Tensor | np.ndarray, v: torch.Tensor | np.ndarray, *, metric: str = 'centered_l2') -> float:
    """Dispatch a numpy-compatible curve distance by metric name."""

    token = _normalize_curve_distance_metric(metric)
    if token == 'centered_l2':
        return centered_pointwise_l2(u, v)
    if token == 'diff_l2':
        return first_difference_l2(u, v)
    raise AssertionError(f'Unhandled curve distance metric: {token!r}.')


def representative_curve_stack_from_values_torch(values: torch.Tensor, *, reducer: str) -> torch.Tensor:
    """Reduce one ``(samples, rows, axis)`` tensor to one ``(samples, axis)`` stack."""

    curves = _ensure_rank3_curve_stack(values)
    return _reduce_rows(curves, reducer)


def representative_curve_from_values_torch(values: torch.Tensor, *, reducer: str) -> torch.Tensor:
    """Reduce one ``(samples, rows, axis)`` tensor to one batch-mean curve."""

    return representative_curve_stack_from_values_torch(values, reducer=reducer).mean(dim=0)


def _representative_psd_values_from_maps(
    signal_maps: torch.Tensor,
    *,
    centering: str,
    curve_space: str,
    userbin_edges: Sequence[float] | None,
) -> torch.Tensor:
    maps = _ensure_maps(signal_maps)
    variant_maps = _variant_maps(maps, centering)
    freqs, exact_psd = exact_periodogram_from_maps(variant_maps)
    curve_space_token = str(curve_space).strip().lower()
    if curve_space_token == 'exact':
        return exact_psd.real
    if curve_space_token == 'userbin':
        if userbin_edges is None:
            raise ValueError('userbin_edges must be provided when curve_space=userbin.')
        userbin_psd, _edges, _centers = aggregate_userbins_torch(exact_psd, freqs, userbin_edges, axis=2)
        return userbin_psd.real
    raise ValueError(f'Unsupported curve_space: {curve_space!r}. Allowed values: {", ".join(_CURVE_SPACES)}.')


def representative_psd_curve_stack_from_maps_torch(
    signal_maps: torch.Tensor,
    *,
    reducer: str,
    centering: str = 'raw',
    scale: str = 'raw',
    curve_space: str = 'exact',
    userbin_edges: Sequence[float] | None = None,
) -> torch.Tensor:
    """Return one sample-wise representative PSD-curve stack.

    ``curve_space='exact'`` returns ``(samples, freq_bins)`` curves.
    ``curve_space='userbin'`` returns ``(samples, userbins)`` curves.
    """

    values = _representative_psd_values_from_maps(
        signal_maps,
        centering=centering,
        curve_space=curve_space,
        userbin_edges=userbin_edges,
    )
    curves = representative_curve_stack_from_values_torch(values, reducer=reducer)
    return _apply_curve_scale(curves, scale)


def representative_psd_curve_from_maps_torch(
    signal_maps: torch.Tensor,
    *,
    reducer: str,
    centering: str = 'raw',
    scale: str = 'raw',
    curve_space: str = 'exact',
    userbin_edges: Sequence[float] | None = None,
) -> torch.Tensor:
    """Return one batch-mean representative PSD curve from row signals."""

    values = _representative_psd_values_from_maps(
        signal_maps,
        centering=centering,
        curve_space=curve_space,
        userbin_edges=userbin_edges,
    )
    batch_curve = representative_curve_stack_from_values_torch(values, reducer=reducer).mean(dim=0)
    return _apply_curve_scale(batch_curve, scale)


def representative_psd_minibatch_curve_from_maps_torch(
    signal_maps: torch.Tensor,
    *,
    reducer: str,
    centering: str = 'raw',
    scale: str = 'raw',
    curve_space: str = 'exact',
    userbin_edges: Sequence[float] | None = None,
) -> torch.Tensor:
    """Return the minibatch-level PSD curve used by training regularization.

    The row reducer is applied after PSD computation for each sample, then the
    minibatch mean is taken, and only then the selected value scale is applied.
    """

    values = _representative_psd_values_from_maps(
        signal_maps,
        centering=centering,
        curve_space=curve_space,
        userbin_edges=userbin_edges,
    )
    batch_curve = representative_curve_stack_from_values_torch(values, reducer=reducer).mean(dim=0)
    return _apply_curve_scale(batch_curve, scale)


def representative_psd_curves_from_maps_torch(
    signal_maps: torch.Tensor,
    *,
    reducers: Sequence[str] = _REDUCERS,
    centering: str = 'raw',
    scale: str = 'raw',
    curve_space: str = 'exact',
    userbin_edges: Sequence[float] | None = None,
) -> dict[str, torch.Tensor]:
    """Return one reducer->curve mapping for PSD-first representative curves."""

    return {
        str(reducer): representative_psd_curve_from_maps_torch(
            signal_maps,
            reducer=str(reducer),
            centering=centering,
            scale=scale,
            curve_space=curve_space,
            userbin_edges=userbin_edges,
        )
        for reducer in reducers
    }


def _row_dispersion_curves(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    row_var = values.var(dim=1, unbiased=False)
    row_med = values.median(dim=1, keepdim=True).values
    row_mad = (values - row_med).abs().median(dim=1).values
    return row_var.mean(dim=0), row_mad.mean(dim=0)


def _to_numpy(value: torch.Tensor | None) -> np.ndarray | None:
    if value is None:
        return None
    return value.detach().cpu().numpy()


def compute_family_spectral_summary(
    signal_maps: torch.Tensor,
    *,
    window: int | None,
    overlap: int | None,
    userbin_edges: Sequence[float] | None = None,
    include_spectrogram: bool = True,
    include_userbin: bool = False,
) -> dict[str, Any]:
    """Compute the core PSD-first family summary on the active device."""

    if bool(include_userbin):
        raise ValueError('psd_userbin is disabled for analysis summaries; use psd_exact only.')

    maps = _ensure_maps(signal_maps)
    num_samples, num_rows, sequence_length = [int(v) for v in maps.shape]
    summary: dict[str, Any] = {
        'num_samples': int(num_samples),
        'num_rows': int(num_rows),
        'sequence_length': int(sequence_length),
        'representative_reducers': list(_REDUCERS),
        'curve_extractors': ['psd_exact'],
        'spectrogram_saved': bool(include_spectrogram),
        'representative': {reducer: {} for reducer in _REDUCERS},
        'dispersion': {},
    }

    freq_ref: torch.Tensor | None = None
    frame_axis_ref: torch.Tensor | None = None

    for centering in _CENTERINGS:
        variant_maps = _variant_maps(maps, centering)
        freqs, exact_psd = exact_periodogram_from_maps(variant_maps)
        if freq_ref is None:
            freq_ref = freqs

        exact_values = exact_psd.real

        for reducer in _REDUCERS:
            exact_stack = representative_curve_stack_from_values_torch(exact_values, reducer=reducer)
            rep_exact_raw = exact_stack.mean(dim=0)
            for scale in _SCALES:
                rep_exact = _apply_curve_scale(rep_exact_raw, scale)
                summary['representative'][reducer].setdefault('psd_exact', {}).setdefault(scale, {})[centering] = _to_numpy(rep_exact)
        var_exact_raw, mad_exact_raw = _row_dispersion_curves(exact_values)
        for scale in _SCALES:
            summary['dispersion'].setdefault('psd_exact', {}).setdefault('variance', {}).setdefault(scale, {})[centering] = _to_numpy(_apply_curve_scale(var_exact_raw, scale))
            summary['dispersion'].setdefault('psd_exact', {}).setdefault('mad', {}).setdefault(scale, {})[centering] = _to_numpy(_apply_curve_scale(mad_exact_raw, scale))
        if include_spectrogram:
            spec_freqs, frame_centers, exact_spec = exact_spectrogram_from_maps(variant_maps, window=window, overlap=overlap)
            if frame_axis_ref is None:
                frame_axis_ref = frame_centers
            for reducer in _REDUCERS:
                rep_spec_exact = _reduce_rows(exact_spec.real, reducer).mean(dim=0)
                summary['representative'][reducer].setdefault('spectrogram_exact', {})[centering] = _to_numpy(rep_spec_exact)
    if freq_ref is None:
        raise RuntimeError('Failed to compute family spectral summary.')

    summary['freq'] = _to_numpy(freq_ref)
    summary['userbin_edges'] = None
    summary['userbin_centers'] = None
    summary['frame_axis'] = _to_numpy(frame_axis_ref)
    return summary


def representative_curve_from_summary(
    summary: dict[str, Any],
    *,
    reducer: str,
    extractor: str,
    centering: str = 'raw',
    scale: str = 'raw',
) -> np.ndarray:
    payload = summary['representative'][str(reducer)][str(extractor)]
    scale_token = str(scale)
    centering_token = _normalize_centering(centering)
    if scale_token in payload:
        return np.asarray(payload[scale_token][centering_token], dtype=np.float64)
    return np.asarray(payload[centering_token], dtype=np.float64)


def curve_axis_from_summary(summary: dict[str, Any], extractor: str) -> np.ndarray:
    extractor = str(extractor)
    if extractor == 'psd_exact':
        return np.asarray(summary['freq'], dtype=np.float64)
    if extractor == 'psd_userbin':
        return np.asarray(summary['userbin_centers'], dtype=np.float64)
    raise ValueError(f'Unsupported curve extractor: {extractor!r}.')


def _distance_key_for_variant(*, metric: str, centering: str) -> str:
    token = _normalize_curve_distance_metric(metric)
    suffix = 'cen' if _normalize_centering(centering) == 'cen' else 'raw'
    if token == 'centered_l2':
        return f'distance_{suffix}'
    return f'{token}_{suffix}'


def pair_distance_from_summaries(
    left_summary: dict[str, Any],
    right_summary: dict[str, Any],
    *,
    reducer: str,
    extractor: str,
    scale: str = 'raw',
    distance_metric: str = 'centered_l2',
) -> dict[str, Any]:
    metric_token = _normalize_curve_distance_metric(distance_metric)
    left_raw = representative_curve_from_summary(left_summary, reducer=reducer, extractor=extractor, centering='raw', scale=scale)
    right_raw = representative_curve_from_summary(right_summary, reducer=reducer, extractor=extractor, centering='raw', scale=scale)
    left_cen = representative_curve_from_summary(left_summary, reducer=reducer, extractor=extractor, centering='cen', scale=scale)
    right_cen = representative_curve_from_summary(right_summary, reducer=reducer, extractor=extractor, centering='cen', scale=scale)
    return {
        _distance_key_for_variant(metric=metric_token, centering='raw'): float(curve_pointwise_distance(left_raw, right_raw, metric=metric_token)),
        _distance_key_for_variant(metric=metric_token, centering='cen'): float(curve_pointwise_distance(left_cen, right_cen, metric=metric_token)),
        'reference_curve_axis': curve_axis_from_summary(left_summary, extractor),
    }


def family_self_drift_from_summaries(
    prev_summary: dict[str, Any],
    curr_summary: dict[str, Any],
    *,
    reducer: str,
    extractor: str,
    scale: str = 'raw',
    distance_metric: str = 'centered_l2',
) -> dict[str, Any]:
    return pair_distance_from_summaries(prev_summary, curr_summary, reducer=reducer, extractor=extractor, scale=scale, distance_metric=distance_metric)


__all__ = [
    'centered_pointwise_l2_torch',
    'curve_pointwise_distance',
    'curve_pointwise_distance_torch',
    'compute_family_spectral_summary',
    'curve_axis_from_summary',
    'family_self_drift_from_summaries',
    'first_difference_l2_torch',
    'pair_distance_from_summaries',
    'representative_curve_from_summary',
    'representative_curve_from_values_torch',
    'representative_curve_stack_from_values_torch',
    'representative_psd_curve_from_maps_torch',
    'representative_psd_minibatch_curve_from_maps_torch',
    'representative_psd_curve_stack_from_maps_torch',
    'representative_psd_curves_from_maps_torch',
]
