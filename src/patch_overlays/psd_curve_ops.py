"""Runtime PSD curve computations used by the token overlay patches."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.patch_overlays.psd_curve_config import PSDCurveSpec, normalize_userbin_reducer
from src.signal.psd_utils import apply_centering, exact_periodogram_from_maps, power_to_db


def _row_reduce(values: torch.Tensor, reducer: str) -> torch.Tensor:
    if reducer == 'mean':
        return values.mean(dim=1)
    if reducer == 'median':
        return values.median(dim=1).values
    raise ValueError(f'Unsupported row reducer: {reducer!r}.')


def _dispersion(values: torch.Tensor, metric: str) -> torch.Tensor:
    if metric == 'variance':
        return values.var(dim=1, unbiased=False)
    if metric == 'mad':
        med = values.median(dim=1, keepdim=True).values
        return (values - med).abs().median(dim=1).values
    raise ValueError(f'Unsupported dispersion metric: {metric!r}.')


def _scale(values: torch.Tensor, scale: str) -> torch.Tensor:
    return values if scale == 'raw' else power_to_db(values)


def _userbin_reduce(values: torch.Tensor, freqs: torch.Tensor, edges: Sequence[float], reducer: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reducer = normalize_userbin_reducer(reducer)
    edges_t = torch.as_tensor([float(v) for v in edges], device=values.device, dtype=values.dtype)
    centers = 0.5 * (edges_t[:-1] + edges_t[1:])
    out = []
    for i in range(int(edges_t.numel()) - 1):
        left = edges_t[i]
        right = edges_t[i + 1]
        if i == int(edges_t.numel()) - 2:
            mask = (freqs >= left) & (freqs <= right)
        else:
            mask = (freqs >= left) & (freqs < right)
        if int(mask.sum().item()) < 1:
            raise ValueError(f'userbin [{float(left):.8g}, {float(right):.8g}] contains no native normalized frequency bin.')
        chunk = values[..., mask]
        if reducer == 'mean':
            out.append(chunk.mean(dim=-1))
        elif reducer == 'median':
            out.append(chunk.median(dim=-1).values)
        else:
            out.append(chunk.sum(dim=-1))
    return torch.stack(out, dim=-1), edges_t, centers


def psd_values_for_spec(
    maps: torch.Tensor,
    spec: PSDCurveSpec,
    *,
    userbin_edges: Sequence[float] | None = None,
    userbin_reducer: str = 'mean',
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if maps.ndim != 3:
        raise ValueError(f'Expected PSD maps with shape (samples, rows, time), got {tuple(maps.shape)}.')
    x = maps if spec.centering == 'raw' else apply_centering(maps)
    freqs, exact = exact_periodogram_from_maps(x)
    values = exact.real
    if spec.extractor == 'psd_exact':
        return values, freqs, None
    if userbin_edges is None:
        raise ValueError(f'PSD token {spec.token!r} requires userbin edges.')
    binned, edges, centers = _userbin_reduce(values, freqs, userbin_edges, userbin_reducer)
    return binned, centers, edges


def representative_curve_tensor(
    maps: torch.Tensor,
    spec: PSDCurveSpec,
    *,
    userbin_edges: Sequence[float] | None = None,
    userbin_reducer: str = 'mean',
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    values, axis, edges = psd_values_for_spec(maps, spec, userbin_edges=userbin_edges, userbin_reducer=userbin_reducer)
    curve = _row_reduce(values, spec.reducer).mean(dim=0)
    return _scale(curve, spec.scale), axis, edges


def curve_distance(u: torch.Tensor | np.ndarray, v: torch.Tensor | np.ndarray, metric: str) -> float:
    u_arr = np.asarray(u.detach().cpu().numpy() if isinstance(u, torch.Tensor) else u, dtype=np.float64).reshape(-1)
    v_arr = np.asarray(v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v, dtype=np.float64).reshape(-1)
    if u_arr.shape != v_arr.shape:
        raise ValueError(f'Curve shape mismatch: {u_arr.shape} vs {v_arr.shape}.')
    if metric == 'centered_l2':
        diff = (u_arr - u_arr.mean()) - (v_arr - v_arr.mean())
    elif metric == 'diff_l2':
        if u_arr.size < 2:
            return 0.0
        diff = np.diff(u_arr) - np.diff(v_arr)
    else:
        raise ValueError(f'Unsupported curve distance metric: {metric!r}.')
    return float(np.linalg.norm(diff, ord=2))


def row_value_unit(scale: str, metric: str | None = None) -> str:
    if scale == 'db':
        return 'dB'
    if metric == 'variance':
        return 'power^2'
    return 'power'


def curve_rows_for_maps(
    *,
    common_row,
    base: Mapping[str, Any],
    maps: torch.Tensor,
    specs: Sequence[PSDCurveSpec],
    userbin_edges: Sequence[float] | None,
    userbin_reducer: str,
    category: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, np.ndarray]]:
    curve_rows: list[dict[str, str]] = []
    dispersion_rows: list[dict[str, str]] = []
    curves: dict[str, np.ndarray] = {}
    for spec in specs:
        values, axis, edges = psd_values_for_spec(maps, spec, userbin_edges=userbin_edges, userbin_reducer=userbin_reducer)
        rep = _scale(_row_reduce(values, spec.reducer).mean(dim=0), spec.scale)
        rep_np = rep.detach().cpu().numpy().reshape(-1)
        axis_np = axis.detach().cpu().numpy().reshape(-1)
        edge_np = None if edges is None else edges.detach().cpu().numpy().reshape(-1)
        curves[spec.token] = rep_np
        for i, value in enumerate(rep_np):
            kwargs = dict(base)
            kwargs.update(
                category=category,
                extractor=spec.extractor,
                reducer=spec.reducer,
                variant=spec.centering,
                scale=spec.scale,
                psd_token=spec.token,
                userbin_reducer=userbin_reducer if spec.extractor == 'psd_userbin' else '',
                frequency=float(axis_np[i]) if i < len(axis_np) else '',
                frequency_bin=int(i),
                frequency_unit='normalized_frequency',
                value=float(value),
                value_unit=row_value_unit(spec.scale),
            )
            if edge_np is not None and i + 1 < len(edge_np):
                kwargs['bin_left'] = float(edge_np[i])
                kwargs['bin_right'] = float(edge_np[i + 1])
            curve_rows.append(common_row(**kwargs))
        for metric in ('variance', 'mad'):
            disp = _scale(_dispersion(values, metric).mean(dim=0), spec.scale).detach().cpu().numpy().reshape(-1)
            for i, value in enumerate(disp):
                kwargs = dict(base)
                kwargs.update(
                    category='dataset_dispersion' if category == 'dataset_curve' else 'analysis_dispersion',
                    extractor=spec.extractor,
                    variant=spec.centering,
                    scale=spec.scale,
                    psd_token=spec.token,
                    userbin_reducer=userbin_reducer if spec.extractor == 'psd_userbin' else '',
                    statistic=metric,
                    frequency=float(axis_np[i]) if i < len(axis_np) else '',
                    frequency_unit='normalized_frequency',
                    value=float(value),
                    value_unit=row_value_unit(spec.scale, metric),
                )
                if edge_np is not None and i + 1 < len(edge_np):
                    kwargs['bin_left'] = float(edge_np[i])
                    kwargs['bin_right'] = float(edge_np[i + 1])
                dispersion_rows.append(common_row(**kwargs))
    return curve_rows, dispersion_rows, curves


def token_distance_rows(
    *,
    common_row,
    base: Mapping[str, Any],
    curves: Mapping[str, np.ndarray],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for left, right in combinations(sorted(curves), 2):
        if np.asarray(curves[left]).shape != np.asarray(curves[right]).shape:
            continue
        for metric in ('centered_l2', 'diff_l2'):
            kwargs = dict(base)
            kwargs.update(
                category='psd_curve_distance',
                left_psd_token=left,
                right_psd_token=right,
                distance_metric=metric,
                value=curve_distance(curves[left], curves[right], metric),
                value_unit='dimensionless',
            )
            rows.append(common_row(**kwargs))
    return rows
