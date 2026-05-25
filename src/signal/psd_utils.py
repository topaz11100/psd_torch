"""GPU-first exact Hann-windowed spectral primitives and analysis helpers."""

from __future__ import annotations

import json
import os
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


DB_EPSILON = 1.0e-12
PSD_VARIANTS = ('raw', 'centered')


def _analysis_real_dtype(*, device: torch.device | None = None, like: torch.Tensor | None = None) -> torch.dtype:
    """Choose one real compute dtype that favors CUDA throughput.

    The official signal-processing path is GPU-first. When tensors already live on
    CUDA we keep the compute path in ``float32`` unless the caller explicitly
    passed ``float64`` tensors. CPU fallbacks stay in ``float64``.
    """

    if like is not None:
        if like.dtype == torch.float64:
            return torch.float64
        if like.device.type == 'cuda':
            return torch.float32
        return torch.float64
    if device is not None and device.type == 'cuda':
        return torch.float32
    return torch.float64


def _as_float_tensor(
    value: torch.Tensor | np.ndarray | Sequence[float],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Internal helper for ``as float tensor`` in the ``psd_utils`` module."""
    if isinstance(value, torch.Tensor):
        resolved_dtype = dtype or _analysis_real_dtype(like=value)
        return value.to(device=device or value.device, dtype=resolved_dtype)
    resolved_device = device or torch.device('cpu')
    resolved_dtype = dtype or _analysis_real_dtype(device=resolved_device)
    return torch.as_tensor(value, device=resolved_device, dtype=resolved_dtype)


def tensor_to_channel_major_maps(
    batch_sequence: torch.Tensor,
    *,
    expected_time: int | None = None,
    expected_channels: int | None = None,
) -> torch.Tensor:
    """Convert tensors into official ``(samples, rows, time)`` PSD maps.

    The Spec treats the last PSD axis as the only time/frequency axis.
    For membrane or frame-like tensors, every non-time axis is flattened into the
    row axis while the explicit time axis is preserved.
    """

    if batch_sequence.ndim == 2:
        return batch_sequence.unsqueeze(1)
    if batch_sequence.ndim == 3:
        if expected_time is not None and expected_channels is not None:
            if batch_sequence.shape[1] == int(expected_time) and batch_sequence.shape[2] == int(expected_channels):
                return batch_sequence.transpose(1, 2).contiguous()
            if batch_sequence.shape[1] == int(expected_channels) and batch_sequence.shape[2] == int(expected_time):
                return batch_sequence.contiguous()
        raise ValueError(
            'Ambiguous rank-3 PSD tensor requires explicit expected_time and expected_channels. '
            f'Got shape {tuple(batch_sequence.shape)}.'
        )
    if batch_sequence.ndim == 4:
        batch, channels, height, width = [int(v) for v in batch_sequence.shape]
        return batch_sequence.reshape(batch, channels, height * width).contiguous()
    if batch_sequence.ndim == 5:
        batch, time_steps, channels, height, width = [int(v) for v in batch_sequence.shape]
        return batch_sequence.permute(0, 2, 3, 4, 1).reshape(batch, channels * height * width, time_steps).contiguous()
    raise ValueError(f'Expected a 2D, 3D, 4D, or 5D tensor, but received shape {tuple(batch_sequence.shape)}.')


def tensor_to_channel_major_maps_explicit(
    batch_sequence: torch.Tensor,
    *,
    psd_axis_kind: str,
    psd_time_axis: str | int | None,
    psd_flatten_rule: str | None = None,
    psd_logical_shape: dict[str, Any] | Sequence[int] | None = None,
    expected_time: int | None = None,
    expected_rows: int | None = None,
) -> torch.Tensor:
    """Convert prepared tensors to ``(samples, rows, time)`` without axis heuristics.

    Rank-3 tensors are never guessed by comparing axis lengths.  The caller must
    supply manifest-derived metadata that identifies whether the tensor is already
    row-major ``(B,rows,time)`` or sequence-major ``(B,T,C)``.
    """

    tensor = batch_sequence if isinstance(batch_sequence, torch.Tensor) else torch.as_tensor(batch_sequence)
    axis_kind = str(psd_axis_kind or '').strip().lower()
    time_axis = str(psd_time_axis if psd_time_axis is not None else '').strip().lower()
    flatten_rule = str(psd_flatten_rule or '').strip().lower()

    if tensor.ndim == 2:
        maps = tensor.unsqueeze(1)
    elif tensor.ndim == 3:
        row_major_markers = {'last', '-1', '2', 'time_last', 'rows_time', 'row_time', 'nrt', 'brows_time'}
        sequence_markers = {'1', 't', 'time', 'time_axis_1', 'model_time', 'sequence'}
        if time_axis in row_major_markers or flatten_rule in {'already_flattened_rows_time', 'rows_time', 'channel_major_time_last'}:
            maps = tensor.contiguous()
        elif time_axis in sequence_markers or axis_kind == 'temporal':
            maps = tensor.transpose(1, 2).contiguous()
        elif expected_time is not None and int(tensor.shape[2]) == int(expected_time):
            maps = tensor.contiguous()
        elif expected_time is not None and int(tensor.shape[1]) == int(expected_time):
            maps = tensor.transpose(1, 2).contiguous()
        else:
            raise ValueError(
                'Ambiguous rank-3 PSD tensor cannot be converted without manifest axis metadata. '
                f'psd_axis_kind={psd_axis_kind!r}, psd_time_axis={psd_time_axis!r}, '
                f'psd_flatten_rule={psd_flatten_rule!r}, shape={tuple(tensor.shape)}.'
            )
    elif tensor.ndim == 4:
        if axis_kind in {'static_repeat', 'image_temporal'}:
            raise ValueError(f'Expected rank-5 frame input or rank-3 PSD view for {axis_kind}, got {tuple(tensor.shape)}.')
        batch, channels, height, width = [int(v) for v in tensor.shape]
        maps = tensor.reshape(batch, channels * height * width, 1).contiguous()
    elif tensor.ndim == 5:
        batch, time_steps, channels, height, width = [int(v) for v in tensor.shape]
        maps = tensor.permute(0, 2, 3, 4, 1).reshape(batch, channels * height * width, time_steps).contiguous()
    else:
        raise ValueError(f'Expected a 2D, 3D, 4D, or 5D tensor, but received shape {tuple(tensor.shape)}.')

    if maps.ndim != 3:
        raise ValueError(f'PSD maps must have shape (samples, rows, time), got {tuple(maps.shape)}.')
    if expected_time is not None and int(maps.shape[-1]) != int(expected_time):
        raise ValueError(f'PSD time axis mismatch: expected {expected_time}, got {int(maps.shape[-1])}.')
    if expected_rows is not None and int(maps.shape[1]) != int(expected_rows):
        raise ValueError(f'PSD row axis mismatch: expected {expected_rows}, got {int(maps.shape[1])}.')
    return maps


def sequence_batch_to_channel_major_maps(batch_sequence: torch.Tensor) -> torch.Tensor:
    """Convert a known ``(batch, time, channels)`` sequence batch into ``(batch, rows, time)``.

    Hidden/output traces produced by the project model stack always use the
    canonical ``(B, T, C)`` layout. Using an explicit converter avoids ambiguous
    time-vs-channel heuristics, which could silently flip axes when ``C > T``.
    """

    if batch_sequence.ndim == 2:
        return batch_sequence.unsqueeze(1)
    if batch_sequence.ndim != 3:
        raise ValueError(f'Expected a 2D or 3D tensor, but received shape {tuple(batch_sequence.shape)}.')
    return batch_sequence.transpose(1, 2).contiguous()


def trace_tensor_to_channel_major_maps(batch_signal: torch.Tensor) -> torch.Tensor:
    """Convert dense/CNN trace tensors into official ``(samples, rows, time)`` maps.

    For frame/membrane tensors ``(B,T,C,H,W)``, the time axis stays last and all
    channel/spatial axes are flattened into one row axis, including ``T=1``.
    """

    if batch_signal.ndim == 2:
        return batch_signal.unsqueeze(1)
    if batch_signal.ndim == 3:
        return sequence_batch_to_channel_major_maps(batch_signal)
    if batch_signal.ndim == 4:
        batch, channels, height, width = [int(v) for v in batch_signal.shape]
        return batch_signal.reshape(batch, channels, height * width).contiguous()
    if batch_signal.ndim == 5:
        batch, time_steps, channels, height, width = [int(v) for v in batch_signal.shape]
        return batch_signal.permute(0, 2, 3, 4, 1).reshape(batch, channels * height * width, time_steps).contiguous()
    raise ValueError(f'Expected a 2D, 3D, 4D, or 5D tensor, but received shape {tuple(batch_signal.shape)}.')


def apply_centering(signal_maps: torch.Tensor) -> torch.Tensor:
    """Handle ``apply centering`` for the ``psd_utils`` module."""
    return signal_maps - signal_maps.mean(dim=-1, keepdim=True)


def hann_window(length: int, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Handle ``hann window`` for the ``psd_utils`` module."""
    length = int(length)
    if length <= 0:
        raise ValueError('Signal length must be positive.')
    if dtype is None:
        dtype = _analysis_real_dtype(device=device)
    if length == 1:
        return torch.ones(1, device=device, dtype=dtype)
    t = torch.arange(length, device=device, dtype=dtype)
    return 0.5 - 0.5 * torch.cos(2.0 * torch.pi * t / float(length - 1))


def exact_one_sided_freqs(length: int, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Handle ``exact one sided freqs`` for the ``psd_utils`` module."""
    length = int(length)
    if length <= 0:
        raise ValueError('Signal length must be positive.')
    if dtype is None:
        dtype = _analysis_real_dtype(device=device)
    return torch.arange(length // 2 + 1, device=device, dtype=dtype) / float(length)


def one_sided_scaling(length: int, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Handle ``one sided scaling`` for the ``psd_utils`` module."""
    length = int(length)
    if dtype is None:
        dtype = _analysis_real_dtype(device=device)
    freq_count = length // 2 + 1
    scale = torch.ones(freq_count, device=device, dtype=dtype)
    if length % 2 == 0:
        if freq_count > 2:
            scale[1:-1] = 2.0
    else:
        if freq_count > 1:
            scale[1:] = 2.0
    return scale


def exact_hann_rfft(signal_maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return one-sided frequency grid, complex spectrum, one-sided scale, and Hann power constant."""

    if signal_maps.ndim != 3:
        raise ValueError(f'Expected shape (samples, rows, time), got {tuple(signal_maps.shape)}')
    signal_maps = _as_float_tensor(signal_maps)
    length = int(signal_maps.shape[-1])
    device = signal_maps.device
    dtype = signal_maps.dtype
    freqs = exact_one_sided_freqs(length, device=device, dtype=dtype)
    window = hann_window(length, device=device, dtype=dtype)
    window_power = window.square().sum() / float(length)
    spectrum = torch.fft.rfft(signal_maps * window.view(1, 1, -1), dim=-1)
    scale = one_sided_scaling(length, device=device, dtype=dtype)
    return freqs, spectrum, scale, window_power


def exact_periodogram_from_maps(signal_maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact full-length Hann-windowed one-sided periodogram."""

    freqs, spectrum, scale, window_power = exact_hann_rfft(signal_maps)
    length = int(signal_maps.shape[-1])
    power = scale.view(1, 1, -1) * spectrum.abs().square() / (float(length) * window_power)
    return freqs, power.real


def exact_cross_periodogram_from_maps(x_maps: torch.Tensor, y_maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Handle ``exact cross periodogram from maps`` for the ``psd_utils`` module."""
    if x_maps.shape != y_maps.shape:
        raise ValueError(f'x_maps and y_maps shape mismatch: {tuple(x_maps.shape)} vs {tuple(y_maps.shape)}.')
    freqs, x_spec, scale, window_power = exact_hann_rfft(x_maps)
    _freqs_y, y_spec, _scale_y, _window_power_y = exact_hann_rfft(y_maps)
    length = int(x_maps.shape[-1])
    cross = scale.view(1, 1, -1) * (y_spec * x_spec.conj()) / (float(length) * window_power)
    return freqs, cross


def _resolve_spectrogram_params(length: int, window: int | None, overlap: int | None) -> tuple[int, int, list[int]]:
    """Resolve spectrogram params; ``None`` means the official full-length window."""
    length = int(length)
    if window is None:
        window = length
    if int(window) <= 0:
        window = length
    window = min(int(window), int(length))
    overlap = 0 if overlap is None else min(max(0, int(overlap)), max(0, window - 1))
    hop = max(1, window - overlap)
    starts = list(range(0, max(1, length - window + 1), hop))
    if not starts:
        starts = [0]
    if starts[-1] != max(0, length - window):
        starts.append(max(0, length - window))
    starts = sorted(set(starts))
    return window, overlap, starts


def _gather_frames(signal_maps: torch.Tensor, *, starts: Sequence[int], window: int) -> torch.Tensor:
    """Internal helper for ``gather frames`` in the ``psd_utils`` module."""
    device = signal_maps.device
    num_frames = len(starts)
    starts_t = torch.as_tensor([int(v) for v in starts], device=device, dtype=torch.long)
    offsets = torch.arange(int(window), device=device, dtype=torch.long)
    frame_index = starts_t.view(-1, 1) + offsets.view(1, -1)
    gather_index = frame_index.view(1, 1, num_frames, int(window)).expand(signal_maps.shape[0], signal_maps.shape[1], -1, -1)
    expanded = signal_maps.unsqueeze(-2).expand(-1, -1, num_frames, -1)
    return torch.gather(expanded, dim=-1, index=gather_index)


def exact_spectrogram_from_maps(signal_maps: torch.Tensor, *, window: int | None, overlap: int | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the full-length Hann spectrogram unless an internal caller supplies a window."""
    if signal_maps.ndim != 3:
        raise ValueError(f'Expected shape (samples, rows, time), got {tuple(signal_maps.shape)}')
    signal_maps = _as_float_tensor(signal_maps)
    _samples, _rows, length = [int(v) for v in signal_maps.shape]
    window, overlap, starts = _resolve_spectrogram_params(length, window, overlap)
    if window == length and len(starts) == 1:
        freqs, power = exact_periodogram_from_maps(signal_maps)
        centers = torch.tensor([(length - 1) / 2.0], device=signal_maps.device, dtype=signal_maps.dtype)
        return freqs, centers, power.unsqueeze(-1)
    framed = _gather_frames(signal_maps, starts=starts, window=window)  # (samples, rows, frames, window)
    freq_grid = exact_one_sided_freqs(window, device=signal_maps.device, dtype=signal_maps.dtype)
    taper = hann_window(window, device=signal_maps.device, dtype=signal_maps.dtype)
    window_power = taper.square().sum() / float(window)
    scale = one_sided_scaling(window, device=signal_maps.device, dtype=signal_maps.dtype)
    spectrum = torch.fft.rfft(framed * taper.view(1, 1, 1, -1), dim=-1)
    power = scale.view(1, 1, 1, -1) * spectrum.abs().square() / (float(window) * window_power)
    power = power.permute(0, 1, 3, 2).contiguous()  # (samples, rows, freq, frames)
    centers = torch.as_tensor(
        [start + 0.5 * float(window - 1) for start in starts],
        device=signal_maps.device,
        dtype=signal_maps.dtype,
    )
    return freq_grid, centers, power.real


def aggregate_userbins_torch(
    values: torch.Tensor,
    freqs: torch.Tensor,
    userbin_edges: Sequence[float],
    *,
    axis: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate frequency bins on-device without per-bin host synchronizations."""

    values = _as_float_tensor(values)
    freqs = _as_float_tensor(freqs, device=values.device, dtype=values.dtype)
    axis = int(axis)
    moved = values.movedim(axis, -1)
    edges = _as_float_tensor([float(v) for v in userbin_edges], device=values.device, dtype=values.dtype)
    if edges.ndim != 1 or edges.numel() < 2:
        raise ValueError('userbin_edges must be a one-dimensional sequence with at least two values.')

    num_bins = int(edges.numel() - 1)
    interior = edges[1:-1]
    bin_ids = torch.bucketize(freqs, interior, right=False)
    valid = (freqs >= edges[0]) & (freqs <= edges[-1])

    assignment = F.one_hot(bin_ids.clamp(min=0, max=num_bins - 1), num_classes=num_bins).to(dtype=values.dtype)
    assignment = assignment * valid.to(dtype=values.dtype).unsqueeze(-1)
    counts = assignment.sum(dim=0)
    safe_counts = torch.where(counts > 0, counts, torch.ones_like(counts))
    weights = assignment / safe_counts.unsqueeze(0)

    flat = moved.reshape(-1, moved.shape[-1])
    aggregated = flat @ weights
    aggregated = aggregated.reshape(*moved.shape[:-1], num_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return aggregated.movedim(-1, axis), edges, centers


def aggregate_userbins_numpy(values: np.ndarray, freqs: np.ndarray, userbin_edges: Sequence[float], *, axis: int = -1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate userbins numpy."""
    tensor_values = torch.as_tensor(values, dtype=torch.float64)
    tensor_freqs = torch.as_tensor(freqs, dtype=torch.float64)
    binned, edges, centers = aggregate_userbins_torch(tensor_values, tensor_freqs, userbin_edges, axis=axis)
    return binned.detach().cpu().numpy(), edges.detach().cpu().numpy(), centers.detach().cpu().numpy()


def power_to_db(values: torch.Tensor | np.ndarray, *, epsilon: float = DB_EPSILON) -> torch.Tensor | np.ndarray:
    """Handle ``power to db`` for the ``psd_utils`` module."""
    if isinstance(values, torch.Tensor):
        return 10.0 * torch.log10(torch.clamp(values, min=0.0) + float(epsilon))
    array = np.asarray(values, dtype=np.float64)
    return 10.0 * np.log10(np.clip(array, a_min=0.0, a_max=None) + float(epsilon))


def centered_pointwise_l2(u: torch.Tensor | np.ndarray, v: torch.Tensor | np.ndarray) -> float:
    """Handle ``centered pointwise l2`` for the ``psd_utils`` module."""
    u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
    v_arr = np.asarray(v, dtype=np.float64).reshape(-1)
    if u_arr.shape != v_arr.shape:
        raise ValueError(f'Curve shape mismatch: {u_arr.shape} vs {v_arr.shape}.')
    diff = (u_arr - u_arr.mean()) - (v_arr - v_arr.mean())
    return float(np.linalg.norm(diff, ord=2))


def first_difference_l2(u: torch.Tensor | np.ndarray, v: torch.Tensor | np.ndarray) -> float:
    """Return the unnormalized first-difference L2 curve distance."""
    u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
    v_arr = np.asarray(v, dtype=np.float64).reshape(-1)
    if u_arr.shape != v_arr.shape:
        raise ValueError(f'Curve shape mismatch: {u_arr.shape} vs {v_arr.shape}.')
    if u_arr.size < 2:
        return 0.0
    diff = np.diff(u_arr) - np.diff(v_arr)
    return float(np.linalg.norm(diff, ord=2))


def centered_rowwise_l2(u: torch.Tensor | np.ndarray, v: torch.Tensor | np.ndarray) -> np.ndarray:
    """Handle ``centered rowwise l2`` for the ``psd_utils`` module."""
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    if u_arr.shape != v_arr.shape:
        raise ValueError(f'Row-wise curve shape mismatch: {u_arr.shape} vs {v_arr.shape}.')
    if u_arr.ndim != 2:
        raise ValueError(f'Expected rank-2 matrices for centered_rowwise_l2, got {u_arr.shape}.')
    centered = (u_arr - u_arr.mean(axis=1, keepdims=True)) - (v_arr - v_arr.mean(axis=1, keepdims=True))
    return np.linalg.norm(centered, axis=1)


def time_domain_summary_from_maps(signal_maps: torch.Tensor) -> dict[str, Any]:
    """Handle ``time domain summary from maps`` for the ``psd_utils`` module."""
    if signal_maps.ndim != 3:
        raise ValueError(f'Expected shape (samples, rows, time), got {tuple(signal_maps.shape)}')
    maps = _as_float_tensor(signal_maps)
    centered = apply_centering(maps)
    raw_heatmap = maps.mean(dim=0)
    centered_heatmap = centered.mean(dim=0)
    raw_mean = maps.mean(dim=(0, 1))
    centered_mean = centered.mean(dim=(0, 1))
    return {
        'time': np.arange(int(maps.shape[-1]), dtype=np.float64),
        'time_domain_heatmap': raw_heatmap.detach().cpu().numpy(),
        'time_domain_mean': raw_mean.detach().cpu().numpy(),
        'time_domain_heatmap_raw': raw_heatmap.detach().cpu().numpy(),
        'time_domain_heatmap_centered': centered_heatmap.detach().cpu().numpy(),
        'time_domain_mean_raw': raw_mean.detach().cpu().numpy(),
        'time_domain_mean_centered': centered_mean.detach().cpu().numpy(),
    }


def scalar_representative_maps(signal_maps: torch.Tensor, *, reducer: str) -> torch.Tensor:
    """Handle ``scalar representative maps`` for the ``psd_utils`` module."""
    reducer = str(reducer).strip().lower()
    if signal_maps.ndim != 3:
        raise ValueError(f'Expected shape (samples, rows, time), got {tuple(signal_maps.shape)}')
    if reducer == 'mean':
        return signal_maps.mean(dim=1, keepdim=True)
    if reducer == 'median':
        return signal_maps.median(dim=1, keepdim=True).values
    raise ValueError(f'Unsupported representative reducer: {reducer}')


def pca_dim_from_cli_vector(
    cli_dims: Sequence[int] | Sequence[str] | None,
    layer_index_zero_based: int,
    row_count: int,
) -> int:
    """Resolve one layer PCA dimension from CLI vector semantics."""
    rows = max(1, int(row_count))
    layer_index = max(0, int(layer_index_zero_based))
    if cli_dims is None or len(cli_dims) == 0:
        return max(1, min(rows, 4))
    dims = [int(v) for v in cli_dims]
    selected = dims[layer_index] if layer_index < len(dims) else dims[-1]
    return max(1, min(rows, int(selected)))


def compute_fixed_pca_basis(reference_maps: torch.Tensor, target_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute fixed PCA basis and centroid from ``(samples, rows, time)`` maps."""
    if reference_maps.ndim != 3:
        raise ValueError(f'Expected shape (samples, rows, time), got {tuple(reference_maps.shape)}')
    maps = _as_float_tensor(reference_maps)
    samples, rows, time_steps = [int(v) for v in maps.shape]
    observations = max(1, samples * time_steps)
    dim = max(1, min(int(target_dim), rows, observations))
    obs = maps.permute(0, 2, 1).reshape(observations, rows)
    centroid = obs.mean(dim=0)
    centered = obs - centroid
    try:
        _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
        basis = vh[:dim, :].transpose(0, 1).contiguous()
    except Exception:
        cov = centered.transpose(0, 1) @ centered / float(max(1, observations - 1))
        evals, evecs = torch.linalg.eigh(cov)
        order = torch.argsort(evals, descending=True)
        basis = evecs[:, order[:dim]].contiguous()
    return basis.detach(), centroid.detach()


def apply_fixed_pca_basis(signal_maps: torch.Tensor, basis: torch.Tensor, centroid: torch.Tensor) -> torch.Tensor:
    """Apply a fixed PCA basis to ``(samples, rows, time)`` maps -> ``(samples, dim, time)``."""
    if signal_maps.ndim != 3:
        raise ValueError(f'Expected shape (samples, rows, time), got {tuple(signal_maps.shape)}')
    maps = _as_float_tensor(signal_maps)
    rows = int(maps.shape[1])
    basis_local = torch.as_tensor(basis, device=maps.device, dtype=maps.dtype)
    centroid_local = torch.as_tensor(centroid, device=maps.device, dtype=maps.dtype).reshape(-1)
    if basis_local.ndim != 2:
        raise ValueError(f'basis must be rank-2 (rows, dim), got {tuple(basis_local.shape)}')
    if int(basis_local.shape[0]) != rows:
        raise ValueError(f'basis row mismatch: expected {rows}, got {int(basis_local.shape[0])}')
    if int(centroid_local.numel()) != rows:
        raise ValueError(f'centroid row mismatch: expected {rows}, got {int(centroid_local.numel())}')
    centered = maps - centroid_local.view(1, rows, 1)
    return torch.einsum('srt,rd->sdt', centered, basis_local).contiguous()


def row_variance_mad_summary(signal_maps: torch.Tensor) -> dict[str, float]:
    """Handle ``row variance mad summary`` for the ``psd_utils`` module."""
    if signal_maps.ndim != 3:
        raise ValueError(f'Expected shape (samples, rows, time), got {tuple(signal_maps.shape)}')
    maps = _as_float_tensor(signal_maps)
    row_var = maps.var(dim=1, unbiased=False)
    row_median = maps.median(dim=1, keepdim=True).values
    row_mad = (maps - row_median).abs().median(dim=1).values
    return {
        'row_variance_mean': float(row_var.mean().item()),
        'row_mad_mean': float(row_mad.mean().item()),
    }


def scalar_io_spectral_objects(x_maps: torch.Tensor, y_maps: torch.Tensor, *, epsilon: float = DB_EPSILON) -> dict[str, Any]:
    """Handle ``scalar io spectral objects`` for the ``psd_utils`` module."""
    x_maps = tensor_to_channel_major_maps(x_maps) if x_maps.ndim == 2 else x_maps
    y_maps = tensor_to_channel_major_maps(y_maps) if y_maps.ndim == 2 else y_maps
    if x_maps.ndim != 3 or y_maps.ndim != 3:
        raise ValueError('scalar_io_spectral_objects expects rank-3 maps.')
    if int(x_maps.shape[1]) != 1 or int(y_maps.shape[1]) != 1:
        raise ValueError('scalar_io_spectral_objects expects scalar maps with one row.')
    payload: dict[str, Any] = {}
    for variant in PSD_VARIANTS:
        x_variant = x_maps if variant == 'raw' else apply_centering(x_maps)
        y_variant = y_maps if variant == 'raw' else apply_centering(y_maps)
        freqs, s_xx_each = exact_periodogram_from_maps(x_variant)
        _freqs_y, s_yy_each = exact_periodogram_from_maps(y_variant)
        _freqs_xy, s_yx_each = exact_cross_periodogram_from_maps(x_variant, y_variant)
        s_xx = s_xx_each.mean(dim=(0, 1)).real
        s_yy = s_yy_each.mean(dim=(0, 1)).real
        s_yx = s_yx_each.mean(dim=(0, 1))
        coherence = s_yx.abs().square() / torch.clamp(s_xx * s_yy, min=float(epsilon))
        H = s_yx / torch.clamp(s_xx, min=float(epsilon))
        payload.update(
            {
                f'freq_{variant}': freqs.detach().cpu().numpy(),
                f'S_xx_{variant}': s_xx.detach().cpu().numpy(),
                f'S_yy_{variant}': s_yy.detach().cpu().numpy(),
                f'S_yx_mag_{variant}': s_yx.abs().detach().cpu().numpy(),
                f'S_yx_phase_{variant}': torch.angle(s_yx).detach().cpu().numpy(),
                f'coherence_{variant}': coherence.real.detach().cpu().numpy(),
                f'H_mag_{variant}': H.abs().detach().cpu().numpy(),
                f'H_phase_{variant}': torch.angle(H).detach().cpu().numpy(),
            }
        )
    return payload


def _variant_maps(signal_maps: torch.Tensor, variant: str) -> torch.Tensor:
    """Internal helper for ``variant maps`` in the ``psd_utils`` module."""
    return signal_maps if variant == 'raw' else apply_centering(signal_maps)


def _sample_reduce_rows(values: torch.Tensor, reducer: str) -> torch.Tensor:
    """Reduce one ``(samples, rows, ...)`` tensor over the row axis only."""

    token = str(reducer).strip().lower()
    if token == 'mean':
        return values.mean(dim=1)
    if token == 'median':
        return values.median(dim=1).values
    raise ValueError(f'Unsupported reducer: {reducer!r}.')


def combined_exact_psd_payload_from_maps_torch(
    signal_maps: torch.Tensor,
    *,
    window: int | None,
    overlap: int | None,
    userbin_edges: Sequence[float],
    include_spectrogram: bool = True,
) -> dict[str, Any]:
    """Compute the official raw/centered dataset-input spectral payload on GPU when available."""

    if signal_maps.ndim != 3:
        raise ValueError(f'Expected shape (samples, rows, time), got {tuple(signal_maps.shape)}')
    maps = _as_float_tensor(signal_maps)
    num_samples, num_rows, sequence_length = [int(v) for v in maps.shape]
    payload: dict[str, Any] = {
        'variants_saved': list(PSD_VARIANTS),
        'taper_window_applied': True,
        'taper_window_name': 'hann',
        'db_plots_saved': True,
        'db_plot_scale': '10log10_power_plus_epsilon',
        'db_plot_epsilon': DB_EPSILON,
        'num_samples': num_samples,
        'num_rows': num_rows,
        'sequence_length': sequence_length,
        'userbin_edges': [float(v) for v in userbin_edges],
        'spectrogram_saved': bool(include_spectrogram),
    }

    for variant in PSD_VARIANTS:
        variant_maps = _variant_maps(maps, variant)
        freq_exact, exact_psd = exact_periodogram_from_maps(variant_maps)
        userbin_psd, edges, centers = aggregate_userbins_torch(exact_psd, freq_exact, userbin_edges, axis=2)

        mean_waveform_exact = _sample_reduce_rows(exact_psd.real, 'mean').mean(dim=0)
        median_waveform_exact = _sample_reduce_rows(exact_psd.real, 'median').mean(dim=0)
        mean_waveform_userbin = _sample_reduce_rows(userbin_psd.real, 'mean').mean(dim=0)
        median_waveform_userbin = _sample_reduce_rows(userbin_psd.real, 'median').mean(dim=0)
        row_mean_psd_exact = exact_psd.real.mean(dim=0)

        payload[f'freq_exact_{variant}'] = freq_exact.detach().cpu().numpy()
        payload[f'userbin_edges_{variant}'] = edges.detach().cpu().numpy()
        payload[f'userbin_centers_{variant}'] = centers.detach().cpu().numpy()
        payload[f'mean_psd_waveform_exact_{variant}'] = mean_waveform_exact.detach().cpu().numpy()
        payload[f'median_psd_waveform_exact_{variant}'] = median_waveform_exact.detach().cpu().numpy()
        payload[f'mean_psd_waveform_userbin_{variant}'] = mean_waveform_userbin.detach().cpu().numpy()
        payload[f'median_psd_waveform_userbin_{variant}'] = median_waveform_userbin.detach().cpu().numpy()
        payload[f'row_mean_psd_exact_{variant}'] = row_mean_psd_exact.detach().cpu().numpy()
        payload[f'mean_psd_waveform_exact_{variant}_db'] = power_to_db(mean_waveform_exact).detach().cpu().numpy()
        payload[f'median_psd_waveform_exact_{variant}_db'] = power_to_db(median_waveform_exact).detach().cpu().numpy()
        payload[f'mean_psd_waveform_userbin_{variant}_db'] = power_to_db(mean_waveform_userbin).detach().cpu().numpy()
        payload[f'median_psd_waveform_userbin_{variant}_db'] = power_to_db(median_waveform_userbin).detach().cpu().numpy()
        payload[f'scalar_summary_{variant}'] = json.dumps(
            {
                'mean_psd_waveform_mean': float(mean_waveform_exact.mean().item()),
                'mean_psd_waveform_std': float(mean_waveform_exact.std(unbiased=False).item()),
                'median_psd_waveform_mean': float(median_waveform_exact.mean().item()),
                'median_psd_waveform_std': float(median_waveform_exact.std(unbiased=False).item()),
            },
            ensure_ascii=False,
        )

        if include_spectrogram:
            spec_freqs, frame_centers, spectrogram = exact_spectrogram_from_maps(variant_maps, window=window, overlap=overlap)
            userbin_spec, _spec_edges, spec_centers = aggregate_userbins_torch(spectrogram, spec_freqs, userbin_edges, axis=2)

            mean_spectrogram_exact = _sample_reduce_rows(spectrogram.real, 'mean').mean(dim=0)
            median_spectrogram_exact = _sample_reduce_rows(spectrogram.real, 'median').mean(dim=0)
            mean_spectrogram_userbin = _sample_reduce_rows(userbin_spec.real, 'mean').mean(dim=0)
            median_spectrogram_userbin = _sample_reduce_rows(userbin_spec.real, 'median').mean(dim=0)

            payload[f'spectrogram_freqs_{variant}'] = spec_freqs.detach().cpu().numpy()
            payload[f'spectrogram_frame_centers_{variant}'] = frame_centers.detach().cpu().numpy()
            payload[f'spectrogram_userbin_centers_{variant}'] = spec_centers.detach().cpu().numpy()
            payload[f'mean_spectrogram_exact_{variant}'] = mean_spectrogram_exact.detach().cpu().numpy()
            payload[f'median_spectrogram_exact_{variant}'] = median_spectrogram_exact.detach().cpu().numpy()
            payload[f'mean_spectrogram_userbin_{variant}'] = mean_spectrogram_userbin.detach().cpu().numpy()
            payload[f'median_spectrogram_userbin_{variant}'] = median_spectrogram_userbin.detach().cpu().numpy()
            payload[f'mean_spectrogram_exact_{variant}_db'] = power_to_db(mean_spectrogram_exact).detach().cpu().numpy()
            payload[f'median_spectrogram_exact_{variant}_db'] = power_to_db(median_spectrogram_exact).detach().cpu().numpy()
            payload[f'mean_spectrogram_userbin_{variant}_db'] = power_to_db(mean_spectrogram_userbin).detach().cpu().numpy()
            payload[f'median_spectrogram_userbin_{variant}_db'] = power_to_db(median_spectrogram_userbin).detach().cpu().numpy()
    return payload


def combined_exact_psd_payload_from_map_batches_torch(
    signal_map_batches: Iterable[torch.Tensor],
    *,
    window: int | None,
    overlap: int | None,
    userbin_edges: Sequence[float],
    include_spectrogram: bool = True,
) -> dict[str, Any]:
    """Compute the official dataset-input PSD bundle exactly from channel-major batches."""

    num_samples_total = 0
    num_rows: int | None = None
    sequence_length: int | None = None
    rep_exact_curve_sums: dict[str, dict[str, torch.Tensor | None]] = {
        variant: {'mean': None, 'median': None} for variant in PSD_VARIANTS
    }
    rep_userbin_curve_sums: dict[str, dict[str, torch.Tensor | None]] = {
        variant: {'mean': None, 'median': None} for variant in PSD_VARIANTS
    }
    row_psd_exact_sums: dict[str, torch.Tensor | None] = {variant: None for variant in PSD_VARIANTS}
    rep_exact_spectrogram_sums: dict[str, dict[str, torch.Tensor | None]] = {
        variant: {'mean': None, 'median': None} for variant in PSD_VARIANTS
    }
    rep_userbin_spectrogram_sums: dict[str, dict[str, torch.Tensor | None]] = {
        variant: {'mean': None, 'median': None} for variant in PSD_VARIANTS
    }
    freq_refs: dict[str, torch.Tensor] = {}
    userbin_edge_refs: dict[str, torch.Tensor] = {}
    userbin_center_refs: dict[str, torch.Tensor] = {}
    spec_freq_refs: dict[str, torch.Tensor] = {}
    frame_center_refs: dict[str, torch.Tensor] = {}
    spec_userbin_center_refs: dict[str, torch.Tensor] = {}

    for batch_maps in signal_map_batches:
        maps = _as_float_tensor(batch_maps)
        if maps.ndim != 3:
            raise ValueError(f'Expected shape (samples, rows, time), got {tuple(maps.shape)}')
        batch_samples, batch_rows, batch_length = [int(v) for v in maps.shape]
        if batch_samples <= 0:
            continue
        if num_rows is None:
            num_rows = int(batch_rows)
            sequence_length = int(batch_length)
        elif int(batch_rows) != int(num_rows) or int(batch_length) != int(sequence_length):
            raise ValueError(
                'All channel-major batches must share the same (rows, time) shape. '
                f'Expected ({num_rows}, {sequence_length}), got ({batch_rows}, {batch_length}).'
            )

        for variant in PSD_VARIANTS:
            variant_maps = _variant_maps(maps, variant)
            freq_exact, exact_psd = exact_periodogram_from_maps(variant_maps)
            userbin_psd, edges, centers = aggregate_userbins_torch(exact_psd, freq_exact, userbin_edges, axis=2)
            row_psd_exact_batch = exact_psd.real.sum(dim=0).detach().cpu()
            row_psd_exact_sums[variant] = row_psd_exact_batch if row_psd_exact_sums[variant] is None else row_psd_exact_sums[variant] + row_psd_exact_batch
            for reducer in ('mean', 'median'):
                rep_exact_batch = _sample_reduce_rows(exact_psd.real, reducer).sum(dim=0).detach().cpu()
                rep_userbin_batch = _sample_reduce_rows(userbin_psd.real, reducer).sum(dim=0).detach().cpu()
                rep_exact_curve_sums[variant][reducer] = rep_exact_batch if rep_exact_curve_sums[variant][reducer] is None else rep_exact_curve_sums[variant][reducer] + rep_exact_batch
                rep_userbin_curve_sums[variant][reducer] = rep_userbin_batch if rep_userbin_curve_sums[variant][reducer] is None else rep_userbin_curve_sums[variant][reducer] + rep_userbin_batch
            if variant not in freq_refs:
                freq_refs[variant] = freq_exact.detach().cpu()
                userbin_edge_refs[variant] = edges.detach().cpu()
                userbin_center_refs[variant] = centers.detach().cpu()

            if include_spectrogram:
                spec_freqs, frame_centers, spectrogram = exact_spectrogram_from_maps(variant_maps, window=window, overlap=overlap)
                userbin_spec, _spec_edges, spec_centers = aggregate_userbins_torch(spectrogram, spec_freqs, userbin_edges, axis=2)
                for reducer in ('mean', 'median'):
                    rep_spec_exact_batch = _sample_reduce_rows(spectrogram.real, reducer).sum(dim=0).detach().cpu()
                    rep_spec_userbin_batch = _sample_reduce_rows(userbin_spec.real, reducer).sum(dim=0).detach().cpu()
                    rep_exact_spectrogram_sums[variant][reducer] = (
                        rep_spec_exact_batch if rep_exact_spectrogram_sums[variant][reducer] is None else rep_exact_spectrogram_sums[variant][reducer] + rep_spec_exact_batch
                    )
                    rep_userbin_spectrogram_sums[variant][reducer] = (
                        rep_spec_userbin_batch if rep_userbin_spectrogram_sums[variant][reducer] is None else rep_userbin_spectrogram_sums[variant][reducer] + rep_spec_userbin_batch
                    )
                if variant not in spec_freq_refs:
                    spec_freq_refs[variant] = spec_freqs.detach().cpu()
                    frame_center_refs[variant] = frame_centers.detach().cpu()
                    spec_userbin_center_refs[variant] = spec_centers.detach().cpu()
        num_samples_total += int(batch_samples)

    if num_samples_total <= 0 or num_rows is None or sequence_length is None:
        raise RuntimeError('Cannot build PSD payload from an empty iterable of channel-major batches.')

    payload: dict[str, Any] = {
        'variants_saved': list(PSD_VARIANTS),
        'taper_window_applied': True,
        'taper_window_name': 'hann',
        'db_plots_saved': True,
        'db_plot_scale': '10log10_power_plus_epsilon',
        'db_plot_epsilon': DB_EPSILON,
        'num_samples': int(num_samples_total),
        'num_rows': int(num_rows),
        'sequence_length': int(sequence_length),
        'userbin_edges': [float(v) for v in userbin_edges],
        'spectrogram_saved': bool(include_spectrogram),
    }

    for variant in PSD_VARIANTS:
        if row_psd_exact_sums[variant] is None:
            raise RuntimeError(f'PSD accumulation failed for variant {variant!r}.')
        freq_exact = freq_refs[variant]
        edges = userbin_edge_refs[variant]
        centers = userbin_center_refs[variant]
        mean_waveform_exact = rep_exact_curve_sums[variant]['mean'] / float(num_samples_total)
        median_waveform_exact = rep_exact_curve_sums[variant]['median'] / float(num_samples_total)
        mean_waveform_userbin = rep_userbin_curve_sums[variant]['mean'] / float(num_samples_total)
        median_waveform_userbin = rep_userbin_curve_sums[variant]['median'] / float(num_samples_total)
        row_psd_exact = row_psd_exact_sums[variant] / float(num_samples_total)

        payload[f'freq_exact_{variant}'] = freq_exact.detach().cpu().numpy()
        payload[f'userbin_edges_{variant}'] = edges.detach().cpu().numpy()
        payload[f'userbin_centers_{variant}'] = centers.detach().cpu().numpy()
        payload[f'mean_psd_waveform_exact_{variant}'] = mean_waveform_exact.detach().cpu().numpy()
        payload[f'median_psd_waveform_exact_{variant}'] = median_waveform_exact.detach().cpu().numpy()
        payload[f'mean_psd_waveform_userbin_{variant}'] = mean_waveform_userbin.detach().cpu().numpy()
        payload[f'median_psd_waveform_userbin_{variant}'] = median_waveform_userbin.detach().cpu().numpy()
        payload[f'row_mean_psd_exact_{variant}'] = row_psd_exact.detach().cpu().numpy()
        payload[f'mean_psd_waveform_exact_{variant}_db'] = power_to_db(mean_waveform_exact).detach().cpu().numpy()
        payload[f'median_psd_waveform_exact_{variant}_db'] = power_to_db(median_waveform_exact).detach().cpu().numpy()
        payload[f'mean_psd_waveform_userbin_{variant}_db'] = power_to_db(mean_waveform_userbin).detach().cpu().numpy()
        payload[f'median_psd_waveform_userbin_{variant}_db'] = power_to_db(median_waveform_userbin).detach().cpu().numpy()
        payload[f'scalar_summary_{variant}'] = json.dumps(
            {
                'mean_psd_waveform_mean': float(mean_waveform_exact.mean().item()),
                'mean_psd_waveform_std': float(mean_waveform_exact.std(unbiased=False).item()),
                'median_psd_waveform_mean': float(median_waveform_exact.mean().item()),
                'median_psd_waveform_std': float(median_waveform_exact.std(unbiased=False).item()),
            },
            ensure_ascii=False,
        )

        if include_spectrogram:
            spec_freqs = spec_freq_refs[variant]
            frame_centers = frame_center_refs[variant]
            spec_userbin_centers = spec_userbin_center_refs[variant]
            mean_spectrogram_exact = rep_exact_spectrogram_sums[variant]['mean'] / float(num_samples_total)
            median_spectrogram_exact = rep_exact_spectrogram_sums[variant]['median'] / float(num_samples_total)
            mean_spectrogram_userbin = rep_userbin_spectrogram_sums[variant]['mean'] / float(num_samples_total)
            median_spectrogram_userbin = rep_userbin_spectrogram_sums[variant]['median'] / float(num_samples_total)
            payload[f'spectrogram_freqs_{variant}'] = spec_freqs.detach().cpu().numpy()
            payload[f'spectrogram_frame_centers_{variant}'] = frame_centers.detach().cpu().numpy()
            payload[f'spectrogram_userbin_centers_{variant}'] = spec_userbin_centers.detach().cpu().numpy()
            payload[f'mean_spectrogram_exact_{variant}'] = mean_spectrogram_exact.detach().cpu().numpy()
            payload[f'median_spectrogram_exact_{variant}'] = median_spectrogram_exact.detach().cpu().numpy()
            payload[f'mean_spectrogram_userbin_{variant}'] = mean_spectrogram_userbin.detach().cpu().numpy()
            payload[f'median_spectrogram_userbin_{variant}'] = median_spectrogram_userbin.detach().cpu().numpy()
            payload[f'mean_spectrogram_exact_{variant}_db'] = power_to_db(mean_spectrogram_exact).detach().cpu().numpy()
            payload[f'median_spectrogram_exact_{variant}_db'] = power_to_db(median_spectrogram_exact).detach().cpu().numpy()
            payload[f'mean_spectrogram_userbin_{variant}_db'] = power_to_db(mean_spectrogram_userbin).detach().cpu().numpy()
            payload[f'median_spectrogram_userbin_{variant}_db'] = power_to_db(median_spectrogram_userbin).detach().cpu().numpy()
    return payload


def combined_spatial_2d_psd_payload_from_original_input_batches(
    original_input_batches: Iterable[torch.Tensor],
) -> dict[str, Any]:
    """Compute mean 2D spatial PSDs exactly from an iterable of original-input batches."""

    batch_size = max(1, int(os.environ.get('PSD_SPATIAL_FFT_BATCH', '1024')))
    target_device: torch.device | None = None
    target_dtype: torch.dtype | None = None
    raw_sum: torch.Tensor | None = None
    centered_sum: torch.Tensor | None = None
    total_count = 0
    spatial_shape: tuple[int, int] | None = None

    for original_inputs in original_input_batches:
        inputs = original_inputs if isinstance(original_inputs, torch.Tensor) else torch.as_tensor(original_inputs)
        if inputs.ndim < 3:
            raise ValueError(f'Expected original image-like inputs with rank >= 3, got {tuple(inputs.shape)}.')
        spatial_h = int(inputs.shape[-2])
        spatial_w = int(inputs.shape[-1])
        if spatial_shape is None:
            spatial_shape = (spatial_h, spatial_w)
        elif spatial_shape != (spatial_h, spatial_w):
            raise ValueError(
                'All original-input batches must share the same spatial shape. '
                f'Expected {spatial_shape}, got {(spatial_h, spatial_w)}.'
            )
        maps = inputs.reshape(-1, spatial_h, spatial_w)
        if int(maps.shape[0]) <= 0:
            continue
        if target_device is None:
            if maps.device.type == 'cuda':
                target_device = maps.device
            elif torch.cuda.is_available():
                target_device = torch.device(f'cuda:{torch.cuda.current_device()}')
            else:
                target_device = maps.device
            target_dtype = _analysis_real_dtype(device=target_device, like=maps if maps.device.type == 'cuda' else None)
        assert target_device is not None and target_dtype is not None
        for start in range(0, int(maps.shape[0]), batch_size):
            stop = min(int(maps.shape[0]), start + batch_size)
            chunk = maps[start:stop].to(
                device=target_device,
                dtype=target_dtype,
                non_blocking=(maps.device.type == 'cpu' and target_device.type == 'cuda'),
            )
            raw_fft = torch.fft.fftshift(torch.fft.fft2(chunk, dim=(-2, -1)), dim=(-2, -1))
            raw_chunk_sum = raw_fft.abs().square().sum(dim=0).detach().cpu()
            centered_chunk = chunk - chunk.mean(dim=(-2, -1), keepdim=True)
            centered_fft = torch.fft.fftshift(torch.fft.fft2(centered_chunk, dim=(-2, -1)), dim=(-2, -1))
            centered_chunk_sum = centered_fft.abs().square().sum(dim=0).detach().cpu()
            raw_sum = raw_chunk_sum if raw_sum is None else raw_sum + raw_chunk_sum
            centered_sum = centered_chunk_sum if centered_sum is None else centered_sum + centered_chunk_sum
            total_count += int(chunk.shape[0])

    if total_count <= 0 or raw_sum is None or centered_sum is None:
        raise RuntimeError('Cannot build spatial PSD payload from an empty iterable of original-input batches.')
    return {
        'mean_spatial_psd_2d_raw': (raw_sum / float(total_count)).detach().cpu().numpy(),
        'mean_spatial_psd_2d_centered': (centered_sum / float(total_count)).detach().cpu().numpy(),
    }


def mean_spatial_psd_2d_from_original_inputs(original_inputs: torch.Tensor, *, centered: bool) -> dict[str, Any]:
    """Handle ``mean spatial psd 2d from original inputs`` for the ``psd_utils`` module."""
    inputs = original_inputs if isinstance(original_inputs, torch.Tensor) else torch.as_tensor(original_inputs)
    if inputs.ndim < 3:
        raise ValueError(f'Expected original image-like inputs with rank >= 3, got {tuple(inputs.shape)}.')

    spatial_h = int(inputs.shape[-2])
    spatial_w = int(inputs.shape[-1])
    maps = inputs.reshape(-1, spatial_h, spatial_w)
    if int(maps.shape[0]) <= 0:
        raise ValueError('original_inputs must contain at least one spatial map.')

    if maps.device.type == 'cuda':
        target_device = maps.device
    elif torch.cuda.is_available():
        target_device = torch.device(f'cuda:{torch.cuda.current_device()}')
    else:
        target_device = maps.device
    target_dtype = _analysis_real_dtype(device=target_device, like=maps if maps.device.type == 'cuda' else None)
    batch_size = max(1, int(os.environ.get('PSD_SPATIAL_FFT_BATCH', '1024')))

    power_sum: torch.Tensor | None = None
    total_count = 0
    for start in range(0, int(maps.shape[0]), batch_size):
        stop = min(int(maps.shape[0]), start + batch_size)
        chunk = maps[start:stop].to(
            device=target_device,
            dtype=target_dtype,
            non_blocking=(maps.device.type == 'cpu' and target_device.type == 'cuda'),
        )
        if centered:
            chunk = chunk - chunk.mean(dim=(-2, -1), keepdim=True)
        fft = torch.fft.fftshift(torch.fft.fft2(chunk, dim=(-2, -1)), dim=(-2, -1))
        chunk_power_sum = fft.abs().square().sum(dim=0)
        power_sum = chunk_power_sum if power_sum is None else power_sum + chunk_power_sum
        total_count += int(chunk.shape[0])

    power = power_sum / float(max(1, total_count))
    return {'mean_spatial_psd_2d': power.detach().cpu().numpy()}


def combined_spatial_2d_psd_payload_from_original_inputs(original_inputs: torch.Tensor) -> dict[str, Any]:
    """Handle ``combined spatial 2d psd payload from original inputs`` for the ``psd_utils`` module."""
    raw = mean_spatial_psd_2d_from_original_inputs(original_inputs, centered=False)
    centered = mean_spatial_psd_2d_from_original_inputs(original_inputs, centered=True)
    return {
        'mean_spatial_psd_2d_raw': raw['mean_spatial_psd_2d'],
        'mean_spatial_psd_2d_centered': centered['mean_spatial_psd_2d'],
    }


def build_scalar_waveform_bundle_from_maps(
    signal_maps: torch.Tensor,
    *,
    window: int | None,
    overlap: int | None,
    userbin_edges: Sequence[float],
) -> dict[str, Any]:
    """Build scalar waveform bundle from maps."""
    maps = signal_maps if signal_maps.ndim == 3 else tensor_to_channel_major_maps(signal_maps)
    if maps.ndim != 3:
        raise ValueError(f'Expected rank-3 scalar maps, got {tuple(maps.shape)}.')
    return combined_exact_psd_payload_from_maps_torch(maps, window=window, overlap=overlap, userbin_edges=userbin_edges, include_spectrogram=True)


def auto_spectral_matrix_from_mode_maps(mode_maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Handle ``auto spectral matrix from mode maps`` for the ``psd_utils`` module."""
    if mode_maps.ndim != 3:
        raise ValueError(f'Expected shape (samples, modes, time), got {tuple(mode_maps.shape)}')
    freqs, spectrum, scale, window_power = exact_hann_rfft(mode_maps)
    length = int(mode_maps.shape[-1])
    matrix = torch.einsum('nmf,nkf->fmk', spectrum, spectrum.conj()) / max(1, int(mode_maps.shape[0]))
    matrix = scale.view(-1, 1, 1) * matrix / (float(length) * window_power)
    return freqs, matrix


def cross_spectral_matrix_from_mode_maps(x_mode_maps: torch.Tensor, y_mode_maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Handle ``cross spectral matrix from mode maps`` for the ``psd_utils`` module."""
    if x_mode_maps.ndim != 3 or y_mode_maps.ndim != 3:
        raise ValueError('cross_spectral_matrix_from_mode_maps expects rank-3 mode maps.')
    if x_mode_maps.shape[0] != y_mode_maps.shape[0] or x_mode_maps.shape[-1] != y_mode_maps.shape[-1]:
        raise ValueError('x_mode_maps and y_mode_maps must agree in sample count and time length.')
    freqs, x_spec, scale, window_power = exact_hann_rfft(x_mode_maps)
    _freqs_y, y_spec, _scale_y, _window_power_y = exact_hann_rfft(y_mode_maps)
    length = int(x_mode_maps.shape[-1])
    matrix = torch.einsum('nyf,nxf->fyx', y_spec, x_spec.conj()) / max(1, int(x_mode_maps.shape[0]))
    matrix = scale.view(-1, 1, 1) * matrix / (float(length) * window_power)
    return freqs, matrix


__all__ = [
    'DB_EPSILON',
    'PSD_VARIANTS',
    'aggregate_userbins_numpy',
    'aggregate_userbins_torch',
    'apply_centering',
    'auto_spectral_matrix_from_mode_maps',
    'build_scalar_waveform_bundle_from_maps',
    'centered_pointwise_l2',
    'centered_rowwise_l2',
    'combined_exact_psd_payload_from_map_batches_torch',
    'combined_exact_psd_payload_from_maps_torch',
    'combined_spatial_2d_psd_payload_from_original_input_batches',
    'combined_spatial_2d_psd_payload_from_original_inputs',
    'cross_spectral_matrix_from_mode_maps',
    'exact_cross_periodogram_from_maps',
    'exact_hann_rfft',
    'exact_one_sided_freqs',
    'exact_periodogram_from_maps',
    'exact_spectrogram_from_maps',
    'first_difference_l2',
    'hann_window',
    'mean_spatial_psd_2d_from_original_inputs',
    'one_sided_scaling',
    'power_to_db',
    'pca_dim_from_cli_vector',
    'row_variance_mad_summary',
    'scalar_io_spectral_objects',
    'scalar_representative_maps',
    'sequence_batch_to_channel_major_maps',
    'trace_tensor_to_channel_major_maps',
    'tensor_to_channel_major_maps',
    'tensor_to_channel_major_maps_explicit',
    'time_domain_summary_from_maps',
]
    'compute_fixed_pca_basis',
    'apply_fixed_pca_basis',
