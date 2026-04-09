"""Exact periodogram and exact sliding-periodogram helpers."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


def _to_numpy_3d(maps: np.ndarray) -> np.ndarray:
    arr = np.asarray(maps, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"Expected (S,R,T) maps, got shape={arr.shape}")
    return arr


def exact_periodogram(maps: np.ndarray, centered: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Compute exact one-sided simple periodogram for maps (S,R,T)."""

    arr = _to_numpy_3d(maps)
    if centered:
        arr = arr - arr.mean(axis=-1, keepdims=True)
    t = arr.shape[-1]
    spec = np.fft.rfft(arr, axis=-1)
    pxx = (np.abs(spec) ** 2) / float(t)
    freqs = np.fft.rfftfreq(t, d=1.0)
    return freqs, pxx


def sliding_exact_spectrogram(maps: np.ndarray, window: int, overlap: int, centered: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact sliding simple periodogram with no taper window."""

    arr = _to_numpy_3d(maps)
    if window <= 0 or overlap < 0 or overlap >= window:
        raise ValueError("window and overlap are invalid")
    step = window - overlap
    t = arr.shape[-1]
    starts = list(range(0, max(t - window + 1, 1), step))
    if not starts:
        starts = [0]
    frames = []
    centers = []
    for s in starts:
        e = min(s + window, t)
        frame = arr[..., s:e]
        if frame.shape[-1] < window:
            pad = np.zeros((*frame.shape[:-1], window - frame.shape[-1]), dtype=frame.dtype)
            frame = np.concatenate([frame, pad], axis=-1)
        if centered:
            frame = frame - frame.mean(axis=-1, keepdims=True)
        spec = np.fft.rfft(frame, axis=-1)
        frames.append((np.abs(spec) ** 2) / float(window))
        centers.append(s + (window - 1) / 2.0)
    spectrogram = np.stack(frames, axis=-1)  # (S,R,F,U)
    freqs = np.fft.rfftfreq(window, d=1.0)
    return freqs, np.asarray(centers, dtype=np.float64), spectrogram


def bin_by_user_edges(freqs: np.ndarray, values: np.ndarray, edges: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate values by user-provided frequency bins.

    values must have frequency on the last axis or second-last axis.
    """

    f = np.asarray(freqs)
    e = np.asarray(list(edges), dtype=np.float64)
    if e.ndim != 1 or len(e) < 2:
        raise ValueError("userbin edges must contain at least 2 values")
    centers = 0.5 * (e[:-1] + e[1:])
    out = []
    for lo, hi in zip(e[:-1], e[1:]):
        mask = (f >= lo) & (f < hi if hi < e[-1] else f <= hi)
        if not np.any(mask):
            out.append(np.zeros(values.shape[:-1], dtype=np.float64))
        else:
            out.append(values[..., mask].mean(axis=-1))
    return centers, np.stack(out, axis=-1)


def db10(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert linear power-like values to decibel scale."""

    return 10.0 * np.log10(np.asarray(x) + eps)


def summary_scalars(array: np.ndarray) -> Dict[str, float]:
    """Small scalar summary used for bundle metadata."""

    arr = np.asarray(array, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
