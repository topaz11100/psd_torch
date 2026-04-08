from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from src.common.fft_analysis import band_edges_to_bin_ranges, bin_spectrum


_DEFAULT_USERBIN_EDGES: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


def normalize_userbin_edges(edges: Optional[Sequence[float]]) -> Tuple[list[float], str]:
    if edges is None:
        return [float(v) for v in _DEFAULT_USERBIN_EDGES], "default"
    vals = [float(v) for v in edges]
    if len(vals) < 2:
        raise ValueError("userbin_edges must contain at least two values")
    for i in range(len(vals) - 1):
        if not (vals[i] < vals[i + 1]):
            raise ValueError(f"userbin_edges must be strictly increasing, got {vals}")
    return vals, "cli"



def userbin_centers(edges: Sequence[float]) -> np.ndarray:
    vals = [float(v) for v in edges]
    if len(vals) < 2:
        return np.zeros((0,), dtype=float)
    return np.asarray([(vals[i] + vals[i + 1]) * 0.5 for i in range(len(vals) - 1)], dtype=float)



def effective_psd_window(T: int, psd_window: int, psd_overlap: int) -> Tuple[int, int]:
    """Return the effective spectrogram frame length and overlap.

    The public CLI still uses the historical ``psd_window`` / ``psd_overlap``
    names, but under the current specification these values control only the
    sliding simple-periodogram spectrogram frame geometry. The exact waveform
    path always uses the full signal length with no taper window.
    """
    T = int(T)
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")
    nperseg = max(2, min(int(psd_window), T))
    noverlap = max(0, min(int(psd_overlap), nperseg - 1))
    return int(nperseg), int(noverlap)



def build_window(window_fn: str, length: int, *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Legacy taper helper kept for non-PSD-analysis utilities.

    The official ``psd_analysis`` and ``dataset_psd`` exact periodogram /
    exact spectrogram paths must not apply taper windows. This helper remains
    available only for auxiliary code paths such as generic Welch utilities.
    """
    name = str(window_fn).strip().lower()
    if int(length) <= 0:
        raise ValueError(f"window length must be > 0, got {length}")
    if name == "hann":
        return torch.hann_window(int(length), periodic=False, device=device, dtype=dtype)
    if name == "hamming":
        return torch.hamming_window(int(length), periodic=False, device=device, dtype=dtype)
    if name == "blackman":
        return torch.blackman_window(int(length), periodic=False, device=device, dtype=dtype)
    raise ValueError(f"Unsupported window_fn: {window_fn}")



def _center_last_dim(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=-1, keepdim=True)



def exact_simple_periodogram_torch(
    x: torch.Tensor,
    *,
    centered: bool = False,
) -> torch.Tensor:
    """Exact simple periodogram with no taper window.

    For a signal of length ``T`` along the last axis this returns

    ``(1 / T) * |rfft(x)|^2``

    where ``x`` is either the raw signal or its mean-centered version.
    """
    if x.ndim < 1:
        raise ValueError(f"x must have at least one dimension, got {tuple(x.shape)}")
    x_f = x.to(torch.float32)
    if bool(centered):
        x_f = _center_last_dim(x_f)
    T = int(x_f.shape[-1])
    if T <= 0:
        raise ValueError(f"signal length must be > 0, got {T}")
    spec = torch.fft.rfft(x_f, dim=-1)
    return (spec.abs() ** 2) / float(T)



def periodogram_psd_torch(
    x: torch.Tensor,
    *,
    window_fn: str = "hann",
    centered: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Backward-compatible alias for the exact windowless periodogram.

    ``window_fn`` and ``eps`` are accepted for API compatibility but ignored by
    the official exact path.
    """
    del window_fn, eps
    return exact_simple_periodogram_torch(x, centered=bool(centered))



def _windowed_rfft_power(
    x: torch.Tensor,
    *,
    window: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    seg = x - x.mean(dim=-1, keepdim=True)
    seg = seg * window
    spec = torch.fft.rfft(seg, dim=-1)
    denom = torch.clamp(torch.sum(window * window), min=float(eps))
    return (spec.abs() ** 2) / denom



def welch_psd_torch(
    x: torch.Tensor,
    *,
    nperseg: int,
    noverlap: int,
    window_fn: str = "hann",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Generic Welch PSD helper for non-official auxiliary use."""
    if x.ndim < 1:
        raise ValueError(f"x must have at least one dimension, got {tuple(x.shape)}")
    T = int(x.shape[-1])
    nperseg_eff, noverlap_eff = effective_psd_window(T, int(nperseg), int(noverlap))
    step = max(1, nperseg_eff - noverlap_eff)
    if T < nperseg_eff:
        raise ValueError(f"effective nperseg ({nperseg_eff}) cannot exceed T ({T})")
    num_seg = 1 + max(0, (T - nperseg_eff) // step)
    if num_seg <= 0:
        raise ValueError(f"No Welch segments for T={T}, nperseg={nperseg_eff}, noverlap={noverlap_eff}")

    win = build_window(window_fn, nperseg_eff, device=x.device, dtype=x.dtype)
    segs = x.unfold(-1, nperseg_eff, step)
    power = _windowed_rfft_power(segs, window=win, eps=float(eps))
    return power.mean(dim=-2)



def exact_sliding_simple_spectrogram_torch(
    x: torch.Tensor,
    *,
    nperseg: int,
    noverlap: int,
    centered: bool = False,
) -> torch.Tensor:
    """Exact sliding simple-periodogram spectrogram with no taper window.

    Input shape:
      - (T,)
      - (..., T)
    Output shape:
      - (..., F, U) where F is the exact one-sided frequency-bin count and U is
        the number of sliding frames.
    """
    if x.ndim < 1:
        raise ValueError(f"x must have at least one dimension, got {tuple(x.shape)}")
    T = int(x.shape[-1])
    nperseg_eff, noverlap_eff = effective_psd_window(T, int(nperseg), int(noverlap))
    step = max(1, nperseg_eff - noverlap_eff)
    if T < nperseg_eff:
        raise ValueError(f"effective nperseg ({nperseg_eff}) cannot exceed T ({T})")
    num_seg = 1 + max(0, (T - nperseg_eff) // step)
    if num_seg <= 0:
        raise ValueError(f"No spectrogram segments for T={T}, nperseg={nperseg_eff}, noverlap={noverlap_eff}")

    segs = x.to(torch.float32).unfold(-1, nperseg_eff, step)
    if bool(centered):
        segs = _center_last_dim(segs)
    spec = torch.fft.rfft(segs, dim=-1)
    power = (spec.abs() ** 2) / float(nperseg_eff)  # (..., U, F)
    return power.movedim(-1, -2).contiguous()  # (..., F, U)



def spectrogram_exact_torch(
    x: torch.Tensor,
    *,
    nperseg: int,
    noverlap: int,
    window_fn: str = "hann",
    centered: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Backward-compatible alias for the exact windowless sliding spectrogram.

    ``window_fn`` and ``eps`` are accepted for API compatibility but ignored by
    the official exact path.
    """
    del window_fn, eps
    return exact_sliding_simple_spectrogram_torch(
        x,
        nperseg=int(nperseg),
        noverlap=int(noverlap),
        centered=bool(centered),
    )



def spectrogram_frame_centers(T: int, *, nperseg: int, noverlap: int) -> np.ndarray:
    T_i = int(T)
    nperseg_eff, noverlap_eff = effective_psd_window(T_i, int(nperseg), int(noverlap))
    step = max(1, nperseg_eff - noverlap_eff)
    num_seg = 1 + max(0, (T_i - nperseg_eff) // step)
    starts = np.arange(num_seg, dtype=float) * float(step)
    return starts + 0.5 * float(max(nperseg_eff - 1, 0))



def userbin_from_psd_numpy(psd: np.ndarray, band_ranges: Sequence[Tuple[int, int]]) -> np.ndarray:
    return np.asarray(bin_spectrum(np.asarray(psd, dtype=float), band_ranges, dim=-1, reduce="mean"), dtype=float)



def userbin_from_psd_torch(psd: torch.Tensor, band_ranges: Sequence[Tuple[int, int]]) -> torch.Tensor:
    if psd.ndim < 1:
        raise ValueError(f"psd must have at least one dimension, got {tuple(psd.shape)}")
    bands = []
    for lo, hi in band_ranges:
        lo_i = int(lo)
        hi_i = int(hi)
        if hi_i < lo_i:
            bands.append(torch.zeros(psd.shape[:-1], device=psd.device, dtype=psd.dtype))
        else:
            bands.append(psd[..., lo_i:hi_i + 1].mean(dim=-1))
    if len(bands) == 0:
        return torch.zeros((*psd.shape[:-1], 0), device=psd.device, dtype=psd.dtype)
    return torch.stack(bands, dim=-1)



def spectrogram_userbin_torch(
    x: torch.Tensor,
    *,
    nperseg: int,
    noverlap: int,
    band_ranges: Sequence[Tuple[int, int]],
    window_fn: str = "hann",
    centered: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Windowless exact spectrogram followed by userbin aggregation."""
    exact = spectrogram_exact_torch(
        x,
        nperseg=int(nperseg),
        noverlap=int(noverlap),
        window_fn=str(window_fn),
        centered=bool(centered),
        eps=float(eps),
    )  # (..., F, U)
    banded = userbin_from_psd_torch(exact.movedim(-2, -1), band_ranges)  # (..., U, B)
    return banded.movedim(-1, -2).contiguous()  # (..., B, U)



def exp_filter_torch(x: torch.Tensor, tau: float, *, dt: float = 1.0) -> torch.Tensor:
    """Exponential causal filter along the last axis.

    z[t] = a z[t-1] + (1-a) x[t], a = exp(-dt/tau)
    """
    if float(tau) <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau}")
    if x.ndim < 1:
        raise ValueError(f"x must have at least one dimension, got {tuple(x.shape)}")
    a = float(np.exp(-float(dt) / float(tau)))
    coeff_x = 1.0 - a
    out = torch.zeros_like(x)
    out[..., 0] = coeff_x * x[..., 0]
    for t in range(1, int(x.shape[-1])):
        out[..., t] = a * out[..., t - 1] + coeff_x * x[..., t]
    return out



def mean_bandpower_summary(userbin: np.ndarray, edges: Sequence[float], *, prefix: str) -> dict[str, float]:
    arr = np.asarray(userbin, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    out: dict[str, float] = {}
    for b in range(arr.shape[-1]):
        lo = float(edges[b])
        hi = float(edges[b + 1])
        key = f"{prefix}_{b}_{lo:.4f}_{hi:.4f}"
        out[key] = float(np.mean(arr[..., b]))
    return out



def temporal_band_ranges_from_edges(T_like: int, edges: Sequence[float]) -> list[Tuple[int, int]]:
    return band_edges_to_bin_ranges(int(T_like), [float(v) for v in edges], d=1.0)
