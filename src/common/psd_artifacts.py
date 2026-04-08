from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.common.plotting import save_heatmap_plot, save_line_plot, save_multiline_series_plot
from src.common.psd_utils import (
    periodogram_psd_torch,
    spectrogram_exact_torch,
    spectrogram_frame_centers,
    userbin_from_psd_torch,
)
from src.common.snn_builder import SNNConfig, SNNClassifier, build_snn_classifier
from src.common.utils import save_json


FeedForwardSNNWithReadout = SNNClassifier


# -----------------------------------------------------------------------------
# Common model / sample helpers
# -----------------------------------------------------------------------------


def build_common_classifier(
    model_name: str,
    input_dim: int,
    hidden_dims: Sequence[int],
    num_classes: int,
    cfg: SNNConfig,
) -> FeedForwardSNNWithReadout:
    hidden_list = [int(h) for h in hidden_dims]
    cfg_eff = replace(
        cfg,
        model_name=str(model_name),
        input_dim=int(input_dim),
        hidden_dim=int(hidden_list[0]) if hidden_list else int(input_dim),
        hidden_dims=hidden_list,
        num_classes=int(num_classes),
    )
    return build_snn_classifier(cfg_eff)



def layer_names_from_hidden(hidden: Sequence[int]) -> List[str]:
    return [f"hidden_{i+1}" for i in range(len(hidden))]



def save_heatmap_image(
    path: str,
    mat: np.ndarray,
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    log1p: bool = False,
    center_zero: bool = False,
) -> None:
    save_heatmap_plot(
        path,
        np.asarray(mat, dtype=float),
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        use_log1p=bool(log1p),
        center_zero=bool(center_zero),
    )



def save_multiline_plot(
    path: str,
    ys: np.ndarray,
    *,
    x: Optional[np.ndarray] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    legend_labels: Optional[Sequence[str]] = None,
) -> None:
    save_multiline_series_plot(
        path,
        np.asarray(ys, dtype=float),
        x=None if x is None else np.asarray(x, dtype=float),
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        legend_labels=legend_labels,
    )


# -----------------------------------------------------------------------------
# Signal transforms
# -----------------------------------------------------------------------------


def input_time_matrix(x_seq: np.ndarray | torch.Tensor) -> np.ndarray:
    arr = x_seq.detach().cpu().numpy() if torch.is_tensor(x_seq) else np.asarray(x_seq)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"expected single-sample batch or sample tensor, got {arr.shape}")
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"input sequence must be (T,C), got {arr.shape}")
    return np.asarray(arr, dtype=float).T  # (C,T)



def _signal_single_sample_array(sig: np.ndarray | torch.Tensor) -> np.ndarray:
    arr = sig.detach().cpu().numpy() if torch.is_tensor(sig) else np.asarray(sig)
    if arr.ndim >= 1 and arr.shape[0] == 1 and arr.ndim in (3, 4):
        arr = arr[0]
    return np.asarray(arr, dtype=float)



def layer_signal_matrix(sig: np.ndarray | torch.Tensor, *, active_mask: Optional[np.ndarray] = None) -> np.ndarray:
    arr = _signal_single_sample_array(sig)
    if arr.ndim == 2:  # (T,N)
        return arr.T
    if arr.ndim == 3:  # (T,N,D)
        T, N, D = arr.shape
        mat = arr.transpose(1, 2, 0).reshape(N * D, T)
        if active_mask is not None:
            m = np.asarray(active_mask, dtype=float).reshape(-1)
            if m.size == N * D:
                mat = mat[m > 0.0]
        return mat
    raise ValueError(f"unexpected signal shape for layer heatmap: {arr.shape}")



def maybe_active_mask(layer: nn.Module) -> Optional[np.ndarray]:
    if hasattr(layer, "soft_mask") and callable(getattr(layer, "soft_mask")):
        try:
            m = layer.soft_mask(torch.float32)  # type: ignore[attr-defined]
            if torch.is_tensor(m):
                return m.detach().cpu().numpy().astype(float)
        except Exception:
            return None
    return None



def _layer_semantic_aliases(layer: nn.Module, rec: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    aliases: Dict[str, np.ndarray] = {}
    cls = layer.__class__.__name__
    if "output" in rec:
        aliases["spk"] = rec["output"]
        aliases["spike"] = rec["output"]
    if "soma_state" in rec:
        aliases["membrane"] = rec["soma_state"]

    if cls == "LIFDenseLayer":
        if "soma_state" in rec:
            aliases["mem"] = rec["soma_state"]
    elif cls == "RFDenseLayer":
        if "soma_state" in rec:
            aliases["x"] = rec["soma_state"]
        if "state_y" in rec:
            aliases["y"] = rec["state_y"]
    elif cls == "TCLIFDenseLayer":
        if "dendrite_state" in rec:
            aliases["v1"] = rec["dendrite_state"]
        if "soma_state" in rec:
            aliases["v2"] = rec["soma_state"]
    elif cls == "TSLIFDenseLayer":
        if "dendrite_state" in rec:
            aliases["vd"] = rec["dendrite_state"]
        if "soma_state" in rec:
            aliases["vs"] = rec["soma_state"]
    elif cls in ("DHSNNDenseLayer", "MyDHSNNDenseLayer", "MyReverseDHSNNDenseLayer"):
        if "dendrite_state" in rec:
            aliases["d_state"] = rec["dendrite_state"]
        if "soma_state" in rec:
            aliases["mem"] = rec["soma_state"]
    elif cls == "DRFDenseLayer":
        if "dendrite_state" in rec:
            aliases["u"] = rec["dendrite_state"]
        if "state_v" in rec:
            aliases["v"] = rec["state_v"]
        if "pre_hist" in rec:
            aliases["pre_hist"] = rec["pre_hist"]
        if "V_th" in rec:
            aliases["V_th"] = rec["V_th"]
    elif cls == "MyDRFDenseLayer":
        if "dendrite_state" in rec:
            aliases["u"] = rec["dendrite_state"]
        if "state_v" in rec:
            aliases["v"] = rec["state_v"]
        if "p_hist" in rec:
            aliases["p_hist"] = rec["p_hist"]
        if "V_th" in rec:
            aliases["V_th"] = rec["V_th"]
    return aliases



def collect_sample_result(model, x_seq: torch.Tensor, device: torch.device) -> Dict[str, Any]:
    with torch.no_grad():
        _, hidden_recs, out_rec = model.forward_with_records(x_seq.to(device).unsqueeze(0))
    recs: List[Dict[str, np.ndarray]] = []
    layers = list(getattr(model, "hidden_layers", getattr(model, "layers", [])))
    for layer, rec in zip(layers, hidden_recs):
        arr_rec = {k: _signal_single_sample_array(v) for k, v in rec.items()}
        arr_rec.update(_layer_semantic_aliases(layer, arr_rec))
        recs.append(arr_rec)
    output_rec = {k: _signal_single_sample_array(v) for k, v in out_rec.items()}
    output_layer = getattr(model, "output_layer", None)
    if output_layer is not None:
        output_rec.update(_layer_semantic_aliases(output_layer, output_rec))
    if "output" in output_rec:
        output_rec.setdefault("spk", output_rec["output"])
        output_rec.setdefault("spike", output_rec["output"])
    if "soma_state" in output_rec:
        output_rec.setdefault("membrane", output_rec["soma_state"])
    return {
        "input": x_seq.detach().cpu().numpy(),
        "recs": recs,
        "output_rec": output_rec,
    }


# -----------------------------------------------------------------------------
# PSD / spectrogram bundle helpers
# -----------------------------------------------------------------------------


_PSD_VARIANTS: tuple[str, ...] = ("raw", "centered")


def _spectrogram_userbin_torch_from_exact(exact: torch.Tensor, band_ranges) -> torch.Tensor:
    banded = userbin_from_psd_torch(exact.movedim(-2, -1), band_ranges)
    return banded.movedim(-1, -2).contiguous()



def combined_exact_psd_payload_from_maps_torch(
    maps_t: torch.Tensor,
    *,
    periodogram_band_ranges,
    spectrogram_band_ranges,
    nperseg_eff: int,
    noverlap_eff: int,
    window_fn: Optional[str] = None,
) -> Dict[str, Any]:
    if maps_t.dim() != 3:
        raise ValueError(f"maps_t must be (S,R,T), got {tuple(maps_t.shape)}")
    sample_count = int(maps_t.shape[0])
    row_count = int(maps_t.shape[1])
    T = int(maps_t.shape[2])
    flat = maps_t.to(torch.float32).reshape(sample_count * row_count, T)

    payload: Dict[str, Any] = {
        "exact_freqs": np.fft.rfftfreq(T, d=1.0).astype(np.float32),
        "spectrogram_freqs": np.fft.rfftfreq(int(nperseg_eff), d=1.0).astype(np.float32),
        "spectrogram_frame_centers": spectrogram_frame_centers(
            T,
            nperseg=int(nperseg_eff),
            noverlap=int(noverlap_eff),
        ).astype(np.float32),
        "periodogram_length": int(T),
        "spectrogram_window_length": int(nperseg_eff),
        "spectrogram_overlap_length": int(noverlap_eff),
        "num_samples": int(sample_count),
        "num_rows": int(row_count),
        "variants_saved": list(_PSD_VARIANTS),
        "taper_window_applied": False,
        "legacy_window_fn_ignored": None if window_fn is None else str(window_fn),
    }

    for variant_name, centered in (("raw", False), ("centered", True)):
        row_psd_exact = periodogram_psd_torch(flat, centered=bool(centered)).reshape(sample_count, row_count, -1)
        row_psd_user = userbin_from_psd_torch(row_psd_exact, periodogram_band_ranges)
        sample_mean_psd_exact = row_psd_exact.mean(dim=1)

        row_spectrogram_exact = spectrogram_exact_torch(
            flat,
            nperseg=int(nperseg_eff),
            noverlap=int(noverlap_eff),
            centered=bool(centered),
        )
        row_spectrogram_exact = row_spectrogram_exact.reshape(
            sample_count,
            row_count,
            row_spectrogram_exact.shape[-2],
            row_spectrogram_exact.shape[-1],
        )
        row_spectrogram_user = _spectrogram_userbin_torch_from_exact(row_spectrogram_exact, spectrogram_band_ranges)
        sample_mean_spectrogram_exact = row_spectrogram_exact.mean(dim=1)

        payload[f"set_mean_psd_exact_{variant_name}"] = sample_mean_psd_exact.mean(dim=0).detach().cpu().numpy().astype(np.float32)
        payload[f"set_mean_heatmap_user_{variant_name}"] = row_psd_user.mean(dim=0).detach().cpu().numpy().astype(np.float32)
        payload[f"set_mean_spectrogram_exact_{variant_name}"] = sample_mean_spectrogram_exact.mean(dim=0).detach().cpu().numpy().astype(np.float32)
        payload[f"set_mean_spectrogram_user_{variant_name}"] = row_spectrogram_user.mean(dim=0).detach().cpu().numpy().astype(np.float32)

    return payload



def merge_exact_psd_payloads(payloads: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(payloads) == 0:
        raise ValueError("cannot merge an empty payload list")

    first = payloads[0]
    exact_freqs = np.asarray(first["exact_freqs"], dtype=float)
    spectrogram_freqs = np.asarray(first["spectrogram_freqs"], dtype=float)
    spectrogram_frames = np.asarray(first["spectrogram_frame_centers"], dtype=float)
    num_rows = int(first["num_rows"])
    periodogram_length = int(first["periodogram_length"])
    spectrogram_window_length = int(first["spectrogram_window_length"])
    spectrogram_overlap_length = int(first["spectrogram_overlap_length"])

    total_samples = 0
    legacy_window_names: set[str] = set()
    accumulators: Dict[str, Optional[np.ndarray]] = {
        f"set_mean_psd_exact_{variant}": None for variant in _PSD_VARIANTS
    }
    for variant in _PSD_VARIANTS:
        accumulators[f"set_mean_heatmap_user_{variant}"] = None
        accumulators[f"set_mean_spectrogram_exact_{variant}"] = None
        accumulators[f"set_mean_spectrogram_user_{variant}"] = None

    for payload in payloads:
        cur_freqs = np.asarray(payload["exact_freqs"], dtype=float)
        cur_spec_freqs = np.asarray(payload["spectrogram_freqs"], dtype=float)
        cur_spec_frames = np.asarray(payload["spectrogram_frame_centers"], dtype=float)
        if cur_freqs.shape != exact_freqs.shape or not np.allclose(cur_freqs, exact_freqs):
            raise ValueError("exact frequency grids must match across payloads")
        if cur_spec_freqs.shape != spectrogram_freqs.shape or not np.allclose(cur_spec_freqs, spectrogram_freqs):
            raise ValueError("spectrogram frequency grids must match across payloads")
        if cur_spec_frames.shape != spectrogram_frames.shape or not np.allclose(cur_spec_frames, spectrogram_frames):
            raise ValueError("spectrogram frame-center grids must match across payloads")
        if int(payload["num_rows"]) != num_rows:
            raise ValueError("row counts must match across payloads")
        if int(payload["periodogram_length"]) != periodogram_length:
            raise ValueError("periodogram lengths must match across payloads")
        if int(payload["spectrogram_window_length"]) != spectrogram_window_length:
            raise ValueError("spectrogram window lengths must match across payloads")
        if int(payload["spectrogram_overlap_length"]) != spectrogram_overlap_length:
            raise ValueError("spectrogram overlap lengths must match across payloads")
        legacy_name = payload.get("legacy_window_fn_ignored")
        if legacy_name is not None:
            legacy_window_names.add(str(legacy_name))
        weight = int(payload.get("num_samples", 0))
        if weight <= 0:
            continue
        total_samples += weight
        for key in list(accumulators.keys()):
            arr = np.asarray(payload[key], dtype=float)
            if accumulators[key] is None:
                accumulators[key] = arr * float(weight)
            else:
                accumulators[key] = accumulators[key] + arr * float(weight)

    if total_samples <= 0:
        raise ValueError("all payloads were empty")

    merged: Dict[str, Any] = {
        "exact_freqs": exact_freqs.astype(np.float32),
        "spectrogram_freqs": spectrogram_freqs.astype(np.float32),
        "spectrogram_frame_centers": spectrogram_frames.astype(np.float32),
        "periodogram_length": int(periodogram_length),
        "spectrogram_window_length": int(spectrogram_window_length),
        "spectrogram_overlap_length": int(spectrogram_overlap_length),
        "num_samples": int(total_samples),
        "num_rows": int(num_rows),
        "variants_saved": list(_PSD_VARIANTS),
        "taper_window_applied": False,
        "legacy_window_fn_ignored": None if len(legacy_window_names) == 0 else sorted(legacy_window_names)[0],
    }
    if len(legacy_window_names) > 1:
        raise ValueError(f"legacy window names must match across payloads, got {sorted(legacy_window_names)}")
    for key, value in accumulators.items():
        if value is None:
            raise ValueError(f"missing accumulator for {key}")
        merged[key] = (value / float(total_samples)).astype(np.float32)
    return merged



def _large_heatmap_figsize(num_rows: int, num_cols: int) -> tuple[float, float]:
    width = max(28.0, 2.4 * max(int(num_cols), 1) + 8.0)
    height = max(16.0, 0.28 * max(int(num_rows), 1) + 5.0)
    return float(width), float(height)



def _spectrogram_figsize(num_freqs: int, num_frames: int) -> tuple[float, float]:
    width = max(12.0, 0.9 * max(int(num_frames), 1) + 6.0)
    height = max(8.0, 0.16 * max(int(num_freqs), 1) + 4.0)
    return float(width), float(height)



def _spectrogram_user_heatmap_figsize(num_rows: int, num_cols: int) -> tuple[float, float]:
    width = max(12.0, 0.55 * max(int(num_cols), 1) + 6.0)
    height = max(8.0, 0.18 * max(int(num_rows), 1) + 4.0)
    return float(width), float(height)



def _spectrogram_user_heatmap_labels(frame_centers: np.ndarray, band_centers: np.ndarray, *, max_labels: int) -> Optional[List[str]]:
    frame_centers = np.asarray(frame_centers, dtype=float).reshape(-1)
    band_centers = np.asarray(band_centers, dtype=float).reshape(-1)
    total = int(frame_centers.size * band_centers.size)
    if total <= 0 or total > int(max_labels):
        return None
    labels: List[str] = []
    for frame in frame_centers.tolist():
        for band in band_centers.tolist():
            labels.append(f"{float(frame):.1f}\n{float(band):.3f}")
    return labels



def _variant_scalar_summary(
    *,
    waveform: np.ndarray,
    heatmap: np.ndarray,
    spectrogram: np.ndarray,
    spectrogram_user: np.ndarray,
) -> Dict[str, Any]:
    def _safe_scalar(arr: np.ndarray, reducer) -> float:
        arr_f = np.asarray(arr, dtype=float)
        if arr_f.size == 0:
            return 0.0
        return float(reducer(arr_f))

    return {
        "waveform_mean": _safe_scalar(waveform, np.mean),
        "waveform_max": _safe_scalar(waveform, np.max),
        "periodogram_heatmap_mean": _safe_scalar(heatmap, np.mean),
        "periodogram_heatmap_max": _safe_scalar(heatmap, np.max),
        "spectrogram_mean": _safe_scalar(spectrogram, np.mean),
        "spectrogram_max": _safe_scalar(spectrogram, np.max),
        "spectrogram_heatmap_mean": _safe_scalar(spectrogram_user, np.mean),
        "spectrogram_heatmap_max": _safe_scalar(spectrogram_user, np.max),
        "spectrogram_heatmap_shape": list(np.asarray(spectrogram_user, dtype=int).shape),
    }



def _power_like_to_db(values: np.ndarray, *, eps: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return 10.0 * np.log10(np.maximum(arr, 0.0) + float(eps))



def save_psd_bundle(
    base_dir: str,
    *,
    payload: Mapping[str, Any],
    userbin_centers_np: np.ndarray,
    title_prefix: str,
    signal_scope: str,
    epoch: Optional[int],
    save_summary_json: bool = True,
    save_db_plots: bool = False,
    db_eps: float = 1.0e-12,
) -> None:
    os.makedirs(base_dir, exist_ok=True)
    exact_freqs = np.asarray(payload["exact_freqs"], dtype=float)
    spectrogram_freqs = np.asarray(payload["spectrogram_freqs"], dtype=float)
    spectrogram_frames = np.asarray(payload["spectrogram_frame_centers"], dtype=float)
    num_rows = int(payload["num_rows"])
    userbin_centers_arr = np.asarray(userbin_centers_np, dtype=float)

    width, height = _large_heatmap_figsize(num_rows, int(userbin_centers_arr.size))
    spec_x_tick_labels = None
    spec_y_tick_labels = [f"{float(v):.3f}" for v in spectrogram_freqs.tolist()] if int(spectrogram_freqs.size) <= 64 else None
    x_tick_labels = [f"{float(v):.3f}" for v in userbin_centers_arr.reshape(-1).tolist()]
    y_tick_labels = [str(i) for i in range(num_rows)] if num_rows <= 128 else None
    db_plot_files: List[str] = []

    summary_variants: Dict[str, Any] = {}
    for variant in _PSD_VARIANTS:
        waveform = np.asarray(payload[f"set_mean_psd_exact_{variant}"], dtype=float)
        heatmap = np.asarray(payload[f"set_mean_heatmap_user_{variant}"], dtype=float)
        spectrogram_mean = np.asarray(payload[f"set_mean_spectrogram_exact_{variant}"], dtype=float)
        spectrogram_user = np.asarray(payload[f"set_mean_spectrogram_user_{variant}"], dtype=float)
        if spectrogram_user.ndim != 3:
            raise ValueError(f"set_mean_spectrogram_user_{variant} must be (rows, bands, frames), got {spectrogram_user.shape}")
        num_bands = int(spectrogram_user.shape[1])
        num_frames = int(spectrogram_user.shape[2])
        spectrogram_user_heatmap = np.transpose(spectrogram_user, (0, 2, 1)).reshape(num_rows, num_frames * num_bands)

        spec_width, spec_height = _spectrogram_figsize(int(spectrogram_mean.shape[0]), int(spectrogram_mean.shape[1]))
        spec_user_width, spec_user_height = _spectrogram_user_heatmap_figsize(num_rows, int(spectrogram_user_heatmap.shape[1]))
        spec_x_tick_labels = [f"{float(v):.1f}" for v in spectrogram_frames.tolist()] if int(spectrogram_mean.shape[1]) <= 24 else None
        spec_user_x_tick_labels = _spectrogram_user_heatmap_labels(spectrogram_frames, userbin_centers_arr, max_labels=24)

        save_line_plot(
            os.path.join(base_dir, f"mean_psd_waveform_exact_{variant}.png"),
            {f"mean_psd_exact_{variant}": waveform},
            x=exact_freqs,
            xlabel="frequency (cycles/sample)",
            ylabel="mean PSD",
            title=f"{title_prefix} mean PSD waveform (exact periodogram, {variant})",
        )
        save_heatmap_plot(
            os.path.join(base_dir, f"element_psd_heatmap_userbin_{variant}.png"),
            heatmap,
            title=f"{title_prefix} element PSD heatmap (userbin, {variant})",
            xlabel="frequency (cycles/sample)",
            ylabel="element index",
            use_log1p=False,
            center_zero=False,
            origin="lower",
            annotate_all_cells=True,
            value_format="{:.3e}",
            figsize=(width, height),
            x_tick_labels=x_tick_labels,
            y_tick_labels=y_tick_labels,
        )
        save_heatmap_plot(
            os.path.join(base_dir, f"mean_spectrogram_exact_{variant}.png"),
            spectrogram_mean,
            title=f"{title_prefix} mean spectrogram (sliding simple periodogram, {variant})",
            xlabel="time step (frame center)",
            ylabel="frequency (cycles/sample)",
            use_log1p=False,
            center_zero=False,
            origin="lower",
            annotate_all_cells=False,
            figsize=(spec_width, spec_height),
            x_tick_labels=spec_x_tick_labels,
            y_tick_labels=spec_y_tick_labels,
        )
        save_heatmap_plot(
            os.path.join(base_dir, f"element_spectrogram_heatmap_userbin_{variant}.png"),
            spectrogram_user_heatmap,
            title=f"{title_prefix} element spectrogram heatmap (frame-major userbin, {variant})",
            xlabel="frame center / frequency userbin",
            ylabel="element index",
            use_log1p=False,
            center_zero=False,
            origin="lower",
            annotate_all_cells=False,
            figsize=(spec_user_width, spec_user_height),
            x_tick_labels=spec_user_x_tick_labels,
            y_tick_labels=y_tick_labels,
        )

        if bool(save_db_plots):
            waveform_db = _power_like_to_db(waveform, eps=float(db_eps))
            heatmap_db = _power_like_to_db(heatmap, eps=float(db_eps))
            spectrogram_mean_db = _power_like_to_db(spectrogram_mean, eps=float(db_eps))
            spectrogram_user_heatmap_db = _power_like_to_db(spectrogram_user_heatmap, eps=float(db_eps))

            waveform_db_name = f"mean_psd_waveform_exact_{variant}_db.png"
            heatmap_db_name = f"element_psd_heatmap_userbin_{variant}_db.png"
            spectrogram_db_name = f"mean_spectrogram_exact_{variant}_db.png"
            spectrogram_user_db_name = f"element_spectrogram_heatmap_userbin_{variant}_db.png"

            save_line_plot(
                os.path.join(base_dir, waveform_db_name),
                {f"mean_psd_exact_{variant}_db": waveform_db},
                x=exact_freqs,
                xlabel="frequency (cycles/sample)",
                ylabel="mean PSD (dB)",
                title=f"{title_prefix} mean PSD waveform (exact periodogram, {variant}, dB)",
            )
            save_heatmap_plot(
                os.path.join(base_dir, heatmap_db_name),
                heatmap_db,
                title=f"{title_prefix} element PSD heatmap (userbin, {variant}, dB)",
                xlabel="frequency (cycles/sample)",
                ylabel="element index",
                use_log1p=False,
                center_zero=False,
                origin="lower",
                annotate_all_cells=True,
                value_format="{:.2f}",
                figsize=(width, height),
                x_tick_labels=x_tick_labels,
                y_tick_labels=y_tick_labels,
            )
            save_heatmap_plot(
                os.path.join(base_dir, spectrogram_db_name),
                spectrogram_mean_db,
                title=f"{title_prefix} mean spectrogram (sliding simple periodogram, {variant}, dB)",
                xlabel="time step (frame center)",
                ylabel="frequency (cycles/sample)",
                use_log1p=False,
                center_zero=False,
                origin="lower",
                annotate_all_cells=False,
                figsize=(spec_width, spec_height),
                x_tick_labels=spec_x_tick_labels,
                y_tick_labels=spec_y_tick_labels,
            )
            save_heatmap_plot(
                os.path.join(base_dir, spectrogram_user_db_name),
                spectrogram_user_heatmap_db,
                title=f"{title_prefix} element spectrogram heatmap (frame-major userbin, {variant}, dB)",
                xlabel="frame center / frequency userbin",
                ylabel="element index",
                use_log1p=False,
                center_zero=False,
                origin="lower",
                annotate_all_cells=False,
                figsize=(spec_user_width, spec_user_height),
                x_tick_labels=spec_user_x_tick_labels,
                y_tick_labels=y_tick_labels,
            )
            db_plot_files.extend(
                [
                    waveform_db_name,
                    heatmap_db_name,
                    spectrogram_db_name,
                    spectrogram_user_db_name,
                ]
            )

        summary_variants[variant] = {
            "mean_psd_waveform_exact": waveform,
            "element_psd_heatmap_userbin": heatmap,
            "mean_spectrogram_exact": spectrogram_mean,
            "element_spectrogram_heatmap_userbin": spectrogram_user,
            "spectrogram_user_heatmap_flat_shape": list(np.asarray(spectrogram_user_heatmap.shape, dtype=np.int64)),
            "scalar_summary": _variant_scalar_summary(
                waveform=waveform,
                heatmap=heatmap,
                spectrogram=spectrogram_mean,
                spectrogram_user=spectrogram_user,
            ),
        }

    if bool(save_summary_json):
        save_json(
            os.path.join(base_dir, "summary.json"),
            {
                "signal_scope": str(signal_scope),
                "epoch": None if epoch is None else int(epoch),
                "num_samples": int(payload["num_samples"]),
                "num_rows": int(num_rows),
                "frequency_unit": "cycles_per_sample_with_nyquist_0p5",
                "variants_saved": list(_PSD_VARIANTS),
                "periodogram_length": int(payload.get("periodogram_length", exact_freqs.size)),
                "spectrogram_window_length": int(payload.get("spectrogram_window_length", 0)),
                "spectrogram_overlap_length": int(payload.get("spectrogram_overlap_length", 0)),
                "taper_window_applied": False,
                "legacy_window_fn_ignored": payload.get("legacy_window_fn_ignored"),
                "exact_freqs": exact_freqs,
                "userbin_centers": userbin_centers_arr,
                "spectrogram_freqs": spectrogram_freqs,
                "spectrogram_frame_centers": spectrogram_frames,
                "spectrogram_user_heatmap_column_order": "frame_major_then_userbin",
                "heatmap_low_index_at_bottom": True,
                "periodogram_heatmap_annotate_all_cells": True,
                "spectrogram_heatmap_annotate_all_cells": False,
                "db_plots_saved": bool(save_db_plots),
                "db_plot_scale": None if not bool(save_db_plots) else "10log10_power_plus_epsilon",
                "db_plot_epsilon": None if not bool(save_db_plots) else float(db_eps),
                "waveform_representation": "exact_full_length_simple_periodogram_saved_for_raw_and_centered",
                "periodogram_heatmap_representation": "userbin_from_exact_periodogram_saved_for_raw_and_centered",
                "spectrogram_representation": "exact_sliding_simple_periodogram_saved_for_raw_and_centered",
                "spectrogram_heatmap_representation": "userbin_from_exact_spectrogram_frame_major_per_element_saved_for_raw_and_centered",
                "variants": summary_variants,
                "plot_files": [
                    "mean_psd_waveform_exact_raw.png",
                    "mean_psd_waveform_exact_centered.png",
                    "element_psd_heatmap_userbin_raw.png",
                    "element_psd_heatmap_userbin_centered.png",
                    "mean_spectrogram_exact_raw.png",
                    "mean_spectrogram_exact_centered.png",
                    "element_spectrogram_heatmap_userbin_raw.png",
                    "element_spectrogram_heatmap_userbin_centered.png",
                    *db_plot_files,
                ],
            },
        )
