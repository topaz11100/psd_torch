"""PSD/spectrogram payload builder and saver."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from .psd_utils import bin_by_user_edges, db10, exact_periodogram, sliding_exact_spectrogram, summary_scalars
from src.plot.plotting import save_heatmap, save_lineplot


def combined_exact_psd_payload_from_maps_torch(maps, userbin_edges: Iterable[float], psd_window: int, psd_overlap: int) -> Dict[str, object]:
    """Build canonical PSD payload for (S,R,T) map tensor/ndarray input."""

    arr = np.asarray(maps, dtype=np.float64)
    exact_f, p_raw = exact_periodogram(arr, centered=False)
    _, p_ctr = exact_periodogram(arr, centered=True)
    wave_raw = p_raw.mean(axis=1).mean(axis=0)
    wave_ctr = p_ctr.mean(axis=1).mean(axis=0)
    user_f, heat_raw = bin_by_user_edges(exact_f, p_raw, userbin_edges)
    _, heat_ctr = bin_by_user_edges(exact_f, p_ctr, userbin_edges)
    heat_raw = heat_raw.mean(axis=0)
    heat_ctr = heat_ctr.mean(axis=0)

    sf, centers, s_raw = sliding_exact_spectrogram(arr, psd_window, psd_overlap, centered=False)
    _, _, s_ctr = sliding_exact_spectrogram(arr, psd_window, psd_overlap, centered=True)
    spec_raw = s_raw.mean(axis=1).mean(axis=0)
    spec_ctr = s_ctr.mean(axis=1).mean(axis=0)

    sraw_sr = np.transpose(s_raw.mean(axis=0), (0, 2, 1))
    sctr_sr = np.transpose(s_ctr.mean(axis=0), (0, 2, 1))
    _, sraw_user = bin_by_user_edges(sf, sraw_sr, userbin_edges)
    _, sctr_user = bin_by_user_edges(sf, sctr_sr, userbin_edges)

    return {
        "exact_freqs": exact_f,
        "spectrogram_freqs": sf,
        "spectrogram_frame_centers": centers,
        "set_mean_psd_exact_raw": wave_raw,
        "set_mean_psd_exact_centered": wave_ctr,
        "set_mean_heatmap_user_raw": heat_raw,
        "set_mean_heatmap_user_centered": heat_ctr,
        "set_mean_spectrogram_exact_raw": spec_raw,
        "set_mean_spectrogram_exact_centered": spec_ctr,
        "set_mean_spectrogram_user_raw": sraw_user,
        "set_mean_spectrogram_user_centered": sctr_user,
        "periodogram_length": int(arr.shape[-1]),
        "spectrogram_window_length": int(psd_window),
        "spectrogram_overlap_length": int(psd_overlap),
        "num_samples": int(arr.shape[0]),
        "num_rows": int(arr.shape[1]),
        "variants_saved": ["raw", "centered"],
        "taper_window_applied": False,
        "legacy_window_fn_ignored": True,
        "userbin_centers": user_f,
    }


def merge_exact_psd_payloads(dst: Dict[str, object], src: Dict[str, object]) -> Dict[str, object]:
    """Merge payloads with num_samples weighted averaging."""

    if not dst:
        return src
    n0 = float(dst["num_samples"])
    n1 = float(src["num_samples"])
    total = n0 + n1
    merged = dict(dst)
    for key in [
        "set_mean_psd_exact_raw",
        "set_mean_psd_exact_centered",
        "set_mean_heatmap_user_raw",
        "set_mean_heatmap_user_centered",
        "set_mean_spectrogram_exact_raw",
        "set_mean_spectrogram_exact_centered",
        "set_mean_spectrogram_user_raw",
        "set_mean_spectrogram_user_centered",
    ]:
        merged[key] = (np.asarray(dst[key]) * n0 + np.asarray(src[key]) * n1) / total
    merged["num_samples"] = int(total)
    return merged


def save_psd_bundle(payload: Dict[str, object], out_dir: str | Path, save_db_plots: bool = True, save_summary_json: bool = True) -> None:
    """Save canonical PNG bundle and summary json for PSD payload."""

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    ef = np.asarray(payload["exact_freqs"])
    uf = np.asarray(payload["userbin_centers"])
    sf = np.asarray(payload["spectrogram_freqs"])

    save_lineplot(ef, np.asarray(payload["set_mean_psd_exact_raw"]), root / "mean_psd_waveform_exact_raw.png")
    save_lineplot(ef, np.asarray(payload["set_mean_psd_exact_centered"]), root / "mean_psd_waveform_exact_centered.png")
    save_heatmap(np.asarray(payload["set_mean_heatmap_user_raw"]), root / "element_psd_heatmap_userbin_raw.png", x_ticks=uf, annotate=True)
    save_heatmap(np.asarray(payload["set_mean_heatmap_user_centered"]), root / "element_psd_heatmap_userbin_centered.png", x_ticks=uf, annotate=True)
    save_heatmap(np.asarray(payload["set_mean_spectrogram_exact_raw"]), root / "mean_spectrogram_exact_raw.png", y_ticks=sf)
    save_heatmap(np.asarray(payload["set_mean_spectrogram_exact_centered"]), root / "mean_spectrogram_exact_centered.png", y_ticks=sf)

    for variant in ["raw", "centered"]:
        x = np.asarray(payload[f"set_mean_spectrogram_user_{variant}"])
        flat = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        save_heatmap(flat, root / f"element_spectrogram_heatmap_userbin_{variant}.png")

    if save_db_plots:
        save_lineplot(ef, db10(np.asarray(payload["set_mean_psd_exact_raw"])), root / "mean_psd_waveform_exact_raw_db.png")
        save_lineplot(ef, db10(np.asarray(payload["set_mean_psd_exact_centered"])), root / "mean_psd_waveform_exact_centered_db.png")
        save_heatmap(db10(np.asarray(payload["set_mean_heatmap_user_raw"])), root / "element_psd_heatmap_userbin_raw_db.png", x_ticks=uf, annotate=True)
        save_heatmap(db10(np.asarray(payload["set_mean_heatmap_user_centered"])), root / "element_psd_heatmap_userbin_centered_db.png", x_ticks=uf, annotate=True)
        save_heatmap(db10(np.asarray(payload["set_mean_spectrogram_exact_raw"])), root / "mean_spectrogram_exact_raw_db.png", y_ticks=sf)
        save_heatmap(db10(np.asarray(payload["set_mean_spectrogram_exact_centered"])), root / "mean_spectrogram_exact_centered_db.png", y_ticks=sf)
        for variant in ["raw", "centered"]:
            x = np.asarray(payload[f"set_mean_spectrogram_user_{variant}"])
            flat = db10(x.reshape(x.shape[0], x.shape[1] * x.shape[2]))
            save_heatmap(flat, root / f"element_spectrogram_heatmap_userbin_{variant}_db.png")

    if save_summary_json:
        summary = {
            "metadata": {
                "variants_saved": payload["variants_saved"],
                "taper_window_applied": False,
                "legacy_window_fn_ignored": True,
                "db_epsilon": 1e-12,
            },
            "raw": summary_scalars(np.asarray(payload["set_mean_psd_exact_raw"])),
            "centered": summary_scalars(np.asarray(payload["set_mean_psd_exact_centered"])),
        }
        (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
