from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class SHDPSDAnalysisExperimentConfig:
    model: str
    out_root: str
    data_root: str
    hidden: Sequence[int]
    epochs: int = 50
    soft_mask_epochs: Optional[int] = None
    stabilize_epochs: int = 0
    ste_epochs: int = 0
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    weight_decay_dend_soma: Optional[float] = None
    seed: int = 0
    S_min: float = 1.0
    S_max: float = 8.0
    th_len: int = 4
    v_th: float = 1.0
    v_pre: float = 1.0
    T_event: int = 250
    num_workers: int = 4
    download: bool = False
    shd_max_time: float = 1.0
    shd_binning: str = "origin"
    shd_unit_indexing: str = "auto"
    shd_channel_flip: bool = True
    shd_align_to_first_event: bool = False
    shd_use_event_counts: bool = False
    lambda_ortho: float = 0.0
    lambda_s: float = 0.0
    same_label_n_per_label: int = 4
    balanced_global_n_per_label: int = 4
    probe_plot: bool = False
    plot_epochs: Optional[Sequence[int]] = None
    psd_window: int = 64
    psd_overlap: int = 32
    window_fn: str = "hann"
    userbin_edges: Optional[Sequence[float]] = None
    rf_reset_mode: str = "no_reset"
    w_clip_edges: Optional[Sequence[float]] = None
    alpha_clip_edges: Optional[Sequence[float]] = None
    band_neuron_ends: Optional[Sequence[str]] = None
    tear: int = 1
    readout_mode: str = "final_membrane"
    exp_name: Optional[str] = None
    timestamp: Optional[str] = None
    device: str = "auto"
