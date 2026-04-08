from __future__ import annotations

from typing import Any

from src.common.dataset_psd_driver import run_dataset_psd_shd


def run_from_config(cfg: Any) -> str:
    return run_dataset_psd_shd(
        data_root=cfg.data_root,
        out_root=cfg.out_root,
        T=cfg.T,
        num_units=cfg.num_units,
        max_time=cfg.max_time,
        binning=cfg.binning,
        unit_indexing=cfg.unit_indexing,
        channel_flip=cfg.channel_flip,
        align_to_first_event=cfg.align_to_first_event,
        use_event_counts=cfg.use_event_counts,
        batch_size=cfg.batch_size,
        download=cfg.download,
        psd_window=cfg.psd_window,
        psd_overlap=cfg.psd_overlap,
        window_fn=cfg.window_fn,
        userbin_edges=cfg.userbin_edges,
        max_samples=cfg.max_samples,
        exp_name=cfg.exp_name,
        timestamp=cfg.timestamp,
        device=cfg.device,
    )
