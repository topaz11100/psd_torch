from __future__ import annotations

from src.common.datasets import get_shd_loaders


def build_shd_loaders(
    data_root: str,
    *,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = False,
    T_event: int = 250,
    seed: int = 0,
    shd_max_time: float = 1.0,
    shd_binning: str = "origin",
    shd_unit_indexing: str = "auto",
    shd_channel_flip: bool = True,
    shd_align_to_first_event: bool = False,
    shd_use_event_counts: bool = False,
):
    return get_shd_loaders(
        data_root,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        download=bool(download),
        T=int(T_event),
        seed=int(seed),
        max_time=float(shd_max_time),
        binning=str(shd_binning),
        unit_indexing=str(shd_unit_indexing),
        channel_flip=bool(shd_channel_flip),
        align_to_first_event=bool(shd_align_to_first_event),
        use_event_counts=bool(shd_use_event_counts),
    )
