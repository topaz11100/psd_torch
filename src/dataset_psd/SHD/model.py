from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class SHDDatasetPSDExperimentConfig:
    data_root: str
    out_root: str
    T: int = 250
    num_units: int = 700
    max_time: float = 1.0
    binning: str = "origin"
    unit_indexing: str = "auto"
    channel_flip: bool = True
    align_to_first_event: bool = False
    use_event_counts: bool = False
    batch_size: int = 256
    download: bool = False
    psd_window: int = 64
    psd_overlap: int = 32
    window_fn: str = "hann"
    userbin_edges: Optional[Sequence[float]] = None
    max_samples: Optional[int] = None
    exp_name: Optional[str] = None
    timestamp: Optional[str] = None
    device: str = "auto"
