"""Output path helpers for consistent run directory layout."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def build_run_root(out_root: str, dataset: str, model: str, readout_mode: str, timestamp: str | None = None) -> Path:
    """Build standardized run root path."""

    ts = timestamp or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    root = Path(out_root) / dataset / model / readout_mode / ts
    root.mkdir(parents=True, exist_ok=True)
    return root
