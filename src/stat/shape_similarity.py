"""Curve-shape semi-metric helpers for centered pointwise L2 summaries."""

from __future__ import annotations

import numpy as np


def centered_l2_semi_metric(x: np.ndarray, y: np.ndarray) -> float:
    """Compute centered pointwise L2 semi-metric between two mean curves."""

    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    xa = xa - xa.mean()
    ya = ya - ya.mean()
    return float(np.sqrt(np.mean((xa - ya) ** 2)))
