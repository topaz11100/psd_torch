"""Attenuation statistic summary helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class AttenuationStats:
    """Basic attenuation statistics container."""

    mean: float
    std: float
    min: float
    max: float


def summarize(values) -> Dict[str, float]:
    """Summarize attenuation-like parameter values."""

    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
