"""Minimal plot writer interface used by analysis modules."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PLOT_TASKS = []


def save_lineplot(x: np.ndarray, y: np.ndarray, out_path: str | Path, title: str = "") -> None:
    """Save a simple line plot for waveform-style data."""

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(x, y)
    if title:
        plt.title(title)
    plt.xlabel("frequency")
    plt.ylabel("value")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def save_heatmap(matrix: np.ndarray, out_path: str | Path, x_ticks: Iterable[float] | None = None, y_ticks: Iterable[float] | None = None, annotate: bool = False) -> None:
    """Save a heatmap with lower origin as required by spec."""

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(matrix)
    plt.figure(figsize=(10, 5))
    plt.imshow(arr, aspect="auto", origin="lower")
    plt.colorbar()
    if annotate:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                plt.text(j, i, f"{arr[i, j]:.2g}", ha="center", va="center", fontsize=6)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def enqueue_plot_task(callable_obj, *args, **kwargs) -> None:
    """Store deferred plot task in process-local queue."""

    PLOT_TASKS.append((callable_obj, args, kwargs))


def flush_plot_tasks() -> None:
    """Compatibility no-op for async writer interface."""

    return None


def shutdown_plot_worker(wait: bool = True) -> None:
    """Compatibility no-op for async writer interface."""

    return None
