"""Deferred plot payload queue and renderer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Tuple

from tqdm import tqdm


@dataclass
class DeferredPlotTask:
    """Deferred plotting task object."""

    fn: Callable[..., Any]
    args: Tuple[Any, ...]
    kwargs: dict


_TASKS: List[DeferredPlotTask] = []


def add_deferred_plot_task(fn: Callable[..., Any], *args, **kwargs) -> None:
    """Append task to process-local deferred list."""

    _TASKS.append(DeferredPlotTask(fn=fn, args=args, kwargs=kwargs))


def render_deferred_plot_tasks() -> int:
    """Render tasks sequentially with tqdm and remove after success."""

    count = 0
    while _TASKS:
        task = _TASKS.pop(0)
        for _ in tqdm([0], desc="render_plot", leave=False):
            task.fn(*task.args, **task.kwargs)
        count += 1
    return count


def flush_plot_tasks() -> None:
    """Compatibility alias used by main entry points."""

    return None


def shutdown_plot_worker(wait: bool = True) -> None:
    """Compatibility alias used by main entry points."""

    return None
