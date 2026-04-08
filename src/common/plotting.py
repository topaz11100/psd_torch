from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import traceback
from typing import Any, Dict, Optional, Sequence

import numpy as np


# -----------------------------------------------------------------------------
# Global plot-writer settings
# -----------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    return int(raw)


_DEFAULT_WORKERS = max(1, min(4, max(1, (os.cpu_count() or 2) // 2)))
_PLOT_PROCESSES: list[mp.Process] = []
_PLOT_TASK_QUEUE: Optional[Any] = None
_PLOT_ERROR_QUEUE: Optional[Any] = None
_PLOT_WORKERS: int = max(1, _env_int("PSD_PLOT_WRITER_WORKERS", _DEFAULT_WORKERS))
_PLOT_QUEUE_MAXSIZE: int = max(4, _env_int("PSD_PLOT_QUEUE_MAXSIZE", max(16, _PLOT_WORKERS * 8)))
_PLOT_PAYLOAD_DTYPE: str = "float32"
_PLOT_START_METHOD: str = str(os.environ.get("PSD_PLOT_WRITER_START_METHOD", "fork" if os.name == "posix" else "spawn"))
_PLOT_BACKEND: str = str(os.environ.get("PSD_PLOT_BACKEND", "Agg"))
_PLOT_DPI: int = max(72, _env_int("PSD_PLOT_WRITER_DPI", 180))
_PLOT_SKIP_EXISTING: bool = str(os.environ.get("PSD_PLOT_SKIP_EXISTING", "0")).strip().lower() in {"1", "true", "yes", "y"}


# -----------------------------------------------------------------------------
# Main-process worker lifecycle helpers
# -----------------------------------------------------------------------------


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(str(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def configure_plot_writer(
    *,
    workers: Optional[int] = None,
    queue_maxsize: Optional[int] = None,
    start_method: Optional[str] = None,
    dpi: Optional[int] = None,
    skip_existing: Optional[bool] = None,
    backend: Optional[str] = None,
) -> None:
    """Update plot-writer defaults before any worker process is started.

    The PSD analysis and dataset-level PSD entry points share the same plotting
    backend, but their preferred runtime settings differ. This helper lets a
    caller pin the worker topology for a specific experiment while keeping the
    public save_* API unchanged.
    """

    global _PLOT_WORKERS, _PLOT_QUEUE_MAXSIZE, _PLOT_START_METHOD, _PLOT_DPI, _PLOT_SKIP_EXISTING, _PLOT_BACKEND

    if _PLOT_PROCESSES:
        raise RuntimeError("plot writer configuration cannot be changed after workers have started")

    if workers is not None:
        _PLOT_WORKERS = max(1, int(workers))
    if queue_maxsize is not None:
        _PLOT_QUEUE_MAXSIZE = max(4, int(queue_maxsize))
    elif workers is not None:
        _PLOT_QUEUE_MAXSIZE = max(4, max(16, int(_PLOT_WORKERS) * 8))
    if start_method is not None:
        _PLOT_START_METHOD = str(start_method)
    if dpi is not None:
        _PLOT_DPI = max(72, int(dpi))
    if skip_existing is not None:
        _PLOT_SKIP_EXISTING = bool(skip_existing)
    if backend is not None:
        _PLOT_BACKEND = str(backend)



def _maybe_get_error() -> Optional[Dict[str, str]]:
    if _PLOT_ERROR_QUEUE is None:
        return None
    try:
        return _PLOT_ERROR_QUEUE.get_nowait()
    except Exception:
        return None



def _check_worker_health() -> None:
    err = _maybe_get_error()
    if err is not None:
        msg = str(err.get("message", "plot worker failed"))
        tb = str(err.get("traceback", ""))
        raise RuntimeError(f"plot worker failed: {msg}\n{tb}".rstrip())
    if _PLOT_PROCESSES:
        for proc in _PLOT_PROCESSES:
            if not proc.is_alive():
                raise RuntimeError("plot worker process exited unexpectedly")



def _start_plot_worker() -> None:
    global _PLOT_TASK_QUEUE, _PLOT_ERROR_QUEUE
    if _PLOT_PROCESSES:
        _check_worker_health()
        return

    ctx = mp.get_context(_PLOT_START_METHOD)
    _PLOT_TASK_QUEUE = ctx.JoinableQueue(maxsize=int(_PLOT_QUEUE_MAXSIZE))
    _PLOT_ERROR_QUEUE = ctx.Queue(maxsize=max(1, int(_PLOT_WORKERS)))
    for _ in range(int(_PLOT_WORKERS)):
        proc = ctx.Process(
            target=_plot_worker_main,
            args=(_PLOT_TASK_QUEUE, _PLOT_ERROR_QUEUE),
            daemon=True,
        )
        proc.start()
        _PLOT_PROCESSES.append(proc)
    _check_worker_health()



def _submit_plot_task(kind: str, payload: Dict[str, Any]) -> None:
    _start_plot_worker()
    _check_worker_health()
    if _PLOT_TASK_QUEUE is None:
        raise RuntimeError("plot worker queue is not initialized")
    _PLOT_TASK_QUEUE.put({"kind": str(kind), "payload": payload})
    _check_worker_health()



def flush_plot_tasks() -> None:
    if not _PLOT_PROCESSES:
        return
    _check_worker_health()
    if _PLOT_TASK_QUEUE is None:
        raise RuntimeError("plot worker queue is not initialized")
    _PLOT_TASK_QUEUE.join()
    _check_worker_health()



def shutdown_plot_worker(wait: bool = True) -> None:
    global _PLOT_PROCESSES, _PLOT_TASK_QUEUE, _PLOT_ERROR_QUEUE
    procs = list(_PLOT_PROCESSES)
    task_queue = _PLOT_TASK_QUEUE
    err_queue = _PLOT_ERROR_QUEUE
    _PLOT_PROCESSES = []
    _PLOT_TASK_QUEUE = None
    _PLOT_ERROR_QUEUE = None
    if not procs:
        return

    stored_exc: Optional[BaseException] = None
    try:
        if wait:
            try:
                if task_queue is not None:
                    task_queue.join()
                err = None
                if err_queue is not None:
                    try:
                        err = err_queue.get_nowait()
                    except Exception:
                        err = None
                if err is not None:
                    stored_exc = RuntimeError(
                        f"plot worker failed: {err.get('message', 'plot worker failed')}\n{err.get('traceback', '')}".rstrip()
                    )
            except BaseException as exc:
                stored_exc = exc
    finally:
        try:
            if task_queue is not None:
                for _ in range(len(procs)):
                    task_queue.put(None)
                task_queue.join()
        except Exception:
            pass
        for proc in procs:
            try:
                proc.join(timeout=5.0 if wait else 0.5)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=1.0)
            except Exception:
                pass
        try:
            if task_queue is not None:
                task_queue.close()
        except Exception:
            pass
        try:
            if err_queue is not None:
                err_queue.close()
        except Exception:
            pass
    if stored_exc is not None:
        raise stored_exc



def plot_writer_metadata() -> Dict[str, int | str | bool]:
    return {
        "plot_writer_mode": "async_multiprocess_queue",
        "plot_writer_workers": int(_PLOT_WORKERS),
        "plot_writer_queue_maxsize": int(_PLOT_QUEUE_MAXSIZE),
        "plot_writer_start_method": str(_PLOT_START_METHOD),
        "plot_writer_backend": str(_PLOT_BACKEND),
        "plot_writer_payload_dtype": str(_PLOT_PAYLOAD_DTYPE),
        "plot_writer_dpi": int(_PLOT_DPI),
        "plot_writer_skip_existing": bool(_PLOT_SKIP_EXISTING),
    }


@atexit.register
def _cleanup_plot_worker() -> None:
    try:
        shutdown_plot_worker(wait=True)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Worker-side rendering
# -----------------------------------------------------------------------------


def _worker_imports():
    import matplotlib

    matplotlib.use(_PLOT_BACKEND)
    import matplotlib.pyplot as plt

    return plt



def _maybe_skip_existing(path: str) -> bool:
    return bool(_PLOT_SKIP_EXISTING) and os.path.exists(path)



def _worker_save_figure(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=int(_PLOT_DPI), bbox_inches="tight")



def _worker_draw_line_plot(
    path: str,
    y_dict: Dict[str, np.ndarray],
    x: Optional[np.ndarray] = None,
    title: str = "",
    xlabel: str = "epoch",
    ylabel: str = "",
    figsize: Optional[np.ndarray] = None,
) -> None:
    if _maybe_skip_existing(path):
        return
    plt = _worker_imports()
    _ensure_parent(path)
    fig = plt.figure(figsize=(6.4, 4.2) if figsize is None else tuple(np.asarray(figsize, dtype=float).reshape(2)))
    ax = fig.add_subplot(111)
    for name, y in y_dict.items():
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        x_arr = np.arange(len(y_arr), dtype=float) if x is None else np.asarray(x, dtype=float).reshape(-1)
        ax.plot(x_arr, y_arr, label=str(name), linewidth=1.6)
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.28)
    if len(y_dict) > 1:
        ax.legend(frameon=False)
    _worker_save_figure(fig, path)
    plt.close(fig)



def _worker_draw_hist_line(
    path: str,
    values: np.ndarray,
    bins: int = 60,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "count",
    histtype: str = "step",
    linewidth: float = 1.6,
) -> None:
    if _maybe_skip_existing(path):
        return
    plt = _worker_imports()
    _ensure_parent(path)
    v = np.asarray(values, dtype=float).reshape(-1)
    fig = plt.figure(figsize=(6.4, 4.2))
    ax = fig.add_subplot(111)
    histtype_eff = str(histtype).strip().lower()
    if histtype_eff == "step":
        ax.hist(v, bins=int(bins), histtype="step", linewidth=float(linewidth))
    elif histtype_eff == "bar":
        ax.hist(v, bins=int(bins), histtype="bar", linewidth=float(linewidth))
    else:
        raise ValueError(f"unsupported histogram type: {histtype}")
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.28)
    _worker_save_figure(fig, path)
    plt.close(fig)



def _auto_heatmap_figsize(rows: int, cols: int, annotate_all_cells: bool) -> tuple[float, float]:
    if annotate_all_cells:
        width = max(24.0, 2.0 * max(cols, 1) + 6.0)
        height = max(14.0, 0.22 * max(rows, 1) + 4.0)
    else:
        width = max(7.2, 0.42 * max(cols, 1) + 4.0)
        height = max(4.6, 0.16 * max(rows, 1) + 3.0)
    return float(width), float(height)



def _worker_draw_heatmap_plot(
    path: str,
    mat: np.ndarray,
    title: str = "",
    xlabel: str = "neuron (or neuron×branch)",
    ylabel: str = "epoch",
    use_log1p: bool = False,
    center_zero: bool = False,
    origin: str = "lower",
    annotate_all_cells: bool = False,
    value_format: str = "{:.3e}",
    figsize: Optional[np.ndarray] = None,
    x_tick_labels: Optional[Sequence[str]] = None,
    y_tick_labels: Optional[Sequence[str]] = None,
    extent: Optional[np.ndarray] = None,
) -> None:
    if _maybe_skip_existing(path):
        return
    plt = _worker_imports()
    _ensure_parent(path)
    arr = np.asarray(mat, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"heatmap input must be 2D, got {arr.shape}")
    arr_plot = np.log1p(np.maximum(arr, 0.0)) if use_log1p else arr
    fig_size = _auto_heatmap_figsize(arr.shape[0], arr.shape[1], bool(annotate_all_cells)) if figsize is None else tuple(np.asarray(figsize, dtype=float).reshape(2))
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    vmin = vmax = None
    if center_zero:
        vmax = float(np.nanmax(np.abs(arr_plot))) if arr_plot.size > 0 else 1.0
        vmin = -vmax
    mm = np.ma.masked_invalid(arr_plot)
    cmap = plt.get_cmap().copy()
    cmap.set_bad(alpha=0.0)
    extent_eff = None if extent is None else tuple(np.asarray(extent, dtype=float).reshape(4))
    im = ax.imshow(mm, aspect="auto", interpolation="nearest", origin=str(origin), cmap=cmap, vmin=vmin, vmax=vmax, extent=extent_eff)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if x_tick_labels is not None:
        ax.set_xticks(np.arange(len(x_tick_labels), dtype=float))
        ax.set_xticklabels(list(x_tick_labels), rotation=45, ha="right")
    if y_tick_labels is not None:
        if len(y_tick_labels) == arr.shape[0]:
            ax.set_yticks(np.arange(len(y_tick_labels), dtype=float))
            ax.set_yticklabels(list(y_tick_labels))
        else:
            ax.set_yticks(np.arange(arr.shape[0], dtype=float))

    if annotate_all_cells:
        num_rows, num_cols = arr.shape
        font_size = max(4.0, min(14.0, 220.0 / max(num_rows, num_cols, 1)))
        for i in range(num_rows):
            for j in range(num_cols):
                value = arr[i, j]
                if np.isnan(value):
                    continue
                ax.text(float(j), float(i), value_format.format(value), ha="center", va="center", fontsize=font_size)

    # Intentionally avoid per-cell grid rendering here. Large annotated heatmaps
    # are the dominant bottleneck, and grid lines multiply artist count without
    # adding information.
    _worker_save_figure(fig, path)
    plt.close(fig)



def _worker_draw_multiline_plot(
    path: str,
    ys: np.ndarray,
    x: Optional[np.ndarray] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    legend_labels: Optional[Sequence[str]] = None,
) -> None:
    if _maybe_skip_existing(path):
        return
    plt = _worker_imports()
    _ensure_parent(path)
    arr = np.asarray(ys, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    xvals = np.arange(arr.shape[1], dtype=float) if x is None else np.asarray(x, dtype=float).reshape(-1)
    fig = plt.figure(figsize=(6.6, 4.3))
    ax = fig.add_subplot(111)
    for i in range(arr.shape[0]):
        label = None if legend_labels is None or i >= len(legend_labels) else str(legend_labels[i])
        ax.plot(xvals, arr[i], linewidth=1.2, label=label)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.28)
    if legend_labels is not None and len(legend_labels) > 1:
        ax.legend(frameon=False, fontsize=7, ncol=2)
    _worker_save_figure(fig, path)
    plt.close(fig)



def _worker_draw_bar_plot(
    path: str,
    labels: Sequence[str],
    values: np.ndarray,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    rotation: float = 0.0,
    figsize: Optional[np.ndarray] = None,
) -> None:
    if _maybe_skip_existing(path):
        return
    plt = _worker_imports()
    _ensure_parent(path)
    vals = np.asarray(values, dtype=float).reshape(-1)
    labs = [str(v) for v in labels]
    fig = plt.figure(figsize=(7.0, 4.6) if figsize is None else tuple(np.asarray(figsize, dtype=float).reshape(2)))
    ax = fig.add_subplot(111)
    xpos = np.arange(len(labs), dtype=float)
    ax.bar(xpos, vals)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labs, rotation=float(rotation), ha="right" if float(rotation) != 0.0 else "center")
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.28)
    _worker_save_figure(fig, path)
    plt.close(fig)



def _plot_worker_main(task_queue, error_queue) -> None:
    failed = False
    while True:
        task = task_queue.get()
        try:
            if task is None:
                return
            if failed:
                continue
            kind = str(task["kind"])
            payload = dict(task["payload"])
            if kind == "line":
                _worker_draw_line_plot(**payload)
            elif kind == "hist":
                _worker_draw_hist_line(**payload)
            elif kind == "heatmap":
                _worker_draw_heatmap_plot(**payload)
            elif kind == "multiline":
                _worker_draw_multiline_plot(**payload)
            elif kind == "bar":
                _worker_draw_bar_plot(**payload)
            else:
                raise ValueError(f"unsupported plot task kind: {kind}")
        except Exception as exc:
            failed = True
            try:
                error_queue.put_nowait(
                    {
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            except Exception:
                pass
        finally:
            task_queue.task_done()


# -----------------------------------------------------------------------------
# Payload normalization
# -----------------------------------------------------------------------------


def _float32_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)



def _float32_optional_array(values: Optional[Any]) -> Optional[np.ndarray]:
    if values is None:
        return None
    return _float32_array(values)



def _float32_series_dict(y_dict: Dict[str, Sequence[float]]) -> Dict[str, np.ndarray]:
    return {str(name): _float32_array(values).reshape(-1) for name, values in y_dict.items()}


# -----------------------------------------------------------------------------
# Public APIs
# -----------------------------------------------------------------------------


def save_line_plot(
    path: str,
    y_dict: Dict[str, Sequence[float]],
    x: Optional[Sequence[float]] = None,
    title: str = "",
    xlabel: str = "epoch",
    ylabel: str = "",
    figsize: Optional[Sequence[float]] = None,
) -> None:
    _submit_plot_task(
        "line",
        {
            "path": str(path),
            "y_dict": _float32_series_dict(y_dict),
            "x": _float32_optional_array(x),
            "title": str(title),
            "xlabel": str(xlabel),
            "ylabel": str(ylabel),
            "figsize": _float32_optional_array(figsize),
        },
    )



def save_hist_line(
    path: str,
    values: Sequence[float],
    bins: int = 60,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "count",
) -> None:
    _submit_plot_task(
        "hist",
        {
            "path": str(path),
            "values": _float32_array(values).reshape(-1),
            "bins": int(bins),
            "title": str(title),
            "xlabel": str(xlabel),
            "ylabel": str(ylabel),
            "histtype": "step",
            "linewidth": 1.6,
        },
    )



def save_hist_bar(
    path: str,
    values: Sequence[float],
    bins: int = 60,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "count",
    linewidth: float = 0.8,
) -> None:
    _submit_plot_task(
        "hist",
        {
            "path": str(path),
            "values": _float32_array(values).reshape(-1),
            "bins": int(bins),
            "title": str(title),
            "xlabel": str(xlabel),
            "ylabel": str(ylabel),
            "histtype": "bar",
            "linewidth": float(linewidth),
        },
    )



def save_heatmap_plot(
    path: str,
    mat: np.ndarray,
    title: str = "",
    xlabel: str = "neuron (or neuron×branch)",
    ylabel: str = "epoch",
    use_log1p: bool = False,
    center_zero: bool = False,
    origin: str = "lower",
    annotate_all_cells: bool = False,
    value_format: str = "{:.3e}",
    figsize: Optional[Sequence[float]] = None,
    x_tick_labels: Optional[Sequence[str]] = None,
    y_tick_labels: Optional[Sequence[str]] = None,
    extent: Optional[Sequence[float]] = None,
) -> None:
    _submit_plot_task(
        "heatmap",
        {
            "path": str(path),
            "mat": _float32_array(mat),
            "title": str(title),
            "xlabel": str(xlabel),
            "ylabel": str(ylabel),
            "use_log1p": bool(use_log1p),
            "center_zero": bool(center_zero),
            "origin": str(origin),
            "annotate_all_cells": bool(annotate_all_cells),
            "value_format": str(value_format),
            "figsize": _float32_optional_array(figsize),
            "x_tick_labels": None if x_tick_labels is None else [str(v) for v in x_tick_labels],
            "y_tick_labels": None if y_tick_labels is None else [str(v) for v in y_tick_labels],
            "extent": _float32_optional_array(extent),
        },
    )



def save_multiline_series_plot(
    path: str,
    ys: np.ndarray,
    x: Optional[np.ndarray] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    legend_labels: Optional[Sequence[str]] = None,
) -> None:
    _submit_plot_task(
        "multiline",
        {
            "path": str(path),
            "ys": _float32_array(ys),
            "x": _float32_optional_array(x),
            "title": str(title),
            "xlabel": str(xlabel),
            "ylabel": str(ylabel),
            "legend_labels": None if legend_labels is None else [str(v) for v in legend_labels],
        },
    )



def save_bar_plot(
    path: str,
    labels: Sequence[str],
    values: Sequence[float],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    rotation: float = 0.0,
    figsize: Optional[Sequence[float]] = None,
) -> None:
    _submit_plot_task(
        "bar",
        {
            "path": str(path),
            "labels": [str(v) for v in labels],
            "values": _float32_array(values).reshape(-1),
            "title": str(title),
            "xlabel": str(xlabel),
            "ylabel": str(ylabel),
            "rotation": float(rotation),
            "figsize": _float32_optional_array(figsize),
        },
    )
