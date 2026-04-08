from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from src.plot.plotting import flush_plot_tasks, save_bar_plot, save_hist_bar, save_line_plot
from src.signal.psd_artifacts import save_psd_bundle


# Keep deferred plot payloads in process-local CPU memory during training so
# the training loop never blocks on Matplotlib rendering or per-task disk I/O.
# Each training job runs in its own Python process, so per-process globals are
# sufficient even when multiple scenarios are launched in parallel via bash.
_DEFERRED_PLOT_TASKS: List[Optional[Dict[str, Any]]] = []
_DEFERRED_PLOT_TASK_COUNTER: int = 0


def deferred_plot_metadata() -> Dict[str, Any]:
    return {
        "plot_generation_strategy": "hold_numeric_plot_payloads_in_process_memory_render_after_training",
        "plot_payload_storage_backend": "process_local_ram",
        "plot_payload_file_suffix": None,
        "plot_payload_cleanup": "drop_from_memory_after_successful_render",
        "plot_render_progress": "tqdm",
    }



def _copy_ndarray(value: Any, *, dtype: Optional[np.dtype] = None) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    return np.array(arr, copy=True)



def _normalize_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().copy()
    if isinstance(value, Mapping):
        return {str(key): _normalize_value(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return value



def _append_task(task: Mapping[str, Any], *, normalize: bool = True) -> int:
    global _DEFERRED_PLOT_TASK_COUNTER
    _DEFERRED_PLOT_TASK_COUNTER += 1
    if normalize:
        stored = {str(key): _normalize_value(value) for key, value in dict(task).items()}
    else:
        stored = {str(key): value for key, value in dict(task).items()}
    stored["_task_id"] = int(_DEFERRED_PLOT_TASK_COUNTER)
    _DEFERRED_PLOT_TASKS.append(stored)
    return int(_DEFERRED_PLOT_TASK_COUNTER)



def _task_target(task: Mapping[str, Any]) -> Optional[str]:
    for key in ("path", "base_dir"):
        value = task.get(key)
        if value is None:
            continue
        try:
            return os.path.abspath(str(value))
        except Exception:
            return str(value)
    return None



def _relative_target(run_root: str, task: Mapping[str, Any]) -> str:
    target = _task_target(task)
    if target is None:
        return str(task.get("kind", "plot"))
    try:
        return os.path.relpath(target, start=str(run_root))
    except Exception:
        return str(target)



def _task_belongs_to_run_root(task: Mapping[str, Any], run_root: str) -> bool:
    target = _task_target(task)
    if target is None:
        return False
    try:
        return os.path.commonpath([str(run_root), str(target)]) == str(run_root)
    except Exception:
        return False



def _compact_task_slots() -> None:
    global _DEFERRED_PLOT_TASKS
    _DEFERRED_PLOT_TASKS = [task for task in _DEFERRED_PLOT_TASKS if task is not None]



def save_deferred_psd_bundle(
    base_dir: str,
    *,
    payload: Mapping[str, Any],
    userbin_centers_np: np.ndarray,
    title_prefix: str,
    signal_scope: str,
    epoch: Optional[int],
    save_summary_json: bool = True,
    save_db_plots: bool = False,
    db_eps: float = 1.0e-12,
) -> str:
    os.makedirs(base_dir, exist_ok=True)
    _append_task(
        {
            "kind": "psd_bundle",
            "base_dir": str(base_dir),
            "payload": _normalize_value(payload),
            "userbin_centers_np": _copy_ndarray(userbin_centers_np, dtype=np.float32),
            "title_prefix": str(title_prefix),
            "signal_scope": str(signal_scope),
            "epoch": None if epoch is None else int(epoch),
            "save_summary_json": bool(save_summary_json),
            "save_db_plots": bool(save_db_plots),
            "db_eps": float(db_eps),
        },
        normalize=False,
    )
    return str(base_dir)



def save_deferred_line_plot(
    path: str,
    y_dict: Mapping[str, Sequence[float]],
    *,
    x: Optional[Sequence[float]] = None,
    title: str = "",
    xlabel: str = "epoch",
    ylabel: str = "",
    figsize: Optional[Sequence[float]] = None,
) -> str:
    _append_task(
        {
            "kind": "line_plot",
            "path": str(path),
            "y_dict": {str(name): _copy_ndarray(values, dtype=np.float32).reshape(-1) for name, values in y_dict.items()},
            "x": None if x is None else _copy_ndarray(x, dtype=np.float32).reshape(-1),
            "title": str(title),
            "xlabel": str(xlabel),
            "ylabel": str(ylabel),
            "figsize": None if figsize is None else _copy_ndarray(figsize, dtype=np.float32).reshape(-1),
        },
        normalize=False,
    )
    return str(path)



def save_deferred_hist_bar(
    path: str,
    values: Sequence[float],
    *,
    bins: int = 60,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "count",
    linewidth: float = 0.8,
) -> str:
    _append_task(
        {
            "kind": "hist_bar",
            "path": str(path),
            "values": _copy_ndarray(values, dtype=np.float32).reshape(-1),
            "bins": int(bins),
            "title": str(title),
            "xlabel": str(xlabel),
            "ylabel": str(ylabel),
            "linewidth": float(linewidth),
        },
        normalize=False,
    )
    return str(path)



def save_deferred_bar_plot(
    path: str,
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    rotation: float = 0.0,
    figsize: Optional[Sequence[float]] = None,
) -> str:
    _append_task(
        {
            "kind": "bar_plot",
            "path": str(path),
            "labels": [str(v) for v in labels],
            "values": _copy_ndarray(values, dtype=np.float32).reshape(-1),
            "title": str(title),
            "xlabel": str(xlabel),
            "ylabel": str(ylabel),
            "rotation": float(rotation),
            "figsize": None if figsize is None else _copy_ndarray(figsize, dtype=np.float32).reshape(-1),
        },
        normalize=False,
    )
    return str(path)



def _render_task(task: Mapping[str, Any]) -> None:
    kind = str(task["kind"])
    if kind == "psd_bundle":
        save_psd_bundle(
            str(task["base_dir"]),
            payload=task["payload"],
            userbin_centers_np=np.asarray(task["userbin_centers_np"], dtype=np.float32),
            title_prefix=str(task["title_prefix"]),
            signal_scope=str(task["signal_scope"]),
            epoch=None if task.get("epoch") is None else int(task["epoch"]),
            save_summary_json=bool(task.get("save_summary_json", True)),
            save_db_plots=bool(task.get("save_db_plots", False)),
            db_eps=float(task.get("db_eps", 1.0e-12)),
        )
        return
    if kind == "line_plot":
        save_line_plot(
            str(task["path"]),
            task["y_dict"],
            x=task.get("x"),
            title=str(task.get("title", "")),
            xlabel=str(task.get("xlabel", "epoch")),
            ylabel=str(task.get("ylabel", "")),
            figsize=task.get("figsize"),
        )
        return
    if kind == "hist_bar":
        save_hist_bar(
            str(task["path"]),
            task["values"],
            bins=int(task.get("bins", 60)),
            title=str(task.get("title", "")),
            xlabel=str(task.get("xlabel", "")),
            ylabel=str(task.get("ylabel", "count")),
            linewidth=float(task.get("linewidth", 0.8)),
        )
        return
    if kind == "bar_plot":
        save_bar_plot(
            str(task["path"]),
            task["labels"],
            task["values"],
            title=str(task.get("title", "")),
            xlabel=str(task.get("xlabel", "")),
            ylabel=str(task.get("ylabel", "")),
            rotation=float(task.get("rotation", 0.0)),
            figsize=task.get("figsize"),
        )
        return
    raise ValueError(f"unknown deferred plot task kind: {kind}")



def count_deferred_plot_tasks(run_root: str) -> int:
    run_root_abs = os.path.abspath(str(run_root))
    count = 0
    for task in _DEFERRED_PLOT_TASKS:
        if task is None:
            continue
        if _task_belongs_to_run_root(task, run_root_abs):
            count += 1
    return int(count)



def render_deferred_plot_tasks(run_root: str, *, progress_desc: str = "render-plots") -> int:
    run_root_abs = os.path.abspath(str(run_root))
    selected: List[tuple[str, int]] = []
    for idx, task in enumerate(_DEFERRED_PLOT_TASKS):
        if task is None:
            continue
        if _task_belongs_to_run_root(task, run_root_abs):
            selected.append((_relative_target(run_root_abs, task), int(idx)))
    if len(selected) == 0:
        return 0
    selected.sort(key=lambda item: (item[0], item[1]))
    rendered = 0
    with tqdm(selected, total=len(selected), desc=str(progress_desc), leave=True) as pbar:
        for rel_target, idx in pbar:
            task = _DEFERRED_PLOT_TASKS[idx]
            if task is None:
                continue
            pbar.set_postfix_str(rel_target)
            _render_task(task)
            flush_plot_tasks()
            _DEFERRED_PLOT_TASKS[idx] = None
            rendered += 1
            del task
    _compact_task_slots()
    return int(rendered)
