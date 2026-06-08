"""Integrated frequency-bin discriminative-index extractor.

Run with ``python -m src.DI --config config/DI.yaml``.  The module remains
independent of the model-training pipeline, but now uses the project YAML/path
helpers and accepts the same nested clean-config style as the other stages.
It reads prepared ``data_prep`` single-structured ``.npy`` bundles and their
``manifest.yaml`` files directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.util.config import compact_yaml, load_manifest, load_structured, resolve_manifest_path, save_key_value_csv, save_yaml

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting failure is handled at runtime.
    matplotlib = None
    plt = None


SOURCE_PROGRAM = "DI"
STORAGE_FORMAT = "single_structured_npy_v1"
DEFAULT_STAGE_KEYS = ("DI", "di", "discriminative_index")
KST = timezone(timedelta(hours=9), name="Asia/Seoul")
DB_EPSILON = 1.0e-12

DATASET_ALIASES = {
    "smnist": "s-mnist",
    "s_mnist": "s-mnist",
    "s-mnist": "s-mnist",
    "psmnist": "ps-mnist",
    "ps_mnist": "ps-mnist",
    "ps-mnist": "ps-mnist",
    "scifar10": "s-cifar10",
    "s_cifar10": "s-cifar10",
    "s-cifar10": "s-cifar10",
    "cifar10": "cifar-10",
    "cifar_10": "cifar-10",
    "cifar-10": "cifar-10",
    "cifar100": "cifar-100",
    "cifar_100": "cifar-100",
    "cifar-100": "cifar-100",
    "nmnist": "n-mnist",
    "n_mnist": "n-mnist",
    "n-mnist": "n-mnist",
    "cifar10_dvs": "cifar10-dvs",
    "cifar10-dvs": "cifar10-dvs",
    "dvs128_gesture": "dvs128-gesture",
    "dvs128-gesture": "dvs128-gesture",
    "ucihar": "uci-har",
    "uci_har": "uci-har",
    "uci-har": "uci-har",
    "shd": "shd",
    "ssc": "ssc",
    "mnist": "mnist",
    "deap": "deap",
}


def canonicalize_dataset_name(name: str) -> str:
    token = str(name).strip().lower()
    return DATASET_ALIASES.get(token, token)


def safe_token(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "_")
    text = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text)
    return text.strip("_") or "value"


def load_mapping(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = load_structured(resolved)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must be a mapping: {resolved}")
    return dict(payload)


def write_metadata_csv(path: str | Path, payload: Mapping[str, Any]) -> None:
    save_key_value_csv(path, payload)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [item for item in value if item not in (None, "")]
    if isinstance(value, str):
        if value.strip() == "":
            return []
        if "," in value:
            return [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    return [value]


def flatten_config_section(section: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten nested DI config groups by leaf argument name."""

    flattened: dict[str, Any] = {}

    def walk(mapping: Mapping[str, Any], prefix: str = "") -> None:
        for key, value in mapping.items():
            name = str(key)
            child = f"{prefix}.{name}" if prefix else name
            if isinstance(value, Mapping):
                walk(value, child)
                continue
            if name in flattened:
                raise ValueError(f"Duplicate DI config leaf key after flattening: {name!r} at {child!r}")
            flattened[name] = value

    walk(section)
    return flattened




def config_value(section: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Return a DI config value while treating clean-template blanks as missing."""

    value = section.get(key, default)
    if value in (None, ""):
        return default
    if isinstance(value, (list, tuple)) and len(value) == 0:
        return default
    return value

def str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "on", "y"}:
        return True
    if token in {"0", "false", "no", "off", "n", "none", "null"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def maybe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def normalize_signal_window(value: Any) -> str:
    if isinstance(value, bool):
        return "hann" if value else "none"
    if value is None:
        return "hann"
    token = str(value).strip().lower().replace("-", "_")
    aliases = {
        "hann": "hann",
        "hanning": "hann",
        "none": "none",
        "rect": "none",
        "rectangular": "none",
        "boxcar": "none",
        "off": "none",
        "false": "none",
        "0": "none",
        "no": "none",
    }
    if token not in aliases:
        raise ValueError(f"Unsupported signal window {value!r}; allowed: hann, none.")
    return aliases[token]


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def timestamp_token() -> str:
    return datetime.now(KST).strftime("%Y%m%d_%H%M%S")


def timestamped_output_root(
    output_root: str | Path,
    *,
    run_timestamp: str | None,
    enabled: bool,
    prefix: str = SOURCE_PROGRAM,
) -> Path:
    base = Path(output_root).expanduser().resolve()
    if not enabled:
        base.mkdir(parents=True, exist_ok=True)
        return base
    stamp = str(run_timestamp).strip() if run_timestamp not in (None, "") else timestamp_token()
    resolved = base / f"{stamp}__{safe_token(prefix)}"
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_device(device_token: str, gpu_index: int, allow_cpu_fallback: bool) -> torch.device:
    token = str(device_token or "cuda").strip().lower()
    if token.startswith("cuda"):
        if not torch.cuda.is_available():
            if allow_cpu_fallback:
                print("[DI] WARNING: CUDA unavailable; falling back to CPU.", file=sys.stderr, flush=True)
                return torch.device("cpu")
            raise RuntimeError("CUDA is unavailable. Set allow_cpu_fallback=true or device=cpu to run on CPU.")
        index = int(gpu_index)
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f"gpu_index={index} is invalid; visible CUDA device count={torch.cuda.device_count()}.")
        torch.cuda.set_device(index)
        return torch.device(f"cuda:{index}")
    if token == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device token: {device_token!r}")


def resolve_torch_dtype(dtype_token: str, *, default: torch.dtype = torch.float64) -> torch.dtype:
    token = str(dtype_token or "").strip().lower()
    aliases = {
        "float64": torch.float64,
        "double": torch.float64,
        "fp64": torch.float64,
        "float32": torch.float32,
        "single": torch.float32,
        "fp32": torch.float32,
    }
    if token == "":
        return default
    if token not in aliases:
        raise ValueError("stats_dtype must be one of: float64, float32.")
    return aliases[token]


@dataclass(frozen=True)
class DIConfig:
    dataset: list[str]
    prep_root: Path
    output_root: Path
    split: str
    batch_size: int
    num_workers: int
    gpu_index: int
    device: str
    allow_cpu_fallback: bool
    seed: int
    max_samples: int | None
    view_name: str | None
    demean: bool
    epsilon: float
    psd_windows: tuple[str, ...]
    psd_row_reducer: str
    psd_value_transform: str
    timestamped_output: bool
    run_timestamp: str | None
    plot_dpi: int
    plot_log_y: bool
    save_plots: bool
    pin_memory: bool
    stats_dtype: str


def config_from_mapping(payload: Mapping[str, Any]) -> DIConfig:
    section: Mapping[str, Any] = payload
    for key in DEFAULT_STAGE_KEYS:
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            section = candidate
            break
    section = flatten_config_section(section)

    dataset_tokens = [canonicalize_dataset_name(str(v)) for v in as_list(config_value(section, "dataset"))]
    if not dataset_tokens:
        raise ValueError("Config field dataset must not be empty.")

    psd_windows = tuple(dict.fromkeys(normalize_signal_window(v) for v in as_list(config_value(section, "psd_windows", ["none", "hann"]))))
    if not psd_windows:
        raise ValueError("Config field psd_windows must not be empty.")
    reducer = str(config_value(section, "psd_row_reducer", "mean")).strip().lower()
    if reducer not in {"mean", "median"}:
        raise ValueError("psd_row_reducer must be mean or median.")
    transform = str(config_value(section, "psd_value_transform", "raw")).strip().lower()
    if transform not in {"raw", "db", "area"}:
        raise ValueError("psd_value_transform must be raw, db, or area.")

    return DIConfig(
        dataset=dataset_tokens,
        prep_root=Path(str(config_value(section, "prep_root", "data/prep_data"))).expanduser(),
        output_root=Path(str(config_value(section, "output_root", "result/DI"))).expanduser(),
        split=str(config_value(section, "split", "train")).strip().lower(),
        batch_size=int(config_value(section, "batch_size", 256)),
        num_workers=int(config_value(section, "num_workers", 0)),
        gpu_index=int(config_value(section, "gpu_index", 0)),
        device=str(config_value(section, "device", "cuda")),
        allow_cpu_fallback=str_to_bool(config_value(section, "allow_cpu_fallback", False)),
        seed=int(config_value(section, "seed", 0)),
        max_samples=maybe_int(config_value(section, "max_samples")),
        view_name=None if config_value(section, "view_name") in (None, "") else str(config_value(section, "view_name")),
        demean=str_to_bool(config_value(section, "demean", True)),
        epsilon=float(config_value(section, "epsilon", 1.0e-12)),
        psd_windows=psd_windows,
        psd_row_reducer=reducer,
        psd_value_transform=transform,
        timestamped_output=str_to_bool(config_value(section, "timestamped_output", True)),
        run_timestamp=None if config_value(section, "run_timestamp") in (None, "") else str(config_value(section, "run_timestamp")),
        plot_dpi=int(config_value(section, "plot_dpi", 180)),
        plot_log_y=str_to_bool(config_value(section, "plot_log_y", False)),
        save_plots=str_to_bool(config_value(section, "save_plots", True)),
        pin_memory=str_to_bool(config_value(section, "pin_memory", True)),
        stats_dtype=str(config_value(section, "stats_dtype", "float64")).strip().lower(),
    )


def dataclass_to_jsonable(config: DIConfig) -> dict[str, Any]:
    return {
        "dataset": list(config.dataset),
        "prep_root": str(config.prep_root),
        "output_root": str(config.output_root),
        "split": config.split,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "gpu_index": config.gpu_index,
        "device": config.device,
        "allow_cpu_fallback": config.allow_cpu_fallback,
        "seed": config.seed,
        "max_samples": config.max_samples,
        "view_name": config.view_name,
        "demean": config.demean,
        "epsilon": config.epsilon,
        "psd_windows": list(config.psd_windows),
        "psd_row_reducer": config.psd_row_reducer,
        "psd_value_transform": config.psd_value_transform,
        "timestamped_output": config.timestamped_output,
        "run_timestamp": config.run_timestamp,
        "plot_dpi": config.plot_dpi,
        "plot_log_y": config.plot_log_y,
        "save_plots": config.save_plots,
        "pin_memory": config.pin_memory,
        "stats_dtype": config.stats_dtype,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Integrated dataset-level frequency DI extractor.")
    parser.add_argument("--config", default="config/DI.yaml", help="Path to DI YAML config.")
    parser.add_argument("--dataset", default=None, help="Override dataset token/list. Comma separated is accepted.")
    parser.add_argument("--prep_root", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--split", default=None, choices=("train", "test", "all"))
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--gpu_index", type=int, default=None)
    parser.add_argument("--device", default=None, choices=("cuda", "cpu"))
    parser.add_argument("--allow_cpu_fallback", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--view_name", default=None)
    parser.add_argument("--demean", default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--psd_windows", nargs="*", default=None, choices=("none", "hann"))
    parser.add_argument("--psd_row_reducer", default=None, choices=("mean", "median"))
    parser.add_argument("--psd_value_transform", default=None, choices=("raw", "db", "area"))
    parser.add_argument("--timestamped_output", default=None)
    parser.add_argument("--run_timestamp", default=None)
    parser.add_argument("--plot_dpi", type=int, default=None)
    parser.add_argument("--plot_log_y", default=None)
    parser.add_argument("--save_plots", default=None)
    parser.add_argument("--pin_memory", default=None)
    parser.add_argument("--stats_dtype", default=None, choices=("float64", "float32"))
    return parser


def merge_cli_overrides(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    merged = dict(payload)
    section_key = None
    for key in DEFAULT_STAGE_KEYS:
        if isinstance(merged.get(key), Mapping):
            section_key = key
            break
    section_raw = dict(merged.get(section_key, merged)) if section_key else dict(merged)
    section = flatten_config_section(section_raw)
    for key in (
        "dataset",
        "prep_root",
        "output_root",
        "split",
        "batch_size",
        "num_workers",
        "gpu_index",
        "device",
        "allow_cpu_fallback",
        "seed",
        "max_samples",
        "view_name",
        "demean",
        "epsilon",
        "psd_windows",
        "psd_row_reducer",
        "psd_value_transform",
        "timestamped_output",
        "run_timestamp",
        "plot_dpi",
        "plot_log_y",
        "save_plots",
        "pin_memory",
        "stats_dtype",
    ):
        value = getattr(args, key, None)
        if value is not None:
            section[key] = value
    if section_key:
        merged[section_key] = section
    else:
        merged = section
    return merged


def resolve_split_file(dataset_root: Path, manifest: Mapping[str, Any], split: str, view_name: str) -> Path:
    split = str(split)
    view_name = str(view_name)
    files_by_view = manifest.get("files_by_view")
    if isinstance(files_by_view, Mapping):
        view_entry = files_by_view.get(view_name)
        if isinstance(view_entry, Mapping) and isinstance(view_entry.get(split), str):
            return (dataset_root / view_entry[split]).expanduser().resolve()
    files_entry = manifest.get("files", {"train": "train.npy", "test": "test.npy"})
    if not isinstance(files_entry, Mapping):
        raise ValueError("manifest.files must be an object.")
    rel = files_entry.get(split, f"{split}.npy")
    if not isinstance(rel, str):
        raise ValueError(f"manifest.files.{split} must be a string path.")
    return (dataset_root / rel).expanduser().resolve()


def validate_records(records: np.ndarray, path: Path) -> None:
    if not isinstance(records, np.ndarray):
        raise TypeError(f"Structured split payload must be a numpy array: {path}")
    if records.ndim != 1:
        raise ValueError(f"Structured split payload must have shape (N,), got {records.shape}: {path}")
    fields = records.dtype.fields
    if fields is None:
        raise ValueError(f"Structured split payload must use a structured dtype: {path}")
    missing = {"sample_index", "label", "input"}.difference(fields.keys())
    if missing:
        raise ValueError(f"Structured split payload missing fields {sorted(missing)}: {path}")


def load_records(path: Path, max_samples: int | None) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Prepared split file not found: {path}")
    records = np.load(path, mmap_mode="r", allow_pickle=False)
    validate_records(records, path)
    if max_samples is not None:
        return records[: int(max_samples)]
    return records


def validate_manifest(manifest: Mapping[str, Any], *, dataset: str, manifest_path: Path) -> None:
    canonical = canonicalize_dataset_name(dataset)
    manifest_dataset = manifest.get("dataset_name", canonical)
    if canonicalize_dataset_name(str(manifest_dataset)) != canonical:
        raise ValueError(
            f"Manifest dataset_name mismatch: expected {canonical!r}, got {manifest_dataset!r}. "
            f"Manifest path: {manifest_path}"
        )
    storage_format = str(manifest.get("storage_format", ""))
    if storage_format != STORAGE_FORMAT:
        raise ValueError(f"Unsupported storage_format={storage_format!r}; expected {STORAGE_FORMAT!r}. Re-run data_prep.")
    if manifest.get("split_internal_order_preserved") is not True:
        raise ValueError("Prepared manifest must declare split_internal_order_preserved=true.")
    missing_axis = [
        key
        for key in ("psd_time_axis", "psd_row_axes", "psd_flatten_rule", "psd_logical_shape")
        if manifest.get(key) in (None, "", [])
    ]
    if missing_axis:
        raise ValueError(f"Prepared manifest is missing PSD axis metadata {missing_axis}. Re-run current data_prep.")


def resolve_manifest(prep_root: Path, dataset: str) -> tuple[Path, dict[str, Any]]:
    dataset_root = prep_root.expanduser().resolve() / canonicalize_dataset_name(dataset)
    manifest_path = resolve_manifest_path(dataset_root)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Prepared manifest missing: {manifest_path}")
    manifest = load_manifest(manifest_path)
    validate_manifest(manifest, dataset=dataset, manifest_path=manifest_path)
    return dataset_root, manifest


class PreparedSplitViewDataset(Dataset[tuple[np.ndarray, int]]):
    """Self-contained view reader for one prepared split."""

    def __init__(self, *, dataset_name: str, records: np.ndarray, manifest: Mapping[str, Any], view_name: str) -> None:
        self.dataset_name = canonicalize_dataset_name(dataset_name)
        self.records = records
        self.manifest = dict(manifest)
        self.view_name = str(view_name)
        self.sequence_input_rule = self.manifest.get("sequence_input_rule")
        self.labels_np = np.asarray(self.records["label"], dtype=np.int64).reshape(-1).copy()

    def __len__(self) -> int:
        return int(self.records.shape[0])

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        idx = int(index)
        x = self._reconstruct_view(self.records["input"][idx])
        y = int(self.records["label"][idx])
        return x, y

    def _reconstruct_view(self, stored_input: np.ndarray) -> np.ndarray:
        """Mirror the project view rules needed for PSD/DI without importing project code."""

        dataset = self.dataset_name
        view = self.view_name
        stored = np.asarray(stored_input, dtype=np.float32)

        if dataset in {"s-mnist", "ps-mnist", "s-cifar10", "deap"}:
            if view == "model_input":
                return np.ascontiguousarray(stored, dtype=np.float32)
            if view in {"psd_input", "model_input_psd_view"}:
                return np.ascontiguousarray(stored.T, dtype=np.float32)
            # If the selected file is already a view file, fall through to generic maps.
            return np.ascontiguousarray(stored, dtype=np.float32)

        if dataset in {"uci-har", "shd", "ssc"}:
            if view == "model_input":
                return np.ascontiguousarray(stored, dtype=np.float32)
            if view in {"psd_input", "model_input_psd_view", "sequence_input"}:
                return np.ascontiguousarray(stored.T, dtype=np.float32)
            return np.ascontiguousarray(stored, dtype=np.float32)

        if dataset in {"mnist", "cifar-10", "cifar-100"}:
            # Static image datasets may store either repeated CNN frames (T,C,H,W)
            # or a flattened repeated sequence (T,F), depending on the selected view.
            if stored.ndim == 4:
                repeated = stored
                first_frame = repeated[0]
                flatten_channel_major = first_frame.reshape(int(first_frame.shape[0]), -1)
                sequence_flatten = repeated.reshape(int(repeated.shape[0]), -1)
                if view == "original_input":
                    return np.ascontiguousarray(first_frame, dtype=np.float32)
                if view in {"model_input", "model_input_cnn", "cnn_input", "psd_input", "image_psd_view"}:
                    return np.ascontiguousarray(repeated, dtype=np.float32)
                if view in {"sequence_input", "model_input_flatten"}:
                    return np.ascontiguousarray(sequence_flatten, dtype=np.float32)
                if view == "flatten_input":
                    return np.ascontiguousarray(flatten_channel_major, dtype=np.float32)
            return np.ascontiguousarray(stored, dtype=np.float32)

        if dataset in {"n-mnist", "cifar10-dvs", "dvs128-gesture"}:
            original = stored
            if original.ndim >= 2:
                flatten = original.reshape(int(original.shape[0]), -1)
            else:
                flatten = original.reshape(1, -1)
            if view in {"model_input", "original_input"}:
                return np.ascontiguousarray(original, dtype=np.float32)
            if view in {"flatten_input", "sequence_input", "model_input_flatten"}:
                return np.ascontiguousarray(flatten, dtype=np.float32)
            if view in {"psd_input", "event_frame_psd_view"}:
                return np.ascontiguousarray(flatten.T, dtype=np.float32)
            return np.ascontiguousarray(stored, dtype=np.float32)

        if str(self.sequence_input_rule) == "model_input_transpose":
            if view in {"model_input", "psd_input"}:
                return np.ascontiguousarray(stored, dtype=np.float32)
            if view == "sequence_input":
                return np.ascontiguousarray(stored.T, dtype=np.float32)

        return np.ascontiguousarray(stored, dtype=np.float32)


def collect_class_values(datasets: Sequence[PreparedSplitViewDataset]) -> np.ndarray:
    labels = []
    for dataset in datasets:
        labels.append(dataset.labels_np)
    if not labels:
        raise ValueError("No split dataset was provided.")
    merged = np.concatenate(labels, axis=0)
    if merged.size < 1:
        raise ValueError("Selected split is empty.")
    return np.unique(merged.astype(np.int64))


def expected_rows_time_from_manifest(manifest: Mapping[str, Any]) -> tuple[int | None, int | None]:
    logical = manifest.get("psd_logical_shape")
    if isinstance(logical, (list, tuple)) and len(logical) == 2:
        return int(logical[0]), int(logical[1])
    return None, None


def batch_to_maps(
    batch: torch.Tensor,
    *,
    manifest: Mapping[str, Any],
    expected_rows: int | None,
    expected_time: int | None,
) -> torch.Tensor:
    """Convert a collated batch to official (B, rows, time) maps."""

    x = torch.as_tensor(batch, dtype=torch.float32)
    axis_kind = str(manifest.get("psd_axis_kind", "")).strip().lower()
    if x.ndim == 2:
        # (B,T) scalar temporal signal.
        maps = x.unsqueeze(1)
    elif x.ndim == 3:
        # Common PSD view: (B,rows,time). A model-input view may be (B,time,rows).
        shape_rt = (int(x.shape[1]), int(x.shape[2]))
        if expected_rows is not None and expected_time is not None:
            if shape_rt == (int(expected_rows), int(expected_time)):
                maps = x.contiguous()
            elif shape_rt == (int(expected_time), int(expected_rows)):
                maps = x.transpose(1, 2).contiguous()
            else:
                time_axis = str(manifest.get("psd_time_axis", "")).strip().lower()
                flatten_rule = str(manifest.get("psd_flatten_rule", "")).strip().lower()
                if time_axis in {"last", "-1", "2", "time_last", "rows_time", "row_time", "nrt", "brows_time"} or flatten_rule in {
                    "already_flattened_rows_time",
                    "rows_time",
                    "channel_major_time_last",
                }:
                    maps = x.contiguous()
                elif time_axis in {"1", "t", "time", "time_axis_1", "model_time", "sequence"} or axis_kind == "temporal":
                    maps = x.transpose(1, 2).contiguous()
                else:
                    raise ValueError(
                        "Ambiguous rank-3 batch. Set config.view_name to the manifest psd_view_name or check psd axis metadata. "
                        f"shape={tuple(x.shape)}, expected_rows={expected_rows}, expected_time={expected_time}."
                    )
        else:
            time_axis = str(manifest.get("psd_time_axis", "")).strip().lower()
            flatten_rule = str(manifest.get("psd_flatten_rule", "")).strip().lower()
            if time_axis in {"last", "-1", "2", "time_last", "rows_time", "row_time", "nrt", "brows_time"} or flatten_rule in {
                "already_flattened_rows_time",
                "rows_time",
                "channel_major_time_last",
            }:
                maps = x.contiguous()
            elif time_axis in {"1", "t", "time", "time_axis_1", "model_time", "sequence"} or axis_kind == "temporal":
                maps = x.transpose(1, 2).contiguous()
            else:
                # Conservative default for explicitly selected PSD views.
                maps = x.contiguous()
    elif x.ndim == 4:
        # (B,C,H,W) has no explicit temporal axis. Keep a singleton time axis.
        b = int(x.shape[0])
        maps = x.reshape(b, -1, 1).contiguous()
    elif x.ndim == 5:
        # (B,T,C,H,W) -> (B,C*H*W,T)
        b, t, c, h, w = [int(v) for v in x.shape]
        maps = x.permute(0, 2, 3, 4, 1).reshape(b, c * h * w, t).contiguous()
    else:
        raise ValueError(f"Unsupported input batch rank {x.ndim}; shape={tuple(x.shape)}.")

    if maps.ndim != 3:
        raise ValueError(f"DI maps must be rank-3 (B,rows,time), got {tuple(maps.shape)}")
    if expected_time is not None and int(maps.shape[-1]) != int(expected_time):
        raise ValueError(f"PSD time mismatch: expected {expected_time}, got {int(maps.shape[-1])}.")
    if expected_rows is not None and int(maps.shape[1]) != int(expected_rows):
        raise ValueError(f"PSD row mismatch: expected {expected_rows}, got {int(maps.shape[1])}.")
    return maps


def one_sided_freqs(length: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.arange(int(length) // 2 + 1, device=device, dtype=dtype) / float(length)


def one_sided_scaling(length: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    freq_count = int(length) // 2 + 1
    scale = torch.ones(freq_count, device=device, dtype=dtype)
    if int(length) % 2 == 0:
        if freq_count > 2:
            scale[1:-1] = 2.0
    else:
        if freq_count > 1:
            scale[1:] = 2.0
    return scale


def temporal_window(length: int, *, signal_window: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    token = normalize_signal_window(signal_window)
    if token == "none":
        return torch.ones(int(length), device=device, dtype=dtype)
    if int(length) <= 1:
        return torch.ones(int(length), device=device, dtype=dtype)
    t = torch.arange(int(length), device=device, dtype=dtype)
    return 0.5 - 0.5 * torch.cos(2.0 * torch.pi * t / float(int(length) - 1))


def exact_project_periodogram(maps: torch.Tensor, *, signal_window: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Project-compatible one-sided exact periodogram.

    Formula matches the project's ``exact_periodogram_from_maps``:
    scale * |rfft(window*x)|^2 / (L * mean(window^2)).
    """

    if maps.ndim != 3:
        raise ValueError(f"Expected maps shape (B,rows,time), got {tuple(maps.shape)}")
    length = int(maps.shape[-1])
    dtype = maps.dtype
    device = maps.device
    window = temporal_window(length, signal_window=signal_window, device=device, dtype=dtype)
    window_power = window.square().sum() / float(length)
    spectrum = torch.fft.rfft(maps * window.view(1, 1, -1), dim=-1)
    scale = one_sided_scaling(length, device=device, dtype=dtype)
    power = scale.view(1, 1, -1) * spectrum.abs().square() / (float(length) * window_power)
    return one_sided_freqs(length, device=device, dtype=dtype), power.real


def reduce_rows(values: torch.Tensor, reducer: str) -> torch.Tensor:
    token = str(reducer).strip().lower()
    if token == "mean":
        return values.mean(dim=1)
    if token == "median":
        return values.median(dim=1).values
    raise ValueError(f"Unsupported row reducer: {reducer!r}")


def power_to_db(values: torch.Tensor) -> torch.Tensor:
    return 10.0 * torch.log10(torch.clamp(values, min=0.0) + DB_EPSILON)


def area_normalize_power(values: torch.Tensor, *, epsilon: float = DB_EPSILON) -> torch.Tensor:
    denom = torch.clamp(values.sum(dim=-1, keepdim=True), min=float(epsilon))
    return values / denom


class DIAccumulator:
    """GPU-resident sufficient statistics for per-frequency Fisher DI."""

    def __init__(self, *, class_values: np.ndarray, num_freqs: int, device: torch.device, dtype: torch.dtype) -> None:
        if len(class_values) < 2:
            raise ValueError("DI requires at least two classes.")
        self.class_values_np = np.asarray(class_values, dtype=np.int64)
        self.label_to_pos = {int(v): i for i, v in enumerate(self.class_values_np.tolist())}
        self.class_values = torch.as_tensor(self.class_values_np, device=device, dtype=torch.long)
        self.counts = torch.zeros((len(class_values),), device=device, dtype=dtype)
        self.sums = torch.zeros((len(class_values), int(num_freqs)), device=device, dtype=dtype)
        self.sumsq = torch.zeros((len(class_values), int(num_freqs)), device=device, dtype=dtype)
        self.total = 0

    def _label_positions(self, labels: torch.Tensor, *, device: torch.device) -> torch.Tensor:
        labels_device = torch.as_tensor(labels, device=device, dtype=torch.long).reshape(-1)
        positions = torch.searchsorted(self.class_values, labels_device)
        in_range = positions < int(self.class_values.numel())
        matched = torch.zeros_like(in_range, dtype=torch.bool)
        if bool(in_range.any().item()):
            matched[in_range] = self.class_values[positions[in_range]] == labels_device[in_range]
        if not bool(matched.all().item()):
            bad = labels_device[~matched].detach().cpu().reshape(-1).tolist()
            raise KeyError(f"Encountered label(s) {bad[:8]!r} not present in selected split class_values.")
        return positions

    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        if features.ndim != 2:
            raise ValueError(f"DI features must be shape (B,F), got {tuple(features.shape)}")
        if int(features.shape[0]) != int(labels.shape[0]):
            raise ValueError(f"Feature/label batch mismatch: {features.shape[0]} vs {labels.shape[0]}")
        if not torch.isfinite(features).all():
            raise ValueError("Non-finite spectral features encountered.")
        positions = self._label_positions(labels, device=features.device)
        dtype = self.sums.dtype
        self.counts.index_add_(0, positions, torch.ones_like(positions, dtype=dtype))
        self.sums.index_add_(0, positions, features.to(dtype=dtype))
        self.sumsq.index_add_(0, positions, features.to(dtype=dtype).square())
        self.total += int(features.shape[0])

    def finalize(self, *, epsilon: float) -> dict[str, np.ndarray | int | float | list[int]]:
        if self.total < 1:
            raise ValueError("Cannot finalize an empty DI accumulator.")
        if bool((self.counts < 2).any().item()):
            counts = [int(v) for v in self.counts.detach().cpu().tolist()]
            raise ValueError(f"Every class needs at least two samples for unbiased variance; counts={counts}")
        counts_col = self.counts.view(-1, 1)
        means = self.sums / counts_col
        raw_var = (self.sumsq - self.sums.square() / counts_col) / torch.clamp(counts_col - 1.0, min=1.0)
        variances = torch.clamp(raw_var, min=0.0)
        priors = self.counts / float(self.total)
        global_mean = (priors.view(-1, 1) * means).sum(dim=0)
        sb = (priors.view(-1, 1) * (means - global_mean.view(1, -1)).square()).sum(dim=0)
        sw = (priors.view(-1, 1) * variances).sum(dim=0)
        di = sb / (sw + float(epsilon))
        di_sum = di.sum()
        if float(di_sum.detach().cpu().item()) > 0.0:
            di_norm = di / di_sum
        else:
            di_norm = torch.zeros_like(di)
        return {
            "count_total": int(self.total),
            "class_values": [int(v) for v in self.class_values_np.tolist()],
            "class_counts": np.asarray(self.counts.detach().cpu().numpy(), dtype=np.float64),
            "between_scatter": np.asarray(sb.detach().cpu().numpy(), dtype=np.float64),
            "within_scatter": np.asarray(sw.detach().cpu().numpy(), dtype=np.float64),
            "di": np.asarray(di.detach().cpu().numpy(), dtype=np.float64),
            "di_norm": np.asarray(di_norm.detach().cpu().numpy(), dtype=np.float64),
            "di_sum": float(di_sum.detach().cpu().item()),
        }


def output_specs(psd_windows: Sequence[str], value_transform: str = "raw") -> list[dict[str, str]]:
    specs = [
        {
            "key": "dft_magnitude",
            "feature_kind": "dft_magnitude",
            "signal_window": "none",
            "description": "DI from one-sided DFT coefficient magnitude of row-mean scalar sequence.",
        }
    ]
    for window in psd_windows:
        specs.append(
            {
                "key": f"project_psd_{normalize_signal_window(window)}",
                "feature_kind": "project_psd",
                "signal_window": normalize_signal_window(window),
                "value_transform": str(value_transform),
            "description": "DI from project-compatible exact one-sided periodogram after row reduction and configured value transform.",
            }
        )
    return specs


def update_accumulators_for_batch(
    accumulators: dict[str, DIAccumulator],
    *,
    maps: torch.Tensor,
    labels: torch.Tensor,
    class_values: np.ndarray,
    config: DIConfig,
    device: torch.device,
    dtype: torch.dtype,
    stats_dtype: torch.dtype,
) -> torch.Tensor:
    if config.demean:
        maps_for_features = maps - maps.mean(dim=-1, keepdim=True)
    else:
        maps_for_features = maps

    length = int(maps_for_features.shape[-1])
    freqs = one_sided_freqs(length, device=device, dtype=dtype)
    labels_device = torch.as_tensor(labels, device=device, dtype=torch.long).reshape(-1)

    # 1/2. DFT coefficient magnitude DI and normalized DI.
    scalar_signal = maps_for_features.mean(dim=1)
    dft_feature = torch.fft.rfft(scalar_signal, dim=-1).abs().real
    if "dft_magnitude" not in accumulators:
        accumulators["dft_magnitude"] = DIAccumulator(
            class_values=class_values,
            num_freqs=int(dft_feature.shape[-1]),
            device=device,
            dtype=stats_dtype,
        )
    accumulators["dft_magnitude"].update(dft_feature, labels_device)

    # 3/4. Project PSD DI and normalized DI, with/without Hann window.
    for window in config.psd_windows:
        key = f"project_psd_{normalize_signal_window(window)}"
        _freqs, power = exact_project_periodogram(maps_for_features, signal_window=window)
        feature = reduce_rows(power, config.psd_row_reducer)
        if config.psd_value_transform == "db":
            feature = power_to_db(feature)
        elif config.psd_value_transform == "area":
            feature = area_normalize_power(feature, epsilon=float(config.epsilon))
        if key not in accumulators:
            accumulators[key] = DIAccumulator(
                class_values=class_values,
                num_freqs=int(feature.shape[-1]),
                device=device,
                dtype=stats_dtype,
            )
        accumulators[key].update(feature, labels_device)
    return freqs


def write_di_csv(
    path: Path,
    *,
    dataset: str,
    split: str,
    spec: Mapping[str, Any],
    normalized: bool,
    result: Mapping[str, Any],
    freqs: np.ndarray,
    config: DIConfig,
    manifest: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    di_raw = np.asarray(result["di"], dtype=np.float64)
    di_norm = np.asarray(result["di_norm"], dtype=np.float64)
    sb = np.asarray(result["between_scatter"], dtype=np.float64)
    sw = np.asarray(result["within_scatter"], dtype=np.float64)
    values = di_norm if normalized else di_raw
    value_name = "di_norm" if normalized else "di"
    class_values = result.get("class_values", [])
    class_counts = np.asarray(result.get("class_counts", []), dtype=np.float64).reshape(-1)
    class_counts_text = compact_yaml({str(int(c)): int(class_counts[i]) for i, c in enumerate(class_values)})
    fieldnames = [
        "source_program",
        "dataset",
        "split",
        "feature_kind",
        "signal_window",
        "normalized",
        "value_name",
        "frequency_index",
        "normalized_frequency",
        "angular_frequency_rad",
        "value",
        "di_raw",
        "di_norm",
        "between_scatter",
        "within_scatter",
        "epsilon",
        "count_total",
        "num_classes",
        "class_counts",
        "demean",
        "psd_row_reducer",
        "psd_value_transform",
        "view_name",
        "psd_axis_kind",
        "psd_time_axis",
        "psd_flatten_rule",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, value in enumerate(values.reshape(-1)):
            freq = float(freqs[idx]) if idx < len(freqs) else float("nan")
            writer.writerow(
                {
                    "source_program": SOURCE_PROGRAM,
                    "dataset": dataset,
                    "split": split,
                    "feature_kind": spec["feature_kind"],
                    "signal_window": spec["signal_window"],
                    "normalized": str(bool(normalized)).lower(),
                    "value_name": value_name,
                    "frequency_index": idx,
                    "normalized_frequency": f"{freq:.12g}",
                    "angular_frequency_rad": f"{2.0 * math.pi * freq:.12g}",
                    "value": f"{float(value):.12g}",
                    "di_raw": f"{float(di_raw[idx]):.12g}",
                    "di_norm": f"{float(di_norm[idx]):.12g}",
                    "between_scatter": f"{float(sb[idx]):.12g}",
                    "within_scatter": f"{float(sw[idx]):.12g}",
                    "epsilon": f"{float(config.epsilon):.12g}",
                    "count_total": int(result["count_total"]),
                    "num_classes": len(class_values),
                    "class_counts": class_counts_text,
                    "demean": str(bool(config.demean)).lower(),
                    "psd_row_reducer": config.psd_row_reducer,
                    "psd_value_transform": config.psd_value_transform,
                    "view_name": config.view_name or str(manifest.get("psd_view_name", "psd_input")),
                    "psd_axis_kind": str(manifest.get("psd_axis_kind", "")),
                    "psd_time_axis": str(manifest.get("psd_time_axis", "")),
                    "psd_flatten_rule": str(manifest.get("psd_flatten_rule", "")),
                }
            )


def plot_di_csv(csv_path: Path, png_path: Path, *, title: str, ylabel: str, dpi: int, log_y: bool) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not available; cannot save plots.")
    xs: list[float] = []
    ys: list[float] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            xs.append(float(row["normalized_frequency"]))
            ys.append(float(row["value"]))
    fig = plt.figure(figsize=(9.0, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xs, ys, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Normalized frequency")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)
    if log_y:
        positive = [v for v in ys if v > 0.0]
        if positive:
            ax.set_yscale("log")
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=int(dpi))
    plt.close(fig)


def analyze_one_dataset(config: DIConfig, dataset_token: str, base_output_root: Path, device: torch.device) -> dict[str, Any]:
    dataset_token = canonicalize_dataset_name(dataset_token)
    dataset_root, manifest = resolve_manifest(config.prep_root, dataset_token)
    view_name = config.view_name or str(manifest.get("psd_view_name", "psd_input"))
    splits = ("train", "test") if config.split == "all" else (config.split,)
    if any(split not in {"train", "test"} for split in splits):
        raise ValueError("split must be train, test, or all.")

    split_datasets: list[PreparedSplitViewDataset] = []
    split_paths: dict[str, str] = {}
    for split in splits:
        path = resolve_split_file(dataset_root, manifest, split, view_name)
        records = load_records(path, config.max_samples)
        split_paths[split] = str(path)
        split_datasets.append(
            PreparedSplitViewDataset(dataset_name=dataset_token, records=records, manifest=manifest, view_name=view_name)
        )
    class_values = collect_class_values(split_datasets)
    runtime_dataset: Dataset[Any]
    if len(split_datasets) == 1:
        runtime_dataset = split_datasets[0]
    else:
        runtime_dataset = ConcatDataset(split_datasets)

    output_root = base_output_root / dataset_token if len(config.dataset) > 1 else base_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    write_metadata_csv(
        output_root / "DI_resolved_config.csv",
        {
            "config": dataclass_to_jsonable(config),
            "dataset": dataset_token,
            "view_name": view_name,
            "split_paths": split_paths,
            "manifest_path": str(resolve_manifest_path(dataset_root)),
            "class_values": [int(v) for v in class_values.tolist()],
        },
    )

    pin_memory = bool(config.pin_memory and device.type == "cuda")
    loader = DataLoader(
        runtime_dataset,
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        pin_memory=pin_memory,
        drop_last=False,
    )
    expected_rows, expected_time = expected_rows_time_from_manifest(manifest)
    dtype = torch.float32
    stats_dtype = resolve_torch_dtype(config.stats_dtype, default=torch.float64)
    accumulators: dict[str, DIAccumulator] = {}
    freq_ref: torch.Tensor | None = None
    batch_count = 0
    sample_count = 0

    start = time.time()
    print(
        f"[DI] dataset={dataset_token} split={config.split} view={view_name} device={device} "
        f"batch_size={config.batch_size}",
        flush=True,
    )
    with torch.no_grad():
        for inputs, labels in loader:
            batch_count += 1
            sample_count += int(torch.as_tensor(labels).numel())
            maps = batch_to_maps(
                inputs,
                manifest=manifest,
                expected_rows=expected_rows,
                expected_time=expected_time,
            ).to(device=device, dtype=dtype, non_blocking=True)
            freqs = update_accumulators_for_batch(
                accumulators,
                maps=maps,
                labels=torch.as_tensor(labels),
                class_values=class_values,
                config=config,
                device=device,
                dtype=dtype,
                stats_dtype=stats_dtype,
            )
            if freq_ref is None:
                freq_ref = freqs.detach().clone()
            elif int(freq_ref.numel()) != int(freqs.numel()):
                raise ValueError("Frequency axis length changed across batches.")
            if batch_count % 25 == 0:
                print(f"[DI] processed batches={batch_count} samples={sample_count}", flush=True)

    if freq_ref is None:
        raise ValueError("No batch was processed.")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.time() - start

    freqs_np = np.asarray(freq_ref.detach().cpu().numpy(), dtype=np.float64)
    manifest_rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    for spec in output_specs(config.psd_windows, config.psd_value_transform):
        key = spec["key"]
        if key not in accumulators:
            raise KeyError(f"Missing accumulator for output key {key!r}")
        result = accumulators[key].finalize(epsilon=float(config.epsilon))
        results[key] = {
            "count_total": result["count_total"],
            "di_sum": result["di_sum"],
            "class_values": result["class_values"],
        }
        for normalized in (False, True):
            suffix = "norm" if normalized else "raw"
            if spec["feature_kind"] == "dft_magnitude":
                stem = f"DI__{dataset_token}__{config.split}__dft_magnitude__{suffix}"
            else:
                stem = f"DI__{dataset_token}__{config.split}__project_psd__{spec['signal_window']}__{config.psd_value_transform}__{suffix}"
            csv_path = output_root / "csv" / f"{stem}.csv"
            png_path = output_root / "plot" / f"{stem}.png"
            write_di_csv(
                csv_path,
                dataset=dataset_token,
                split=config.split,
                spec=spec,
                normalized=normalized,
                result=result,
                freqs=freqs_np,
                config=config,
                manifest=manifest,
            )
            if config.save_plots:
                ylabel = "DI_norm" if normalized else "DI"
                plot_title = f"{dataset_token} {config.split} {spec['feature_kind']} window={spec['signal_window']} {ylabel}"
                plot_di_csv(
                    csv_path,
                    png_path,
                    title=plot_title,
                    ylabel=ylabel,
                    dpi=int(config.plot_dpi),
                    log_y=bool(config.plot_log_y and not normalized),
                )
            manifest_rows.append(
                {
                    "dataset": dataset_token,
                    "split": config.split,
                    "feature_kind": spec["feature_kind"],
                    "signal_window": spec["signal_window"],
                    "normalized": str(bool(normalized)).lower(),
                    "value_transform": str(spec.get("value_transform", "")),
                    "csv_path": str(csv_path),
                    "plot_path": str(png_path) if config.save_plots else "",
                }
            )

    manifest_path = output_root / "DI_manifest.yaml"
    save_yaml(
        manifest_path,
        {
            "schema_version": "di_manifest_v1",
            "rows": manifest_rows,
        },
    )

    write_metadata_csv(
        output_root / "DI_summary.csv",
        {
            "status": "ok",
            "dataset": dataset_token,
            "split": config.split,
            "view_name": view_name,
            "device": str(device),
            "batches": batch_count,
            "samples": sample_count,
            "elapsed_seconds": elapsed,
            "outputs": manifest_rows,
            "results": results,
        },
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "dataset": dataset_token,
                "samples": sample_count,
                "elapsed_seconds": round(elapsed, 3),
                "output_root": str(output_root),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return {"dataset": dataset_token, "output_root": str(output_root), "manifest": str(manifest_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    raw_config = load_mapping(args.config)
    merged_config = merge_cli_overrides(raw_config, args)
    config = config_from_mapping(merged_config)

    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if config.num_workers < 0:
        raise ValueError("num_workers must be >= 0.")
    if config.split not in {"train", "test", "all"}:
        raise ValueError("split must be train, test, or all.")
    if config.epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    _ = resolve_torch_dtype(config.stats_dtype, default=torch.float64)

    seed_everything(config.seed)
    device = resolve_device(config.device, config.gpu_index, config.allow_cpu_fallback)
    base_output_root = timestamped_output_root(
        config.output_root,
        run_timestamp=config.run_timestamp,
        enabled=config.timestamped_output,
        prefix=SOURCE_PROGRAM,
    )
    write_metadata_csv(base_output_root / "DI_run_config.csv", dataclass_to_jsonable(config))

    outputs = []
    for dataset_token in config.dataset:
        outputs.append(analyze_one_dataset(config, dataset_token, base_output_root, device))
    print(json.dumps({"status": "ok", "source_program": SOURCE_PROGRAM, "outputs": outputs}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
