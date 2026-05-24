#!/usr/bin/env python3
"""
psd_analysis batch CSV recursive plotter.

This script is intentionally standalone. It does not import the project's src package.
It reads a batch directory such as batch_0001, recursively finds CSV artifacts,
and renders plots for the categories requested below.

Rendered categories
- analysis_curve: line plot, x = normalized frequency, y = power
- analysis_dispersion: line plot, x = normalized frequency, y = PSD dispersion
- filter_snapshot: bar plot over filter statistics
- layer_distance_profile/layer_distance_trend: PSD curve-shape distance plots
- layer_dispersion_profile/layer_dispersion_trend: layer dispersion scalar plots

Skipped categories
- pair_distance
- traces/accuracy_loss_join and accuracy_loss_join
- manifests and unsupported categories

Default output
- If --output is not given and --input is a directory: <input>/plots
- If --output is not given and --input is a CSV file: <csv-parent>/plots

External output
- If --output is given, plots are written there.
- For directory input, the directory tree under --input is mirrored under --output.
  Example:
    input : batch_0001/checkpoint_epoch_000001/analysis_curve/a.csv
    output: plots_out/checkpoint_epoch_000001/analysis_curve/a.png

Example
python plot_psd_batch_recursive.py \
  --input /home/yongokhan/바탕화면/psd_outputs/psd_analysis/run1/batch_0001 \
  --output /home/yongokhan/바탕화면/psd_outputs/plots/run1/batch_0001 \
  --overwrite
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == '':
    _SCRIPT_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _SCRIPT_DIR.parent
    try:
        sys.path.remove(str(_SCRIPT_DIR))
    except ValueError:
        pass
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import csv
import math
import re
import sys
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from src.util.config_cli import parse_args_with_config


FIGSIZE = (14, 4)          # 3.5:1, matching the requested/existing style
LINEWIDTH = 3.5
TITLE_SIZE = 26
LABEL_SIZE = 21
TICK_SIZE = 19
DPI = 300

PSD_CATEGORIES = {
    "analysis_curve",
    "analysis_dispersion",
    # These are not normally inside psd_analysis/checkpoint output, but keeping them
    # here makes the script safe for dataset PSD CSVs with the same schema.
    "dataset_curve",
    "dataset_dispersion",
}
RENDERED_CATEGORIES = PSD_CATEGORIES | {"filter_snapshot", "filter_trend", "layer_distance_profile", "layer_distance_trend", "layer_dispersion_profile", "layer_dispersion_trend"}
SKIPPED_CATEGORIES = {
    "pair_distance",
    "accuracy_loss_join",
    "analysis_manifest",
    "dataset_psd_manifest",
    "plotting_manifest",
    "training_metric",
    "drift_distance",
    "pairwise_dependency_appendix",
}
KNOWN_CATEGORIES = RENDERED_CATEGORIES | SKIPPED_CATEGORIES

COMMON_PREFIX_COLUMNS = {
    "schema_version",
    "category",
    "source_program",
    "status",
    "message",
    "dataset",
    "run_id",
    "created_at",
}


@dataclass(frozen=True)
class CsvArtifact:
    path: Path
    category: str
    rows: list[dict[str, str]]


@dataclass(frozen=True)
class ManifestRow:
    input_csv_path: str
    output_figure_path: str
    category: str
    status: str
    message: str
    render_seconds: str = ""


@dataclass(frozen=True)
class DriftPoint:
    metric: str
    epoch_a: str
    epoch_b: str
    x: float | None
    y: float
    transition_label: str


@dataclass(frozen=True)
class DriftGroup:
    key: tuple[str, ...]
    rows: list[dict[str, str]]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recursively render plots from psd_analysis batch CSV output."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="A batch directory such as batch_0001, or a single CSV file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output directory. When --input is a directory, the relative "
            "directory tree under --input is mirrored here. Default: <input>/plots "
            "for directories, or <csv-parent>/plots for files."
        ),
    )
    parser.add_argument(
        "--output_root",
        default=None,
        help="Deprecated alias of --output. Kept for compatibility.",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=("png",),
        help="Figure format. Currently only png is supported.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing figure files.",
    )
    parser.add_argument('--config', default=None, help='JSON 설정 파일 경로(.json)')
    parser.add_argument(
        "--manifest_name",
        default="recursive_plot_manifest.csv",
        help="Manifest CSV filename to write under output_root.",
    )
    parser.add_argument(
        "--include_filter_count",
        action="store_true",
        help="Include the count statistic in filter_snapshot bar plots. Default excludes count because its unit differs from parameter values.",
    )
    return parser


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _infer_category_from_path(path: Path) -> str:
    stem = path.stem
    prefix = stem.split("__", 1)[0]
    if prefix in KNOWN_CATEGORIES:
        return prefix
    for category in sorted(KNOWN_CATEGORIES, key=len, reverse=True):
        if stem.startswith(category):
            return category
    return ""


def _infer_category(path: Path, rows: Sequence[dict[str, str]]) -> str:
    categories = sorted(
        {
            str(row.get("category", "")).strip()
            for row in rows
            if str(row.get("category", "")).strip()
        }
    )
    if len(categories) == 1:
        return categories[0]
    if len(categories) > 1:
        raise ValueError(f"one CSV file must contain one category, got {categories}")
    return _infer_category_from_path(path)


def _path_contains(path: Path, token: str) -> bool:
    token_l = token.lower()
    return any(part.lower() == token_l for part in path.parts)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _discover_csv_files(input_path: Path, output_root: Path, manifest_name: str) -> list[Path]:
    input_path = input_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"--input does not exist: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() != ".csv":
            raise ValueError(f"file input must be a .csv file: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise ValueError(f"--input must be a CSV file or directory: {input_path}")

    files: list[Path] = []
    for path in input_path.rglob("*.csv"):
        resolved = path.resolve()
        if _is_relative_to(resolved, output_root):
            continue
        if path.name == manifest_name:
            continue
        files.append(resolved)
    return sorted(files, key=lambda p: str(p))


def _default_output_root(input_path: Path) -> Path:
    input_path = input_path.expanduser().resolve()
    if input_path.is_dir():
        return input_path / "plots"
    return input_path.parent / "plots"


def _safe_token(value: object, fallback: str = "value") -> str:
    text = str(value if value is not None else "").strip().lower()
    text = text.replace("->", "_to_")
    text = text.replace("-", "_")
    text = re.sub(r"[^a-zA-Z0-9가-힣_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or fallback


def _human_token(value: object, fallback: str = "") -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        return fallback
    return text.replace("_", " ").replace("-", " ").title()


def _first_nonempty(rows: Sequence[dict[str, str]], column: str, default: str = "") -> str:
    for row in rows:
        value = str(row.get(column, "")).strip()
        if value:
            return value
    return default


def _unique_nonempty(rows: Sequence[dict[str, str]], column: str) -> list[str]:
    values = sorted({str(row.get(column, "")).strip() for row in rows if str(row.get(column, "")).strip()})
    return values


def _float_or_none(value: object) -> float | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        number = float(text)
        if not math.isfinite(number):
            return None
        return number
    except Exception:
        return None


def _numeric_values(rows: Sequence[dict[str, str]], column: str = "value") -> list[float]:
    values: list[float] = []
    for row in rows:
        if str(row.get("status", "ok")).strip().lower() not in {"", "ok"}:
            continue
        value = _float_or_none(row.get(column, ""))
        if value is not None:
            values.append(value)
    return values


def _expanded_ylim(values: Sequence[float], *, zero_floor: bool = False) -> tuple[float, float] | None:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return None
    ymin = min(clean)
    ymax = max(clean)
    if zero_floor and ymin >= 0:
        ymin = 0.0
    if ymin == ymax:
        if ymin == 0:
            return (-1.0, 1.0)
        pad = abs(ymin) * 0.05
        return (ymin - pad, ymax + pad)
    pad = (ymax - ymin) * 0.05
    return (ymin - pad, ymax + pad)


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _new_figure():
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.tick_params(axis="both", labelsize=TICK_SIZE, width=1.2, length=6)
    return plt, fig, ax


def _save_figure(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")


def _relative_output_path(csv_path: Path, input_root: Path, output_root: Path, file_format: str) -> Path:
    csv_path = csv_path.resolve()
    input_root = input_root.resolve()
    if input_root.is_dir():
        try:
            rel = csv_path.relative_to(input_root)
        except ValueError:
            rel = Path(csv_path.name)
        return output_root / rel.with_suffix(f".{file_format}")
    return output_root / f"{_safe_token(csv_path.stem)}.{file_format}"


def _multi_output_path(base_path: Path, suffix: str, file_format: str) -> Path:
    return base_path.with_name(f"{base_path.stem}__{_safe_token(suffix)}.{file_format}")


def _layer_display(rows: Sequence[dict[str, str]]) -> str:
    raw_index = _first_nonempty(rows, "layer_index")
    raw_layer = _first_nonempty(rows, "layer")
    try:
        layer_index = int(float(raw_index))
        if layer_index <= 0:
            return "Input"
        return f"Layer {layer_index}"
    except Exception:
        pass
    if raw_layer:
        return _human_token(raw_layer)
    return "Layer"


def _psd_title(category: str, rows: Sequence[dict[str, str]]) -> str:
    layer = _layer_display(rows)
    if category.endswith("curve"):
        return f"PSD of {layer}"
    return f"PSD Dispersion of {layer}"


def _psd_ylabel(category: str, rows: Sequence[dict[str, str]]) -> str:
    scale = _first_nonempty(rows, "scale").lower()
    unit = _first_nonempty(rows, "value_unit")
    statistic = _first_nonempty(rows, "statistic")
    if category.endswith("curve"):
        return "Power (dB)" if scale == "db" else "Power"
    stat_name = _human_token(statistic, "Dispersion")
    if unit:
        return f"{stat_name} ({unit})"
    return stat_name


def _series_label(row: dict[str, str], category: str) -> str:
    if category.endswith("curve"):
        columns = ["probe_family", "label", "signal_kind", "series", "extractor", "reducer", "variant", "scale"]
    elif category.endswith("dispersion"):
        columns = ["probe_family", "label", "signal_kind", "series", "extractor", "statistic", "variant", "scale"]
    else:
        columns = ["metric", "statistic"]
    tokens = [_human_token(row.get(column, "")) for column in columns if str(row.get(column, "")).strip()]
    # In the usual split psd_analysis files, all rows have one effective series.
    # Keep the label short to avoid unreadable legends.
    if len(tokens) <= 2:
        return " / ".join(tokens) if tokens else "Series"
    return " / ".join(tokens[-4:])


def _curve_ylim_key(artifact: CsvArtifact) -> tuple[str, ...]:
    rows = artifact.rows
    fields = [
        "category",
        "dataset",
        "run_id",
        "model_token",
        "model_family",
        "readout_mode",
        "seed",
        "checkpoint_epoch",
        "scope",
        "probe_family",
        "label",
        "signal_kind",
        "series",
        "extractor",
        "reducer",
        "statistic",
        "variant",
        "scale",
        "value_unit",
    ]
    values = []
    for field in fields:
        if field == "category":
            values.append(artifact.category)
        else:
            values.append(_first_nonempty(rows, field))
    return tuple(values)


def _build_curve_ylim_map(artifacts: Sequence[CsvArtifact]) -> dict[tuple[str, ...], tuple[float, float]]:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for artifact in artifacts:
        if artifact.category not in PSD_CATEGORIES:
            continue
        key = _curve_ylim_key(artifact)
        grouped[key].extend(_numeric_values(artifact.rows))
    result: dict[tuple[str, ...], tuple[float, float]] = {}
    for key, values in grouped.items():
        ylim = _expanded_ylim(values, zero_floor=False)
        if ylim is not None:
            result[key] = ylim
    return result


def _render_psd_artifact(
    artifact: CsvArtifact,
    *,
    input_root: Path,
    output_root: Path,
    file_format: str,
    overwrite: bool,
    ylimit_map: dict[tuple[str, ...], tuple[float, float]],
) -> ManifestRow:
    start = time.perf_counter()
    output_path = _relative_output_path(artifact.path, input_root, output_root, file_format)
    if output_path.exists() and not overwrite:
        return ManifestRow(str(artifact.path), str(output_path), artifact.category, "skipped_existing", "output exists")

    try:
        groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for row in artifact.rows:
            if str(row.get("status", "ok")).strip().lower() not in {"", "ok"}:
                continue
            x = _float_or_none(row.get("frequency", ""))
            y = _float_or_none(row.get("value", ""))
            if x is None or y is None:
                continue
            groups[_series_label(row, artifact.category)].append((x, y))
        if not groups:
            raise ValueError("no numeric frequency/value rows")

        plt, fig, ax = _new_figure()
        for label, points in sorted(groups.items(), key=lambda item: item[0]):
            ordered = sorted(points, key=lambda item: item[0])
            xs = [p[0] for p in ordered]
            ys = [p[1] for p in ordered]
            ax.plot(xs, ys, linewidth=LINEWIDTH, label=label)

        all_x = [x for points in groups.values() for x, _y in points]
        if all_x and min(all_x) >= -1e-9 and max(all_x) <= 0.5000001:
            ax.set_xlim(0.0, 0.5)
            ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ylim = ylimit_map.get(_curve_ylim_key(artifact))
        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.set_title(_psd_title(artifact.category, artifact.rows), fontsize=TITLE_SIZE, fontweight="bold", pad=10)
        ax.set_xlabel("Normalized Frequency", fontsize=LABEL_SIZE, fontweight="bold", labelpad=8)
        ax.set_ylabel(_psd_ylabel(artifact.category, artifact.rows), fontsize=LABEL_SIZE, fontweight="bold", labelpad=8)
        if len(groups) > 1:
            ax.legend(fontsize=10, frameon=False, loc="best")
        _save_figure(fig, output_path)
        plt.close(fig)
        return ManifestRow(
            str(artifact.path),
            str(output_path),
            artifact.category,
            "rendered",
            "",
            f"{time.perf_counter() - start:.6f}",
        )
    except Exception as exc:
        return ManifestRow(str(artifact.path), str(output_path), artifact.category, "failed", str(exc))


def _filter_group_key(row: dict[str, str]) -> tuple[str, ...]:
    fields = [
        "dataset",
        "run_id",
        "model_token",
        "model_family",
        "seed",
        "checkpoint_epoch",
        "layer",
        "layer_index",
        "parameter_name",
    ]
    return tuple(str(row.get(field, "")).strip() for field in fields)


def _filter_group_suffix(rows: Sequence[dict[str, str]]) -> str:
    epoch = _first_nonempty(rows, "checkpoint_epoch", "epoch")
    layer = _first_nonempty(rows, "layer_index", "layer")
    parameter = _first_nonempty(rows, "parameter_name", "parameter")
    return f"epoch_{epoch}__layer_{layer}__{parameter}"


def _filter_title(rows: Sequence[dict[str, str]]) -> str:
    layer = _layer_display(rows)
    parameter = _human_token(_first_nonempty(rows, "parameter_name"), "Parameter")
    return f"Filter Snapshot of {layer} - {parameter}"


def _render_filter_artifact(
    artifact: CsvArtifact,
    *,
    input_root: Path,
    output_root: Path,
    file_format: str,
    overwrite: bool,
    include_count: bool,
) -> list[ManifestRow]:
    base_output_path = _relative_output_path(artifact.path, input_root, output_root, file_format)
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in artifact.rows:
        grouped[_filter_group_key(row)].append(row)

    results: list[ManifestRow] = []
    if not grouped:
        return [ManifestRow(str(artifact.path), str(base_output_path), artifact.category, "failed", "empty filter_snapshot rows")]

    multiple_groups = len(grouped) > 1
    for _key, rows in sorted(grouped.items(), key=lambda item: item[0]):
        start = time.perf_counter()
        output_path = base_output_path
        if multiple_groups:
            output_path = _multi_output_path(base_output_path, _filter_group_suffix(rows), file_format)
        if output_path.exists() and not overwrite:
            results.append(ManifestRow(str(artifact.path), str(output_path), artifact.category, "skipped_existing", "output exists"))
            continue
        try:
            bars: list[tuple[str, float, str]] = []
            for row in rows:
                if str(row.get("status", "ok")).strip().lower() not in {"", "ok"}:
                    continue
                statistic = str(row.get("statistic", "")).strip()
                if not include_count and statistic.lower() == "count":
                    continue
                value = _float_or_none(row.get("value", ""))
                if value is None:
                    continue
                bars.append((statistic or "value", value, str(row.get("value_unit", "")).strip()))
            if not bars and not include_count:
                for row in rows:
                    value = _float_or_none(row.get("value", ""))
                    if value is not None:
                        bars.append((str(row.get("statistic", "value")).strip() or "value", value, str(row.get("value_unit", "")).strip()))
            if not bars:
                raise ValueError("no numeric filter_snapshot values")

            order = ["mean", "std", "min", "q25", "q50", "q75", "max", "count"]
            order_index = {name: index for index, name in enumerate(order)}
            bars = sorted(bars, key=lambda item: (order_index.get(item[0].lower(), 999), item[0]))
            labels = [_human_token(item[0]) for item in bars]
            values = [item[1] for item in bars]
            units = sorted({unit for _stat, _value, unit in bars if unit})
            ylabel = units[0] if len(units) == 1 else "Value"
            if ylabel == "parameter_value":
                ylabel = "Parameter Value"

            plt, fig, ax = _new_figure()
            ax.bar(range(len(values)), values)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ylim = _expanded_ylim(values, zero_floor=(min(values) >= 0 if values else True))
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.set_title(_filter_title(rows), fontsize=TITLE_SIZE, fontweight="bold", pad=10)
            ax.set_xlabel("Statistic", fontsize=LABEL_SIZE, fontweight="bold", labelpad=8)
            ax.set_ylabel(_human_token(ylabel, "Value"), fontsize=LABEL_SIZE, fontweight="bold", labelpad=8)
            _save_figure(fig, output_path)
            plt.close(fig)
            results.append(
                ManifestRow(
                    str(artifact.path),
                    str(output_path),
                    artifact.category,
                    "rendered",
                    "",
                    f"{time.perf_counter() - start:.6f}",
                )
            )
        except Exception as exc:
            results.append(ManifestRow(str(artifact.path), str(output_path), artifact.category, "failed", str(exc)))
    return results


def _drift_measure_type(metric: str) -> str:
    token = str(metric).strip().lower()
    if token.startswith("dispersion_variance"):
        return "dispersion_variance"
    if token.startswith("dispersion_mad"):
        return "dispersion_mad"
    return "curve_distance"


def _drift_group_key(row: dict[str, str], *, measure_type: str) -> tuple[str, ...]:
    fields = [
        "dataset",
        "run_id",
        "model_token",
        "model_family",
        "readout_mode",
        "seed",
        "layer",
        "layer_index",
        "scope",
        "signal_kind",
        "series",
        "reference_signal_kind",
        "reference_series",
        "extractor",
        "variant",
        "scale",
    ]
    values = [str(row.get(field, "")).strip() for field in fields]
    values.append(str(row.get("reducer", "")).strip() if measure_type == "curve_distance" else "none")
    values.append(measure_type)
    return tuple(values)


def _drift_row_fingerprint(row: dict[str, str]) -> tuple[str, ...]:
    measure_type = _drift_measure_type(row.get("distance_metric", ""))
    fields = [
        "dataset",
        "run_id",
        "model_token",
        "model_family",
        "readout_mode",
        "seed",
        "checkpoint_epoch_a",
        "checkpoint_epoch_b",
        "layer",
        "layer_index",
        "scope",
        "signal_kind",
        "series",
        "reference_signal_kind",
        "reference_series",
        "extractor",
        "variant",
        "scale",
        "distance_metric",
        "value",
    ]
    values = [str(row.get(field, "")).strip() for field in fields]
    values.append(str(row.get("reducer", "")).strip() if measure_type == "curve_distance" else "none")
    return tuple(values)


def _drift_filename(rows: Sequence[dict[str, str]], file_format: str, *, measure_type: str) -> str:
    dataset = _safe_token(_first_nonempty(rows, "dataset", "dataset"), "dataset")
    model = _safe_token(_first_nonempty(rows, "model_token", "model"), "model")
    seed = _safe_token(_first_nonempty(rows, "seed", "seed"), "seed")
    layer = _safe_token(_first_nonempty(rows, "layer_index", "layer"), "layer")
    scope = _safe_token(_first_nonempty(rows, "scope", "scope"), "scope")
    signal = _safe_token(_first_nonempty(rows, "signal_kind", "signal"), "signal")
    series = _safe_token(_first_nonempty(rows, "series", "series"), "series")
    reference = _safe_token(_first_nonempty(rows, "reference_series", "input"), "input")
    extractor = _safe_token(_first_nonempty(rows, "extractor", "extractor"), "extractor")
    reducer = _safe_token(_first_nonempty(rows, "reducer", "none"), "none")
    variant = _safe_token(_first_nonempty(rows, "variant", "variant"), "variant")
    scale = _safe_token(_first_nonempty(rows, "scale", "scale"), "scale")
    measure = _safe_token(measure_type, "measure")
    base = f"{measure}__{dataset}__{model}__seed_{seed}__layer_{layer}__{scope}__input_{reference}__to__{signal}__{series}__{extractor}"
    if measure_type == "curve_distance":
        return f"{base}__{reducer}__{variant}__{scale}.{file_format}"
    return f"{base}__{variant}__{scale}.{file_format}"


def _drift_title(rows: Sequence[dict[str, str]], *, measure_type: str) -> str:
    layer = _layer_display(rows)
    names = {
        "curve_distance": "Input-Reference PSD Curve Distance",
        "dispersion_variance": "Input-Reference PSD Variance Distance",
        "dispersion_mad": "Input-Reference PSD MAD Distance",
    }
    return f"{names.get(measure_type, 'Input-Reference PSD Distance')} of {layer}"


def _drift_ylabel(measure_type: str) -> str:
    return {
        "curve_distance": "Curve Shape Distance",
        "dispersion_variance": "Variance Shape Distance",
        "dispersion_mad": "MAD Shape Distance",
    }.get(measure_type, "Distance")


def _drift_output_subdir(measure_type: str) -> str:
    return {
        "curve_distance": "curve_distance",
        "dispersion_variance": "dispersion_variance",
        "dispersion_mad": "dispersion_mad",
    }.get(measure_type, _safe_token(measure_type, "distance"))


def _drift_point(row: dict[str, str]) -> DriftPoint | None:
    if str(row.get("status", "ok")).strip().lower() not in {"", "ok"}:
        return None
    y = _float_or_none(row.get("value", ""))
    if y is None:
        return None
    epoch_a = str(row.get("checkpoint_epoch_a", "")).strip()
    epoch_b = str(row.get("checkpoint_epoch_b", "")).strip()
    metric = str(row.get("distance_metric", "")).strip() or "distance"
    x = _float_or_none(epoch_b or epoch_a)
    if epoch_a and epoch_b and epoch_a != epoch_b:
        label = f"{epoch_a}->{epoch_b}"
    else:
        label = epoch_b or epoch_a or "epoch"
    return DriftPoint(metric=metric, epoch_a=epoch_a, epoch_b=epoch_b, x=x, y=y, transition_label=label)


def _render_drift_groups(
    drift_artifacts: Sequence[CsvArtifact],
    *,
    input_root: Path,
    output_root: Path,
    file_format: str,
    overwrite: bool,
) -> list[ManifestRow]:
    if not drift_artifacts:
        return []

    dedup: dict[tuple[str, ...], dict[str, str]] = {}
    for artifact in drift_artifacts:
        for row in artifact.rows:
            dedup[_drift_row_fingerprint(row)] = row

    grouped_rows: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in dedup.values():
        measure_type = _drift_measure_type(row.get("distance_metric", ""))
        grouped_rows[_drift_group_key(row, measure_type=measure_type)].append(row)

    results: list[ManifestRow] = []
    drift_root_dir = output_root / "traces" / "drift_distance"
    source_hint = str(input_root / "**" / "drift_distance*.csv")

    for key, rows in sorted(grouped_rows.items(), key=lambda item: item[0]):
        measure_type = str(key[-1]) if key else "curve_distance"
        start = time.perf_counter()
        output_path = drift_root_dir / _drift_output_subdir(measure_type) / _drift_filename(rows, file_format, measure_type=measure_type)
        if output_path.exists() and not overwrite:
            results.append(ManifestRow(source_hint, str(output_path), "drift_distance", "skipped_existing", "output exists"))
            continue
        try:
            by_metric: dict[str, list[DriftPoint]] = defaultdict(list)
            for row in rows:
                point = _drift_point(row)
                if point is not None:
                    by_metric[point.metric].append(point)
            if not by_metric:
                raise ValueError("no numeric drift_distance values")

            plt, fig, ax = _new_figure()
            all_points: list[DriftPoint] = []
            categorical = any(point.x is None for points in by_metric.values() for point in points)
            category_index: dict[str, int] = {}

            if categorical:
                ordered_labels = sorted(
                    {point.transition_label for points in by_metric.values() for point in points},
                    key=lambda label: [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", label)],
                )
                category_index = {label: idx for idx, label in enumerate(ordered_labels)}

            for metric, points in sorted(by_metric.items(), key=lambda item: item[0]):
                if categorical:
                    ordered = sorted(points, key=lambda point: category_index.get(point.transition_label, 0))
                    xs = [float(category_index[point.transition_label]) for point in ordered]
                else:
                    ordered = sorted(points, key=lambda point: (float(point.x), point.transition_label))
                    xs = [float(point.x) for point in ordered if point.x is not None]
                ys = [point.y for point in ordered]
                n = min(len(xs), len(ys))
                xs = xs[:n]
                ys = ys[:n]
                ax.plot(xs, ys, linewidth=LINEWIDTH, marker="o", label=_human_token(metric, metric))
                all_points.extend(ordered)

            if categorical:
                labels = [label.replace("_", " ") for label in sorted(category_index, key=lambda item: category_index[item])]
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha="right")
            else:
                tick_by_x = {float(point.x): point.transition_label for point in all_points if point.x is not None}
                ticks = sorted(tick_by_x)
                if ticks:
                    ax.set_xticks(ticks)
                    ax.set_xticklabels([tick_by_x[t] for t in ticks], rotation=45, ha="right")
            ax.set_xlabel("Epoch", fontsize=LABEL_SIZE, fontweight="bold", labelpad=8)
            y_values = [point.y for point in all_points]
            ylim = _expanded_ylim(y_values, zero_floor=(min(y_values) >= 0 if y_values else True))
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.set_title(_drift_title(rows, measure_type=measure_type), fontsize=TITLE_SIZE, fontweight="bold", pad=10)
            ax.set_ylabel(_drift_ylabel(measure_type), fontsize=LABEL_SIZE, fontweight="bold", labelpad=8)
            if len(by_metric) > 1:
                ax.legend(fontsize=10, frameon=False, loc="best")
            _save_figure(fig, output_path)
            plt.close(fig)
            results.append(
                ManifestRow(
                    source_hint,
                    str(output_path),
                    "drift_distance",
                    "rendered",
                    f"aggregated {measure_type} trend plot",
                    f"{time.perf_counter() - start:.6f}",
                )
            )
        except Exception as exc:
            results.append(ManifestRow(source_hint, str(output_path), "drift_distance", "failed", str(exc)))
    return results



def _render_layer_distance_artifact(
    artifact: CsvArtifact,
    *,
    input_root: Path,
    output_root: Path,
    file_format: str,
    overwrite: bool,
) -> ManifestRow:
    output_path = _relative_output_path(artifact.path, input_root, output_root, file_format)
    start = time.perf_counter()
    if output_path.exists() and not overwrite:
        return ManifestRow(str(artifact.path), str(output_path), artifact.category, "skipped_existing", "output exists")
    try:
        rows = [row for row in artifact.rows if str(row.get("status", "ok")).strip().lower() in {"", "ok"}]
        if not rows:
            raise ValueError("no successful layer-distance rows")
        plt, fig, ax = _new_figure()
        group_key = "distance_metric"
        by_metric: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_metric[str(row.get(group_key, "distance"))].append(row)
        categorical_labels: list[str] = []
        all_y: list[float] = []
        for metric, metric_rows in sorted(by_metric.items(), key=lambda item: item[0]):
            if artifact.category == "layer_distance_profile":
                ordered = sorted(metric_rows, key=lambda row: (_float_or_none(row.get("comparison_index")) or 0.0, str(row.get("comparison_label", ""))))
                xs = [float(index) for index, _row in enumerate(ordered)]
                labels = [str(row.get("comparison_label", row.get("target_layer", ""))) for row in ordered]
                categorical_labels = labels
            else:
                ordered = sorted(metric_rows, key=lambda row: (_float_or_none(row.get("checkpoint_epoch")) or 0.0, str(row.get("comparison_label", ""))))
                xs = [float(_float_or_none(row.get("checkpoint_epoch")) or 0.0) for row in ordered]
            ys = [float(_float_or_none(row.get("value")) or 0.0) for row in ordered]
            all_y.extend(ys)
            ax.plot(xs, ys, linewidth=LINEWIDTH, marker="o", label=_human_token(metric, metric))
        if artifact.category == "layer_distance_profile" and categorical_labels:
            ax.set_xticks(range(len(categorical_labels)))
            ax.set_xticklabels([label.replace("_", " ") for label in categorical_labels], rotation=45, ha="right")
            ax.set_xlabel("Layer relation", fontsize=LABEL_SIZE, fontweight="bold", labelpad=8)
        else:
            ax.set_xlabel("Epoch", fontsize=LABEL_SIZE, fontweight="bold", labelpad=8)
        ylim = _expanded_ylim(all_y, zero_floor=True)
        if ylim is not None:
            ax.set_ylim(*ylim)
        relation = _first_nonempty(rows, "relation_type", "relation")
        track = _first_nonempty(rows, "track_name", "track")
        title = f"{_human_token(artifact.category)} - {_human_token(relation)} - {_human_token(track)}"
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=10)
        ax.set_ylabel("Curve Shape Distance", fontsize=LABEL_SIZE, fontweight="bold", labelpad=8)
        if len(by_metric) > 1:
            ax.legend(fontsize=10, frameon=False, loc="best")
        _save_figure(fig, output_path)
        plt.close(fig)
        return ManifestRow(str(artifact.path), str(output_path), artifact.category, "rendered", "layer distance plot", f"{time.perf_counter() - start:.6f}")
    except Exception as exc:
        return ManifestRow(str(artifact.path), str(output_path), artifact.category, "failed", str(exc))


def _render_layer_dispersion_artifact(
    artifact: CsvArtifact,
    *,
    input_root: Path,
    output_root: Path,
    file_format: str,
    overwrite: bool,
) -> ManifestRow:
    output_path = _relative_output_path(artifact.path, input_root, output_root, file_format)
    start = time.perf_counter()
    if output_path.exists() and not overwrite:
        return ManifestRow(str(artifact.path), str(output_path), artifact.category, "skipped_existing", "output exists")
    try:
        rows = [row for row in artifact.rows if str(row.get("status", "ok")).strip().lower() in {"", "ok"}]
        if not rows:
            raise ValueError("no successful layer-dispersion rows")
        plt, fig, ax = _new_figure()
        by_stat: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_stat[str(row.get("dispersion_statistic", row.get("statistic", "dispersion")))].append(row)
        all_y: list[float] = []
        categorical_labels: list[str] = []
        for stat, stat_rows in sorted(by_stat.items(), key=lambda item: item[0]):
            if artifact.category == "layer_dispersion_profile":
                ordered = sorted(stat_rows, key=lambda row: (_float_or_none(row.get("layer_index")) or 0.0, str(row.get("layer", ""))))
                xs = [float(index) for index, _row in enumerate(ordered)]
                categorical_labels = [str(row.get("layer", row.get("layer_index", ""))) for row in ordered]
            else:
                ordered = sorted(stat_rows, key=lambda row: (_float_or_none(row.get("checkpoint_epoch")) or 0.0, str(row.get("layer", ""))))
                xs = [float(_float_or_none(row.get("checkpoint_epoch")) or 0.0) for row in ordered]
            ys = [float(_float_or_none(row.get("value")) or 0.0) for row in ordered]
            all_y.extend(ys)
            ax.plot(xs, ys, linewidth=LINEWIDTH, marker="o", label=_human_token(stat, stat))
        if artifact.category == "layer_dispersion_profile" and categorical_labels:
            ax.set_xticks(range(len(categorical_labels)))
            ax.set_xticklabels([label.replace("_", " ") for label in categorical_labels], rotation=45, ha="right")
            ax.set_xlabel("Layer", fontsize=LABEL_SIZE, fontweight="bold", labelpad=8)
        else:
            ax.set_xlabel("Epoch", fontsize=LABEL_SIZE, fontweight="bold", labelpad=8)
        ylim = _expanded_ylim(all_y, zero_floor=True)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(_human_token(artifact.category), fontsize=TITLE_SIZE, fontweight="bold", pad=10)
        ax.set_ylabel("PSD Dispersion", fontsize=LABEL_SIZE, fontweight="bold", labelpad=8)
        if len(by_stat) > 1:
            ax.legend(fontsize=10, frameon=False, loc="best")
        _save_figure(fig, output_path)
        plt.close(fig)
        return ManifestRow(str(artifact.path), str(output_path), artifact.category, "rendered", "layer dispersion plot", f"{time.perf_counter() - start:.6f}")
    except Exception as exc:
        return ManifestRow(str(artifact.path), str(output_path), artifact.category, "failed", str(exc))


def _write_manifest(path: Path, rows: Sequence[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["input_csv_path", "output_figure_path", "category", "status", "message", "render_seconds"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "input_csv_path": row.input_csv_path,
                    "output_figure_path": row.output_figure_path,
                    "category": row.category,
                    "status": row.status,
                    "message": row.message,
                    "render_seconds": row.render_seconds,
                }
            )


def _load_artifacts(csv_files: Sequence[Path]) -> tuple[list[CsvArtifact], list[ManifestRow]]:
    artifacts: list[CsvArtifact] = []
    manifest: list[ManifestRow] = []
    for path in csv_files:
        try:
            rows = _read_csv(path)
            category = _infer_category(path, rows)
            if not category:
                manifest.append(ManifestRow(str(path), "", "", "skipped_unsupported", "cannot infer category"))
                continue
            if _path_contains(path, "pair_distance") or category == "pair_distance":
                manifest.append(ManifestRow(str(path), "", category, "skipped_requested", "pair_distance is intentionally skipped"))
                continue
            if _path_contains(path, "accuracy_loss_join") or category == "accuracy_loss_join":
                manifest.append(ManifestRow(str(path), "", category, "skipped_requested", "accuracy_loss_join is intentionally skipped"))
                continue
            if category in SKIPPED_CATEGORIES:
                manifest.append(ManifestRow(str(path), "", category, "skipped_unsupported", f"category is not rendered: {category}"))
                continue
            if category not in RENDERED_CATEGORIES:
                manifest.append(ManifestRow(str(path), "", category, "skipped_unsupported", f"unsupported category: {category}"))
                continue
            artifacts.append(CsvArtifact(path=path, category=category, rows=rows))
        except Exception as exc:
            manifest.append(ManifestRow(str(path), "", "", "failed_read", str(exc)))
    return artifacts, manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parse_args_with_config(parser, argv=argv, stage_key='plotting')

    input_path = Path(args.input).expanduser().resolve()
    if args.output and args.output_root:
        parser.error("use only one of --output or --output_root, not both.")
    output_arg = args.output if args.output is not None else args.output_root
    output_root = Path(output_arg).expanduser().resolve() if output_arg else _default_output_root(input_path)
    manifest_name = str(args.manifest_name).strip()
    if not manifest_name or Path(manifest_name).name != manifest_name or not manifest_name.endswith(".csv"):
        parser.error("--manifest_name must be a CSV filename without directory separators.")

    output_root.mkdir(parents=True, exist_ok=True)
    csv_files = _discover_csv_files(input_path, output_root, manifest_name)
    artifacts, manifest_rows = _load_artifacts(csv_files)

    psd_artifacts = [artifact for artifact in artifacts if artifact.category in PSD_CATEGORIES]
    filter_artifacts = [artifact for artifact in artifacts if artifact.category == "filter_snapshot"]
    layer_distance_artifacts = [artifact for artifact in artifacts if artifact.category in {"layer_distance_profile", "layer_distance_trend"}]
    layer_dispersion_artifacts = [artifact for artifact in artifacts if artifact.category in {"layer_dispersion_profile", "layer_dispersion_trend"}]

    ylimit_map = _build_curve_ylim_map(psd_artifacts)
    for artifact in psd_artifacts:
        manifest_rows.append(
            _render_psd_artifact(
                artifact,
                input_root=input_path,
                output_root=output_root,
                file_format=args.format,
                overwrite=bool(args.overwrite),
                ylimit_map=ylimit_map,
            )
        )

    for artifact in filter_artifacts:
        manifest_rows.extend(
            _render_filter_artifact(
                artifact,
                input_root=input_path,
                output_root=output_root,
                file_format=args.format,
                overwrite=bool(args.overwrite),
                include_count=bool(args.include_filter_count),
            )
        )

    for artifact in layer_distance_artifacts:
        manifest_rows.append(
            _render_layer_distance_artifact(
                artifact,
                input_root=input_path,
                output_root=output_root,
                file_format=args.format,
                overwrite=bool(args.overwrite),
            )
        )

    for artifact in layer_dispersion_artifacts:
        manifest_rows.append(
            _render_layer_dispersion_artifact(
                artifact,
                input_root=input_path,
                output_root=output_root,
                file_format=args.format,
                overwrite=bool(args.overwrite),
            )
        )

    if not csv_files:
        manifest_rows.append(ManifestRow(str(input_path), "", "", "skipped_empty_input", "no CSV files found"))

    manifest_path = output_root / manifest_name
    manifest_rows_sorted = sorted(manifest_rows, key=lambda row: (row.category, row.input_csv_path, row.output_figure_path, row.status))
    _write_manifest(manifest_path, manifest_rows_sorted)

    rendered = sum(1 for row in manifest_rows if row.status == "rendered")
    skipped = sum(1 for row in manifest_rows if row.status.startswith("skipped"))
    failed = sum(1 for row in manifest_rows if row.status.startswith("failed"))
    print(
        json.dumps(
            {
                "status": "ok" if failed == 0 else "partial",
                "input": str(input_path),
                "output_root": str(output_root),
                "manifest": str(manifest_path),
                "csv_files": len(csv_files),
                "rendered": rendered,
                "skipped": skipped,
                "failed": failed,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
