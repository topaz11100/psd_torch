#!/usr/bin/env python3
"""
Recursive CSV -> heatmap plot converter for element_psd-style CSV files.

Expected CSV structure:
- metadata columns followed by many columns named freq_000000, freq_000001, ...
- each row is one element/neuron
- each freq_* column is a PSD value for a normalized frequency bin

Example:
python plot_element_psd_heatmaps.py \
  --csv-root /path/to/csv_root \
  --plot-root /path/to/plot_root \
  --overwrite
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _str_value(value: object, default: str = "unknown") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    text = str(value)
    return text if text else default


def _first_row_value(first_row: pd.DataFrame, column: str, default: str = "unknown") -> str:
    if column not in first_row.columns or first_row.empty:
        return default
    return _str_value(first_row.iloc[0][column], default=default)


def _find_freq_columns(columns: Iterable[str]) -> list[str]:
    freq_cols = [c for c in columns if c.startswith("freq_")]

    def key(name: str) -> int:
        suffix = name.split("freq_", 1)[1]
        try:
            return int(suffix)
        except ValueError:
            return 10**12

    return sorted(freq_cols, key=key)


def _make_title(first_row: pd.DataFrame, max_chars: int = 120) -> str:
    dataset = _first_row_value(first_row, "dataset")
    layer = _first_row_value(first_row, "layer")
    signal_kind = _first_row_value(first_row, "signal_kind")
    series = _first_row_value(first_row, "series")
    scope = _first_row_value(first_row, "scope")
    label = _first_row_value(first_row, "label", default="")
    variant = _first_row_value(first_row, "variant")
    scale = _first_row_value(first_row, "scale")

    signal = f"{signal_kind}/{series}"
    label_part = f" | label {label}" if label not in {"", "unknown"} else ""
    title = (
        "Element PSD Heatmap\n"
        f"{dataset} | {layer} | {signal} | {scope}{label_part} | {variant}/{scale}"
    )

    lines = []
    for line in title.splitlines():
        if len(line) <= max_chars:
            lines.append(line)
        else:
            lines.append(line[: max_chars - 3] + "...")
    return "\n".join(lines)


def _default_ylabel(first_row: pd.DataFrame) -> str:
    signal_kind = _first_row_value(first_row, "signal_kind", default="")
    if signal_kind == "input":
        return "Element Index"
    return "Neuron Index"


def _normalized_frequency_ticks(n_bins: int, tick_count: int = 6) -> tuple[np.ndarray, list[str]]:
    if n_bins <= 1:
        return np.array([0]), ["0.0"]
    positions = np.linspace(0, n_bins - 1, tick_count)
    labels = [f"{v:.1f}" for v in np.linspace(0.0, 0.5, tick_count)]
    return positions, labels


def plot_csv_to_heatmap(
    csv_path: Path,
    out_path: Path,
    *,
    dpi: int,
    fig_width: float,
    fig_height: float,
    overwrite: bool,
    vmin: float | None,
    vmax: float | None,
    title_max_chars: int,
) -> dict[str, object]:
    if out_path.exists() and not overwrite:
        return {
            "status": "skipped_exists",
            "csv": str(csv_path),
            "png": str(out_path),
        }

    first_row = pd.read_csv(csv_path, nrows=1)
    freq_cols = _find_freq_columns(first_row.columns)
    if not freq_cols:
        return {
            "status": "skipped_no_freq_columns",
            "csv": str(csv_path),
            "png": str(out_path),
        }

    matrix = pd.read_csv(csv_path, usecols=freq_cols).to_numpy(dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        return {
            "status": "skipped_empty_matrix",
            "csv": str(csv_path),
            "png": str(out_path),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)

    x_positions, x_labels = _normalized_frequency_ticks(matrix.shape[1])
    title = _make_title(first_row, max_chars=title_max_chars)
    ylabel = _default_ylabel(first_row)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(
        matrix,
        aspect="auto",
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("PSD (dB)")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Normalized Frequency")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    floor_rows = int(np.isclose(matrix, -120.0).all(axis=1).sum())
    return {
        "status": "ok",
        "csv": str(csv_path),
        "png": str(out_path),
        "shape": f"{matrix.shape[0]}x{matrix.shape[1]}",
        "min": float(np.nanmin(matrix)),
        "max": float(np.nanmax(matrix)),
        "all_floor_rows": floor_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively convert element_psd-style CSV files into heatmap PNG files, "
            "preserving the input directory structure under a separate output root."
        )
    )
    parser.add_argument(
        "--csv-root",
        required=True,
        type=Path,
        help="Root directory containing element_psd CSV files.",
    )
    parser.add_argument(
        "--plot-root",
        required=True,
        type=Path,
        help="Independent root directory where PNG plots will be written.",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern searched recursively under --csv-root. Default: *.csv",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output image DPI. Default: 200",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=12.0,
        help="Figure width in inches. Default: 12.0",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=6.0,
        help="Figure height in inches. Default: 6.0",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Optional fixed color lower bound.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Optional fixed color upper bound.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG files.",
    )
    parser.add_argument(
        "--title-max-chars",
        type=int,
        default=120,
        help="Maximum characters per title line before truncation. Default: 120",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional path to save a conversion summary CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    csv_root = args.csv_root.resolve()
    plot_root = args.plot_root.resolve()

    if not csv_root.exists():
        raise FileNotFoundError(f"csv-root does not exist: {csv_root}")
    if not csv_root.is_dir():
        raise NotADirectoryError(f"csv-root is not a directory: {csv_root}")

    csv_files = sorted(p for p in csv_root.rglob(args.pattern) if p.is_file())

    if not csv_files:
        print(f"[WARN] No CSV files found under {csv_root} with pattern {args.pattern!r}.")
        return 0

    results: list[dict[str, object]] = []
    total = len(csv_files)

    for index, csv_path in enumerate(csv_files, start=1):
        rel_path = csv_path.relative_to(csv_root)
        out_path = (plot_root / rel_path).with_suffix(".png")

        try:
            result = plot_csv_to_heatmap(
                csv_path,
                out_path,
                dpi=args.dpi,
                fig_width=args.fig_width,
                fig_height=args.fig_height,
                overwrite=args.overwrite,
                vmin=args.vmin,
                vmax=args.vmax,
                title_max_chars=args.title_max_chars,
            )
        except Exception as exc:
            result = {
                "status": "error",
                "csv": str(csv_path),
                "png": str(out_path),
                "error": repr(exc),
            }

        results.append(result)
        status = result.get("status", "unknown")
        print(f"[{index}/{total}] {status}: {csv_path} -> {out_path}")

    if args.summary_csv is not None:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(args.summary_csv, index=False)
        print(f"[INFO] Summary written: {args.summary_csv}")

    ok_count = sum(1 for r in results if r.get("status") == "ok")
    error_count = sum(1 for r in results if r.get("status") == "error")
    skipped_count = total - ok_count - error_count

    print(
        "[DONE] "
        f"total={total}, ok={ok_count}, skipped={skipped_count}, error={error_count}, "
        f"plot_root={plot_root}"
    )

    return 1 if error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
