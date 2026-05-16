"""Category-based CSV schema helpers for PSD numeric artifacts."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = 'psd_category_csv_20260502'
COMMON_PREFIX: tuple[str, ...] = (
    'schema_version',
    'category',
    'source_program',
    'status',
    'message',
    'dataset',
    'run_id',
    'created_at',
)

AXIS_METADATA_COLUMNS: tuple[str, ...] = (
    'prep_profile',
    'psd_axis_kind',
    'psd_time_axis',
    'psd_row_axes',
    'psd_flatten_rule',
    'psd_logical_shape',
    'static_repeat_T',
)

CATEGORY_COLUMNS: dict[str, tuple[str, ...]] = {
    'training_metric': (*AXIS_METADATA_COLUMNS, 'model_token', 'model_family', 'readout_mode', 'seed', 'epoch', 'scope', 'metric', 'value', 'value_unit'),
    'dataset_curve': (*AXIS_METADATA_COLUMNS, 'scope', 'probe_family', 'label', 'signal_kind', 'extractor', 'reducer', 'variant', 'scale', 'frequency', 'frequency_unit', 'bin_left', 'bin_right', 'value', 'value_unit'),
    'dataset_dispersion': (*AXIS_METADATA_COLUMNS, 'scope', 'probe_family', 'label', 'signal_kind', 'extractor', 'variant', 'scale', 'statistic', 'frequency', 'frequency_unit', 'bin_left', 'bin_right', 'value', 'value_unit'),
    'analysis_curve': (*AXIS_METADATA_COLUMNS, 'model_token', 'model_family', 'readout_mode', 'seed', 'checkpoint_path', 'checkpoint_epoch', 'layer', 'layer_index', 'scope', 'probe_family', 'label', 'signal_kind', 'series', 'extractor', 'reducer', 'variant', 'scale', 'frequency', 'frequency_unit', 'bin_left', 'bin_right', 'value', 'value_unit'),
    'analysis_dispersion': (*AXIS_METADATA_COLUMNS, 'model_token', 'model_family', 'readout_mode', 'seed', 'checkpoint_path', 'checkpoint_epoch', 'layer', 'layer_index', 'scope', 'probe_family', 'label', 'signal_kind', 'series', 'extractor', 'variant', 'scale', 'statistic', 'frequency', 'frequency_unit', 'bin_left', 'bin_right', 'value', 'value_unit'),
    'pair_distance': (*AXIS_METADATA_COLUMNS, 'model_token', 'model_family', 'readout_mode', 'seed', 'checkpoint_epoch', 'layer', 'layer_index', 'source_scope', 'target_scope', 'source_signal_kind', 'source_series', 'target_signal_kind', 'target_series', 'extractor', 'reducer', 'variant', 'scale', 'distance_metric', 'value', 'value_unit'),
    'layer_distance_profile': (*AXIS_METADATA_COLUMNS, 'model_token', 'model_family', 'readout_mode', 'seed', 'checkpoint_path', 'checkpoint_epoch', 'scope', 'probe_family', 'label', 'track_name', 'source_layer', 'source_layer_index', 'source_signal_kind', 'source_series', 'target_layer', 'target_layer_index', 'target_signal_kind', 'target_series', 'relation_type', 'comparison_index', 'comparison_label', 'extractor', 'reducer', 'variant', 'scale', 'distance_metric', 'value', 'value_unit'),
    'layer_distance_trend': (*AXIS_METADATA_COLUMNS, 'model_token', 'model_family', 'readout_mode', 'seed', 'checkpoint_path', 'checkpoint_epoch', 'scope', 'probe_family', 'label', 'track_name', 'source_layer', 'source_layer_index', 'source_signal_kind', 'source_series', 'target_layer', 'target_layer_index', 'target_signal_kind', 'target_series', 'relation_type', 'comparison_index', 'comparison_label', 'extractor', 'reducer', 'variant', 'scale', 'distance_metric', 'value', 'value_unit'),
    'layer_dispersion_profile': (*AXIS_METADATA_COLUMNS, 'model_token', 'model_family', 'readout_mode', 'seed', 'checkpoint_path', 'checkpoint_epoch', 'scope', 'probe_family', 'label', 'layer', 'layer_index', 'signal_kind', 'series', 'extractor', 'variant', 'scale', 'dispersion_statistic', 'dispersion_reduction', 'value', 'value_unit'),
    'layer_dispersion_trend': (*AXIS_METADATA_COLUMNS, 'model_token', 'model_family', 'readout_mode', 'seed', 'checkpoint_path', 'checkpoint_epoch', 'scope', 'probe_family', 'label', 'layer', 'layer_index', 'signal_kind', 'series', 'extractor', 'variant', 'scale', 'dispersion_statistic', 'dispersion_reduction', 'value', 'value_unit'),
    'drift_distance': (*AXIS_METADATA_COLUMNS, 'model_token', 'model_family', 'readout_mode', 'seed', 'checkpoint_epoch_a', 'checkpoint_epoch_b', 'layer', 'layer_index', 'scope', 'signal_kind', 'series', 'reference_signal_kind', 'reference_series', 'extractor', 'reducer', 'variant', 'scale', 'distance_metric', 'value', 'value_unit'),
    'filter_snapshot': (*AXIS_METADATA_COLUMNS, 'model_token', 'model_family', 'readout_mode', 'seed', 'checkpoint_path', 'checkpoint_epoch', 'layer', 'layer_index', 'parameter_name', 'statistic', 'value', 'value_unit'),
    'filter_trend': (*AXIS_METADATA_COLUMNS, 'model_token', 'model_family', 'readout_mode', 'seed', 'checkpoint_path', 'layer', 'layer_index', 'parameter_name', 'checkpoint_epoch', 'statistic', 'value', 'value_unit'),
    'accuracy_loss_join': (*AXIS_METADATA_COLUMNS, 'model_token', 'model_family', 'readout_mode', 'seed', 'checkpoint_epoch', 'metric', 'value', 'value_unit'),
    'pairwise_dependency_appendix': (*AXIS_METADATA_COLUMNS, 'model_token', 'model_family', 'readout_mode', 'seed', 'checkpoint_epoch', 'layer', 'layer_index', 'source_scope', 'target_scope', 'source_signal_kind', 'source_series', 'target_signal_kind', 'target_series', 'extractor', 'reducer', 'variant', 'scale', 'metric', 'value', 'value_unit'),
    'analysis_manifest': (*AXIS_METADATA_COLUMNS, 'checkpoint_path', 'checkpoint_epoch', 'artifact_name', 'output_csv_path'),
    'dataset_psd_manifest': (*AXIS_METADATA_COLUMNS, 'scope', 'artifact_name', 'output_csv_path'),
    'plotting_manifest': ('input_csv_path', 'output_figure_path', 'render_seconds'),
    'reinterpretation_metric': (*AXIS_METADATA_COLUMNS, 'experiment_id', 'model_family', 'seed', 'scope', 'signal_kind', 'extractor', 'variant', 'scale', 'metric', 'statistic', 'value', 'value_unit'),
}
REQUIRED_COLUMNS = COMMON_PREFIX
CATEGORY_NAMES = frozenset(CATEGORY_COLUMNS)

_SNAKE_CASE_RE = re.compile(r'^[a-z][a-z0-9_]*$')


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp suitable for artifact rows."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def encode_csv_value(value: Any) -> str:
    """Encode one scalar CSV cell."""

    if value is None:
        return ''
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return str(value)


def _category_from_row(row: Mapping[str, Any]) -> str:
    return str(row.get('category', '')).strip()


def category_columns(category: str) -> tuple[str, ...]:
    token = str(category).strip()
    if token not in CATEGORY_COLUMNS:
        raise ValueError(f'Unsupported CSV category: {category!r}.')
    return (*COMMON_PREFIX, *CATEGORY_COLUMNS[token])


def common_row(**overrides: Any) -> dict[str, str]:
    """Build one category-aware CSV row."""

    created_at = encode_csv_value(overrides.pop('created_at', utc_now_iso()))
    category = _category_from_row(overrides)
    if not category:
        raise ValueError('common_row requires category=...')
    if category not in CATEGORY_COLUMNS:
        raise ValueError(f'Unsupported CSV category: {category!r}.')
    row = {column: '' for column in category_columns(category)}
    row['schema_version'] = SCHEMA_VERSION
    row['category'] = category
    row['status'] = encode_csv_value(overrides.pop('status', 'ok'))
    row['created_at'] = created_at
    if 'split' in overrides and 'scope' not in overrides:
        overrides['scope'] = overrides.get('split')
    if 'metric' in overrides and 'parameter_name' not in overrides and category in {'filter_snapshot', 'filter_trend'}:
        overrides['parameter_name'] = overrides.get('metric')
    if 'metric' in overrides and 'distance_metric' not in overrides and category in {'pair_distance', 'drift_distance', 'layer_distance_profile', 'layer_distance_trend'}:
        overrides['distance_metric'] = overrides.get('metric')
    if 'input_csv_path' in overrides and 'output_csv_path' not in overrides and category in {'analysis_manifest', 'dataset_psd_manifest'}:
        overrides['output_csv_path'] = overrides.get('input_csv_path')
    for key, value in overrides.items():
        key_s = str(key)
        if key_s in row:
            row[key_s] = encode_csv_value(value)
    return row


def normalize_common_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Normalize row mappings to category-specific rows."""

    return [common_row(**dict(row)) for row in rows]


def infer_category_from_path(path: Path | str) -> str | None:
    """Infer a category from a stable artifact filename."""

    stem = Path(path).stem
    token = stem.split('__', 1)[0]
    if token in CATEGORY_COLUMNS:
        return token
    for category in CATEGORY_COLUMNS:
        if stem.startswith(category):
            return category
    return None


def _resolve_category(normalized: Sequence[Mapping[str, str]], path: Path) -> str:
    categories = sorted({str(row.get('category', '')).strip() for row in normalized if str(row.get('category', '')).strip()})
    if len(categories) > 1:
        raise ValueError(f'One CSV file must contain one category; got {categories} for {path}.')
    if len(categories) == 1:
        return categories[0]
    inferred = infer_category_from_path(path)
    if inferred is None:
        raise ValueError(f'Cannot infer CSV category for empty row set: {path}')
    return inferred


def _resolve_extra_columns(rows: Sequence[Mapping[str, Any]], base_columns: Sequence[str], extra_columns: Sequence[str] | None) -> list[str]:
    explicit = [] if extra_columns is None else [str(column) for column in extra_columns]
    discovered: list[str] = []
    known = set(base_columns)
    for row in rows:
        for key in row.keys():
            key_s = str(key)
            if key_s not in known and key_s not in discovered and key_s not in explicit:
                discovered.append(key_s)
    resolved = explicit + discovered
    for column in resolved:
        if not _SNAKE_CASE_RE.fullmatch(column):
            raise ValueError(f'Extra CSV column must be lowercase snake_case: {column!r}')
    return resolved


def write_common_csv(path: Path | str, rows: Iterable[Mapping[str, Any]], *, extra_columns: Sequence[str] | None = None) -> Path:
    """Write one category-specific CSV file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_rows = [dict(row) for row in rows]
    normalized = normalize_common_rows(raw_rows) if raw_rows else []
    category = _resolve_category(normalized, path)
    base_columns = list(category_columns(category))
    extra_resolved = _resolve_extra_columns(raw_rows, base_columns, extra_columns)
    fieldnames = base_columns + extra_resolved
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for raw, row in zip(raw_rows, normalized):
            restored = dict(row)
            for column in extra_resolved:
                restored[column] = encode_csv_value(raw.get(column, restored.get(column, '')))
            writer.writerow({column: encode_csv_value(restored.get(column, '')) for column in fieldnames})
    return path


def validate_common_csv_header(path: Path | str) -> list[str]:
    """Validate that a CSV begins with the category schema prefix."""

    path = Path(path)
    with path.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f'CSV file is empty: {path}') from exc
    if header[: len(COMMON_PREFIX)] != list(COMMON_PREFIX):
        raise ValueError(f'CSV does not start with the required category schema prefix: {path}')
    return list(header)


__all__ = [
    'CATEGORY_COLUMNS',
    'CATEGORY_NAMES',
    'AXIS_METADATA_COLUMNS',
    'COMMON_PREFIX',
    'REQUIRED_COLUMNS',
    'SCHEMA_VERSION',
    'category_columns',
    'common_row',
    'encode_csv_value',
    'infer_category_from_path',
    'normalize_common_rows',
    'utc_now_iso',
    'validate_common_csv_header',
    'write_common_csv',
]
