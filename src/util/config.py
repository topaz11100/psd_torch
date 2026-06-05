"""Filesystem and serialization helpers shared across the project.

Project output-format convention:
- user-authored configs and manifests are YAML;
- tabular experiment/analysis artifacts are CSV;
- checkpoint metadata stays JSON-serializable inside ``.pt`` checkpoint payloads.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

YAML_SUFFIXES = {'.yaml', '.yml'}
STRUCTURED_SUFFIXES = YAML_SUFFIXES
MANIFEST_FILENAMES = ('manifest.yaml', 'manifest.yml')


def ensure_dir(path: Path | str) -> Path:
    """Create one directory when missing and return it as ``Path``."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _require_yaml():
    """Import PyYAML with one clear project-level error message."""

    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError('YAML config/manifest support requires PyYAML. Install with: pip install pyyaml') from exc
    return yaml


def to_jsonable(value: Any, *, _raise_on_unknown: bool = True) -> Any:
    """Convert common scientific Python objects into JSON/YAML-safe values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v, _raise_on_unknown=_raise_on_unknown) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v, _raise_on_unknown=_raise_on_unknown) for v in value]
    try:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer, np.bool_)):
            return value.item()
    except Exception:
        pass
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except Exception:
        pass
    if _raise_on_unknown:
        raise TypeError(f'Object is not JSON/YAML serializable: {type(value).__name__}')
    return value


def _yaml_dumper():
    yaml = _require_yaml()

    class NoAliasSafeDumper(yaml.SafeDumper):
        def ignore_aliases(self, data: Any) -> bool:  # noqa: D401 - PyYAML hook
            return True

    return NoAliasSafeDumper


def save_yaml(path: Path | str, payload: dict[str, Any] | list[Any] | Mapping[str, Any]) -> None:
    """Save one YAML payload using UTF-8."""

    yaml = _require_yaml()
    path = Path(path)
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as handle:
        yaml.dump(
            to_jsonable(payload),
            handle,
            Dumper=_yaml_dumper(),
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )


def load_yaml(path: Path | str) -> Any:
    """Load one YAML payload."""

    yaml = _require_yaml()
    with Path(path).open('r', encoding='utf-8') as handle:
        payload = yaml.safe_load(handle)
    return {} if payload is None else payload


def resolve_structured_path(path: Path | str) -> Path:
    """Return an existing structured YAML file.

    If the supplied path exists it is returned as-is. Otherwise the same stem is
    probed with YAML suffix alternatives. Official config/manifest files use
    ``.yaml``.
    """

    resolved = Path(path).expanduser()
    if resolved.exists():
        return resolved
    suffix = resolved.suffix.lower()
    base = resolved.with_suffix('') if suffix in STRUCTURED_SUFFIXES else resolved
    for candidate in (base.with_suffix('.yaml'), base.with_suffix('.yml')):
        if candidate.exists():
            return candidate
    return resolved


def load_structured(path: Path | str) -> Any:
    """Load a structured config/manifest file from YAML."""

    resolved = resolve_structured_path(path)
    suffix = resolved.suffix.lower()
    if suffix in YAML_SUFFIXES:
        return load_yaml(resolved)
    raise ValueError(f'Unsupported structured file extension {suffix!r}: {resolved}')


def save_structured(path: Path | str, payload: dict[str, Any] | list[Any]) -> None:
    """Save a structured config/manifest file as YAML."""

    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix in YAML_SUFFIXES:
        save_yaml(resolved, payload)
        return
    raise ValueError(f'Unsupported structured file extension {suffix!r}: {resolved}')


def manifest_path_for_dir(directory: Path | str) -> Path:
    """Resolve the prepared-data manifest path under one dataset directory."""

    root = Path(directory)
    for filename in MANIFEST_FILENAMES:
        candidate = root / filename
        if candidate.exists():
            return candidate
    return root / 'manifest.yaml'


def resolve_manifest_path(directory: Path | str) -> Path:
    """Resolve ``manifest.yaml`` under a prepared dataset directory."""

    return manifest_path_for_dir(directory)


def load_manifest(path: Path | str) -> dict[str, Any]:
    """Load a YAML manifest."""

    payload = load_structured(path)
    if not isinstance(payload, dict):
        raise ValueError(f'Manifest root must be a mapping: {path}')
    return dict(payload)


def save_manifest(path: Path | str, payload: dict[str, Any] | list[Any]) -> None:
    """Save a manifest as YAML."""

    save_yaml(path, payload)


def compact_yaml(value: Any) -> str:
    """Serialize a scalar/list/dict compactly for embedding inside CSV cells."""

    if value in (None, ''):
        return ''
    if isinstance(value, str):
        return value
    yaml = _require_yaml()
    text = yaml.dump(
        to_jsonable(value),
        Dumper=_yaml_dumper(),
        allow_unicode=True,
        sort_keys=True,
        default_flow_style=True,
    ).strip()
    return text[:-4].strip() if text.endswith('\n...') else text


def save_text(path: Path | str, text: str) -> None:
    """Save UTF-8 plain text."""

    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(text, encoding='utf-8')


def append_csv_row(path: Path | str, fieldnames: Sequence[str], row: dict[str, Any]) -> None:
    """Append one CSV row, creating the header when necessary."""

    path = Path(path)
    ensure_dir(path.parent)
    write_header = not path.exists()
    with path.open('a', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_csv(path: Path | str, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    """Write a full CSV file from rows of dictionaries."""

    path = Path(path)
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_mapping_csv(path: Path | str, payload: Mapping[str, Any], *, key_name: str = 'key', value_name: str = 'value') -> None:
    """Save a small mapping as two-column CSV with compact YAML-encoded nested values."""

    rows = [{key_name: str(key), value_name: compact_yaml(value)} for key, value in payload.items()]
    save_csv(path, [key_name, value_name], rows)


def save_key_value_csv(path: Path | str, payload: Mapping[str, Any], *, key_name: str = 'key', value_name: str = 'value') -> None:
    """Alias for two-column metadata CSV output."""

    save_mapping_csv(path, payload, key_name=key_name, value_name=value_name)


def read_csv_rows(path: Path | str) -> list[dict[str, str]]:
    """Read one CSV file into memory."""

    with Path(path).open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


__all__ = [
    'MANIFEST_FILENAMES',
    'STRUCTURED_SUFFIXES',
    'YAML_SUFFIXES',
    'append_csv_row',
    'compact_yaml',
    'ensure_dir',
    'load_manifest',
    'load_structured',
    'load_yaml',
    'manifest_path_for_dir',
    'read_csv_rows',
    'resolve_manifest_path',
    'resolve_structured_path',
    'save_csv',
    'save_key_value_csv',
    'save_manifest',
    'save_mapping_csv',
    'save_structured',
    'save_text',
    'save_yaml',
    'to_jsonable',
]
