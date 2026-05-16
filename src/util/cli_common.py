"""Shared CLI parsing helpers for experiment entrypoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

_TRUE_TOKENS = {'1', 'true', 't', 'yes', 'y', 'on'}
_FALSE_TOKENS = {'0', 'false', 'f', 'no', 'n', 'off'}
_DEFAULT_USERBIN_EDGES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def parse_bool_token(value: str | bool | int | None, *, default: bool = False) -> bool:
    """Parse project boolean flags from common shell and argparse spellings."""

    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    token = str(value).strip().lower()
    if token == '':
        return bool(default)
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from '{value}'.")


def ensure_absolute_path(path: str | Path, *, arg_name: str) -> Path:
    """Validate that one CLI path is absolute and return it as ``Path``."""

    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        raise ValueError(f"{arg_name} must be an absolute path, got '{path}'.")
    return resolved.resolve()


def normalize_userbin_edges(values: Sequence[float] | None) -> list[float]:
    """Return validated normalized-frequency userbin edges."""

    if values is None or len(values) == 0:
        edges = list(_DEFAULT_USERBIN_EDGES)
    else:
        edges = [float(v) for v in values]
    if len(edges) < 2:
        raise ValueError('userbin_edges must contain at least two increasing values.')
    if any(right <= left for left, right in zip(edges[:-1], edges[1:])):
        raise ValueError('userbin_edges must be strictly increasing.')
    if edges[0] != 0.0 or edges[-1] != 0.5:
        raise ValueError('userbin_edges must start at 0.0 and end at 0.5.')
    return edges


__all__ = [
    'ensure_absolute_path',
    'normalize_userbin_edges',
    'parse_bool_token',
]
