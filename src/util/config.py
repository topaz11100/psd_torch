"""Small filesystem and serialization helpers shared across the project."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def ensure_dir(path: Path | str) -> Path:
    """Create one directory when missing and return it as ``Path``."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


class _NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that gracefully handles NumPy, Torch, and ``Path`` objects."""

    def default(self, obj: Any) -> Any:
        """Handle ``default`` for the ``config`` module."""
        try:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
        except Exception:
            pass
        try:
            import torch

            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().tolist()
        except Exception:
            pass
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_json(path: Path | str, payload: dict[str, Any] | list[Any], *, indent: int = 2) -> None:
    """Save one JSON payload using UTF-8."""

    path = Path(path)
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=indent, cls=_NumpyJSONEncoder)
        handle.write('\n')


def load_json(path: Path | str) -> Any:
    """Load one JSON payload."""

    with Path(path).open('r', encoding='utf-8') as handle:
        return json.load(handle)


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


def read_csv_rows(path: Path | str) -> list[dict[str, str]]:
    """Read one CSV file into memory."""

    with Path(path).open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


__all__ = [
    'append_csv_row',
    'ensure_dir',
    'load_json',
    'read_csv_rows',
    'save_csv',
    'save_json',
    'save_text',
]
