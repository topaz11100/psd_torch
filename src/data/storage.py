"""Official structured prepared-split I/O helpers."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


SINGLE_STRUCTURED_NPY_STORAGE_FORMAT = 'single_structured_npy_v1'


def _atomic_tmp_path(path: Path) -> Path:
    """Return one sibling temporary path suitable for atomic replacement."""

    return path.parent / f'.{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}'


def fsync_path(path: Path | str) -> None:
    """Best-effort fsync for one existing file path."""

    resolved = Path(path).expanduser().resolve()
    with resolved.open('rb+') as handle:
        os.fsync(handle.fileno())


def npy_load_mmap(path: Path | str, *, mmap_mode: str = 'c') -> np.ndarray:
    """Load one ``.npy`` array with NumPy memory mapping enabled by default."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f'Prepared array artifact is missing: {resolved}')
    return np.load(resolved, mmap_mode=mmap_mode, allow_pickle=False)


def validate_single_structured_split_array(records: np.ndarray, *, path: Path | str | None = None) -> None:
    """Validate one structured split mmap/payload against the single-file contract."""

    location = '' if path is None else f' at {Path(path).expanduser().resolve()}'
    if not isinstance(records, np.ndarray):
        raise TypeError(f'Structured split payload{location} must be a NumPy array.')
    if records.ndim != 1:
        raise ValueError(f'Structured split payload{location} must have shape (N,), got {tuple(records.shape)}.')
    fields = records.dtype.fields
    if fields is None or not isinstance(fields, Mapping):
        raise ValueError(f'Structured split payload{location} must use a structured dtype.')
    required = {'sample_index', 'label', 'input'}
    missing = required.difference(fields.keys())
    if missing:
        raise ValueError(f'Structured split payload{location} missing fields: {sorted(missing)}.')
    sample_index_dtype = np.dtype(fields['sample_index'][0])
    label_dtype = np.dtype(fields['label'][0])
    input_dtype = np.dtype(fields['input'][0])
    if sample_index_dtype != np.dtype(np.int64):
        raise ValueError(f'Structured split payload{location} must store sample_index as int64; got {sample_index_dtype}.')
    if label_dtype != np.dtype(np.int64):
        raise ValueError(f'Structured split payload{location} must store label as int64; got {label_dtype}.')
    if input_dtype == np.dtype(object):
        raise ValueError(f'Structured split payload{location} must not store object-dtype inputs.')


def save_single_structured_split_atomic(records: Any, path: Path | str) -> Path:
    """Atomically save one already-structured split payload as a ``.npy`` file.

    This helper intentionally refuses generic iterables/lists so it cannot hide
    split-level materialization behind ``np.asarray``. Official ``data_prep``
    writers use ``open_memmap`` and record-wise assignment directly.
    """

    if not isinstance(records, np.ndarray):
        raise TypeError(
            'save_single_structured_split_atomic requires an existing NumPy structured array or memmap; '
            'official data_prep writers must stream records directly with open_memmap.'
        )
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _atomic_tmp_path(resolved).with_suffix('.npy')
    try:
        validate_single_structured_split_array(records)
        writer = np.lib.format.open_memmap(tmp_path, mode='w+', dtype=records.dtype, shape=records.shape)
        for index in range(int(records.shape[0])):
            writer[index] = records[index]
        writer.flush()
        del writer
        fsync_path(tmp_path)
        validate_single_structured_split_array(npy_load_mmap(tmp_path, mmap_mode='r'), path=tmp_path)
        os.replace(tmp_path, resolved)
        return resolved
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


def load_single_structured_split(path: Path | str, *, mmap_mode: str = 'r') -> np.ndarray:
    """Load one single-structured prepared split ``.npy`` payload."""

    records = npy_load_mmap(path, mmap_mode=mmap_mode)
    validate_single_structured_split_array(records, path=path)
    return records


__all__ = [
    'SINGLE_STRUCTURED_NPY_STORAGE_FORMAT',
    'fsync_path',
    'load_single_structured_split',
    'npy_load_mmap',
    'save_single_structured_split_atomic',
    'validate_single_structured_split_array',
]
