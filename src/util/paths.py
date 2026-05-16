"""Run-directory naming helpers shared by ``dataset_psd`` and ``psd_analysis``."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


_SEOUL_TZ = ZoneInfo('Asia/Seoul')


def sanitize_token(text: str) -> str:
    """Convert a free-form token into a filesystem-safe fragment."""

    safe: list[str] = []
    for ch in text.strip():
        if ch.isalnum() or ch in {'-', '_', '.'}:
            safe.append(ch)
        else:
            safe.append('-')
    token = ''.join(safe).strip('-')
    return token or 'run'


@dataclass(frozen=True)
class RunNameParts:
    """Structured components used for run-root naming."""

    dataset: str
    model: str | None = None
    readout: str | None = None
    experiment: str | None = None
    timestamp: str | None = None


def make_timestamp(explicit: str | None = None) -> str:
    """Return ``explicit`` when provided, otherwise a compact Asia/Seoul timestamp."""

    if explicit:
        return sanitize_token(explicit)
    return datetime.now(_SEOUL_TZ).strftime('%Y%m%d_%H%M%S')


def make_run_root(base_output_root: Path | str, parts: RunNameParts) -> Path:
    """Construct a run root for a single dataset/model/readout experiment."""

    base = Path(base_output_root)
    fragments: list[str] = []
    if parts.experiment:
        fragments.append(sanitize_token(parts.experiment))
    fragments.append(sanitize_token(parts.dataset))
    if parts.model:
        fragments.append(sanitize_token(parts.model))
    if parts.readout:
        fragments.append(sanitize_token(parts.readout))
    fragments.append(make_timestamp(parts.timestamp))
    return base.joinpath('__'.join(fragments))


__all__ = ['RunNameParts', 'make_run_root', 'make_timestamp', 'sanitize_token']
