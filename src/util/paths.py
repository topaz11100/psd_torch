"""Run-directory naming helpers shared by output-producing entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo


_SEOUL_TZ = ZoneInfo('Asia/Seoul')
_TRUE_TOKENS = {'1', 'true', 't', 'yes', 'y', 'on'}
_FALSE_TOKENS = {'0', 'false', 'f', 'no', 'n', 'off'}
_RUN_LEAF_NAMES = {'checkpoints', 'metrics', 'train'}


def sanitize_token(text: Any) -> str:
    """Convert a free-form token into a filesystem-safe fragment."""

    safe: list[str] = []
    for ch in str(text).strip():
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


def make_timestamp(explicit: Any | None = None) -> str:
    """Return ``explicit`` when provided, otherwise a compact Asia/Seoul timestamp."""

    if explicit is not None and str(explicit).strip() != '':
        return sanitize_token(explicit)
    # Include microseconds so fast repeated launches do not collide while keeping
    # lexical order equal to execution-time order.
    return datetime.now(_SEOUL_TZ).strftime('%Y%m%d_%H%M%S_%f')


def parse_timestamped_output(value: Any, *, default: bool = True) -> bool:
    """Parse the public timestamped-output switch."""

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
    raise ValueError(f"Cannot parse timestamped_output from {value!r}.")


def add_timestamped_output_args(parser: Any) -> Any:
    """Add common output timestamp CLI arguments to an argparse parser."""

    known = {getattr(action, 'dest', None) for action in getattr(parser, '_actions', [])}
    if 'timestamped_output' not in known:
        parser.add_argument(
            '--timestamped_output',
            default='true',
            help='true이면 실제 산출물을 output/checkpoint/metric root 아래 실행시각 run_<timestamp> 폴더에 저장한다.',
        )
    if 'run_timestamp' not in known:
        parser.add_argument(
            '--run_timestamp',
            default=None,
            help='자동 실행시각 대신 사용할 timestamp token. DDP에서는 rank0 값이 전체 rank에 공유된다.',
        )
    return parser


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


def timestamped_output_path(
    path: Path | str,
    *,
    timestamp: Any | None = None,
    enabled: Any = True,
    leaf_names: Iterable[str] | None = None,
    prefix: str = 'run',
) -> Path:
    """Return the actual output path, optionally nested under a run timestamp.

    For ordinary output roots, ``/result/stage/case`` becomes
    ``/result/stage/case/run_YYYYmmdd_HHMMSS_ffffff``.

    For training roots whose final component is one of ``checkpoints``,
    ``metrics`` or ``train``, the timestamp is inserted above that leaf so the
    related folders share one run directory:
    ``/result/case/checkpoints`` -> ``/result/case/run_<ts>/checkpoints``.
    """

    base = Path(path).expanduser().resolve()
    if not parse_timestamped_output(enabled, default=True):
        return base
    token = make_timestamp(timestamp)
    run_leaf = f'{sanitize_token(prefix)}_{token}' if prefix else token
    leaves = {str(v) for v in (_RUN_LEAF_NAMES if leaf_names is None else leaf_names)}
    if base.name in leaves:
        return (base.parent / run_leaf / base.name).resolve()
    return (base / run_leaf).resolve()


def timestamped_output_root(
    base_output_root: Path | str,
    *,
    run_timestamp: Any | None = None,
    prefix: str = 'run',
    enabled: Any = True,
) -> Path:
    """Backward-compatible wrapper for output roots using a timestamped child folder."""

    return timestamped_output_path(
        base_output_root,
        timestamp=run_timestamp,
        enabled=enabled,
        leaf_names=(),
        prefix=prefix,
    )


__all__ = [
    'RunNameParts',
    'add_timestamped_output_args',
    'make_run_root',
    'make_timestamp',
    'parse_timestamped_output',
    'sanitize_token',
    'timestamped_output_path',
    'timestamped_output_root',
]
