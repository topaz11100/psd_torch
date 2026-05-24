"""CLI 시작 초기에 필요한 시드 환경만 맞추는 헬퍼."""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Sequence


def _parse_seed_from_argv(argv: Sequence[str]) -> str:
    """argv에서 시드를 읽고, 없으면 --config의 seed를 보조로 사용한다."""
    argv = list(argv)
    config_path: str | None = None
    stage_key = 'data_prep'
    for index, token in enumerate(argv):
        if token == '--seed' and index + 1 < len(argv):
            return str(argv[index + 1])
        if token.startswith('--seed='):
            return str(token.split('=', 1)[1])
        if token == '--config' and index + 1 < len(argv):
            config_path = str(argv[index + 1])
        if token.startswith('--config='):
            config_path = str(token.split('=', 1)[1])
    if config_path:
        try:
            path = Path(config_path)
            if path.suffix.lower() == '.json' and path.exists():
                payload = json.loads(path.read_text(encoding='utf-8'))
                if isinstance(payload, dict):
                    if stage_key in payload and isinstance(payload[stage_key], dict) and 'seed' in payload[stage_key]:
                        return str(payload[stage_key]['seed'])
                    if 'seed' in payload:
                        return str(payload['seed'])
        except Exception:
            pass
    return '0'


def _reexec_argv() -> list[str]:
    """Internal helper for ``reexec argv`` in the ``early_seed`` module."""
    orig = list(getattr(sys, 'orig_argv', []))
    if orig:
        return [sys.executable, *orig[1:]]
    return [sys.executable, *sys.argv]


def ensure_entrypoint_deterministic_env(argv: Sequence[str] | None = None) -> None:
    """프로세스 시작 시점에 필요한 해시 시드만 보장하기 위해 재실행한다."""

    argv = sys.argv if argv is None else argv
    desired_seed = _parse_seed_from_argv(argv)
    current_hash_seed = os.environ.get('PYTHONHASHSEED')
    if current_hash_seed == desired_seed:
        return

    new_env = dict(os.environ)
    new_env['PYTHONHASHSEED'] = desired_seed
    os.execve(sys.executable, _reexec_argv(), new_env)


__all__ = ['ensure_entrypoint_deterministic_env']
