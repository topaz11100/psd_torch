"""Early-process deterministic environment helpers for CLI entrypoints."""

from __future__ import annotations

import os
import sys
from typing import Sequence


_DEFAULT_CUBLAS_WORKSPACE_CONFIG = ':4096:8'


def _parse_seed_from_argv(argv: Sequence[str]) -> str:
    """Internal helper that parse seed from argv."""
    argv = list(argv)
    for index, token in enumerate(argv):
        if token == '--seed' and index + 1 < len(argv):
            return str(argv[index + 1])
        if token.startswith('--seed='):
            return str(token.split('=', 1)[1])
    return '0'


def _reexec_argv() -> list[str]:
    """Internal helper for ``reexec argv`` in the ``early_seed`` module."""
    orig = list(getattr(sys, 'orig_argv', []))
    if orig:
        return [sys.executable, *orig[1:]]
    return [sys.executable, *sys.argv]


def ensure_entrypoint_deterministic_env(argv: Sequence[str] | None = None) -> None:
    """Re-exec the current Python CLI so seed-dependent env vars are present from process start."""

    argv = sys.argv if argv is None else argv
    desired_seed = _parse_seed_from_argv(argv)
    current_hash_seed = os.environ.get('PYTHONHASHSEED')
    current_cublas = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if current_hash_seed == desired_seed and current_cublas is not None:
        return

    new_env = dict(os.environ)
    new_env['PYTHONHASHSEED'] = desired_seed
    new_env.setdefault('CUBLAS_WORKSPACE_CONFIG', _DEFAULT_CUBLAS_WORKSPACE_CONFIG)
    os.execve(sys.executable, _reexec_argv(), new_env)


__all__ = ['ensure_entrypoint_deterministic_env']
