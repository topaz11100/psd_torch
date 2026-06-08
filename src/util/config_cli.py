"""CLI용 YAML 설정 로더와 병합 도우미."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

from src.util.config import load_structured


def _argv_list(argv: Sequence[str] | None) -> list[str]:
    """명시 인자가 없으면 실제 CLI 인자(sys.argv[1:])를 사용한다."""

    return list(sys.argv[1:] if argv is None else argv)


def extract_config_path(argv: Sequence[str] | None) -> str | None:
    """argv에서 --config 경로를 추출한다."""

    args = _argv_list(argv)
    for index, token in enumerate(args):
        if token == '--config' and index + 1 < len(args):
            return str(args[index + 1])
        if token.startswith('--config='):
            return str(token.split('=', 1)[1])
    return None


def _flatten_config_payload(payload: dict[str, Any], *, path: str = '') -> dict[str, Any]:
    """Flatten a nested stage config by leaf argument name.

    Public YAML files are intentionally grouped by semantic blocks such as
    ``data:``, ``model:``, ``regularization:``, and ``runtime:``.  Existing
    entrypoints still use argparse destination names.  This helper preserves the
    cleaner YAML hierarchy while keeping CLI compatibility by mapping each leaf
    key to its argparse destination.  Duplicate leaf names are rejected because
    they would make the resolved command ambiguous.
    """

    flattened: dict[str, Any] = {}

    def walk(mapping: dict[str, Any], prefix: str) -> None:
        for key, value in mapping.items():
            name = str(key)
            child_path = f'{prefix}.{name}' if prefix else name
            if isinstance(value, dict):
                walk(value, child_path)
                continue
            if name in flattened:
                raise ValueError(f'--config leaf key is duplicated after flattening: {name!r} at {child_path!r}')
            flattened[name] = value

    walk(payload, path)
    return flattened


def load_config_dict(config_path: str, *, stage_key: str | None = None) -> dict[str, Any]:
    """YAML 설정 파일을 읽고 stage 내부의 중첩 구조를 CLI 인수 dict로 평탄화한다."""

    path = Path(config_path).expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix not in {'.yaml', '.yml'}:
        raise ValueError(f'--config 파일 확장자는 .yaml/.yml만 허용됩니다: {path}')
    payload = load_structured(path)
    if not isinstance(payload, dict):
        raise ValueError(f'--config 루트는 객체(dict)여야 합니다: {path}')
    if stage_key and stage_key in payload:
        stage_payload = payload[stage_key]
        if not isinstance(stage_payload, dict):
            raise ValueError(f'--config의 {stage_key!r} 값은 객체(dict)여야 합니다: {path}')
        payload = stage_payload
    return _flatten_config_payload(dict(payload), path=stage_key or '')



def parse_args_with_config(
    parser: argparse.ArgumentParser,
    *,
    argv: Sequence[str] | None,
    stage_key: str | None = None,
) -> argparse.Namespace:
    """YAML 설정을 반영한 뒤 인자를 파싱하고 필수 누락을 한국어로 검증한다."""

    required_dests = [a.dest for a in parser._actions if getattr(a, 'required', False)]
    resolved_argv = _argv_list(argv)
    config_path = extract_config_path(resolved_argv)
    if config_path:
        loaded = load_config_dict(config_path, stage_key=stage_key)
        action_by_dest = {action.dest: action for action in parser._actions}
        known = set(action_by_dest)
        unknown = sorted(key for key in loaded.keys() if key not in known)
        if unknown:
            raise ValueError(f'--config에 알 수 없는 키가 있습니다: {", ".join(unknown)}')

        def is_blank_template_value(value: Any) -> bool:
            if value is None or value == '':
                return True
            if isinstance(value, (list, tuple)) and all(item in (None, '') for item in value):
                return True
            return False

        # Clean templates intentionally use blank values.  Do not pass those
        # blanks into argparse because typed actions would try to convert ""
        # and fail before the project can report missing required settings.
        resolved_defaults = {key: value for key, value in loaded.items() if not is_blank_template_value(value)}
        parser.set_defaults(**resolved_defaults)
    for action in parser._actions:
        if getattr(action, 'required', False):
            action.required = False
    args = parser.parse_args(resolved_argv)
    missing: list[str] = []
    for dest in required_dests:
        value = getattr(args, dest, None)
        if value is None:
            missing.append(dest)
    if missing:
        raise ValueError(f'필수 인자가 누락되었습니다: {", ".join(sorted(missing))}. --config 또는 CLI 인자를 확인하세요.')
    return args
