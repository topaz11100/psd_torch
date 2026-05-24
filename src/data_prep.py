"""Official CLI entrypoint for prepared dataset bundle generation."""

from __future__ import annotations

import sys
import os
from pathlib import Path

if __package__ is None or __package__ == '':
    _SCRIPT_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _SCRIPT_DIR.parent
    try:
        sys.path.remove(str(_SCRIPT_DIR))
    except ValueError:
        pass
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from src.util.early_seed import ensure_entrypoint_deterministic_env

import argparse

from src.data.specs import available_dataset_tokens
from src.util.cli_common import ensure_absolute_path, parse_bool_token
from src.util.config_cli import parse_args_with_config


def build_arg_parser() -> argparse.ArgumentParser:
    from src.data.prep_profiles import available_prep_profiles
    parser = argparse.ArgumentParser(description='Preprocess raw datasets into canonical prepared single-file mmap bundles.')
    parser.add_argument('--dataset', required=True, help=f"Dataset token. Registered canonical datasets: {', '.join(available_dataset_tokens())}")
    parser.add_argument('--raw_data_root', required=True, help='Absolute raw-data root path.')
    parser.add_argument('--prep_root', required=True, help='Absolute prepared-bundle root path.')
    parser.add_argument('--seed', type=int, default=0, help='Global preprocessing seed recorded in manifest metadata.')
    parser.add_argument('--config', default=None, help='JSON 설정 파일 경로(.json)')
    parser.add_argument('--force_overwrite', default='false', help='Overwrite an existing prepared bundle for the selected dataset.')
    parser.add_argument('--download', default='false', help='원본 데이터 자동 다운로드 여부.')
    parser.add_argument('--max_samples', type=int, default=None, help='split별 최대 샘플 수(양의 정수 또는 생략).')
    parser.add_argument('--prep_profile', default=None, help=f"전처리 프로필. 사용 가능: {', '.join(available_prep_profiles())}")
    parser.add_argument('--deap_label_axis', default='valence', choices=('valence', 'arousal'))
    parser.add_argument('--deap_num_classes', type=int, default=3, choices=(2, 3))
    parser.add_argument('--shd_dt_ms', type=float, default=1.0)
    parser.add_argument('--shd_max_time', type=float, default=1.2)
    parser.add_argument('--ssc_dt_ms', type=float, default=1.0)
    parser.add_argument('--ssc_max_time', type=float, default=1.0)
    return parser


def _validate_args(args: argparse.Namespace) -> argparse.Namespace:
    args.raw_data_root = ensure_absolute_path(args.raw_data_root, arg_name='raw_data_root')
    args.prep_root = ensure_absolute_path(args.prep_root, arg_name='prep_root')
    args.force_overwrite = parse_bool_token(args.force_overwrite, default=False)
    args.download = parse_bool_token(args.download, default=False)
    if args.max_samples is not None and int(args.max_samples) <= 0:
        raise ValueError('max_samples는 양의 정수 또는 null이어야 합니다.')
    for key in ('shd_dt_ms', 'shd_max_time', 'ssc_dt_ms', 'ssc_max_time'):
        if float(getattr(args, key)) <= 0.0:
            raise ValueError(f'{key}는 양수여야 합니다.')
    return args


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--help" not in argv and "-h" not in argv and "PYTEST_CURRENT_TEST" not in os.environ and not any("pytest" in str(token) for token in argv):
        ensure_entrypoint_deterministic_env()
    parser = build_arg_parser()
    args = _validate_args(parse_args_with_config(parser, argv=argv, stage_key='data_prep'))

    from src.data.preprocessing import prepare_dataset_bundle

    out_dir = prepare_dataset_bundle(
        args.dataset,
        raw_data_root=args.raw_data_root,
        prep_root=args.prep_root,
        seed=int(args.seed),
        overwrite=bool(args.force_overwrite),
        download=bool(args.download),
        deap_label_axis=str(args.deap_label_axis),
        deap_num_classes=int(args.deap_num_classes),
        shd_dt_ms=float(args.shd_dt_ms),
        shd_max_time=float(args.shd_max_time),
        ssc_dt_ms=float(args.ssc_dt_ms),
        ssc_max_time=float(args.ssc_max_time),
        max_samples=args.max_samples,
        prep_profile=args.prep_profile,
    )
    print(str(out_dir))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
