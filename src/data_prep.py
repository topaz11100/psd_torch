"""Official CLI entrypoint for prepared dataset bundle generation."""

from __future__ import annotations

import sys
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

ensure_entrypoint_deterministic_env()

import argparse

from src.data.specs import available_dataset_tokens
from src.util.cli_common import ensure_absolute_path, parse_bool_token


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Preprocess raw datasets into canonical prepared single-file mmap bundles.')
    parser.add_argument('--dataset', required=True, help=f"Dataset token. Registered canonical datasets: {', '.join(available_dataset_tokens())}")
    parser.add_argument('--raw_data_root', required=True, help='Absolute raw-data root path.')
    parser.add_argument('--prep_root', required=True, help='Absolute prepared-bundle root path.')
    parser.add_argument('--seed', type=int, default=0, help='Global preprocessing seed recorded in manifest metadata.')
    parser.add_argument('--force_overwrite', default='false', help='Overwrite an existing prepared bundle for the selected dataset.')
    return parser


def _validate_args(args: argparse.Namespace) -> argparse.Namespace:
    args.raw_data_root = ensure_absolute_path(args.raw_data_root, arg_name='raw_data_root')
    args.prep_root = ensure_absolute_path(args.prep_root, arg_name='prep_root')
    args.force_overwrite = parse_bool_token(args.force_overwrite, default=False)
    return args


def main() -> int:
    parser = build_arg_parser()
    args = _validate_args(parser.parse_args())

    from src.data.preprocessing import prepare_dataset_bundle

    out_dir = prepare_dataset_bundle(
        args.dataset,
        raw_data_root=args.raw_data_root,
        prep_root=args.prep_root,
        seed=int(args.seed),
        overwrite=bool(args.force_overwrite),
        deap_label_axis='valence',
        deap_num_classes=3,
        prep_profile=None,
    )
    print(str(out_dir))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
