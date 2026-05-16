"""CLI driver for independent paper-experiment PSD reinterpretation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.reinterpretation.dh_snn import CASE_SPEC as DH_SNN_CASE
from src.reinterpretation.drf import CASE_SPEC as DRF_CASE
from src.reinterpretation.need_high import CASE_SPEC as NEED_HIGH_CASE
from src.util.cli_common import ensure_absolute_path, normalize_userbin_edges, parse_bool_token
from src.util.config import ensure_dir, save_json
from src.util.csv_schema import common_row, write_common_csv
from src.util.early_seed import ensure_entrypoint_deterministic_env

ensure_entrypoint_deterministic_env()

_CASES = {
    'need_high': NEED_HIGH_CASE,
    'drf': DRF_CASE,
    'dh_snn': DH_SNN_CASE,
}


def _parse_gpu_map(text: str) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for item in str(text).split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise ValueError(f'gpu_map item must be experiment_id:gpu_index, got {item!r}.')
        key, value = [part.strip() for part in item.split(':', 1)]
        if key not in _CASES:
            raise ValueError(f'Unknown reinterpretation experiment id in gpu_map: {key!r}.')
        gpu_index = int(value)
        if gpu_index < 0:
            raise ValueError(f'gpu_index must be >= 0 for {key!r}.')
        mapping[key] = gpu_index
    return mapping


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run fixed paper-experiment PSD reinterpretation hooks.')
    parser.add_argument('--run_need_high', default='false')
    parser.add_argument('--run_drf', default='false')
    parser.add_argument('--run_dh_snn', default='false')
    parser.add_argument('--gpu_map', required=True, help='Comma-separated experiment_id:gpu_index entries, e.g. need_high:0,drf:1,dh_snn:0.')
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--log_root', required=True)
    parser.add_argument('--userbin_edges', nargs='*', type=float, default=None)
    parser.add_argument('--seed_bundle', default='0')
    return parser


def _enabled_cases(args: argparse.Namespace) -> list[str]:
    flags = {
        'need_high': parse_bool_token(args.run_need_high, default=False),
        'drf': parse_bool_token(args.run_drf, default=False),
        'dh_snn': parse_bool_token(args.run_dh_snn, default=False),
    }
    return [case_id for case_id, enabled in flags.items() if enabled]


def _case_metadata(case: dict[str, Any], *, args: argparse.Namespace, gpu_index: int) -> dict[str, Any]:
    return {
        **case,
        'gpu_index': int(gpu_index),
        'psd_config': {
            'estimator': 'full_length_periodogram',
            'frequency_unit': 'normalized_frequency',
            'userbin_edges': normalize_userbin_edges(args.userbin_edges),
            'scales': ['raw', 'db'],
        },
        'csv_output_policy': 'category_csv_per_artifact',
        'plot_output_policy': 'external_plotting_stage',
        'figure_rendering_policy': 'disabled_in_reinterpretation_driver',
        'seed_bundle': str(args.seed_bundle),
        'source_optimizer_family': 'author_code_fixed_setting',
        'project_side_optimizer_override': None,
        'checkpoint_policy': 'disabled',
        'raw_state_archive_policy': 'disabled',
        'driver_scope': 'fixed paper experiment plus PSD observation hooks',
    }


def _case_common_base(case_id: str, case: dict[str, Any], *, args: argparse.Namespace, gpu_index: int) -> dict[str, Any]:
    dataset_token = str(case.get('dataset', '')).strip().lower().replace('_', '-')
    return {
        'source_program': 'reinterpretation',
        'run_id': f'{case_id}_seed_bundle_{args.seed_bundle}',
        'status': 'pending_author_hook',
        'dataset': dataset_token,
        'seed': str(args.seed_bundle),
        'model_family': str(case_id),
        'scope': 'paper_experiment',
        'message': 'author-code hook execution must populate numeric values without changing paper settings',
    }


def _hook_status_rows(case_id: str, case: dict[str, Any], *, args: argparse.Namespace, gpu_index: int, artifact_name: str) -> list[dict[str, str]]:
    base = _case_common_base(case_id, case, args=args, gpu_index=int(gpu_index))
    rows: list[dict[str, str]] = []
    scale_values = ('raw', 'db') if 'curve' in str(artifact_name) else ('',)
    for family in case['hook_families']:
        for scale in scale_values:
            kwargs = dict(base)
            kwargs.update(
                category='reinterpretation_metric',
                experiment_id=case_id,
                signal_kind=str(family),
                scale=scale,
                metric='hook_family_status',
                statistic='pending',
            )
            rows.append(common_row(**kwargs))
    return rows


def _case_metric_rows(case_id: str, case: dict[str, Any], *, args: argparse.Namespace, gpu_index: int) -> list[dict[str, str]]:
    base = _case_common_base(case_id, case, args=args, gpu_index=int(gpu_index))
    rows = []
    kwargs = dict(base)
    kwargs.update(
        category='reinterpretation_metric',
        experiment_id=case_id,
        status='ok',
        message='',
        metric='gpu_index',
        value=float(gpu_index),
    )
    rows.append(common_row(**kwargs))
    kwargs = dict(base)
    kwargs.update(
        category='reinterpretation_metric',
        experiment_id=case_id,
        metric='author_forward_status',
        statistic='pending',
    )
    rows.append(common_row(**kwargs))
    return rows


def _write_case_outputs(case_id: str, case: dict[str, Any], *, args: argparse.Namespace, gpu_index: int) -> None:
    case_root = ensure_dir(Path(args.output_root) / 'reinterpretation' / case_id)
    metadata = _case_metadata(case, args=args, gpu_index=int(gpu_index))
    save_json(case_root / 'metadata.json', metadata)

    metrics_dir = ensure_dir(case_root / 'metrics')
    dataset_dir = ensure_dir(case_root / 'dataset_psd')
    analysis_dir = ensure_dir(case_root / 'psd_analysis')

    write_common_csv(metrics_dir / 'missing_family_table.csv', _hook_status_rows(case_id, case, args=args, gpu_index=int(gpu_index), artifact_name='missing_family_table'))
    write_common_csv(metrics_dir / 'case_specific_metric_table.csv', _case_metric_rows(case_id, case, args=args, gpu_index=int(gpu_index)))
    write_common_csv(dataset_dir / 'dataset_input_curve.csv', _hook_status_rows(case_id, case, args=args, gpu_index=int(gpu_index), artifact_name='dataset_input_curve'))
    write_common_csv(analysis_dir / 'hook_family_curve.csv', _hook_status_rows(case_id, case, args=args, gpu_index=int(gpu_index), artifact_name='hook_family_curve'))


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.output_root = ensure_absolute_path(args.output_root, arg_name='output_root')
    args.log_root = ensure_absolute_path(args.log_root, arg_name='log_root')
    normalize_userbin_edges(args.userbin_edges)
    gpu_map = _parse_gpu_map(args.gpu_map)
    enabled = _enabled_cases(args)
    if not enabled:
        raise ValueError('At least one reinterpretation run flag must be true.')
    for case_id in enabled:
        if case_id not in gpu_map:
            raise ValueError(f'Enabled experiment {case_id!r} is missing from gpu_map.')
        _write_case_outputs(case_id, _CASES[case_id], args=args, gpu_index=gpu_map[case_id])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
