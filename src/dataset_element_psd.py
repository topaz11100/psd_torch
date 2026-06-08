
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == '':
    _SCRIPT_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _SCRIPT_DIR.parent
    try:
        sys.path.remove(str(_SCRIPT_DIR))
    except ValueError:
        pass
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from src.util.config import compact_yaml, load_structured, save_yaml
from src.util.config_cli import parse_args_with_config
from src.util.csv_schema import common_row, write_common_csv, write_manifest_yaml
from src.util.paths import timestamped_output_root


def _load_runtime_dependencies() -> None:
    global torch, ConcatDataset, dataset_for_view, make_loader, resolve_dataset_bundle, seed_everything, normalize_signal_window
    import torch as _torch
    from torch.utils.data import ConcatDataset as _ConcatDataset
    from src.data.registry import dataset_for_view as _dataset_for_view
    from src.data.registry import make_loader as _make_loader
    from src.data.registry import resolve_dataset_bundle as _resolve_dataset_bundle
    from src.signal.psd_utils import normalize_signal_window as _normalize_signal_window
    from src.util.random import seed_everything as _seed_everything
    torch = _torch
    ConcatDataset = _ConcatDataset
    dataset_for_view = _dataset_for_view
    make_loader = _make_loader
    resolve_dataset_bundle = _resolve_dataset_bundle
    seed_everything = _seed_everything
    normalize_signal_window = _normalize_signal_window


def _as_list(value: Any, default: Sequence[str]) -> list[str]:
    if value is None or value == '':
        return [str(v) for v in default]
    if isinstance(value, (list, tuple)):
        values = [str(v).strip() for v in value if str(v).strip()]
        return values or [str(v) for v in default]
    text = str(value).strip()
    if not text:
        return [str(v) for v in default]
    if ',' in text:
        return [chunk.strip() for chunk in text.split(',') if chunk.strip()]
    return [text]


def _boolish(value: Any, *, default: bool = False) -> bool:
    if value is None or value == '':
        return bool(default)
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if token in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise ValueError(f'Cannot parse boolean value: {value!r}')


def _require_device(device_name: str, gpu_index: int):
    token = str(device_name or 'cuda').strip().lower()
    if token != 'cuda':
        raise ValueError('dataset_element analysis is configured for cuda-only execution in this project.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for dataset_element analysis.')
    index = int(gpu_index)
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(f'gpu_index={index} is invalid for {torch.cuda.device_count()} CUDA device(s).')
    torch.cuda.set_device(index)
    return torch.device(f'cuda:{index}')


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = load_structured(path)
    if not isinstance(payload, dict):
        raise ValueError(f'Prepared manifest must be a mapping: {path}')
    return dict(payload)


def _axis_metadata_columns(manifest: Mapping[str, Any], *, psd_axis_kind: str) -> dict[str, Any]:
    logical_shape = manifest.get('psd_logical_shape')
    static_repeat_t = manifest.get('static_repeat_T')
    return {
        'prep_profile': str(manifest.get('prep_profile', manifest.get('psd_axis_kind', psd_axis_kind))),
        'psd_axis_kind': str(manifest.get('psd_axis_kind', psd_axis_kind)),
        'psd_time_axis': manifest.get('psd_time_axis', ''),
        'psd_row_axes': compact_yaml(manifest.get('psd_row_axes', [])),
        'psd_flatten_rule': str(manifest.get('psd_flatten_rule', '')), 
        'psd_logical_shape': compact_yaml(logical_shape if logical_shape is not None else []),
        'static_repeat_T': '' if static_repeat_t in (None, '') else int(static_repeat_t),
    }


def _safe_token(value: Any) -> str:
    token = str(value).strip().lower().replace('-', '_')
    token = ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in token)
    token = '_'.join(part for part in token.split('_') if part)
    return token or 'value'


def _maps_from_batch(inputs: Any, *, device: Any, dtype: Any) -> Any:
    x = torch.as_tensor(inputs, dtype=dtype, device=device)
    if x.ndim == 2:
        # (B,T) -> one row with T samples.
        return x.unsqueeze(1).contiguous()
    if x.ndim == 3:
        # (B,T,F) -> (B,F,T).
        return x.transpose(1, 2).contiguous()
    if x.ndim >= 4:
        # (B,T,...) -> flatten all non-time axes into rows.
        return x.reshape(int(x.shape[0]), int(x.shape[1]), -1).transpose(1, 2).contiguous()
    raise ValueError(f'Expected input batch with rank >=2, got shape {tuple(x.shape)}.')


def _centered(maps: Any) -> Any:
    return maps - maps.mean(dim=-1, keepdim=True)


def _variant_maps(maps: Any, variant: str) -> Any:
    token = str(variant).strip().lower()
    if token == 'raw':
        return maps
    if token in {'centered', 'demean', 'demeaned'}:
        return _centered(maps)
    raise ValueError(f'Unsupported variant: {variant!r}')


def _scale_psd(power: Any, scale: str, eps: float = 1.0e-12) -> Any:
    token = str(scale).strip().lower()
    if token == 'raw':
        return power
    if token == 'db':
        return 10.0 * torch.log10(torch.clamp(power, min=eps))
    if token == 'area':
        denom = torch.clamp(power.sum(dim=-1, keepdim=True), min=eps)
        return power / denom
    raise ValueError(f'Unsupported PSD scale: {scale!r}')


def _value_unit_psd(scale: str) -> str:
    token = str(scale).strip().lower()
    if token == 'db':
        return 'dB'
    if token == 'area':
        return 'area_normalized_power_fraction'
    return 'power'


def _scopes(value: Any) -> list[str]:
    scopes = _as_list(value, ('train', 'test'))
    normalized = []
    for scope in scopes:
        token = str(scope).strip().lower().replace('-', '_')
        if token in {'train_full', 'train'}:
            normalized.append('train')
        elif token in {'test_full', 'test'}:
            normalized.append('test')
        elif token in {'all', 'all_full'}:
            normalized.append('all')
        else:
            raise ValueError(f'Unsupported dataset element scope: {scope!r}')
    return normalized


def _dataset_for_scope(bundle: Any, scope: str, view_name: str):
    selected = str(view_name or '').strip() or str(bundle.psd_view_name)
    if scope == 'train':
        return 'train', dataset_for_view(bundle.train_dataset, selected)
    if scope == 'test':
        return 'test', dataset_for_view(bundle.test_dataset, selected)
    if scope == 'all':
        return 'all', ConcatDataset([dataset_for_view(bundle.train_dataset, selected), dataset_for_view(bundle.test_dataset, selected)])
    raise ValueError(scope)


def _frequency_axis(bin_count: int, time_length: int) -> list[float]:
    if time_length <= 0:
        return [float(i) for i in range(bin_count)]
    return [float(i) / float(time_length) for i in range(bin_count)]


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    save_yaml(path, {'source_program': SOURCE_PROGRAM, 'rows': rows})

SOURCE_PROGRAM = 'dataset_element_psd'


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Prepared dataset row/element-wise PSD analysis.')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--prep_root', required=True)
    parser.add_argument('--view_name', default='')
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--batch_size', required=True, type=int)
    parser.add_argument('--gpu_index', required=True, type=int)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin_memory', default='true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stats_dtype', default='float32', choices=('float32', 'float64'))
    parser.add_argument('--signal_window', default='none', choices=('hann', 'none'))
    parser.add_argument('--variants', nargs='*', default=['raw', 'centered'])
    parser.add_argument('--scales', nargs='*', default=['raw', 'db', 'area'])
    parser.add_argument('--scopes', nargs='*', default=None)
    parser.add_argument('--sample_reducer', default='mean', choices=('mean',))
    parser.add_argument('--config', default=None)
    parser.add_argument('--run_timestamp', default=None)
    parser.add_argument('--timestamped_output', default='true')
    return parser


def _windowed(maps: Any, signal_window: str) -> Any:
    token = normalize_signal_window(signal_window)
    if token == 'none':
        return maps
    if token == 'hann':
        win = torch.hann_window(int(maps.shape[-1]), periodic=False, device=maps.device, dtype=maps.dtype)
        return maps * win.view(1, 1, -1)
    raise ValueError(f'Unsupported signal_window: {signal_window!r}')


def _accumulate_scope(dataset: Any, args: argparse.Namespace, *, device: Any, dtype: Any) -> tuple[dict[str, Any], int, int, int]:
    variants = _as_list(args.variants, ('raw', 'centered'))
    accum: dict[str, Any] = {variant: None for variant in variants}
    sample_count = 0
    row_count = 0
    time_length = 0
    loader = make_loader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=_boolish(args.pin_memory, default=True), seed=int(args.seed))
    for inputs, _targets in loader:
        maps = _maps_from_batch(inputs, device=device, dtype=dtype)
        sample_count += int(maps.shape[0])
        row_count = int(maps.shape[1])
        time_length = int(maps.shape[2])
        for variant in variants:
            prepared = _windowed(_variant_maps(maps, variant), str(args.signal_window))
            power = torch.fft.rfft(prepared, dim=-1).abs().square()
            if accum[variant] is None:
                accum[variant] = power.sum(dim=0).detach().clone()
            else:
                accum[variant] = accum[variant] + power.sum(dim=0).detach()
    if sample_count > 0:
        for variant in variants:
            if accum[variant] is not None:
                accum[variant] = accum[variant] / float(sample_count)
    return accum, sample_count, row_count, time_length


def _rows_for_matrix(*, base: Mapping[str, Any], matrix: Any, variant: str, scale: str, time_length: int) -> list[dict[str, Any]]:
    matrix_cpu = matrix.detach().cpu().to(dtype=torch.float64)
    row_count = int(matrix_cpu.shape[0])
    freq_count = int(matrix_cpu.shape[1])
    freqs = _frequency_axis(freq_count, int(time_length))
    rows: list[dict[str, Any]] = []
    for element_index in range(row_count):
        values = matrix_cpu[element_index]
        for frequency_bin, value in enumerate(values.tolist()):
            item = dict(base)
            item.update(
                category='dataset_element_psd',
                signal_kind='input',
                series='x_probe',
                variant=str(variant),
                scale=str(scale),
                element_index=int(element_index),
                element_axis_order='input_row_or_flattened_feature_index_zero_based',
                time_length=int(time_length),
                frequency=float(freqs[frequency_bin]),
                frequency_unit='normalized_frequency',
                frequency_bin=int(frequency_bin),
                frequency_bin_count=int(freq_count),
                frequency_grid='rfft_one_sided_index_over_time_length',
                signal_window=str(args_signal_window_global),
                value=float(value),
                value_unit=_value_unit_psd(scale),
            )
            rows.append(common_row(**item))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parse_args_with_config(parser, argv=argv, stage_key='dataset_element_psd')
    if int(args.batch_size) < 1:
        parser.error('--batch_size must be >= 1.')
    if int(args.num_workers) < 0:
        parser.error('--num_workers must be >= 0.')
    _load_runtime_dependencies()
    seed = seed_everything(int(args.seed))
    dtype = torch.float32 if str(args.stats_dtype) == 'float32' else torch.float64
    device = _require_device(str(args.device), int(args.gpu_index))
    bundle = resolve_dataset_bundle(str(args.dataset), prep_root=str(args.prep_root))
    manifest = _load_manifest(Path(bundle.manifest_path))
    base_output = timestamped_output_root(args.output_root, run_timestamp=getattr(args, 'run_timestamp', None), prefix=SOURCE_PROGRAM, enabled=getattr(args, 'timestamped_output', True))
    base_output = Path(base_output).expanduser().resolve()
    base_output.mkdir(parents=True, exist_ok=True)
    global args_signal_window_global
    args_signal_window_global = normalize_signal_window(str(args.signal_window))
    common_base = {
        'source_program': SOURCE_PROGRAM,
        'dataset': str(bundle.dataset_name),
        'run_id': f'{bundle.dataset_name}_dataset_element_psd_seed{seed}',
        'seed': int(seed),
    }
    common_base.update(_axis_metadata_columns(manifest, psd_axis_kind=str(bundle.psd_axis_kind)))
    manifest_rows: list[dict[str, Any]] = []
    for scope in _scopes(args.scopes):
        split_name, scoped_dataset = _dataset_for_scope(bundle, scope, str(args.view_name or ''))
        accum, sample_count, _row_count, time_length = _accumulate_scope(scoped_dataset, args, device=device, dtype=dtype)
        if sample_count == 0:
            continue
        for variant, raw_matrix in accum.items():
            if raw_matrix is None:
                continue
            for scale in _as_list(args.scales, ('raw', 'db', 'area')):
                matrix = _scale_psd(raw_matrix, scale)
                base = dict(common_base)
                base.update(split=split_name, scope=f'{split_name}_full', sample_count=int(sample_count))
                rows = _rows_for_matrix(base=base, matrix=matrix, variant=variant, scale=scale, time_length=time_length)
                out_path = base_output / 'dataset_element_psd' / f'dataset_element_psd__{_safe_token(split_name)}__{_safe_token(variant)}__{_safe_token(scale)}.csv'
                write_common_csv(out_path, rows, extra_columns=['sample_count'])
                manifest_rows.append({'scope': f'{split_name}_full', 'artifact_name': 'dataset_element_psd', 'output_csv_path': str(out_path), 'sample_count': int(sample_count)})
    manifest_path = base_output / 'dataset_element_psd_manifest.yaml'
    _write_manifest(manifest_path, manifest_rows)
    print(json.dumps({'status': 'ok', 'source_program': SOURCE_PROGRAM, 'output_root': str(base_output), 'manifest': str(manifest_path)}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
