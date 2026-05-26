"""Dataset-level input/probe PSD baseline entrypoint."""

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

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


from src.util.csv_schema import common_row, write_common_csv
from src.util.config_cli import parse_args_with_config


SOURCE_PROGRAM = 'dataset_psd'


def _load_runtime_dependencies() -> None:
    """실제 데이터셋 PSD 분석 시점에만 무거운 의존성을 불러온다."""

    global np, torch, tqdm, seed_everything
    global dataset_for_view, make_loader, resolve_dataset_bundle
    global curve_axis_from_summary
    global apply_centering, exact_periodogram_from_maps, power_to_db, tensor_to_channel_major_maps_explicit
    global build_probe_index_bundle, build_probe_scopes, dataset_targets, subset_from_indices

    import numpy as _np
    import torch as _torch
    from tqdm import tqdm as _tqdm

    from src.data.registry import dataset_for_view as _dataset_for_view
    from src.data.registry import make_loader as _make_loader
    from src.data.registry import resolve_dataset_bundle as _resolve_dataset_bundle
    from src.signal.family_spectral_analysis import curve_axis_from_summary as _curve_axis_from_summary
    from src.signal.psd_utils import apply_centering as _apply_centering
    from src.signal.psd_utils import exact_periodogram_from_maps as _exact_periodogram_from_maps
    from src.signal.psd_utils import power_to_db as _power_to_db
    from src.signal.psd_utils import tensor_to_channel_major_maps_explicit as _tensor_to_channel_major_maps_explicit
    from src.stat.probe_selection import build_probe_index_bundle as _build_probe_index_bundle, build_probe_scopes as _build_probe_scopes
    from src.stat.probe_selection import dataset_targets as _dataset_targets
    from src.stat.probe_selection import subset_from_indices as _subset_from_indices
    from src.util.random import seed_everything as _seed_everything

    np = _np
    torch = _torch
    tqdm = _tqdm
    dataset_for_view = _dataset_for_view
    make_loader = _make_loader
    resolve_dataset_bundle = _resolve_dataset_bundle
    curve_axis_from_summary = _curve_axis_from_summary
    apply_centering = _apply_centering
    exact_periodogram_from_maps = _exact_periodogram_from_maps
    power_to_db = _power_to_db
    tensor_to_channel_major_maps_explicit = _tensor_to_channel_major_maps_explicit
    build_probe_index_bundle = _build_probe_index_bundle; build_probe_scopes = _build_probe_scopes
    dataset_targets = _dataset_targets
    subset_from_indices = _subset_from_indices
    seed_everything = _seed_everything


def _load_json_light(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
ALL_CURVE_EXTRACTORS = ('psd_exact',)
ALL_VALUE_SCALES = ('raw', 'db')


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Dataset-level input PSD baseline producer.')
    parser.add_argument('--dataset', required=True, help='Canonical dataset token.')
    parser.add_argument('--prep_root', required=True, help='Prepared data root containing <dataset>/manifest.json.')
    parser.add_argument('--output_root', required=True, help='Output directory for category CSV files.')
    parser.add_argument('--batch_size', required=True, type=int)
    parser.add_argument('--gpu_index', required=True, type=int)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--config', default=None, help='JSON 설정 파일 경로(.json)')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--userbin_edges', nargs='*', default=None)
    parser.add_argument('--userbin_width', type=float, default=None)
    parser.add_argument('--userbin_count', type=int, default=None)
    parser.add_argument('--userbin_reducer', default=None)
    return parser


def _require_cuda_device(gpu_index: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError('--gpu_index was requested, but CUDA is unavailable.')
    index = int(gpu_index)
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(f'--gpu_index {index} is invalid for {torch.cuda.device_count()} CUDA device(s).')
    torch.cuda.set_device(index)
    return torch.device(f'cuda:{index}')


def _validate_axis_metadata(manifest: Mapping[str, Any]) -> None:
    for key in ('psd_time_axis', 'psd_row_axes', 'psd_flatten_rule', 'psd_logical_shape'):
        if manifest.get(key) in (None, '', []):
            raise ValueError(f'Prepared manifest is missing required axis metadata: {key}')



def _axis_metadata_columns(manifest: Mapping[str, Any], *, psd_axis_kind: str) -> dict[str, Any]:
    logical_shape = manifest.get('psd_logical_shape')
    static_repeat_t = manifest.get('static_repeat_T')
    return {
        'prep_profile': str(manifest.get('prep_profile', manifest.get('psd_axis_kind', psd_axis_kind))),
        'psd_axis_kind': str(manifest.get('psd_axis_kind', psd_axis_kind)),
        'psd_time_axis': manifest.get('psd_time_axis', ''),
        'psd_row_axes': json.dumps(manifest.get('psd_row_axes', []), ensure_ascii=False),
        'psd_flatten_rule': str(manifest.get('psd_flatten_rule', '')),
        'psd_logical_shape': json.dumps(logical_shape if logical_shape is not None else [], ensure_ascii=False),
        'static_repeat_T': '' if static_repeat_t in (None, '') else int(static_repeat_t),
    }


def _expected_rows_time(manifest: Mapping[str, Any]) -> tuple[int | None, int | None]:
    logical = manifest.get('psd_logical_shape')
    if isinstance(logical, (list, tuple)) and len(logical) == 2:
        return int(logical[0]), int(logical[1])
    return None, None

def _seed_everything(seed: int) -> None:
    seed_everything(int(seed))


def _quota(dataset: Any) -> int:
    targets = dataset_targets(dataset)
    counts: dict[int, int] = {}
    for target in targets:
        counts[int(target)] = counts.get(int(target), 0) + 1
    if not counts:
        raise ValueError('Cannot build probes for an empty dataset split.')

    target_total = 100
    num_classes = len(counts)
    per_label = max(1, target_total // max(1, num_classes))

    return max(1, min(per_label, min(counts.values())))


def _probe_subsets(dataset: Any, *, split_name: str, seed: int):
    quota = _quota(dataset)
    bundle = build_probe_index_bundle(
        dataset,
        split_name=split_name,
        seed=int(seed),
        same_label_n_per_label=quota,
        balanced_global_n_per_label=quota,
        distribution_global_min_class_n=quota,
    )

    for scope in build_probe_scopes(dataset, split_name=split_name, bundle=bundle):
        yield (scope.scope, scope.probe_family, scope.label, scope.subset)


def _value_unit_for_power_scale(scale: str) -> str:
    return 'dB' if str(scale) == 'db' else 'power'


def _value_unit_for_dispersion(metric: str, scale: str) -> str:
    if str(scale) == 'db':
        return 'dB'
    return 'power^2' if str(metric) == 'variance' else 'power'


def _iter_scaled_centering_maps(payload: Mapping[str, Any]):
    for scale in ALL_VALUE_SCALES:
        if scale in payload:
            yield scale, payload[scale]


def _curve_rows(*, summary: Mapping[str, Any], base: Mapping[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    curve_rows: list[dict[str, str]] = []
    dispersion_rows: list[dict[str, str]] = []
    for reducer, extractor_map in summary.get('representative', {}).items():
        for extractor, scaled_payload in extractor_map.items():
            if extractor not in ALL_CURVE_EXTRACTORS:
                continue
            axis = curve_axis_from_summary(dict(summary), 'psd_exact')
            edges = None
            for scale, centering_map in _iter_scaled_centering_maps(scaled_payload):
                for centering, values in centering_map.items():
                    for idx, value in enumerate(np.asarray(values, dtype=np.float64).reshape(-1)):
                        kwargs = dict(base)
                        kwargs.update(
                            category='dataset_curve',
                            extractor=extractor,
                            reducer=reducer,
                            variant='centered' if str(centering) == 'cen' else 'raw',
                            scale=scale,
                            frequency=float(axis[idx]) if idx < len(axis) else '',
                            frequency_unit='normalized_frequency',
                            value=float(value),
                            value_unit=_value_unit_for_power_scale(scale),
                        )
                        if edges is not None and idx + 1 < len(edges):
                            kwargs['bin_left'] = float(edges[idx])
                            kwargs['bin_right'] = float(edges[idx + 1])
                        curve_rows.append(common_row(**kwargs))
    for extractor, metric_map in summary.get('dispersion', {}).items():
        if extractor not in ALL_CURVE_EXTRACTORS:
            continue
        axis = curve_axis_from_summary(dict(summary), 'psd_exact')
        edges = None
        for metric, scaled_payload in metric_map.items():
            for scale, centering_map in _iter_scaled_centering_maps(scaled_payload):
                for centering, values in centering_map.items():
                    for idx, value in enumerate(np.asarray(values, dtype=np.float64).reshape(-1)):
                        kwargs = dict(base)
                        kwargs.update(
                            category='dataset_dispersion',
                            extractor=extractor,
                            variant='centered' if str(centering) == 'cen' else 'raw',
                            scale=scale,
                            statistic=str(metric),
                            frequency=float(axis[idx]) if idx < len(axis) else '',
                            frequency_unit='normalized_frequency',
                            value=float(value),
                            value_unit=_value_unit_for_dispersion(str(metric), scale),
                        )
                        if edges is not None and idx + 1 < len(edges):
                            kwargs['bin_left'] = float(edges[idx])
                            kwargs['bin_right'] = float(edges[idx + 1])
                        dispersion_rows.append(common_row(**kwargs))
    return curve_rows, dispersion_rows



def _scaled_tensor(values: torch.Tensor, scale: str) -> torch.Tensor:
    scale_token = str(scale).strip().lower()
    if scale_token == 'raw':
        return values
    if scale_token == 'db':
        return power_to_db(values)
    raise ValueError(f'Unsupported PSD value scale: {scale!r}.')


def _reduce_row_tensor(values: torch.Tensor, reducer: str) -> torch.Tensor:
    reducer_token = str(reducer).strip().lower()
    if reducer_token == 'mean':
        return values.mean(dim=1)
    if reducer_token == 'median':
        return values.median(dim=1).values
    raise ValueError(f'Unsupported representative reducer: {reducer!r}.')


def _row_dispersion_stack(values: torch.Tensor, metric: str) -> torch.Tensor:
    metric_token = str(metric).strip().lower()
    if metric_token == 'variance':
        return values.var(dim=1, unbiased=False)
    if metric_token == 'mad':
        row_median = values.median(dim=1, keepdim=True).values
        return (values - row_median).abs().median(dim=1).values
    raise ValueError(f'Unsupported dispersion metric: {metric!r}.')


def _add_accumulator(accumulator: dict[tuple[str, ...], torch.Tensor], key: tuple[str, ...], values: torch.Tensor) -> None:
    batch_sum = values.detach().sum(dim=0)
    if key not in accumulator:
        accumulator[key] = batch_sum.clone()
        return
    if tuple(accumulator[key].shape) != tuple(batch_sum.shape):
        raise ValueError(f'PSD accumulator shape mismatch for {key}: {tuple(accumulator[key].shape)} vs {tuple(batch_sum.shape)}.')
    accumulator[key] = accumulator[key] + batch_sum


def _numpy_mean(accumulator: Mapping[tuple[str, ...], torch.Tensor], key: tuple[str, ...], total_samples: int) -> np.ndarray:
    if key not in accumulator:
        raise KeyError(f'Missing PSD accumulator key: {key}')
    return (accumulator[key] / float(total_samples)).detach().cpu().numpy()


def _streaming_summary(
    dataset: Any,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
    manifest: Mapping[str, Any],
    psd_axis_kind: str,
) -> dict[str, Any]:
    loader = make_loader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.type == 'cuda',
        seed=int(seed),
    )
    representative_acc: dict[tuple[str, ...], torch.Tensor] = {}
    dispersion_acc: dict[tuple[str, ...], torch.Tensor] = {}
    total_samples = 0
    num_rows: int | None = None
    sequence_length: int | None = None
    freq_ref: torch.Tensor | None = None

    expected_rows, expected_time = _expected_rows_time(manifest)
    for inputs, _target in tqdm(loader, desc='dataset_psd batch', leave=False):
        maps = tensor_to_channel_major_maps_explicit(
            torch.as_tensor(inputs, dtype=torch.float32),
            psd_axis_kind=str(psd_axis_kind),
            psd_time_axis=manifest.get('psd_time_axis'),
            psd_flatten_rule=manifest.get('psd_flatten_rule'),
            psd_logical_shape=manifest.get('psd_logical_shape'),
            expected_time=expected_time,
            expected_rows=expected_rows,
        ).to(device=device, non_blocking=True)
        if maps.ndim != 3:
            raise ValueError(f'PSD maps must have shape (samples, rows, time), got {tuple(maps.shape)}.')
        if int(maps.shape[0]) < 1:
            continue
        batch_rows = int(maps.shape[1])
        batch_time = int(maps.shape[2])
        if num_rows is None:
            num_rows = batch_rows
            sequence_length = batch_time
        elif num_rows != batch_rows or sequence_length != batch_time:
            raise ValueError(
                'All batches within one dataset PSD scope must share row/time shape; '
                f'expected ({num_rows}, {sequence_length}), got ({batch_rows}, {batch_time}).'
            )
        total_samples += int(maps.shape[0])

        for centering in ('raw', 'cen'):
            variant_maps = maps if centering == 'raw' else apply_centering(maps)
            freqs, exact_psd = exact_periodogram_from_maps(variant_maps)
            if freq_ref is None:
                freq_ref = freqs.detach().clone()
            elif int(freq_ref.numel()) != int(freqs.numel()):
                raise ValueError('Frequency axis length changed across streaming batches in one dataset PSD scope.')

            extractor_values = (
                ('psd_exact', exact_psd.real),
            )
            for extractor, values in extractor_values:
                for reducer in ('mean', 'median'):
                    reduced = _reduce_row_tensor(values, reducer)
                    _add_accumulator(
                        representative_acc,
                        (reducer, extractor, centering),
                        reduced,
                    )
                for metric in ('variance', 'mad'):
                    _add_accumulator(
                        dispersion_acc,
                        (extractor, metric, centering),
                        _row_dispersion_stack(values, metric),
                    )

    if total_samples < 1 or num_rows is None or sequence_length is None or freq_ref is None:
        raise ValueError('Selected dataset PSD scope is empty.')

    summary: dict[str, Any] = {
        'num_samples': int(total_samples),
        'num_rows': int(num_rows),
        'sequence_length': int(sequence_length),
        'representative_reducers': ['mean', 'median'],
        'curve_extractors': list(ALL_CURVE_EXTRACTORS),
        'spectrogram_saved': False,
        'freq': freq_ref.detach().cpu().numpy(),
        'userbin_centers': None,
        'representative': {'mean': {}, 'median': {}},
        'dispersion': {},
    }
    for reducer in ('mean', 'median'):
        for extractor in ALL_CURVE_EXTRACTORS:
            for centering in ('raw', 'cen'):
                raw_curve = _numpy_mean(
                    representative_acc,
                    (reducer, extractor, centering),
                    total_samples,
                )
                for scale in ALL_VALUE_SCALES:
                    value = raw_curve if scale == 'raw' else power_to_db(raw_curve)
                    summary['representative'].setdefault(reducer, {}).setdefault(extractor, {}).setdefault(scale, {})[centering] = value
    for extractor in ALL_CURVE_EXTRACTORS:
        for metric in ('variance', 'mad'):
            for centering in ('raw', 'cen'):
                raw_dispersion = _numpy_mean(
                    dispersion_acc,
                    (extractor, metric, centering),
                    total_samples,
                )
                for scale in ALL_VALUE_SCALES:
                    value = raw_dispersion if scale == 'raw' else power_to_db(raw_dispersion)
                    summary['dispersion'].setdefault(extractor, {}).setdefault(metric, {}).setdefault(scale, {})[centering] = value
    return summary


def _safe_token(value: Any) -> str:
    text = str(value).strip().lower().replace('-', '_')
    return ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in text).strip('_') or 'value'


def _row_output_name(row: Mapping[str, str]) -> str:
    category = row.get('category', '')
    scope = _safe_token(row.get('scope', 'scope'))
    extractor = _safe_token(row.get('extractor', 'extractor'))
    variant = _safe_token(row.get('variant', 'raw'))
    scale = _safe_token(row.get('scale', 'raw'))
    signal = _safe_token(row.get('signal_kind', 'input'))
    if category == 'dataset_curve':
        reducer = _safe_token(row.get('reducer', 'mean'))
        return f'dataset_curve__{scope}__{signal}__{extractor}__{reducer}__{variant}__{scale}.csv'
    if category == 'dataset_dispersion':
        statistic = _safe_token(row.get('statistic', 'statistic'))
        return f'dataset_dispersion__{scope}__{signal}__{extractor}__{variant}__{scale}__{statistic}.csv'
    return f'{_safe_token(category)}__{scope}.csv'


def _manifest_row(*, base: Mapping[str, Any], path: Path, artifact_name: str, scope: str, status: str = 'ok', message: str = '') -> dict[str, str]:
    kwargs = dict(base)
    kwargs.update(
        category='dataset_psd_manifest',
        status=status,
        message=message,
        artifact_name=artifact_name,
        output_csv_path=str(path),
        scope=scope,
    )
    return common_row(**kwargs)


def _write_grouped(root_dir: Path, rows: list[dict[str, str]], *, manifest_rows: list[dict[str, str]], manifest_base: Mapping[str, Any], artifact_name: str) -> None:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out_path = root_dir / _safe_token(row.get('category', 'category')) / _row_output_name(row)
        groups[str(out_path)].append(row)
    for out_path_text, group_rows in sorted(groups.items()):
        out_path = Path(out_path_text)
        write_common_csv(out_path, group_rows)
        scope = group_rows[0].get('scope', '') if group_rows else ''
        manifest_rows.append(_manifest_row(base=manifest_base, path=out_path, artifact_name=artifact_name, scope=scope))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parse_args_with_config(parser, argv=argv, stage_key='dataset_psd')
    if int(args.batch_size) < 1:
        parser.error('--batch_size must be >= 1.')
    if int(args.num_workers) < 0:
        parser.error('--num_workers must be >= 0.')

    _load_runtime_dependencies()
    _seed_everything(int(args.seed))
    device = _require_cuda_device(int(args.gpu_index))
    dataset_token = str(args.dataset)
    prep_root = Path(args.prep_root).expanduser().resolve()
    bundle = resolve_dataset_bundle(dataset_token, prep_root=prep_root)
    manifest = _load_json_light(bundle.manifest_path)
    if not isinstance(manifest, Mapping):
        raise ValueError(f'Prepared manifest must be a JSON object: {bundle.manifest_path}')
    _validate_axis_metadata(manifest)

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    prep_profile = str(manifest.get('prep_profile', manifest.get('psd_axis_kind', bundle.psd_axis_kind)))
    run_id = f'{dataset_token}_dataset_psd_seed{int(args.seed)}'
    common_base = {
        'source_program': SOURCE_PROGRAM,
        'run_id': run_id,
        'dataset': dataset_token,
        **_axis_metadata_columns(manifest, psd_axis_kind=bundle.psd_axis_kind),
    }
    manifest_rows: list[dict[str, str]] = []
    split_items = (('train', bundle.train_dataset), ('test', bundle.test_dataset))
    for split_name, split_dataset in tqdm(split_items, desc='dataset_psd split', leave=False):
        psd_dataset = dataset_for_view(split_dataset, bundle.psd_view_name)
        views: list[tuple[str, str, int | None, Any]] = [(f'{split_name}_full', 'full_dataset', None, psd_dataset)]
        views.extend((f'{split_name}_{family_id}', family, label, subset) for family_id, family, label, subset in _probe_subsets(psd_dataset, split_name=split_name, seed=int(args.seed)))
        for scope, family, label, subset in tqdm(views, desc=f'dataset_psd {split_name} scope', leave=False):
            summary = _streaming_summary(
                subset,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                seed=int(args.seed),
                device=device,
                manifest=manifest,
                psd_axis_kind=bundle.psd_axis_kind,
            )
            base = dict(common_base)
            base.update(scope=scope, probe_family=family, label='' if label is None else int(label), signal_kind='input')
            curve_rows, dispersion_rows = _curve_rows(summary=summary, base=base)
            _write_grouped(output_root, curve_rows, manifest_rows=manifest_rows, manifest_base=common_base, artifact_name='dataset_curve')
            _write_grouped(output_root, dispersion_rows, manifest_rows=manifest_rows, manifest_base=common_base, artifact_name='dataset_dispersion')
    manifest_path = output_root / 'dataset_psd_manifest.csv'
    write_common_csv(manifest_path, manifest_rows)
    print(json.dumps({'status': 'ok', 'source_program': SOURCE_PROGRAM, 'output_root': str(output_root), 'manifest': str(manifest_path)}, sort_keys=True))
    return 0


try:
    from src.patch_overlays.runtime_patch import patch_dataset_psd as _patch_dataset_psd
    _patch_dataset_psd(globals())
except Exception:
    pass

if __name__ == '__main__':
    raise SystemExit(main())
