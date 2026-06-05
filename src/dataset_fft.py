"""데이터셋 입력 FFT 독립 분석 엔트리포인트."""

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
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


from src.util.csv_schema import common_row, write_common_csv, write_manifest_yaml
from src.util.config import compact_yaml, load_structured
from src.util.config import load_structured
from src.util.config_cli import parse_args_with_config
from src.util.paths import timestamped_output_root


SOURCE_PROGRAM = 'dataset_fft'


def _load_runtime_dependencies() -> None:
    """실제 데이터셋 FFT 분석 시점에만 무거운 의존성을 불러온다."""

    global torch, tqdm, seed_everything
    global dataset_for_view, make_loader, resolve_dataset_bundle
    global apply_centering, power_to_db, tensor_to_channel_major_maps_explicit
    global build_probe_index_bundle, dataset_targets, subset_from_indices

    import torch as _torch
    from tqdm import tqdm as _tqdm

    from src.data.registry import dataset_for_view as _dataset_for_view
    from src.data.registry import make_loader as _make_loader
    from src.data.registry import resolve_dataset_bundle as _resolve_dataset_bundle
    from src.signal.psd_utils import apply_centering as _apply_centering
    from src.signal.psd_utils import power_to_db as _power_to_db
    from src.signal.psd_utils import tensor_to_channel_major_maps_explicit as _tensor_to_channel_major_maps_explicit
    from src.stat.probe_selection import build_probe_index_bundle as _build_probe_index_bundle
    from src.stat.probe_selection import dataset_targets as _dataset_targets
    from src.stat.probe_selection import subset_from_indices as _subset_from_indices
    from src.util.random import seed_everything as _seed_everything

    torch = _torch
    tqdm = _tqdm
    dataset_for_view = _dataset_for_view
    make_loader = _make_loader
    resolve_dataset_bundle = _resolve_dataset_bundle
    apply_centering = _apply_centering
    power_to_db = _power_to_db
    tensor_to_channel_major_maps_explicit = _tensor_to_channel_major_maps_explicit
    build_probe_index_bundle = _build_probe_index_bundle
    dataset_targets = _dataset_targets
    subset_from_indices = _subset_from_indices
    seed_everything = _seed_everything


def _load_structured_light(path: Path) -> dict[str, Any]:
    payload = load_structured(path)
    if not isinstance(payload, dict):
        raise ValueError(f'구조화 파일 루트는 mapping이어야 합니다: {path}')
    return dict(payload)
CATEGORY = 'dataset_fft'
VARIANTS = ('raw', 'centered')
SCALES = ('raw', 'db')


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Prepared dataset 입력 FFT 분석')
    parser.add_argument('--dataset', required=True, help='정규화된 데이터셋 토큰')
    parser.add_argument('--prep_root', required=True, help='prepared 데이터 루트')
    parser.add_argument('--output_root', required=True, help='분석 출력 루트')
    parser.add_argument('--batch_size', required=True, type=int)
    parser.add_argument('--gpu_index', required=True, type=int)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--config', default=None, help='YAML 설정 파일 경로(.yaml)')
    parser.add_argument('--run_timestamp', default=None, help='결과 output_root 아래에 생성할 실행시각 폴더명 suffix. 생략 시 Asia/Seoul 현재시각을 사용한다.')
    parser.add_argument('--timestamped_output', default='true', help='true이면 output_root 아래 실행시각 폴더를 자동 생성한다. false이면 기존 경로에 직접 저장한다.')
    return parser


def _require_cuda_device(gpu_index: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError('--gpu_index가 지정되었지만 CUDA를 사용할 수 없습니다.')
    index = int(gpu_index)
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(f'--gpu_index {index} 값이 유효하지 않습니다. (CUDA 장치 수: {torch.cuda.device_count()})')
    torch.cuda.set_device(index)
    return torch.device(f'cuda:{index}')


def _validate_axis_metadata(manifest: Mapping[str, Any]) -> None:
    for key in ('psd_time_axis', 'psd_row_axes', 'psd_flatten_rule', 'psd_logical_shape'):
        if manifest.get(key) in (None, '', []):
            raise ValueError(f'Prepared manifest에 필수 축 메타데이터가 없습니다: {key}')


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


def _expected_rows_time(manifest: Mapping[str, Any]) -> tuple[int | None, int | None]:
    """manifest의 논리 PSD shape에서 row/time 기대값을 읽는다."""

    logical = manifest.get('psd_logical_shape')
    if isinstance(logical, (list, tuple)) and len(logical) == 2:
        return int(logical[0]), int(logical[1])
    return None, None


def _quota(dataset: Any) -> int:
    targets = dataset_targets(dataset)
    counts: dict[int, int] = {}
    for target in targets:
        counts[int(target)] = counts.get(int(target), 0) + 1
    if not counts:
        raise ValueError('빈 데이터셋 split에서는 probe scope를 만들 수 없습니다.')
    target_total = 100
    num_classes = len(counts)
    per_label = max(1, target_total // max(1, num_classes))
    return max(1, min(per_label, min(counts.values())))


def _probe_subsets(dataset: Any, *, split_name: str, seed: int):
    yield f'{split_name}_full', 'full', None, dataset

    quota = _quota(dataset)
    bundle = build_probe_index_bundle(
        dataset,
        split_name=split_name,
        seed=int(seed),
        same_label_n_per_label=quota,
        balanced_global_n_per_label=quota,
        distribution_global_min_class_n=quota,
    )
    yield f'{split_name}_balanced_global', 'balanced_global', None, subset_from_indices(dataset, bundle.balanced_global)


def _safe_token(value: Any) -> str:
    token = str(value).strip().lower()
    normalized = ''.join(ch if ch.isalnum() else '_' for ch in token)
    normalized = normalized.strip('_')
    return normalized or 'na'


def _manifest_row(*, base: Mapping[str, Any], path: Path, artifact_name: str, scope: str, status: str = 'ok', message: str = '') -> dict[str, Any]:
    row = dict(base)
    row.update(category='dataset_fft_manifest', artifact_name=artifact_name, scope=scope, status=status, message=message, output_csv_path=str(path))
    return common_row(**row)


def _row_output_name(row: Mapping[str, Any]) -> str:
    scope = _safe_token(row.get('scope', 'scope'))
    variant = _safe_token(row.get('variant', 'raw'))
    scale = _safe_token(row.get('scale', 'raw'))
    return f'{CATEGORY}__{scope}__{variant}__{scale}.csv'


def _write_grouped(root_dir: Path, rows: list[dict[str, Any]], *, manifest_rows: list[dict[str, Any]], manifest_base: Mapping[str, Any], artifact_name: str) -> None:
    grouped: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out_path = root_dir / CATEGORY / _row_output_name(row)
        grouped[out_path].append(row)
    for out_path, output_rows in grouped.items():
        write_common_csv(out_path, output_rows)
        scope = str(output_rows[0].get('scope', '')) if output_rows else ''
        manifest_rows.append(_manifest_row(base=manifest_base, path=out_path, artifact_name=artifact_name, scope=scope))


def _update_accumulator(acc: torch.Tensor | None, values: torch.Tensor) -> torch.Tensor:
    batch_sum = values.detach().sum(dim=0)
    if acc is None:
        return batch_sum.clone()
    if tuple(acc.shape) != tuple(batch_sum.shape):
        raise ValueError(f'FFT 누적 shape가 일치하지 않습니다: acc={tuple(acc.shape)} batch={tuple(batch_sum.shape)}')
    return acc + batch_sum


def _scaled(values: torch.Tensor, scale: str) -> torch.Tensor:
    if str(scale) == 'raw':
        return values
    if str(scale) == 'db':
        return power_to_db(values)
    raise ValueError(f'지원하지 않는 scale 값입니다: {scale}')


def _value_unit(scale: str) -> str:
    return 'dB' if str(scale) == 'db' else 'power'


def _fft_rows(*, base: Mapping[str, Any], fft_values: torch.Tensor, variant: str, scale: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    arr = fft_values.detach().cpu().to(dtype=torch.float64).reshape(-1)
    for frequency_index, value in enumerate(arr.tolist()):
        row = dict(base)
        row.update(
            category=CATEGORY,
            signal_kind='input',
            series='x_probe',
            variant=str(variant),
            scale=str(scale),
            row_frequency_index=int(frequency_index),
            frequency_bin_count=int(arr.numel()),
            frequency_grid='rfft_one_sided_index',
            value=float(value),
            value_unit=_value_unit(scale),
        )
        rows.append(common_row(**row))
    return rows


def _dataset_tokens(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        tokens = [str(v).strip() for v in value if str(v).strip()]
    else:
        tokens = [str(value).strip()]
    if not tokens:
        raise ValueError('dataset 배열은 비어 있을 수 없습니다.')
    return tokens


def _dataset_output_root(base_output_root: str | Path, dataset_token: str, *, multi_dataset: bool) -> Path:
    root = Path(base_output_root).expanduser().resolve()
    return root / str(dataset_token) if multi_dataset else root


def _run_dataset_fft(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.batch_size) < 1:
        raise ValueError('--batch_size는 1 이상이어야 합니다.')
    if int(args.num_workers) < 0:
        raise ValueError('--num_workers는 0 이상이어야 합니다.')

    seed = seed_everything(int(args.seed))
    device = _require_cuda_device(int(args.gpu_index))
    bundle = resolve_dataset_bundle(str(args.dataset), prep_root=str(args.prep_root))

    manifest = _load_structured_light(Path(bundle.manifest_path))
    if not isinstance(manifest, dict):
        raise ValueError(f'Prepared manifest 형식이 올바르지 않습니다: {bundle.manifest_path}')
    _validate_axis_metadata(manifest)

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_id = f'{bundle.dataset_name}_dataset_fft_seed{seed}'
    common_base = {
        'source_program': SOURCE_PROGRAM,
        'run_id': run_id,
        'dataset': str(bundle.dataset_name),
        'seed': int(seed),
    }
    common_base.update(_axis_metadata_columns(manifest, psd_axis_kind=str(bundle.psd_axis_kind)))

    fft_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for split_name, split_dataset in tqdm((('train', bundle.train_dataset), ('test', bundle.test_dataset)), desc=f'dataset_fft {bundle.dataset_name} split', leave=False):
        analysis_dataset = dataset_for_view(split_dataset, bundle.psd_view_name)
        for scope, probe_family, label, subset in tqdm(list(_probe_subsets(analysis_dataset, split_name=split_name, seed=seed)), desc=f'dataset_fft {bundle.dataset_name} {split_name} scope', leave=False):
            accumulators: dict[str, torch.Tensor | None] = {v: None for v in VARIANTS}
            sample_count = 0
            loader = make_loader(
                subset,
                batch_size=int(args.batch_size),
                shuffle=False,
                num_workers=int(args.num_workers),
                pin_memory=device.type == 'cuda',
                seed=int(seed),
            )
            expected_rows, expected_time = _expected_rows_time(manifest)
            for inputs, _targets in tqdm(loader, desc=f'dataset_fft {scope} batch', leave=False):
                maps = tensor_to_channel_major_maps_explicit(
                    torch.as_tensor(inputs, dtype=torch.float32).to(device=device, non_blocking=True),
                    psd_axis_kind=str(bundle.psd_axis_kind),
                    psd_time_axis=manifest.get('psd_time_axis'),
                    psd_flatten_rule=manifest.get('psd_flatten_rule'),
                    psd_logical_shape=manifest.get('psd_logical_shape'),
                    expected_time=expected_time,
                    expected_rows=expected_rows,
                )
                sample_count += int(maps.shape[0])
                for variant in VARIANTS:
                    prepared = maps if variant == 'raw' else apply_centering(maps)
                    power = torch.fft.rfft(prepared, dim=-1).abs().square().mean(dim=1)
                    accumulators[variant] = _update_accumulator(accumulators[variant], power)

            if sample_count == 0:
                continue

            for variant in VARIANTS:
                summed = accumulators[variant]
                if summed is None:
                    continue
                mean_power = summed / float(sample_count)
                base = dict(common_base)
                base.update(scope=scope, probe_family=probe_family, label='' if label is None else int(label), split=split_name)
                for scale in SCALES:
                    fft_rows.extend(_fft_rows(base=base, fft_values=_scaled(mean_power, scale), variant=variant, scale=scale))

    _write_grouped(output_root, fft_rows, manifest_rows=manifest_rows, manifest_base=common_base, artifact_name=CATEGORY)
    manifest_path = output_root / 'dataset_fft_manifest.yaml'
    write_manifest_yaml(manifest_path, manifest_rows)
    return {'dataset': str(bundle.dataset_name), 'output_root': str(output_root), 'manifest': str(manifest_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parse_args_with_config(parser, argv=argv, stage_key='dataset_fft')
    _load_runtime_dependencies()
    dataset_tokens = _dataset_tokens(args.dataset)
    base_output_root = timestamped_output_root(args.output_root, run_timestamp=getattr(args, 'run_timestamp', None), prefix=SOURCE_PROGRAM, enabled=getattr(args, 'timestamped_output', True))
    results: list[dict[str, Any]] = []
    for index, dataset_token in enumerate(dataset_tokens, start=1):
        print(f'[dataset_fft] 시작 {index}/{len(dataset_tokens)} dataset={dataset_token}', flush=True)
        run_args = argparse.Namespace(**vars(args))
        run_args.dataset = dataset_token
        run_args.output_root = str(_dataset_output_root(base_output_root, dataset_token, multi_dataset=len(dataset_tokens) > 1))
        results.append(_run_dataset_fft(run_args))
        print(f'[dataset_fft] 완료 {index}/{len(dataset_tokens)} dataset={dataset_token}', flush=True)
    print(json.dumps({'status': 'ok', 'source_program': SOURCE_PROGRAM, 'outputs': results}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
