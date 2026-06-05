"""Checkpoint-only 2-D FFT analysis for MLP probe output maps."""

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
from pathlib import Path
from typing import Any, Mapping, Sequence


from src.util.csv_schema import common_row, write_common_csv, write_manifest_yaml
from src.util.config_cli import parse_args_with_config
from src.util.paths import timestamped_output_root


SOURCE_PROGRAM = '2d_fft_analysis'
VARIANTS = ('raw', 'centered')
SCALES = ('raw', 'db')


def _load_runtime_dependencies() -> None:
    """2D FFT 실행에 필요한 무거운 의존성을 지연 로드한다."""

    global torch, tqdm, psd_common
    global ProbeMapKey, checkpoint_output_dir, collect_mlp_output_maps, common_base_for_key
    global layer_folder, manifest_row, safe_token, validate_mlp_only, value_columns, write_matrix_csv
    global dataset_for_view, power_to_db
    import torch as _torch
    from tqdm import tqdm as _tqdm
    import src.psd_analysis as _psd_common

    _psd_common._load_runtime_dependencies()
    from src.analysis_matrix_common import ProbeMapKey as _ProbeMapKey, checkpoint_output_dir as _checkpoint_output_dir, collect_mlp_output_maps as _collect_mlp_output_maps, common_base_for_key as _common_base_for_key, layer_folder as _layer_folder, manifest_row as _manifest_row, safe_token as _safe_token, validate_mlp_only as _validate_mlp_only, value_columns as _value_columns, write_matrix_csv as _write_matrix_csv
    from src.data.registry import dataset_for_view as _dataset_for_view
    from src.signal.psd_utils import power_to_db as _power_to_db
    torch = _torch
    tqdm = _tqdm
    psd_common = _psd_common
    ProbeMapKey = _ProbeMapKey
    checkpoint_output_dir = _checkpoint_output_dir
    collect_mlp_output_maps = _collect_mlp_output_maps
    common_base_for_key = _common_base_for_key
    layer_folder = _layer_folder
    manifest_row = _manifest_row
    safe_token = _safe_token
    validate_mlp_only = _validate_mlp_only
    value_columns = _value_columns
    write_matrix_csv = _write_matrix_csv
    dataset_for_view = _dataset_for_view
    power_to_db = _power_to_db


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Checkpoint-only 2-D FFT analysis for MLP probe output maps.')
    parser.add_argument('--checkpoint', required=True, help='Single .pt checkpoint file or strict .pt-only checkpoint directory.')
    parser.add_argument('--dataset', required=True, help='Canonical dataset token stored in checkpoint metadata.')
    parser.add_argument('--prep_root', required=True, help='Prepared data root containing <dataset>/manifest.yaml.')
    parser.add_argument('--output_root', required=True, help='Root directory for 2-D FFT CSV outputs.')
    parser.add_argument('--anal_batch', required=True, type=int, help='Maximum samples per analysis forward pass.')
    parser.add_argument('--gpu_index', required=True, type=int, help='CUDA device index for analysis.')
    parser.add_argument('--seed', type=int, default=None, help='Analysis seed. Defaults to checkpoint seed when omitted.')
    parser.add_argument('--num_workers', type=int, default=0, help='Probe loading DataLoader worker count.')
    parser.add_argument('--low_vram', type=int, default=0, help='Use CPU trace staging to reduce VRAM use: 0 or 1.')
    parser.add_argument('--config', default=None, help='YAML 설정 파일 경로(.yaml)')
    parser.add_argument('--run_timestamp', default=None, help='결과 output_root 아래에 생성할 실행시각 폴더명 suffix. 생략 시 Asia/Seoul 현재시각을 사용한다.')
    parser.add_argument('--timestamped_output', default='true', help='true이면 output_root 아래 실행시각 폴더를 자동 생성한다. false이면 기존 경로에 직접 저장한다.')
    return parser


def _variant_maps(maps: torch.Tensor, *, variant: str) -> torch.Tensor:
    token = str(variant)
    if token == 'raw':
        return maps
    if token == 'centered':
        return maps - maps.mean(dim=(-2, -1), keepdim=True)
    raise ValueError(f'Unsupported 2-D FFT variant: {variant!r}.')


def _two_d_fft_power(maps: torch.Tensor, *, variant: str) -> torch.Tensor:
    prepared = _variant_maps(maps, variant=variant)
    if prepared.ndim != 3:
        raise ValueError(f'Expected output maps with shape (samples, neurons, time), got {tuple(prepared.shape)}.')
    fft = torch.fft.fftshift(torch.fft.fft2(prepared, dim=(-2, -1)), dim=(-2, -1))
    return fft.abs().square().mean(dim=0).real.contiguous()


def _scale_power(power: torch.Tensor, *, scale: str) -> torch.Tensor:
    token = str(scale)
    if token == 'raw':
        return power
    if token == 'db':
        return power_to_db(power)
    raise ValueError(f'Unsupported 2-D FFT scale: {scale!r}.')


def _fftshift_frequency_axis(length: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.fftfreq(int(length), d=1.0, device=device, dtype=dtype))


def _value_unit(scale: str) -> str:
    return 'dB' if str(scale) == 'db' else 'power'


def _artifact_name(base: Mapping[str, Any], key: ProbeMapKey, *, variant: str, scale: str) -> str:
    layer_name, layer_index, signal_kind, series, scope, _probe_family, label, sample_role, _sample_index = key
    label_part = 'all_labels' if label is None else f'label_{int(label)}'
    return (
        f'analysis_2d_fft__epoch_{safe_token(base.get("checkpoint_epoch", "epoch"))}'
        f'__layer_{safe_token(layer_index)}__{safe_token(layer_name)}'
        f'__{safe_token(scope)}__{label_part}'
        f'__{safe_token(signal_kind)}__{safe_token(series)}'
        f'__{safe_token(sample_role)}__{safe_token(variant)}__{safe_token(scale)}.csv'
    )


def _rows_for_power_matrix(
    *,
    base: Mapping[str, Any],
    power: torch.Tensor,
    variant: str,
    scale: str,
    value_column_names: Sequence[str],
) -> list[dict[str, Any]]:
    if power.ndim != 2:
        raise ValueError(f'Expected 2-D power matrix, got {tuple(power.shape)}.')
    row_count = int(power.shape[0])
    time_length = int(power.shape[1])
    row_axis = _fftshift_frequency_axis(row_count, device=power.device, dtype=power.dtype).detach().cpu().tolist()
    values = power.detach().cpu().to(dtype=torch.float64)
    rows: list[dict[str, Any]] = []
    for row_index in range(row_count):
        row_kwargs = dict(base)
        row_kwargs.update(
            category='analysis_2d_fft',
            variant=variant,
            scale=scale,
            row_frequency_index=row_index,
            row_frequency=float(row_axis[row_index]),
            row_frequency_unit='cycles_per_row_index',
            row_count=row_count,
            time_length=time_length,
            time_frequency_bin_count=time_length,
            time_frequency_grid='fftshift_fftfreq_time_index',
            value_unit=_value_unit(scale),
        )
        row = common_row(**row_kwargs)
        for column_index, column_name in enumerate(value_column_names):
            row[column_name] = float(values[row_index, column_index].item())
        rows.append(row)
    return rows


def _write_2d_fft_artifact(
    *,
    checkpoint_dir: Path,
    checkpoint_base: Mapping[str, Any],
    key: ProbeMapKey,
    maps: torch.Tensor,
    variant: str,
    scale: str,
    manifest_rows: list[dict[str, str]],
) -> None:
    raw_power = _two_d_fft_power(maps, variant=variant)
    scaled_power = _scale_power(raw_power, scale=scale)
    value_column_names = value_columns('time_freq', int(scaled_power.shape[1]))
    base = common_base_for_key(checkpoint_base=checkpoint_base, key=key)
    rows = _rows_for_power_matrix(base=base, power=scaled_power, variant=variant, scale=scale, value_column_names=value_column_names)
    layer_name, layer_index, *_rest = key
    out_path = checkpoint_dir / 'layers' / layer_folder(layer_name, layer_index) / 'analysis_2d_fft' / _artifact_name(base, key, variant=variant, scale=scale)
    write_matrix_csv(out_path, rows, value_column_names=value_column_names)
    manifest_rows.append(manifest_row(base=dict(checkpoint_base), artifact_name='analysis_2d_fft', path=out_path))


def _checkpoint_base(
    *,
    payload: Mapping[str, Any],
    checkpoint_path: Path,
    model_spec: Any,
    readout_mode: str,
    prep_profile: str,
    seed: int,
    manifest: Mapping[str, Any],
    psd_axis_kind: str,
) -> dict[str, Any]:
    run_id = f'{payload.get("dataset_token", "dataset")}_{model_spec.canonical_token}_{readout_mode}_2d_fft_seed{seed}'
    base = psd_common._checkpoint_common_base(
        payload=payload,
        checkpoint_path=checkpoint_path,
        model_spec=model_spec,
        readout_mode=readout_mode,
        run_id=run_id,
        prep_profile=prep_profile,
        seed=seed,
    )
    base['source_program'] = SOURCE_PROGRAM
    base.update(psd_common._axis_metadata_columns(manifest, psd_axis_kind=psd_axis_kind))
    return base


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parse_args_with_config(parser, argv=argv, stage_key='fft2d_analysis')
    if int(args.anal_batch) < 1:
        parser.error('--anal_batch must be >= 1.')
    if int(args.num_workers) < 0:
        parser.error('--num_workers must be >= 0.')
    _load_runtime_dependencies()
    psd_common.LOW_VRAM = bool(int(args.low_vram))

    output_root = timestamped_output_root(args.output_root, run_timestamp=getattr(args, 'run_timestamp', None), prefix=SOURCE_PROGRAM, enabled=getattr(args, 'timestamped_output', True))
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_input = Path(args.checkpoint).expanduser().resolve()
    input_is_single_file = checkpoint_input.is_file()
    checkpoint_files, ordering_warnings = psd_common._resolve_checkpoint_files(checkpoint_input)
    device = psd_common._require_cuda_device(int(args.gpu_index))

    manifest_rows: list[dict[str, str]] = []
    first_manifest_base: dict[str, Any] | None = None

    for checkpoint_path in tqdm(checkpoint_files, desc='2d_fft_analysis:checkpoints', leave=False):
        payload = psd_common._load_checkpoint(checkpoint_path, map_location='cpu')
        seed = int(args.seed if args.seed is not None else payload.get('seed', 0))
        psd_common._seed_everything(seed)
        bundle = psd_common._resolve_bundle(payload, cli_dataset=args.dataset, cli_prep_root=args.prep_root)
        manifest = psd_common._manifest_dict(bundle.manifest_path)
        psd_common._validate_axis_metadata(manifest, payload)
        model, _readout, model_spec, readout_mode = psd_common._build_model_from_checkpoint(payload, device=device)
        validate_mlp_only(model_spec)
        prep_profile = str(manifest.get('prep_profile', manifest.get('psd_axis_kind', bundle.psd_axis_kind)))
        checkpoint_base = _checkpoint_base(
            payload=payload,
            checkpoint_path=checkpoint_path,
            model_spec=model_spec,
            readout_mode=readout_mode,
            prep_profile=prep_profile,
            seed=seed,
            manifest=manifest,
            psd_axis_kind=bundle.psd_axis_kind,
        )
        if first_manifest_base is None:
            first_manifest_base = dict(checkpoint_base)

        maps_by_key: dict[ProbeMapKey, torch.Tensor] = {}
        for split_name, split_dataset in (('train', bundle.train_dataset), ('test', bundle.test_dataset)):
            analysis_dataset = dataset_for_view(split_dataset, bundle.training_view_name)
            maps_by_key.update(
                collect_mlp_output_maps(
                    model=model,
                    dataset=analysis_dataset,
                    split_name=split_name,
                    seed=seed,
                    anal_batch=int(args.anal_batch),
                    num_workers=int(args.num_workers),
                    device=device,
                )
            )

        checkpoint_dir = checkpoint_output_dir(
            output_root=output_root,
            checkpoint_path=checkpoint_path,
            checkpoint_payload=payload,
            input_is_single_file=input_is_single_file,
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for key, maps in tqdm(sorted(maps_by_key.items(), key=lambda item: (item[0][1], item[0][0], item[0][4], item[0][6] if item[0][6] is not None else -1, item[0][3])), desc='2d_fft_analysis:artifacts', leave=False):
            for variant in VARIANTS:
                for scale in SCALES:
                    _write_2d_fft_artifact(
                        checkpoint_dir=checkpoint_dir,
                        checkpoint_base=checkpoint_base,
                        key=key,
                        maps=maps,
                        variant=variant,
                        scale=scale,
                        manifest_rows=manifest_rows,
                    )

    manifest_base = first_manifest_base or {'source_program': SOURCE_PROGRAM, 'dataset': str(args.dataset), 'run_id': '2d_fft_analysis'}
    for warning in ordering_warnings:
        manifest_rows.append(manifest_row(base=dict(manifest_base), artifact_name='checkpoint_ordering', path=Path(args.checkpoint), status='ok', message=warning))
    write_manifest_yaml(output_root / 'analysis_manifest.yaml', manifest_rows)
    print(json.dumps({'status': 'ok', 'source_program': SOURCE_PROGRAM, 'output_root': str(output_root), 'checkpoints': [str(p) for p in checkpoint_files]}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
