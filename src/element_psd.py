"""Checkpoint-only element-wise PSD analysis for MLP probe output maps."""

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


from src.util.csv_schema import common_row, write_common_csv
from src.util.config_cli import parse_args_with_config


SOURCE_PROGRAM = 'element_psd'
VARIANTS = ('raw', 'centered')
SCALES = ('raw', 'db')


def _load_runtime_dependencies() -> None:
    """Element PSD 실행에 필요한 무거운 의존성을 지연 로드한다."""

    global torch, tqdm, psd_common
    global ProbeMapKey, checkpoint_output_dir, collect_mlp_output_maps, common_base_for_key
    global layer_folder, manifest_row, safe_token, validate_mlp_only, value_columns, write_matrix_csv
    global dataset_for_view, apply_centering, exact_periodogram_from_maps, power_to_db
    import torch as _torch
    from tqdm import tqdm as _tqdm
    import src.psd_analysis as _psd_common

    _psd_common._load_runtime_dependencies()
    from src.analysis_matrix_common import ProbeMapKey as _ProbeMapKey, checkpoint_output_dir as _checkpoint_output_dir, collect_mlp_output_maps as _collect_mlp_output_maps, common_base_for_key as _common_base_for_key, layer_folder as _layer_folder, manifest_row as _manifest_row, safe_token as _safe_token, validate_mlp_only as _validate_mlp_only, value_columns as _value_columns, write_matrix_csv as _write_matrix_csv
    from src.data.registry import dataset_for_view as _dataset_for_view
    from src.signal.psd_utils import apply_centering as _apply_centering, exact_periodogram_from_maps as _exact_periodogram_from_maps, power_to_db as _power_to_db
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
    apply_centering = _apply_centering
    exact_periodogram_from_maps = _exact_periodogram_from_maps
    power_to_db = _power_to_db


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Checkpoint-only element-wise PSD analysis for MLP probe output maps.')
    parser.add_argument('--checkpoint', required=True, help='Single .pt checkpoint file or strict .pt-only checkpoint directory.')
    parser.add_argument('--dataset', required=True, help='Canonical dataset token stored in checkpoint metadata.')
    parser.add_argument('--prep_root', required=True, help='Prepared data root containing <dataset>/manifest.json.')
    parser.add_argument('--output_root', required=True, help='Root directory for element-wise PSD CSV outputs.')
    parser.add_argument('--anal_batch', required=True, type=int, help='Maximum samples per analysis forward pass.')
    parser.add_argument('--gpu_index', required=True, type=int, help='CUDA device index for analysis.')
    parser.add_argument('--seed', type=int, default=None, help='Analysis seed. Defaults to checkpoint seed when omitted.')
    parser.add_argument('--num_workers', type=int, default=0, help='Probe loading DataLoader worker count.')
    parser.add_argument('--low_vram', type=int, default=0, help='Use CPU trace staging to reduce VRAM use: 0 or 1.')
    parser.add_argument('--config', default=None, help='JSON 설정 파일 경로(.json)')
    return parser


def _variant_maps(maps: torch.Tensor, *, variant: str) -> torch.Tensor:
    token = str(variant)
    if token == 'raw':
        return maps
    if token == 'centered':
        return apply_centering(maps)
    raise ValueError(f'Unsupported element PSD variant: {variant!r}.')


def _element_psd_matrix(maps: torch.Tensor, *, variant: str) -> tuple[torch.Tensor, torch.Tensor]:
    prepared = _variant_maps(maps, variant=variant)
    if prepared.ndim != 3:
        raise ValueError(f'Expected output maps with shape (samples, neurons, time), got {tuple(prepared.shape)}.')
    freqs, psd = exact_periodogram_from_maps(prepared)
    return freqs, psd.mean(dim=0).real.contiguous()


def _scale_power(power: torch.Tensor, *, scale: str) -> torch.Tensor:
    token = str(scale)
    if token == 'raw':
        return power
    if token == 'db':
        return power_to_db(power)
    raise ValueError(f'Unsupported element PSD scale: {scale!r}.')


def _value_unit(scale: str) -> str:
    return 'dB' if str(scale) == 'db' else 'power'


def _artifact_name(base: Mapping[str, Any], key: ProbeMapKey, *, variant: str, scale: str) -> str:
    layer_name, layer_index, signal_kind, series, scope, _probe_family, label, sample_role, _sample_index = key
    label_part = 'all_labels' if label is None else f'label_{int(label)}'
    return (
        f'element_psd__epoch_{safe_token(base.get("checkpoint_epoch", "epoch"))}'
        f'__layer_{safe_token(layer_index)}__{safe_token(layer_name)}'
        f'__{safe_token(scope)}__{label_part}'
        f'__{safe_token(signal_kind)}__{safe_token(series)}'
        f'__{safe_token(sample_role)}__{safe_token(variant)}__{safe_token(scale)}.csv'
    )


def _rows_for_element_matrix(
    *,
    base: Mapping[str, Any],
    matrix: torch.Tensor,
    time_length: int,
    variant: str,
    scale: str,
    value_column_names: Sequence[str],
) -> list[dict[str, Any]]:
    if matrix.ndim != 2:
        raise ValueError(f'Expected element PSD matrix with shape (neurons, frequencies), got {tuple(matrix.shape)}.')
    neuron_count = int(matrix.shape[0])
    frequency_bin_count = int(matrix.shape[1])
    values = matrix.detach().cpu().to(dtype=torch.float64)
    rows: list[dict[str, Any]] = []
    for neuron_index in range(neuron_count):
        row_kwargs = dict(base)
        row_kwargs.update(
            category='element_psd',
            variant=variant,
            scale=scale,
            neuron_index=neuron_index,
            neuron_axis_order='dense_layer_output_index_zero_based',
            time_length=int(time_length),
            frequency_bin_count=frequency_bin_count,
            frequency_grid='exact_one_sided_index_over_time_length',
            frequency_unit='normalized_frequency',
            value_unit=_value_unit(scale),
        )
        row = common_row(**row_kwargs)
        for column_index, column_name in enumerate(value_column_names):
            row[column_name] = float(values[neuron_index, column_index].item())
        rows.append(row)
    return rows


def _write_element_psd_artifact(
    *,
    checkpoint_dir: Path,
    checkpoint_base: Mapping[str, Any],
    key: ProbeMapKey,
    maps: torch.Tensor,
    variant: str,
    scale: str,
    manifest_rows: list[dict[str, str]],
) -> None:
    _freqs, raw_matrix = _element_psd_matrix(maps, variant=variant)
    scaled_matrix = _scale_power(raw_matrix, scale=scale)
    value_column_names = value_columns('freq', int(scaled_matrix.shape[1]))
    base = common_base_for_key(checkpoint_base=checkpoint_base, key=key)
    rows = _rows_for_element_matrix(
        base=base,
        matrix=scaled_matrix,
        time_length=int(maps.shape[-1]),
        variant=variant,
        scale=scale,
        value_column_names=value_column_names,
    )
    layer_name, layer_index, *_rest = key
    out_path = checkpoint_dir / 'layers' / layer_folder(layer_name, layer_index) / 'element_psd' / _artifact_name(base, key, variant=variant, scale=scale)
    write_matrix_csv(out_path, rows, value_column_names=value_column_names)
    manifest_rows.append(manifest_row(base=dict(checkpoint_base), artifact_name='element_psd', path=out_path))


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
    run_id = f'{payload.get("dataset_token", "dataset")}_{model_spec.canonical_token}_{readout_mode}_element_psd_seed{seed}'
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
    args = parse_args_with_config(parser, argv=argv, stage_key='element_psd')
    if int(args.anal_batch) < 1:
        parser.error('--anal_batch must be >= 1.')
    if int(args.num_workers) < 0:
        parser.error('--num_workers must be >= 0.')
    _load_runtime_dependencies()
    psd_common.LOW_VRAM = bool(int(args.low_vram))

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_input = Path(args.checkpoint).expanduser().resolve()
    input_is_single_file = checkpoint_input.is_file()
    checkpoint_files, ordering_warnings = psd_common._resolve_checkpoint_files(checkpoint_input)
    device = psd_common._require_cuda_device(int(args.gpu_index))

    manifest_rows: list[dict[str, str]] = []
    first_manifest_base: dict[str, Any] | None = None

    for checkpoint_path in tqdm(checkpoint_files, desc='element_psd:checkpoints', leave=False):
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
        for key, maps in tqdm(sorted(maps_by_key.items(), key=lambda item: (item[0][1], item[0][0], item[0][4], item[0][6] if item[0][6] is not None else -1, item[0][3])), desc='element_psd:artifacts', leave=False):
            for variant in VARIANTS:
                for scale in SCALES:
                    _write_element_psd_artifact(
                        checkpoint_dir=checkpoint_dir,
                        checkpoint_base=checkpoint_base,
                        key=key,
                        maps=maps,
                        variant=variant,
                        scale=scale,
                        manifest_rows=manifest_rows,
                    )

    manifest_base = first_manifest_base or {'source_program': SOURCE_PROGRAM, 'dataset': str(args.dataset), 'run_id': 'element_psd'}
    for warning in ordering_warnings:
        manifest_rows.append(manifest_row(base=dict(manifest_base), artifact_name='checkpoint_ordering', path=Path(args.checkpoint), status='ok', message=warning))
    write_common_csv(output_root / 'analysis_manifest.csv', manifest_rows)
    print(json.dumps({'status': 'ok', 'source_program': SOURCE_PROGRAM, 'output_root': str(output_root), 'checkpoints': [str(p) for p in checkpoint_files]}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
