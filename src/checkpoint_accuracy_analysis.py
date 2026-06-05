"""Checkpoint-only train/test accuracy analysis entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import src.psd_analysis as psd_common
from src.data.registry import make_loader
from src.model.training import EpochMetrics, evaluate_one_epoch
from src.util.csv_schema import common_row, write_common_csv, write_manifest_yaml
from src.util.paths import timestamped_output_root

SOURCE_PROGRAM = 'checkpoint_accuracy_analysis'


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Checkpoint-only train/test accuracy analysis entrypoint.')
    parser.add_argument('--checkpoint', required=True, help='Single .pt checkpoint file or strict .pt-only checkpoint directory.')
    parser.add_argument('--dataset', required=True, help='Canonical dataset token stored in the checkpoint metadata.')
    parser.add_argument('--prep_root', required=True, help='Prepared data root containing <dataset>/manifest.yaml.')
    parser.add_argument('--output_root', required=True, help='Root directory for checkpoint accuracy CSV outputs.')
    parser.add_argument('--anal_batch', required=True, type=int, help='Maximum samples per evaluation forward pass.')
    parser.add_argument('--gpu_index', required=True, type=int, help='CUDA device index for accuracy evaluation.')
    parser.add_argument('--seed', type=int, default=None, help='Evaluation seed. Defaults to checkpoint seed when omitted.')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader worker count.')
    parser.add_argument('--splits', nargs='+', choices=('train', 'test'), default=('train', 'test'), help='Dataset splits to evaluate. Default: train test.')
    parser.add_argument('--run_timestamp', default=None, help='Execution timestamp suffix for the output run directory. Defaults to Asia/Seoul current time.')
    parser.add_argument('--timestamped_output', default='true', help='true이면 output_root 아래 실행시각 폴더를 자동 생성한다. false이면 기존 경로에 직접 저장한다.')
    return parser


def _checkpoint_seed(payload: Mapping[str, Any]) -> int:
    value = payload.get('seed', payload.get('training_args', {}).get('seed', 0))
    return int(value)


def _checkpoint_epoch_value(payload: Mapping[str, Any], checkpoint_path: Path) -> tuple[int, str]:
    epoch, warning = psd_common._checkpoint_epoch(payload, checkpoint_path)
    if epoch is None:
        return -1, warning
    return int(epoch), warning


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
    dataset_token = str(payload.get('dataset_token') or payload.get('training_args', {}).get('dataset') or 'dataset')
    run_id = f'{dataset_token}_{model_spec.canonical_token}_{readout_mode}_checkpoint_accuracy_seed{seed}'
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


def _accuracy_row(*, base: Mapping[str, Any], scope: str, metrics: EpochMetrics) -> dict[str, str]:
    kwargs = dict(base)
    kwargs.update(
        category='checkpoint_accuracy',
        scope=str(scope),
        accuracy=float(metrics.accuracy),
        correct=int(metrics.correct),
        total=int(metrics.total),
        value_unit='fraction',
    )
    return common_row(**kwargs)


def _evaluate_split(
    *,
    model: torch.nn.Module,
    readout: torch.nn.Module,
    dataset: Any,
    split_name: str,
    anal_batch: int,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> EpochMetrics:
    loader = make_loader(
        dataset,
        batch_size=int(anal_batch),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.type == 'cuda',
        seed=int(seed),
    )
    return evaluate_one_epoch(
        model,
        loader,
        readout=readout,
        device=device,
        progress_desc=f'accuracy {split_name}',
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if int(args.anal_batch) < 1:
        parser.error('--anal_batch must be >= 1.')
    if int(args.num_workers) < 0:
        parser.error('--num_workers must be >= 0.')

    output_root = timestamped_output_root(args.output_root, run_timestamp=getattr(args, 'run_timestamp', None), prefix=SOURCE_PROGRAM, enabled=getattr(args, 'timestamped_output', True))
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_input = Path(args.checkpoint).expanduser().resolve()
    checkpoint_files, ordering_warnings = psd_common._resolve_checkpoint_files(checkpoint_input)
    device = psd_common._require_cuda_device(int(args.gpu_index))

    rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, str]] = []
    first_manifest_base: dict[str, Any] | None = None
    checkpoint_summaries: list[dict[str, Any]] = []

    for checkpoint_path in checkpoint_files:
        payload = psd_common._load_checkpoint(checkpoint_path, map_location='cpu')
        seed = int(args.seed if args.seed is not None else _checkpoint_seed(payload))
        psd_common._seed_everything(seed)
        bundle = psd_common._resolve_bundle(payload, cli_dataset=args.dataset, cli_prep_root=args.prep_root)
        manifest = psd_common._manifest_dict(bundle.manifest_path)
        model, readout, model_spec, readout_mode = psd_common._build_model_from_checkpoint(payload, device=device)
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
        checkpoint_epoch, epoch_warning = _checkpoint_epoch_value(payload, checkpoint_path)
        checkpoint_base['dataset'] = str(bundle.dataset_name)
        checkpoint_base['checkpoint_epoch'] = int(checkpoint_epoch)
        if first_manifest_base is None:
            first_manifest_base = dict(checkpoint_base)

        split_to_dataset = {
            'train': bundle.train_dataset,
            'test': bundle.test_dataset,
        }
        summary: dict[str, Any] = {
            'checkpoint_path': str(checkpoint_path),
            'checkpoint_epoch': int(checkpoint_epoch),
        }
        if epoch_warning:
            summary['warning'] = epoch_warning

        for split_name in args.splits:
            metrics = _evaluate_split(
                model=model,
                readout=readout,
                dataset=split_to_dataset[str(split_name)],
                split_name=str(split_name),
                anal_batch=int(args.anal_batch),
                num_workers=int(args.num_workers),
                seed=seed,
                device=device,
            )
            rows.append(_accuracy_row(base=checkpoint_base, scope=str(split_name), metrics=metrics))
            summary[f'{split_name}_accuracy'] = float(metrics.accuracy)
            summary[f'{split_name}_correct'] = int(metrics.correct)
            summary[f'{split_name}_total'] = int(metrics.total)

        checkpoint_summaries.append(summary)
        del model, readout
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    csv_path = output_root / 'checkpoint_accuracy.csv'
    write_common_csv(csv_path, rows)
    manifest_base = first_manifest_base or {'source_program': SOURCE_PROGRAM, 'dataset': str(args.dataset), 'run_id': 'checkpoint_accuracy_analysis'}
    manifest_rows.append(psd_common._manifest_row(base=dict(manifest_base), artifact_name='checkpoint_accuracy', path=csv_path))
    for warning in ordering_warnings:
        manifest_rows.append(psd_common._manifest_row(base=dict(manifest_base), artifact_name='checkpoint_ordering', path=Path(args.checkpoint), status='ok', message=warning))
    write_manifest_yaml(output_root / 'analysis_manifest.yaml', manifest_rows)

    print(json.dumps({
        'status': 'ok',
        'source_program': SOURCE_PROGRAM,
        'output_root': str(output_root),
        'csv_path': str(csv_path),
        'checkpoint_count': len(checkpoint_files),
        'checkpoints': checkpoint_summaries,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
