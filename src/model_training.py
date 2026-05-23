"""Supervised training entrypoint for the split PSD pipeline.

This program trains and evaluates a model, writes selected `.pt` checkpoints,
and records scalar training metrics in the common long-form CSV schema. It does
not run model/layer PSD analysis and does not render figures.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import shutil
import random
from typing import Any, Sequence

import numpy as np
import torch
from tqdm import tqdm

from src.data.registry import make_loader, resolve_dataset_bundle, select_training_view_for_model
from src.model.model_registry import ModelSpec, canonicalize_model_token
from src.model.training import build_optimizer, evaluate_one_epoch, train_one_epoch
from src.model.snn_builder import build_snn_classifier
from src.readout.readout import build_readout
from src.util.config import load_json
from src.util.csv_schema import common_row, write_common_csv

CHECKPOINT_SCHEMA_VERSION = 'psd_checkpoint_v1'
SOURCE_PROGRAM = 'model_training'


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Supervised model training entrypoint for selected checkpoint production.')
    parser.add_argument('--dataset', required=True, help='Canonical dataset token.')
    parser.add_argument('--prep_root', required=True, help='Prepared data root containing <dataset>/manifest.json.')
    parser.add_argument('--model', required=True, help='Canonical model token.')
    parser.add_argument('--hidden_spec', required=True, help='Dense hidden widths or - for fixed-CNN backbone tokens.')
    parser.add_argument('--readout_mode', required=True, choices=('temporal_membrane', 'first_spike', 'max_rate', 'spikegru_max_over_time'))
    parser.add_argument('--epochs', required=True, type=int)
    parser.add_argument('--batch_size', required=True, type=int)
    parser.add_argument('--lr', required=True, type=float)
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader worker process count. Use 0 for single-process loading.')
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--regularization_lambda1', default=0.0, type=float, help='Signed weight for input-to-hidden PSD curve-shape distances.')
    parser.add_argument('--regularization_lambda2', default=0.0, type=float, help='Signed weight for adjacent hidden PSD curve-shape distances, including input-to-first-hidden.')
    parser.add_argument('--regularization_signal', default='y_mem', choices=('y_mem', 'y_spike'))
    parser.add_argument('--regularization_curve_space', default='exact', choices=('exact',))
    parser.add_argument('--regularization_curve_scale', default='raw', choices=('raw', 'db'))
    parser.add_argument('--regularization_centering', default='raw', choices=('raw', 'centered'))
    parser.add_argument('--regularization_reducer', default='mean', choices=('mean', 'median'))
    parser.add_argument('--regularization_distance_metric', default='centered_l2', choices=('centered_l2', 'diff_l2'), help='PSD curve distance for regularization: centered_l2 or unnormalized first-difference L2.')
    parser.add_argument('--anal_epoch_list', nargs='*', default=None, help='Selected checkpoint epochs; empty means final epoch only.')
    parser.add_argument('--checkpoint_root', required=True, help='Directory that will contain selected .pt checkpoint files only.')
    parser.add_argument('--metric_root', required=True, help='Directory for training_metrics.csv; must be outside checkpoint_root.')
    parser.add_argument('--output_root', default=None, help='Optional parent output root recorded in training metadata only.')
    parser.add_argument('--v_th', type=float, default=1.0, help='threshold')
    parser.add_argument('--resume_checkpoint',
                        default=None,
                        help='Optional .pt checkpoint path. Loads model state_dict and continues from checkpoint epoch + 1.',
                    )
    return parser


def _normalize_anal_epoch_list(values: Sequence[str] | None, *, epochs: int) -> list[int]:
    epochs = int(epochs)
    if epochs < 1:
        raise ValueError('--epochs must be >= 1.')
    if values is None or len(values) == 0 or all(str(v).strip() == '' for v in values):
        values = [str(epochs)]
    normalized: set[int] = set()
    for raw in values:
        text = str(raw).strip()
        if text == '':
            continue
        try:
            epoch = int(text)
        except ValueError as exc:
            raise ValueError(f'--anal_epoch_list values must be integers; got {raw!r}.') from exc
        if epoch < 1 or epoch > epochs:
            raise ValueError(f'--anal_epoch_list value {epoch} is outside 1 <= epoch <= epochs ({epochs}).')
        normalized.add(epoch)
    if not normalized:
        normalized.add(epochs)
    return sorted(normalized)


def _resolve_prepared_paths(dataset: str, prep_root: str) -> tuple[Path, Path]:
    root = Path(str(prep_root)).expanduser().resolve()
    dataset_root = (root / str(dataset)).resolve()
    manifest_path = dataset_root / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f'--prep_root must contain <dataset>/manifest.json; missing {manifest_path}')
    manifest = load_json(manifest_path)
    manifest_dataset = str(manifest.get('dataset_name', dataset)) if isinstance(manifest, dict) else str(dataset)
    if manifest_dataset != str(dataset):
        raise ValueError(f'--dataset {dataset!r} does not match manifest dataset_name {manifest_dataset!r}.')
    return root, dataset_root


def _strict_prepare_checkpoint_dir(checkpoint_root: Path) -> None:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(checkpoint_root.iterdir(), key=lambda item: item.name)
    if existing:
        offenders = ', '.join(str(child) for child in existing[:5])
        raise ValueError(
            'Checkpoint root must be empty before training so the completed directory contains only ' 
            f'this run\'s selected .pt files; found {offenders}'
        )


def _assert_clean_checkpoint_dir(checkpoint_root: Path) -> None:
    for child in checkpoint_root.iterdir():
        if child.is_dir():
            raise ValueError(f'Checkpoint directory contains invalid subdirectory: {child}')
        if child.is_file() and child.suffix != '.pt':
            raise ValueError(f'Checkpoint directory contains non-.pt file: {child}')


def _resolve_device(gpu_index: int) -> torch.device:
    if torch.cuda.is_available():
        index = int(gpu_index)
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f'--gpu_index {index} is invalid for {torch.cuda.device_count()} CUDA device(s).')
        torch.cuda.set_device(index)
        return torch.device(f'cuda:{index}')
    return torch.device('cpu')


def _seed_everything(seed: int) -> None:
    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _model_family(spec: ModelSpec) -> str:
    if spec.family in {'cnn_lif', 'cnn_rf'}:
        return 'cnn'
    if spec.family in {'lif', 'rf'}:
        return 'dense_snn'
    return str(spec.family)


def _input_shape_from_manifest(manifest: dict[str, Any], model_spec: ModelSpec) -> list[int] | None:
    if model_spec.family not in {'cnn_lif', 'cnn_rf'}:
        return None
    for key in ('cnn_input_shape', 'original_shape', 'stored_shape'):
        value = manifest.get(key)
        if isinstance(value, (list, tuple)):
            return [int(v) for v in value]
    return None


def _hidden_sizes_for_model(model_spec: ModelSpec, default_hidden_sizes: Sequence[int]) -> tuple[int, ...] | None:
    if model_spec.family in {'cnn_lif', 'cnn_rf', 'spikingssm', 'spikformer', 'spikegru'}:
        return None
    return tuple(int(v) for v in default_hidden_sizes)


def _hidden_spec_normalized(model_spec: ModelSpec, model_metadata: dict[str, Any]) -> str | None:
    if model_spec.family in {'cnn_lif', 'cnn_rf'}:
        return None
    value = model_metadata.get('hidden_spec') or model_metadata.get('arch_spec')
    return None if value is None else str(value)


def _read_manifest(bundle_manifest: Path) -> dict[str, Any]:
    value = load_json(bundle_manifest)
    if not isinstance(value, dict):
        raise ValueError(f'Prepared manifest must be a JSON object: {bundle_manifest}')
    return value


def _checkpoint_payload(
    *,
    epoch: int,
    model: torch.nn.Module,
    model_spec: ModelSpec,
    model_config: dict[str, Any],
    readout_config: dict[str, Any],
    dataset_token: str,
    prep_root: Path,
    prepared_dataset_path: Path,
    prepared_data_ref: dict[str, Any],
    axis_metadata_ref: dict[str, Any],
    seed: int,
    training_args: dict[str, Any],
    normalization_metadata: dict[str, Any],
    hidden_spec_normalized: str | None,
    metric_snapshot: dict[str, float],
) -> dict[str, Any]:
    return {
        'schema_version': CHECKPOINT_SCHEMA_VERSION,
        'epoch': int(epoch),
        'model_token': model_spec.canonical_token,
        'model_config': dict(model_config),
        'state_dict': model.state_dict(),
        'readout_config': dict(readout_config),
        'dataset_token': str(dataset_token),
        'prep_root': str(prep_root),
        'prepared_dataset_path': str(prepared_dataset_path),
        'prepared_data_ref': dict(prepared_data_ref),
        'axis_metadata_ref': dict(axis_metadata_ref),
        'seed': int(seed),
        'training_args': dict(training_args),
        'normalization_metadata': dict(normalization_metadata),
        'hidden_spec_normalized': hidden_spec_normalized,
        'metric_snapshot': dict(metric_snapshot),
    }


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix='checkpoint_write_', dir=str(path.parent.parent)))
    temp_path = temp_dir / path.name
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    except Exception:
        try:
            if temp_path.exists():
                temp_path.unlink()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    shutil.rmtree(temp_dir, ignore_errors=True)


def _training_metric_rows(
    *,
    run_id: str,
    dataset: str,
    prep_profile: str,
    seed: int,
    model_spec: ModelSpec,
    readout_mode: str,
    epoch: int,
    scope: str,
    metrics: dict[str, float | int],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for metric_name, value in metrics.items():
        rows.append(
            common_row(
                category='training_metric',
                source_program=SOURCE_PROGRAM,
                run_id=run_id,
                dataset=dataset,
                scope=scope,
                seed=seed,
                model_token=model_spec.canonical_token,
                model_family=_model_family(model_spec),
                readout_mode=readout_mode,
                epoch=epoch,
                metric=metric_name,
                value=value,
            )
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if int(args.batch_size) < 1:
        parser.error('--batch_size must be >= 1.')
    if float(args.lr) <= 0.0:
        parser.error('--lr must be > 0.')
    if int(args.num_workers) < 0:
        parser.error('--num_workers must be >= 0.')
    try:
        anal_epochs = _normalize_anal_epoch_list(args.anal_epoch_list, epochs=int(args.epochs))
    except ValueError as exc:
        parser.error(str(exc))
    dataset_token = str(args.dataset)
    prep_root, prepared_dataset_path = _resolve_prepared_paths(dataset_token, args.prep_root)
    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    metric_root = Path(args.metric_root).expanduser().resolve()
    resume_checkpoint = None if args.resume_checkpoint is None else Path(args.resume_checkpoint).expanduser().resolve()

    if checkpoint_root == metric_root or checkpoint_root in metric_root.parents:
        parser.error('--metric_root must be outside --checkpoint_root.')

    if resume_checkpoint is None:
        _strict_prepare_checkpoint_dir(checkpoint_root)
    else:
        if not resume_checkpoint.exists():
            parser.error(f'--resume_checkpoint does not exist: {resume_checkpoint}')
        if resume_checkpoint.suffix != '.pt':
            parser.error(f'--resume_checkpoint must be a .pt file: {resume_checkpoint}')

        # resume 때는 기존 checkpoint들이 남아 있어도 허용해야 함
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        _assert_clean_checkpoint_dir(checkpoint_root)

    metric_root.mkdir(parents=True, exist_ok=True)

    _seed_everything(int(args.seed))
    device = _resolve_device(int(args.gpu_index))
    model_spec = canonicalize_model_token(args.model)
    bundle = resolve_dataset_bundle(dataset_token, prep_root=prep_root)
    bundle = select_training_view_for_model(bundle, model_family=model_spec.family)
    manifest = _read_manifest(bundle.manifest_path)
    input_shape = _input_shape_from_manifest(manifest, model_spec)

    hidden_spec_text = str(args.hidden_spec).strip()
    hidden_sizes = _hidden_sizes_for_model(model_spec, bundle.default_hidden_sizes)
    effective_readout_mode = 'spikegru_max_over_time' if model_spec.family == 'spikegru' else str(args.readout_mode)
    readout = build_readout(effective_readout_mode, num_classes=bundle.num_classes, sequence_length=bundle.sequence_length, device=device)
    model = build_snn_classifier(
        model_token=model_spec,
        input_dim=bundle.input_dim,
        sequence_length=bundle.sequence_length,
        num_classes=bundle.num_classes,
        input_shape=input_shape,
        hidden_sizes=hidden_sizes,
        arch_spec=hidden_spec_text,
        output_layer_overrides=readout.output_layer_overrides(),
        v_th=float(args.v_th),
    ).to(device)
    readout.to(device)
    optimizer = build_optimizer(model, lr=float(args.lr))

    resume_epoch = 0

    if resume_checkpoint is not None:
        try:
            payload = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        except TypeError:
            payload = torch.load(resume_checkpoint, map_location=device)

        if not isinstance(payload, dict):
            raise ValueError(f'Resume checkpoint must be a dict: {resume_checkpoint}')

        if 'state_dict' not in payload:
            raise ValueError(f'Resume checkpoint is missing state_dict: {resume_checkpoint}')

        if 'epoch' not in payload:
            raise ValueError(f'Resume checkpoint is missing epoch: {resume_checkpoint}')

        # 안전장치: 다른 실험 checkpoint를 잘못 넣는 것 방지
        ckpt_dataset = str(payload.get('dataset_token', '')).strip()
        ckpt_model = str(payload.get('model_token', '')).strip()
        ckpt_readout = str(payload.get('readout_config', {}).get('mode', '')).strip()

        if ckpt_dataset and ckpt_dataset != dataset_token:
            raise ValueError(f'Resume dataset mismatch: checkpoint={ckpt_dataset}, current={dataset_token}')

        if ckpt_model and ckpt_model != model_spec.canonical_token:
            raise ValueError(f'Resume model mismatch: checkpoint={ckpt_model}, current={model_spec.canonical_token}')

        if ckpt_readout and ckpt_readout != effective_readout_mode:
            raise ValueError(f'Resume readout mismatch: checkpoint={ckpt_readout}, current={effective_readout_mode}')

        model.load_state_dict(payload['state_dict'])
        resume_epoch = int(payload['epoch'])

        if resume_epoch >= int(args.epochs):
            raise ValueError(
                f'Resume checkpoint epoch={resume_epoch} is already >= target --epochs={int(args.epochs)}. '
                'Increase --epochs or use an earlier checkpoint.'
            )

        tqdm.write(
            f'[model_training] resumed from {resume_checkpoint} at epoch {resume_epoch}; '
            f'next epoch = {resume_epoch + 1}'
        )

    train_loader = make_loader(
        bundle.train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=device.type == 'cuda',
        seed=int(args.seed),
    )
    test_loader = make_loader(
        bundle.test_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == 'cuda',
        seed=int(args.seed),
    )

    training_args = {
        'dataset': dataset_token,
        'prep_root': str(prep_root),
        'prepared_dataset_path': str(prepared_dataset_path),
        'model': model_spec.canonical_token,
        'hidden_spec': None if model_spec.family in {'cnn_lif', 'cnn_rf'} else hidden_spec_text,
        'readout_mode': effective_readout_mode,
        'epochs': int(args.epochs),
        'batch_size': int(args.batch_size),
        'lr': float(args.lr),
        'seed': int(args.seed),
        'gpu_index': int(args.gpu_index),
        'num_workers': int(args.num_workers),
        'regularization_lambda1': float(args.regularization_lambda1),
        'regularization_lambda2': float(args.regularization_lambda2),
        'regularization_signal': str(args.regularization_signal),
        'regularization_curve_space': str(args.regularization_curve_space),
        'regularization_curve_scale': str(args.regularization_curve_scale),
        'regularization_centering': str(args.regularization_centering),
        'regularization_reducer': str(args.regularization_reducer),
        'regularization_distance_metric': str(args.regularization_distance_metric),
        'anal_epoch_list': list(anal_epochs),
        'checkpoint_root': str(checkpoint_root),
        'metric_root': str(metric_root),
        'output_root': '' if args.output_root is None else str(Path(args.output_root).expanduser().resolve()),
    }
    prep_profile = str(manifest.get('prep_profile', manifest.get('psd_axis_kind', bundle.psd_axis_kind)))
    run_id = f'{dataset_token}_{model_spec.canonical_token}_{effective_readout_mode}_seed{int(args.seed)}'
    model_metadata = model.model_metadata() if hasattr(model, 'model_metadata') else {}
    hidden_spec_normalized = _hidden_spec_normalized(model_spec, model_metadata)
    model_config = {
        'input_dim': int(bundle.input_dim),
        'sequence_length': int(bundle.sequence_length),
        'num_classes': int(bundle.num_classes),
        'input_shape': input_shape,
        'hidden_spec': hidden_spec_normalized,
        'arch_spec': str(model_metadata.get('arch_spec', hidden_spec_text)),
        'v_th': float(args.v_th),
        'model_metadata': model_metadata,
    }
    readout_config = {'mode': effective_readout_mode, 'num_classes': int(bundle.num_classes), 'sequence_length': int(bundle.sequence_length)}
    prepared_data_ref = {'prep_root': str(prep_root), 'prepared_dataset_path': str(prepared_dataset_path), 'manifest_path': str(bundle.manifest_path)}
    axis_metadata_ref = {
        'manifest_path': str(bundle.manifest_path),
        'training_view_name': bundle.training_view_name,
        'psd_view_name': bundle.psd_view_name,
        'psd_axis_kind': bundle.psd_axis_kind,
        'psd_sample_axis': manifest.get('psd_sample_axis'),
        'psd_batch_axis': manifest.get('psd_batch_axis'),
        'psd_time_axis': manifest.get('psd_time_axis'),
        'psd_row_axes': manifest.get('psd_row_axes'),
        'psd_feature_axes': manifest.get('psd_feature_axes'),
        'psd_token_axes': manifest.get('psd_token_axes'),
        'psd_flatten_rule': manifest.get('psd_flatten_rule'),
        'psd_logical_shape': manifest.get('psd_logical_shape'),
    }
    normalization_metadata = {
        'stored_dtype': manifest.get('stored_dtype'),
        'label_dtype': manifest.get('label_dtype', 'int64'),
        'sample_index_dtype': manifest.get('sample_index_dtype', 'int64'),
        'model_input_axis_order': manifest.get('model_input_axis_order'),
        'stored_order_is_model_input_order': manifest.get('stored_order_is_model_input_order'),
    }

    all_metric_rows: list[dict[str, str]] = []
    latest_snapshot: dict[str, float] = {}
    for epoch in range(resume_epoch + 1, int(args.epochs) + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            readout=readout,
            optimizer=optimizer,
            device=device,
            progress_desc=f'train epoch {epoch}',
            regularization_lambda1=float(args.regularization_lambda1),
            regularization_lambda2=float(args.regularization_lambda2),
            regularization_signal=str(args.regularization_signal),
            regularization_curve_space=str(args.regularization_curve_space),
            regularization_curve_scale=str(args.regularization_curve_scale),
            regularization_centering=str(args.regularization_centering),
            regularization_reducer=str(args.regularization_reducer),
            regularization_distance_metric=str(args.regularization_distance_metric),
        )

        if hasattr(model, 'clamp_projected_parameters'):
            model.clamp_projected_parameters()

        # 추가: 분석 epoch이 아니면 기록/평가/체크포인트 전부 생략
        if epoch not in anal_epochs:
            continue

        test_metrics = evaluate_one_epoch(
            model,
            test_loader,
            readout=readout,
            device=device,
            progress_desc=f'test epoch {epoch}',
        )

        train_values = {
            'loss': train_metrics.loss,
            'task_loss': train_metrics.task_loss,
            'regularization_loss': train_metrics.regularization_loss,
            'regularization_global_loss': train_metrics.regularization_global_loss,
            'regularization_adjacent_loss': train_metrics.regularization_adjacent_loss,
            'accuracy': train_metrics.accuracy,
            'correct': train_metrics.correct,
            'total': train_metrics.total,
        }

        test_values = {
            'loss': test_metrics.loss,
            'accuracy': test_metrics.accuracy,
            'correct': test_metrics.correct,
            'total': test_metrics.total,
        }

        tqdm.write(
            (
                f'[model_training] epoch {epoch:04d}/{int(args.epochs):04d} | '
                f'train_loss={train_metrics.loss:.6f} '
                f'train_acc={train_metrics.accuracy:.4f} '
                f'train_task={train_metrics.task_loss:.6f} '
                f'train_reg={train_metrics.regularization_loss:.6f} | '
                f'test_loss={test_metrics.loss:.6f} '
                f'test_acc={test_metrics.accuracy:.4f} '
                f'test_correct={test_metrics.correct}/{test_metrics.total}'
            )
        )

        all_metric_rows.extend(
            _training_metric_rows(
                run_id=run_id,
                dataset=dataset_token,
                prep_profile=prep_profile,
                seed=int(args.seed),
                model_spec=model_spec,
                readout_mode=effective_readout_mode,
                epoch=epoch,
                scope='train',
                metrics=train_values,
            )
        )

        all_metric_rows.extend(
            _training_metric_rows(
                run_id=run_id,
                dataset=dataset_token,
                prep_profile=prep_profile,
                seed=int(args.seed),
                model_spec=model_spec,
                readout_mode=effective_readout_mode,
                epoch=epoch,
                scope='test',
                metrics=test_values,
            )
        )

        latest_snapshot = {
            'train_loss': float(train_metrics.loss),
            'train_task_loss': float(train_metrics.task_loss),
            'train_regularization_loss': float(train_metrics.regularization_loss),
            'train_regularization_global_loss': float(train_metrics.regularization_global_loss),
            'train_regularization_adjacent_loss': float(train_metrics.regularization_adjacent_loss),
            'train_accuracy': float(train_metrics.accuracy),
            'test_loss': float(test_metrics.loss),
            'test_accuracy': float(test_metrics.accuracy),
        }

        checkpoint_path = checkpoint_root / f'checkpoint_epoch_{epoch:06d}.pt'
        payload = _checkpoint_payload(
            epoch=epoch,
            model=model,
            model_spec=model_spec,
            model_config=model_config,
            readout_config=readout_config,
            dataset_token=dataset_token,
            prep_root=prep_root,
            prepared_dataset_path=prepared_dataset_path,
            prepared_data_ref=prepared_data_ref,
            axis_metadata_ref=axis_metadata_ref,
            seed=int(args.seed),
            training_args=training_args,
            normalization_metadata=normalization_metadata,
            hidden_spec_normalized=hidden_spec_normalized,
            metric_snapshot=latest_snapshot,
        )
        _atomic_torch_save(payload, checkpoint_path)
        _assert_clean_checkpoint_dir(checkpoint_root)

    metrics_path = metric_root / 'training_metrics.csv'
    write_common_csv(metrics_path, all_metric_rows)
    _assert_clean_checkpoint_dir(checkpoint_root)
    print(
        json.dumps(
            {
                'status': 'ok',
                'source_program': SOURCE_PROGRAM,
                'checkpoint_root': str(checkpoint_root),
                'metric_csv': str(metrics_path),
                'selected_epochs': anal_epochs,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
