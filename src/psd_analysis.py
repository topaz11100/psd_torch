"""Checkpoint-only PSD analysis entrypoint for prepared probe sets."""

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
import itertools
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


from src.util.csv_schema import common_row, write_common_csv
from src.util.config_cli import parse_args_with_config
from src.util.cli_common import parse_bool_token


SOURCE_PROGRAM = 'psd_analysis'
PCA_ANALYSIS_SCHEMA_VERSION = 1


def _parse_bool_config_value(value: Any, *, default: bool) -> bool:
    return parse_bool_token(value, default=default)


def _load_json_light(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
ALL_CURVE_EXTRACTORS = ('psd_exact',)
ALL_VALUE_SCALES = ('raw', 'db')
LOW_VRAM = False


def _load_runtime_dependencies() -> None:
    """PSD 분석 실행에 필요한 무거운 의존성을 지연 로드한다."""

    global np, torch, tqdm, seed_everything
    global canonicalize_model_input_batch
    global dataset_for_view, make_loader, resolve_dataset_bundle, select_training_view_for_model
    global ModelSpec, canonicalize_model_token, build_snn_classifier, build_readout, canonicalize_readout_mode
    global compute_family_spectral_summary, curve_axis_from_summary, curve_pointwise_distance
    global pair_distance_from_summaries, representative_curve_from_summary
    global trace_tensor_to_channel_major_maps, pca_dim_from_cli_vector, compute_fixed_pca_basis, apply_fixed_pca_basis
    global auto_spectral_matrix_from_mode_maps, cross_spectral_matrix_from_mode_maps
    global build_probe_index_bundle, build_probe_scopes, dataset_targets, subset_from_indices
    import numpy as _np
    import torch as _torch
    from tqdm import tqdm as _tqdm
    from src.util.random import seed_everything as _seed_everything
    from src.data.base import canonicalize_model_input_batch as _canonicalize_model_input_batch
    from src.data.registry import dataset_for_view as _dataset_for_view, make_loader as _make_loader, resolve_dataset_bundle as _resolve_dataset_bundle, select_training_view_for_model as _select_training_view_for_model
    from src.model.model_registry import ModelSpec as _ModelSpec, canonicalize_model_token as _canonicalize_model_token
    from src.model.snn_builder import build_snn_classifier as _build_snn_classifier
    from src.readout.readout import build_readout as _build_readout, canonicalize_readout_mode as _canonicalize_readout_mode
    from src.signal.family_spectral_analysis import compute_family_spectral_summary as _compute_family_spectral_summary, curve_axis_from_summary as _curve_axis_from_summary, curve_pointwise_distance as _curve_pointwise_distance, pair_distance_from_summaries as _pair_distance_from_summaries, representative_curve_from_summary as _representative_curve_from_summary
    from src.signal.psd_utils import (
        trace_tensor_to_channel_major_maps as _trace_tensor_to_channel_major_maps,
        pca_dim_from_cli_vector as _pca_dim_from_cli_vector,
        compute_fixed_pca_basis as _compute_fixed_pca_basis,
        apply_fixed_pca_basis as _apply_fixed_pca_basis,
        auto_spectral_matrix_from_mode_maps as _auto_spectral_matrix_from_mode_maps,
        cross_spectral_matrix_from_mode_maps as _cross_spectral_matrix_from_mode_maps,
    )
    from src.stat.probe_selection import build_probe_index_bundle as _build_probe_index_bundle, build_probe_scopes as _build_probe_scopes, dataset_targets as _dataset_targets, subset_from_indices as _subset_from_indices
    np = _np
    torch = _torch
    tqdm = _tqdm
    seed_everything = _seed_everything
    canonicalize_model_input_batch = _canonicalize_model_input_batch
    dataset_for_view = _dataset_for_view
    make_loader = _make_loader
    resolve_dataset_bundle = _resolve_dataset_bundle
    select_training_view_for_model = _select_training_view_for_model
    ModelSpec = _ModelSpec
    canonicalize_model_token = _canonicalize_model_token
    build_snn_classifier = _build_snn_classifier
    build_readout = _build_readout
    canonicalize_readout_mode = _canonicalize_readout_mode
    compute_family_spectral_summary = _compute_family_spectral_summary
    curve_axis_from_summary = _curve_axis_from_summary
    curve_pointwise_distance = _curve_pointwise_distance
    pair_distance_from_summaries = _pair_distance_from_summaries
    representative_curve_from_summary = _representative_curve_from_summary
    trace_tensor_to_channel_major_maps = _trace_tensor_to_channel_major_maps
    pca_dim_from_cli_vector = _pca_dim_from_cli_vector
    compute_fixed_pca_basis = _compute_fixed_pca_basis
    apply_fixed_pca_basis = _apply_fixed_pca_basis
    auto_spectral_matrix_from_mode_maps = _auto_spectral_matrix_from_mode_maps
    cross_spectral_matrix_from_mode_maps = _cross_spectral_matrix_from_mode_maps
    build_probe_index_bundle = _build_probe_index_bundle; build_probe_scopes = _build_probe_scopes
    dataset_targets = _dataset_targets
    subset_from_indices = _subset_from_indices

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Checkpoint-only PSD analysis entrypoint.')
    parser.add_argument('--checkpoint', required=True, help='Single .pt checkpoint file or strict .pt-only checkpoint directory.')
    parser.add_argument('--dataset', required=True, help='Canonical dataset token stored in the checkpoint metadata.')
    parser.add_argument('--prep_root', required=True, help='Prepared data root containing <dataset>/manifest.json.')
    parser.add_argument('--output_root', required=True, help='Root directory for analysis CSV outputs.')
    parser.add_argument('--anal_batch', required=True, type=int, help='Maximum samples per analysis forward pass.')
    parser.add_argument('--gpu_index', required=True, type=int, help='CUDA device index for analysis.')
    parser.add_argument('--enable_pairwise_dependency_appendix', action='store_true')
    parser.add_argument('--analysis_distance_metric', nargs='*', default=['centered_l2'], choices=('centered_l2', 'diff_l2'), help='PSD curve distance metrics to sweep for pair/layer shape distances.')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--config', default=None, help='JSON 설정 파일 경로(.json)')
    parser.add_argument('--low_vram', type=int, default=0)
    parser.add_argument('--enable_pca_1d', default='false')
    parser.add_argument('--enable_pca_mimo', default='false')
    parser.add_argument('--pca_ref_epoch', type=int, default=None)
    parser.add_argument('--pca_min_train_accuracy', type=float, default=0.0)
    parser.add_argument('--pca_dim_per_layer', nargs='*', default=None)
    parser.add_argument('--psd_curve_tokens', nargs='*', default=None)
    parser.add_argument('--analysis_userbin_edges', nargs='*', default=None)
    parser.add_argument('--analysis_userbin_reducer', nargs='*', default=['mean'], choices=('mean', 'median', 'sum'))
    parser.add_argument('--filter_alpha_userbin_edges', nargs='*', type=float, default=None)
    parser.add_argument('--filter_alpha_userbin_count', type=int, default=10)
    parser.add_argument('--filter_frequency_userbin_edges', nargs='*', type=float, default=None)
    parser.add_argument('--filter_frequency_userbin_count', type=int, default=10)
    parser.add_argument('--filter_damping_userbin_edges', nargs='*', type=float, default=None)
    parser.add_argument('--filter_damping_userbin_count', type=int, default=10)
    return parser


def _load_checkpoint(path: Path, *, map_location: str | torch.device = 'cpu') -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f'Checkpoint must load to a mapping: {path}')
    return payload


def _checkpoint_epoch(payload: Mapping[str, Any], path: Path) -> tuple[int | None, str]:
    try:
        return int(payload['epoch']), ''
    except Exception:
        return None, f'checkpoint {path.name} is missing integer epoch metadata; lexical order was used'


def _resolve_checkpoint_files(path: Path) -> tuple[list[Path], list[str]]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f'--checkpoint path does not exist: {path}')
    warnings: list[str] = []
    if path.is_file():
        if path.suffix != '.pt':
            raise ValueError(f'File mode requires a .pt input: {path}')
        return [path], warnings
    if not path.is_dir():
        raise ValueError(f'--checkpoint must be a .pt file or directory: {path}')
    children = sorted(path.iterdir(), key=lambda item: item.name)
    pt_files: list[Path] = []
    for child in children:
        if child.is_dir():
            raise ValueError(f'Strict checkpoint directory may not contain subdirectories: {child}')
        if child.is_file():
            if child.suffix != '.pt':
                raise ValueError(f'Strict checkpoint directory may contain .pt regular files only; found {child}')
            pt_files.append(child)
    if not pt_files:
        raise ValueError(f'Strict checkpoint directory contains no .pt files: {path}')
    sortable: list[tuple[int, str, Path]] = []
    missing_epoch = False
    for pt_file in pt_files:
        payload = _load_checkpoint(pt_file, map_location='cpu')
        epoch, warning = _checkpoint_epoch(payload, pt_file)
        if warning:
            warnings.append(warning)
            missing_epoch = True
        sortable.append((10**18 if epoch is None else int(epoch), pt_file.name, pt_file))
    if missing_epoch:
        return [item[2] for item in sorted(sortable, key=lambda item: item[1])], warnings
    return [item[2] for item in sorted(sortable, key=lambda item: (item[0], item[1]))], warnings


def _require_cuda_device(gpu_index: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError('--gpu_index was requested, but CUDA is unavailable. The analysis contract has no implicit CPU fallback.')
    index = int(gpu_index)
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(f'--gpu_index {index} is invalid for {torch.cuda.device_count()} CUDA device(s).')
    torch.cuda.set_device(index)
    return torch.device(f'cuda:{index}')


def _seed_everything(seed: int) -> None:
    seed_everything(int(seed))


def _model_family(spec: ModelSpec) -> str:
    if spec.family in {'cnn_lif', 'cnn_rf'}:
        return 'cnn'
    if spec.family in {'lif', 'rf'}:
        return 'dense_snn'
    return str(spec.family)


def _resolve_bundle(
    payload: Mapping[str, Any],
    *,
    cli_dataset: str,
    cli_prep_root: str,
    model_spec: ModelSpec | None = None,
):
    requested_dataset = str(cli_dataset)
    checkpoint_dataset = str(payload.get('dataset_token') or payload.get('training_args', {}).get('dataset') or '')
    if not checkpoint_dataset:
        raise ValueError('Checkpoint is missing dataset_token metadata.')
    if checkpoint_dataset != requested_dataset:
        raise ValueError(f'--dataset {requested_dataset!r} does not match checkpoint dataset_token {checkpoint_dataset!r}.')
    prep_root = Path(cli_prep_root).expanduser().resolve()
    bundle = resolve_dataset_bundle(requested_dataset, prep_root=prep_root)
    resolved_spec = model_spec
    if resolved_spec is None:
        token = str(payload.get('model_token') or payload.get('training_args', {}).get('model') or '')
        if token:
            resolved_spec = canonicalize_model_token(token)
    if resolved_spec is not None:
        bundle = select_training_view_for_model(bundle, model_family=resolved_spec.family)
    return bundle


def _manifest_dict(path: Path) -> dict[str, Any]:
    payload = _load_json_light(path)
    if not isinstance(payload, dict):
        raise ValueError(f'Prepared manifest must be a JSON object: {path}')
    return payload


def _validate_axis_metadata(manifest: Mapping[str, Any], checkpoint_payload: Mapping[str, Any]) -> None:
    axis_ref = checkpoint_payload.get('axis_metadata_ref')
    sources = [manifest]
    if isinstance(axis_ref, Mapping):
        sources.append(axis_ref)
    for key in ('psd_time_axis', 'psd_row_axes', 'psd_flatten_rule', 'psd_logical_shape'):
        if not any(source.get(key) not in (None, '', []) for source in sources):
            raise ValueError(f'Prepared/checkpoint metadata is missing required axis metadata: {key}')




def _json_metadata(value: Any) -> str:
    if value in (None, ''):
        return ''
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _axis_metadata_columns(manifest: Mapping[str, Any], *, psd_axis_kind: str) -> dict[str, str]:
    logical_shape = manifest.get('psd_logical_shape')
    static_repeat_T = ''
    if isinstance(logical_shape, Mapping):
        static_repeat_T = logical_shape.get('T', logical_shape.get('time', ''))
    return {
        'prep_profile': str(manifest.get('prep_profile', psd_axis_kind)),
        'psd_axis_kind': str(manifest.get('psd_axis_kind', psd_axis_kind)),
        'psd_time_axis': str(manifest.get('psd_time_axis', '')),
        'psd_row_axes': _json_metadata(manifest.get('psd_row_axes')),
        'psd_flatten_rule': str(manifest.get('psd_flatten_rule', '')),
        'psd_logical_shape': _json_metadata(logical_shape),
        'static_repeat_T': '' if static_repeat_T is None else str(static_repeat_T),
    }

def _build_model_from_checkpoint(payload: Mapping[str, Any], *, device: torch.device):
    model_token = str(payload.get('model_token') or '')
    if not model_token:
        raise ValueError('Checkpoint is missing model_token.')
    spec = canonicalize_model_token(model_token)
    model_config = payload.get('model_config')
    readout_config = payload.get('readout_config')
    if not isinstance(model_config, Mapping):
        raise ValueError('Checkpoint model_config must be a mapping.')
    if not isinstance(readout_config, Mapping):
        raise ValueError('Checkpoint readout_config must be a mapping.')
    mode = str(readout_config.get('mode') or readout_config.get('readout_mode') or '')
    if not mode:
        raise ValueError('Checkpoint readout_config is missing mode.')
    mode = canonicalize_readout_mode(mode)
    input_dim = int(model_config['input_dim'])
    sequence_length = int(model_config['sequence_length'])
    num_classes = int(model_config['num_classes'])
    input_shape = model_config.get('input_shape')
    if input_shape is not None:
        input_shape = [int(v) for v in input_shape]
    if spec.family in {'cnn_lif', 'cnn_rf'}:
        hidden_spec = '-'
    else:
        hidden_spec = str(model_config.get('hidden_spec') or model_config.get('arch_spec') or '')
    v_th = float(model_config.get('v_th', 1.0))
    readout = build_readout(mode, num_classes=num_classes, sequence_length=sequence_length, device=device)
    model = build_snn_classifier(
        model_token=spec,
        input_dim=input_dim,
        sequence_length=sequence_length,
        num_classes=num_classes,
        input_shape=input_shape,
        hidden_sizes=None,
        arch_spec=hidden_spec,
        output_layer_overrides=readout.output_layer_overrides(),
        v_th=v_th,
    ).to(device)
    state_dict = payload.get('state_dict')
    if not isinstance(state_dict, Mapping):
        raise ValueError('Checkpoint is missing state_dict mapping.')
    model.load_state_dict(state_dict)
    model.eval()
    readout.to(device)
    readout.eval()
    return model, readout, spec, mode


def _prepared_input_for_model(model: torch.nn.Module, inputs: Any, *, device: torch.device) -> torch.Tensor:
    spec_family = getattr(getattr(model, 'spec', None), 'family', None)
    if spec_family in {'cnn_lif', 'cnn_rf'}:
        tensor = torch.as_tensor(inputs)
    else:
        tensor = canonicalize_model_input_batch(inputs, input_dim=int(model.input_dim), sequence_length=int(model.sequence_length))
    return tensor.to(device=device, dtype=torch.float32, non_blocking=True)


def _probe_quota(dataset: Any) -> int:
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
    quota = _probe_quota(dataset)
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

def _trace_to_cpu_maps(tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f'Expected torch.Tensor, got {type(tensor)!r}.')

    cpu_tensor = tensor.detach().to(device='cpu', non_blocking=False)

    if cpu_tensor.dtype != torch.float32:
        cpu_tensor = cpu_tensor.to(dtype=torch.float32)

    return trace_tensor_to_channel_major_maps(cpu_tensor).contiguous()

def _maps_from_record(record: Any) -> list[tuple[str, str, torch.Tensor]]:
    if not LOW_VRAM:
        output: list[tuple[str, str, torch.Tensor]] = []
        explicit_signal_kind = getattr(record, 'signal_kind', None)
        explicit_series = getattr(record, 'series', None)
        if explicit_signal_kind and explicit_series:
            tensor = getattr(record, 'membrane', None)
            if isinstance(tensor, torch.Tensor):
                return [(str(explicit_signal_kind), str(explicit_series), trace_tensor_to_channel_major_maps(tensor))]
        if getattr(record, 'layer_input', None) is not None:
            output.append(('hidden', 'layer_input', trace_tensor_to_channel_major_maps(record.layer_input)))
        output.append(('hidden', 'membrane', trace_tensor_to_channel_major_maps(record.membrane)))
        output.append(('hidden', 'spike', trace_tensor_to_channel_major_maps(record.spike)))
        for attr_name in ('x_layer', 'i_current', 'z_gate', 'y_mem', 'y_spike'):
            tensor = getattr(record, attr_name, None)
            if isinstance(tensor, torch.Tensor):
                output.append(('hidden', attr_name, trace_tensor_to_channel_major_maps(tensor)))
        return output
    else:
        output: list[tuple[str, str, torch.Tensor]] = []
        explicit_signal_kind = getattr(record, 'signal_kind', None)
        explicit_series = getattr(record, 'series', None)

        if explicit_signal_kind and explicit_series:
            tensor = getattr(record, 'membrane', None)
            if isinstance(tensor, torch.Tensor):
                return [
                    (
                        str(explicit_signal_kind),
                        str(explicit_series),
                        _trace_to_cpu_maps(tensor),
                    )
                ]

        layer_input = getattr(record, 'layer_input', None)
        if isinstance(layer_input, torch.Tensor):
            output.append(('hidden', 'layer_input', _trace_to_cpu_maps(layer_input)))

        membrane = getattr(record, 'membrane', None)
        if isinstance(membrane, torch.Tensor):
            output.append(('hidden', 'membrane', _trace_to_cpu_maps(membrane)))

        spike = getattr(record, 'spike', None)
        if isinstance(spike, torch.Tensor):
            output.append(('hidden', 'spike', _trace_to_cpu_maps(spike)))

        for attr_name in ('x_layer', 'i_current', 'z_gate', 'y_mem', 'y_spike'):
            tensor = getattr(record, attr_name, None)
            if isinstance(tensor, torch.Tensor):
                output.append(('hidden', attr_name, _trace_to_cpu_maps(tensor)))

        return output


def _collect_signal_maps(
    *,
    model: torch.nn.Module,
    dataset: Any,
    split_name: str,
    seed: int,
    anal_batch: int,
    num_workers: int,
    device: torch.device,
) -> dict[tuple[str, int, str, str, str, str, int | None], torch.Tensor]:
    if not LOW_VRAM:
        collected: dict[tuple[str, int, str, str, str, str, int | None], list[torch.Tensor]] = defaultdict(list)
        layer_index_by_name: dict[str, int] = {}
        for idx, (name, _layer) in enumerate(model.iter_named_layers() if hasattr(model, 'iter_named_layers') else [], start=1):
            layer_index_by_name[str(name)] = idx
        with torch.inference_mode():
            for family_id, family, label, subset in _probe_subsets(dataset, split_name=split_name, seed=seed):
                scope = f'{split_name}_{family_id}'
                loader = make_loader(
                    subset,
                    batch_size=int(anal_batch),
                    shuffle=False,
                    num_workers=int(num_workers),
                    pin_memory=device.type == 'cuda',
                    seed=int(seed),
                )
                for inputs, _target in tqdm(loader, desc=f'{SOURCE_PROGRAM}:{scope}', leave=False):
                    model_inputs = _prepared_input_for_model(model, inputs, device=device)
                    result = model(model_inputs, capture_hidden=True)
                    for record in list(result.hidden_records):
                        layer_name = str(record.layer_name)
                        layer_index = int(layer_index_by_name.get(layer_name, len(layer_index_by_name) + 1))
                        for signal_kind, series, maps in _maps_from_record(record):
                            collected[(layer_name, layer_index, signal_kind, series, scope, family, label)].append(maps.detach())
                    output_record = result.output_record
                    output_index = int(layer_index_by_name.get('output', 999))
                    if getattr(output_record, 'layer_input', None) is not None:
                        collected[('output', output_index, 'output', 'layer_input', scope, family, label)].append(trace_tensor_to_channel_major_maps(output_record.layer_input).detach())
                    collected[('output', output_index, 'output', 'membrane', scope, family, label)].append(trace_tensor_to_channel_major_maps(output_record.membrane).detach())
                    collected[('output', output_index, 'output', 'spike', scope, family, label)].append(trace_tensor_to_channel_major_maps(output_record.spike).detach())
                    readout_mem = getattr(output_record, 'readout_mem', None)
                    if isinstance(readout_mem, torch.Tensor):
                        collected[('output', output_index, 'output', 'readout_mem', scope, family, label)].append(trace_tensor_to_channel_major_maps(readout_mem).detach())
        return {key: torch.cat(values, dim=0) for key, values in collected.items() if values}
    else:
        collected: dict[tuple[str, int, str, str, str, str, int | None], list[torch.Tensor]] = defaultdict(list)
        layer_index_by_name: dict[str, int] = {}

        for idx, (name, _layer) in enumerate(model.iter_named_layers() if hasattr(model, 'iter_named_layers') else [], start=1):
            layer_index_by_name[str(name)] = idx

        with torch.inference_mode():
            for family_id, family, label, subset in _probe_subsets(dataset, split_name=split_name, seed=seed):
                scope = f'{split_name}_{family_id}'
                loader = make_loader(
                    subset,
                    batch_size=int(anal_batch),
                    shuffle=False,
                    num_workers=int(num_workers),
                    pin_memory=device.type == 'cuda',
                    seed=int(seed),
                )

                for inputs, _target in tqdm(loader, desc=f'{SOURCE_PROGRAM}:{scope}', leave=False):
                    result = None
                    model_inputs = None

                    try:
                        model_inputs = _prepared_input_for_model(model, inputs, device=device)
                        result = model(model_inputs, capture_hidden=True)

                        for record in result.hidden_records:
                            layer_name = str(record.layer_name)
                            layer_index = int(layer_index_by_name.get(layer_name, len(layer_index_by_name) + 1))

                            for signal_kind, series, maps in _maps_from_record(record):
                                collected[(layer_name, layer_index, signal_kind, series, scope, family, label)].append(maps)

                        output_record = result.output_record
                        output_index = int(layer_index_by_name.get('output', 999))

                        output_layer_input = getattr(output_record, 'layer_input', None)
                        if isinstance(output_layer_input, torch.Tensor):
                            collected[('output', output_index, 'output', 'layer_input', scope, family, label)].append(
                                _trace_to_cpu_maps(output_layer_input)
                            )

                        output_membrane = getattr(output_record, 'membrane', None)
                        if isinstance(output_membrane, torch.Tensor):
                            collected[('output', output_index, 'output', 'membrane', scope, family, label)].append(
                                _trace_to_cpu_maps(output_membrane)
                            )

                        output_spike = getattr(output_record, 'spike', None)
                        if isinstance(output_spike, torch.Tensor):
                            collected[('output', output_index, 'output', 'spike', scope, family, label)].append(
                                _trace_to_cpu_maps(output_spike)
                            )

                        readout_mem = getattr(output_record, 'readout_mem', None)
                        if isinstance(readout_mem, torch.Tensor):
                            collected[('output', output_index, 'output', 'readout_mem', scope, family, label)].append(
                                _trace_to_cpu_maps(readout_mem)
                            )

                    finally:
                        del result
                        del model_inputs
                        del inputs

                        if device.type == 'cuda':
                            torch.cuda.empty_cache()

        return {
            key: torch.cat(values, dim=0).contiguous()
            for key, values in collected.items()
            if values
        }


def _axis_values(summary: Mapping[str, Any], extractor: str) -> tuple[np.ndarray, np.ndarray | None]:
    if str(extractor) != 'psd_exact':
        raise ValueError(f'Unsupported PSD extractor for exact-only analysis: {extractor!r}.')
    axis = curve_axis_from_summary(dict(summary), 'psd_exact')
    return axis, None


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


def _summary_curve_rows(*, common: dict[str, Any], summary: Mapping[str, Any], extractors: tuple[str, ...]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    curve_rows: list[dict[str, str]] = []
    dispersion_rows: list[dict[str, str]] = []
    for reducer, extractor_map in summary.get('representative', {}).items():
        for extractor, scaled_payload in extractor_map.items():
            if extractor not in extractors:
                continue
            axis, edges = _axis_values(summary, extractor)
            for scale, centering_map in _iter_scaled_centering_maps(scaled_payload):
                for centering, values in centering_map.items():
                    for idx, value in enumerate(np.asarray(values, dtype=np.float64).reshape(-1)):
                        row_kwargs = dict(common)
                        row_kwargs.update(
                            category='analysis_curve',
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
                            row_kwargs['bin_left'] = float(edges[idx])
                            row_kwargs['bin_right'] = float(edges[idx + 1])
                        curve_rows.append(common_row(**row_kwargs))
    for extractor, metric_map in summary.get('dispersion', {}).items():
        if extractor not in extractors:
            continue
        axis, edges = _axis_values(summary, extractor)
        for metric, scaled_payload in metric_map.items():
            for scale, centering_map in _iter_scaled_centering_maps(scaled_payload):
                for centering, values in centering_map.items():
                    for idx, value in enumerate(np.asarray(values, dtype=np.float64).reshape(-1)):
                        row_kwargs = dict(common)
                        row_kwargs.update(
                            category='analysis_dispersion',
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
                            row_kwargs['bin_left'] = float(edges[idx])
                            row_kwargs['bin_right'] = float(edges[idx + 1])
                        dispersion_rows.append(common_row(**row_kwargs))
    return curve_rows, dispersion_rows


def _filter_vectors(model: torch.nn.Module) -> dict[str, dict[str, np.ndarray]]:
    collected: dict[str, dict[str, np.ndarray]] = {}
    if not hasattr(model, 'iter_named_layers'):
        return collected
    for layer_name, layer in model.iter_named_layers():
        if not hasattr(layer, 'filter_stats_vectors'):
            continue
        raw = layer.filter_stats_vectors()
        if not isinstance(raw, Mapping):
            continue
        normalized: dict[str, np.ndarray] = {}
        for key, value in raw.items():
            name = str(key)
            if name == 'damping_per_sample':
                name = 'damping'
            if name == 'f_cyc_per_sample':
                name = 'center_frequency'
            if name in {'alpha', 'damping', 'center_frequency'}:
                arr = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
                if arr.size > 0:
                    normalized[name] = np.asarray(arr, dtype=np.float64).reshape(-1)
        if normalized:
            collected[str(layer_name)] = normalized
    return collected


_FILTER_DEFAULT_USERBIN_BOUNDS: dict[str, tuple[float, float]] = {
    'alpha': (0.0, 1.0),
    'center_frequency': (0.0, 0.5),
    'damping': (0.1, 1.0),
}

_FILTER_VALUE_UNITS: dict[str, str] = {
    'alpha': 'parameter_value',
    'damping': 'parameter_value',
    'center_frequency': 'normalized_frequency_cyc_per_sample_nyquist_0p5',
}


def _filter_cli_prefix(parameter: str) -> str:
    if str(parameter) == 'center_frequency':
        return 'frequency'
    return str(parameter)


def _filter_value_unit(parameter: str) -> str:
    return _FILTER_VALUE_UNITS.get(str(parameter), 'parameter_value')


def _strict_filter_edges(raw_edges: Any, *, name: str) -> np.ndarray:
    if isinstance(raw_edges, str):
        text = raw_edges.strip()
        if text.startswith('['):
            raw_edges = json.loads(text)
        else:
            raw_edges = [part for part in re.split(r'[\s,]+', text) if part]
    values = np.asarray([float(value) for value in raw_edges], dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError(f'{name} must contain at least two edge values.')
    if not np.all(np.isfinite(values)):
        raise ValueError(f'{name} must contain finite edge values only.')
    if np.any(np.diff(values) <= 0.0):
        raise ValueError(f'{name} must be strictly increasing.')
    return values


def _filter_userbin_edges(parameter: str, values: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    prefix = _filter_cli_prefix(parameter)
    edge_attr = f'filter_{prefix}_userbin_edges'
    count_attr = f'filter_{prefix}_userbin_count'
    raw_edges = getattr(args, edge_attr, None)
    if raw_edges not in (None, '', []):
        return _strict_filter_edges(raw_edges, name=edge_attr)

    count = int(getattr(args, count_attr, 10) or 10)
    if count < 1:
        raise ValueError(f'{count_attr} must be >= 1.')
    lower, upper = _FILTER_DEFAULT_USERBIN_BOUNDS.get(str(parameter), (float(np.min(values)), float(np.max(values))))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower = float(np.min(values))
        upper = float(np.max(values))
        if upper <= lower:
            delta = max(abs(lower) * 0.01, 1.0e-6)
            lower -= delta
            upper += delta
    return np.linspace(float(lower), float(upper), int(count) + 1, dtype=np.float64)


def _filter_frequency_value(parameter: str, scalar: float) -> float | str:
    if str(parameter) == 'center_frequency':
        return float(scalar)
    return ''


def _filter_distribution_rows_for_values(
    *,
    common_base: dict[str, Any],
    layer_name: str,
    layer_index: int | str,
    distribution_scope: str,
    parameter: str,
    values: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return rows
    unit = _filter_value_unit(parameter)
    for index, scalar in enumerate(flat):
        kwargs = dict(common_base)
        kwargs.update(
            category='filter_distribution',
            layer=str(layer_name),
            layer_index=layer_index,
            distribution_scope=str(distribution_scope),
            parameter_name=str(parameter),
            distribution_kind='exact',
            neuron_index=int(index),
            parameter_value=float(scalar),
            frequency=_filter_frequency_value(parameter, float(scalar)),
            frequency_unit=unit if str(parameter) == 'center_frequency' else '',
            value=float(scalar),
            value_unit=unit,
        )
        rows.append(common_row(**kwargs))

    edges = _filter_userbin_edges(parameter, flat, args)
    counts, bin_edges = np.histogram(flat, bins=edges)
    total = int(np.sum(counts))
    for bin_index, count in enumerate(np.asarray(counts, dtype=np.int64).reshape(-1)):
        left = float(bin_edges[bin_index])
        right = float(bin_edges[bin_index + 1])
        width = right - left
        center = 0.5 * (left + right)
        probability = 0.0 if total <= 0 else float(count) / float(total)
        density = 0.0 if total <= 0 or width <= 0.0 else float(count) / (float(total) * width)
        kwargs = dict(common_base)
        kwargs.update(
            category='filter_distribution',
            layer=str(layer_name),
            layer_index=layer_index,
            distribution_scope=str(distribution_scope),
            parameter_name=str(parameter),
            distribution_kind='userbin',
            bin_index=int(bin_index),
            bin_left=left,
            bin_right=right,
            bin_count=int(count),
            bin_probability=probability,
            bin_density=density,
            parameter_value=center,
            frequency=_filter_frequency_value(parameter, center),
            frequency_unit=unit if str(parameter) == 'center_frequency' else '',
            value=int(count),
            value_unit='count',
        )
        rows.append(common_row(**kwargs))
    return rows


def _filter_model_vectors(filter_vectors: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for param_map in filter_vectors.values():
        for parameter, values in param_map.items():
            arr = np.asarray(values, dtype=np.float64).reshape(-1)
            if arr.size > 0:
                grouped[str(parameter)].append(arr)
    return {parameter: np.concatenate(chunks, axis=0) for parameter, chunks in grouped.items() if chunks}


def _filter_distribution_rows(*, common_base: dict[str, Any], model: torch.nn.Module, layer_index_by_name: Mapping[str, int], args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    vectors = _filter_vectors(model)
    for layer_name, param_map in vectors.items():
        layer_index = layer_index_by_name.get(layer_name, '')
        for parameter, values in param_map.items():
            rows.extend(
                _filter_distribution_rows_for_values(
                    common_base=common_base,
                    layer_name=str(layer_name),
                    layer_index=layer_index,
                    distribution_scope='layer',
                    parameter=str(parameter),
                    values=np.asarray(values, dtype=np.float64),
                    args=args,
                )
            )
    for parameter, values in _filter_model_vectors(vectors).items():
        rows.extend(
            _filter_distribution_rows_for_values(
                common_base=common_base,
                layer_name='model',
                layer_index=0,
                distribution_scope='model',
                parameter=str(parameter),
                values=np.asarray(values, dtype=np.float64),
                args=args,
            )
        )
    return rows


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return {'count': 0.0, 'mean': 0.0, 'std': 0.0, 'min': 0.0, 'q25': 0.0, 'q50': 0.0, 'q75': 0.0, 'max': 0.0}
    return {
        'count': float(flat.size),
        'mean': float(np.mean(flat)),
        'std': float(np.std(flat)),
        'min': float(np.min(flat)),
        'q25': float(np.quantile(flat, 0.25)),
        'q50': float(np.quantile(flat, 0.50)),
        'q75': float(np.quantile(flat, 0.75)),
        'max': float(np.max(flat)),
    }


def _filter_snapshot_rows(*, common_base: dict[str, Any], model: torch.nn.Module, layer_index_by_name: Mapping[str, int]) -> tuple[list[dict[str, str]], dict[str, dict[str, dict[str, float]]]]:
    rows: list[dict[str, str]] = []
    trend_source: dict[str, dict[str, dict[str, float]]] = {}
    vectors = _filter_vectors(model)
    for layer_name, param_map in vectors.items():
        trend_source[layer_name] = {}
        for parameter, values in param_map.items():
            stats = _summary_stats(values)
            trend_source[layer_name][parameter] = dict(stats)
            for stat_name, stat_value in stats.items():
                kwargs = dict(common_base)
                kwargs.update(
                    category='filter_snapshot',
                    layer=layer_name,
                    layer_index=layer_index_by_name.get(layer_name, ''),
                    parameter_name=parameter,
                    statistic=stat_name,
                    value=stat_value,
                    value_unit='count' if stat_name == 'count' else _filter_value_unit(parameter),
                )
                rows.append(common_row(**kwargs))
    model_vectors = _filter_model_vectors(vectors)
    if model_vectors:
        trend_source['model'] = {}
    for parameter, values in model_vectors.items():
        stats = _summary_stats(values)
        trend_source['model'][parameter] = dict(stats)
        for stat_name, stat_value in stats.items():
            kwargs = dict(common_base)
            kwargs.update(
                category='filter_snapshot',
                layer='model',
                layer_index=0,
                parameter_name=parameter,
                statistic=stat_name,
                value=stat_value,
                value_unit='count' if stat_name == 'count' else _filter_value_unit(parameter),
            )
            rows.append(common_row(**kwargs))
    return rows, trend_source


def _manifest_row(*, base: dict[str, Any], artifact_name: str, path: Path, status: str = 'ok', message: str = '') -> dict[str, str]:
    kwargs = dict(base)
    kwargs.update(category='analysis_manifest', status=status, message=message, artifact_name=artifact_name, output_csv_path=str(path))
    return common_row(**kwargs)


def _safe_token(value: Any) -> str:
    text = str(value).strip().lower().replace('-', '_')
    return ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in text).strip('_') or 'value'


def _row_output_name(row: Mapping[str, str]) -> str:
    category = row.get('category', '')
    epoch = _safe_token(row.get('checkpoint_epoch', 'epoch'))
    layer_index = _safe_token(row.get('layer_index', 'layer'))
    scope = _safe_token(row.get('scope', 'scope'))
    signal = _safe_token(row.get('signal_kind', 'signal'))
    series = _safe_token(row.get('series', 'series'))
    extractor = _safe_token(row.get('extractor', 'extractor'))
    reducer = _safe_token(row.get('reducer', 'none'))
    variant = _safe_token(row.get('variant', 'raw'))
    scale = _safe_token(row.get('scale', 'raw'))
    if category == 'analysis_curve':
        return f'analysis_curve__epoch_{epoch}__layer_{layer_index}__{scope}__{signal}__{series}__{extractor}__{reducer}__{variant}__{scale}.csv'
    if category == 'analysis_dispersion':
        statistic = _safe_token(row.get('statistic', 'statistic'))
        return f'analysis_dispersion__epoch_{epoch}__layer_{layer_index}__{scope}__{signal}__{series}__{extractor}__{variant}__{scale}__{statistic}.csv'
    if category == 'pair_distance':
        source = _safe_token(row.get('source_scope', 'source'))
        target = _safe_token(row.get('target_scope', 'target'))
        source_signal = _safe_token(row.get('source_signal_kind', 'source_signal'))
        target_signal = _safe_token(row.get('target_signal_kind', 'target_signal'))
        source_series = _safe_token(row.get('source_series', 'source_series'))
        target_series = _safe_token(row.get('target_series', 'target_series'))
        metric = _safe_token(row.get('distance_metric', 'distance'))
        return f'pair_distance__epoch_{epoch}__layer_{layer_index}__{source}__{source_signal}_{source_series}__to__{target}__{target_signal}_{target_series}__{extractor}__{reducer}__{variant}__{scale}__{metric}.csv'
    if category == 'drift_distance':
        epoch_a = _safe_token(row.get('checkpoint_epoch_a', 'a'))
        epoch_b = _safe_token(row.get('checkpoint_epoch_b', 'b'))
        metric = _safe_token(row.get('distance_metric', 'distance'))
        reference_series = _safe_token(row.get('reference_series', 'reference'))
        return f'drift_distance__epoch_{epoch_a}__to__epoch_{epoch_b}__layer_{layer_index}__{scope}__input_{reference_series}__to__{signal}__{series}__{extractor}__{reducer}__{variant}__{scale}__{metric}.csv'
    if category == 'filter_snapshot':
        parameter = _safe_token(row.get('parameter_name', 'parameter'))
        return f'filter_snapshot__epoch_{epoch}__layer_{layer_index}__{parameter}.csv'
    if category == 'filter_distribution':
        parameter = _safe_token(row.get('parameter_name', 'parameter'))
        scope = _safe_token(row.get('distribution_scope', 'scope'))
        kind = _safe_token(row.get('distribution_kind', 'kind'))
        return f'filter_distribution__epoch_{epoch}__layer_{layer_index}__{parameter}__{scope}__{kind}.csv'
    if category == 'filter_trend':
        parameter = _safe_token(row.get('parameter_name', 'parameter'))
        statistic = _safe_token(row.get('statistic', 'statistic'))
        return f'filter_trend__layer_{layer_index}__{parameter}__{statistic}.csv'
    if category == 'accuracy_loss_join':
        return f'accuracy_loss_join__epoch_{epoch}.csv'
    if category in {'layer_distance_profile', 'layer_distance_trend'}:
        relation = _safe_token(row.get('relation_type', 'relation'))
        track = _safe_token(row.get('track_name', 'track'))
        metric = _safe_token(row.get('distance_metric', 'distance'))
        if category == 'layer_distance_profile':
            return f'layer_distance_profile__epoch_{epoch}__{relation}__{scope}__{track}__{extractor}__{reducer}__{variant}__{scale}__{metric}.csv'
        source_layer = _safe_token(row.get('source_layer', 'source'))
        source_index = _safe_token(row.get('source_layer_index', 'source_index'))
        target_layer = _safe_token(row.get('target_layer', 'target'))
        target_index = _safe_token(row.get('target_layer_index', 'target_index'))
        return f'layer_distance_trend__{relation}__{scope}__{track}__source_{source_index}_{source_layer}__target_{target_index}_{target_layer}__{extractor}__{reducer}__{variant}__{scale}__{metric}.csv'
    if category in {'layer_dispersion_profile', 'layer_dispersion_trend'}:
        statistic = _safe_token(row.get('dispersion_statistic', row.get('statistic', 'statistic')))
        reduction = _safe_token(row.get('dispersion_reduction', 'reduction'))
        if category == 'layer_dispersion_profile':
            return f'layer_dispersion_profile__epoch_{epoch}__{scope}__{signal}__{series}__{extractor}__{variant}__{scale}__{statistic}__{reduction}.csv'
        return f'layer_dispersion_trend__layer_{layer_index}__{scope}__{signal}__{series}__{extractor}__{variant}__{scale}__{statistic}__{reduction}.csv'
    if category == 'pairwise_dependency_appendix':
        source = _safe_token(row.get('source_scope', 'source'))
        target = _safe_token(row.get('target_scope', 'target'))
        source_signal = _safe_token(row.get('source_signal_kind', 'source_signal'))
        target_signal = _safe_token(row.get('target_signal_kind', 'target_signal'))
        source_series = _safe_token(row.get('source_series', 'source_series'))
        target_series = _safe_token(row.get('target_series', 'target_series'))
        metric = _safe_token(row.get('metric', 'metric'))
        return f'pairwise_dependency_appendix__epoch_{epoch}__layer_{layer_index}__{source}__{source_signal}_{source_series}__to__{target}__{target_signal}_{target_series}__{extractor}__{reducer}__{variant}__{scale}__{metric}.csv'
    return f'{_safe_token(category)}.csv'

def _write_artifact(path: Path, rows: list[dict[str, str]], *, manifest_rows: list[dict[str, str]], manifest_base: dict[str, Any], artifact_name: str) -> None:
    if not rows:
        return
    root_dir = path.parent / _safe_token(artifact_name)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out_path = root_dir / _row_output_name(row)
        groups[str(out_path)].append(row)
    for out_path_text, group_rows in sorted(groups.items()):
        out_path = Path(out_path_text)
        write_common_csv(out_path, group_rows)
        manifest_rows.append(_manifest_row(base=manifest_base, artifact_name=artifact_name, path=out_path))



def _write_rows_to_dir(root_dir: Path, rows: list[dict[str, str]], *, manifest_rows: list[dict[str, str]], manifest_base: dict[str, Any], artifact_name: str) -> None:
    if not rows:
        return
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out_path = root_dir / _row_output_name(row)
        groups[str(out_path)].append(row)
    for out_path_text, group_rows in sorted(groups.items()):
        out_path = Path(out_path_text)
        write_common_csv(out_path, group_rows)
        manifest_rows.append(_manifest_row(base=manifest_base, artifact_name=artifact_name, path=out_path))


def _layer_folder(layer: str, layer_index: int | str) -> str:
    try:
        idx = int(layer_index)
    except Exception:
        idx = 999
    return f'layer_{idx:03d}__{_safe_token(layer)}'


def _write_layer_rows(checkpoint_dir: Path, rows: list[dict[str, str]], *, manifest_rows: list[dict[str, str]], manifest_base: dict[str, Any], artifact_name: str) -> None:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get('layer', ''), row.get('layer_index', ''))].append(row)
    for (layer, layer_index), group_rows in sorted(grouped.items(), key=lambda item: (str(item[0][1]), str(item[0][0]))):
        root_dir = checkpoint_dir / 'layers' / _layer_folder(layer, layer_index) / _safe_token(artifact_name)
        _write_rows_to_dir(root_dir, group_rows, manifest_rows=manifest_rows, manifest_base=manifest_base, artifact_name=artifact_name)


def _pair_rows_for_checkpoint(
    *,
    summaries: Mapping[tuple[str, int, str, str, str, str, int | None], Mapping[str, Any]],
    common_by_key: Mapping[tuple[str, int, str, str, str, str, int | None], dict[str, Any]],
    enable_appendix: bool,
    extractors: tuple[str, ...],
    distance_metric: str = 'centered_l2',
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    pair_rows: list[dict[str, str]] = []
    appendix_rows: list[dict[str, str]] = []
    grouped: dict[tuple[str, int, str, str], list[tuple[tuple[str, int, str, str, str, str, int | None], Mapping[str, Any]]]] = defaultdict(list)
    for key, summary in summaries.items():
        layer, layer_index, signal_kind, series, _scope, _family, _label = key
        grouped[(layer, layer_index, signal_kind, series)].append((key, summary))
    for (_layer, _layer_index, _signal_kind, _series), items in grouped.items():
        if len(items) < 2:
            continue
        for (left_key, left_summary), (right_key, right_summary) in itertools.combinations(sorted(items, key=lambda item: item[0][4]), 2):
            base = dict(common_by_key[left_key])
            left_scope = left_key[4]
            right_scope = right_key[4]
            left_series = left_key[3]
            right_series = right_key[3]
            right_common = common_by_key[right_key]
            for reducer in ('mean', 'median'):
                for extractor in extractors:
                    for scale in ALL_VALUE_SCALES:
                        try:
                            distances = pair_distance_from_summaries(dict(left_summary), dict(right_summary), reducer=reducer, extractor=extractor, scale=scale, distance_metric=distance_metric)
                        except Exception as exc:
                            failure = dict(base)
                            failure.update(
                                category='pair_distance', status='failed', message=str(exc), extractor=extractor, reducer=reducer, variant='raw', scale=scale,
                                source_scope=left_scope, target_scope=right_scope,
                                source_signal_kind=base.get('signal_kind', ''), source_series=left_series,
                                target_signal_kind=right_common.get('signal_kind', ''), target_series=right_series,
                                distance_metric='failed',
                            )
                            pair_rows.append(common_row(**failure))
                            continue
                        for metric, value in distances.items():
                            if metric == 'reference_curve_axis':
                                continue
                            kwargs = dict(base)
                            kwargs.update(
                                category='pair_distance', extractor=extractor, reducer=reducer, variant='centered' if metric.endswith('_cen') else 'raw', scale=scale,
                                source_scope=left_scope, target_scope=right_scope,
                                source_signal_kind=base.get('signal_kind', ''), source_series=left_series,
                                target_signal_kind=right_common.get('signal_kind', ''), target_series=right_series,
                                distance_metric=metric, value=float(value), value_unit='dimensionless',
                            )
                            pair_rows.append(common_row(**kwargs))
                        if enable_appendix:
                            try:
                                left_curve = representative_curve_from_summary(dict(left_summary), reducer=reducer, extractor=extractor, centering='raw', scale=scale).reshape(-1)
                                right_curve = representative_curve_from_summary(dict(right_summary), reducer=reducer, extractor=extractor, centering='raw', scale=scale).reshape(-1)
                                if left_curve.size == right_curve.size and left_curve.size > 1 and np.std(left_curve) > 0 and np.std(right_curve) > 0:
                                    corr = float(np.corrcoef(left_curve, right_curve)[0, 1])
                                else:
                                    corr = 0.0
                                appendix = dict(base)
                                appendix.update(
                                    category='pairwise_dependency_appendix', extractor=extractor, reducer=reducer, variant='raw', scale=scale,
                                    source_scope=left_scope, target_scope=right_scope,
                                    source_signal_kind=base.get('signal_kind', ''), source_series=left_series,
                                    target_signal_kind=right_common.get('signal_kind', ''), target_series=right_series,
                                    metric='representative_curve_correlation', value=corr, value_unit='pearson_r',
                                )
                                appendix_rows.append(common_row(**appendix))
                            except Exception as exc:
                                failure = dict(base)
                                failure.update(
                                    category='pairwise_dependency_appendix', status='failed', message=str(exc), extractor=extractor, reducer=reducer, variant='raw', scale=scale,
                                    source_scope=left_scope, target_scope=right_scope,
                                    source_signal_kind=base.get('signal_kind', ''), source_series=left_series,
                                    target_signal_kind=right_common.get('signal_kind', ''), target_series=right_series,
                                )
                                appendix_rows.append(common_row(**failure))
    return pair_rows, appendix_rows

def _target_track_name(signal_kind: str, series: str) -> str | None:
    signal = str(signal_kind)
    name = str(series)
    if signal == 'feature' or name in {'block_output', 'feature'}:
        return 'feature'
    if name in {'spike', 'y_spike'}:
        return 'spike'
    if name in {'membrane', 'y_mem', 'readout_mem'}:
        return 'membrane'
    if name in {'layer_input', 'x_layer'}:
        return 'layer_input'
    return None


def _distance_rows_for_pair(
    *,
    category: str,
    source_key: tuple[str, int, str, str, str, str, int | None],
    source_summary: Mapping[str, Any],
    target_key: tuple[str, int, str, str, str, str, int | None],
    target_summary: Mapping[str, Any],
    common_by_key: Mapping[tuple[str, int, str, str, str, str, int | None], dict[str, Any]],
    relation_type: str,
    comparison_index: int,
    track_name: str,
    extractors: tuple[str, ...],
    distance_metric: str = 'centered_l2',
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    source_common = common_by_key[source_key]
    target_common = common_by_key[target_key]
    source_layer, source_layer_index, source_signal, source_series, *_ = source_key
    target_layer, target_layer_index, target_signal, target_series, *_ = target_key
    comparison_label = f'{source_layer}->{target_layer}'
    for reducer in ('mean', 'median'):
        for extractor in extractors:
            for scale in ALL_VALUE_SCALES:
                try:
                    distances = pair_distance_from_summaries(dict(source_summary), dict(target_summary), reducer=reducer, extractor=extractor, scale=scale, distance_metric=distance_metric)
                except Exception as exc:
                    base = dict(target_common)
                    base.update(
                        category=category,
                        status='failed',
                        message=str(exc),
                        source_layer=source_layer,
                        source_layer_index=source_layer_index,
                        source_signal_kind=source_signal,
                        source_series=source_series,
                        target_layer=target_layer,
                        target_layer_index=target_layer_index,
                        target_signal_kind=target_signal,
                        target_series=target_series,
                        relation_type=relation_type,
                        comparison_index=comparison_index,
                        comparison_label=comparison_label,
                        track_name=track_name,
                        extractor=extractor,
                        reducer=reducer,
                        variant='raw',
                        scale=scale,
                        distance_metric='failed',
                    )
                    rows.append(common_row(**base))
                    continue
                for metric, value in distances.items():
                    if metric == 'reference_curve_axis':
                        continue
                    base = dict(target_common)
                    base.update(
                        category=category,
                        source_layer=source_layer,
                        source_layer_index=source_layer_index,
                        source_signal_kind=source_signal,
                        source_series=source_series,
                        target_layer=target_layer,
                        target_layer_index=target_layer_index,
                        target_signal_kind=target_signal,
                        target_series=target_series,
                        relation_type=relation_type,
                        comparison_index=comparison_index,
                        comparison_label=comparison_label,
                        track_name=track_name,
                        extractor=extractor,
                        reducer=reducer,
                        variant='centered' if str(metric).endswith('_cen') else 'raw',
                        scale=scale,
                        distance_metric=metric,
                        value=float(value),
                        value_unit='dimensionless',
                    )
                    rows.append(common_row(**base))
    return rows


def _layer_distance_rows_for_checkpoint(
    *,
    summaries: Mapping[tuple[str, int, str, str, str, str, int | None], Mapping[str, Any]],
    common_by_key: Mapping[tuple[str, int, str, str, str, str, int | None], dict[str, Any]],
    extractors: tuple[str, ...],
    distance_metric: str = 'centered_l2',
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    profile_rows: list[dict[str, str]] = []
    trend_rows: list[dict[str, str]] = []
    targets: dict[tuple[str, str, int | None, str], list[tuple[tuple[str, int, str, str, str, str, int | None], Mapping[str, Any]]]] = defaultdict(list)
    for key, summary in summaries.items():
        layer, layer_index, signal_kind, series, scope, family, label = key
        track = _target_track_name(str(signal_kind), str(series))
        if track is not None:
            targets[(scope, family, label, track)].append((key, summary))
    for (_scope, _family, _label, track), nodes in sorted(targets.items(), key=lambda item: (item[0][0], item[0][3])):
        ordered = sorted(nodes, key=lambda item: (int(item[0][1]), str(item[0][0]), str(item[0][2]), str(item[0][3])))
        adjacency_chain = ordered
        for index, ((source_key, source_summary), (target_key, target_summary)) in enumerate(zip(adjacency_chain, adjacency_chain[1:]), start=1):
            rows = _distance_rows_for_pair(
                category='layer_distance_profile',
                source_key=source_key,
                source_summary=source_summary,
                target_key=target_key,
                target_summary=target_summary,
                common_by_key=common_by_key,
                relation_type='adjacent',
                comparison_index=index,
                track_name=track,
                extractors=extractors,
                distance_metric=distance_metric,
            )
            profile_rows.extend(rows)
            for row in rows:
                trend = dict(row)
                trend['category'] = 'layer_distance_trend'
                trend_rows.append(common_row(**trend))
    return profile_rows, trend_rows


def _layer_dispersion_rows_for_checkpoint(
    *,
    summaries: Mapping[tuple[str, int, str, str, str, str, int | None], Mapping[str, Any]],
    common_by_key: Mapping[tuple[str, int, str, str, str, str, int | None], dict[str, Any]],
    extractors: tuple[str, ...],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    profile_rows: list[dict[str, str]] = []
    trend_rows: list[dict[str, str]] = []
    for key, summary in sorted(summaries.items(), key=lambda item: (item[0][4], item[0][1], item[0][0], item[0][2], item[0][3])):
        base_common = common_by_key[key]
        for extractor in extractors:
            for metric, scaled_payload in summary.get('dispersion', {}).get(extractor, {}).items():
                if str(metric) not in {'variance', 'mad'}:
                    continue
                for scale, centering_map in _iter_scaled_centering_maps(scaled_payload):
                    for centering, values in centering_map.items():
                        arr = np.asarray(values, dtype=np.float64).reshape(-1)
                        value = float(np.mean(arr)) if arr.size else 0.0
                        row_base = dict(base_common)
                        row_base.update(
                            category='layer_dispersion_profile',
                            extractor=extractor,
                            variant='centered' if str(centering) == 'cen' else 'raw',
                            scale=scale,
                            dispersion_statistic=str(metric),
                            dispersion_reduction='mean_over_frequency',
                            value=value,
                            value_unit=_value_unit_for_dispersion(str(metric), scale),
                        )
                        row = common_row(**row_base)
                        profile_rows.append(row)
                        trend = dict(row)
                        trend['category'] = 'layer_dispersion_trend'
                        trend_rows.append(common_row(**trend))
    return profile_rows, trend_rows


def _checkpoint_common_base(*, payload: Mapping[str, Any], checkpoint_path: Path, model_spec: ModelSpec, readout_mode: str, run_id: str, prep_profile: str, seed: int) -> dict[str, Any]:
    epoch = int(payload.get('epoch', -1))
    return {
        'source_program': SOURCE_PROGRAM,
        'run_id': run_id,
        'dataset': str(payload.get('dataset_token', '')),
        'prep_profile': prep_profile,
        'seed': int(seed),
        'model_token': model_spec.canonical_token,
        'model_family': _model_family(model_spec),
        'readout_mode': readout_mode,
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_epoch': epoch,
    }


def _validate_pca_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[int]:
    if args.pca_ref_epoch is not None and int(args.pca_ref_epoch) < 1:
        parser.error('--pca_ref_epoch must be a positive integer.')
    gate = float(args.pca_min_train_accuracy)
    if gate < 0.0 or gate > 1.0:
        parser.error('--pca_min_train_accuracy must be in [0.0, 1.0].')
    values = [] if args.pca_dim_per_layer is None else list(args.pca_dim_per_layer)
    dims: list[int] = []
    for value in values:
        parsed = int(value)
        if parsed < 1:
            raise ValueError('pca_dim_per_layer must contain positive integers only.')
        dims.append(parsed)
    return dims


def _resolve_train_accuracy(payload: Mapping[str, Any]) -> float | None:
    metric_snapshot = payload.get('metric_snapshot')
    if isinstance(metric_snapshot, Mapping):
        value = metric_snapshot.get('train_accuracy')
        if value is not None:
            return float(value)
    value = payload.get('train_accuracy')
    if value is not None:
        return float(value)
    metrics = payload.get('training_metrics')
    if isinstance(metrics, Mapping):
        value = metrics.get('train_accuracy')
        if value is not None:
            return float(value)
    return None


def _parse_bool_like(value: Any, *, default: bool) -> bool:
    return _parse_bool_config_value(value, default=default)


def _split_cli_values(values: Any, *, default: Sequence[str]) -> list[str]:
    if values is None:
        return [str(v) for v in default]
    if isinstance(values, str):
        raw = [values]
    else:
        raw = list(values)
    out: list[str] = []
    for item in raw:
        for chunk in str(item).replace(',', ' ').split():
            token = chunk.strip()
            if token:
                out.append(token)
    return out or [str(v) for v in default]


def _normalize_analysis_distance_metrics(values: Any) -> tuple[str, ...]:
    allowed = {'centered_l2', 'diff_l2'}
    out: list[str] = []
    seen: set[str] = set()
    for value in _split_cli_values(values, default=('centered_l2',)):
        token = str(value).strip().lower()
        if token not in allowed:
            raise ValueError(f'Unsupported analysis_distance_metric {value!r}. Allowed: {tuple(sorted(allowed))}.')
        if token not in seen:
            out.append(token)
            seen.add(token)
    return tuple(out)


def _safe_basis_filename(basis_id: str) -> str:
    base = re.sub(r'[^a-zA-Z0-9._-]+', '_', str(basis_id)).strip('._')
    if not base:
        base = 'basis'
    return f'{base}.pt'


def main(argv: Sequence[str] | None = None) -> int:
    global LOW_VRAM
    parser = build_arg_parser()
    args = parse_args_with_config(parser, argv=argv, stage_key='psd_analysis')
    LOW_VRAM = bool(int(args.low_vram))
    if int(args.anal_batch) < 1:
        parser.error('--anal_batch must be >= 1.')
    if int(args.num_workers) < 0:
        parser.error('--num_workers must be >= 0.')
    distance_metrics = _normalize_analysis_distance_metrics(args.analysis_distance_metric)
    pca_dims_cli = _validate_pca_args(args, parser)
    enable_pca_1d = _parse_bool_like(args.enable_pca_1d, default=False)
    enable_pca_mimo = _parse_bool_like(args.enable_pca_mimo, default=False)
    pca_enabled = bool(enable_pca_1d or enable_pca_mimo or args.pca_ref_epoch is not None)
    if pca_enabled and args.pca_ref_epoch is None:
        raise ValueError('pca_ref_epoch must be provided when enable_pca_1d or enable_pca_mimo is enabled.')
    # Base path keeps exact PSD summaries; runtime token overlay expands psd_curve_tokens/userbins.

    _load_runtime_dependencies()

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_input = Path(args.checkpoint).expanduser().resolve()
    input_is_single_file = checkpoint_input.is_file()
    selected_extractors = ALL_CURVE_EXTRACTORS
    checkpoint_files, ordering_warnings = _resolve_checkpoint_files(checkpoint_input)
    pca_ref_checkpoint: Path | None = None
    if pca_enabled:
        ref_candidates = [p for p in checkpoint_files if int(_load_checkpoint(p, map_location='cpu').get('epoch', -1)) == int(args.pca_ref_epoch)]
        if not ref_candidates:
            raise ValueError(f'pca_ref_epoch={int(args.pca_ref_epoch)} is not present in checkpoint list.')
        pca_ref_checkpoint = ref_candidates[0]
    device = _require_cuda_device(int(args.gpu_index))

    manifest_rows: list[dict[str, str]] = []
    layer_distance_trend_rows: list[dict[str, str]] = []
    layer_dispersion_trend_rows: list[dict[str, str]] = []
    filter_trend_rows: list[dict[str, str]] = []
    filter_trend_history: dict[tuple[str, int, str, str], list[tuple[int, float, dict[str, Any]]]] = defaultdict(list)
    first_manifest_base: dict[str, Any] | None = None
    pca_basis_cache: dict[tuple[str, int, str, str, str, int | None], tuple[torch.Tensor, torch.Tensor, dict[str, Any]]] = {}

    pca_reference_dir = output_root / 'pca_reference' / 'basis'
    for checkpoint_path in tqdm(checkpoint_files, desc='psd_analysis:checkpoints', leave=False):
        payload = _load_checkpoint(checkpoint_path, map_location='cpu')
        seed = int(args.seed if args.seed is not None else payload.get('seed', 0))
        _seed_everything(seed)
        model, _readout, model_spec, readout_mode = _build_model_from_checkpoint(payload, device=device)
        bundle = _resolve_bundle(payload, cli_dataset=args.dataset, cli_prep_root=args.prep_root, model_spec=model_spec)
        manifest = _manifest_dict(bundle.manifest_path)
        _validate_axis_metadata(manifest, payload)
        prep_profile = str(manifest.get('prep_profile', manifest.get('psd_axis_kind', bundle.psd_axis_kind)))
        run_id = f'{bundle.dataset_name}_{model_spec.canonical_token}_{readout_mode}_analysis_seed{seed}'
        checkpoint_base = _checkpoint_common_base(payload=payload, checkpoint_path=checkpoint_path, model_spec=model_spec, readout_mode=readout_mode, run_id=run_id, prep_profile=prep_profile, seed=seed)
        checkpoint_base.update(_axis_metadata_columns(manifest, psd_axis_kind=bundle.psd_axis_kind))
        if first_manifest_base is None:
            first_manifest_base = dict(checkpoint_base)

        layer_index_by_name: dict[str, int] = {}
        if hasattr(model, 'iter_named_layers'):
            for idx, (name, _layer) in enumerate(model.iter_named_layers(), start=1):
                layer_index_by_name[str(name)] = idx
        maps_by_key: dict[tuple[str, int, str, str, str, str, int | None], torch.Tensor] = {}
        for split_name, split_dataset in (('train', bundle.train_dataset), ('test', bundle.test_dataset)):
            analysis_dataset = dataset_for_view(split_dataset, bundle.training_view_name)
            maps_by_key.update(
                _collect_signal_maps(
                    model=model,
                    dataset=analysis_dataset,
                    split_name=split_name,
                    seed=seed,
                    anal_batch=int(args.anal_batch),
                    num_workers=int(args.num_workers),
                    device=device,
                )
            )
        if pca_enabled and checkpoint_path == pca_ref_checkpoint:
            train_accuracy = _resolve_train_accuracy(payload)
            if float(args.pca_min_train_accuracy) > 0.0 and train_accuracy is None:
                raise ValueError('pca_min_train_accuracy > 0 requires reference checkpoint train accuracy metadata.')
            if train_accuracy is not None and train_accuracy < float(args.pca_min_train_accuracy):
                raise ValueError(f'pca_ref_epoch train accuracy {train_accuracy:.6f} is below pca_min_train_accuracy={float(args.pca_min_train_accuracy):.6f}')
            for key, maps in maps_by_key.items():
                layer_name, layer_index, signal_kind, _series, scope, family, label = key
                row_count = int(maps.shape[1])
                dim = pca_dim_from_cli_vector(pca_dims_cli, int(layer_index) - 1, row_count)
                basis, centroid = compute_fixed_pca_basis(maps, dim)
                meta = {
                    'pca_analysis_schema_version': PCA_ANALYSIS_SCHEMA_VERSION,
                    'dataset': str(args.dataset),
                    'run_id': str(run_id),
                    'checkpoint_path': str(checkpoint_path),
                    'checkpoint_epoch': int(payload.get('epoch', -1)),
                    'reference_checkpoint_path': str(checkpoint_path),
                    'split': str(scope),
                    'scope': str(scope),
                    'layer': str(layer_name),
                    'layer_name': str(layer_name),
                    'layer_index': int(layer_index),
                    'family': str(family),
                    'label': '' if label is None else int(label),
                    'signal_kind': str(signal_kind),
                    'row_count': row_count,
                    'requested_dim': int((pca_dims_cli[int(layer_index) - 1] if pca_dims_cli and int(layer_index) - 1 < len(pca_dims_cli) else (pca_dims_cli[-1] if pca_dims_cli else min(row_count, 4)))),
                    'resolved_dim': int(dim),
                    'basis_shape': list(basis.shape),
                    'centroid_shape': list(centroid.shape),
                    'basis_id': f'dataset={args.dataset}|run={run_id}|ref_epoch={int(payload.get("epoch",-1))}|split={scope}|scope={scope}|layer={layer_name}|family={family}|kind={signal_kind}|dim={int(dim)}',
                    'reference_epoch': int(payload.get('epoch', -1)),
                    'reference_train_accuracy': None if train_accuracy is None else float(train_accuracy),
                    'pca_min_train_accuracy': float(args.pca_min_train_accuracy),
                    'source_dtype': str(maps.dtype),
                    'source_device_after_fit': 'cpu',
                }
                pca_reference_dir.mkdir(parents=True, exist_ok=True)
                basis_file = _safe_basis_filename(str(meta['basis_id']))
                basis_path = pca_reference_dir / basis_file
                basis_payload = {
                    'basis': basis.detach().cpu().requires_grad_(False),
                    'centroid': centroid.detach().cpu().requires_grad_(False),
                    'basis_id': str(meta['basis_id']),
                    'reference_epoch': int(meta['reference_epoch']),
                    'requested_dim': int(meta['requested_dim']),
                    'resolved_dim': int(meta['resolved_dim']),
                    'row_count': int(meta['row_count']),
                    'layer_index': int(meta['layer_index']),
                    'layer_name': str(meta['layer_name']),
                    'family': str(meta['family']),
                    'signal_kind': str(meta['signal_kind']),
                }
                torch.save(basis_payload, basis_path)
                meta['basis_file'] = str(basis_path.relative_to(output_root))
                pca_basis_cache[key] = (basis.cpu(), centroid.cpu(), meta)
                if LOW_VRAM:
                    del basis, centroid

        if input_is_single_file:
            checkpoint_dir = output_root / checkpoint_path.stem
        else:
            checkpoint_dir = output_root / f'checkpoint_epoch_{int(payload.get("epoch", len(manifest_rows))):06d}'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        summaries: dict[tuple[str, int, str, str, str, str, int | None], Mapping[str, Any]] = {}
        common_by_key: dict[tuple[str, int, str, str, str, str, int | None], dict[str, Any]] = {}
        family_rows: list[dict[str, str]] = []
        dispersion_rows: list[dict[str, str]] = []
        pca_reference_rows: list[dict[str, str]] = []
        pca_mode_rows: list[dict[str, str]] = []
        pca_mimo_rows: list[dict[str, str]] = []
        pca_cross_rows: list[dict[str, str]] = []

        for key, maps in tqdm(sorted(maps_by_key.items(), key=lambda item: (item[0][1], item[0][0], item[0][2], item[0][3], item[0][4])), desc='psd_analysis:summaries', leave=False):
            layer_name, layer_index, signal_kind, series, scope, family, label = key
            summary = compute_family_spectral_summary(maps, window=None, overlap=0, userbin_edges=None, include_spectrogram=False, include_userbin=False)
            summaries[key] = summary
            base = dict(checkpoint_base)
            base.update(layer=layer_name, layer_index=layer_index, scope=scope, probe_family=family, label='' if label is None else int(label), signal_kind=signal_kind, series=series)
            common_by_key[key] = base
            curve_rows, disp_rows = _summary_curve_rows(common=base, summary=summary, extractors=selected_extractors)
            family_rows.extend(curve_rows)
            dispersion_rows.extend(disp_rows)
            cache = pca_basis_cache.get(key)
            if pca_enabled and cache is not None:
                basis, centroid, meta = cache
                pca_maps = apply_fixed_pca_basis(maps, basis, centroid)
                for mode in range(int(pca_maps.shape[1])):
                    mode_summary = compute_family_spectral_summary(
                        pca_maps[:, mode : mode + 1, :], window=None, overlap=0, userbin_edges=None, include_spectrogram=False, include_userbin=False
                    )
                    if enable_pca_1d:
                        mode_curve = representative_curve_from_summary(dict(mode_summary), reducer='mean', extractor='psd_exact', centering='raw', scale='raw').reshape(-1)
                        mode_freq = curve_axis_from_summary(dict(mode_summary), extractor='psd_exact', centering='raw')
                        for fi, (freq_value, curve_value) in enumerate(zip(mode_freq, mode_curve)):
                            kwargs = dict(base)
                            kwargs.update(category='analysis_curve', extractor='psd_exact', reducer='mean', variant='raw', scale='raw', frequency=float(freq_value), frequency_bin=int(fi), value=float(curve_value), value_unit='power', series=f'pca_mode_{mode:03d}', pca_analysis_schema_version=PCA_ANALYSIS_SCHEMA_VERSION, basis_id=str(meta.get('basis_id', '')), pca_mode=int(mode), reference_epoch=int(meta.get('reference_epoch', -1)), resolved_dim=int(meta.get('resolved_dim', pca_maps.shape[1])))
                            pca_mode_rows.append(common_row(**kwargs))
                if enable_pca_mimo:
                    freqs, mimo = auto_spectral_matrix_from_mode_maps(pca_maps)
                    for fi, freq_value in enumerate(freqs.detach().cpu().numpy().reshape(-1)):
                        for mi in range(int(mimo.shape[1])):
                            for mj in range(int(mimo.shape[2])):
                                kwargs = dict(base)
                                entry = mimo[fi, mi, mj]
                                kwargs.update(category='analysis_curve', extractor='psd_exact', reducer='mean', variant='raw', scale='raw', frequency=float(freq_value), frequency_bin=int(fi), value=float(entry.abs().detach().cpu().item()), value_real=float(entry.real.detach().cpu().item()), value_imag=float(entry.imag.detach().cpu().item()), value_unit='power', series=f'pca_mimo_{mi:03d}_{mj:03d}', pca_analysis_schema_version=PCA_ANALYSIS_SCHEMA_VERSION, basis_id=str(meta.get('basis_id', '')), source_mode=int(mi), target_mode=int(mj), reference_epoch=int(meta.get('reference_epoch', -1)), resolved_dim=int(meta.get('resolved_dim', pca_maps.shape[1])))
                                pca_mimo_rows.append(common_row(**kwargs))
                ref_kwargs = dict(base)
                ref_kwargs.update(category='analysis_manifest', status='ok', message=json.dumps(meta, ensure_ascii=False), artifact_name='pca_reference', output_csv_path=str(checkpoint_dir / 'pca_reference'), pca_analysis_schema_version=PCA_ANALYSIS_SCHEMA_VERSION, basis_id=str(meta.get('basis_id', '')), reference_epoch=int(meta.get('reference_epoch', -1)), resolved_dim=int(meta.get('resolved_dim', 0)))
                pca_reference_rows.append(common_row(**ref_kwargs))

        layer_distance_profile_rows: list[dict[str, str]] = []
        layer_distance_checkpoint_trend_rows: list[dict[str, str]] = []
        for distance_metric in distance_metrics:
            profile_rows, trend_rows = _layer_distance_rows_for_checkpoint(
                summaries=summaries,
                common_by_key=common_by_key,
                extractors=selected_extractors,
                distance_metric=str(distance_metric),
            )
            layer_distance_profile_rows.extend(profile_rows)
            layer_distance_checkpoint_trend_rows.extend(trend_rows)
        layer_dispersion_profile_rows, layer_dispersion_checkpoint_trend_rows = _layer_dispersion_rows_for_checkpoint(summaries=summaries, common_by_key=common_by_key, extractors=selected_extractors)
        layer_distance_trend_rows.extend(layer_distance_checkpoint_trend_rows)
        layer_dispersion_trend_rows.extend(layer_dispersion_checkpoint_trend_rows)

        pair_rows: list[dict[str, str]] = []
        appendix_rows: list[dict[str, str]] = []
        for distance_metric in distance_metrics:
            current_pair_rows, current_appendix_rows = _pair_rows_for_checkpoint(
                summaries=summaries,
                common_by_key=common_by_key,
                enable_appendix=bool(args.enable_pairwise_dependency_appendix),
                extractors=selected_extractors,
                distance_metric=str(distance_metric),
            )
            pair_rows.extend(current_pair_rows)
            appendix_rows.extend(current_appendix_rows)
        filter_rows, filter_snapshot = _filter_snapshot_rows(common_base=checkpoint_base, model=model, layer_index_by_name=layer_index_by_name)
        filter_distribution_rows = _filter_distribution_rows(common_base=checkpoint_base, model=model, layer_index_by_name=layer_index_by_name, args=args)
        for layer_name, param_map in filter_snapshot.items():
            layer_index = 0 if str(layer_name) == 'model' else int(layer_index_by_name.get(layer_name, 999))
            for parameter, stats in param_map.items():
                for stat_name, stat_value in stats.items():
                    filter_trend_history[(layer_name, layer_index, parameter, stat_name)].append((int(payload.get('epoch', 0)), float(stat_value), dict(checkpoint_base)))
        if pca_enabled and enable_pca_mimo:
            sorted_keys = sorted(maps_by_key.keys(), key=lambda item: (item[4], item[1], item[0], item[5], str(item[6])))
            for key in sorted_keys:
                layer_name, layer_index, _signal_kind, _series, scope, family, label = key
                li_key = (layer_name, layer_index, 'hidden', 'layer_input', scope, family, label)
                mem_key = (layer_name, layer_index, 'hidden', 'membrane', scope, family, label)
                spk_key = (layer_name, layer_index, 'hidden', 'spike', scope, family, label)
                for src_key, dst_key, trace_name in ((li_key, mem_key, 'layer_input_to_membrane'), (li_key, spk_key, 'layer_input_to_spike')):
                    if src_key in maps_by_key and dst_key in maps_by_key and src_key in pca_basis_cache and dst_key in pca_basis_cache:
                        x_basis, x_centroid, x_meta = pca_basis_cache[src_key]
                        y_basis, y_centroid, y_meta = pca_basis_cache[dst_key]
                        src_modes = apply_fixed_pca_basis(maps_by_key[src_key], x_basis, x_centroid)
                        dst_modes = apply_fixed_pca_basis(maps_by_key[dst_key], y_basis, y_centroid)
                        freqs, cross = cross_spectral_matrix_from_mode_maps(src_modes, dst_modes)
                        base = dict(common_by_key[src_key])
                        for fi, freq_value in enumerate(freqs.detach().cpu().numpy().reshape(-1)):
                            for mi in range(int(cross.shape[1])):
                                for mj in range(int(cross.shape[2])):
                                    kwargs = dict(base)
                                    entry = cross[fi, mi, mj]
                                    kwargs.update(category='analysis_curve', extractor='psd_exact', reducer='mean', variant='raw', scale='raw', frequency=float(freq_value), frequency_bin=int(fi), value=float(entry.abs().detach().cpu().item()), value_real=float(entry.real.detach().cpu().item()), value_imag=float(entry.imag.detach().cpu().item()), value_unit='power', pca_analysis_schema_version=PCA_ANALYSIS_SCHEMA_VERSION, series=f'pca_cross_{trace_name}_{mi:03d}_{mj:03d}', relation=str(trace_name), source_layer_index=int(src_key[1]), source_layer_name=str(src_key[0]), source_signal_kind=str(src_key[3]), target_layer_index=int(dst_key[1]), target_layer_name=str(dst_key[0]), target_signal_kind=str(dst_key[3]), source_mode=int(mi), target_mode=int(mj), x_basis_id=str(x_meta.get('basis_id', '')), y_basis_id=str(y_meta.get('basis_id', '')), x_resolved_dim=int(x_meta.get('resolved_dim', src_modes.shape[1])), y_resolved_dim=int(y_meta.get('resolved_dim', dst_modes.shape[1])), reference_epoch=int(x_meta.get('reference_epoch', -1)))
                                    pca_cross_rows.append(common_row(**kwargs))

            # adjacent hidden output-output relation
            grouped_keys: dict[tuple[str, str, int | None], list[tuple[str, int, str, str, str, str, int | None]]] = defaultdict(list)
            for key in sorted_keys:
                layer_name, layer_index, _signal_kind, series, scope, family, label = key
                if series in {'spike', 'membrane'}:
                    grouped_keys[(scope, family, label)].append(key)
            for (_scope, _family, _label), keys in grouped_keys.items():
                by_index: dict[int, tuple[str, int, str, str, str, str, int | None]] = {}
                for key in keys:
                    idx = int(key[1])
                    # prefer spike over membrane deterministically
                    if idx not in by_index or (by_index[idx][3] != 'spike' and key[3] == 'spike'):
                        by_index[idx] = key
                ordered = [by_index[i] for i in sorted(by_index.keys())]
                for src_key, dst_key in zip(ordered[:-1], ordered[1:]):
                    if int(dst_key[1]) != int(src_key[1]) + 1:
                        continue
                    if src_key not in maps_by_key or dst_key not in maps_by_key:
                        continue
                    if src_key not in pca_basis_cache or dst_key not in pca_basis_cache:
                        continue
                    x_basis, x_centroid, x_meta = pca_basis_cache[src_key]
                    y_basis, y_centroid, y_meta = pca_basis_cache[dst_key]
                    src_modes = apply_fixed_pca_basis(maps_by_key[src_key], x_basis, x_centroid)
                    dst_modes = apply_fixed_pca_basis(maps_by_key[dst_key], y_basis, y_centroid)
                    freqs, cross = cross_spectral_matrix_from_mode_maps(src_modes, dst_modes)
                    base = dict(common_by_key[src_key])
                    for fi, freq_value in enumerate(freqs.detach().cpu().numpy().reshape(-1)):
                        for mi in range(int(cross.shape[1])):
                            for mj in range(int(cross.shape[2])):
                                kwargs = dict(base)
                                entry = cross[fi, mi, mj]
                                kwargs.update(category='analysis_curve', extractor='psd_exact', reducer='mean', variant='raw', scale='raw', frequency=float(freq_value), frequency_bin=int(fi), value=float(entry.abs().detach().cpu().item()), value_real=float(entry.real.detach().cpu().item()), value_imag=float(entry.imag.detach().cpu().item()), value_unit='power', pca_analysis_schema_version=PCA_ANALYSIS_SCHEMA_VERSION, series=f'pca_cross_adjacent_hidden_output_{mi:03d}_{mj:03d}', relation='adjacent_hidden_output', source_layer_index=int(src_key[1]), source_layer_name=str(src_key[0]), source_signal_kind=str(src_key[3]), target_layer_index=int(dst_key[1]), target_layer_name=str(dst_key[0]), target_signal_kind=str(dst_key[3]), source_mode=int(mi), target_mode=int(mj), x_basis_id=str(x_meta.get('basis_id', '')), y_basis_id=str(y_meta.get('basis_id', '')), x_resolved_dim=int(x_meta.get('resolved_dim', src_modes.shape[1])), y_resolved_dim=int(y_meta.get('resolved_dim', dst_modes.shape[1])), reference_epoch=int(x_meta.get('reference_epoch', -1)))
                                pca_cross_rows.append(common_row(**kwargs))

        _write_layer_rows(checkpoint_dir, family_rows, manifest_rows=manifest_rows, manifest_base=checkpoint_base, artifact_name='analysis_curve')
        _write_layer_rows(checkpoint_dir, dispersion_rows, manifest_rows=manifest_rows, manifest_base=checkpoint_base, artifact_name='analysis_dispersion')
        _write_artifact(checkpoint_dir / 'pair_distance.csv', pair_rows, manifest_rows=manifest_rows, manifest_base=checkpoint_base, artifact_name='pair_distance')
        _write_layer_rows(checkpoint_dir, filter_rows, manifest_rows=manifest_rows, manifest_base=checkpoint_base, artifact_name='filter_snapshot')
        _write_layer_rows(checkpoint_dir, filter_distribution_rows, manifest_rows=manifest_rows, manifest_base=checkpoint_base, artifact_name='filter_distribution')
        for relation_type in ('adjacent',):
            relation_rows = [row for row in layer_distance_profile_rows if row.get('relation_type') == relation_type]
            _write_rows_to_dir(checkpoint_dir / 'layer_distance_profile' / relation_type, relation_rows, manifest_rows=manifest_rows, manifest_base=checkpoint_base, artifact_name='layer_distance_profile')
        for statistic in ('variance', 'mad'):
            statistic_rows = [row for row in layer_dispersion_profile_rows if row.get('dispersion_statistic') == statistic]
            _write_rows_to_dir(checkpoint_dir / 'layer_dispersion_profile' / statistic, statistic_rows, manifest_rows=manifest_rows, manifest_base=checkpoint_base, artifact_name='layer_dispersion_profile')
        if bool(args.enable_pairwise_dependency_appendix):
            _write_artifact(checkpoint_dir / 'pairwise_dependency_appendix.csv', appendix_rows, manifest_rows=manifest_rows, manifest_base=checkpoint_base, artifact_name='pairwise_dependency_appendix')
        if pca_enabled:
            _write_layer_rows(checkpoint_dir, pca_reference_rows, manifest_rows=manifest_rows, manifest_base=checkpoint_base, artifact_name='pca_reference')
        if pca_enabled and enable_pca_1d:
            _write_layer_rows(checkpoint_dir, pca_mode_rows, manifest_rows=manifest_rows, manifest_base=checkpoint_base, artifact_name='pca_mode_traces')
        if pca_enabled and enable_pca_mimo:
            _write_layer_rows(checkpoint_dir, pca_mimo_rows, manifest_rows=manifest_rows, manifest_base=checkpoint_base, artifact_name='pca_mimo_traces')
            _write_layer_rows(checkpoint_dir, pca_cross_rows, manifest_rows=manifest_rows, manifest_base=checkpoint_base, artifact_name='pca_cross_traces')

    traces_dir = output_root / 'traces'
    traces_dir.mkdir(parents=True, exist_ok=True)
    manifest_base = first_manifest_base or {'source_program': SOURCE_PROGRAM, 'dataset': str(args.dataset), 'run_id': 'analysis'}
    for (layer_name, layer_index, parameter, stat_name), history in sorted(filter_trend_history.items()):
        for epoch, stat_value, base in sorted(history, key=lambda item: item[0]):
            kwargs = dict(base)
            kwargs.update(category='filter_trend', layer=layer_name, layer_index=layer_index, checkpoint_epoch=epoch, parameter_name=parameter, statistic=stat_name, value=stat_value, value_unit='count' if stat_name == 'count' else _filter_value_unit(parameter))
            filter_trend_rows.append(common_row(**kwargs))
    for relation_type in ('adjacent',):
        relation_rows = [row for row in layer_distance_trend_rows if row.get('relation_type') == relation_type]
        _write_rows_to_dir(traces_dir / 'layer_distance_trend' / relation_type, relation_rows, manifest_rows=manifest_rows, manifest_base=manifest_base, artifact_name='layer_distance_trend')
    for statistic in ('variance', 'mad'):
        statistic_rows = [row for row in layer_dispersion_trend_rows if row.get('dispersion_statistic') == statistic]
        _write_rows_to_dir(traces_dir / 'layer_dispersion_trend' / statistic, statistic_rows, manifest_rows=manifest_rows, manifest_base=manifest_base, artifact_name='layer_dispersion_trend')
    for (layer_name, layer_index, parameter, _stat_name), group_rows in itertools.groupby(sorted(filter_trend_rows, key=lambda r: (r.get('layer_index', ''), r.get('layer', ''), r.get('parameter_name', ''), r.get('statistic', ''))), key=lambda r: (r.get('layer', ''), r.get('layer_index', ''), r.get('parameter_name', ''), r.get('statistic', ''))):
        _write_rows_to_dir(traces_dir / 'filter_trend' / _layer_folder(layer_name, layer_index), list(group_rows), manifest_rows=manifest_rows, manifest_base=manifest_base, artifact_name='filter_trend')
    for warning in ordering_warnings:
        manifest_rows.append(_manifest_row(base=manifest_base, artifact_name='checkpoint_ordering', path=Path(args.checkpoint), status='ok', message=warning))
    manifest_path = output_root / 'analysis_manifest.csv'
    write_common_csv(manifest_path, manifest_rows)
    print(json.dumps({'status': 'ok', 'source_program': SOURCE_PROGRAM, 'output_root': str(output_root), 'analysis_distance_metrics': list(distance_metrics), 'checkpoints': [str(p) for p in checkpoint_files]}, sort_keys=True))
    return 0


try:
    from src.patch_overlays.runtime_patch import patch_psd_analysis as _patch_psd_analysis
    _patch_psd_analysis(globals())
except Exception:
    pass

if __name__ == '__main__':
    raise SystemExit(main())
