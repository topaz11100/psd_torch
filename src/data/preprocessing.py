"""Official ``data_prep`` preprocessing for prepared experiment bundles."""

from __future__ import annotations

import inspect
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data.specs import available_dataset_tokens, canonicalize_dataset_name, get_dataset_spec
from src.data.storage import SINGLE_STRUCTURED_NPY_STORAGE_FORMAT, fsync_path, load_single_structured_split
from src.util.config import ensure_dir, save_json


_PREPROCESSING_SPEC_DOC = 'Spec/theory/data_prep/data_prep.md'
_PREPROCESSING_IMPL_SPEC_DOC = 'Spec/impl/spec/data_prep.md'
_DEAP_LABEL_AXIS_TO_INDEX = {
    'valence': 0,
    'arousal': 1,
}
_DEAP_SEGMENTS_PER_TRIAL = 20
_DEAP_SUBJECT_TEST_RATIO = 0.2


@dataclass(frozen=True)
class DatasetPrepContext:
    """Class representing ``DatasetPrepContext`` in the ``preprocessing`` module."""
    dataset_token: str
    raw_data_root: Path
    seed: int
    download: bool
    deap_label_axis: str
    deap_num_classes: int
    shd_dt_ms: float
    shd_max_time: float
    ssc_dt_ms: float
    ssc_max_time: float
    prep_profile: str


StreamingDatasetWriter = Callable[[DatasetPrepContext, Path, bool, int | None], Path]
_STREAMING_WRITERS: dict[str, StreamingDatasetWriter] = {}

_PROJECT_STANDARD_PREP_PROFILE = 'project_standard'
_STATIC_IMAGE_REPEAT_T = 4
_REINTERPRETATION_PREP_PROFILES = {
    'need_high_cifar10_dvs_t16': {
        'dataset_name': 'cifar10-dvs',
        'origin_code_root': 'Origin/need-high/event',
        'origin_paper': 'Need High Section 4.1 Table 2 CIFAR10-DVS Max-Former vs MS-QKFormer',
        'origin_config_path': 'Origin/need-high/event/cifar10dvs.yaml',
        'num_frames': 16,
        'training_view_name': 'model_input',
        'psd_view_name': 'event_frame_psd_view',
        'stored_shape': [16, 2, 128, 128],
        'psd_sample_axis': None,
        'psd_batch_axis': 0,
        'psd_time_axis': 1,
        'psd_row_axes': [2, 3, 4],
        'psd_feature_axes': [],
        'psd_token_axes': [],
        'psd_flatten_rule': 'flatten_polarity_y_x_axes_to_rows_preserve_frame_time',
        'psd_logical_shape': [32768, 16],
        'layout_source': 'author_code_profile',
    },
    'drf_shd_t250': {
        'dataset_name': 'shd',
        'origin_code_root': 'Origin/neuron_model/D-RF',
        'origin_paper': 'D-RF Section 5.1 Table 1 SHD D-RF vs BRF',
        'origin_config_path': 'Origin/neuron_model/D-RF/main_training_parallel.py',
        'shd_dt_ms': 1.0,
        'shd_max_time': 0.25,
        'training_view_name': 'model_input',
        'psd_view_name': 'model_input_psd_view',
        'stored_shape': [250, 700],
        'psd_sample_axis': None,
        'psd_batch_axis': 0,
        'psd_time_axis': 1,
        'psd_row_axes': [2],
        'psd_feature_axes': [],
        'psd_token_axes': [],
        'psd_flatten_rule': 'flatten_unit_axis_to_rows_preserve_time',
        'psd_logical_shape': [700, 250],
        'layout_source': 'author_code_profile',
    },
    'dh_snn_shd_t1000': {
        'dataset_name': 'shd',
        'origin_code_root': 'Origin/neuron_model/DH-SNN',
        'origin_paper': 'DH-SNN Fig. 3f Fig. 4f Table 1 SHD vanilla SFNN vs DH-SFNN',
        'origin_config_path': 'Origin/neuron_model/DH-SNN/README.md',
        'shd_dt_ms': 1.0,
        'shd_max_time': 1.0,
        'training_view_name': 'model_input',
        'psd_view_name': 'model_input_psd_view',
        'stored_shape': [1000, 700],
        'psd_sample_axis': None,
        'psd_batch_axis': 0,
        'psd_time_axis': 1,
        'psd_row_axes': [2],
        'psd_feature_axes': [],
        'psd_token_axes': [],
        'psd_flatten_rule': 'flatten_unit_axis_to_rows_preserve_time',
        'psd_logical_shape': [700, 1000],
        'layout_source': 'author_code_profile',
    },
}


def available_prep_profiles() -> tuple[str, ...]:
    return (_PROJECT_STANDARD_PREP_PROFILE, *sorted(_REINTERPRETATION_PREP_PROFILES))


def _resolve_prep_profile(dataset_token: str, prep_profile: str | None) -> tuple[str, dict[str, Any]]:
    profile = _PROJECT_STANDARD_PREP_PROFILE if prep_profile is None or str(prep_profile).strip() == '' else str(prep_profile).strip()
    dataset_token = canonicalize_dataset_name(dataset_token)
    if profile == _PROJECT_STANDARD_PREP_PROFILE:
        return profile, {
            'prep_profile': profile,
            'prep_profile_role': 'project_standard',
            'layout_source': 'project_standard_profile',
        }
    if profile not in _REINTERPRETATION_PREP_PROFILES:
        allowed = ', '.join(available_prep_profiles())
        raise ValueError(f'Unsupported prep_profile {profile!r}. Available: {allowed}.')
    payload = dict(_REINTERPRETATION_PREP_PROFILES[profile])
    expected_dataset = canonicalize_dataset_name(str(payload['dataset_name']))
    if expected_dataset != dataset_token:
        raise ValueError(f'prep_profile {profile!r} is only valid for dataset {expected_dataset!r}, got {dataset_token!r}.')
    payload['prep_profile'] = profile
    payload['prep_profile_role'] = 'reinterpretation'
    return profile, payload


def register_streaming_dataset_writer(*dataset_tokens: str) -> Callable[[StreamingDatasetWriter], StreamingDatasetWriter]:
    """Register one direct-to-disk writer for large-memory dataset families."""

    def decorator(function: StreamingDatasetWriter) -> StreamingDatasetWriter:
        """Handle ``decorator`` for the ``preprocessing`` module."""
        for token in dataset_tokens:
            canonical = canonicalize_dataset_name(token)
            existing = _STREAMING_WRITERS.get(canonical)
            if existing is not None and existing is not function:
                raise RuntimeError(f'Streaming dataset writer already registered for {canonical!r}.')
            _STREAMING_WRITERS[canonical] = function
        return function

    return decorator


def _resolve_streaming_dataset_writer(dataset_token: str) -> StreamingDatasetWriter | None:
    """Return one registered direct-to-disk writer when available."""

    return _STREAMING_WRITERS.get(canonicalize_dataset_name(dataset_token))


def _required_view_names(
    dataset_token: str,
    *,
    psd_axis_kind: str,
    training_view_name: str,
    psd_view_name: str,
) -> tuple[str, ...]:
    """Internal helper for ``required view names`` in the ``preprocessing`` module."""
    spec = get_dataset_spec(dataset_token)
    views = list(spec.required_view_names(psd_axis_kind=psd_axis_kind))
    for extra in (training_view_name, psd_view_name):
        if extra not in views:
            views.append(extra)
    return tuple(sorted(views))


def _as_float_tensor(array: Any) -> torch.Tensor:
    """Internal helper for ``as float tensor`` in the ``preprocessing`` module."""
    return torch.as_tensor(array, dtype=torch.float32)


def _as_long_tensor(array: Any) -> torch.Tensor:
    """Internal helper for ``as long tensor`` in the ``preprocessing`` module."""
    return torch.as_tensor(array, dtype=torch.long)


def _quantize_deap_labels(scores: np.ndarray, *, num_classes: int) -> np.ndarray:
    """Internal helper that quantize deap labels."""
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if int(num_classes) == 2:
        return np.where(scores <= 5.0, 0, 1).astype(np.int64)
    if int(num_classes) == 3:
        quantized = np.full(scores.shape, 1, dtype=np.int64)
        quantized[scores <= 3.0] = 0
        quantized[scores >= 7.0] = 2
        return quantized
    raise ValueError('DEAP deap_num_classes must be 2 or 3 according to Spec/theory/data_prep/data_prep.md.')


def _remove_deap_baseline(subject_eeg: np.ndarray) -> np.ndarray:
    """Internal helper that remove deap baseline."""
    subject_eeg = np.asarray(subject_eeg, dtype=np.float32)
    if subject_eeg.ndim != 3 or subject_eeg.shape[1] != 32 or subject_eeg.shape[2] < 8064:
        raise ValueError(f'Unexpected DEAP EEG tensor shape {subject_eeg.shape}; expected (trials, 32, >=8064).')
    baseline = subject_eeg[:, :, :384]
    baseline_template = (baseline[:, :, 0:128] + baseline[:, :, 128:256] + baseline[:, :, 256:384]) / 3.0
    stimulus = subject_eeg[:, :, 384:8064]
    if stimulus.shape[-1] != 7680:
        raise ValueError(f'Unexpected DEAP post-baseline length {stimulus.shape[-1]}; expected 7680.')
    tiled_baseline = np.tile(baseline_template, (1, 1, 60))
    return (stimulus - tiled_baseline).astype(np.float32)


def _base_split_metadata(dataset_token: str, split_name: str, *, psd_axis_kind: str, **extra: Any) -> dict[str, Any]:
    """Internal helper for ``base split metadata`` in the ``preprocessing`` module."""
    payload = {
        'preprocessing_spec_doc': _PREPROCESSING_SPEC_DOC,
        'preprocessing_impl_spec_doc': _PREPROCESSING_IMPL_SPEC_DOC,
        'dataset_token': dataset_token,
        'split': split_name,
        'split_internal_order_preserved': True,
        'psd_axis_kind': psd_axis_kind,
    }
    payload.update(extra)
    return payload


def _bundle_manifest(
    dataset_token: str,
    *,
    raw_data_root: Path,
    seed: int,
    psd_axis_kind: str,
    training_view_name: str | None = None,
    psd_view_name: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Internal helper for ``bundle manifest`` in the ``preprocessing`` module."""
    spec = get_dataset_spec(dataset_token)
    payload = {
        'dataset_name': dataset_token,
        'files': {'train': 'train.npy', 'test': 'test.npy'},
        'preprocessing_spec_doc': _PREPROCESSING_SPEC_DOC,
        'preprocessing_impl_spec_doc': _PREPROCESSING_IMPL_SPEC_DOC,
        'split_internal_order_preserved': True,
        'raw_data_root': str(raw_data_root),
        'seed': int(seed),
        'psd_axis_kind': str(psd_axis_kind),
        'training_view_name': spec.training_view_name if training_view_name is None else str(training_view_name),
        'psd_view_name': spec.psd_view_name if psd_view_name is None else str(psd_view_name),
        'progress_logger': 'tqdm',
        'default_hidden_sizes': list(spec.default_hidden_sizes),
        'storage_format': SINGLE_STRUCTURED_NPY_STORAGE_FORMAT,
    }
    payload.update(extra)
    return payload


@dataclass(frozen=True)
class _SingleFileStorageContract:
    """Storage/view contract for one dataset under the single-file mmap format."""

    stored_view_name: str
    training_view_name: str
    psd_view_name: str
    available_views: tuple[str, ...]


def _singlefile_storage_contract(dataset_token: str) -> _SingleFileStorageContract:
    """Return the dataset-specific single-file storage contract."""

    canonical = canonicalize_dataset_name(dataset_token)
    spec = get_dataset_spec(canonical)
    if canonical in {'s-mnist', 'ps-mnist', 's-cifar10', 'deap'}:
        return _SingleFileStorageContract(
            stored_view_name='model_input',
            training_view_name='model_input',
            psd_view_name=spec.psd_view_name,
            available_views=('model_input', 'psd_input', spec.psd_view_name),
        )
    if canonical in {'uci-har', 'shd', 'ssc'}:
        return _SingleFileStorageContract(
            stored_view_name='model_input',
            training_view_name='model_input',
            psd_view_name=spec.psd_view_name,
            available_views=('model_input', 'psd_input', spec.psd_view_name, 'sequence_input'),
        )
    if canonical in {'mnist', 'cifar-10', 'cifar-100'}:
        return _SingleFileStorageContract(
            stored_view_name='model_input',
            training_view_name='model_input',
            psd_view_name=spec.psd_view_name,
            available_views=('model_input', 'psd_input', spec.psd_view_name, 'original_input', 'flatten_input', 'sequence_input'),
        )
    if canonical in {'n-mnist', 'cifar10-dvs', 'dvs128-gesture'}:
        return _SingleFileStorageContract(
            stored_view_name='model_input',
            training_view_name='model_input',
            psd_view_name=spec.psd_view_name,
            available_views=('model_input', 'psd_input', spec.psd_view_name, 'original_input', 'flatten_input', 'sequence_input'),
        )
    raise ValueError(f'Unsupported single-file storage contract dataset: {dataset_token!r}.')


def _runtime_sequence_shape_for_dataset(dataset_token: str, stored_shape: tuple[int, ...]) -> tuple[int, int]:
    """Return the runtime ``(sequence_length, input_dim)`` implied by one stored sample shape."""

    canonical = canonicalize_dataset_name(dataset_token)
    shape = tuple(int(v) for v in stored_shape)
    if canonical in {'s-mnist', 'ps-mnist', 's-cifar10', 'deap'}:
        if len(shape) != 2:
            raise ValueError(f'{canonical} stored shape must be rank 2, got {shape}.')
        return int(shape[0]), int(shape[1])
    if canonical == 'uci-har':
        if len(shape) != 2:
            raise ValueError(f'{canonical} stored shape must be rank 2, got {shape}.')
        return int(shape[0]), int(shape[1])
    if canonical in {'shd', 'ssc'}:
        if len(shape) != 2:
            raise ValueError(f'{canonical} stored shape must be rank 2, got {shape}.')
        return int(shape[0]), int(shape[1])
    if canonical in {'mnist', 'cifar-10', 'cifar-100'}:
        if len(shape) != 4:
            raise ValueError(f'{canonical} stored shape must be rank 4 as (T,C,H,W), got {shape}.')
        return int(shape[0]), int(shape[1]) * int(shape[2]) * int(shape[3])
    if canonical in {'n-mnist', 'cifar10-dvs', 'dvs128-gesture'}:
        if len(shape) != 4:
            raise ValueError(f'{canonical} stored shape must be rank 4, got {shape}.')
        return int(shape[0]), int(shape[1]) * int(shape[2]) * int(shape[3])
    raise ValueError(f'Unsupported runtime sequence-shape dataset: {dataset_token!r}.')


def _psd_axis_metadata_for_dataset(dataset_token: str, stored_shape: tuple[int, ...], *, psd_axis_kind: str) -> dict[str, Any]:
    """Return required manifest axis metadata for the official logical PSD view.

    ``psd_sample_axis`` intentionally stays ``None`` because each structured
    record stores one sample and therefore the record-local ``input`` field has
    no sample dimension. ``psd_batch_axis`` and the other PSD axes describe the
    runtime batch tensor produced by the prepared-bundle loader for
    ``psd_view_name``.
    """

    canonical = canonicalize_dataset_name(dataset_token)
    shape = tuple(int(v) for v in stored_shape)
    _ = str(psd_axis_kind)
    static_repeat_metadata = {
        'is_static_repeat': False,
        'static_repeat_T': None,
        'repeat_schedule': None,
    }
    if canonical in {'s-mnist', 'ps-mnist', 's-cifar10', 'shd', 'ssc', 'deap', 'uci-har'}:
        if len(shape) != 2:
            raise ValueError(f'{canonical} stored shape must be rank 2 for temporal axis metadata, got {shape}.')
        time_steps, rows = int(shape[0]), int(shape[1])
        semantics = {
            's-mnist': 'sequence_time',
            'ps-mnist': 'sequence_time',
            's-cifar10': 'sequence_time',
            'shd': 'event_time',
            'ssc': 'event_time',
            'deap': 'sensor_time',
            'uci-har': 'sensor_time',
        }.get(canonical, 'sequence_time')
        return {
            'model_input_axis_order': ['time', 'channel'],
            'psd_sample_axis': None,
            'psd_batch_axis': 0,
            'psd_time_axis': 2,
            'psd_row_axes': [1],
            'psd_feature_axes': [],
            'psd_token_axes': [],
            'psd_flatten_rule': 'prepared_psd_view_rows_time_from_time_major_model_input',
            'psd_logical_shape': [rows, time_steps],
            'physical_input_shape': list(shape),
            'time_axis_semantics': semantics,
            'stored_order_is_model_input_order': True,
            **static_repeat_metadata,
        }
    if canonical in {'mnist', 'cifar-10', 'cifar-100'}:
        if len(shape) != 4:
            raise ValueError(f'{canonical} stored shape must be rank 4 as (T,C,H,W) for image axis metadata, got {shape}.')
        repeat_t, channels, height, width = [int(v) for v in shape]
        return {
            'model_input_axis_order': ['time', 'channel', 'height', 'width'],
            'psd_sample_axis': None,
            'psd_batch_axis': 0,
            'psd_time_axis': 1,
            'psd_row_axes': [2, 3, 4],
            'psd_feature_axes': [],
            'psd_token_axes': [],
            'psd_flatten_rule': 'flatten_channel_height_width_axes_to_rows_preserve_static_repeat_time',
            'psd_logical_shape': [channels * height * width, repeat_t],
            'physical_input_shape': list(shape),
            'time_axis_semantics': 'static_repeat_time',
            'stored_order_is_model_input_order': True,
            'is_static_repeat': True,
            'static_repeat_T': repeat_t,
            'repeat_schedule': 'prepared_storage_repeats_same_image_frame_each_timestep',
        }
    if canonical in {'n-mnist', 'cifar10-dvs', 'dvs128-gesture'}:
        if len(shape) != 4:
            raise ValueError(f'{canonical} stored shape must be rank 4 for DVS axis metadata, got {shape}.')
        frames, channels, height, width = [int(v) for v in shape]
        return {
            'model_input_axis_order': ['time', 'channel', 'height', 'width'],
            'psd_sample_axis': None,
            'psd_batch_axis': 0,
            'psd_time_axis': 2,
            'psd_row_axes': [1],
            'psd_feature_axes': [],
            'psd_token_axes': [],
            'psd_flatten_rule': 'flatten_channel_height_width_axes_to_rows_preserve_frame_time',
            'psd_logical_shape': [channels * height * width, frames],
            'physical_input_shape': list(shape),
            'time_axis_semantics': 'event_time',
            'stored_order_is_model_input_order': True,
            **static_repeat_metadata,
        }
    raise ValueError(f'Unsupported dataset for PSD axis metadata: {dataset_token!r}.')


def _stored_input_array_for_dataset(dataset_token: str, value: torch.Tensor) -> np.ndarray:
    """Convert one canonical stored view tensor into the official on-disk NumPy dtype."""

    canonical = canonicalize_dataset_name(dataset_token)
    array = np.asarray(torch.as_tensor(value).detach().cpu())
    if canonical in {'shd', 'ssc'}:
        if not np.all((array == 0) | (array == 1)):
            raise ValueError(f'{canonical} binary occupancy storage expected only 0/1 values.')
        return array.astype(np.uint8, copy=False)
    if canonical in {'n-mnist', 'cifar10-dvs', 'dvs128-gesture'}:
        rounded = np.rint(array)
        if np.allclose(array, rounded) and float(rounded.min(initial=0.0)) >= 0.0:
            max_value = int(rounded.max(initial=0.0))
            if max_value <= np.iinfo(np.uint8).max:
                return rounded.astype(np.uint8)
            if max_value <= np.iinfo(np.uint16).max:
                return rounded.astype(np.uint16)
            return rounded.astype(np.uint32)
    return array.astype(np.float32, copy=False)


def _validate_singlefile_split_reopen(
    path: Path,
    *,
    expected_count: int,
    expected_dtype: np.dtype,
    expected_input_shape: tuple[int, ...],
) -> None:
    """Reopen and validate one newly-written structured split payload."""

    reopened = load_single_structured_split(path, mmap_mode='r')
    if int(reopened.shape[0]) != int(expected_count):
        raise ValueError(f'Reopened split {path} length mismatch: {int(reopened.shape[0])} vs {int(expected_count)}.')
    if reopened.dtype != expected_dtype:
        raise ValueError(f'Reopened split {path} dtype mismatch: {reopened.dtype} vs {expected_dtype}.')
    if tuple(int(v) for v in reopened.dtype.fields['input'][0].shape) != tuple(int(v) for v in expected_input_shape):
        raise ValueError(
            f'Reopened split {path} input shape mismatch: '
            f'{tuple(int(v) for v in reopened.dtype.fields["input"][0].shape)} vs {tuple(int(v) for v in expected_input_shape)}.'
        )
    sample_indices = np.asarray(reopened['sample_index'])
    if sample_indices.size > 0:
        expected_indices = np.arange(int(sample_indices.shape[0]), dtype=np.int64)
        if not np.array_equal(sample_indices, expected_indices):
            raise ValueError(f'Reopened split {path} sample_index values are not the canonical contiguous prefix.')
        _ = reopened[0]
        _ = reopened[-1]


def _prepare_output_dir_for_streaming(
    *,
    dataset_token: str,
    prep_root: Path,
    overwrite: bool,
    prep_profile_name: str,
) -> Path:
    """Create the official output directory for a streaming data_prep writer."""

    output_root = Path(prep_root).expanduser().resolve()
    if prep_profile_name != _PROJECT_STANDARD_PREP_PROFILE:
        output_root = output_root / prep_profile_name
    out_dir = output_root / canonicalize_dataset_name(dataset_token)
    if out_dir.exists():
        if not overwrite and any(out_dir.iterdir()):
            raise FileExistsError(
                f'Prepared bundle already exists under {out_dir}. '
                'Use --force_overwrite=true to replace it.'
            )
        if overwrite:
            shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    return out_dir


def _write_streamed_structured_split(
    *,
    dataset_token: str,
    split_name: str,
    dataset_root: Path,
    total_count: int,
    input_shape: tuple[int, ...],
    input_dtype: np.dtype,
    sample_iter: Any,
    max_samples: int | None,
) -> dict[str, Any]:
    """Write one split by consuming one already-preprocessed sample at a time."""

    total = int(total_count)
    if max_samples is not None:
        resolved_max = int(max_samples)
        if resolved_max <= 0:
            raise ValueError('max_samples must be positive when provided.')
        total = min(total, resolved_max)
    if total <= 0:
        raise ValueError(f'Prepared split {split_name!r} is empty after applying max_samples.')

    input_shape = tuple(int(v) for v in input_shape)
    input_dtype = np.dtype(input_dtype)
    record_dtype = np.dtype([
        ('sample_index', np.int64),
        ('label', np.int64),
        ('input', input_dtype, input_shape),
    ])
    final_path = dataset_root / f'{split_name}.npy'
    tmp_path = dataset_root / f'{split_name}.tmp.npy'
    written = 0
    try:
        writer = np.lib.format.open_memmap(tmp_path, mode='w+', dtype=record_dtype, shape=(int(total),))
        with tqdm(total=int(total), desc=f'{dataset_token}:{split_name}:write', leave=False) as progress:
            for raw_sample_index, label, sample_array in sample_iter:
                if written >= total:
                    break
                sample = np.asarray(sample_array)
                if tuple(int(v) for v in sample.shape) != input_shape:
                    raise ValueError(
                        f'{dataset_token}:{split_name} sample {raw_sample_index} stored shape '
                        f'{tuple(int(v) for v in sample.shape)} does not match expected {input_shape}.'
                    )
                if np.dtype(sample.dtype) != input_dtype:
                    sample = sample.astype(input_dtype, copy=False)
                writer['sample_index'][written] = int(raw_sample_index)
                writer['label'][written] = int(label)
                writer['input'][written] = sample
                written += 1
                progress.update(1)
        if written != total:
            raise ValueError(f'{dataset_token}:{split_name} writer produced {written} records, expected {total}.')
        writer.flush()
        del writer
        fsync_path(tmp_path)
        _validate_singlefile_split_reopen(
            tmp_path,
            expected_count=int(total),
            expected_dtype=record_dtype,
            expected_input_shape=input_shape,
        )
        tmp_path.replace(final_path)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise
    return {
        'path': final_path.name,
        'num_samples': int(total),
        'stored_shape': [int(v) for v in input_shape],
        'stored_dtype': str(input_dtype),
    }


def _save_streaming_manifest(
    *,
    dataset_token: str,
    context: DatasetPrepContext,
    out_dir: Path,
    contract: _SingleFileStorageContract,
    split_infos: dict[str, dict[str, Any]],
    psd_axis_kind: str,
    profile_payload: dict[str, Any],
    metadata_extra: dict[str, Any] | None = None,
    max_samples: int | None = None,
) -> None:
    """Save the official manifest shared by all one-sample streaming writers."""

    stored_shape = tuple(int(v) for v in split_infos['train']['stored_shape'])
    test_shape = tuple(int(v) for v in split_infos['test']['stored_shape'])
    if stored_shape != test_shape:
        raise ValueError(f'Stored input shape mismatch between train/test: {stored_shape} vs {test_shape}.')
    stored_dtype = str(split_infos['train']['stored_dtype'])
    if stored_dtype != str(split_infos['test']['stored_dtype']):
        raise ValueError(f'Stored dtype mismatch between train/test: {stored_dtype} vs {split_infos["test"]["stored_dtype"]}.')
    sequence_length, input_dim = _runtime_sequence_shape_for_dataset(dataset_token, stored_shape)
    manifest_axis_metadata = _psd_axis_metadata_for_dataset(
        dataset_token,
        stored_shape,
        psd_axis_kind=str(psd_axis_kind),
    )

    extra_payload: dict[str, Any] = {}
    if metadata_extra is not None:
        extra_payload.update(dict(metadata_extra))
    extra_payload.update(dict(profile_payload))
    extra_payload.update(dict(manifest_axis_metadata))

    manifest = _bundle_manifest(
        dataset_token,
        raw_data_root=context.raw_data_root,
        seed=context.seed,
        psd_axis_kind=str(psd_axis_kind),
        training_view_name=contract.training_view_name,
        psd_view_name=contract.psd_view_name,
        stored_view_name=contract.stored_view_name,
        stored_view_name_by_split={'train': contract.stored_view_name, 'test': contract.stored_view_name},
        available_views=list(contract.available_views),
        stored_shape=list(stored_shape),
        stored_dtype=stored_dtype,
        label_dtype='int64',
        sample_index_dtype='int64',
        split_sizes={'train': int(split_infos['train']['num_samples']), 'test': int(split_infos['test']['num_samples'])},
        sequence_length=int(sequence_length),
        input_dim=int(input_dim),
        writer_backend='open_memmap_one_sample_streaming',
        one_sample_streaming_writer=True,
        **extra_payload,
    )
    if max_samples is not None:
        manifest['max_samples_truncated'] = int(max_samples)
    save_json(out_dir / 'manifest.json', manifest)


def _channel_major_flatten_from_static_image(images: torch.Tensor) -> torch.Tensor:
    """Internal helper for ``channel major flatten from static image`` in the ``preprocessing`` module."""
    return images.reshape(images.shape[0], images.shape[1], -1)


def _sequential_from_static_image(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal helper for ``sequential from static image`` in the ``preprocessing`` module."""
    flat = _channel_major_flatten_from_static_image(images)
    model_input = flat.transpose(1, 2)
    psd_input = flat
    return model_input, psd_input


def _torchvision_image_to_numpy(image: Any) -> np.ndarray:
    """Convert one torchvision image sample to channel-first float32 numpy."""

    tensor = torch.as_tensor(image, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim == 3 and int(tensor.shape[-1]) in {1, 3} and int(tensor.shape[0]) not in {1, 3}:
        tensor = tensor.permute(2, 0, 1).contiguous()
    if tensor.ndim != 3:
        raise ValueError(f'Expected one channel-first image tensor, got shape {tuple(tensor.shape)}.')
    if float(tensor.max().item()) > 1.0:
        tensor = tensor / 255.0
    return np.asarray(tensor.detach().cpu(), dtype=np.float32)


def _iter_torchvision_stored_samples(
    dataset: Any,
    *,
    dataset_token: str,
    permutation: torch.Tensor | None = None,
):
    """Yield one official stored sample at a time from a torchvision split."""

    canonical = canonicalize_dataset_name(dataset_token)
    perm_np = None if permutation is None else np.asarray(permutation.detach().cpu(), dtype=np.int64)
    for index in range(len(dataset)):
        image, label = dataset[index]
        array = _torchvision_image_to_numpy(image)
        if canonical in {'s-mnist', 'ps-mnist', 's-cifar10'}:
            model_input = array.reshape(int(array.shape[0]), -1).T.astype(np.float32, copy=False)
            if perm_np is not None:
                model_input = model_input[perm_np, :]
            yield int(index), int(label), model_input
        elif canonical in {'mnist', 'cifar-10', 'cifar-100'}:
            repeated = np.repeat(array[None, ...], _STATIC_IMAGE_REPEAT_T, axis=0).astype(np.float32, copy=False)
            yield int(index), int(label), repeated
        else:
            raise ValueError(f'Unsupported torchvision streaming dataset {dataset_token!r}.')


@register_streaming_dataset_writer('s-mnist', 'ps-mnist', 'mnist')
def _stream_write_mnist_family_bundle(
    context: DatasetPrepContext,
    prep_root: Path,
    overwrite: bool,
    max_samples: int | None,
) -> Path:
    """One-sample streaming writer for MNIST-derived official datasets."""

    from torchvision import datasets, transforms

    dataset_token = canonicalize_dataset_name(context.dataset_token)
    profile_name, profile_payload = _resolve_prep_profile(dataset_token, context.prep_profile)
    out_dir = _prepare_output_dir_for_streaming(
        dataset_token=dataset_token,
        prep_root=Path(prep_root),
        overwrite=bool(overwrite),
        prep_profile_name=profile_name,
    )
    if dataset_token in {'s-mnist', 'ps-mnist'}:
        input_shape = (784, 1)
        psd_axis_kind = 'temporal'
        dataset_root = context.raw_data_root / dataset_token
        permutation = None
        metadata_extra: dict[str, Any] = {'normalization_rule': 'ToTensor_[0,1]', 'flatten_order': 'raster'}
        if dataset_token == 'ps-mnist':
            generator = torch.Generator().manual_seed(int(context.seed))
            permutation = torch.randperm(input_shape[0], generator=generator)
            metadata_extra['permutation_seed'] = int(context.seed)
            metadata_extra['permutation'] = [int(v) for v in permutation.tolist()]
    else:
        input_shape = (_STATIC_IMAGE_REPEAT_T, 1, 28, 28)
        psd_axis_kind = 'static_repeat'
        dataset_root = context.raw_data_root / 'mnist'
        permutation = None
        metadata_extra = {
            'normalization_rule': 'ToTensor_[0,1]',
            'flatten_order': 'raster',
            'original_shape': [1, 28, 28],
            'sequence_input_rule': 'prepared_static_repeat_TCHW',
            'cnn_input_shape': [_STATIC_IMAGE_REPEAT_T, 1, 28, 28],
        }
    contract = _singlefile_storage_contract(dataset_token)
    split_infos: dict[str, dict[str, Any]] = {}
    for split_name, train_flag in tqdm((('train', True), ('test', False)), desc=f'{dataset_token}:splits', leave=False):
        raw_dataset = datasets.MNIST(root=str(dataset_root), train=bool(train_flag), transform=transforms.ToTensor(), download=bool(context.download))
        split_infos[split_name] = _write_streamed_structured_split(
            dataset_token=dataset_token,
            split_name=split_name,
            dataset_root=out_dir,
            total_count=len(raw_dataset),
            input_shape=tuple(input_shape),
            input_dtype=np.dtype(np.float32),
            sample_iter=_iter_torchvision_stored_samples(raw_dataset, dataset_token=dataset_token, permutation=permutation),
            max_samples=max_samples,
        )
    _save_streaming_manifest(
        dataset_token=dataset_token,
        context=context,
        out_dir=out_dir,
        contract=contract,
        split_infos=split_infos,
        psd_axis_kind=psd_axis_kind,
        profile_payload=profile_payload,
        metadata_extra=metadata_extra,
        max_samples=max_samples,
    )
    return out_dir


@register_streaming_dataset_writer('s-cifar10', 'cifar-10', 'cifar-100')
def _stream_write_cifar10_family_bundle(
    context: DatasetPrepContext,
    prep_root: Path,
    overwrite: bool,
    max_samples: int | None,
) -> Path:
    """One-sample streaming writer for CIFAR-10-derived official datasets."""

    from torchvision import datasets, transforms

    dataset_token = canonicalize_dataset_name(context.dataset_token)
    profile_name, profile_payload = _resolve_prep_profile(dataset_token, context.prep_profile)
    out_dir = _prepare_output_dir_for_streaming(
        dataset_token=dataset_token,
        prep_root=Path(prep_root),
        overwrite=bool(overwrite),
        prep_profile_name=profile_name,
    )
    if dataset_token == 's-cifar10':
        input_shape = (1024, 3)
        psd_axis_kind = 'temporal'
        dataset_root = context.raw_data_root / 's-cifar10'
        dataset_cls = datasets.CIFAR10
        metadata_extra: dict[str, Any] = {
            'normalization_rule': 'ToTensor_[0,1]',
            'flatten_order': 'raster',
            'color_preserving': True,
        }
    else:
        input_shape = (_STATIC_IMAGE_REPEAT_T, 3, 32, 32)
        psd_axis_kind = 'static_repeat'
        dataset_root = context.raw_data_root / dataset_token
        dataset_cls = datasets.CIFAR100 if dataset_token == 'cifar-100' else datasets.CIFAR10
        metadata_extra = {
            'normalization_rule': 'ToTensor_[0,1]',
            'flatten_order': 'raster',
            'original_shape': [3, 32, 32],
            'color_preserving': True,
            'sequence_input_rule': 'prepared_static_repeat_TCHW',
            'cnn_input_shape': [_STATIC_IMAGE_REPEAT_T, 3, 32, 32],
        }
    contract = _singlefile_storage_contract(dataset_token)
    split_infos: dict[str, dict[str, Any]] = {}
    for split_name, train_flag in tqdm((('train', True), ('test', False)), desc=f'{dataset_token}:splits', leave=False):
        raw_dataset = dataset_cls(root=str(dataset_root), train=bool(train_flag), transform=transforms.ToTensor(),
                                  download=bool(context.download),)
        split_infos[split_name] = _write_streamed_structured_split(
            dataset_token=dataset_token,
            split_name=split_name,
            dataset_root=out_dir,
            total_count=len(raw_dataset),
            input_shape=tuple(input_shape),
            input_dtype=np.dtype(np.float32),
            sample_iter=_iter_torchvision_stored_samples(raw_dataset, dataset_token=dataset_token),
            max_samples=max_samples,
        )
    _save_streaming_manifest(
        dataset_token=dataset_token,
        context=context,
        out_dir=out_dir,
        contract=contract,
        split_infos=split_infos,
        psd_axis_kind=psd_axis_kind,
        profile_payload=profile_payload,
        metadata_extra=metadata_extra,
        max_samples=max_samples,
    )
    return out_dir


def _candidate_existing_dirs(root: Path, names: list[str]) -> Path:
    """Internal helper for ``candidate existing dirs`` in the ``preprocessing`` module."""
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'Could not locate expected raw dataset directory under {root}: {names}')


_UCI_HAR_CHANNELS = [
    'total_acc_x',
    'total_acc_y',
    'total_acc_z',
    'body_gyro_x',
    'body_gyro_y',
    'body_gyro_z',
]


def _resolve_uci_har_root(raw_data_root: Path) -> Path:
    dataset_root = _candidate_existing_dirs(raw_data_root, ['uci-har', 'UCI HAR Dataset'])
    if dataset_root.name != 'UCI HAR Dataset' and (dataset_root / 'UCI HAR Dataset').exists():
        dataset_root = dataset_root / 'UCI HAR Dataset'
    return dataset_root


def _uci_har_count_samples(dataset_root: Path, split_name: str) -> int:
    labels_path = dataset_root / split_name / f'y_{split_name}.txt'
    if not labels_path.exists():
        raise FileNotFoundError(f'Missing UCI-HAR label file: {labels_path}')
    with labels_path.open('r', encoding='utf-8') as handle:
        return sum(1 for line in handle if line.strip())


def _iter_uci_har_raw_samples(dataset_root: Path, split_name: str):
    signals_dir = dataset_root / split_name / 'Inertial Signals'
    signal_paths = []
    for channel in _UCI_HAR_CHANNELS:
        path = signals_dir / f'{channel}_{split_name}.txt'
        if not path.exists():
            raise FileNotFoundError(f'Missing UCI-HAR signal file: {path}')
        signal_paths.append(path)
    labels_path = dataset_root / split_name / f'y_{split_name}.txt'
    handles = [path.open('r', encoding='utf-8') for path in signal_paths]
    label_handle = labels_path.open('r', encoding='utf-8')
    try:
        for sample_index, lines in enumerate(zip(*handles, label_handle)):
            *signal_lines, label_line = lines
            sample = np.empty((128, len(_UCI_HAR_CHANNELS)), dtype=np.float32)
            for channel_index, line in enumerate(signal_lines):
                values = np.fromstring(line.strip(), sep=' ', dtype=np.float32)
                if values.shape[0] != 128:
                    raise ValueError(f'UCI-HAR {split_name} sample {sample_index} channel length {values.shape[0]} != 128.')
                sample[:, channel_index] = values
            label = int(label_line.strip()) - 1
            yield int(sample_index), int(label), sample
    finally:
        for handle in handles:
            handle.close()
        label_handle.close()


def _compute_uci_har_train_stats(dataset_root: Path) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    channel_sum = np.zeros((len(_UCI_HAR_CHANNELS),), dtype=np.float64)
    channel_sq_sum = np.zeros((len(_UCI_HAR_CHANNELS),), dtype=np.float64)
    for _sample_index, _label, sample in tqdm(_iter_uci_har_raw_samples(dataset_root, 'train'), desc='uci-har:train_stats', leave=False):
        channel_sum += sample.sum(axis=0, dtype=np.float64)
        channel_sq_sum += np.square(sample.astype(np.float64, copy=False)).sum(axis=0)
        count += int(sample.shape[0])
    if count <= 0:
        raise ValueError('UCI-HAR train split is empty.')
    mean = channel_sum / float(count)
    variance = np.maximum(channel_sq_sum / float(count) - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std = np.where(std < 1.0e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _iter_uci_har_normalized_samples(dataset_root: Path, split_name: str, *, mean: np.ndarray, std: np.ndarray):
    mean = np.asarray(mean, dtype=np.float32).reshape(1, -1)
    std = np.asarray(std, dtype=np.float32).reshape(1, -1)
    for sample_index, label, sample in _iter_uci_har_raw_samples(dataset_root, split_name):
        normalized = ((sample.astype(np.float32, copy=False) - mean) / std).astype(np.float32, copy=False)
        yield int(sample_index), int(label), normalized


@register_streaming_dataset_writer('uci-har')
def _stream_write_uci_har_bundle(
    context: DatasetPrepContext,
    prep_root: Path,
    overwrite: bool,
    max_samples: int | None,
) -> Path:
    """One-sample streaming writer for UCI-HAR."""

    dataset_token = 'uci-har'
    profile_name, profile_payload = _resolve_prep_profile(dataset_token, context.prep_profile)
    dataset_root = _resolve_uci_har_root(context.raw_data_root)
    out_dir = _prepare_output_dir_for_streaming(
        dataset_token=dataset_token,
        prep_root=Path(prep_root),
        overwrite=bool(overwrite),
        prep_profile_name=profile_name,
    )
    mean, std = _compute_uci_har_train_stats(dataset_root)
    contract = _singlefile_storage_contract(dataset_token)
    split_infos: dict[str, dict[str, Any]] = {}
    for split_name in tqdm(('train', 'test'), desc='uci-har:splits', leave=False):
        split_infos[split_name] = _write_streamed_structured_split(
            dataset_token=dataset_token,
            split_name=split_name,
            dataset_root=out_dir,
            total_count=_uci_har_count_samples(dataset_root, split_name),
            input_shape=(128, len(_UCI_HAR_CHANNELS)),
            input_dtype=np.dtype(np.float32),
            sample_iter=_iter_uci_har_normalized_samples(dataset_root, split_name, mean=mean, std=std),
            max_samples=max_samples,
        )
    metadata_extra = {
        'normalization_rule': 'train_only_channel_zscore',
        'normalization_mean': [float(v) for v in mean.reshape(-1).tolist()],
        'normalization_std': [float(v) for v in std.reshape(-1).tolist()],
        'channels': list(_UCI_HAR_CHANNELS),
        'window_length': 128,
        'sampling_rate_hz': 50,
        'sequence_input_rule': 'time_major_model_input',
        'psd_logical_view_rule': 'metadata_time_axis_and_row_axes',
    }
    _save_streaming_manifest(
        dataset_token=dataset_token,
        context=context,
        out_dir=out_dir,
        contract=contract,
        split_infos=split_infos,
        psd_axis_kind='temporal',
        profile_payload=profile_payload,
        metadata_extra=metadata_extra,
        max_samples=max_samples,
    )
    return out_dir


def _resolve_deap_root(raw_data_root: Path) -> Path:
    dataset_root = _candidate_existing_dirs(raw_data_root, ['deap', 'data_preprocessed_python'])
    if dataset_root.name != 'data_preprocessed_python' and (dataset_root / 'data_preprocessed_python').exists():
        dataset_root = dataset_root / 'data_preprocessed_python'
    return dataset_root


def _deap_subject_split_plan(
    *,
    subject_files: list[Path],
    axis_index: int,
    num_classes: int,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Build deterministic subject-independent DEAP split metadata.

    Split unit is subject, not segment.
    Therefore, no subject appears in both train and test.
    """

    if not subject_files:
        raise ValueError('DEAP subject_files must not be empty.')

    rng = np.random.default_rng(int(seed))
    subject_files = list(sorted(subject_files))
    subject_order = rng.permutation(len(subject_files)).astype(np.int64)

    test_subject_count = max(1, int(round(len(subject_files) * _DEAP_SUBJECT_TEST_RATIO)))
    test_subject_positions = set(int(v) for v in subject_order[:test_subject_count].tolist())

    plans: dict[str, list[dict[str, Any]]] = {'train': [], 'test': []}
    counts = {'train': 0, 'test': 0}

    for subject_position, subject_path in enumerate(tqdm(subject_files, desc='deap:split_plan', leave=False)):
        with subject_path.open('rb') as handle:
            subject = pickle.load(handle, encoding='latin1')

        labels = np.asarray(subject['labels'], dtype=np.float32)[:, int(axis_index)]
        quantized_labels = _quantize_deap_labels(labels, num_classes=int(num_classes))

        segment_labels = np.repeat(quantized_labels, _DEAP_SEGMENTS_PER_TRIAL).astype(np.int64)
        segment_indices = np.arange(segment_labels.shape[0], dtype=np.int64)

        split_name = 'test' if int(subject_position) in test_subject_positions else 'train'

        plans[split_name].append(
            {
                'path': subject_path,
                'indices': segment_indices,
                'labels': segment_labels,
            }
        )
        counts[split_name] += int(segment_indices.shape[0])

    return plans, counts

def _deap_label_distribution_summary(
    plans: dict[str, list[dict[str, Any]]],
    *,
    num_classes: int,
) -> dict[str, dict[str, Any]]:
    """Summarize DEAP label distribution and majority-class baseline."""

    summary: dict[str, dict[str, Any]] = {}

    for split_name, entries in plans.items():
        if entries:
            labels = np.concatenate(
                [np.asarray(entry['labels'], dtype=np.int64).reshape(-1) for entry in entries]
            )
        else:
            labels = np.empty((0,), dtype=np.int64)

        counts = np.bincount(labels, minlength=int(num_classes)).astype(np.int64)
        total = int(counts.sum())

        if total == 0:
            majority_class = None
            majority_baseline_accuracy = 0.0
        else:
            majority_class = int(counts.argmax())
            majority_baseline_accuracy = float(counts.max() / total)

        summary[str(split_name)] = {
            'total': total,
            'class_counts': [int(v) for v in counts.tolist()],
            'majority_class': majority_class,
            'majority_baseline_accuracy': majority_baseline_accuracy,
        }

    return summary

def _deap_trial_segments(trial_eeg: np.ndarray) -> np.ndarray:
    trial = np.asarray(trial_eeg, dtype=np.float32)
    if trial.ndim != 2 or trial.shape[0] != 32 or trial.shape[1] < 8064:
        raise ValueError(f'Unexpected DEAP trial EEG shape {trial.shape}; expected (32, >=8064).')
    baseline = trial[:, :384]
    baseline_template = (baseline[:, 0:128] + baseline[:, 128:256] + baseline[:, 256:384]) / 3.0
    stimulus = trial[:, 384:8064]
    if stimulus.shape[-1] != 7680:
        raise ValueError(f'Unexpected DEAP stimulus length {stimulus.shape[-1]}; expected 7680.')
    corrected = stimulus - np.tile(baseline_template, (1, 60))
    return corrected.reshape(32, 20, 384).transpose(1, 2, 0).astype(np.float32, copy=False)


def _iter_deap_split_samples(split_plan: list[dict[str, Any]], *, axis_index: int, num_classes: int):
    written_index = 0
    for entry in split_plan:
        subject_path = Path(entry['path'])
        wanted_indices = np.asarray(entry['indices'], dtype=np.int64).reshape(-1)
        wanted_labels = np.asarray(entry['labels'], dtype=np.int64).reshape(-1)
        label_by_index = {int(index): int(label) for index, label in zip(wanted_indices.tolist(), wanted_labels.tolist())}
        with subject_path.open('rb') as handle:
            subject = pickle.load(handle, encoding='latin1')
        data = np.asarray(subject['data'], dtype=np.float32)[:, :32, :]
        for trial_index in sorted({int(index) // 20 for index in label_by_index.keys()}):
            segments = _deap_trial_segments(data[trial_index])
            for segment_offset in range(20):
                original_index = int(trial_index) * 20 + int(segment_offset)
                if original_index not in label_by_index:
                    continue
                yield int(written_index), int(label_by_index[original_index]), segments[segment_offset]
                written_index += 1


@register_streaming_dataset_writer('deap')
def _stream_write_deap_bundle(
    context: DatasetPrepContext,
    prep_root: Path,
    overwrite: bool,
    max_samples: int | None,
) -> Path:
    """One-sample streaming writer for DEAP."""

    normalized_axis = str(context.deap_label_axis).strip().lower()
    axis_index = _DEAP_LABEL_AXIS_TO_INDEX.get(normalized_axis)
    if axis_index is None:
        allowed = ', '.join(sorted(_DEAP_LABEL_AXIS_TO_INDEX))
        raise ValueError(f'Unsupported deap_label_axis {context.deap_label_axis!r}. Allowed: {allowed}.')
    if int(context.deap_num_classes) not in {2, 3}:
        raise ValueError('DEAP deap_num_classes must be 2 or 3 according to Spec/theory/data_prep/data_prep.md.')
    dataset_token = 'deap'
    profile_name, profile_payload = _resolve_prep_profile(dataset_token, context.prep_profile)
    dataset_root = _resolve_deap_root(context.raw_data_root)
    subject_files = sorted(dataset_root.glob('s*.dat'))
    if not subject_files:
        raise FileNotFoundError(f'Could not locate DEAP subject .dat files under {dataset_root}')
    plans, counts = _deap_subject_split_plan(
        subject_files=subject_files,
        axis_index=int(axis_index),
        num_classes=int(context.deap_num_classes),
        seed=int(context.seed),
    )
    label_distribution = _deap_label_distribution_summary(
        plans,
        num_classes=int(context.deap_num_classes),
    )
    out_dir = _prepare_output_dir_for_streaming(
        dataset_token=dataset_token,
        prep_root=Path(prep_root),
        overwrite=bool(overwrite),
        prep_profile_name=profile_name,
    )
    contract = _singlefile_storage_contract(dataset_token)
    split_infos: dict[str, dict[str, Any]] = {}
    for split_name in tqdm(('train', 'test'), desc='deap:splits', leave=False):
        split_infos[split_name] = _write_streamed_structured_split(
            dataset_token=dataset_token,
            split_name=split_name,
            dataset_root=out_dir,
            total_count=int(counts[split_name]),
            input_shape=(384, 32),
            input_dtype=np.dtype(np.float32),
            sample_iter=_iter_deap_split_samples(plans[split_name], axis_index=int(axis_index), num_classes=int(context.deap_num_classes)),
            max_samples=max_samples,
        )
    metadata_extra = {
        'segment_length': 384,
        'num_segments_per_trial': _DEAP_SEGMENTS_PER_TRIAL,
        'channels_used': 32,
        'baseline_removed_samples': 384,
        'baseline_template_length': 128,
        'baseline_removal_rule': 'mean_of_three_128_sample_baseline_chunks_tiled_60_times',
        'label_axis': normalized_axis,
        'num_classes': int(context.deap_num_classes),
        'label_binning_rule': '1-5_to_0_and_6-9_to_1' if int(context.deap_num_classes) == 2 else '1-3_to_0_4-6_to_1_7-9_to_2',
        'split_unit': 'subject',
        'subject_split_random_state': int(context.seed),
        'subject_test_ratio': float(_DEAP_SUBJECT_TEST_RATIO),
        'label_distribution': label_distribution,
    }
    _save_streaming_manifest(
        dataset_token=dataset_token,
        context=context,
        out_dir=out_dir,
        contract=contract,
        split_infos=split_infos,
        psd_axis_kind='temporal',
        profile_payload=profile_payload,
        metadata_extra=metadata_extra,
        max_samples=max_samples,
    )
    return out_dir


def _find_hdf5_file(root: Path, candidates: list[str]) -> Path:
    """Internal helper for ``find hdf5 file`` in the ``preprocessing`` module."""
    for name in candidates:
        path = root / name
        if path.exists():
            return path
    raise FileNotFoundError(f'Could not locate expected HDF5 file under {root}: {candidates}')


def _hdf5_event_label_key(handle: h5py.File) -> str:
    if 'labels' in handle:
        return 'labels'
    if 'y' in handle:
        return 'y'
    raise ValueError('HDF5 event file missing labels/y dataset.')


def _hdf5_event_sample_count(path: Path) -> int:
    with h5py.File(path, 'r') as handle:
        return int(handle[_hdf5_event_label_key(handle)].shape[0])


def _infer_event_unit_shift_from_hdf5_files(paths: list[Path], *, num_units: int) -> tuple[int, str]:
    global_min: int | None = None
    global_max: int | None = None
    for path in paths:
        with h5py.File(path, 'r') as handle:
            units_dataset = handle['spikes']['units']
            labels = handle[_hdf5_event_label_key(handle)]
            for index in tqdm(range(int(labels.shape[0])), desc=f'{path.name}:unit_scan', leave=False):
                units = np.asarray(units_dataset[index], dtype=np.int64).reshape(-1)
                if units.size == 0:
                    continue
                sample_min = int(units.min())
                sample_max = int(units.max())
                global_min = sample_min if global_min is None else min(global_min, sample_min)
                global_max = sample_max if global_max is None else max(global_max, sample_max)
    if global_min is None or global_max is None:
        return 0, 'all_empty_assumed_zero_based'
    if 0 <= global_min and global_max <= int(num_units) - 1:
        return 0, 'zero_based'
    if 1 <= global_min and global_max <= int(num_units):
        return -1, 'one_based_shifted_to_zero_based'
    raise ValueError(
        f'Unsupported event-unit indexing range [{global_min}, {global_max}] for num_units={int(num_units)}. '
        'Expected either 0-based [0, C-1] or 1-based [1, C].'
    )


def _event_sample_to_binary_time_major(
    times: np.ndarray,
    units: np.ndarray,
    *,
    num_units: int,
    num_steps: int,
    dt_s: float,
) -> np.ndarray:
    raster = np.zeros((int(num_steps), int(num_units)), dtype=np.uint8)
    if times.size == 0 or units.size == 0:
        return raster
    bins = np.floor(times.astype(np.float64) / float(dt_s)).astype(np.int64)
    units = units.astype(np.int64, copy=False)
    valid = (bins >= 0) & (bins < int(num_steps)) & (units >= 0) & (units < int(num_units))
    if np.any(valid):
        raster[bins[valid], units[valid]] = 1
    return raster


def _iter_heidelberg_hdf5_samples(paths: list[Path], *, unit_shift: int, num_units: int, num_steps: int, dt_s: float):
    sample_index = 0
    for path in paths:
        with h5py.File(path, 'r') as handle:
            if 'spikes' not in handle:
                raise ValueError(f'HDF5 file missing spikes group: {path}')
            times_dataset = handle['spikes']['times']
            units_dataset = handle['spikes']['units']
            labels = handle[_hdf5_event_label_key(handle)]
            for local_index in range(int(labels.shape[0])):
                times = np.asarray(times_dataset[local_index], dtype=np.float32).reshape(-1)
                units = np.asarray(units_dataset[local_index], dtype=np.int64).reshape(-1)
                if int(unit_shift) != 0:
                    units = units + int(unit_shift)
                label = int(np.asarray(labels[local_index]).reshape(-1)[0])
                yield int(sample_index), int(label), _event_sample_to_binary_time_major(
                    times,
                    units,
                    num_units=int(num_units),
                    num_steps=int(num_steps),
                    dt_s=float(dt_s),
                )
                sample_index += 1

@register_streaming_dataset_writer('shd', 'ssc')
def _stream_write_heidelberg_events_bundle(
    context: DatasetPrepContext,
    prep_root: Path,
    overwrite: bool,
    max_samples: int | None,
) -> Path:
    """Direct-to-disk single-file writer for Heidelberg event datasets."""

    dataset_token = canonicalize_dataset_name(context.dataset_token)
    profile_name, profile_payload = _resolve_prep_profile(dataset_token, context.prep_profile)
    dt_ms = float(context.shd_dt_ms if dataset_token == 'shd' else context.ssc_dt_ms)
    max_time_s = float(context.shd_max_time if dataset_token == 'shd' else context.ssc_max_time)
    dataset_root = _candidate_existing_dirs(context.raw_data_root, [dataset_token, dataset_token.upper(), dataset_token.lower()])
    train_path = _find_hdf5_file(dataset_root, [f'{dataset_token}_train.h5', 'train.h5', f'{dataset_token}-train.h5'])
    test_path = _find_hdf5_file(dataset_root, [f'{dataset_token}_test.h5', 'test.h5', f'{dataset_token}-test.h5'])
    train_paths = [train_path]
    if dataset_token == 'ssc':
        valid_path = _find_hdf5_file(dataset_root, [f'{dataset_token}_valid.h5', 'valid.h5', f'{dataset_token}-valid.h5'])
        train_paths.append(valid_path)
    test_paths = [test_path]
    out_dir = _prepare_output_dir_for_streaming(
        dataset_token=dataset_token,
        prep_root=Path(prep_root),
        overwrite=bool(overwrite),
        prep_profile_name=profile_name,
    )

    num_units = 700
    unit_shift, unit_indexing_rule = _infer_event_unit_shift_from_hdf5_files(train_paths + test_paths, num_units=num_units)
    num_steps = int(round(float(max_time_s) / (float(dt_ms) * 1.0e-3)))
    contract = _singlefile_storage_contract(dataset_token)
    split_infos = {
        'train': _write_streamed_structured_split(
            dataset_token=dataset_token,
            split_name='train',
            dataset_root=out_dir,
            total_count=sum(_hdf5_event_sample_count(path) for path in train_paths),
            input_shape=(int(num_steps), int(num_units)),
            input_dtype=np.dtype(np.uint8),
            sample_iter=_iter_heidelberg_hdf5_samples(
                train_paths,
                unit_shift=int(unit_shift),
                num_units=int(num_units),
                num_steps=int(num_steps),
                dt_s=float(dt_ms) * 1.0e-3,
            ),
            max_samples=max_samples,
        ),
        'test': _write_streamed_structured_split(
            dataset_token=dataset_token,
            split_name='test',
            dataset_root=out_dir,
            total_count=sum(_hdf5_event_sample_count(path) for path in test_paths),
            input_shape=(int(num_steps), int(num_units)),
            input_dtype=np.dtype(np.uint8),
            sample_iter=_iter_heidelberg_hdf5_samples(
                test_paths,
                unit_shift=int(unit_shift),
                num_units=int(num_units),
                num_steps=int(num_steps),
                dt_s=float(dt_ms) * 1.0e-3,
            ),
            max_samples=max_samples,
        ),
    }
    metadata_extra = {
        'dt_ms': float(dt_ms),
        'max_time_s': float(max_time_s),
        'binarization': 'binary_occupancy',
        'num_units': int(num_units),
        'unit_index_shift': int(unit_shift),
        'unit_indexing_rule': unit_indexing_rule,
        'sequence_input_rule': 'time_major_model_input',
        'psd_logical_view_rule': 'metadata_time_axis_and_row_axes',
        'train_source_files': [path.name for path in train_paths],
        'test_source_files': [path.name for path in test_paths],
    }
    _save_streaming_manifest(
        dataset_token=dataset_token,
        context=context,
        out_dir=out_dir,
        contract=contract,
        split_infos=split_infos,
        psd_axis_kind='temporal',
        profile_payload=profile_payload,
        metadata_extra=metadata_extra,
        max_samples=max_samples,
    )
    return out_dir


class _IndexedDatasetView:
    """Small map-style view used when a DVS backend exposes one unsplit dataset."""

    def __init__(self, dataset: Any, indices: np.ndarray):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int) -> Any:
        return self.dataset[int(self.indices[int(index)])]


def _spikingjelly_constructor_parameters(dataset_cls: Any) -> set[str]:
    try:
        signature = inspect.signature(dataset_cls)
    except (TypeError, ValueError):
        return set()
    return set(signature.parameters)


def _make_spikingjelly_frame_dataset(
    dataset_cls: Any,
    dataset_root: Path,
    *,
    train: bool | None,
    num_frames: int,
    split_ratio: float | None = None,
) -> Any:
    """Instantiate a SpikingJelly DVS dataset across old and current API names."""

    try:
        signature = inspect.signature(dataset_cls)
        params = set(signature.parameters)
        accepts_var_keyword = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    except (TypeError, ValueError):
        params = set()
        accepts_var_keyword = False

    def accepts(name: str) -> bool:
        return accepts_var_keyword or name in params

    kwargs: dict[str, Any] = {}
    if train is not None and accepts('train'):
        kwargs['train'] = bool(train)
    if accepts('data_type'):
        kwargs['data_type'] = 'frame'
    elif accepts('use_frame'):
        kwargs['use_frame'] = True
    if accepts('frames_number'):
        kwargs['frames_number'] = int(num_frames)
    elif accepts('frames_num'):
        kwargs['frames_num'] = int(num_frames)
    if accepts('split_by'):
        kwargs['split_by'] = 'number'
    if accepts('duration'):
        kwargs['duration'] = None
    if accepts('normalization'):
        kwargs['normalization'] = None
    if split_ratio is not None and accepts('split_ratio'):
        kwargs['split_ratio'] = float(split_ratio)

    return dataset_cls(str(dataset_root), **kwargs)


def _deterministic_index_split(dataset: Any, *, train_ratio: float, seed: int) -> tuple[_IndexedDatasetView, _IndexedDatasetView]:
    total = int(len(dataset))
    if total < 2:
        raise ValueError('Cannot split a DVS dataset with fewer than 2 samples.')
    indices = np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(indices)
    train_count = int(round(float(train_ratio) * total))
    train_count = max(1, min(total - 1, train_count))
    train_indices = np.sort(indices[:train_count])
    test_indices = np.sort(indices[train_count:])
    return _IndexedDatasetView(dataset, train_indices), _IndexedDatasetView(dataset, test_indices)


def _load_dvs_raw_splits(dataset_token: str, dataset_root: Path, *, num_frames: int, seed: int = 0) -> tuple[Any, Any, tuple[int, int, int], str]:
    """Load DVS frame datasets without split-level dense materialization."""

    failures: list[str] = []

    def load_with_spikingjelly() -> tuple[Any, Any, tuple[int, int, int], str]:
        try:
            if dataset_token == 'n-mnist':
                from spikingjelly.datasets.n_mnist import NMNIST

                train_raw = _make_spikingjelly_frame_dataset(NMNIST, dataset_root, train=True, num_frames=int(num_frames))
                test_raw = _make_spikingjelly_frame_dataset(NMNIST, dataset_root, train=False, num_frames=int(num_frames))
                sensor_shape = (2, 34, 34)
            elif dataset_token == 'cifar10-dvs':
                from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS

                params = _spikingjelly_constructor_parameters(CIFAR10DVS)
                sensor_shape = (2, 128, 128)
                if 'train' in params:
                    train_raw = _make_spikingjelly_frame_dataset(CIFAR10DVS, dataset_root, train=True, num_frames=int(num_frames), split_ratio=0.9)
                    test_raw = _make_spikingjelly_frame_dataset(CIFAR10DVS, dataset_root, train=False, num_frames=int(num_frames), split_ratio=0.9)
                else:
                    full_raw = _make_spikingjelly_frame_dataset(CIFAR10DVS, dataset_root, train=None, num_frames=int(num_frames))
                    train_raw, test_raw = _deterministic_index_split(full_raw, train_ratio=0.9, seed=int(seed))
            elif dataset_token == 'dvs128-gesture':
                from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

                train_raw = _make_spikingjelly_frame_dataset(DVS128Gesture, dataset_root, train=True, num_frames=int(num_frames))
                test_raw = _make_spikingjelly_frame_dataset(DVS128Gesture, dataset_root, train=False, num_frames=int(num_frames))
                sensor_shape = (2, 128, 128)
            else:
                raise ValueError(f'Unsupported DVS dataset: {dataset_token}')
            return train_raw, test_raw, sensor_shape, 'spikingjelly'
        except Exception as exc:
            failures.append(f'spikingjelly: {type(exc).__name__}: {exc}')
            return None, None, (), ''

    def load_with_tonic() -> tuple[Any, Any, tuple[int, int, int], str]:
        try:
            import tonic
            from tonic.transforms import ToFrame

            if dataset_token != 'n-mnist':
                return None, None, (), ''
            sensor_shape = (2, 34, 34)
            tonic_sensor_size = (34, 34, 2)
            transform = ToFrame(sensor_size=tonic_sensor_size, n_event_bins=int(num_frames))
            train_raw = tonic.datasets.NMNIST(save_to=str(dataset_root), train=True, transform=transform)
            test_raw = tonic.datasets.NMNIST(save_to=str(dataset_root), train=False, transform=transform)
            return train_raw, test_raw, sensor_shape, 'tonic'
        except Exception as exc:
            failures.append(f'tonic: {type(exc).__name__}: {exc}')
            return None, None, (), ''

    train_raw, test_raw, sensor_shape, loader_backend = load_with_spikingjelly()
    if (train_raw is None or test_raw is None) and dataset_token == 'n-mnist':
        train_raw, test_raw, sensor_shape, loader_backend = load_with_tonic()
    if train_raw is None or test_raw is None:
        detail = '; '.join(failures) if failures else 'no backend attempted'
        backend_requirement = (
            'SpikingJelly frame dataset support for DVS frame conversion'
            if dataset_token in {'cifar10-dvs', 'dvs128-gesture'}
            else 'SpikingJelly or tonic DVS frame conversion support'
        )
        raise RuntimeError(f'{dataset_token} preprocessing requires {backend_requirement}. Backend failure detail: {detail}')
    return train_raw, test_raw, sensor_shape, loader_backend


def _canonicalize_dvs_frame_sample(frames: Any, *, sensor_shape: tuple[int, int, int], num_frames: int) -> np.ndarray:
    """Return one DVS sample as ``(T,C,H,W)`` unsigned event-count frames."""

    array = np.asarray(frames)
    if array.ndim != 4:
        raise ValueError(f'Expected DVS frame tensor rank 4, got shape {tuple(array.shape)}.')
    expected_t = int(num_frames)
    expected_c, expected_h, expected_w = [int(v) for v in sensor_shape]
    if tuple(int(v) for v in array.shape) == (expected_t, expected_c, expected_h, expected_w):
        canonical = array
    elif tuple(int(v) for v in array.shape) == (expected_t, expected_h, expected_w, expected_c):
        canonical = np.transpose(array, (0, 3, 1, 2))
    else:
        raise ValueError(
            f'Unsupported DVS frame shape {tuple(array.shape)}; expected '
            f'({expected_t}, {expected_c}, {expected_h}, {expected_w}) or '
            f'({expected_t}, {expected_h}, {expected_w}, {expected_c}).'
        )
    rounded = np.rint(canonical)
    if not np.allclose(canonical, rounded):
        raise ValueError('DVS frame counts must be integer-valued after official frame integration.')
    if float(rounded.min(initial=0.0)) < 0.0:
        raise ValueError('DVS frame counts must be non-negative.')
    if int(rounded.max(initial=0.0)) > int(np.iinfo(np.uint16).max):
        raise ValueError('DVS frame count exceeds uint16 storage range; update the official dtype policy before writing.')
    return rounded.astype(np.uint16, copy=False)


def _iter_dvs_stored_samples(raw_dataset: Any, *, sensor_shape: tuple[int, int, int], num_frames: int):
    for index in range(len(raw_dataset)):
        frames, label = raw_dataset[index]
        yield int(index), int(label), _canonicalize_dvs_frame_sample(frames, sensor_shape=sensor_shape, num_frames=int(num_frames))


@register_streaming_dataset_writer('n-mnist', 'cifar10-dvs', 'dvs128-gesture')
def _stream_write_dvs_bundle(
    context: DatasetPrepContext,
    prep_root: Path,
    overwrite: bool,
    max_samples: int | None,
) -> Path:
    """One-sample streaming writer for DVS frame-count datasets."""

    dataset_token = canonicalize_dataset_name(context.dataset_token)
    profile_name, profile_payload = _resolve_prep_profile(dataset_token, context.prep_profile)
    num_frames = int(profile_payload.get('num_frames', 10 if dataset_token == 'n-mnist' else 20))
    dataset_root = context.raw_data_root / dataset_token
    train_raw, test_raw, sensor_shape, loader_backend = _load_dvs_raw_splits(dataset_token, dataset_root, num_frames=int(num_frames), seed=int(context.seed))
    out_dir = _prepare_output_dir_for_streaming(
        dataset_token=dataset_token,
        prep_root=Path(prep_root),
        overwrite=bool(overwrite),
        prep_profile_name=profile_name,
    )
    contract = _singlefile_storage_contract(dataset_token)
    split_infos: dict[str, dict[str, Any]] = {}
    for split_name, raw_dataset in tqdm((('train', train_raw), ('test', test_raw)), desc=f'{dataset_token}:splits', leave=False):
        split_infos[split_name] = _write_streamed_structured_split(
            dataset_token=dataset_token,
            split_name=split_name,
            dataset_root=out_dir,
            total_count=len(raw_dataset),
            input_shape=(int(num_frames), *[int(v) for v in sensor_shape]),
            input_dtype=np.dtype(np.uint16),
            sample_iter=_iter_dvs_stored_samples(raw_dataset, sensor_shape=sensor_shape, num_frames=int(num_frames)),
            max_samples=max_samples,
        )
    metadata_extra = {
        'num_frames': int(num_frames),
        'split_by': 'number',
        'normalization': 'None',
        'sensor_shape': [int(v) for v in sensor_shape],
        'download': bool(context.download),
        'loader_backend': loader_backend,
        'loader_backend_policy': 'spikingjelly_preferred_with_tonic_fallback_for_n_mnist_only',
        'sequence_input_rule': 'flatten_input_identity',
        'original_shape': [int(num_frames), *[int(v) for v in sensor_shape]],
    }
    if dataset_token == 'cifar10-dvs':
        metadata_extra['split_ratio'] = 0.9
    _save_streaming_manifest(
        dataset_token=dataset_token,
        context=context,
        out_dir=out_dir,
        contract=contract,
        split_infos=split_infos,
        psd_axis_kind='image_temporal',
        profile_payload=profile_payload,
        metadata_extra=metadata_extra,
        max_samples=max_samples,
    )
    return out_dir


def prepare_dataset_bundle(
    dataset_name: str,
    *,
    raw_data_root: Path | str,
    prep_root: Path | str,
    seed: int = 0,
    download: bool = False,
    overwrite: bool = False,
    deap_label_axis: str = 'valence',
    deap_num_classes: int = 2,
    shd_dt_ms: float = 1.0,
    shd_max_time: float = 1.2,
    ssc_dt_ms: float = 1.0,
    ssc_max_time: float = 1.0,
    max_samples: int | None = None,
    prep_profile: str | None = None,
) -> Path:
    """Prepare dataset bundle."""
    dataset_token = canonicalize_dataset_name(dataset_name)
    prep_profile_name, prep_profile_payload = _resolve_prep_profile(dataset_token, prep_profile)
    raw_data_root = Path(raw_data_root).expanduser().resolve()
    prep_root = Path(prep_root).expanduser().resolve()
    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples <= 0:
            raise ValueError('max_samples must be positive when provided.')
    if 'shd_dt_ms' in prep_profile_payload:
        shd_dt_ms = float(prep_profile_payload['shd_dt_ms'])
    if 'shd_max_time' in prep_profile_payload:
        shd_max_time = float(prep_profile_payload['shd_max_time'])
    context = DatasetPrepContext(
        dataset_token=dataset_token,
        raw_data_root=raw_data_root,
        seed=int(seed),
        download=bool(download),
        deap_label_axis=str(deap_label_axis),
        deap_num_classes=int(deap_num_classes),
        shd_dt_ms=float(shd_dt_ms),
        shd_max_time=float(shd_max_time),
        ssc_dt_ms=float(ssc_dt_ms),
        ssc_max_time=float(ssc_max_time),
        prep_profile=prep_profile_name,
    )
    streaming_writer = _resolve_streaming_dataset_writer(dataset_token)
    if streaming_writer is not None:
        return streaming_writer(
            context,
            prep_root,
            bool(overwrite),
            max_samples,
        )

    raise RuntimeError(
        f'No Spec-compliant one-sample streaming writer is registered for {dataset_token!r}. '
        'The official data_prep path cannot fall back to split-level in-memory materialization.'
    )


__all__ = [
    'prepare_dataset_bundle',
]
