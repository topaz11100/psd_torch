"""Prepared-bundle dataset registry for the official experiment scope."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import os
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.base import (
    DatasetBundle,
    PreparedDatasetProtocol,
    PreparedStructuredSplitDataset,
)
from src.data.specs import available_dataset_tokens as _available_dataset_tokens
from src.data.specs import canonicalize_dataset_name as _canonicalize_dataset_name
from src.data.specs import get_dataset_spec
from src.data.storage import SINGLE_STRUCTURED_NPY_STORAGE_FORMAT, load_single_structured_split
from src.util.config import load_json
from src.util.random import build_torch_generator, seed_dataloader_worker


_EMITTED_LOADER_POLICY_WARNINGS: set[str] = set()
_STATIC_IMAGE_REPEAT_T = 4


@dataclass(frozen=True)
class ViewDataset(Dataset):
    """Alternate view wrapper preserving canonical labels and sample indices."""

    parent: PreparedDatasetProtocol
    view_name: str
    _resolved_parent: PreparedDatasetProtocol | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        """Resolve one efficient alternate-view dataset when supported."""
        resolved = self.parent.with_primary_view(self.view_name) if hasattr(self.parent, 'with_primary_view') else None
        object.__setattr__(self, '_resolved_parent', resolved)

    @property
    def _runtime_parent(self) -> PreparedDatasetProtocol:
        """Return the dataset actually serving items for this view wrapper."""
        return self.parent if self._resolved_parent is None else self._resolved_parent

    def __len__(self) -> int:
        """Return the number of items available from this object."""
        return len(self._runtime_parent)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return one item from this object by index."""
        runtime_parent = self._runtime_parent
        if self._resolved_parent is not None:
            return runtime_parent[int(index)]
        tensor = runtime_parent.view_tensor(self.view_name)[int(index)]
        target = int(runtime_parent.labels[int(index)].item())
        return tensor, target

    @property
    def labels(self) -> torch.Tensor:
        """Return the labels."""
        return self._runtime_parent.labels

    @property
    def targets(self) -> torch.Tensor:
        """Return the targets."""
        return self._runtime_parent.targets

    @property
    def sample_indices(self) -> list[int]:
        """Return the indices."""
        return list(self._runtime_parent.sample_indices)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the metadata."""
        return dict(self._runtime_parent.metadata)

    @property
    def data_views(self) -> dict[str, Any]:
        """Return the data views."""
        return self._runtime_parent.data_views


def available_dataset_tokens() -> tuple[str, ...]:
    """Return available dataset tokens."""
    return _available_dataset_tokens()


def canonicalize_dataset_name(name: str) -> str:
    """Canonicalize dataset name."""
    return _canonicalize_dataset_name(name)


def _dataset_root(prep_root: Path, dataset_name: str) -> Path:
    """Internal helper for ``dataset root`` in the ``registry`` module."""
    return prep_root / canonicalize_dataset_name(dataset_name)


def _sequence_shape_from_singlefile_manifest(manifest: Mapping[str, Any], dataset_name: str) -> tuple[int, int]:
    """Resolve runtime ``(sequence_length, input_dim)`` for the structured single-file format."""

    seq = manifest.get('sequence_length')
    inp = manifest.get('input_dim')
    if seq is not None and inp is not None:
        return int(seq), int(inp)
    stored_shape = tuple(int(v) for v in manifest.get('stored_shape', ()))
    canonical = canonicalize_dataset_name(dataset_name)
    if canonical in {'s-mnist', 'ps-mnist', 's-cifar10', 'deap'}:
        if len(stored_shape) != 2:
            raise ValueError(f'{canonical} stored_shape must be rank 2 in the structured manifest, got {stored_shape}.')
        return int(stored_shape[0]), int(stored_shape[1])
    if canonical in {'uci-har', 'shd', 'ssc'}:
        if len(stored_shape) != 2:
            raise ValueError(f'{canonical} stored_shape must be rank 2 in the structured manifest, got {stored_shape}.')
        return int(stored_shape[0]), int(stored_shape[1])
    if canonical in {'mnist', 'cifar-10', 'cifar-100'}:
        if len(stored_shape) != 4:
            raise ValueError(f'{canonical} stored_shape must be rank 4 as (T,C,H,W) in the structured manifest, got {stored_shape}.')
        return int(stored_shape[0]), int(stored_shape[1]) * int(stored_shape[2]) * int(stored_shape[3])
    if canonical in {'n-mnist', 'cifar10-dvs', 'dvs128-gesture'}:
        if len(stored_shape) != 4:
            raise ValueError(f'{canonical} stored_shape must be rank 4 in the structured manifest, got {stored_shape}.')
        return int(stored_shape[0]), int(stored_shape[1]) * int(stored_shape[2]) * int(stored_shape[3])
    raise ValueError(f'Unsupported structured-manifest dataset {dataset_name!r}.')


def dataset_for_view(dataset: PreparedDatasetProtocol, view_name: str) -> PreparedDatasetProtocol | ViewDataset:
    """Expose one alternate saved view as a dataset without changing canonical order."""

    if hasattr(dataset, 'with_primary_view'):
        return dataset.with_primary_view(view_name)
    return ViewDataset(dataset, view_name)


def _select_static_image_training_view(manifest: Mapping[str, Any], *, model_family: str, default_view: str) -> str:
    """Select the runtime training view for static-image CNN vs dense model families."""

    psd_axis_kind = str(manifest.get('psd_axis_kind', ''))
    available = tuple(str(v) for v in manifest.get('available_views', ()))
    if psd_axis_kind != 'static_repeat':
        return str(default_view)
    family = str(model_family)
    if family in {'cnn_lif', 'cnn_rf', 'cnn'}:
        for candidate in (manifest.get('cnn_training_view_name'), 'model_input_cnn', 'cnn_input', 'model_input'):
            if isinstance(candidate, str) and (not available or candidate in available):
                return candidate
        return str(default_view)
    for candidate in (manifest.get('flatten_training_view_name'), 'model_input_flatten', 'sequence_input', 'flatten_input'):
        if isinstance(candidate, str) and (not available or candidate in available):
            return candidate
    return str(default_view)


def select_training_view_for_model(bundle: DatasetBundle, *, model_family: str) -> DatasetBundle:
    """Return a bundle whose primary split view matches the model family.

    Static image datasets may physically store both CNN-shaped and flattened
    split payloads. CNN families consume the CNN view; dense LIF/RF families
    consume the flattened time-major view. Other datasets keep the manifest
    default training view unchanged.
    """

    manifest = load_json(bundle.manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f'Prepared manifest must be a JSON object: {bundle.manifest_path}')
    selected = _select_static_image_training_view(manifest, model_family=str(model_family), default_view=bundle.training_view_name)
    if selected == bundle.training_view_name:
        return bundle
    return replace(
        bundle,
        train_dataset=dataset_for_view(bundle.train_dataset, selected),
        test_dataset=dataset_for_view(bundle.test_dataset, selected),
        training_view_name=selected,
    )


def resolve_dataset_bundle(
    dataset_name: str,
    *,
    prep_root: Path | str,
    max_samples: int | None = None,
) -> DatasetBundle:
    """Load one prepared dataset bundle from ``data_prep`` outputs."""

    canonical = canonicalize_dataset_name(dataset_name)
    spec = get_dataset_spec(canonical)
    prep_root = Path(prep_root).expanduser().resolve()
    dataset_root = _dataset_root(prep_root, canonical)
    manifest_path = dataset_root / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(
            f'Prepared dataset manifest is missing: {manifest_path}. Run python -m src.data_prep first.'
        )

    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f'manifest.json must contain a JSON object: {manifest_path}')
    if str(manifest.get('dataset_name', canonical)) != canonical:
        raise ValueError(f'Manifest dataset_name mismatch for {canonical}: {manifest.get("dataset_name")!r}')
    if manifest.get('split_internal_order_preserved') is not True:
        raise ValueError('Prepared manifest must declare split_internal_order_preserved=true.')


    storage_format = str(manifest.get('storage_format', ''))
    if storage_format != SINGLE_STRUCTURED_NPY_STORAGE_FORMAT:
        raise ValueError(
            'Prepared manifest storage_format must be '
            f'{SINGLE_STRUCTURED_NPY_STORAGE_FORMAT!r} under the locked project specification; '
            f'got {storage_format!r}. Re-run python -m src.data_prep to regenerate the official structured bundle.'
        )
    files_entry = manifest.get('files', {'train': 'train.npy', 'test': 'test.npy'})
    if not isinstance(files_entry, dict):
        raise ValueError('Prepared manifest files entry must be a JSON object.')

    resolved_paths: dict[str, Path] = {}
    for split_name in ('train', 'test'):
        relpath = files_entry.get(split_name, f'{split_name}.npy')
        if not isinstance(relpath, str):
            raise ValueError(f'Structured manifest files.{split_name} must be a single relative .npy path.')
        path = (dataset_root / relpath).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f'Prepared manifest references missing {split_name} structured payload: {path}. '
                f'Manifest path: {manifest_path}.'
            )
        resolved_paths[split_name] = path

    files_by_view_raw = manifest.get('files_by_view', {})
    files_by_view = files_by_view_raw if isinstance(files_by_view_raw, Mapping) else {}
    view_paths: dict[str, dict[str, Path]] = {}
    for view_name, split_map in files_by_view.items():
        if not isinstance(split_map, Mapping):
            continue
        resolved_split_paths: dict[str, Path] = {}
        for split_name in ('train', 'test'):
            relpath = split_map.get(split_name)
            if not isinstance(relpath, str):
                continue
            path = (dataset_root / relpath).resolve()
            if not path.exists():
                raise FileNotFoundError(
                    f'Prepared manifest references missing {view_name}.{split_name} payload: {path}. '
                    f'Manifest path: {manifest_path}.'
                )
            resolved_split_paths[split_name] = path
        if set(resolved_split_paths) == {'train', 'test'}:
            view_paths[str(view_name)] = resolved_split_paths

    training_view_name = str(manifest.get('training_view_name', spec.training_view_name))
    psd_view_name = str(manifest.get('psd_view_name', spec.psd_view_name))
    psd_axis_kind = str(manifest.get('psd_axis_kind', spec.psd_axis_kind))
    available_views = tuple(str(v) for v in manifest.get('available_views', []))
    missing_declared = {training_view_name, psd_view_name}.difference(available_views or {training_view_name, psd_view_name})
    if missing_declared:
        raise ValueError(
            f'Structured manifest is missing declared runtime view(s) {sorted(missing_declared)}. '
            f'Available views: {sorted(available_views)}.'
        )
    train_records = load_single_structured_split(resolved_paths['train'], mmap_mode='r')
    test_records = load_single_structured_split(resolved_paths['test'], mmap_mode='r')
    train_records_by_view = {
        view_name: load_single_structured_split(paths['train'], mmap_mode='r')
        for view_name, paths in view_paths.items()
        if paths['train'] != resolved_paths['train']
    }
    test_records_by_view = {
        view_name: load_single_structured_split(paths['test'], mmap_mode='r')
        for view_name, paths in view_paths.items()
        if paths['test'] != resolved_paths['test']
    }
    shared_metadata = dict(manifest)
    train_dataset = PreparedStructuredSplitDataset(
        dataset_name=canonical,
        split_name='train',
        records=train_records,
        metadata={**shared_metadata, 'split': 'train', 'dataset_token': canonical},
        primary_view_name=training_view_name,
        max_samples=max_samples,
        records_by_view=train_records_by_view,
    )
    test_dataset = PreparedStructuredSplitDataset(
        dataset_name=canonical,
        split_name='test',
        records=test_records,
        metadata={**shared_metadata, 'split': 'test', 'dataset_token': canonical},
        primary_view_name=training_view_name,
        max_samples=max_samples,
        records_by_view=test_records_by_view,
    )
    sequence_length, input_dim = _sequence_shape_from_singlefile_manifest(manifest, canonical)
    class_label_values: set[int] = set()
    for records in (train_records, test_records):
        class_label_values.update(int(value) for value in np.unique(records['label']).tolist())
    class_labels = tuple(sorted(class_label_values))
    default_hidden_sizes = tuple(int(v) for v in manifest.get('default_hidden_sizes', list(spec.default_hidden_sizes)))
    return DatasetBundle(
        dataset_name=canonical,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        input_dim=int(input_dim),
        sequence_length=int(sequence_length),
        num_classes=int(len(class_labels)),
        class_labels=class_labels,
        default_hidden_sizes=default_hidden_sizes,
        prep_root=prep_root,
        manifest_path=manifest_path,
        psd_axis_kind=psd_axis_kind,
        training_view_name=training_view_name,
        psd_view_name=psd_view_name,
    )


def _object_nbytes(value: Any) -> int:
    """Best-effort byte estimate for one dataset sample tree."""

    if isinstance(value, torch.Tensor):
        return int(value.numel()) * int(value.element_size())
    if isinstance(value, Mapping):
        return sum(_object_nbytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_object_nbytes(v) for v in value)
    nbytes = getattr(value, 'nbytes', None)
    if nbytes is not None:
        try:
            return int(nbytes)
        except Exception:
            return 0
    return 0


def _estimate_sample_nbytes(dataset: Dataset) -> int | None:
    """Return one best-effort sample-size estimate for loader policy guards."""

    try:
        sample = dataset[0]
    except Exception:
        return None
    estimate = _object_nbytes(sample)
    return None if estimate <= 0 else int(estimate)


def _available_shm_bytes() -> int | None:
    """Return available ``/dev/shm`` bytes when the platform exposes it."""

    shm_path = '/dev/shm'
    if not os.path.exists(shm_path):
        return None
    try:
        stats = os.statvfs(shm_path)
    except OSError:
        return None
    return int(stats.f_bavail) * int(stats.f_frsize)


def _resolve_loader_runtime_policy(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> dict[str, Any]:
    """Resolve one shm-aware DataLoader runtime policy."""

    requested_workers = max(0, int(num_workers))
    actual_workers = requested_workers
    sample_nbytes = _estimate_sample_nbytes(dataset)
    batch_nbytes = None if sample_nbytes is None else int(sample_nbytes) * max(1, int(batch_size))
    shm_available = _available_shm_bytes()
    shm_budget = None if shm_available is None else max(0, int(float(shm_available) * 0.5))
    prefetch_factor = 2 if actual_workers > 0 else None
    notes: list[str] = []

    if actual_workers > 0 and batch_nbytes is not None and shm_budget is not None and batch_nbytes > 0:
        required = int(batch_nbytes) * int(actual_workers) * int(prefetch_factor or 1)
        if required > shm_budget:
            prefetch_factor = 1
            notes.append('prefetch_factor_auto_reduced')
            required = int(batch_nbytes) * int(actual_workers) * int(prefetch_factor)
        if required > shm_budget:
            allowed_workers = int(shm_budget // max(1, int(batch_nbytes) * int(prefetch_factor or 1)))
            actual_workers = max(0, min(actual_workers, allowed_workers))
            if actual_workers < requested_workers:
                notes.append('num_workers_auto_reduced')
        if actual_workers <= 0:
            actual_workers = 0
            prefetch_factor = None
            notes.append('worker_pool_disabled_due_to_shm_budget')

    persistent_workers = bool(actual_workers > 0)
    if actual_workers == 0:
        prefetch_factor = None

    return {
        'requested_num_workers': int(requested_workers),
        'actual_num_workers': int(actual_workers),
        'pin_memory': bool(pin_memory),
        'persistent_workers': bool(persistent_workers),
        'prefetch_factor': None if prefetch_factor is None else int(prefetch_factor),
        'estimated_sample_nbytes': None if sample_nbytes is None else int(sample_nbytes),
        'estimated_batch_nbytes': None if batch_nbytes is None else int(batch_nbytes),
        'shm_available_nbytes': None if shm_available is None else int(shm_available),
        'shm_budget_nbytes': None if shm_budget is None else int(shm_budget),
        'policy_notes': list(dict.fromkeys(str(v) for v in notes)),
    }


def loader_runtime_policy(loader: DataLoader) -> dict[str, Any]:
    """Return the runtime loader policy recorded on one project DataLoader."""

    policy = getattr(loader, '_psd_runtime_policy', None)
    return dict(policy) if isinstance(policy, dict) else {}


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool = True,
    drop_last: bool = False,
    generator: torch.Generator | None = None,
    seed: int | None = None,
) -> DataLoader:
    """Build one project-standard ``DataLoader``.

    The helper keeps worker seeding, persistent-worker policy, and explicit
    generator construction in one place so the official entrypoints do not drift
    apart over time. It also applies a conservative ``/dev/shm`` guard so large
    batches auto-downgrade worker/prefetch settings instead of crashing workers
    with a bus error.
    """

    resolved_policy = _resolve_loader_runtime_policy(
        dataset,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
    )
    worker_count = int(resolved_policy['actual_num_workers'])
    resolved_generator = generator if generator is not None else build_torch_generator(seed)
    loader_kwargs: dict[str, Any] = {
        'dataset': dataset,
        'batch_size': int(batch_size),
        'shuffle': bool(shuffle),
        'num_workers': worker_count,
        'pin_memory': bool(pin_memory),
        'drop_last': bool(drop_last),
        'generator': resolved_generator,
        'worker_init_fn': seed_dataloader_worker,
    }
    if worker_count > 0:
        loader_kwargs['persistent_workers'] = bool(resolved_policy['persistent_workers'])
        loader_kwargs['prefetch_factor'] = int(resolved_policy['prefetch_factor'])
    notes = list(resolved_policy.get('policy_notes', []))
    requested_workers = int(resolved_policy.get('requested_num_workers', worker_count))
    if notes and worker_count != requested_workers:
        warning_token = (
            f'{requested_workers}|{worker_count}|{resolved_policy.get("prefetch_factor")}|{resolved_policy.get("estimated_batch_nbytes")}|{tuple(notes)}'
        )
        if warning_token not in _EMITTED_LOADER_POLICY_WARNINGS:
            _EMITTED_LOADER_POLICY_WARNINGS.add(warning_token)
            warnings.warn(
                'DataLoader worker policy was auto-reduced for shm safety: '
                f'requested num_workers={requested_workers}, actual num_workers={worker_count}, '
                f'prefetch_factor={resolved_policy.get("prefetch_factor")}, notes={notes}.',
                RuntimeWarning,
                stacklevel=2,
            )
    loader = DataLoader(**loader_kwargs)
    setattr(loader, '_psd_runtime_policy', dict(resolved_policy))
    return loader


__all__ = [
    'DatasetBundle',
    'ViewDataset',
    'available_dataset_tokens',
    'canonicalize_dataset_name',
    'dataset_for_view',
    'loader_runtime_policy',
    'make_loader',
    'resolve_dataset_bundle',
    'select_training_view_for_model',
]
