"""Prepared-dataset typing helpers and bundle contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import torch
from torch.utils.data import Dataset

_STATIC_IMAGE_REPEAT_T = 4


@dataclass(frozen=True)
class DatasetMetadata:
    """Common dataset metadata shared by downstream experiments."""

    dataset_name: str
    input_dim: int
    sequence_length: int
    num_classes: int
    class_labels: tuple[int, ...]
    default_hidden_sizes: tuple[int, ...]
    psd_axis_kind: str
    training_view_name: str = 'model_input'
    psd_view_name: str = 'psd_input'


class SequenceDataset(Protocol):
    """Protocol for datasets returning ``(sequence, label)`` pairs."""

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        ...


class PreparedViewAccessor:
    """Lazy split-view accessor that preserves mmap-backed sample access.

    The prepared-bundle loader contract forbids implicit full-split RAM
    materialization. This accessor therefore reconstructs one sample view on
    demand and only materializes the full split when ``materialize()`` is
    called explicitly by non-loader code.
    """

    def __init__(self, parent: 'PreparedStructuredSplitDataset', view_name: str) -> None:
        self.parent = parent
        self.view_name = str(view_name)

    def __len__(self) -> int:
        return len(self.parent)

    def _normalize_scalar_index(self, index: int) -> int:
        resolved = int(index)
        if resolved < 0:
            resolved += len(self)
        if resolved < 0 or resolved >= len(self):
            raise IndexError(f'Prepared view index out of range: {index}')
        return resolved

    def _empty_like_batch(self) -> torch.Tensor:
        first = self.parent._sample_view(0, self.view_name)
        return first.new_empty((0, *tuple(first.shape)))

    def _stack_indices(self, indices: list[int]) -> torch.Tensor:
        if not indices:
            return self._empty_like_batch()
        return torch.stack([self.parent._sample_view(index, self.view_name) for index in indices], dim=0)

    def __getitem__(self, index: Any) -> torch.Tensor:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return self._stack_indices(list(range(start, stop, step)))
        if torch.is_tensor(index):
            if index.ndim == 0:
                return self.parent._sample_view(self._normalize_scalar_index(int(index.item())), self.view_name)
            return self._stack_indices([self._normalize_scalar_index(int(v)) for v in index.reshape(-1).tolist()])
        if isinstance(index, np.ndarray):
            if index.ndim == 0:
                return self.parent._sample_view(self._normalize_scalar_index(int(index.item())), self.view_name)
            return self._stack_indices([self._normalize_scalar_index(int(v)) for v in index.reshape(-1).tolist()])
        if isinstance(index, (list, tuple)):
            return self._stack_indices([self._normalize_scalar_index(int(v)) for v in index])
        return self.parent._sample_view(self._normalize_scalar_index(int(index)), self.view_name)

    @property
    def shape(self) -> tuple[int, ...]:
        sample_shape = tuple(self.parent._sample_view(0, self.view_name).shape)
        return (len(self), *sample_shape)

    @property
    def dtype(self) -> torch.dtype:
        return self.parent._sample_view(0, self.view_name).dtype

    def materialize(self) -> torch.Tensor:
        """Explicitly stack the full split for non-loader callers."""

        cached = self.parent._materialized_view_cache.get(self.view_name)
        if cached is not None:
            return cached
        stacked = self._stack_indices(list(range(len(self))))
        self.parent._materialized_view_cache[self.view_name] = stacked
        return stacked

    def __repr__(self) -> str:
        return (
            f'PreparedViewAccessor(dataset={self.parent.dataset_name!r}, '
            f'split={self.parent.split_name!r}, view={self.view_name!r}, shape={self.shape!r})'
        )


class PreparedDatasetProtocol(Protocol):
    """Common runtime surface shared by prepared split dataset wrappers."""

    dataset_name: str
    split_name: str
    labels: torch.Tensor
    targets: torch.Tensor
    sample_indices: list[int]
    metadata: dict[str, Any]
    primary_view_name: str

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        ...

    def available_views(self) -> tuple[str, ...]:
        ...

    def view_tensor(self, view_name: str) -> PreparedViewAccessor:
        ...

    def with_primary_view(self, view_name: str) -> 'PreparedDatasetProtocol':
        ...


def canonicalize_model_input_batch(batch: Any, *, sequence_length: int, input_dim: int) -> torch.Tensor:
    """Normalize one training batch into the canonical ``(B,T,C)`` model input layout."""

    tensor = torch.as_tensor(batch)
    expected_time = int(sequence_length)
    expected_channels = int(input_dim)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim == 3:
        if int(tensor.shape[1]) == expected_time and int(tensor.shape[2]) == expected_channels:
            return tensor
        if int(tensor.shape[1]) == expected_channels and int(tensor.shape[2]) == expected_time:
            return tensor.transpose(1, 2).contiguous()
        raise ValueError(
            f'Could not canonicalize rank-3 batch with shape {tuple(tensor.shape)} to (B,{expected_time},{expected_channels}).'
        )
    if tensor.ndim == 4:
        batch = int(tensor.shape[0])
        channels = int(tensor.shape[1])
        flattened_spatial = tensor.reshape(batch, channels, -1).transpose(1, 2).contiguous()
        if int(flattened_spatial.shape[1]) == expected_time and int(flattened_spatial.shape[2]) == expected_channels:
            return flattened_spatial
        flattened_image = tensor.reshape(batch, -1).contiguous()
        raise ValueError(
            f'Could not canonicalize rank-4 batch with shape {tuple(tensor.shape)} to (B,{expected_time},{expected_channels}) without inventing a time axis; '
            f'prepared static images must be stored as rank-5 batches (B,T,C,H,W). '
            f'flattened spatial shape is {tuple(flattened_spatial.shape)}, flattened image shape is {tuple(flattened_image.shape)}.'
        )
    if tensor.ndim == 5:
        flattened = tensor.reshape(int(tensor.shape[0]), int(tensor.shape[1]), -1).contiguous()
        if int(flattened.shape[1]) == expected_time and int(flattened.shape[2]) == expected_channels:
            return flattened
        if int(flattened.shape[1]) == expected_channels and int(flattened.shape[2]) == expected_time:
            return flattened.transpose(1, 2).contiguous()
        raise ValueError(
            f'Could not canonicalize rank-5 batch with shape {tuple(tensor.shape)} to (B,{expected_time},{expected_channels}); '
            f'flattened shape is {tuple(flattened.shape)}.'
        )
    raise ValueError(f'Unsupported batch rank {tensor.ndim}; expected rank 2-5 input, got shape {tuple(tensor.shape)}.')


def _reconstruct_structured_view(
    stored_input: np.ndarray,
    *,
    dataset_name: str,
    view_name: str,
    sequence_input_rule: str | None,
) -> torch.Tensor:
    """Reconstruct one runtime view from the canonical stored single-file input."""

    dataset = str(dataset_name)
    view = str(view_name)
    stored = np.array(stored_input, copy=True)
    tensor = torch.as_tensor(stored)

    if dataset in {'s-mnist', 'ps-mnist', 's-cifar10', 'deap'}:
        if view == 'model_input':
            return tensor.to(dtype=torch.float32)
        if view in {'psd_input', 'model_input_psd_view'}:
            return tensor.transpose(0, 1).contiguous().to(dtype=torch.float32)
        raise KeyError(f'View {view!r} is unsupported for dataset {dataset!r}.')

    if dataset in {'uci-har', 'shd', 'ssc'}:
        if view == 'model_input':
            return tensor.to(dtype=torch.float32)
        if view in {'psd_input', 'model_input_psd_view', 'sequence_input'}:
            return tensor.transpose(0, 1).contiguous().to(dtype=torch.float32)
        raise KeyError(f'View {view!r} is unsupported for dataset {dataset!r}.')

    if dataset in {'mnist', 'cifar-10', 'cifar-100'}:
        value = tensor.to(dtype=torch.float32)
        if value.ndim == 4:
            repeated = value
            first_frame = repeated[0].contiguous()
            flatten_channel_major = first_frame.reshape(int(first_frame.shape[0]), -1).contiguous()
            sequence_flatten = repeated.reshape(int(repeated.shape[0]), -1).contiguous()
            if view == 'original_input':
                return first_frame
            if view in {'model_input', 'model_input_cnn', 'cnn_input', 'psd_input', 'image_psd_view'}:
                return repeated.contiguous()
            if view in {'sequence_input', 'model_input_flatten'}:
                return sequence_flatten
            if view == 'flatten_input':
                return flatten_channel_major
            raise KeyError(f'View {view!r} is unsupported for dataset {dataset!r}.')
        if value.ndim == 2:
            if view in {'model_input', 'model_input_flatten', 'flatten_input', 'sequence_input'}:
                return value.contiguous()
            raise ValueError(
                f'{dataset} prepared flat samples have shape (T,F) and can only serve flatten views; '
                f'view {view!r} requested shape {tuple(value.shape)}.'
            )
        raise ValueError(f'{dataset} prepared samples must have shape (T,C,H,W) or (T,F), got {tuple(value.shape)}.')

    if dataset in {'n-mnist', 'cifar10-dvs', 'dvs128-gesture'}:
        original = tensor.to(dtype=torch.float32)
        flatten = original.reshape(int(original.shape[0]), -1)
        if view in {'model_input', 'original_input'}:
            return original
        if view in {'flatten_input', 'sequence_input', 'model_input_flatten'}:
            return flatten
        if view in {'psd_input', 'event_frame_psd_view'}:
            return flatten.transpose(0, 1).contiguous()
        raise KeyError(f'View {view!r} is unsupported for dataset {dataset!r}.')

    if sequence_input_rule == 'model_input_transpose':
        if view in {'model_input', 'psd_input'}:
            return tensor.to(dtype=torch.float32)
        if view == 'sequence_input':
            return tensor.transpose(0, 1).contiguous().to(dtype=torch.float32)
    raise KeyError(f'Could not reconstruct view {view!r} for dataset {dataset!r}.')


class PreparedStructuredSplitDataset(Dataset[tuple[torch.Tensor, int]]):
    """Lazy prepared split backed by one single structured ``.npy`` memmap payload."""

    def __init__(
        self,
        *,
        dataset_name: str,
        split_name: str,
        records: np.ndarray,
        metadata: dict[str, Any],
        primary_view_name: str,
        max_samples: int | None = None,
        records_by_view: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        self.dataset_name = str(dataset_name)
        self.split_name = str(split_name)
        self.metadata = dict(metadata)
        self.primary_view_name = str(primary_view_name)
        self._records = records if max_samples is None else records[: int(max_samples)]
        self._num_samples = int(self._records.shape[0])
        raw_records_by_view = {} if records_by_view is None else dict(records_by_view)
        self._records_by_view: dict[str, np.ndarray] = {}
        for view_name, view_records in raw_records_by_view.items():
            sliced = view_records if max_samples is None else view_records[: int(max_samples)]
            self._records_by_view[str(view_name)] = sliced
        for alias in ('model_input', 'model_input_cnn', 'cnn_input', 'psd_input', 'image_psd_view', 'original_input'):
            self._records_by_view.setdefault(alias, self._records)
        self._num_samples = int(self._records.shape[0])
        if self._num_samples <= 0:
            raise ValueError('PreparedStructuredSplitDataset cannot represent an empty split.')
        label_array = np.array(self._records['label'], copy=True)
        sample_index_array = np.array(self._records['sample_index'], copy=True)
        self.labels = torch.as_tensor(label_array, dtype=torch.long).reshape(-1)
        self.targets = self.labels
        self.sample_indices = [int(v) for v in sample_index_array.reshape(-1).tolist()]
        for view_name, view_records in self._records_by_view.items():
            if int(view_records.shape[0]) != self._num_samples:
                raise ValueError(
                    f'Prepared view {view_name!r} length mismatch for {self.dataset_name}:{self.split_name}: '
                    f'{int(view_records.shape[0])} vs {self._num_samples}.'
                )
            if 'label' in view_records.dtype.fields and not np.array_equal(np.array(view_records['label']), label_array):
                raise ValueError(f'Prepared view {view_name!r} label array does not match the primary split payload.')
            if 'sample_index' in view_records.dtype.fields and not np.array_equal(np.array(view_records['sample_index']), sample_index_array):
                raise ValueError(f'Prepared view {view_name!r} sample_index array does not match the primary split payload.')
        self._sequence_input_rule = None if self.metadata.get('sequence_input_rule') is None else str(self.metadata.get('sequence_input_rule'))
        available = self.available_views()
        if self.primary_view_name not in available:
            raise KeyError(f'Primary view {self.primary_view_name!r} missing. Available: {", ".join(available)}.')
        self._view_accessor_cache: dict[str, PreparedViewAccessor] = {}
        self._materialized_view_cache: dict[str, torch.Tensor] = {}

    @property
    def data_views(self) -> dict[str, PreparedViewAccessor]:
        return {name: self.view_tensor(name) for name in self.available_views()}

    def __len__(self) -> int:
        return self._num_samples

    def _sample_view(self, index: int, view_name: str) -> torch.Tensor:
        records = self._records_by_view.get(str(view_name), self._records)
        return _reconstruct_structured_view(
            records['input'][int(index)],
            dataset_name=self.dataset_name,
            view_name=view_name,
            sequence_input_rule=self._sequence_input_rule,
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        resolved = int(index)
        if resolved < 0:
            resolved += self._num_samples
        if resolved < 0 or resolved >= self._num_samples:
            raise IndexError(f'PreparedStructuredSplitDataset index out of range: {index}')
        return self._sample_view(resolved, self.primary_view_name), int(self.labels[resolved].item())

    def available_views(self) -> tuple[str, ...]:
        if self.dataset_name in {'s-mnist', 'ps-mnist', 's-cifar10', 'deap'}:
            return ('model_input', 'psd_input', 'model_input_psd_view')
        if self.dataset_name in {'uci-har', 'shd', 'ssc'}:
            return ('model_input', 'psd_input', 'model_input_psd_view', 'sequence_input')
        if self.dataset_name in {'mnist', 'cifar-10', 'cifar-100', 'n-mnist', 'cifar10-dvs', 'dvs128-gesture'}:
            if self.dataset_name in {'mnist', 'cifar-10', 'cifar-100'}:
                declared = tuple(str(v) for v in self.metadata.get('available_views', ()))
                defaults = ('model_input', 'model_input_cnn', 'cnn_input', 'model_input_flatten', 'psd_input', 'image_psd_view', 'original_input', 'flatten_input', 'sequence_input')
                return tuple(dict.fromkeys((*declared, *defaults)))
            return ('model_input', 'psd_input', 'event_frame_psd_view', 'original_input', 'flatten_input', 'model_input_flatten', 'sequence_input')
        return (self.primary_view_name,)

    def view_tensor(self, view_name: str) -> PreparedViewAccessor:
        resolved_name = str(view_name)
        if resolved_name not in self.available_views():
            available = ', '.join(self.available_views())
            raise KeyError(f'Prepared split view {resolved_name!r} missing. Available: {available}.')
        cached = self._view_accessor_cache.get(resolved_name)
        if cached is not None:
            return cached
        accessor = PreparedViewAccessor(self, resolved_name)
        self._view_accessor_cache[resolved_name] = accessor
        return accessor

    def with_primary_view(self, view_name: str) -> 'PreparedStructuredSplitDataset':
        return PreparedStructuredSplitDataset(
            dataset_name=self.dataset_name,
            split_name=self.split_name,
            records=self._records,
            metadata=self.metadata,
            primary_view_name=view_name,
            max_samples=self._num_samples,
            records_by_view=self._records_by_view,
        )


@dataclass(frozen=True)
class DatasetBundle:
    """Concrete prepared datasets plus metadata required by experiments."""

    dataset_name: str
    train_dataset: PreparedDatasetProtocol
    test_dataset: PreparedDatasetProtocol
    input_dim: int
    sequence_length: int
    num_classes: int
    class_labels: tuple[int, ...]
    default_hidden_sizes: tuple[int, ...]
    prep_root: Path
    manifest_path: Path
    psd_axis_kind: str
    training_view_name: str = 'model_input'
    psd_view_name: str = 'psd_input'


__all__ = [
    'DatasetBundle',
    'DatasetMetadata',
    'PreparedStructuredSplitDataset',
    'PreparedDatasetProtocol',
    'PreparedViewAccessor',
    'SequenceDataset',
    'canonicalize_model_input_batch',
]
