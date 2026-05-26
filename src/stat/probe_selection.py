"""Deterministic probe-set selection shared by dataset_psd and psd_analysis."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset, Subset


_DISTRIBUTION_ROUNDING_RULE = 'floor_plus_largest_remainder_with_half_up_total_and_class_id_tie_break'


_FAMILY_HASH_TOKEN = {
    # Spec/theory/dataset_psd/dataset_psd.md defines the family-specific seed inputs with
    # the short string literals "same", "bal", and "dist". Using these exact
    # tokens here keeps the implementation byte-for-byte aligned with the spec.
    'same_label': 'same',
    'balanced_global': 'bal',
    'distribution_global': 'dist',
}


@dataclass(frozen=True)
class ProbeIndexBundle:
    """Resolved deterministic probe indices for one split."""

    same_label: dict[int, list[int]]
    balanced_global: list[int]
    distribution_global: list[int]
    same_label_per_label: dict[int, list[int]]
    balanced_global_per_label: dict[int, list[int]]
    distribution_global_per_label: dict[int, list[int]]
    class_counts: dict[int, int]
    distribution_global_ideal_quota: dict[int, float]
    distribution_global_floor_quota: dict[int, int]
    distribution_global_fractional_part: dict[int, float]
    distribution_global_rounded_quota: dict[int, int]
    distribution_global_total_quota: int
    distribution_global_min_class_count: int
    distribution_global_rounding_rule: str = _DISTRIBUTION_ROUNDING_RULE


@dataclass(frozen=True)
class ProbeScope:
    scope: str
    probe_family: str
    label: int | None
    subset: Subset
    sample_role: str | None = None
    sample_index: int | None = None


def _stable_rank_key(*parts: object) -> int:
    """Internal helper for ``stable rank key`` in the ``probe_selection`` module."""
    digest = hashlib.sha1('|'.join(str(part) for part in parts).encode('utf-8')).hexdigest()
    return int(digest, 16)


def dataset_targets(dataset: Dataset) -> list[int]:
    """Handle ``dataset targets`` for the ``probe_selection`` module."""
    if isinstance(dataset, Subset):
        parent_targets = dataset_targets(dataset.dataset)
        return [int(parent_targets[index]) for index in dataset.indices]
    for attribute in ('targets', 'labels', 'ys'):
        if hasattr(dataset, attribute):
            values = getattr(dataset, attribute)
            if isinstance(values, torch.Tensor):
                return [int(v) for v in values.tolist()]
            return [int(v) for v in list(values)]
    return [int(dataset[index][1]) for index in range(len(dataset))]


def dataset_sample_indices(dataset: Dataset) -> list[int]:
    """Handle ``dataset sample indices`` for the ``probe_selection`` module."""
    if isinstance(dataset, Subset):
        parent_indices = dataset_sample_indices(dataset.dataset)
        return [int(parent_indices[index]) for index in dataset.indices]
    if hasattr(dataset, 'sample_indices'):
        values = getattr(dataset, 'sample_indices')
        return [int(v) for v in list(values)]
    return list(range(len(dataset)))


def _group_indices_by_label(
    targets: Sequence[int],
    *,
    sample_indices: Sequence[int],
) -> dict[int, list[tuple[int, int]]]:
    """Internal helper for ``group indices by label`` in the ``probe_selection`` module."""
    grouped: dict[int, list[tuple[int, int]]] = {}
    for dataset_index, (label, sample_index) in enumerate(zip(targets, sample_indices)):
        grouped.setdefault(int(label), []).append((int(dataset_index), int(sample_index)))
    return grouped


def _ordered_indices_for_family(
    label_entries: list[tuple[int, int]],
    *,
    split_name: str,
    seed: int,
    family_token: str,
    label: int,
) -> list[int]:
    """Internal helper for ``ordered indices for family`` in the ``probe_selection`` module."""
    spec_family_token = _FAMILY_HASH_TOKEN.get(str(family_token), str(family_token))
    return [
        dataset_index
        for dataset_index, sample_index in sorted(
            label_entries,
            key=lambda item: (
                _stable_rank_key(split_name, seed, spec_family_token, label, item[1]),
                item[1],
                item[0],
            ),
        )
    ]


def _round_half_up(value: float) -> int:
    """Internal helper for ``round half up`` in the ``probe_selection`` module."""
    return int(math.floor(float(value) + 0.5))


def _distribution_global_quota(
    class_counts: dict[int, int],
    *,
    min_class_n: int,
) -> tuple[dict[int, float], dict[int, int], dict[int, float], dict[int, int], int, int]:
    """Internal helper for ``distribution global quota`` in the ``probe_selection`` module."""
    ordered_labels = sorted(class_counts.keys())
    if not ordered_labels:
        raise ValueError('Cannot build a probe bundle for an empty dataset split.')
    min_count = min(int(class_counts[label]) for label in ordered_labels)
    if min_count <= 0:
        raise ValueError('Every class count must be positive for distribution_global.')
    if int(min_class_n) <= 0:
        raise ValueError('distribution_global_min_class_n must be positive.')
    if int(min_class_n) > int(min_count):
        raise ValueError(
            'distribution_global_min_class_n must not exceed the minimum class count within the split: '
            f'{min_class_n} > {min_count}.'
        )

    ideal: dict[int, float] = {
        int(label): (float(class_counts[label]) / float(min_count)) * float(min_class_n)
        for label in ordered_labels
    }
    floor_quota: dict[int, int] = {int(label): int(math.floor(ideal[label])) for label in ordered_labels}
    fractional: dict[int, float] = {int(label): float(ideal[label] - float(floor_quota[label])) for label in ordered_labels}
    total_quota = _round_half_up(sum(float(ideal[label]) for label in ordered_labels))
    remainder = int(total_quota - sum(int(floor_quota[label]) for label in ordered_labels))
    rounded = dict(floor_quota)
    if remainder > 0:
        ranked_labels = sorted(ordered_labels, key=lambda label: (-fractional[label], int(label)))
        for label in ranked_labels[:remainder]:
            rounded[int(label)] = int(rounded[int(label)] + 1)
    for label in ordered_labels:
        count = int(class_counts[label])
        quota = int(rounded[label])
        if quota < 0 or quota > count:
            raise ValueError(
                f'Rounded distribution_global quota for label {label} became invalid: {quota} with class count {count}.'
            )
    return ideal, floor_quota, fractional, rounded, int(total_quota), int(min_count)


def build_probe_index_bundle(
    dataset: Dataset,
    *,
    split_name: str,
    seed: int,
    same_label_n_per_label: int,
    balanced_global_n_per_label: int,
    distribution_global_min_class_n: int,
) -> ProbeIndexBundle:
    """Build independent same_label, balanced_global, and distribution_global subsets.

    The implementation follows ``Spec/theory/dataset_psd/dataset_psd.md`` exactly: each
    family uses its own hash token, tie-breaks by ``sample_index`` then
    ``dataset_index``, and concatenates multi-label families in ascending label
    order.
    """

    same_n = int(same_label_n_per_label)
    bal_n = int(balanced_global_n_per_label)
    dist_min_n = int(distribution_global_min_class_n)
    targets = dataset_targets(dataset)
    sample_indices = dataset_sample_indices(dataset)
    if len(targets) != len(sample_indices):
        raise ValueError('targets and sample_indices must have the same length.')
    grouped = _group_indices_by_label(targets, sample_indices=sample_indices)
    ordered_labels = sorted(grouped.keys())
    if not ordered_labels:
        raise ValueError('Cannot build a probe bundle for an empty dataset split.')
    class_counts = {int(label): int(len(grouped[label])) for label in ordered_labels}
    min_class_count = min(class_counts.values())
    if same_n <= 0:
        raise ValueError('same_label_n_per_label must be positive.')
    if bal_n <= 0:
        raise ValueError('balanced_global_n_per_label must be positive.')
    if dist_min_n <= 0:
        raise ValueError('distribution_global_min_class_n must be positive.')
    if same_n > min_class_count:
        raise ValueError(
            'same_label_n_per_label must not exceed the minimum class count within the split: '
            f'{same_n} > {min_class_count}.'
        )
    if bal_n > min_class_count:
        raise ValueError(
            'balanced_global_n_per_label must not exceed the minimum class count within the split: '
            f'{bal_n} > {min_class_count}.'
        )

    dist_ideal, dist_floor, dist_fractional, dist_rounded, dist_total, dist_min_count = _distribution_global_quota(
        class_counts,
        min_class_n=dist_min_n,
    )

    same_label_per_label: dict[int, list[int]] = {}
    balanced_global_per_label: dict[int, list[int]] = {}
    distribution_global_per_label: dict[int, list[int]] = {}
    for label in ordered_labels:
        entries = grouped[label]
        same_order = _ordered_indices_for_family(entries, split_name=split_name, seed=seed, family_token='same_label', label=label)
        bal_order = _ordered_indices_for_family(entries, split_name=split_name, seed=seed, family_token='balanced_global', label=label)
        dist_order = _ordered_indices_for_family(entries, split_name=split_name, seed=seed, family_token='distribution_global', label=label)
        same_label_per_label[int(label)] = same_order[:same_n]
        balanced_global_per_label[int(label)] = bal_order[:bal_n]
        distribution_global_per_label[int(label)] = dist_order[: int(dist_rounded[int(label)])]

    balanced_global: list[int] = []
    distribution_global: list[int] = []
    for label in ordered_labels:
        balanced_global.extend(int(index) for index in balanced_global_per_label[label])
        distribution_global.extend(int(index) for index in distribution_global_per_label[label])

    return ProbeIndexBundle(
        same_label={int(label): list(indices) for label, indices in same_label_per_label.items()},
        balanced_global=balanced_global,
        distribution_global=distribution_global,
        same_label_per_label={int(label): list(indices) for label, indices in same_label_per_label.items()},
        balanced_global_per_label={int(label): list(indices) for label, indices in balanced_global_per_label.items()},
        distribution_global_per_label={int(label): list(indices) for label, indices in distribution_global_per_label.items()},
        class_counts={int(label): int(count) for label, count in class_counts.items()},
        distribution_global_ideal_quota={int(label): float(dist_ideal[label]) for label in ordered_labels},
        distribution_global_floor_quota={int(label): int(dist_floor[label]) for label in ordered_labels},
        distribution_global_fractional_part={int(label): float(dist_fractional[label]) for label in ordered_labels},
        distribution_global_rounded_quota={int(label): int(dist_rounded[label]) for label in ordered_labels},
        distribution_global_total_quota=int(dist_total),
        distribution_global_min_class_count=int(dist_min_count),
    )


def subset_from_indices(dataset: Dataset, indices: Sequence[int]) -> Subset:
    """Handle ``subset from indices`` for the ``probe_selection`` module."""
    return Subset(dataset, list(int(index) for index in indices))


def iter_probe_subsets(dataset: Dataset, bundle: ProbeIndexBundle) -> Iterable[tuple[str, int | None, Subset]]:
    """Iterate over probe subsets."""
    yield 'balanced_global', None, subset_from_indices(dataset, bundle.balanced_global)
    yield 'distribution_global', None, subset_from_indices(dataset, bundle.distribution_global)
    for label, indices in sorted(bundle.same_label.items(), key=lambda item: item[0]):
        yield 'same_label', int(label), subset_from_indices(dataset, indices)


def build_probe_scopes(dataset: Dataset, *, split_name: str, bundle: ProbeIndexBundle) -> list[ProbeScope]:
    scopes: list[ProbeScope] = [
        ProbeScope(scope=f'{split_name}_balanced_global', probe_family='balanced_global', label=None, subset=subset_from_indices(dataset, bundle.balanced_global), sample_role='balanced_mean'),
        ProbeScope(scope=f'{split_name}_distribution_global', probe_family='distribution_global', label=None, subset=subset_from_indices(dataset, bundle.distribution_global), sample_role='distribution_mean'),
    ]
    for label, indices in sorted(bundle.same_label.items(), key=lambda item: item[0]):
        scopes.append(ProbeScope(scope=f'{split_name}_same_label_label_{int(label)}', probe_family='same_label', label=int(label), subset=subset_from_indices(dataset, indices), sample_role='same_label_mean'))
    return scopes


__all__ = [
    'ProbeIndexBundle',
    'ProbeScope',
    'build_probe_index_bundle',
    'build_probe_scopes',
    'dataset_sample_indices',
    'dataset_targets',
    'iter_probe_subsets',
    'subset_from_indices',
]
