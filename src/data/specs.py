"""Central dataset catalog for official prepared-bundle experiments.

This module is the single source of truth for

1. canonical dataset tokens and aliases,
2. default hidden-size baselines, and
3. the official view / axis defaults consumed by ``data_prep`` and downstream
   bundle validation.

Keeping this information in one place makes later dataset additions much less
error-prone: adding one dataset should not require repeating token / alias /
default-view metadata in several files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


_PREPROCESSING_SPEC_DOC = 'spec/Theory/data_prep/data_prep.md'
_PREPROCESSING_IMPL_SPEC_DOC = 'spec/Implementation/data_prep.md'


@dataclass(frozen=True)
class DatasetSpec:
    """Official dataset-level contract derived from ``spec/Theory/data_prep/data_prep.md``."""

    canonical_name: str
    aliases: tuple[str, ...]
    default_hidden_sizes: tuple[int, ...]
    psd_axis_kind: str
    training_view_name: str = 'model_input'
    psd_view_name: str = 'psd_input'
    required_views: tuple[str, ...] = ('model_input', 'psd_input')
    spatial_auxiliary_views: tuple[str, ...] = ()
    preprocessing_spec_doc: str = _PREPROCESSING_SPEC_DOC
    preprocessing_impl_spec_doc: str = _PREPROCESSING_IMPL_SPEC_DOC

    def normalized_aliases(self) -> tuple[str, ...]:
        """Handle ``normalized aliases`` for the ``specs`` module."""
        tokens = {self.canonical_name}
        tokens.update(self.aliases)
        return tuple(sorted(_normalize_dataset_token(token) for token in tokens))

    def required_view_names(self, *, psd_axis_kind: str | None = None) -> tuple[str, ...]:
        """Handle ``required view names`` for the ``specs`` module."""
        axis_kind = str(self.psd_axis_kind if psd_axis_kind is None else psd_axis_kind)
        views = list(self.required_views)
        for name in (self.training_view_name, self.psd_view_name):
            if name not in views:
                views.append(name)
        if axis_kind in {'raster_spatial', 'image_temporal'}:
            for name in self.spatial_auxiliary_views:
                if name not in views:
                    views.append(name)
        return tuple(views)


_DATASET_SPECS = (
    DatasetSpec(
        canonical_name='s-mnist',
        aliases=('s_mnist', 'smnist', 'sequential-mnist', 'sequential_mnist'),
        default_hidden_sizes=(64, 256),
        psd_axis_kind='temporal',
        psd_view_name='model_input_psd_view',
    ),
    DatasetSpec(
        canonical_name='ps-mnist',
        aliases=('ps_mnist', 'psmnist', 'permuted-mnist', 'permuted_mnist'),
        default_hidden_sizes=(64, 256),
        psd_axis_kind='temporal',
        psd_view_name='model_input_psd_view',
    ),
    DatasetSpec(
        canonical_name='s-cifar10',
        aliases=('s_cifar10', 'scifar10', 'sequential-cifar10', 'sequential_cifar10'),
        default_hidden_sizes=(128, 256),
        psd_axis_kind='temporal',
        psd_view_name='model_input_psd_view',
    ),
    DatasetSpec(
        canonical_name='shd',
        aliases=('shd',),
        default_hidden_sizes=(256, 256),
        psd_axis_kind='temporal',
        psd_view_name='model_input_psd_view',
        training_view_name='model_input',
    ),
    DatasetSpec(
        canonical_name='ssc',
        aliases=('ssc',),
        default_hidden_sizes=(256, 256),
        psd_axis_kind='temporal',
        psd_view_name='model_input_psd_view',
        training_view_name='model_input',
    ),
    DatasetSpec(
        canonical_name='deap',
        aliases=('deap',),
        default_hidden_sizes=(128, 128),
        psd_axis_kind='temporal',
        psd_view_name='model_input_psd_view',
    ),
    DatasetSpec(
        canonical_name='uci-har',
        aliases=('uci_har', 'ucihar', 'har'),
        default_hidden_sizes=(128, 128),
        psd_axis_kind='temporal',
        psd_view_name='model_input_psd_view',
        training_view_name='model_input',
    ),
    DatasetSpec(
        canonical_name='mnist',
        aliases=('mnist',),
        default_hidden_sizes=(128, 128),
        psd_axis_kind='static_repeat',
        psd_view_name='image_psd_view',
        training_view_name='model_input',
        spatial_auxiliary_views=('original_input', 'flatten_input'),
    ),
    DatasetSpec(
        canonical_name='cifar-10',
        aliases=('cifar10', 'cifar_10'),
        default_hidden_sizes=(256, 256),
        psd_axis_kind='static_repeat',
        psd_view_name='image_psd_view',
        training_view_name='model_input',
        spatial_auxiliary_views=('original_input', 'flatten_input'),
    ),
    DatasetSpec(
    canonical_name='cifar-100',
    aliases=('cifar100', 'cifar_100'),
    default_hidden_sizes=(256, 256),
    psd_axis_kind='static_repeat',
    psd_view_name='image_psd_view',
    training_view_name='model_input',
    spatial_auxiliary_views=('original_input', 'flatten_input'),
    ),
    DatasetSpec(
        canonical_name='n-mnist',
        aliases=('n_mnist', 'nmnist'),
        default_hidden_sizes=(256, 256),
        psd_axis_kind='image_temporal',
        psd_view_name='event_frame_psd_view',
        training_view_name='model_input',
        spatial_auxiliary_views=('original_input', 'flatten_input'),
    ),
    DatasetSpec(
        canonical_name='cifar10-dvs',
        aliases=('cifar10_dvs', 'cifar-dvs', 'cifar10dvs'),
        default_hidden_sizes=(256, 256),
        psd_axis_kind='image_temporal',
        psd_view_name='event_frame_psd_view',
        training_view_name='model_input',
        spatial_auxiliary_views=('original_input', 'flatten_input'),
    ),
    DatasetSpec(
        canonical_name='dvs128-gesture',
        aliases=('dvs128_gesture', 'dvs128gesture', 'dvsgesture', 'gesture128', 'gesture-128'),
        default_hidden_sizes=(256, 256),
        psd_axis_kind='image_temporal',
        psd_view_name='event_frame_psd_view',
        training_view_name='model_input',
        spatial_auxiliary_views=('original_input', 'flatten_input'),
    ),
)


def _normalize_dataset_token(name: str) -> str:
    """Internal helper that normalize dataset token."""
    return str(name).strip().lower().replace('_', '-')


_SPEC_BY_CANONICAL = {spec.canonical_name: spec for spec in _DATASET_SPECS}
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _spec in _DATASET_SPECS:
    for _alias in _spec.normalized_aliases():
        _ALIAS_TO_CANONICAL[_alias] = _spec.canonical_name


def available_dataset_tokens() -> tuple[str, ...]:
    """Return available dataset tokens."""
    return tuple(sorted(_SPEC_BY_CANONICAL.keys()))


def canonicalize_dataset_name(name: str) -> str:
    """Canonicalize dataset name."""
    token = _normalize_dataset_token(name)
    if token not in _ALIAS_TO_CANONICAL:
        available = ', '.join(available_dataset_tokens())
        raise ValueError(f"Unsupported dataset token '{name}'. Registered canonical datasets: {available}.")
    return _ALIAS_TO_CANONICAL[token]


def get_dataset_spec(name: str) -> DatasetSpec:
    """Handle ``get dataset spec`` for the ``specs`` module."""
    return _SPEC_BY_CANONICAL[canonicalize_dataset_name(name)]


def iter_dataset_specs() -> tuple[DatasetSpec, ...]:
    """Iterate over dataset specs."""
    return tuple(sorted(_DATASET_SPECS, key=lambda spec: spec.canonical_name))


def default_hidden_sizes_for_dataset(name: str) -> tuple[int, ...]:
    """Handle ``default hidden sizes for dataset`` for the ``specs`` module."""
    return get_dataset_spec(name).default_hidden_sizes


def required_view_names_for_dataset(name: str, *, psd_axis_kind: str | None = None) -> tuple[str, ...]:
    """Handle ``required view names for dataset`` for the ``specs`` module."""
    return get_dataset_spec(name).required_view_names(psd_axis_kind=psd_axis_kind)


__all__ = [
    'DatasetSpec',
    'available_dataset_tokens',
    'canonicalize_dataset_name',
    'default_hidden_sizes_for_dataset',
    'get_dataset_spec',
    'iter_dataset_specs',
    'required_view_names_for_dataset',
]
