"""Shared helpers for matrix-valued checkpoint analysis artifacts."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import src.psd_analysis as psd_common
from src.data.registry import make_loader
from src.signal.psd_utils import trace_tensor_to_channel_major_maps
from src.stat.probe_selection import (
    build_probe_index_bundle,
    build_probe_scopes,
    dataset_sample_indices,
    dataset_targets,
    subset_from_indices,
)
from src.util.csv_schema import common_row, write_common_csv

MLP_MODEL_FAMILIES = frozenset({'if', 'lif', 'rf', 'tc_lif', 'ts_lif', 'dh_snn', 'd_rf', 'my_dh_snn', 'my_d_rf', 'my_r_dh_snn'})
OUTPUT_SERIES = frozenset({'membrane', 'spike'})
ProbeMapKey = tuple[str, int, str, str, str, str, int | None, str, int | None]


@dataclass(frozen=True)
class ProbeScope:
    """One deterministic probe subset requested by matrix-valued analyses."""

    scope: str
    probe_family: str
    label: int | None
    sample_role: str
    sample_index: int | None
    subset: Any


def validate_mlp_only(model_spec: Any) -> None:
    """Fail fast when a checkpoint does not describe a plain dense MLP SNN."""

    family = str(getattr(model_spec, 'family', ''))
    if family not in MLP_MODEL_FAMILIES:
        allowed = ', '.join(sorted(MLP_MODEL_FAMILIES))
        raise ValueError(
            '2-D FFT and element-wise PSD analysis are currently implemented for MLP-style dense SNNs only. '
            f'Got model family {family!r}; allowed families are {{{allowed}}}.'
        )


def layer_index_by_name(model: torch.nn.Module) -> dict[str, int]:
    """Return stable one-based layer indices matching psd_analysis ordering."""

    mapping: dict[str, int] = {}
    if hasattr(model, 'iter_named_layers'):
        for idx, (name, _layer) in enumerate(model.iter_named_layers(), start=1):
            mapping[str(name)] = idx
    return mapping


def checkpoint_output_dir(*, output_root: Path, checkpoint_path: Path, checkpoint_payload: Mapping[str, Any], input_is_single_file: bool) -> Path:
    """Return the checkpoint-specific output directory used by analysis entrypoints."""

    if input_is_single_file:
        return output_root / checkpoint_path.stem
    return output_root / f'checkpoint_epoch_{int(checkpoint_payload.get("epoch", 0)):06d}'


def safe_token(value: Any) -> str:
    """Expose psd_analysis token normalization for compatible file names."""

    return psd_common._safe_token(value)


def layer_folder(layer: str, layer_index: int | str) -> str:
    """Expose psd_analysis layer folder naming for compatible layout."""

    return psd_common._layer_folder(layer, layer_index)


def manifest_row(*, base: dict[str, Any], artifact_name: str, path: Path, status: str = 'ok', message: str = '') -> dict[str, str]:
    """Build one standard analysis manifest row."""

    return psd_common._manifest_row(base=base, artifact_name=artifact_name, path=path, status=status, message=message)


def _stable_label_single_rank_key(*parts: object) -> int:
    """Return a deterministic rank key for label-single samples outside balanced probes."""

    digest = hashlib.sha1('|'.join(str(part) for part in parts).encode('utf-8')).hexdigest()
    return int(digest, 16)


def _label_single_indices_outside_balanced(
    dataset: Any,
    *,
    split_name: str,
    seed: int,
    balanced_indices: Sequence[int],
) -> dict[int, int]:
    """Choose one deterministic sample per label, excluding balanced-global indices."""

    targets = dataset_targets(dataset)
    sample_indices = dataset_sample_indices(dataset)
    if len(targets) != len(sample_indices):
        raise ValueError('targets and sample_indices must have the same length.')

    balanced_index_set = {int(index) for index in balanced_indices}
    grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for dataset_index, (label, sample_index) in enumerate(zip(targets, sample_indices)):
        if int(dataset_index) in balanced_index_set:
            continue
        grouped[int(label)].append((int(dataset_index), int(sample_index)))

    labels = sorted(set(int(label) for label in targets))
    selected: dict[int, int] = {}
    for label in labels:
        candidates = grouped.get(int(label), [])
        if not candidates:
            raise ValueError(
                'Cannot select a label-single sample outside the balanced_global probe set. '
                f'split={split_name!r}, label={int(label)}, balanced_count={sum(1 for index in balanced_indices if int(targets[int(index)]) == int(label))}. '
                'Use a split with at least one non-balanced sample for every label or reduce the balanced probe quota.'
            )
        dataset_index, _sample_index = min(
            candidates,
            key=lambda item: (
                _stable_label_single_rank_key(split_name, int(seed), 'label_single_excluding_balanced', int(label), item[1]),
                item[1],
                item[0],
            ),
        )
        selected[int(label)] = int(dataset_index)
    return selected




def _probe_quota(dataset: Any) -> int:
    """probe 선택에 사용할 label별 균형 quota를 계산한다."""

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

def iter_matrix_probe_scopes(dataset: Any, *, split_name: str, seed: int) -> list[ProbeScope]:
    """Return balanced-global and one-sample-per-label probe scopes.

    The balanced scope uses the same quota rule as psd_analysis. The single-label
    scopes choose one deterministic sample per label from the same split after
    excluding every index already used by the balanced-global scope.
    """

    quota = _probe_quota(dataset)
    bundle = build_probe_index_bundle(
        dataset,
        split_name=split_name,
        seed=int(seed),
        same_label_n_per_label=quota,
        balanced_global_n_per_label=quota,
        distribution_global_min_class_n=quota,
    )
    sample_indices = dataset_sample_indices(dataset)
    scopes: list[ProbeScope] = [
        ProbeScope(scope=s.scope, probe_family=s.probe_family, label=s.label, sample_role=s.sample_role, sample_index=s.sample_index, subset=s.subset)
        for s in build_probe_scopes(dataset, split_name=split_name, bundle=bundle)
    ]
    try:
        label_single_indices = _label_single_indices_outside_balanced(
            dataset,
            split_name=split_name,
            seed=int(seed),
            balanced_indices=bundle.balanced_global,
        )
    except ValueError:
        label_single_indices = {}
    for label, dataset_index in sorted(label_single_indices.items(), key=lambda item: item[0]):
        sample_index = int(sample_indices[dataset_index]) if dataset_index < len(sample_indices) else dataset_index
        scopes.append(
            ProbeScope(
                scope=f'{split_name}_label_single_label_{int(label)}',
                probe_family='label_single',
                label=int(label),
                sample_role='label_single_sample_excluding_balanced',
                sample_index=sample_index,
                subset=subset_from_indices(dataset, [dataset_index]),
            )
        )
    return scopes


def _trace_maps(tensor: torch.Tensor) -> torch.Tensor:
    if psd_common.LOW_VRAM:
        return psd_common._trace_to_cpu_maps(tensor)
    maps = trace_tensor_to_channel_major_maps(tensor.detach())
    if maps.dtype != torch.float32:
        maps = maps.to(dtype=torch.float32)
    return maps.contiguous()


def _append_output_series(
    collected: dict[ProbeMapKey, list[torch.Tensor]],
    *,
    layer_name: str,
    layer_index: int,
    signal_kind: str,
    scope: ProbeScope,
    record: Any,
) -> None:
    for series in sorted(OUTPUT_SERIES):
        tensor = getattr(record, series, None)
        if not isinstance(tensor, torch.Tensor):
            continue
        key: ProbeMapKey = (
            str(layer_name),
            int(layer_index),
            str(signal_kind),
            str(series),
            str(scope.scope),
            str(scope.probe_family),
            scope.label,
            str(scope.sample_role),
            scope.sample_index,
        )
        collected[key].append(_trace_maps(tensor))


def collect_mlp_output_maps(
    *,
    model: torch.nn.Module,
    dataset: Any,
    split_name: str,
    seed: int,
    anal_batch: int,
    num_workers: int,
    device: torch.device,
) -> dict[ProbeMapKey, torch.Tensor]:
    """균형/라벨 단일 프로브에서 hidden/output 맵만 수집한다."""

    if int(anal_batch) < 1:
        raise ValueError('anal_batch must be >= 1.')
    collected: dict[ProbeMapKey, list[torch.Tensor]] = defaultdict(list)
    layer_indices = layer_index_by_name(model)
    scopes = iter_matrix_probe_scopes(dataset, split_name=split_name, seed=seed)

    with torch.inference_mode():
        for scope in scopes:
            loader = make_loader(
                scope.subset,
                batch_size=int(anal_batch),
                shuffle=False,
                num_workers=int(num_workers),
                pin_memory=device.type == 'cuda',
                seed=int(seed),
            )
            for inputs, _target in loader:
                result = None
                model_inputs = None
                try:
                    if hasattr(model, 'input_dim') and hasattr(model, 'sequence_length'):
                        model_inputs = psd_common._prepared_input_for_model(model, inputs, device=device)
                    else:
                        model_inputs = torch.as_tensor(inputs, dtype=torch.float32, device=device)
                    result = model(model_inputs, capture_hidden=True)
                    for record in result.hidden_records:
                        layer_name = str(record.layer_name)
                        _append_output_series(
                            collected,
                            layer_name=layer_name,
                            layer_index=int(layer_indices.get(layer_name, len(layer_indices) + 1)),
                            signal_kind='hidden',
                            scope=scope,
                            record=record,
                        )
                    output_record = result.output_record
                    output_layer_name = str(getattr(output_record, 'layer_name', 'output') or 'output')
                    _append_output_series(
                        collected,
                        layer_name=output_layer_name,
                        layer_index=int(layer_indices.get(output_layer_name, layer_indices.get('output', 999))),
                        signal_kind='output',
                        scope=scope,
                        record=output_record,
                    )
                finally:
                    del result
                    del model_inputs
                    del inputs
                    if device.type == 'cuda' and psd_common.LOW_VRAM:
                        torch.cuda.empty_cache()

    return {key: torch.cat(values, dim=0).contiguous() for key, values in collected.items() if values}


def common_base_for_key(*, checkpoint_base: Mapping[str, Any], key: ProbeMapKey) -> dict[str, Any]:
    """Build category row metadata shared by matrix-valued analysis outputs."""

    layer_name, layer_index, signal_kind, series, scope, probe_family, label, sample_role, sample_index = key
    base = dict(checkpoint_base)
    base.update(
        layer=layer_name,
        layer_index=int(layer_index),
        scope=scope,
        probe_family=probe_family,
        label='' if label is None else int(label),
        sample_role=sample_role,
        sample_index='' if sample_index is None else int(sample_index),
        signal_kind=signal_kind,
        series=series,
    )
    return base


def value_columns(prefix: str, count: int) -> list[str]:
    """Return stable dynamic matrix value column names."""

    token = safe_token(prefix)
    return [f'{token}_{idx:06d}' for idx in range(int(count))]


def write_matrix_csv(path: Path, rows: list[dict[str, Any]], *, value_column_names: Sequence[str]) -> None:
    """Write one matrix-like category CSV with dynamic numeric value columns."""

    write_common_csv(path, rows, extra_columns=list(value_column_names))


__all__ = [
    'MLP_MODEL_FAMILIES',
    'OUTPUT_SERIES',
    'ProbeMapKey',
    'ProbeScope',
    'checkpoint_output_dir',
    'collect_mlp_output_maps',
    'common_base_for_key',
    'iter_matrix_probe_scopes',
    'layer_folder',
    'layer_index_by_name',
    'manifest_row',
    'safe_token',
    'validate_mlp_only',
    'value_columns',
    'write_matrix_csv',
]
