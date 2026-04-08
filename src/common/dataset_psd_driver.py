from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from src.common.datasets import build_dataset_bundle, normalize_dataset_name
from src.common.plotting import configure_plot_writer, flush_plot_tasks, plot_writer_metadata, shutdown_plot_worker
from src.common.probe_selection import flatten_scope_indices, probe_scope_signature, probe_union_indices, select_fixed_probe_scopes
from src.common.psd_analysis_driver import _materialize_probe_batches, _probe_reference_payloads_for_split
from src.common.psd_artifacts import combined_exact_psd_payload_from_maps_torch, merge_exact_psd_payloads, save_psd_bundle
from src.common.psd_utils import effective_psd_window, normalize_userbin_edges, temporal_band_ranges_from_edges, userbin_centers
from src.common.utils import get_backend_flags, get_device, now_timestamp_seoul, require_absolute_path, save_json


_DEFAULT_BATCH_SIZE = 256


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == '':
        return int(default)
    return int(raw)



def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == '':
        return bool(default)
    return str(raw).strip().lower() in {'1', 'true', 'yes', 'y'}



def _label_hist_to_jsonable(hist: Dict[int, int]) -> Dict[str, int]:
    return {str(int(k)): int(v) for k, v in sorted(hist.items())}



def _jsonable_idx_map(idx_map: Mapping[int, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in sorted(idx_map.items()):
        if isinstance(value, (list, tuple)):
            out[str(int(key))] = [int(v) for v in value]
        else:
            out[str(int(key))] = int(value)
    return out



def _dataset_sequence_to_map_ct(x_seq: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(x_seq):
        x_seq = torch.as_tensor(np.asarray(x_seq, dtype=np.float32))
    x_seq = x_seq.to(torch.float32)
    if x_seq.dim() != 2:
        raise ValueError(f'dataset sample must be (T,C), got {tuple(x_seq.shape)}')
    return x_seq.transpose(0, 1).contiguous()



def _compute_split_payload_from_dataset(
    dataset,
    *,
    split_name: str,
    max_samples: Optional[int],
    batch_size: int,
    device: torch.device,
    periodogram_band_ranges,
    spectrogram_band_ranges,
    nperseg_eff: int,
    noverlap_eff: int,
    window_fn: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    total_len = len(dataset)
    if max_samples is None or int(max_samples) <= 0:
        num_samples = int(total_len)
        cap_value = None
    else:
        num_samples = min(int(total_len), int(max_samples))
        cap_value = int(max_samples)

    payloads = []
    label_hist: Dict[int, int] = {}
    activity_per_sample: list[float] = []
    batch_maps: list[torch.Tensor] = []

    pbar = tqdm(range(num_samples), total=num_samples, desc=f'dataset-psd-{split_name}', leave=True)
    for idx in pbar:
        x_seq, label = dataset[int(idx)]
        map_ct = _dataset_sequence_to_map_ct(x_seq)
        batch_maps.append(map_ct)
        label_hist[int(label)] = int(label_hist.get(int(label), 0) + 1)
        activity_per_sample.append(float(map_ct.sum().item()))
        if len(batch_maps) >= int(batch_size) or int(idx) == int(num_samples) - 1:
            maps_t = torch.stack(batch_maps, dim=0).to(device=device, dtype=torch.float32)
            payload = combined_exact_psd_payload_from_maps_torch(
                maps_t,
                periodogram_band_ranges=periodogram_band_ranges,
                spectrogram_band_ranges=spectrogram_band_ranges,
                nperseg_eff=int(nperseg_eff),
                noverlap_eff=int(noverlap_eff),
                window_fn=str(window_fn),
            )
            payloads.append(payload)
            batch_maps = []
        pbar.set_postfix({'label': int(label)})

    merged = merge_exact_psd_payloads(payloads)
    split_summary = {
        'num_samples': int(num_samples),
        'max_samples_request': cap_value,
        'label_histogram': _label_hist_to_jsonable(label_hist),
        'mean_activity_sum_per_sample': float(np.mean(activity_per_sample)) if activity_per_sample else 0.0,
        'median_activity_sum_per_sample': float(np.median(activity_per_sample)) if activity_per_sample else 0.0,
    }
    return merged, split_summary


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------


@torch.no_grad()
def run_dataset_psd(
    *,
    dataset: str,
    data_root: str,
    out_root: str,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    num_workers: int = 4,
    download: bool = False,
    seed: Optional[int] = None,
    psd_window: int = 64,
    psd_overlap: int = 32,
    window_fn: str = 'hann',
    userbin_edges: Optional[Sequence[float]] = None,
    same_label_n_per_label: int = 4,
    balanced_global_n_per_label: int = 4,
    probe_plot: bool = True,
    max_samples: Optional[int] = None,
    exp_name: Optional[str] = None,
    timestamp: Optional[str] = None,
    device: str = 'auto',
    shd_T: int = 250,
    shd_max_time: float = 1.0,
    shd_binning: str = 'origin',
    shd_unit_indexing: str = 'auto',
    shd_channel_flip: bool = True,
    shd_align_to_first_event: bool = False,
    shd_use_event_counts: bool = False,
    dvsgesture_chunk_size: int = 120,
    dvsgesture_empty_size: int = 40,
    dvsgesture_dt_ms: float = 10.0,
    dvsgesture_ds: int = 4,
    deap_label_axis: int = 0,
    deap_num_classes: int = 3,
) -> str:
    if int(batch_size) <= 0:
        raise ValueError(f'batch_size must be > 0, got {batch_size}')

    dataset_name = normalize_dataset_name(dataset)
    data_root_abs = require_absolute_path(data_root, kind='data_root_abs', must_exist=False, create=True)
    out_root_abs = require_absolute_path(out_root, kind='result_root_abs', must_exist=False, create=True)
    dev = get_device(device)
    seed_value = 0 if seed is None else int(seed)

    configure_plot_writer(
        workers=max(1, _env_int('PSD_PLOT_WRITER_WORKERS', 1)),
        queue_maxsize=max(4, _env_int('PSD_PLOT_QUEUE_MAXSIZE', 8)),
        start_method=str(os.environ.get('PSD_PLOT_WRITER_START_METHOD', 'spawn')),
        dpi=max(72, _env_int('PSD_PLOT_WRITER_DPI', 180)),
        skip_existing=_env_bool('PSD_PLOT_SKIP_EXISTING', False),
    )

    bundle = build_dataset_bundle(
        dataset_name=dataset_name,
        data_root=data_root_abs,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        download=bool(download),
        seed=seed,
        shd_T=int(shd_T),
        shd_max_time=float(shd_max_time),
        shd_binning=str(shd_binning),
        shd_unit_indexing=str(shd_unit_indexing),
        shd_channel_flip=bool(shd_channel_flip),
        shd_align_to_first_event=bool(shd_align_to_first_event),
        shd_use_event_counts=bool(shd_use_event_counts),
        dvsgesture_chunk_size=int(dvsgesture_chunk_size),
        dvsgesture_empty_size=int(dvsgesture_empty_size),
        dvsgesture_dt_ms=float(dvsgesture_dt_ms),
        dvsgesture_ds=int(dvsgesture_ds),
        deap_label_axis=int(deap_label_axis),
        deap_num_classes=int(deap_num_classes),
    )

    edge_values, edge_source = normalize_userbin_edges(userbin_edges)
    nperseg_eff, noverlap_eff = effective_psd_window(int(bundle.T), int(psd_window), int(psd_overlap))
    periodogram_band_ranges = temporal_band_ranges_from_edges(int(bundle.T), edge_values)
    spectrogram_band_ranges = temporal_band_ranges_from_edges(int(nperseg_eff), edge_values)
    userbin_centers_np = userbin_centers(edge_values)

    ts = str(timestamp) if timestamp is not None else now_timestamp_seoul()
    run_name = str(exp_name or f'dataset_psd-{dataset_name}').replace(' ', '').replace('/', '-')
    run_name = f'{run_name}_{ts}'
    run_root = os.path.join(out_root_abs, run_name)
    os.makedirs(run_root, exist_ok=True)

    train_dataset = bundle.train_loader.dataset
    test_dataset = bundle.test_loader.dataset
    fixed_scopes = {
        'train': select_fixed_probe_scopes(
            train_dataset,
            int(bundle.num_classes),
            split_name='train',
            base_seed=int(seed_value),
            same_label_n=int(same_label_n_per_label),
            balanced_n=int(balanced_global_n_per_label),
        ),
        'test': select_fixed_probe_scopes(
            test_dataset,
            int(bundle.num_classes),
            split_name='test',
            base_seed=int(seed_value),
            same_label_n=int(same_label_n_per_label),
            balanced_n=int(balanced_global_n_per_label),
        ),
    }

    probe_batches_by_split = {
        'train': _materialize_probe_batches(train_dataset, fixed_scopes['train'], pin_memory=bool(dev.type == 'cuda')),
        'test': _materialize_probe_batches(test_dataset, fixed_scopes['test'], pin_memory=bool(dev.type == 'cuda')),
    }

    train_payload, train_summary = _compute_split_payload_from_dataset(
        train_dataset,
        split_name='train',
        max_samples=max_samples,
        batch_size=int(batch_size),
        device=dev,
        periodogram_band_ranges=periodogram_band_ranges,
        spectrogram_band_ranges=spectrogram_band_ranges,
        nperseg_eff=int(nperseg_eff),
        noverlap_eff=int(noverlap_eff),
        window_fn=str(window_fn),
    )
    test_payload, test_summary = _compute_split_payload_from_dataset(
        test_dataset,
        split_name='test',
        max_samples=max_samples,
        batch_size=int(batch_size),
        device=dev,
        periodogram_band_ranges=periodogram_band_ranges,
        spectrogram_band_ranges=spectrogram_band_ranges,
        nperseg_eff=int(nperseg_eff),
        noverlap_eff=int(noverlap_eff),
        window_fn=str(window_fn),
    )

    save_psd_bundle(
        os.path.join(run_root, 'train'),
        payload=train_payload,
        userbin_centers_np=np.asarray(userbin_centers_np, dtype=float),
        title_prefix='train input reference',
        signal_scope='input',
        epoch=None,
        save_summary_json=True,
        save_db_plots=True,
    )
    save_psd_bundle(
        os.path.join(run_root, 'test'),
        payload=test_payload,
        userbin_centers_np=np.asarray(userbin_centers_np, dtype=float),
        title_prefix='test input reference',
        signal_scope='input',
        epoch=None,
        save_summary_json=True,
        save_db_plots=True,
    )

    if bool(probe_plot):
        _probe_reference_payloads_for_split(
            run_root,
            'train',
            fixed_scopes['train'],
            periodogram_band_ranges=periodogram_band_ranges,
            spectrogram_band_ranges=spectrogram_band_ranges,
            nperseg_eff=int(nperseg_eff),
            noverlap_eff=int(noverlap_eff),
            window_fn=str(window_fn),
            userbin_centers_np=np.asarray(userbin_centers_np, dtype=float),
            device=dev,
            probe_batches=probe_batches_by_split['train'],
            save_plots=True,
            defer_plots=False,
        )
        _probe_reference_payloads_for_split(
            run_root,
            'test',
            fixed_scopes['test'],
            periodogram_band_ranges=periodogram_band_ranges,
            spectrogram_band_ranges=spectrogram_band_ranges,
            nperseg_eff=int(nperseg_eff),
            noverlap_eff=int(noverlap_eff),
            window_fn=str(window_fn),
            userbin_centers_np=np.asarray(userbin_centers_np, dtype=float),
            device=dev,
            probe_batches=probe_batches_by_split['test'],
            save_plots=True,
            defer_plots=False,
        )

    config: Dict[str, Any] = {
        'run_name': run_name,
        'experiment_name': 'dataset_psd',
        'dataset_name': str(bundle.dataset_name),
        'spec_doc': 'paper/proposed/dataset_psd.md',
        'reference_doc': 'paper/proposed/psd_analysis.md',
        'plotting_spec_doc': 'paper/proposed/psd_userbin_async_plot.md',
        'data_root_abs': data_root_abs,
        'result_root_abs': run_root,
        'run_root_abs': run_root,
        'parent_result_root_abs': out_root_abs,
        'download': bool(download),
        'seed': int(seed_value),
        'batch_size': int(batch_size),
        'num_workers': int(num_workers),
        'psd_window': int(psd_window),
        'psd_overlap': int(psd_overlap),
        'periodogram_length_effective': int(bundle.T),
        'spectrogram_window_effective': int(nperseg_eff),
        'spectrogram_overlap_effective': int(noverlap_eff),
        'psd_window_effective': int(nperseg_eff),
        'psd_overlap_effective': int(noverlap_eff),
        'window_fn_legacy_ignored': str(window_fn),
        'taper_window_applied': False,
        'userbin_edges': list(edge_values),
        'userbin_edges_source': str(edge_source),
        'waveform_psd_representation': 'exact_full_length_simple_periodogram_saved_for_raw_and_centered',
        'heatmap_psd_representation': 'userbin_from_exact_periodogram_saved_for_raw_and_centered',
        'spectrogram_representation': 'exact_sliding_simple_periodogram_saved_for_raw_and_centered',
        'spectrogram_heatmap_representation': 'userbin_from_exact_spectrogram_frame_major_per_element_saved_for_raw_and_centered',
        'variants_saved': ['raw', 'centered'],
        'save_db_psd_plots': True,
        'db_plot_scale': '10log10_power_plus_epsilon',
        'db_plot_epsilon': 1.0e-12,
        'same_label_n_per_label': int(same_label_n_per_label),
        'balanced_global_n_per_label': int(balanced_global_n_per_label),
        'probe_plot': bool(probe_plot),
        'probe_set_reference_root': 'probe_set_reference' if bool(probe_plot) else None,
        'probe_selection_scheme': 'canonical_hash_rank_by_split_seed_label_dataset_index_with_scope_prefix_counts_only',
        'probe_selection_scope_rule': 'same_label_and_balanced_global_take_prefixes_from_the_same_per_label_canonical_order',
        'same_label_indices_train': _jsonable_idx_map(fixed_scopes['train']['same_label']),
        'same_label_indices_test': _jsonable_idx_map(fixed_scopes['test']['same_label']),
        'balanced_global_indices_train': _jsonable_idx_map(fixed_scopes['train']['balanced_global']),
        'balanced_global_indices_test': _jsonable_idx_map(fixed_scopes['test']['balanced_global']),
        'same_label_flat_indices_train': flatten_scope_indices(fixed_scopes['train'], 'same_label'),
        'same_label_flat_indices_test': flatten_scope_indices(fixed_scopes['test'], 'same_label'),
        'balanced_global_flat_indices_train': flatten_scope_indices(fixed_scopes['train'], 'balanced_global'),
        'balanced_global_flat_indices_test': flatten_scope_indices(fixed_scopes['test'], 'balanced_global'),
        'probe_union_indices_train': probe_union_indices(fixed_scopes['train']),
        'probe_union_indices_test': probe_union_indices(fixed_scopes['test']),
        'probe_selection_signature_train': probe_scope_signature(fixed_scopes['train']),
        'probe_selection_signature_test': probe_scope_signature(fixed_scopes['test']),
        'max_samples_per_split': None if max_samples is None or int(max_samples) <= 0 else int(max_samples),
        'device': str(dev),
        'backend_flags': get_backend_flags(),
        'dataset_bundle': {
            'num_classes': int(bundle.num_classes),
            'input_dim': int(bundle.input_dim),
            'T': int(bundle.T),
            'metadata': dict(bundle.metadata),
        },
        **plot_writer_metadata(),
    }
    save_json(os.path.join(run_root, 'config.json'), config)

    summary: Dict[str, Any] = {
        'splits': {
            'train': train_summary,
            'test': test_summary,
        },
        'same_label_n_per_label': int(same_label_n_per_label),
        'balanced_global_n_per_label': int(balanced_global_n_per_label),
        'probe_plot': bool(probe_plot),
        'save_db_psd_plots': True,
        'db_plot_scale': '10log10_power_plus_epsilon',
        'db_plot_epsilon': 1.0e-12,
        'plot_files_per_split': [
            'mean_psd_waveform_exact_raw.png',
            'mean_psd_waveform_exact_centered.png',
            'element_psd_heatmap_userbin_raw.png',
            'element_psd_heatmap_userbin_centered.png',
            'mean_spectrogram_exact_raw.png',
            'mean_spectrogram_exact_centered.png',
            'element_spectrogram_heatmap_userbin_raw.png',
            'element_spectrogram_heatmap_userbin_centered.png',
            'mean_psd_waveform_exact_raw_db.png',
            'mean_psd_waveform_exact_centered_db.png',
            'element_psd_heatmap_userbin_raw_db.png',
            'element_psd_heatmap_userbin_centered_db.png',
            'mean_spectrogram_exact_raw_db.png',
            'mean_spectrogram_exact_centered_db.png',
            'element_spectrogram_heatmap_userbin_raw_db.png',
            'element_spectrogram_heatmap_userbin_centered_db.png',
            'summary.json',
        ],
        'probe_plot_files_per_scope': [
            'mean_psd_waveform_exact_raw.png',
            'mean_psd_waveform_exact_centered.png',
            'element_psd_heatmap_userbin_raw.png',
            'element_psd_heatmap_userbin_centered.png',
            'mean_spectrogram_exact_raw.png',
            'mean_spectrogram_exact_centered.png',
            'element_spectrogram_heatmap_userbin_raw.png',
            'element_spectrogram_heatmap_userbin_centered.png',
            'mean_psd_waveform_exact_raw_db.png',
            'mean_psd_waveform_exact_centered_db.png',
            'element_psd_heatmap_userbin_raw_db.png',
            'element_psd_heatmap_userbin_centered_db.png',
            'mean_spectrogram_exact_raw_db.png',
            'mean_spectrogram_exact_centered_db.png',
            'element_spectrogram_heatmap_userbin_raw_db.png',
            'element_spectrogram_heatmap_userbin_centered_db.png',
            'summary.json',
        ] if bool(probe_plot) else [],
    }
    save_json(os.path.join(run_root, 'summary.json'), summary)
    flush_plot_tasks()
    shutdown_plot_worker(wait=True)
    return run_root


@torch.no_grad()
def run_dataset_psd_shd(
    *,
    data_root: str,
    out_root: str,
    T: int = 250,
    num_units: int = 700,
    max_time: float = 1.0,
    binning: str = 'origin',
    unit_indexing: str = 'auto',
    channel_flip: bool = True,
    align_to_first_event: bool = False,
    use_event_counts: bool = False,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    download: bool = False,
    psd_window: int = 64,
    psd_overlap: int = 32,
    window_fn: str = 'hann',
    userbin_edges: Optional[Sequence[float]] = None,
    same_label_n_per_label: int = 4,
    balanced_global_n_per_label: int = 4,
    probe_plot: bool = True,
    max_samples: Optional[int] = None,
    exp_name: Optional[str] = None,
    timestamp: Optional[str] = None,
    device: str = 'auto',
) -> str:
    if int(num_units) != 700:
        raise ValueError('run_dataset_psd_shd expects standard SHD num_units=700. Use run_dataset_psd(dataset=...) for generic datasets.')
    return run_dataset_psd(
        dataset='shd',
        data_root=data_root,
        out_root=out_root,
        batch_size=int(batch_size),
        num_workers=4,
        download=bool(download),
        seed=0,
        psd_window=int(psd_window),
        psd_overlap=int(psd_overlap),
        window_fn=str(window_fn),
        userbin_edges=userbin_edges,
        same_label_n_per_label=int(same_label_n_per_label),
        balanced_global_n_per_label=int(balanced_global_n_per_label),
        probe_plot=bool(probe_plot),
        max_samples=max_samples,
        exp_name=exp_name,
        timestamp=timestamp,
        device=str(device),
        shd_T=int(T),
        shd_max_time=float(max_time),
        shd_binning=str(binning),
        shd_unit_indexing=str(unit_indexing),
        shd_channel_flip=bool(channel_flip),
        shd_align_to_first_event=bool(align_to_first_event),
        shd_use_event_counts=bool(use_event_counts),
    )
