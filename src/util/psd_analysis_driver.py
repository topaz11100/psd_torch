from __future__ import annotations

import csv
import os
import re
import shutil
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.data.datasets import build_dataset_bundle, get_shd_loaders, normalize_dataset_name
from src.signal.psd_artifacts import (
    FeedForwardSNNWithReadout,
    build_common_classifier,
    combined_exact_psd_payload_from_maps_torch,
    input_time_matrix,
    layer_names_from_hidden,
    layer_signal_matrix,
    maybe_active_mask,
    save_psd_bundle,
)
from src.model.psd_training import configure_structure_schedule, evaluate_model, train_one_epoch
from src.model.model_registry import get_model_spec, resolve_model_name, spike_driving_membrane_key
from src.model.optim import build_adamw
from src.plot.plotting import (
    configure_plot_writer,
    flush_plot_tasks,
    plot_writer_metadata,
    save_bar_plot,
    save_heatmap_plot,
    save_hist_bar,
    save_line_plot,
    shutdown_plot_worker,
)
from src.plot.deferred_plot_tasks import (
    deferred_plot_metadata,
    render_deferred_plot_tasks,
    save_deferred_bar_plot,
    save_deferred_hist_bar,
    save_deferred_line_plot,
    save_deferred_psd_bundle,
)
from src.stat.probe_selection import flatten_scope_indices, probe_scope_signature, probe_union_indices, select_fixed_probe_scopes
from src.model.psd_model_variants import (
    default_band_neuron_ends,
    groups_from_cli,
    infer_num_groups_from_band_neuron_ends,
    parse_psd_model_variant,
    validate_lif_clip_edges,
    validate_rf_clip_edges,
    validate_tear,
)
from src.signal.psd_utils import (
    effective_psd_window,
    normalize_userbin_edges,
    temporal_band_ranges_from_edges,
    userbin_centers,
)
from src.readout.readout import normalize_readout_mode
from src.model.snn_builder import SNNConfig
from src.neurons.surrogate import SpikeFn
from src.model.first_spike_loss import FirstSpikeLoss
from src.util.utils import get_backend_flags, get_device, now_timestamp_seoul, require_absolute_path, save_json, save_text, set_seed
from src.neurons.LIF_neuron import LIFDenseLayer
from src.neurons.RF_neuron import RFDenseLayer


_TRAINING_COMPLETE_STATS_DIRNAME = "training_complete_stats"

_PSD_BUNDLE_REQUIRED_FILES: tuple[str, ...] = (
    "mean_psd_waveform_exact_raw.png",
    "mean_psd_waveform_exact_centered.png",
    "element_psd_heatmap_userbin_raw.png",
    "element_psd_heatmap_userbin_centered.png",
    "mean_spectrogram_exact_raw.png",
    "mean_spectrogram_exact_centered.png",
    "element_spectrogram_heatmap_userbin_raw.png",
    "element_spectrogram_heatmap_userbin_centered.png",
    "mean_psd_waveform_exact_raw_db.png",
    "mean_psd_waveform_exact_centered_db.png",
    "element_psd_heatmap_userbin_raw_db.png",
    "element_psd_heatmap_userbin_centered_db.png",
    "mean_spectrogram_exact_raw_db.png",
    "mean_spectrogram_exact_centered_db.png",
    "element_spectrogram_heatmap_userbin_raw_db.png",
    "element_spectrogram_heatmap_userbin_centered_db.png",
    "summary.json",
)


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _jsonable_idx_map(idx_map: Mapping[int, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in sorted(idx_map.items()):
        if isinstance(value, (list, tuple)):
            out[str(int(key))] = [int(v) for v in value]
        else:
            out[str(int(key))] = int(value)
    return out


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y"}


def _psd_bundle_is_complete(base_dir: str) -> bool:
    root = os.path.abspath(str(base_dir))
    return all(os.path.exists(os.path.join(root, name)) for name in _PSD_BUNDLE_REQUIRED_FILES)


def _copy_file_if_exists(src: str, dst: str) -> None:
    if not os.path.exists(src):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def _normalize_plot_epochs(plot_epochs: Optional[Sequence[int]], total_epochs: int) -> List[int]:
    total = int(total_epochs)
    if total < 0:
        raise ValueError(f"total_epochs must be >= 0, got {total_epochs}")
    if plot_epochs is None:
        return [int(v) for v in range(1, total + 1)]

    normalized: List[int] = []
    seen: set[int] = set()
    for raw_epoch in plot_epochs:
        epoch = int(raw_epoch)
        if epoch < 1 or epoch > total:
            raise ValueError(f"plot_epochs entries must satisfy 1 <= epoch <= {total}, got {epoch}")
        if epoch not in seen:
            normalized.append(epoch)
            seen.add(epoch)
    return sorted(normalized)


def _sanitize_run_name(name: str) -> str:
    return str(name).replace(" ", "").replace("/", "-")



def _fetch_dataset_sample(dataset, idx: int) -> Tuple[np.ndarray, int]:
    x_seq, label = dataset[int(idx)]
    return x_seq.detach().cpu().numpy(), int(label)



def _fetch_dataset_batch(dataset, indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[int] = []
    for idx in indices:
        x_seq, label = dataset[int(idx)]
        xs.append(np.asarray(x_seq.detach().cpu().numpy(), dtype=np.float32))
        ys.append(int(label))
    if len(xs) == 0:
        raise ValueError("indices must contain at least one dataset index")
    return np.stack(xs, axis=0), np.asarray(ys, dtype=np.int64)





def _comparison_items_for_split(split_scopes: Mapping[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    same_label = split_scopes["same_label"]
    for label in sorted(same_label.keys()):
        idxs = [int(v) for v in same_label[int(label)]]
        items.append(
            {
                "rel_dir": os.path.join("same_label", f"label_{int(label)}"),
                "indices": idxs,
                "label": int(label),
                "scope_name": "same_label",
            }
        )
    balanced_flat = flatten_scope_indices(split_scopes, "balanced_global")
    if len(balanced_flat) > 0:
        items.append(
            {
                "rel_dir": os.path.join("balanced_global"),
                "indices": balanced_flat,
                "label": None,
                "scope_name": "balanced_global",
            }
        )
    return items


# -----------------------------------------------------------------------------
# Model construction helpers
# -----------------------------------------------------------------------------


def _build_variant_classifier(
    *,
    model_token: str,
    input_dim: int,
    hidden: Sequence[int],
    num_classes: int,
    v_th: float,
    rf_reset_mode: str,
    rf_clip_edges: Optional[Sequence[float]],
    lif_clip_edges: Optional[Sequence[float]],
    band_neuron_ends: Optional[Sequence[str]],
    tear: int,
    readout_mode: str,
    recurrent: bool,
) -> tuple[FeedForwardSNNWithReadout, Dict[str, Any], str]:
    variant = parse_psd_model_variant(model_token)
    if variant is None:
        raise ValueError(f"unknown PSD model variant: {model_token}")

    hidden = [int(h) for h in hidden]
    if len(hidden) < 1:
        raise ValueError("hidden must contain at least one hidden layer")

    num_groups = 1
    band_ends_text = None if band_neuron_ends is None else [str(x) for x in band_neuron_ends]
    group_ids: Optional[List[np.ndarray]] = None
    group_ends: Optional[List[List[int]]] = None
    clip_values: Optional[List[float]] = None
    if variant.base_model == "RF" and (variant.structure_mask or variant.clip_params):
        if variant.clip_params:
            if rf_clip_edges is None:
                raise ValueError("rf_clip/rf_structclip require --w_clip_edges")
            clip_values = validate_rf_clip_edges(rf_clip_edges)
            num_groups = len(clip_values) - 1
        else:
            num_groups = infer_num_groups_from_band_neuron_ends(band_ends_text) if band_ends_text is not None else 2
        if band_ends_text is None:
            band_ends_text = default_band_neuron_ends(hidden, num_groups=num_groups)
        group_ends, group_ids = groups_from_cli(hidden, band_neuron_ends=band_ends_text, num_groups=num_groups)
    elif variant.base_model == "LIF" and (variant.structure_mask or variant.clip_params):
        if variant.clip_params:
            if lif_clip_edges is None:
                raise ValueError("lif_clip/lif_structclip require --alpha_clip_edges")
            clip_values = validate_lif_clip_edges(lif_clip_edges)
            num_groups = len(clip_values) - 1
        else:
            num_groups = infer_num_groups_from_band_neuron_ends(band_ends_text) if band_ends_text is not None else 2
        if band_ends_text is None:
            band_ends_text = default_band_neuron_ends(hidden, num_groups=num_groups)
        group_ends, group_ids = groups_from_cli(hidden, band_neuron_ends=band_ends_text, num_groups=num_groups)

    tear_eff = validate_tear(int(tear), len(hidden)) if (variant.structure_mask or variant.clip_params) else 1

    dims = [int(input_dim)] + hidden
    hidden_layers = nn.ModuleList()
    spike_fn = SpikeFn(name="mg", lens=0.5, gamma=0.5)

    for li in range(len(hidden)):
        in_dim = dims[li]
        out_dim = dims[li + 1]
        mask = None
        recurrent_mask = None
        clip_bounds = None
        dest_hidden_index = li + 1
        # Structured connection masks act on hidden-to-hidden projections.
        # The input stream itself is not grouped, so the first hidden layer has
        # no structural mask to apply. Once a previous hidden layer exists,
        # the destination hidden-layer index follows the public 1-based ``tear``
        # convention.
        if group_ids is not None and variant.structure_mask and li >= 1 and dest_hidden_index >= int(tear_eff):
            prev_groups = group_ids[li - 1]
            curr_groups = group_ids[li]
            mask = (curr_groups[:, None] == prev_groups[None, :]).astype(np.float32)
        if bool(recurrent) and group_ids is not None and variant.structure_mask and dest_hidden_index >= int(tear_eff):
            curr_groups = group_ids[li]
            recurrent_mask = (curr_groups[:, None] == curr_groups[None, :]).astype(np.float32)

        if group_ids is not None and clip_values is not None:
            # ``tear`` is the public 1-based destination hidden-layer index where
            # clipped parameter constraints begin. Apply it consistently to both
            # clip-only and struct+clip variants.
            clip_enabled_here = bool(variant.clip_params) and dest_hidden_index >= int(tear_eff)
            if clip_enabled_here:
                lows = np.asarray([clip_values[int(g)] for g in group_ids[li]], dtype=np.float32)
                highs = np.asarray([clip_values[int(g) + 1] for g in group_ids[li]], dtype=np.float32)
                clip_bounds = (torch.from_numpy(lows), torch.from_numpy(highs))

        if variant.base_model == "RF":
            layer = RFDenseLayer(
                in_dim,
                out_dim,
                dt=1.0,
                threshold=float(v_th),
                reset_mode=str(rf_reset_mode),
                spike_fn=spike_fn,
                clip_f_bounds=clip_bounds,
                input_group_mask=None if mask is None else torch.from_numpy(mask),
                recurrent=bool(recurrent),
                recurrent_group_mask=None if recurrent_mask is None else torch.from_numpy(recurrent_mask),
            )
        elif variant.base_model == "LIF":
            layer = LIFDenseLayer(
                in_dim,
                out_dim,
                v_th=float(v_th),
                spike_fn=spike_fn,
                alpha_clip_bounds=clip_bounds,
                input_group_mask=None if mask is None else torch.from_numpy(mask),
                recurrent=bool(recurrent),
                recurrent_group_mask=None if recurrent_mask is None else torch.from_numpy(recurrent_mask),
            )
        else:
            raise ValueError(f"unsupported base model for PSD variants: {variant.base_model}")
        hidden_layers.append(layer)

    if variant.base_model == "RF":
        output_layer = RFDenseLayer(
            int(hidden[-1]),
            int(num_classes),
            dt=1.0,
            threshold=float(v_th),
            reset_mode=str(rf_reset_mode),
            spike_fn=spike_fn,
            recurrent=bool(recurrent),
        )
    elif variant.base_model == "LIF":
        output_layer = LIFDenseLayer(
            int(hidden[-1]),
            int(num_classes),
            v_th=float(v_th),
            spike_fn=spike_fn,
            recurrent=bool(recurrent),
        )
    else:
        raise ValueError(f"unsupported base model for PSD variants: {variant.base_model}")

    model = FeedForwardSNNWithReadout(
        hidden_layers=hidden_layers,
        output_layer=output_layer,
        readout_mode=str(readout_mode),
    )
    meta = {
        "model_token": str(model_token),
        "base_model_name": str(variant.base_model),
        "structure_mask": bool(variant.structure_mask),
        "clip_params": bool(variant.clip_params),
        "num_groups": int(num_groups),
        "band_neuron_ends": None if band_ends_text is None else list(band_ends_text),
        "group_cumulative_ends_per_hidden_layer": None if group_ends is None else [[int(v) for v in ends] for ends in group_ends],
        "group_ids_per_hidden_layer": None if group_ids is None else [[int(v) for v in gids.tolist()] for gids in group_ids],
        "tear": int(tear_eff),
        "rf_clip_edges": clip_values if variant.base_model == "RF" else None,
        "rf_clip_edges_unit": "normalized_frequency_cyc_per_sample_nyquist_0p5" if variant.base_model == "RF" else None,
        "lif_clip_edges": clip_values if variant.base_model == "LIF" else None,
        "rf_reset_mode": str(rf_reset_mode),
        "lif_reset_mode": "subtractive_soft_reset_fixed_vth",
        "dynamics_init": dict(
            RFDenseLayer.dynamics_init_metadata() if variant.base_model == "RF" else LIFDenseLayer.dynamics_init_metadata()
        ),
        "readout_mode": str(normalize_readout_mode(readout_mode)),
        "recurrent": bool(recurrent),
        "final_membrane_output_behavior": "disable_output_spike_and_spike_triggered_reset",
        "spike_readout_output_behavior": "ordinary_spiking_output_layer",
        "output_layer_semantics": "same-base-neuron-output-layer_no_extra_nn_head",
    }
    return model, meta, str(variant.base_model)


# -----------------------------------------------------------------------------
# PSD helpers
# -----------------------------------------------------------------------------



_RECORD_ALIAS_KEYS: Dict[str, Dict[str, str]] = {
    "LIFDenseLayer": {"mem": "soma_state", "spk": "output", "spike": "output", "membrane": "soma_state"},
    "RFDenseLayer": {"x": "soma_state", "y": "state_y", "spk": "output", "spike": "output", "membrane": "soma_state"},
    "TCLIFDenseLayer": {"v1": "dendrite_state", "v2": "soma_state", "spk": "output", "spike": "output", "membrane": "soma_state"},
    "TSLIFDenseLayer": {"vd": "dendrite_state", "vs": "soma_state", "spk": "output", "spike": "output", "membrane": "soma_state"},
    "DHSNNDenseLayer": {"d_state": "dendrite_state", "mem": "soma_state", "spk": "output", "spike": "output", "membrane": "soma_state"},
    "MyDHSNNDenseLayer": {"d_state": "dendrite_state", "mem": "soma_state", "spk": "output", "spike": "output", "membrane": "soma_state"},
    "MyReverseDHSNNDenseLayer": {"d_state": "dendrite_state", "mem": "soma_state", "spk": "output", "spike": "output", "membrane": "soma_state"},
    "DRFDenseLayer": {"u": "dendrite_state", "v": "state_v", "pre_hist": "pre_hist", "V_th": "V_th", "spk": "output", "spike": "output", "membrane": "soma_state"},
    "MyDRFDenseLayer": {"u": "dendrite_state", "v": "state_v", "p_hist": "p_hist", "V_th": "V_th", "spk": "output", "spike": "output", "membrane": "soma_state"},
}



def _record_tensor_for_key(layer: nn.Module, rec: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
    if key in rec:
        return rec[key]
    cls = layer.__class__.__name__
    alias = _RECORD_ALIAS_KEYS.get(cls, {}).get(str(key))
    if alias is not None and alias in rec:
        return rec[alias]
    raise KeyError(f"missing record key {key!r} for layer class {cls}")



def _batched_signal_maps(sig: torch.Tensor, *, active_mask: Optional[np.ndarray]) -> torch.Tensor:
    if sig.dim() == 3:  # (B,T,N)
        out = sig.permute(0, 2, 1).contiguous()
    elif sig.dim() == 4:  # (B,T,N,D)
        B, T, N, D = sig.shape
        out = sig.permute(0, 2, 3, 1).reshape(B, N * D, T).contiguous()
    else:
        raise ValueError(f"unexpected record tensor shape for batched PSD maps: {tuple(sig.shape)}")
    if active_mask is not None:
        flat_mask = np.asarray(active_mask, dtype=float).reshape(-1)
        if flat_mask.size == int(out.shape[1]):
            mask_t = torch.as_tensor(flat_mask > 0.0, device=out.device, dtype=torch.bool)
            out = out[:, mask_t, :]
    return out



def _block_specs_from_variant_meta(
    hidden_layer_names: Sequence[str],
    variant_meta: Mapping[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    raw_group_ids = variant_meta.get("group_ids_per_hidden_layer") if isinstance(variant_meta, Mapping) else None
    if raw_group_ids is None:
        return {}

    block_specs: Dict[str, List[Dict[str, Any]]] = {}
    for layer_name, gids_raw in zip(hidden_layer_names, raw_group_ids):
        gids = np.asarray(gids_raw, dtype=np.int64).reshape(-1)
        if gids.size == 0:
            continue
        layer_specs: List[Dict[str, Any]] = []
        for block_offset, group_id in enumerate(sorted(int(v) for v in np.unique(gids)), start=1):
            idx = np.flatnonzero(gids == int(group_id)).astype(np.int64)
            if idx.size == 0:
                continue
            layer_specs.append(
                {
                    "block_name": f"block_{int(block_offset)}",
                    "group_id": int(group_id),
                    "neuron_indices_zero_based": idx,
                    "neuron_indices_one_based": idx + 1,
                }
            )
        if len(layer_specs) > 1:
            block_specs[str(layer_name)] = layer_specs
    return block_specs



def _batched_signal_maps_for_block(
    sig: torch.Tensor,
    *,
    neuron_indices_zero_based: Sequence[int],
    active_mask: Optional[np.ndarray],
) -> torch.Tensor:
    idx_np = np.asarray(list(neuron_indices_zero_based), dtype=np.int64).reshape(-1)
    if idx_np.size == 0:
        raise ValueError("block neuron index list must be non-empty")
    idx_t = torch.as_tensor(idx_np, device=sig.device, dtype=torch.long)

    block_mask: Optional[np.ndarray] = None
    if sig.dim() == 3:
        sig_block = sig.index_select(2, idx_t)
        if active_mask is not None:
            flat_mask = np.asarray(active_mask, dtype=float).reshape(-1)
            if flat_mask.size == int(sig.shape[2]):
                block_mask = flat_mask[idx_np]
        return _batched_signal_maps(sig_block, active_mask=block_mask)

    if sig.dim() == 4:
        sig_block = sig.index_select(2, idx_t)
        if active_mask is not None:
            flat_mask = np.asarray(active_mask, dtype=float).reshape(-1)
            num_neurons = int(sig.shape[2])
            num_branch = int(sig.shape[3])
            if flat_mask.size == num_neurons * num_branch:
                block_mask = flat_mask.reshape(num_neurons, num_branch)[idx_np, :].reshape(-1)
            elif flat_mask.size == num_neurons:
                block_mask = flat_mask[idx_np]
        return _batched_signal_maps(sig_block, active_mask=block_mask)

    raise ValueError(f"unexpected record tensor shape for block PSD maps: {tuple(sig.shape)}")



def _layer_effective_incoming_weight_matrix(layer: nn.Module) -> np.ndarray:
    if hasattr(layer, "effective_weight") and callable(getattr(layer, "effective_weight")):
        weight = layer.effective_weight()  # type: ignore[attr-defined]
        arr = weight.detach().cpu().numpy() if torch.is_tensor(weight) else np.asarray(weight)
        return np.asarray(arr, dtype=float)

    if hasattr(layer, "W_in") and torch.is_tensor(getattr(layer, "W_in")):
        arr = getattr(layer, "W_in").detach().cpu().numpy()
        return np.asarray(arr, dtype=float)

    if hasattr(layer, "fc") and hasattr(getattr(layer, "fc"), "weight"):
        weight = getattr(layer.fc, "weight")
        arr = weight.detach().cpu().numpy().astype(float)
        if hasattr(layer, "mask") and torch.is_tensor(getattr(layer, "mask")):
            mask = getattr(layer, "mask").detach().cpu().numpy().astype(float)
            if mask.shape == arr.shape:
                arr = arr * mask
        elif hasattr(layer, "input_group_mask") and torch.is_tensor(getattr(layer, "input_group_mask")):
            mask = getattr(layer, "input_group_mask").detach().cpu().numpy().astype(float)
            if mask.shape == arr.shape:
                arr = arr * mask
        input_dim = int(getattr(layer, "input_dim", arr.shape[1]))
        if arr.shape[1] > input_dim:
            arr = arr[:, :input_dim]
        output_dim = int(getattr(layer, "output_dim", arr.shape[0]))
        branch = int(getattr(layer, "branch", 1))
        if branch > 1 and arr.shape[0] == output_dim * branch:
            arr = arr.reshape(output_dim, branch, arr.shape[1]).sum(axis=1)
        return np.asarray(arr, dtype=float)

    if hasattr(layer, "W") and torch.is_tensor(getattr(layer, "W")):
        arr = getattr(layer, "W").detach().cpu().numpy().astype(float)
        input_dim = int(getattr(layer, "input_dim", arr.shape[1]))
        if arr.shape[1] > input_dim:
            arr = arr[:, :input_dim]
        output_dim = int(getattr(layer, "output_dim", arr.shape[0]))
        branch = int(getattr(layer, "branch", 1))
        if branch > 1 and arr.shape[0] == output_dim * branch:
            arr = arr.reshape(output_dim, branch, arr.shape[1]).sum(axis=1)
        return np.asarray(arr, dtype=float)

    raise KeyError(f"unable to extract incoming weight matrix from layer class {layer.__class__.__name__}")



def _density_line_from_values(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.asarray([0.0], dtype=float), np.asarray([0.0], dtype=float)
    if np.allclose(arr.min(), arr.max()):
        center = float(arr.reshape(-1)[0])
        return np.asarray([center], dtype=float), np.asarray([1.0], dtype=float)
    hist, edges = np.histogram(arr, bins=_histogram_bin_count(arr), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return np.asarray(centers, dtype=float), np.asarray(hist, dtype=float)



def _save_weight_density_plot(
    path: str,
    weight_matrix: np.ndarray,
    *,
    title: str,
    defer_plot: bool = False,
) -> None:
    centers, density = _density_line_from_values(np.asarray(weight_matrix, dtype=float))
    save_fn = save_deferred_line_plot if bool(defer_plot) else save_line_plot
    save_fn(
        path,
        {"pdf": density.tolist()},
        x=centers.tolist(),
        title=str(title),
        xlabel="incoming weight",
        ylabel="density",
    )



def _save_hidden_weight_visualizations(
    epoch_root: str,
    hidden_layer_names: Sequence[str],
    hidden_layers: Sequence[nn.Module],
    *,
    block_specs_by_layer: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
    epoch: int,
    defer_plots: bool = False,
) -> None:
    block_specs_lookup = {} if block_specs_by_layer is None else {str(k): list(v) for k, v in block_specs_by_layer.items()}
    for layer_name, layer in zip(hidden_layer_names, hidden_layers):
        try:
            full_weight = _layer_effective_incoming_weight_matrix(layer)
        except Exception:
            continue
        layer_root = os.path.join(epoch_root, str(layer_name))
        _save_weight_density_plot(
            os.path.join(layer_root, "w_plot.png"),
            full_weight,
            title=f"epoch {int(epoch)} {layer_name} incoming weight density",
            defer_plot=bool(defer_plots),
        )
        for block_spec in block_specs_lookup.get(str(layer_name), []):
            idx = np.asarray(block_spec["neuron_indices_zero_based"], dtype=np.int64).reshape(-1)
            if idx.size == 0:
                continue
            block_weight = full_weight[idx, :]
            block_root = os.path.join(layer_root, "block", str(block_spec["block_name"]))
            _save_weight_density_plot(
                os.path.join(block_root, "w_plot.png"),
                block_weight,
                title=f"epoch {int(epoch)} {layer_name} {block_spec['block_name']} incoming weight density",
                defer_plot=bool(defer_plots),
            )


def _materialize_probe_batches(dataset, split_scopes: Mapping[str, Any], *, pin_memory: bool) -> Dict[str, Dict[str, Any]]:
    batches: Dict[str, Dict[str, Any]] = {}
    for item in _comparison_items_for_split(split_scopes):
        x_batch_np, y_batch_np = _fetch_dataset_batch(dataset, item["indices"])
        x_batch_cpu = torch.from_numpy(np.asarray(x_batch_np, dtype=np.float32)).contiguous()
        if bool(pin_memory):
            x_batch_cpu = x_batch_cpu.pin_memory()
        batches[str(item["rel_dir"])] = {
            "indices": [int(v) for v in item["indices"]],
            "x_batch_cpu": x_batch_cpu,
            "y_batch_np": np.asarray(y_batch_np, dtype=np.int64),
            "scope_name": str(item["scope_name"]),
            "label": None if item["label"] is None else int(item["label"]),
        }
    return batches



def _probe_batch_to_device(batch_entry: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    x_batch_cpu = batch_entry["x_batch_cpu"]
    if not torch.is_tensor(x_batch_cpu):
        x_batch_cpu = torch.as_tensor(np.asarray(x_batch_cpu, dtype=np.float32))
    return x_batch_cpu.to(device=device, dtype=torch.float32, non_blocking=bool(getattr(x_batch_cpu, "is_pinned", lambda: False)()))




def _write_accuracy_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "test_loss",
        "test_acc",
        "stage",
        "ste_enabled",
        "hardened",
        "readout_mode",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})



def _append_accuracy_csv_row(path: str, row: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "test_loss",
        "test_acc",
        "stage",
        "ste_enabled",
        "hardened",
        "readout_mode",
    ]
    need_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if need_header:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})



def _save_accuracy_plot_from_csv(csv_path: str, png_path: str) -> None:
    epochs: List[float] = []
    train_acc: List[float] = []
    test_acc: List[float] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("epoch", "") == "":
                continue
            epochs.append(float(row["epoch"]))
            train_acc.append(float(row["train_acc"]))
            test_acc.append(float(row["test_acc"]))
    if len(epochs) == 0:
        return
    save_line_plot(
        png_path,
        {"train_acc": train_acc, "test_acc": test_acc},
        x=epochs,
        xlabel="epoch",
        ylabel="accuracy",
        title="train/test accuracy",
    )


def _write_probe_set_accuracy_txt(
    path: str,
    *,
    epoch: int,
    split_name: str,
    probe_type: str,
    label: Optional[int],
    correct: int,
    total: int,
    accuracy: float,
) -> None:
    lines = [
        f"epoch: {int(epoch)}",
        f"split: {str(split_name)}",
        f"probe_type: {str(probe_type)}",
        f"label: {'none' if label is None else int(label)}",
        f"correct: {int(correct)}",
        f"total: {int(total)}",
        f"accuracy: {float(accuracy):.6f}",
    ]
    save_text(path, "\n".join(lines) + "\n")


# -----------------------------------------------------------------------------
# Attenuation statistics
# -----------------------------------------------------------------------------


def _layer_decay_vectors(layer: nn.Module) -> Dict[str, np.ndarray]:
    if isinstance(layer, LIFDenseLayer):
        alpha = layer.alpha().detach().cpu().numpy().astype(float).reshape(-1)
        return {"alpha": alpha}
    if isinstance(layer, RFDenseLayer):
        rho = layer.rho().detach().cpu().numpy().astype(float).reshape(-1)
        f_cyc_per_sample = layer.f_cyc_per_sample().detach().cpu().numpy().astype(float).reshape(-1)
        return {
            "rho": rho.reshape(-1),
            "f_cyc_per_sample": f_cyc_per_sample.reshape(-1),
        }
    return {}



def _summary_stats(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "variance": 0.0,
            "std": 0.0,
            "min": 0.0,
            "q25": 0.0,
            "q50": 0.0,
            "q75": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "variance": float(np.var(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q50": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
    }


def _write_decay_stats_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["parameter", "count", "mean", "variance", "std", "min", "q25", "q50", "q75", "max"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _histogram_bin_count(values: np.ndarray) -> int:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size <= 1:
        return 1
    return int(max(5, min(60, int(np.ceil(np.sqrt(arr.size))))))


def _save_decay_stat_group(
    base_dir: str,
    *,
    title_prefix: str,
    stats_by_name: Mapping[str, Dict[str, float]],
    values_by_name: Mapping[str, np.ndarray],
    defer_plots: bool = False,
) -> None:
    os.makedirs(base_dir, exist_ok=True)
    csv_rows = []
    for param_name, stats in stats_by_name.items():
        labels = ["mean", "variance", "q25", "q50", "q75"]
        stat_values = [float(stats[label]) for label in labels]
        raw_values = np.asarray(values_by_name[param_name], dtype=float).reshape(-1)
        bar_path = os.path.join(base_dir, f"{param_name}_stats_bar.png")
        hist_path = os.path.join(base_dir, f"{param_name}_value_hist_bar.png")
        if bool(defer_plots):
            save_deferred_bar_plot(
                bar_path,
                labels,
                stat_values,
                title=f"{title_prefix} {param_name} statistics",
                xlabel="statistic",
                ylabel=str(param_name),
                rotation=0.0,
            )
            save_deferred_hist_bar(
                hist_path,
                raw_values,
                bins=_histogram_bin_count(raw_values),
                title=f"{title_prefix} {param_name} value histogram",
                xlabel=str(param_name),
                ylabel="neuron count",
            )
        else:
            save_bar_plot(
                bar_path,
                labels,
                stat_values,
                title=f"{title_prefix} {param_name} statistics",
                xlabel="statistic",
                ylabel=str(param_name),
                rotation=0.0,
            )
            save_hist_bar(
                hist_path,
                raw_values,
                bins=_histogram_bin_count(raw_values),
                title=f"{title_prefix} {param_name} value histogram",
                xlabel=str(param_name),
                ylabel="neuron count",
            )
        csv_rows.append({"parameter": str(param_name), **dict(stats)})
    _write_decay_stats_csv(os.path.join(base_dir, "summary_stats.csv"), csv_rows)


def _save_decay_statistics(
    run_root: str,
    epoch: int,
    layer_names: Sequence[str],
    layers: Sequence[nn.Module],
    *,
    stats_root: Optional[str] = None,
    summary_copy_paths: Optional[Sequence[str]] = None,
    defer_plots: bool = False,
) -> None:
    if stats_root is None:
        epoch_dir = os.path.join(run_root, f"epoch_{int(epoch):04d}")
        stats_root = os.path.join(epoch_dir, "attenuation_stats")
        summary_paths: List[str] = [
            os.path.join(stats_root, "all_layers_summary.csv"),
            os.path.join(epoch_dir, "all_layers_summary.csv"),
        ]
    else:
        stats_root = os.path.abspath(str(stats_root))
        summary_paths = [os.path.join(stats_root, "all_layers_summary.csv")]

    if summary_copy_paths is not None:
        for raw_path in summary_copy_paths:
            csv_path = os.path.abspath(str(raw_path))
            if csv_path not in summary_paths:
                summary_paths.append(csv_path)

    layer_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    model_accumulator: Dict[str, List[np.ndarray]] = {}

    for layer_name, layer in zip(layer_names, layers):
        vectors = _layer_decay_vectors(layer)
        if len(vectors) == 0:
            continue
        stats_by_name = {name: _summary_stats(values) for name, values in vectors.items()}
        layer_stats[str(layer_name)] = stats_by_name
        for name, values in vectors.items():
            model_accumulator.setdefault(str(name), []).append(np.asarray(values, dtype=float).reshape(-1))
        _save_decay_stat_group(
            os.path.join(stats_root, "layers", str(layer_name)),
            title_prefix=f"epoch {int(epoch)} {layer_name}",
            stats_by_name=stats_by_name,
            values_by_name=vectors,
            defer_plots=bool(defer_plots),
        )

    if len(model_accumulator) == 0:
        return
    model_stats = {
        name: _summary_stats(np.concatenate(chunks, axis=0) if len(chunks) > 0 else np.zeros((0,), dtype=float))
        for name, chunks in model_accumulator.items()
    }
    _save_decay_stat_group(
        os.path.join(stats_root, "model"),
        title_prefix=f"epoch {int(epoch)} model",
        stats_by_name=model_stats,
        values_by_name={
            name: np.concatenate(chunks, axis=0) if len(chunks) > 0 else np.zeros((0,), dtype=float)
            for name, chunks in model_accumulator.items()
        },
        defer_plots=bool(defer_plots),
    )

    flat_rows = []
    for layer_name, stats_by_name in layer_stats.items():
        for param_name, stats in stats_by_name.items():
            flat_rows.append({"scope": "layer", "layer": str(layer_name), "parameter": str(param_name), **dict(stats)})
    for param_name, stats in model_stats.items():
        flat_rows.append({"scope": "model", "layer": "model", "parameter": str(param_name), **dict(stats)})
    fieldnames = ["scope", "layer", "parameter", "count", "mean", "variance", "std", "min", "q25", "q50", "q75", "max"]
    for csv_path in summary_paths:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in flat_rows:
                writer.writerow({key: row.get(key) for key in fieldnames})


# -----------------------------------------------------------------------------
# Signal extraction and epoch artifact saving
# -----------------------------------------------------------------------------


def _maps_from_sample_result(
    sample_result: Mapping[str, Any],
    layers: Sequence[nn.Module],
    layer_names: Sequence[str],
    *,
    membrane_key: str,
    spike_key: str,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "input": None,
        "layers": OrderedDict(),
    }
    out["input"] = input_time_matrix(sample_result["input"])

    for lname, layer, rec in zip(layer_names, layers, sample_result["recs"]):
        mask = maybe_active_mask(layer)
        spike_value = rec.get(spike_key)
        if spike_value is None:
            spike_alias = _RECORD_ALIAS_KEYS.get(layer.__class__.__name__, {}).get(str(spike_key))
            if spike_alias is not None:
                spike_value = rec.get(str(spike_alias))
        if spike_value is None:
            raise KeyError(f"missing spike key {spike_key!r} in layer {lname}")

        membrane_value = rec.get(membrane_key)
        if membrane_value is None:
            membrane_alias = _RECORD_ALIAS_KEYS.get(layer.__class__.__name__, {}).get(str(membrane_key))
            if membrane_alias is not None:
                membrane_value = rec.get(str(membrane_alias))
        if membrane_value is None:
            raise KeyError(f"missing membrane key {membrane_key!r} in layer {lname}")

        spike_raw = layer_signal_matrix(spike_value, active_mask=mask)
        mem_raw = layer_signal_matrix(membrane_value, active_mask=mask)
        out["layers"][str(lname)] = {
            "spike": spike_raw,
            "membrane": mem_raw,
        }

    output_rec = dict(sample_result.get("output_rec", {}))

    output_spike_value = output_rec.get("output")
    if output_spike_value is None:
        for alt_key in (str(spike_key), "spk", "spike"):
            output_spike_value = output_rec.get(str(alt_key))
            if output_spike_value is not None:
                break
    if output_spike_value is None:
        raise KeyError("missing output spike record in output layer")

    output_membrane_value = output_rec.get("soma_state")
    if output_membrane_value is None:
        for alt_key in (str(membrane_key), "membrane"):
            output_membrane_value = output_rec.get(str(alt_key))
            if output_membrane_value is not None:
                break
    if output_membrane_value is None:
        raise KeyError("missing output membrane record in output layer")

    out["layers"]["output"] = {
        "spike": layer_signal_matrix(output_spike_value, active_mask=None),
        "membrane": layer_signal_matrix(output_membrane_value, active_mask=None),
    }
    return out



def _save_time_domain_plots(
    base_dir: str,
    maps_t: torch.Tensor,
    *,
    title_prefix: str,
) -> None:
    """Save time-domain plots for one signal family.

    The saved heatmap is the probe-set mean over samples, so for spikes it is an
    empirical firing-rate map in [0, 1]. The accompanying line plot is the
    element-wise mean over the same heatmap.
    """
    arr = maps_t.detach().cpu().numpy().astype(np.float32, copy=False)
    if arr.ndim != 3:
        raise ValueError(f"maps_t must be (S,R,T), got {arr.shape}")
    mean_heatmap = np.asarray(arr.mean(axis=0), dtype=np.float32)
    element_mean = np.asarray(mean_heatmap.mean(axis=0), dtype=np.float32)
    timesteps = np.arange(int(mean_heatmap.shape[1]), dtype=np.float32)
    save_heatmap_plot(
        os.path.join(base_dir, "time_domain_heatmap.png"),
        mean_heatmap,
        title=f"{title_prefix} time-domain heatmap (probe-set mean)",
        xlabel="timestep",
        ylabel="element index",
        origin="lower",
    )
    save_line_plot(
        os.path.join(base_dir, "time_domain_element_mean.png"),
        {"element_mean": element_mean},
        x=timesteps,
        title=f"{title_prefix} time-domain element mean",
        xlabel="timestep",
        ylabel="mean over elements",
    )


def _save_epoch_analysis(
    run_root: str,
    epoch: int,
    model,
    hidden_layers: Sequence[nn.Module],
    hidden_layer_names: Sequence[str],
    fixed_scopes: Mapping[str, Any],
    *,
    membrane_key: str,
    spike_key: str,
    nperseg_eff: int,
    spectrogram_band_ranges,
    noverlap_eff: int,
    window_fn: str,
    periodogram_band_ranges,
    userbin_centers_np: np.ndarray,
    device: torch.device,
    probe_batches_by_split: Mapping[str, Mapping[str, Mapping[str, Any]]],
    criterion: Optional[nn.Module] = None,
    save_signal_psd_plots: bool = True,
    artifact_root: Optional[str] = None,
    block_specs_by_layer: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
    probe_reference_payloads_by_split: Optional[Mapping[str, Mapping[str, Mapping[str, Any]]]] = None,
    derivative_tracker: Optional[Dict[str, Any]] = None,
    defer_signal_psd_plots: bool = False,
) -> None:
    epoch_root = os.path.abspath(str(artifact_root)) if artifact_root is not None else os.path.join(run_root, f"epoch_{int(epoch):04d}")
    os.makedirs(epoch_root, exist_ok=True)
    psd_bundle_saver = save_deferred_psd_bundle if bool(defer_signal_psd_plots) else save_psd_bundle
    for split_name in ("train", "test"):
        items = _comparison_items_for_split(fixed_scopes[split_name])
        split_batches = probe_batches_by_split[split_name]
        for item in items:
            batch_entry = split_batches[str(item["rel_dir"])]
            x_batch_t = _probe_batch_to_device(batch_entry, device)
            with torch.no_grad():
                logits, hidden_recs, out_rec = model.forward_with_records(x_batch_t)

            if criterion is not None and bool(getattr(criterion, "requires_output_record", False)):
                analysis = criterion.analyze_output_record(out_rec)
                preds = criterion.predictions_from_analysis(analysis).detach().cpu().numpy().astype(np.int64)
            else:
                preds = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
            targets = np.asarray(batch_entry["y_batch_np"], dtype=np.int64).reshape(-1)
            total = int(targets.size)
            correct = int(np.sum(preds == targets))
            accuracy = 0.0 if total <= 0 else float(correct) / float(total)
            scope_root = os.path.join(epoch_root, split_name, item["rel_dir"])
            _write_probe_set_accuracy_txt(
                os.path.join(scope_root, "probe_set_accuracy.txt"),
                epoch=int(epoch),
                split_name=str(split_name),
                probe_type=str(item["scope_name"]),
                label=None if item["label"] is None else int(item["label"]),
                correct=int(correct),
                total=int(total),
                accuracy=float(accuracy),
            )

            if not bool(save_signal_psd_plots):
                continue

            reference_payload = None
            if probe_reference_payloads_by_split is not None:
                reference_payload = probe_reference_payloads_by_split[str(split_name)].get(str(item["rel_dir"]))

            block_specs_lookup = {} if block_specs_by_layer is None else {str(k): list(v) for k, v in block_specs_by_layer.items()}
            for layer_name, layer, rec in zip(hidden_layer_names, hidden_layers, hidden_recs):
                mask = maybe_active_mask(layer)
                family_to_tensor = {
                    "spike": _record_tensor_for_key(layer, rec, str(spike_key)),
                    "membrane": _record_tensor_for_key(layer, rec, str(membrane_key)),
                }
                for family, sig_t in family_to_tensor.items():
                    maps_t = _batched_signal_maps(sig_t, active_mask=mask)
                    combined = combined_exact_psd_payload_from_maps_torch(
                        maps_t,
                        periodogram_band_ranges=periodogram_band_ranges,
                        spectrogram_band_ranges=spectrogram_band_ranges,
                        nperseg_eff=int(nperseg_eff),
                        noverlap_eff=int(noverlap_eff),
                        window_fn=str(window_fn),
                    )
                    out_dir = os.path.join(epoch_root, split_name, item["rel_dir"], str(layer_name), str(family))
                    psd_bundle_saver(
                        out_dir,
                        payload=combined,
                        userbin_centers_np=np.asarray(userbin_centers_np, dtype=float),
                        title_prefix=f"epoch {int(epoch)} {split_name} {item['rel_dir'].replace(os.sep, ' / ')} {layer_name} {family}",
                        signal_scope=f"{layer_name}/{family}",
                        epoch=int(epoch),
                        save_db_plots=True,
                    )
                    _save_time_domain_plots(
                        out_dir,
                        maps_t,
                        title_prefix=f"epoch {int(epoch)} {split_name} {item['rel_dir'].replace(os.sep, ' / ')} {layer_name} {family}",
                    )
                    if derivative_tracker is not None and reference_payload is not None:
                        _update_derivative_tracker(
                            derivative_tracker,
                            epoch=int(epoch),
                            split_name=str(split_name),
                            rel_dir=str(item['rel_dir']),
                            layer_name=str(layer_name),
                            family=str(family),
                            reference_payload=reference_payload,
                            current_payload=combined,
                        )
                    for block_spec in block_specs_lookup.get(str(layer_name), []):
                        block_maps_t = _batched_signal_maps_for_block(
                            sig_t,
                            neuron_indices_zero_based=block_spec["neuron_indices_zero_based"],
                            active_mask=mask,
                        )
                        block_combined = combined_exact_psd_payload_from_maps_torch(
                            block_maps_t,
                            periodogram_band_ranges=periodogram_band_ranges,
                            spectrogram_band_ranges=spectrogram_band_ranges,
                            nperseg_eff=int(nperseg_eff),
                            noverlap_eff=int(noverlap_eff),
                            window_fn=str(window_fn),
                        )
                        block_out_dir = os.path.join(
                            epoch_root,
                            split_name,
                            item["rel_dir"],
                            str(layer_name),
                            "block",
                            str(block_spec["block_name"]),
                            str(family),
                        )
                        psd_bundle_saver(
                            block_out_dir,
                            payload=block_combined,
                            userbin_centers_np=np.asarray(userbin_centers_np, dtype=float),
                            title_prefix=(
                                f"epoch {int(epoch)} {split_name} {item['rel_dir'].replace(os.sep, ' / ')} "
                                f"{layer_name} {block_spec['block_name']} {family}"
                            ),
                            signal_scope=f"{layer_name}/block/{block_spec['block_name']}/{family}",
                            epoch=int(epoch),
                            save_db_plots=True,
                        )

            output_family_to_maps = {
                "spike": _batched_signal_maps(_record_tensor_for_key(model.output_layer, out_rec, "output"), active_mask=None),
                "membrane": _batched_signal_maps(_record_tensor_for_key(model.output_layer, out_rec, str(membrane_key)), active_mask=None),
            }
            for family, maps_t in output_family_to_maps.items():
                combined = combined_exact_psd_payload_from_maps_torch(
                    maps_t,
                    periodogram_band_ranges=periodogram_band_ranges,
                    spectrogram_band_ranges=spectrogram_band_ranges,
                    nperseg_eff=int(nperseg_eff),
                    noverlap_eff=int(noverlap_eff),
                    window_fn=str(window_fn),
                )
                out_dir = os.path.join(epoch_root, split_name, item["rel_dir"], "output", str(family))
                psd_bundle_saver(
                    out_dir,
                    payload=combined,
                    userbin_centers_np=np.asarray(userbin_centers_np, dtype=float),
                    title_prefix=f"epoch {int(epoch)} {split_name} {item['rel_dir'].replace(os.sep, ' / ')} output {family}",
                    signal_scope=f"output/{family}",
                    epoch=int(epoch),
                    save_db_plots=True,
                )
                _save_time_domain_plots(
                    out_dir,
                    maps_t,
                    title_prefix=f"epoch {int(epoch)} {split_name} {item['rel_dir'].replace(os.sep, ' / ')} output {family}",
                )
                if derivative_tracker is not None and reference_payload is not None:
                    _update_derivative_tracker(
                        derivative_tracker,
                        epoch=int(epoch),
                        split_name=str(split_name),
                        rel_dir=str(item['rel_dir']),
                        layer_name='output',
                        family=str(family),
                        reference_payload=reference_payload,
                        current_payload=combined,
                    )



_DERIVATIVE_DB_EPS = 1.0e-12

_DERIVATIVE_METRIC_SPECS: tuple[dict[str, Any], ...] = (
    {"metric_name": "mean_psd_waveform_exact_raw", "payload_key": "set_mean_psd_exact_raw", "kind": "line", "db": False},
    {"metric_name": "mean_psd_waveform_exact_centered", "payload_key": "set_mean_psd_exact_centered", "kind": "line", "db": False},
    {"metric_name": "mean_spectrogram_exact_raw", "payload_key": "set_mean_spectrogram_exact_raw", "kind": "heatmap", "db": False},
    {"metric_name": "mean_spectrogram_exact_centered", "payload_key": "set_mean_spectrogram_exact_centered", "kind": "heatmap", "db": False},
    {"metric_name": "mean_psd_waveform_exact_raw_db", "payload_key": "set_mean_psd_exact_raw", "kind": "line", "db": True},
    {"metric_name": "mean_psd_waveform_exact_centered_db", "payload_key": "set_mean_psd_exact_centered", "kind": "line", "db": True},
    {"metric_name": "mean_spectrogram_exact_raw_db", "payload_key": "set_mean_spectrogram_exact_raw", "kind": "heatmap", "db": True},
    {"metric_name": "mean_spectrogram_exact_centered_db", "payload_key": "set_mean_spectrogram_exact_centered", "kind": "heatmap", "db": True},
)

_FIXED_BRANCH_CANONICAL_MODELS = {"DH_SNN", "D_RF"}
_VARIABLE_BRANCH_CANONICAL_MODELS = {"my_DH_SNN", "my_R_DH_SNN", "my_D_RF"}
_RECURRENT_ENABLED_CANONICAL_MODELS = {"LIF", "RF", "TC_LIF", "TS_LIF", "DH_SNN"}


def _parse_model_token_recurrence_and_branch(model_token: str) -> tuple[str, bool, Optional[int]]:
    token = str(model_token).strip()
    recurrent = False
    fixed_branch_override: Optional[int] = None

    rec_match = re.match(r"^(.*?)(?:_R(?:_([0-9]+))?)$", token, flags=re.IGNORECASE)
    if rec_match is not None:
        token = str(rec_match.group(1))
        recurrent = True
        if rec_match.group(2) is not None:
            fixed_branch_override = int(rec_match.group(2))
            if int(fixed_branch_override) < 1:
                raise ValueError(f"branch count parsed from {model_token!r} must be >= 1, got {fixed_branch_override}")

    variant = parse_psd_model_variant(token)
    if variant is not None:
        canonical = resolve_model_name(str(variant.base_model))
        if bool(recurrent) and canonical not in _RECURRENT_ENABLED_CANONICAL_MODELS:
            raise ValueError(f"recurrent suffix _R is not supported for {model_token!r}")
        if fixed_branch_override is not None:
            raise ValueError(
                f"model token suffix _R_<int> is not supported for PSD clip/structure variants: {model_token!r}"
            )
        return token, bool(recurrent), None

    match = re.match(r"^(.*)_([0-9]+)$", token)
    if match is not None:
        token = str(match.group(1))
        fixed_branch_override = int(match.group(2))

    canonical = resolve_model_name(token)
    if bool(recurrent) and canonical not in _RECURRENT_ENABLED_CANONICAL_MODELS:
        raise ValueError(f"recurrent suffix _R is not supported for {model_token!r} -> canonical {canonical}.")
    if fixed_branch_override is not None and canonical not in _FIXED_BRANCH_CANONICAL_MODELS:
        raise ValueError(
            f"model token suffix _<int> is reserved for fixed-branch dendritic models. "
            f"Got {model_token!r} -> canonical {canonical}."
        )
    if fixed_branch_override is not None and int(fixed_branch_override) < 1:
        raise ValueError(f"branch count parsed from {model_token!r} must be >= 1, got {fixed_branch_override}")
    return token, bool(recurrent), None if fixed_branch_override is None else int(fixed_branch_override)


def _probe_reference_payloads_for_split(
    run_root: str,
    split_name: str,
    split_scopes: Mapping[str, Any],
    *,
    periodogram_band_ranges,
    spectrogram_band_ranges,
    nperseg_eff: int,
    noverlap_eff: int,
    window_fn: str,
    userbin_centers_np: np.ndarray,
    device: torch.device,
    probe_batches: Mapping[str, Mapping[str, Any]],
    save_plots: bool,
    defer_plots: bool = False,
) -> Dict[str, Dict[str, Any]]:
    payloads: Dict[str, Dict[str, Any]] = {}
    base_root = os.path.join(run_root, "probe_set_reference", str(split_name))
    if bool(save_plots):
        os.makedirs(base_root, exist_ok=True)
    psd_bundle_saver = save_deferred_psd_bundle if bool(defer_plots) else save_psd_bundle
    for item in _comparison_items_for_split(split_scopes):
        rel_dir = str(item["rel_dir"])
        batch_entry = probe_batches.get(rel_dir)
        if batch_entry is None:
            raise KeyError(f"missing prefetched probe batch for {split_name}/{rel_dir}")
        x_maps_t = _probe_batch_to_device(batch_entry, device).permute(0, 2, 1).contiguous()
        combined = combined_exact_psd_payload_from_maps_torch(
            x_maps_t,
            periodogram_band_ranges=periodogram_band_ranges,
            spectrogram_band_ranges=spectrogram_band_ranges,
            nperseg_eff=int(nperseg_eff),
            noverlap_eff=int(noverlap_eff),
            window_fn=str(window_fn),
        )
        payloads[rel_dir] = combined
        if bool(save_plots):
            out_dir = os.path.join(base_root, rel_dir, "input")
            if not _psd_bundle_is_complete(out_dir):
                psd_bundle_saver(
                    out_dir,
                    payload=combined,
                    userbin_centers_np=np.asarray(userbin_centers_np, dtype=float),
                    title_prefix=f"{split_name} {rel_dir.replace(os.sep, ' / ')} input reference",
                    signal_scope="input",
                    epoch=None,
                    save_db_plots=True,
                )
    return payloads


def _payload_metric_array(payload: Mapping[str, Any], payload_key: str, *, use_db: bool) -> np.ndarray:
    arr = np.asarray(payload[payload_key], dtype=float)
    if bool(use_db):
        arr = 10.0 * np.log10(np.maximum(arr, 0.0) + float(_DERIVATIVE_DB_EPS))
    return arr


def _derivative_semi_metric_1d(ref: np.ndarray, cur: np.ndarray) -> float:
    ref_1d = np.asarray(ref, dtype=float).reshape(-1)
    cur_1d = np.asarray(cur, dtype=float).reshape(-1)
    if ref_1d.shape != cur_1d.shape:
        raise ValueError(f'derivative semi-metric requires matching 1D shapes, got {ref_1d.shape} vs {cur_1d.shape}')
    if ref_1d.size <= 1:
        return 0.0
    diff = np.diff(cur_1d) - np.diff(ref_1d)
    return float(np.sqrt(np.sum(diff * diff)))


def _derivative_semi_metric_rows(ref: np.ndarray, cur: np.ndarray) -> np.ndarray:
    ref_2d = np.asarray(ref, dtype=float)
    cur_2d = np.asarray(cur, dtype=float)
    if ref_2d.shape != cur_2d.shape:
        raise ValueError(f'derivative semi-metric requires matching 2D shapes, got {ref_2d.shape} vs {cur_2d.shape}')
    if ref_2d.ndim != 2:
        raise ValueError(f'row-wise derivative semi-metric expects 2D arrays, got {ref_2d.shape}')
    if ref_2d.shape[1] <= 1:
        return np.zeros((ref_2d.shape[0],), dtype=float)
    diff = np.diff(cur_2d, axis=1) - np.diff(ref_2d, axis=1)
    return np.sqrt(np.sum(diff * diff, axis=1)).astype(float)


def _update_derivative_tracker(
    tracker: Dict[str, Any],
    *,
    epoch: int,
    split_name: str,
    rel_dir: str,
    layer_name: str,
    family: str,
    reference_payload: Mapping[str, Any],
    current_payload: Mapping[str, Any],
) -> None:
    key = (str(split_name), str(rel_dir), str(layer_name), str(family))
    bucket = tracker.setdefault(key, {"line": OrderedDict(), "heatmap": OrderedDict()})
    for spec in _DERIVATIVE_METRIC_SPECS:
        ref_arr = _payload_metric_array(reference_payload, str(spec["payload_key"]), use_db=bool(spec["db"]))
        cur_arr = _payload_metric_array(current_payload, str(spec["payload_key"]), use_db=bool(spec["db"]))
        metric_name = str(spec["metric_name"])
        if str(spec["kind"]) == "line":
            value = _derivative_semi_metric_1d(ref_arr, cur_arr)
            bucket["line"].setdefault(metric_name, []).append((int(epoch), float(value)))
        elif str(spec["kind"]) == "heatmap":
            values = _derivative_semi_metric_rows(ref_arr, cur_arr)
            bucket["heatmap"].setdefault(metric_name, []).append((int(epoch), values.astype(float)))
        else:
            raise ValueError(f"unknown derivative metric kind: {spec['kind']}")


def _save_derivative_tracker(root: str, tracker: Mapping[Any, Any]) -> None:
    if len(tracker) == 0:
        return
    os.makedirs(root, exist_ok=True)
    summary: Dict[str, Any] = {
        "derivative_definition": "d(u,v)=sqrt(sum_k ((u[k+1]-u[k])-(v[k+1]-v[k]))^2))",
        "time_step_note": "finite differences use a unit step, so explicit delta_t is omitted",
        "tracked_mean_plot_families_only": True,
        "tracked_metric_names": [str(spec["metric_name"]) for spec in _DERIVATIVE_METRIC_SPECS],
    }
    save_json(os.path.join(root, 'summary.json'), summary)
    for key, bucket in tracker.items():
        split_name, rel_dir, layer_name, family = key
        base_dir = os.path.join(root, str(split_name), str(rel_dir), str(layer_name), str(family))
        os.makedirs(base_dir, exist_ok=True)
        line_bucket = bucket.get("line", {})
        for metric_name, points in line_bucket.items():
            ordered = sorted(points, key=lambda item: int(item[0]))
            epochs = [int(epoch) for epoch, _ in ordered]
            values = [float(value) for _, value in ordered]
            save_line_plot(
                os.path.join(base_dir, f'{metric_name}.png'),
                {'derivative_semi_metric': values},
                x=epochs,
                title=f'{split_name} {rel_dir.replace(os.sep, " / ")} {layer_name} {family} {metric_name}',
                xlabel='selected epoch',
                ylabel='derivative semi-metric',
            )
            save_json(
                os.path.join(base_dir, f'{metric_name}.json'),
                {'epochs': epochs, 'values': values, 'layer_name': str(layer_name), 'family': str(family)},
            )
        heatmap_bucket = bucket.get("heatmap", {})
        for metric_name, points in heatmap_bucket.items():
            ordered = sorted(points, key=lambda item: int(item[0]))
            epochs = [int(epoch) for epoch, _ in ordered]
            mats = [np.asarray(value, dtype=float).reshape(-1) for _, value in ordered]
            if len(mats) == 0:
                continue
            heatmap = np.stack(mats, axis=1)
            save_heatmap_plot(
                os.path.join(base_dir, f'{metric_name}.png'),
                heatmap,
                title=f'{split_name} {rel_dir.replace(os.sep, " / ")} {layer_name} {family} {metric_name}',
                xlabel='selected epoch',
                ylabel='spectrogram frequency bin',
                origin='lower',
                x_tick_labels=[str(epoch) for epoch in epochs],
            )
            save_json(
                os.path.join(base_dir, f'{metric_name}.json'),
                {
                    'epochs': epochs,
                    'values': [np.asarray(value, dtype=float).reshape(-1).tolist() for _, value in ordered],
                    'matrix_shape': list(heatmap.shape),
                    'layer_name': str(layer_name),
                    'family': str(family),
                },
            )


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------


def _first_spike_loss_hparams(dataset_name: str) -> Dict[str, float | int]:
    name = normalize_dataset_name(dataset_name)
    alpha_fs = 0.1 if name == 'dvsgesture' else 0.2
    return {
        'alpha_fs': float(alpha_fs),
        'D': 16,
        'A': 200.0,
        'lambda_treg': 0.01,
        'beta_treg': 0.02,
    }



def run_psd_analysis(
    *,
    dataset: str,
    model: str,
    out_root: str,
    data_root: str,
    hidden: Sequence[int],
    epochs: int = 50,
    soft_mask_epochs: Optional[int] = None,
    stabilize_epochs: int = 0,
    ste_epochs: int = 0,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    weight_decay_dend_soma: Optional[float] = None,
    seed: int = 0,
    S_min: float = 1.0,
    S_max: float = 8.0,
    th_len: int = 4,
    v_th: float = 1.0,
    v_pre: float = 1.0,
    num_workers: int = 4,
    download: bool = False,
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
    lambda_ortho: float = 0.0,
    lambda_s: float = 0.0,
    same_label_n_per_label: int = 4,
    balanced_global_n_per_label: int = 4,
    probe_plot: bool = False,
    plot_epochs: Optional[Sequence[int]] = None,
    psd_window: int = 64,
    psd_overlap: int = 32,
    window_fn: str = 'hann',
    userbin_edges: Optional[Sequence[float]] = None,
    rf_reset_mode: str = 'no_reset',
    w_clip_edges: Optional[Sequence[float]] = None,
    alpha_clip_edges: Optional[Sequence[float]] = None,
    band_neuron_ends: Optional[Sequence[str]] = None,
    tear: int = 1,
    readout_mode: str = 'final_membrane',
    exp_name: Optional[str] = None,
    timestamp: Optional[str] = None,
    device: str = 'auto',
) -> str:
    dataset_name = normalize_dataset_name(dataset)
    # Legacy compatibility only: persistent input probe-reference saving moved to
    # dataset_psd. psd_analysis now always keeps probe-reference payloads internal.
    _ = bool(probe_plot)
    hidden = [int(h) for h in hidden]
    if len(hidden) < 1:
        raise ValueError('hidden must contain at least one hidden layer')
    if int(epochs) < 0:
        raise ValueError(f'epochs must be >= 0, got {epochs}')
    signal_psd_plot_epochs = _normalize_plot_epochs(plot_epochs, int(epochs))
    signal_psd_plot_epoch_set = {int(v) for v in signal_psd_plot_epochs}
    readout_mode_key = normalize_readout_mode(readout_mode)
    out_root_abs = require_absolute_path(out_root, kind='result_root_abs', create=True)
    data_root_abs = require_absolute_path(data_root, kind='data_root_abs', create=True)
    dev = get_device(device)
    set_seed(int(seed))

    configure_plot_writer(
        workers=max(1, _env_int('PSD_PLOT_WRITER_WORKERS', 4)),
        queue_maxsize=max(4, _env_int('PSD_PLOT_QUEUE_MAXSIZE', 64)),
        start_method=str(os.environ.get('PSD_PLOT_WRITER_START_METHOD', 'fork' if os.name == 'posix' else 'spawn')),
        dpi=max(72, _env_int('PSD_PLOT_WRITER_DPI', 120)),
        skip_existing=_env_bool('PSD_PLOT_SKIP_EXISTING', True),
    )

    bundle = build_dataset_bundle(
        dataset_name=dataset_name,
        data_root=data_root_abs,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        download=bool(download),
        seed=int(seed),
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
    train_loader = bundle.train_loader
    test_loader = bundle.test_loader
    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset
    num_classes = int(bundle.num_classes)
    input_dim = int(bundle.input_dim)
    T = int(bundle.T)

    base_model_token, recurrent_mode, fixed_branch_override = _parse_model_token_recurrence_and_branch(model)
    variant = parse_psd_model_variant(base_model_token)
    variant_meta: Dict[str, Any] = {}
    if variant is not None:
        if fixed_branch_override is not None:
            raise ValueError(f'fixed-branch suffix is not supported for PSD clip/structure variants: {model}')
        net, variant_meta, canonical_model = _build_variant_classifier(
            model_token=str(base_model_token),
            input_dim=int(input_dim),
            hidden=hidden,
            num_classes=int(num_classes),
            v_th=float(v_th),
            rf_reset_mode=str(rf_reset_mode),
            rf_clip_edges=None if w_clip_edges is None else [float(v) for v in w_clip_edges],
            lif_clip_edges=None if alpha_clip_edges is None else [float(v) for v in alpha_clip_edges],
            band_neuron_ends=None if band_neuron_ends is None else [str(v) for v in band_neuron_ends],
            tear=int(tear),
            readout_mode=str(readout_mode_key),
            recurrent=bool(recurrent_mode),
        )
        net = net.to(dev)
        spec = get_model_spec(str(canonical_model))
        model_token = str(variant_meta['model_token'])
        if bool(recurrent_mode):
            model_token = f"{model_token}_R"
        effective_branch = None
    else:
        canonical_model = resolve_model_name(base_model_token)
        spec = get_model_spec(canonical_model)
        if canonical_model in _VARIABLE_BRANCH_CANONICAL_MODELS and fixed_branch_override is not None:
            raise ValueError(
                f'{model!r} uses a variable-branch neuron family. Use S_min/S_max instead of a _<int> branch suffix.'
            )
        if canonical_model in _FIXED_BRANCH_CANONICAL_MODELS:
            effective_branch = int(fixed_branch_override if fixed_branch_override is not None else max(1, int(np.ceil(float(S_max)))))
        else:
            effective_branch = 1
        cfg = SNNConfig(
            model_name=str(spec.builder_name),
            input_dim=int(input_dim),
            hidden_dim=int(hidden[0]),
            num_classes=int(num_classes),
            branch=int(effective_branch),
            S_min=float(S_min),
            S_max=float(S_max),
            th_len=int(th_len),
            v_th=float(v_th),
            v_pre=float(v_pre),
            rf_reset_mode=str(rf_reset_mode),
            readout_mode=str(readout_mode_key),
            recurrent=bool(recurrent_mode),
        )
        net = build_common_classifier(
            model_name=str(spec.builder_name),
            input_dim=int(input_dim),
            hidden_dims=hidden,
            num_classes=int(num_classes),
            cfg=cfg,
        ).to(dev)
        if hasattr(net, 'set_readout_mode'):
            net.set_readout_mode(str(readout_mode_key))
        model_token = str(base_model_token if fixed_branch_override is not None else canonical_model)
        if bool(recurrent_mode):
            model_token = f"{model_token}_R" if fixed_branch_override is None else f"{base_model_token}_R_{int(fixed_branch_override)}"
        variant_meta['fixed_branch_count'] = None if effective_branch is None else int(effective_branch)
        variant_meta['recurrent'] = bool(recurrent_mode)

    optimizer, opt_info = build_adamw(
        net,
        lr=float(lr),
        weight_decay=float(weight_decay),
        weight_decay_dend_soma=weight_decay_dend_soma,
    )
    if readout_mode_key == 'earliest_spike':
        fs_hparams = _first_spike_loss_hparams(dataset_name)
        criterion: nn.Module = FirstSpikeLoss(
            num_classes=int(num_classes),
            step=int(T),
            alpha_fs=float(fs_hparams['alpha_fs']),
            D=int(fs_hparams['D']),
            A=float(fs_hparams['A']),
            lambda_treg=float(fs_hparams['lambda_treg']),
            beta_treg=float(fs_hparams['beta_treg']),
        )
    else:
        fs_hparams = {}
        criterion = nn.CrossEntropyLoss()

    ts = str(timestamp) if timestamp is not None else now_timestamp_seoul()
    run_name = _sanitize_run_name(exp_name or f'psd_analysis-{dataset_name}-{model_token}')
    run_root = os.path.join(out_root_abs, f'{run_name}_{ts}')
    os.makedirs(run_root, exist_ok=True)

    fixed_scopes = {
        'train': select_fixed_probe_scopes(train_dataset, int(num_classes), split_name='train', base_seed=int(seed), same_label_n=int(same_label_n_per_label), balanced_n=int(balanced_global_n_per_label)),
        'test': select_fixed_probe_scopes(test_dataset, int(num_classes), split_name='test', base_seed=int(seed), same_label_n=int(same_label_n_per_label), balanced_n=int(balanced_global_n_per_label)),
    }
    probe_batches_by_split = {
        'train': _materialize_probe_batches(train_dataset, fixed_scopes['train'], pin_memory=bool(dev.type == 'cuda')),
        'test': _materialize_probe_batches(test_dataset, fixed_scopes['test'], pin_memory=bool(dev.type == 'cuda')),
    }

    edge_values, edge_source = normalize_userbin_edges(userbin_edges)
    nperseg_eff, noverlap_eff = effective_psd_window(int(T), int(psd_window), int(psd_overlap))
    periodogram_band_ranges = temporal_band_ranges_from_edges(int(T), edge_values)
    spectrogram_band_ranges = temporal_band_ranges_from_edges(int(nperseg_eff), edge_values)
    userbin_centers_np = userbin_centers(edge_values)
    hidden_layer_names = layer_names_from_hidden(hidden)
    block_specs_by_layer = _block_specs_from_variant_meta(hidden_layer_names, variant_meta)
    decay_layers = list(net.hidden_layers) + [net.output_layer]
    decay_layer_names = hidden_layer_names + ['output']
    supports_decay_stats = any(isinstance(layer, (LIFDenseLayer, RFDenseLayer)) for layer in decay_layers)
    spike_key = str(spec.signal_manifest['layer_spike_keys'][0])
    membrane_key = str(spike_driving_membrane_key(canonical_model))

    probe_reference_payloads_by_split = {
        'train': _probe_reference_payloads_for_split(
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
            save_plots=False,
            defer_plots=True,
        ),
        'test': _probe_reference_payloads_for_split(
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
            save_plots=False,
            defer_plots=True,
        ),
    }

    config: Dict[str, Any] = {
        'run_name': os.path.basename(run_root),
        'experiment_name': 'psd_analysis',
        'dataset_name': str(dataset_name),
        'spec_doc': 'paper/proposed/psd_analysis.md',
        'implementation_doc': 'paper/proposed/psd_analysis_implement.md',
        'plotting_spec_doc': 'paper/proposed/psd_userbin_async_plot.md',
        'result_root_abs': run_root,
        'run_root_abs': run_root,
        'parent_result_root_abs': out_root_abs,
        'data_root_abs': data_root_abs,
        'dataset_bundle': {
            'num_classes': int(num_classes),
            'input_dim': int(input_dim),
            'T': int(T),
            'metadata': dict(bundle.metadata),
        },
        'model_name': str(model_token),
        'base_model_name': str(canonical_model),
        'model_builder_name': str(spec.builder_name),
        'hidden': list(hidden),
        'input_layer_semantics': 'data_stream',
        'output_layer_semantics': 'neuron_output_layer_with_functional_readout_no_extra_nn_head',
        'output_layer_saved_families': ['spike', 'membrane'],
        'epochs': int(epochs),
        'analysis_epochs': [int(v) for v in range(1, int(epochs) + 1)],
        'plot_epochs_requested': None if plot_epochs is None else [int(v) for v in plot_epochs],
        'signal_psd_plot_epochs': [int(v) for v in signal_psd_plot_epochs],
        'signal_psd_epoch_selection_mode': 'all_epochs_when_unspecified_else_explicit_epoch_list',
        'save_every_epoch': bool(len(signal_psd_plot_epochs) == int(epochs)),
        'signal_psd_save_every_epoch': bool(len(signal_psd_plot_epochs) == int(epochs)),
        'soft_mask_epochs': None if soft_mask_epochs is None else int(soft_mask_epochs),
        'stabilize_epochs': int(stabilize_epochs),
        'ste_epochs': int(ste_epochs),
        'batch_size': int(batch_size),
        'num_workers': int(num_workers),
        'lr': float(lr),
        'weight_decay': float(weight_decay),
        'weight_decay_dend_soma': None if weight_decay_dend_soma is None else float(weight_decay_dend_soma),
        'optimizer_group_info': {
            'weight_decay': float(opt_info.weight_decay),
            'weight_decay_dend_soma': None if opt_info.weight_decay_dend_soma is None else float(opt_info.weight_decay_dend_soma),
            'num_decay_layer_params': int(opt_info.num_decay_layer_params),
            'num_decay_dend_soma_params': int(opt_info.num_decay_dend_soma_params),
            'num_no_decay_params': int(opt_info.num_no_decay_params),
        },
        'seed': int(seed),
        'S_min': float(S_min),
        'S_max': float(S_max),
        'th_len': int(th_len),
        'v_th': float(v_th),
        'v_pre': float(v_pre),
        'lambda_ortho': float(lambda_ortho),
        'lambda_s': float(lambda_s),
        'same_label_n_per_label': int(same_label_n_per_label),
        'balanced_global_n_per_label': int(balanced_global_n_per_label),
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
        'spike_driving_membrane_key': str(membrane_key),
        'spike_key': str(spike_key),
        'psd_window': int(psd_window),
        'psd_overlap': int(psd_overlap),
        'periodogram_length_effective': int(T),
        'spectrogram_window_effective': int(nperseg_eff),
        'spectrogram_overlap_effective': int(noverlap_eff),
        'window_fn_legacy_ignored': str(window_fn),
        'taper_window_applied': False,
        'userbin_edges': list(edge_values),
        'userbin_edges_source': str(edge_source),
        'waveform_psd_representation': 'exact_full_length_simple_periodogram_saved_for_raw_and_centered',
        'heatmap_psd_representation': 'userbin_from_exact_periodogram_saved_for_raw_and_centered',
        'spectrogram_representation': 'exact_sliding_simple_periodogram_saved_for_raw_and_centered',
        'spectrogram_heatmap_representation': 'userbin_from_exact_spectrogram_frame_major_per_element_saved_for_raw_and_centered',
        'variants_saved': ['raw', 'centered'],
        'probe_psd_method': 'simple_periodogram',
        'probe_set_reference_root': None,
        'save_probe_set_reference_once': False,
        'save_change_plots': False,
        'save_db_psd_plots': True,
        'db_plot_scale': '10log10_power_plus_epsilon',
        'db_plot_epsilon': float(_DERIVATIVE_DB_EPS),
        'save_spectrogram_plots': True,
        'save_spectrogram_userbin_heatmaps': True,
        'save_coherence_plots': False,
        'save_inter_layer_comparison_plots': False,
        'save_trend_plots': False,
        'save_spatial_2d_psd_plots': False,
        'save_output_layer_plots': True,
        'save_attenuation_bar_plots': bool(supports_decay_stats),
        'save_attenuation_value_histograms': bool(supports_decay_stats),
        'rf_saved_filter_stats': ['rho', 'f_cyc_per_sample'],
        'rf_clip_edges_unit': 'normalized_frequency_cyc_per_sample_with_nyquist_0p5',
        'filter_property_spec_doc': 'paper/proposed/filter_analysis.md',
        'filter_property_status': 'attenuation_stats_only',
        'filter_property_reason': 'Selected epoch directories save attenuation statistics and hidden-layer incoming-weight visualizations; training_complete_stats/ keeps the final attenuation snapshot copy.',
        'training_complete_stats_root': _TRAINING_COMPLETE_STATS_DIRNAME,
        'attenuation_stats_root_template': f'{_TRAINING_COMPLETE_STATS_DIRNAME}/attenuation_stats',
        'derivative_semi_metric_root_template': f'{_TRAINING_COMPLETE_STATS_DIRNAME}/derivative_semi_metric',
        'derivative_semi_metric_reference': 'probe input mean plots versus selected-epoch hidden/output mean plots',
        'derivative_semi_metric_tracks_output_layer': True,
        'derivative_semi_metric_excludes_element_plots': True,
        'derivative_semi_metric_tracked_plots': [str(spec_['metric_name']) for spec_ in _DERIVATIVE_METRIC_SPECS],
        'derivative_semi_metric_finite_difference_note': 'finite differences use a unit step, so explicit delta_t is omitted',
        'epoch_all_layers_summary_file': 'epoch_<eeee>/all_layers_summary.csv',
        'training_complete_all_layers_summary_file': f'{_TRAINING_COMPLETE_STATS_DIRNAME}/all_layers_summary.csv',
        'train_test_accuracy_file': 'train_test_accuracy.csv',
        'train_test_accuracy_plot': 'train_test_accuracy.png',
        'training_complete_accuracy_file': f'{_TRAINING_COMPLETE_STATS_DIRNAME}/train_test_accuracy.csv',
        'training_complete_accuracy_plot': f'{_TRAINING_COMPLETE_STATS_DIRNAME}/train_test_accuracy.png',
        'probe_set_accuracy_file': 'probe_set_accuracy.txt',
        'probe_set_accuracy_epoch_root_template': 'epoch_<eeee>/<split>/<scope>/probe_set_accuracy.txt',
        'training_complete_probe_set_accuracy_root': f'{_TRAINING_COMPLETE_STATS_DIRNAME}/probe_set_accuracy',
        'probe_set_accuracy_saved_every_epoch': bool(len(signal_psd_plot_epochs) == int(epochs)),
        'probe_set_accuracy_saved_in_selected_epoch_dirs_only': True,
        'probe_set_accuracy_saved_in_training_complete_stats': True,
        'signal_psd_epoch_dirs_created_only_for_selected_epochs': True,
        'time_domain_plot_files': ['time_domain_heatmap.png', 'time_domain_element_mean.png'],
        'save_time_domain_plots': True,
        'final_accuracy_plot_always_saved': True,
        'attenuation_stats_saved_every_epoch': bool(supports_decay_stats and len(signal_psd_plot_epochs) == int(epochs)),
        'attenuation_stats_saved_in_selected_epoch_dirs_only': bool(supports_decay_stats),
        'final_attenuation_snapshot_always_saved': bool(supports_decay_stats and int(epochs) > 0),
        'hidden_weight_visualizations_saved_in_selected_epoch_dirs': True,
        'hidden_weight_visualization_files': ['w_plot.png'],
        'grouped_hidden_block_psd_saved': bool(len(block_specs_by_layer) > 0),
        'grouped_hidden_block_weight_visualizations_saved': bool(len(block_specs_by_layer) > 0),
        'bundle_plot_files': list(_PSD_BUNDLE_REQUIRED_FILES),
        'readout_mode': str(readout_mode_key),
        'readout_mode_choices': ['final_membrane', 'earliest_spike', 'max_rate'],
        'final_membrane_output_behavior': 'disable_output_spike_and_spike_triggered_reset',
        'spike_readout_output_behavior': 'ordinary_spiking_output_layer',
        'earliest_spike_forward_rule': 'first_spike_then_current_to_past_membrane_state_then_last_membrane_state',
        'earliest_spike_training_rule': 'first-spike released-code timing loss on output spike/membrane records',
        'earliest_spike_time_encoding_origin': 'adapted from the released First-spike timing code',
        'earliest_spike_loss_hparams': fs_hparams,
        'rf_reset_mode': str(rf_reset_mode),
        'lif_reset_mode': 'subtractive_soft_reset_fixed_vth',
        'w_clip_edges': None if w_clip_edges is None else [float(v) for v in w_clip_edges],
        'alpha_clip_edges': None if alpha_clip_edges is None else [float(v) for v in alpha_clip_edges],
        'band_neuron_ends': None if band_neuron_ends is None else [str(v) for v in band_neuron_ends],
        'tear': int(tear),
        'variant_meta': variant_meta,
        'backend_flags': get_backend_flags(),
        **deferred_plot_metadata(),
        **plot_writer_metadata(),
    }
    save_json(os.path.join(run_root, 'config.json'), config)

    accuracy_csv_path = os.path.join(run_root, 'train_test_accuracy.csv')
    accuracy_png_path = os.path.join(run_root, 'train_test_accuracy.png')
    training_complete_stats_root = os.path.join(run_root, _TRAINING_COMPLETE_STATS_DIRNAME)

    if int(epochs) == 0:
        render_deferred_plot_tasks(run_root, progress_desc=f'plot-{dataset_name}-{model_token}')
        flush_plot_tasks()
        shutdown_plot_worker(wait=True)
        return run_root

    accuracy_history: List[Dict[str, Any]] = []
    hardened_state: Dict[str, bool] = {'done': False}
    derivative_tracker: Dict[str, Any] = {}

    pbar = tqdm(range(1, int(epochs) + 1), total=int(epochs), desc=f'psd-{dataset_name}-{model_token}', leave=True)
    for epoch in pbar:
        stage = configure_structure_schedule(
            net,
            int(epoch),
            total_epochs=int(epochs),
            soft_mask_epochs=soft_mask_epochs,
            stabilize_epochs=int(stabilize_epochs),
            ste_epochs=int(ste_epochs),
            hardened_state=hardened_state,
        )
        train_eval_loss, train_eval_acc = train_one_epoch(
            net,
            train_loader,
            optimizer,
            criterion,
            dev,
            lambda_ortho=float(lambda_ortho),
            lambda_s=float(lambda_s),
        )
        test_eval_loss, test_eval_acc = evaluate_model(net, test_loader, criterion, dev)
        accuracy_history.append(
            {
                'epoch': int(epoch),
                'train_loss': float(train_eval_loss),
                'train_acc': float(train_eval_acc),
                'test_loss': float(test_eval_loss),
                'test_acc': float(test_eval_acc),
                'stage': str(stage['stage']),
                'ste_enabled': bool(stage['ste_enabled']),
                'hardened': bool(stage['hardened']),
                'readout_mode': str(readout_mode_key),
            }
        )
        _append_accuracy_csv_row(accuracy_csv_path, accuracy_history[-1])
        save_signal_psd_plots = int(epoch) in signal_psd_plot_epoch_set
        if bool(save_signal_psd_plots):
            _save_epoch_analysis(
                run_root,
                int(epoch),
                net,
                list(net.hidden_layers),
                hidden_layer_names,
                fixed_scopes,
                membrane_key=str(membrane_key),
                spike_key=str(spike_key),
                nperseg_eff=int(nperseg_eff),
                spectrogram_band_ranges=spectrogram_band_ranges,
                noverlap_eff=int(noverlap_eff),
                window_fn=str(window_fn),
                periodogram_band_ranges=periodogram_band_ranges,
                userbin_centers_np=np.asarray(userbin_centers_np, dtype=float),
                device=dev,
                probe_batches_by_split=probe_batches_by_split,
                criterion=criterion,
                save_signal_psd_plots=True,
                block_specs_by_layer=block_specs_by_layer,
                probe_reference_payloads_by_split=probe_reference_payloads_by_split,
                derivative_tracker=derivative_tracker,
                defer_signal_psd_plots=True,
            )
            _save_hidden_weight_visualizations(
                os.path.join(run_root, f'epoch_{int(epoch):04d}'),
                hidden_layer_names,
                list(net.hidden_layers),
                block_specs_by_layer=block_specs_by_layer,
                epoch=int(epoch),
                defer_plots=True,
            )
            if bool(supports_decay_stats):
                _save_decay_statistics(
                    run_root,
                    int(epoch),
                    decay_layer_names,
                    decay_layers,
                    defer_plots=True,
                )
        pbar.set_postfix({'tr': f'{train_eval_acc:.4f}', 'te': f'{test_eval_acc:.4f}', 'stage': str(stage['stage'])})

    if accuracy_history:
        final_epoch = int(accuracy_history[-1]['epoch'])
        os.makedirs(training_complete_stats_root, exist_ok=True)
        _write_accuracy_csv(accuracy_csv_path, accuracy_history)
        _save_epoch_analysis(
            run_root,
            final_epoch,
            net,
            list(net.hidden_layers),
            hidden_layer_names,
            fixed_scopes,
            membrane_key=str(membrane_key),
            spike_key=str(spike_key),
            nperseg_eff=int(nperseg_eff),
            spectrogram_band_ranges=spectrogram_band_ranges,
            noverlap_eff=int(noverlap_eff),
            window_fn=str(window_fn),
            periodogram_band_ranges=periodogram_band_ranges,
            userbin_centers_np=np.asarray(userbin_centers_np, dtype=float),
            device=dev,
            probe_batches_by_split=probe_batches_by_split,
            criterion=criterion,
            save_signal_psd_plots=False,
            artifact_root=os.path.join(training_complete_stats_root, 'probe_set_accuracy'),
            block_specs_by_layer=block_specs_by_layer,
            probe_reference_payloads_by_split=probe_reference_payloads_by_split,
            derivative_tracker=None,
        )
    render_deferred_plot_tasks(run_root, progress_desc=f'plot-{dataset_name}-{model_token}')
    if accuracy_history:
        _save_accuracy_plot_from_csv(accuracy_csv_path, accuracy_png_path)
        if bool(supports_decay_stats):
            _save_decay_statistics(
                run_root,
                final_epoch,
                decay_layer_names,
                decay_layers,
                stats_root=os.path.join(training_complete_stats_root, 'attenuation_stats'),
                summary_copy_paths=[os.path.join(training_complete_stats_root, 'all_layers_summary.csv')],
            )
        if len(derivative_tracker) > 0:
            _save_derivative_tracker(os.path.join(training_complete_stats_root, 'derivative_semi_metric'), derivative_tracker)
    flush_plot_tasks()
    if accuracy_history:
        _copy_file_if_exists(accuracy_csv_path, os.path.join(training_complete_stats_root, 'train_test_accuracy.csv'))
        _copy_file_if_exists(accuracy_png_path, os.path.join(training_complete_stats_root, 'train_test_accuracy.png'))
    shutdown_plot_worker(wait=True)
    return run_root



def run_psd_analysis_shd(
    *,
    model: str,
    out_root: str,
    data_root: str,
    hidden: Sequence[int],
    epochs: int = 50,
    soft_mask_epochs: Optional[int] = None,
    stabilize_epochs: int = 0,
    ste_epochs: int = 0,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    weight_decay_dend_soma: Optional[float] = None,
    seed: int = 0,
    S_min: float = 1.0,
    S_max: float = 8.0,
    th_len: int = 4,
    v_th: float = 1.0,
    v_pre: float = 1.0,
    T_event: int = 250,
    num_workers: int = 4,
    download: bool = False,
    shd_max_time: float = 1.0,
    shd_binning: str = 'origin',
    shd_unit_indexing: str = 'auto',
    shd_channel_flip: bool = True,
    shd_align_to_first_event: bool = False,
    shd_use_event_counts: bool = False,
    lambda_ortho: float = 0.0,
    lambda_s: float = 0.0,
    same_label_n_per_label: int = 4,
    balanced_global_n_per_label: int = 4,
    probe_plot: bool = False,
    plot_epochs: Optional[Sequence[int]] = None,
    psd_window: int = 64,
    psd_overlap: int = 32,
    window_fn: str = 'hann',
    userbin_edges: Optional[Sequence[float]] = None,
    rf_reset_mode: str = 'no_reset',
    w_clip_edges: Optional[Sequence[float]] = None,
    alpha_clip_edges: Optional[Sequence[float]] = None,
    band_neuron_ends: Optional[Sequence[str]] = None,
    tear: int = 1,
    readout_mode: str = 'final_membrane',
    exp_name: Optional[str] = None,
    timestamp: Optional[str] = None,
    device: str = 'auto',
) -> str:
    return run_psd_analysis(
        dataset='shd',
        model=model,
        out_root=out_root,
        data_root=data_root,
        hidden=hidden,
        epochs=int(epochs),
        soft_mask_epochs=soft_mask_epochs,
        stabilize_epochs=int(stabilize_epochs),
        ste_epochs=int(ste_epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        weight_decay=float(weight_decay),
        weight_decay_dend_soma=weight_decay_dend_soma,
        seed=int(seed),
        S_min=float(S_min),
        S_max=float(S_max),
        th_len=int(th_len),
        v_th=float(v_th),
        v_pre=float(v_pre),
        num_workers=int(num_workers),
        download=bool(download),
        shd_T=int(T_event),
        shd_max_time=float(shd_max_time),
        shd_binning=str(shd_binning),
        shd_unit_indexing=str(shd_unit_indexing),
        shd_channel_flip=bool(shd_channel_flip),
        shd_align_to_first_event=bool(shd_align_to_first_event),
        shd_use_event_counts=bool(shd_use_event_counts),
        lambda_ortho=float(lambda_ortho),
        lambda_s=float(lambda_s),
        same_label_n_per_label=int(same_label_n_per_label),
        balanced_global_n_per_label=int(balanced_global_n_per_label),
        probe_plot=bool(probe_plot),
        plot_epochs=plot_epochs,
        psd_window=int(psd_window),
        psd_overlap=int(psd_overlap),
        window_fn=str(window_fn),
        userbin_edges=userbin_edges,
        rf_reset_mode=str(rf_reset_mode),
        w_clip_edges=w_clip_edges,
        alpha_clip_edges=alpha_clip_edges,
        band_neuron_ends=band_neuron_ends,
        tear=int(tear),
        readout_mode=str(readout_mode),
        exp_name=exp_name,
        timestamp=timestamp,
        device=str(device),
    )
