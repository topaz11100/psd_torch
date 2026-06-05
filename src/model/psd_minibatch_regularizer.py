from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from src.model.snn_builder import LayerRecord
from src.signal.family_spectral_analysis import representative_psd_minibatch_curve_from_maps_torch, curve_pointwise_distance_torch
from src.signal.psd_utils import (
    aggregate_userbins_torch,
    apply_fixed_pca_basis,
    auto_spectral_matrix_from_mode_maps,
    compute_fixed_pca_basis,
    pca_dim_from_cli_vector,
    scalar_representative_maps,
)


@dataclass
class FixedPCALayerReference:
    layer_name: str
    layer_index: int | None
    dim: int
    x_basis: torch.Tensor
    x_centroid: torch.Tensor
    y_basis: torch.Tensor
    y_centroid: torch.Tensor
    output_family: str
    basis_id: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class PSDRegularizationBreakdown:
    total: torch.Tensor
    rep_1d: torch.Tensor
    pca_1d: torch.Tensor
    pca_mimo: torch.Tensor
    per_layer_rep_1d: dict[str, torch.Tensor]
    per_layer_pca_1d: dict[str, torch.Tensor]
    per_layer_pca_mimo: dict[str, torch.Tensor]
    metadata: dict[str, Any] | None = None
    rep_input: torch.Tensor | None = None
    rep_adjacent: torch.Tensor | None = None
    pca_1d_input: torch.Tensor | None = None
    pca_1d_adjacent: torch.Tensor | None = None
    pca_mimo_input: torch.Tensor | None = None
    pca_mimo_adjacent: torch.Tensor | None = None


def _pca_mode_from_dim(dim: int) -> str:
    """Resolve PCA regularizer mode from the configured dimension.

    A resolved dimension of 1 is the scalar/1-D PCA branch; dimensions greater
    than 1 use the MIMO spectral-matrix branch.  The public lambda schema is
    split by relation, not by PCA mode.  Metrics keep ``pca_1d`` and
    ``pca_mimo`` buckets so CSV rows still show which mathematical branch was
    active.
    """

    value = int(dim)
    if value < 1:
        raise ValueError('PCA regularizer dimension must be positive.')
    return '1d' if value == 1 else 'mimo'


def _is_fixed_pca_ref(obj: Any) -> bool:
    return all(hasattr(obj, name) for name in ('x_basis', 'x_centroid', 'y_basis', 'y_centroid', 'dim'))


def move_fixed_pca_reference_bank_to_device(
    bank: dict[str, Any] | None,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any] | None:
    """Move fixed PCA references to the active device.

    ``bank`` may be either the legacy flat mapping ``layer -> reference`` or the
    current relation-split mapping ``relation -> layer -> reference``.
    """

    if bank is None:
        return None
    out: dict[str, Any] = {}
    for key, ref in bank.items():
        if isinstance(ref, dict):
            out[str(key)] = move_fixed_pca_reference_bank_to_device(ref, device=device, dtype=dtype)
            continue
        if not _is_fixed_pca_ref(ref):
            raise TypeError(f'Invalid PCA reference bank entry at {key!r}: {type(ref).__name__}')
        out[str(key)] = FixedPCALayerReference(
            layer_name=str(ref.layer_name),
            layer_index=(None if ref.layer_index is None else int(ref.layer_index)),
            dim=int(ref.dim),
            x_basis=ref.x_basis.detach().to(device=device, dtype=dtype).requires_grad_(False),
            x_centroid=ref.x_centroid.detach().to(device=device, dtype=dtype).requires_grad_(False),
            y_basis=ref.y_basis.detach().to(device=device, dtype=dtype).requires_grad_(False),
            y_centroid=ref.y_centroid.detach().to(device=device, dtype=dtype).requires_grad_(False),
            output_family=str(ref.output_family),
            basis_id=getattr(ref, 'basis_id', None),
            metadata={**(getattr(ref, 'metadata', {}) or {}), 'device_resident': str(device), 'dtype_resident': str(dtype)},
        )
    return out


def _to_maps(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x.transpose(1, 2).contiguous()
    if x.ndim in (2, 4, 5):
        from src.signal.psd_utils import trace_tensor_to_channel_major_maps

        return trace_tensor_to_channel_major_maps(x).contiguous()
    raise ValueError(f'Expected a trace tensor convertible to (samples, rows, time), got {tuple(x.shape)}')


def _select_y(record: LayerRecord, output_family: str) -> torch.Tensor:
    tok = str(output_family).strip().lower()
    if tok == 'spike':
        return _to_maps(record.spike)
    if tok == 'membrane':
        return _to_maps(record.membrane)
    raise ValueError('output_family must be spike or membrane')


def compute_fixed_pca_reference_bank(
    input_batch: torch.Tensor,
    hidden_records: list[LayerRecord],
    output_family: str,
    pca_dim_per_layer: list[int] | None,
    *,
    variant: str = 'raw',
    relation: str = 'adjacent',
    layer_names: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, FixedPCALayerReference]:
    with torch.no_grad():
        bank: dict[str, FixedPCALayerReference] = {}
        tok = str(variant).strip().lower()
        if tok not in {'raw', 'centered'}:
            raise ValueError('variant must be raw or centered')
        if len(hidden_records) == 0:
            raise ValueError('hidden_records must not be empty when building PCA reference bank.')
        rel = str(relation).strip().lower()
        if rel not in {'adjacent', 'input'}:
            raise ValueError('relation must be adjacent or input')
        original_input = _to_maps(input_batch)
        current_input = original_input
        for i, record in enumerate(hidden_records):
            x_maps = original_input if rel == 'input' else current_input
            y_maps = _select_y(record, output_family)
            if tok == 'centered':
                current_fit = x_maps - x_maps.mean(dim=-1, keepdim=True)
                y_fit = y_maps - y_maps.mean(dim=-1, keepdim=True)
            else:
                current_fit, y_fit = x_maps, y_maps
            dim = pca_dim_from_cli_vector(pca_dim_per_layer, i, int(x_maps.shape[1]))
            x_basis, x_centroid = compute_fixed_pca_basis(current_fit, dim)
            y_dim = pca_dim_from_cli_vector(pca_dim_per_layer, i, int(y_maps.shape[1]))
            y_basis, y_centroid = compute_fixed_pca_basis(y_fit, y_dim)
            d = min(int(x_basis.shape[1]), int(y_basis.shape[1]))
            layer_name = str(record.layer_name if record.layer_name else f'hidden_{i}')
            if layer_names is not None and i < len(layer_names):
                layer_name = str(layer_names[i])
            pca_mode = _pca_mode_from_dim(d)
            basis_id = f'{rel}|{layer_name}|{output_family}|dim={d}|mode={pca_mode}'
            bank[layer_name] = FixedPCALayerReference(
                layer_name=layer_name,
                layer_index=i,
                dim=d,
                x_basis=x_basis[:, :d].detach().requires_grad_(False),
                x_centroid=x_centroid.detach().requires_grad_(False),
                y_basis=y_basis[:, :d].detach().requires_grad_(False),
                y_centroid=y_centroid.detach().requires_grad_(False),
                output_family=str(output_family),
                basis_id=basis_id,
                metadata={
                    'variant': tok,
                    'relation': rel,
                    'resolved_dim': int(d),
                    'pca_mode': pca_mode,
                    'pca_mode_policy': 'dim==1 -> 1d; dim>=2 -> mimo',
                    **(metadata or {}),
                },
            )
            current_input = _to_maps(record.spike)
        return bank


def _aggregate_complex_spectral_matrix_userbins(
    matrix: torch.Tensor,
    freqs: torch.Tensor,
    userbin_edges: Sequence[float],
    *,
    reducer: str = 'mean',
) -> torch.Tensor:
    """Aggregate a complex [F,D,D] spectral matrix on frequency user bins."""

    real, _edges, _centers = aggregate_userbins_torch(matrix.real, freqs, userbin_edges, axis=0, reducer=str(reducer))
    imag, _edges, _centers = aggregate_userbins_torch(matrix.imag, freqs, userbin_edges, axis=0, reducer=str(reducer))
    return torch.complex(real, imag)


def _spectral_matrix_feature(matrix: torch.Tensor, *, scale: str) -> torch.Tensor:
    token = str(scale or 'raw').strip().lower().replace('-', '_')
    aliases = {'linear': 'raw', 'log': 'db', 'area_norm': 'area', 'area_normalized': 'area', 'normalized': 'area'}
    token = aliases.get(token, token)
    if token == 'raw':
        return torch.view_as_real(matrix)
    if token == 'db':
        return 10.0 * torch.log10(matrix.abs().clamp_min(1.0e-12))
    if token == 'area':
        denom = matrix.abs().sum().clamp_min(1.0e-12)
        return torch.view_as_real(matrix / denom)
    raise ValueError(f'Unsupported spectral-matrix curve scale: {scale!r}. Allowed values: raw, db, area.')


def _spectral_matrix_distance(mx: torch.Tensor, my: torch.Tensor, *, metric: str, scale: str) -> torch.Tensor:
    x = _spectral_matrix_feature(mx, scale=scale)
    y = _spectral_matrix_feature(my, scale=scale)
    if tuple(x.shape) != tuple(y.shape):
        raise ValueError(f'Spectral matrix shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}')
    token = str(metric or 'centered_l2').strip().lower()
    if token == 'centered_l2':
        xf = x.reshape(-1)
        yf = y.reshape(-1)
        return torch.linalg.vector_norm((xf - xf.mean()) - (yf - yf.mean()), ord=2)
    if token == 'diff_l2':
        if int(x.shape[0]) < 2:
            return x.new_zeros(())
        return torch.linalg.vector_norm((torch.diff(x, dim=0) - torch.diff(y, dim=0)).reshape(-1), ord=2)
    raise ValueError(f'Unsupported spectral-matrix distance metric: {metric!r}.')


def _zero_psd_breakdown(reference: torch.Tensor, *, metadata: dict[str, Any] | None = None) -> PSDRegularizationBreakdown:
    zero = reference.new_zeros(())
    return PSDRegularizationBreakdown(
        total=zero,
        rep_1d=zero,
        pca_1d=zero,
        pca_mimo=zero,
        per_layer_rep_1d={},
        per_layer_pca_1d={},
        per_layer_pca_mimo={},
        metadata=dict(metadata or {}),
        rep_input=zero,
        rep_adjacent=zero,
        pca_1d_input=zero,
        pca_1d_adjacent=zero,
        pca_mimo_input=zero,
        pca_mimo_adjacent=zero,
    )


def _prefix_layer_dict(relation: str, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {f'{relation}/{k}': v for k, v in values.items()}


def _select_pca_bank_for_relation(
    pca_reference_banks: dict[str, Any] | None,
    relation: str,
) -> dict[str, FixedPCALayerReference] | None:
    if pca_reference_banks is None:
        return None
    rel = str(relation).strip().lower()
    maybe = pca_reference_banks.get(rel)
    if isinstance(maybe, dict):
        return maybe  # type: ignore[return-value]
    # Legacy flat bank compatibility.
    if pca_reference_banks and all(_is_fixed_pca_ref(v) for v in pca_reference_banks.values()):
        return pca_reference_banks  # type: ignore[return-value]
    return None


def _compute_minibatch_psd_regularizer_single_relation(
    input_batch: torch.Tensor,
    hidden_records: list[LayerRecord],
    variant: str,
    output_family: str,
    lambda_rep_1d: float,
    lambda_pca: float,
    pca_reference_bank: dict[str, FixedPCALayerReference] | None,
    *,
    curve_scale: str = 'raw',
    curve_space: str = 'exact',
    userbin_edges: Sequence[float] | None = None,
    userbin_reducer: str = 'mean',
    reducer: str = 'mean',
    distance_metric: str = 'centered_l2',
    relation: str = 'adjacent',
    signal_window: str | bool | None = 'hann',
) -> PSDRegularizationBreakdown:
    ref = input_batch.new_zeros(())
    if float(lambda_rep_1d) == 0.0 and float(lambda_pca) == 0.0:
        return _zero_psd_breakdown(ref, metadata={
            'variant': str(variant),
            'output_family': str(output_family),
            'curve_scale': str(curve_scale),
            'curve_space': str(curve_space),
            'userbin_reducer': str(userbin_reducer),
            'reducer': str(reducer),
            'distance_metric': str(distance_metric),
            'relation': str(relation),
            'signal_window': str(signal_window),
            'pca_mode_policy': 'dim==1 -> 1d; dim>=2 -> mimo',
        })
    if float(lambda_pca) != 0.0 and not pca_reference_bank:
        raise ValueError(f'PCA lambda is nonzero but pca_reference_bank is missing for relation={relation!r}.')
    rel = str(relation).strip().lower()
    if rel not in {'adjacent', 'input'}:
        raise ValueError('relation must be adjacent or input')
    curve_space_token = str(curve_space).strip().lower()
    if curve_space_token not in {'exact', 'userbin'}:
        raise ValueError('curve_space must be exact or userbin')
    if curve_space_token == 'userbin' and userbin_edges is None:
        raise ValueError('userbin_edges must be provided when curve_space=userbin.')
    rep_parts: dict[str, torch.Tensor] = {}
    pca1_parts: dict[str, torch.Tensor] = {}
    pcam_parts: dict[str, torch.Tensor] = {}
    pca_mode_by_layer: dict[str, str] = {}
    pca_dim_by_layer: dict[str, int] = {}
    rep = ref
    pca1 = ref
    pcam = ref
    original_input = _to_maps(input_batch)
    current_input = original_input
    for record in hidden_records:
        source_input = original_input if rel == 'input' else current_input
        y_maps = _select_y(record, output_family)
        if float(lambda_rep_1d) != 0.0:
            x_rep = scalar_representative_maps(source_input, reducer=str(reducer))
            y_rep = scalar_representative_maps(y_maps, reducer=str(reducer))
            xc = representative_psd_minibatch_curve_from_maps_torch(
                x_rep,
                reducer=str(reducer),
                centering=variant,
                scale=curve_scale,
                curve_space=curve_space_token,
                userbin_edges=userbin_edges,
                userbin_reducer=str(userbin_reducer),
                signal_window=signal_window,
            )
            yc = representative_psd_minibatch_curve_from_maps_torch(
                y_rep,
                reducer=str(reducer),
                centering=variant,
                scale=curve_scale,
                curve_space=curve_space_token,
                userbin_edges=userbin_edges,
                userbin_reducer=str(userbin_reducer),
                signal_window=signal_window,
            )
            v = curve_pointwise_distance_torch(xc, yc, metric=str(distance_metric))
            rep = rep + v
            rep_parts[record.layer_name] = v
        if float(lambda_pca) != 0.0:
            assert pca_reference_bank is not None
            if record.layer_name not in pca_reference_bank:
                raise ValueError(f'Missing PCA layer key: {record.layer_name} for relation={rel}')
            r = pca_reference_bank[record.layer_name]
            x_modes = apply_fixed_pca_basis(source_input, r.x_basis, r.x_centroid)
            y_modes = apply_fixed_pca_basis(y_maps, r.y_basis, r.y_centroid)
            d = min(int(x_modes.shape[1]), int(y_modes.shape[1]), int(r.dim))
            x_modes = x_modes[:, :d, :]
            y_modes = y_modes[:, :d, :]
            mode = _pca_mode_from_dim(d)
            pca_mode_by_layer[record.layer_name] = mode
            pca_dim_by_layer[record.layer_name] = int(d)
            if mode == '1d':
                v1 = curve_pointwise_distance_torch(
                    representative_psd_minibatch_curve_from_maps_torch(
                        x_modes,
                        reducer=str(reducer),
                        centering=variant,
                        scale=curve_scale,
                        curve_space=curve_space_token,
                        userbin_edges=userbin_edges,
                        userbin_reducer=str(userbin_reducer),
                        signal_window=signal_window,
                    ),
                    representative_psd_minibatch_curve_from_maps_torch(
                        y_modes,
                        reducer=str(reducer),
                        centering=variant,
                        scale=curve_scale,
                        curve_space=curve_space_token,
                        userbin_edges=userbin_edges,
                        userbin_reducer=str(userbin_reducer),
                        signal_window=signal_window,
                    ),
                    metric=str(distance_metric),
                )
                pca1 = pca1 + v1
                pca1_parts[record.layer_name] = v1
            else:
                if str(variant).strip().lower() == 'centered':
                    x_matrix_modes = x_modes - x_modes.mean(dim=-1, keepdim=True)
                    y_matrix_modes = y_modes - y_modes.mean(dim=-1, keepdim=True)
                else:
                    x_matrix_modes = x_modes
                    y_matrix_modes = y_modes
                _fx, mx = auto_spectral_matrix_from_mode_maps(x_matrix_modes, signal_window=signal_window)
                _fy, my = auto_spectral_matrix_from_mode_maps(y_matrix_modes, signal_window=signal_window)
                if int(_fx.numel()) != int(_fy.numel()) or torch.max(torch.abs(_fx - _fy)) > 1.0e-6:
                    raise ValueError('PCA-MIMO x/y frequency axes do not match.')
                if curve_space_token == 'userbin':
                    assert userbin_edges is not None
                    mx = _aggregate_complex_spectral_matrix_userbins(mx, _fx, userbin_edges, reducer=str(userbin_reducer))
                    my = _aggregate_complex_spectral_matrix_userbins(my, _fy, userbin_edges, reducer=str(userbin_reducer))
                v2 = _spectral_matrix_distance(mx, my, metric=str(distance_metric), scale=str(curve_scale))
                pcam = pcam + v2
                pcam_parts[record.layer_name] = v2
        current_input = _to_maps(record.spike)
    rep = float(lambda_rep_1d) * rep
    pca1 = float(lambda_pca) * pca1
    pcam = float(lambda_pca) * pcam
    return PSDRegularizationBreakdown(
        rep + pca1 + pcam,
        rep,
        pca1,
        pcam,
        rep_parts,
        pca1_parts,
        pcam_parts,
        {
            'variant': str(variant),
            'output_family': str(output_family),
            'curve_scale': str(curve_scale),
            'curve_space': curve_space_token,
            'userbin_reducer': str(userbin_reducer),
            'reducer': str(reducer),
            'distance_metric': str(distance_metric),
            'relation': rel,
            'signal_window': str(signal_window),
            'pca_mode_policy': 'dim==1 -> 1d; dim>=2 -> mimo',
            'pca_mode_by_layer': dict(pca_mode_by_layer),
            'pca_dim_by_layer': dict(pca_dim_by_layer),
        },
    )


def compute_minibatch_psd_regularizer(
    input_batch: torch.Tensor,
    hidden_records: list[LayerRecord],
    variant: str,
    output_family: str,
    *,
    lambda_rep_input: float = 0.0,
    lambda_rep_adjacent: float = 0.0,
    lambda_pca_input: float = 0.0,
    lambda_pca_adjacent: float = 0.0,
    pca_reference_banks: dict[str, Any] | None = None,
    curve_scale: str = 'raw',
    curve_space: str = 'exact',
    userbin_edges: Sequence[float] | None = None,
    userbin_reducer: str = 'mean',
    reducer: str = 'mean',
    distance_metric: str = 'centered_l2',
    signal_window: str | bool | None = 'hann',
    # Legacy positional-name compatibility. New callers should not use these.
    lambda_rep_1d: float | None = None,
    lambda_pca: float | None = None,
    pca_reference_bank: dict[str, Any] | None = None,
    relation: str | None = None,
) -> PSDRegularizationBreakdown:
    if lambda_rep_1d is not None or lambda_pca is not None or relation is not None or pca_reference_bank is not None:
        rel = str(relation or 'adjacent').strip().lower()
        if rel == 'input':
            lambda_rep_input = float(lambda_rep_input) + float(lambda_rep_1d or 0.0)
            lambda_pca_input = float(lambda_pca_input) + float(lambda_pca or 0.0)
        elif rel == 'adjacent':
            lambda_rep_adjacent = float(lambda_rep_adjacent) + float(lambda_rep_1d or 0.0)
            lambda_pca_adjacent = float(lambda_pca_adjacent) + float(lambda_pca or 0.0)
        else:
            raise ValueError('relation must be adjacent or input')
        if pca_reference_banks is None:
            pca_reference_banks = pca_reference_bank

    ref = input_batch.new_zeros(())
    requested = {
        'input': {'rep': float(lambda_rep_input), 'pca': float(lambda_pca_input)},
        'adjacent': {'rep': float(lambda_rep_adjacent), 'pca': float(lambda_pca_adjacent)},
    }
    if all(v['rep'] == 0.0 and v['pca'] == 0.0 for v in requested.values()):
        return _zero_psd_breakdown(ref, metadata={
            'variant': str(variant),
            'output_family': str(output_family),
            'curve_scale': str(curve_scale),
            'curve_space': str(curve_space),
            'userbin_reducer': str(userbin_reducer),
            'reducer': str(reducer),
            'distance_metric': str(distance_metric),
            'relations': [],
            'signal_window': str(signal_window),
            'pca_mode_policy': 'dim==1 -> 1d; dim>=2 -> mimo',
        })

    total = ref
    rep_total = ref
    pca1_total = ref
    pcam_total = ref
    relation_outputs: dict[str, PSDRegularizationBreakdown] = {}

    for rel in ('input', 'adjacent'):
        lam_rep = requested[rel]['rep']
        lam_pca = requested[rel]['pca']
        if lam_rep == 0.0 and lam_pca == 0.0:
            continue
        bank = _select_pca_bank_for_relation(pca_reference_banks, rel)
        out = _compute_minibatch_psd_regularizer_single_relation(
            input_batch,
            hidden_records,
            variant,
            output_family,
            lambda_rep_1d=lam_rep,
            lambda_pca=lam_pca,
            pca_reference_bank=bank,
            curve_scale=curve_scale,
            curve_space=curve_space,
            userbin_edges=userbin_edges,
            userbin_reducer=userbin_reducer,
            reducer=reducer,
            distance_metric=distance_metric,
            relation=rel,
            signal_window=signal_window,
        )
        relation_outputs[rel] = out
        total = total + out.total
        rep_total = rep_total + out.rep_1d
        pca1_total = pca1_total + out.pca_1d
        pcam_total = pcam_total + out.pca_mimo

    per_layer_rep: dict[str, torch.Tensor] = {}
    per_layer_pca1: dict[str, torch.Tensor] = {}
    per_layer_pcam: dict[str, torch.Tensor] = {}
    for rel, out in relation_outputs.items():
        per_layer_rep.update(_prefix_layer_dict(rel, out.per_layer_rep_1d))
        per_layer_pca1.update(_prefix_layer_dict(rel, out.per_layer_pca_1d))
        per_layer_pcam.update(_prefix_layer_dict(rel, out.per_layer_pca_mimo))

    input_out = relation_outputs.get('input')
    adjacent_out = relation_outputs.get('adjacent')
    return PSDRegularizationBreakdown(
        total=total,
        rep_1d=rep_total,
        pca_1d=pca1_total,
        pca_mimo=pcam_total,
        per_layer_rep_1d=per_layer_rep,
        per_layer_pca_1d=per_layer_pca1,
        per_layer_pca_mimo=per_layer_pcam,
        metadata={
            'variant': str(variant),
            'output_family': str(output_family),
            'curve_scale': str(curve_scale),
            'curve_space': str(curve_space).strip().lower(),
            'userbin_reducer': str(userbin_reducer),
            'reducer': str(reducer),
            'distance_metric': str(distance_metric),
            'relations': sorted(list(relation_outputs.keys())),
            'signal_window': str(signal_window),
            'pca_mode_policy': 'dim==1 -> 1d; dim>=2 -> mimo',
            'relation_metadata': {rel: out.metadata for rel, out in relation_outputs.items()},
        },
        rep_input=(input_out.rep_1d if input_out is not None else ref),
        rep_adjacent=(adjacent_out.rep_1d if adjacent_out is not None else ref),
        pca_1d_input=(input_out.pca_1d if input_out is not None else ref),
        pca_1d_adjacent=(adjacent_out.pca_1d if adjacent_out is not None else ref),
        pca_mimo_input=(input_out.pca_mimo if input_out is not None else ref),
        pca_mimo_adjacent=(adjacent_out.pca_mimo if adjacent_out is not None else ref),
    )
