from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from src.model.snn_builder import LayerRecord
from src.signal.psd_utils import scalar_representative_maps, compute_fixed_pca_basis, apply_fixed_pca_basis, auto_spectral_matrix_from_mode_maps, pca_dim_from_cli_vector
from src.signal.family_spectral_analysis import representative_psd_minibatch_curve_from_maps_torch, curve_pointwise_distance_torch

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


def _to_maps(x: torch.Tensor)->torch.Tensor:
    if x.ndim==3:
        return x.transpose(1,2).contiguous()
    if x.ndim in (2,4,5):
        from src.signal.psd_utils import trace_tensor_to_channel_major_maps
        return trace_tensor_to_channel_major_maps(x).contiguous()
    raise ValueError(f'Expected a trace tensor convertible to (samples, rows, time), got {tuple(x.shape)}')

def _select_y(record: LayerRecord, output_family: str)->torch.Tensor:
    tok=str(output_family).strip().lower()
    if tok=='spike': return _to_maps(record.spike)
    if tok=='membrane': return _to_maps(record.membrane)
    raise ValueError('output_family must be spike or membrane')

def compute_fixed_pca_reference_bank(input_batch: torch.Tensor, hidden_records: list[LayerRecord], output_family: str, pca_dim_per_layer: list[int] | None, *, variant: str = 'raw', relation: str = 'adjacent', layer_names: list[str] | None = None, metadata: dict[str, Any] | None = None) -> dict[str, FixedPCALayerReference]:
    with torch.no_grad():
        bank={}
        tok = str(variant).strip().lower()
        if tok not in {'raw', 'centered'}:
            raise ValueError('variant must be raw or centered')
        if len(hidden_records)==0:
            raise ValueError('hidden_records must not be empty when building PCA reference bank.')
        rel = str(relation).strip().lower()
        if rel not in {'adjacent', 'input'}:
            raise ValueError('relation must be adjacent or input')
        original_input = _to_maps(input_batch)
        current_input = original_input
        for i,record in enumerate(hidden_records):
            x_maps = original_input if rel == 'input' else current_input
            y_maps=_select_y(record, output_family)
            if tok=='centered':
                current_fit = x_maps - x_maps.mean(dim=-1, keepdim=True)
                y_fit = y_maps - y_maps.mean(dim=-1, keepdim=True)
            else:
                current_fit, y_fit = x_maps, y_maps
            dim=pca_dim_from_cli_vector(pca_dim_per_layer, i, int(x_maps.shape[1]))
            x_basis, x_centroid = compute_fixed_pca_basis(current_fit, dim)
            y_dim=pca_dim_from_cli_vector(pca_dim_per_layer, i, int(y_maps.shape[1]))
            y_basis, y_centroid = compute_fixed_pca_basis(y_fit, y_dim)
            d=min(int(x_basis.shape[1]), int(y_basis.shape[1]))
            layer_name = str(record.layer_name if record.layer_name else f'hidden_{i}')
            basis_id = f'{layer_name}|{output_family}|dim={d}'
            bank[layer_name]=FixedPCALayerReference(
                layer_name=layer_name,
                layer_index=i,
                dim=d,
                x_basis=x_basis[:, :d].detach().requires_grad_(False),
                x_centroid=x_centroid.detach().requires_grad_(False),
                y_basis=y_basis[:, :d].detach().requires_grad_(False),
                y_centroid=y_centroid.detach().requires_grad_(False),
                output_family=str(output_family),
                basis_id=basis_id,
                metadata={'variant': tok, 'relation': rel, **(metadata or {})},
            )
            current_input = _to_maps(record.spike)
        return bank

def compute_minibatch_psd_regularizer(input_batch: torch.Tensor, hidden_records: list[LayerRecord], variant: str, output_family: str, lambda_rep_1d: float, lambda_pca: float, pca_reference_bank: dict[str, FixedPCALayerReference] | None, *, curve_scale: str = 'raw', relation: str = 'adjacent') -> PSDRegularizationBreakdown:
    ref = input_batch.new_zeros(())
    if float(lambda_rep_1d)==0.0 and float(lambda_pca)==0.0:
        return PSDRegularizationBreakdown(ref,ref,ref,ref,{},{},{},{'variant': str(variant), 'output_family': str(output_family), 'curve_scale': str(curve_scale), 'relation': str(relation)})
    if float(lambda_pca)!=0.0 and not pca_reference_bank:
        raise ValueError('PCA lambda is nonzero but pca_reference_bank is missing.')
    rel = str(relation).strip().lower()
    if rel not in {'adjacent', 'input'}:
        raise ValueError('relation must be adjacent or input')
    rep_parts={}; pca1_parts={}; pcam_parts={}
    rep=ref; pca1=ref; pcam=ref
    original_input=_to_maps(input_batch)
    current_input=original_input
    for record in hidden_records:
        source_input = original_input if rel == 'input' else current_input
        y_maps=_select_y(record, output_family)
        if float(lambda_rep_1d)!=0.0:
            x_rep=scalar_representative_maps(source_input,reducer='mean')
            y_rep=scalar_representative_maps(y_maps,reducer='mean')
            xc=representative_psd_minibatch_curve_from_maps_torch(x_rep,reducer='mean',centering=variant,scale=curve_scale,curve_space='exact')
            yc=representative_psd_minibatch_curve_from_maps_torch(y_rep,reducer='mean',centering=variant,scale=curve_scale,curve_space='exact')
            v=curve_pointwise_distance_torch(xc,yc,metric='centered_l2'); rep=rep+v; rep_parts[record.layer_name]=v
        if float(lambda_pca)!=0.0:
            if record.layer_name not in pca_reference_bank: raise ValueError(f'Missing PCA layer key: {record.layer_name}')
            r=pca_reference_bank[record.layer_name]
            x_modes=apply_fixed_pca_basis(source_input, r.x_basis, r.x_centroid)
            y_modes=apply_fixed_pca_basis(y_maps, r.y_basis, r.y_centroid)
            d=min(int(x_modes.shape[1]), int(y_modes.shape[1]))
            x_modes=x_modes[:,:d,:]; y_modes=y_modes[:,:d,:]
            v1=curve_pointwise_distance_torch(
                representative_psd_minibatch_curve_from_maps_torch(x_modes,reducer='mean',centering=variant,scale=curve_scale,curve_space='exact'),
                representative_psd_minibatch_curve_from_maps_torch(y_modes,reducer='mean',centering=variant,scale=curve_scale,curve_space='exact'),
                metric='centered_l2')
            pca1=pca1+v1; pca1_parts[record.layer_name]=v1
            _fx,mx=auto_spectral_matrix_from_mode_maps(x_modes); _fy,my=auto_spectral_matrix_from_mode_maps(y_modes)
            v2=torch.linalg.vector_norm((mx.real-my.real).reshape(-1),ord=2)
            pcam=pcam+v2; pcam_parts[record.layer_name]=v2
        current_input=_to_maps(record.spike)
    rep = float(lambda_rep_1d)*rep
    pca1 = float(lambda_pca)*pca1
    pcam = float(lambda_pca)*pcam
    return PSDRegularizationBreakdown(rep+pca1+pcam, rep,pca1,pcam, rep_parts,pca1_parts,pcam_parts, {'variant': str(variant), 'output_family': str(output_family), 'curve_scale': str(curve_scale), 'relation': rel})
