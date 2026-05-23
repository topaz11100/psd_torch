from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class FFT2DResult:
    matrix: torch.Tensor
    row_axis: list[float]
    col_axis: list[float]
    metadata: dict


def _bin_1d(values, edges, reducer='mean', allow_empty=False, empty_policy='error'):
    out=[]; centers=[]; empty=[]
    for lo,hi in zip(edges[:-1], edges[1:]):
        m=(values>=lo)&(values<hi)
        centers.append((lo+hi)/2)
        if not torch.any(m):
            empty.append(True)
            if not allow_empty and empty_policy=='error': raise ValueError('empty bin encountered')
            out.append(float('nan') if empty_policy=='nan' else 0.0)
        else:
            empty.append(False)
            vv=values[m]
            out.append(vv.mean().item() if reducer=='mean' else vv.median().item())
    return out, centers, empty


def fft2d_exact(maps: torch.Tensor, centering='none', window_time='none', window_row='none', to_db=False, eps=1e-12):
    # maps: S,R,T
    x=maps
    if centering=='time_mean': x=x-x.mean(dim=-1, keepdim=True)
    if centering=='global_mean': x=x-x.mean(dim=(-1,-2), keepdim=True)
    if window_time=='hann': x=x*torch.hann_window(x.shape[-1], device=x.device, dtype=x.dtype)
    if window_row=='hann': x=x*torch.hann_window(x.shape[-2], device=x.device, dtype=x.dtype).view(1,-1,1)
    fr = torch.fft.fftfreq(x.shape[-2], d=1.0)
    ft = torch.fft.rfftfreq(x.shape[-1], d=1.0)
    y = torch.fft.fft(x, dim=-2)
    y = torch.fft.rfft(y, dim=-1)
    p=(y.real**2 + y.imag**2)
    m=p.mean(dim=0)
    if to_db: m=10*torch.log10(torch.clamp(m,min=eps))
    return FFT2DResult(matrix=m, row_axis=fr.tolist(), col_axis=ft.tolist(), metadata={'spectral_axis':'exact','fftshift_row':False})


def fft2d_userbin(result: FFT2DResult, userbin_axes: str, reducer='mean', time_edges=None, row_edges=None, allow_empty=False, empty_policy='error', row_axis_semantics='unordered'):
    mat=result.matrix
    row_axis=torch.tensor(result.row_axis)
    col_axis=torch.tensor(result.col_axis)
    if userbin_axes in {'row_frequency','both_frequency_axes'} and row_axis_semantics=='unordered':
        raise ValueError('row_frequency/both_frequency_axes userbin is not allowed for unordered row_axis_semantics')
    if userbin_axes=='time_frequency':
        if time_edges is None: raise ValueError('time_bin_edges required')
        out=[]
        for r in range(mat.shape[0]):
            row_vals=[]
            for lo,hi in zip(time_edges[:-1], time_edges[1:]):
                m=(col_axis>=lo)&(col_axis<hi)
                if not torch.any(m):
                    if not allow_empty and empty_policy=='error': raise ValueError('empty bin encountered')
                    row_vals.append(float('nan') if empty_policy=='nan' else 0.0)
                else:
                    vv=mat[r,m]; row_vals.append(vv.mean().item() if reducer=='mean' else vv.median().item())
            out.append(row_vals)
        return FFT2DResult(matrix=torch.tensor(out), row_axis=result.row_axis, col_axis=[(a+b)/2 for a,b in zip(time_edges[:-1],time_edges[1:])], metadata={'spectral_axis':'userbin','userbin_axes':'time_frequency'})
    if userbin_axes=='row_frequency':
        if row_edges is None: raise ValueError('row_bin_edges required')
        out=[]
        for lo,hi in zip(row_edges[:-1], row_edges[1:]):
            m=(row_axis>=lo)&(row_axis<hi)
            if not torch.any(m):
                if not allow_empty and empty_policy=='error': raise ValueError('empty bin encountered')
                out.append(torch.zeros(mat.shape[1]))
            else:
                vv=mat[m,:]
                out.append(vv.mean(dim=0) if reducer=='mean' else vv.median(dim=0).values)
        return FFT2DResult(matrix=torch.stack(out), row_axis=[(a+b)/2 for a,b in zip(row_edges[:-1],row_edges[1:])], col_axis=result.col_axis, metadata={'spectral_axis':'userbin','userbin_axes':'row_frequency'})
    if userbin_axes=='both_frequency_axes':
        if row_edges is None or time_edges is None: raise ValueError('row_bin_edges and time_bin_edges required')
        tmp=fft2d_userbin(result,'row_frequency',reducer,None,row_edges,allow_empty,empty_policy,row_axis_semantics)
        return fft2d_userbin(tmp,'time_frequency',reducer,time_edges,None,allow_empty,empty_policy,row_axis_semantics)
    raise ValueError('invalid userbin_axes')
