from __future__ import annotations
from dataclasses import dataclass
import hashlib
import torch
from psd_snn.analysis.signal_map.emitter import bt_to_srt
from psd_snn.analysis.spectral.accumulator import PSDAccumulator
from psd_snn.analysis.signal.fft2d import fft2d_exact, fft2d_userbin
from psd_snn.analysis.signal.pca_basis_store import PCABasisStore, PCABasisKey, PCAFitRequest, PCAApplyRequest

@dataclass
class SignalMapRecord:
    maps: torch.Tensor
    metadata: dict


def _apply_window(x, window):
    if window == 'hann':
        w = torch.hann_window(x.shape[-1], device=x.device, dtype=x.dtype)
        return x * w
    return x


def _to_power(maps, centering, window):
    x = maps
    if centering == 'centered':
        x = x - x.mean(dim=-1, keepdim=True)
    x = _apply_window(x, window)
    fft = torch.fft.rfft(x, dim=-1)
    power = (fft.real ** 2 + fft.imag ** 2)
    return power


def _basis_id(w, mu):
    h = hashlib.sha256()
    h.update(w.detach().cpu().numpy().tobytes())
    h.update(mu.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


class SignalAnalysisRunner:
    def __init__(self, spec):
        self.spec = spec
        self.results = []
        self.pca_basis_store = PCABasisStore(getattr(getattr(spec, "artifact", None), "output_dir", None))

    def update_signal_maps(self, map_records: list[SignalMapRecord]):
        ps = self.spec.psd
        for rec in map_records:
            maps = rec.maps  # S,R,T
            p = _to_power(maps, ps.centering, ps.window)
            freq = torch.fft.rfftfreq(maps.shape[-1]).tolist()
            method = ps.representative.method
            if method == 'mean':
                sxf = p.mean(dim=1)
                acc = PSDAccumulator(axis_policy=ps.spectral_axis, userbin_edges=ps.userbin_edges, userbin_reducer=ps.userbin_reducer, allow_empty_bins=ps.allow_empty_bins, empty_bin_fill='nan' if ps.empty_bin_policy=='nan' else 'zero')
                acc.update(freq, sxf.tolist())
                self.results.append({'type':'spectral_curve','representative':'mean',**acc.finalize(to_db=(ps.scale_outputs in {'db','both'})), 'meta':rec.metadata})
            elif method == 'median':
                sxf = p.median(dim=1).values
                acc = PSDAccumulator(axis_policy=ps.spectral_axis, userbin_edges=ps.userbin_edges, userbin_reducer=ps.userbin_reducer, allow_empty_bins=ps.allow_empty_bins, empty_bin_fill='nan' if ps.empty_bin_policy=='nan' else 'zero')
                acc.update(freq, sxf.tolist()); self.results.append({'type':'spectral_curve','representative':'median',**acc.finalize(to_db=(ps.scale_outputs in {'db','both'})), 'meta':rec.metadata})
            elif method == 'element_psd':
                mat = p.mean(dim=0)  # R,F
                if ps.spectral_axis == 'userbin':
                    rows=[]
                    for r in range(mat.shape[0]):
                        acc=PSDAccumulator(axis_policy='userbin', userbin_edges=ps.userbin_edges, userbin_reducer=ps.userbin_reducer, allow_empty_bins=ps.allow_empty_bins, empty_bin_fill='nan' if ps.empty_bin_policy=='nan' else 'zero')
                        acc.update(freq, [mat[r].tolist()]); out=acc.finalize(to_db=(ps.scale_outputs in {'db','both'})); rows.append(out['power']); uf=out['freq']
                    self.results.append({'type':'spectral_matrix_1d','representative':'element_psd','matrix':rows,'freq':uf,'spectral_axis':'userbin','meta':rec.metadata})
                else:
                    self.results.append({'type':'spectral_matrix_1d','representative':'element_psd','matrix':mat.tolist(),'freq':freq,'spectral_axis':'exact','meta':rec.metadata})
            elif method == 'pca':
                k = ps.representative.pca.n_components
                pca = ps.representative.pca
                md = rec.metadata
                key = PCABasisKey(reference_checkpoint_epoch=pca.reference_checkpoint, reference_checkpoint_id=md.get('reference_checkpoint_id'), reference_split=md.get('reference_split', md.get('split','unknown')), reference_scope=md.get('reference_scope', md.get('scope','unknown')), reference_probe_family=md.get('reference_probe_family', md.get('probe_family','unknown')), layer_index=int(md.get('layer_index',0)), layer_name=str(md.get('layer_name','unknown')), signal_kind=str(md.get('signal_kind','hidden')), series=str(md.get('series','spike')), n_components=k, row_count=int(maps.shape[1]), centering=bool(pca.center))
                if pca.basis_mode == 'fixed_reference':
                    y, basis = self.pca_basis_store.apply(PCAApplyRequest(maps=maps, key=key))
                    basis_id = basis.basis_id
                else:
                    basis = self.pca_basis_store.fit(PCAFitRequest(maps=maps, key=key, created_from=md))
                    self.pca_basis_store.save_tensor_artifact(basis)
                    basis_id = basis.basis_id
                    y = ((maps.permute(0,2,1) - basis.mean) @ basis.components).permute(0,2,1)
                pk = _to_power(y, ps.centering, ps.window).mean(dim=0)  # K,F
                if ps.spectral_axis == 'userbin':
                    rows=[]
                    for ci in range(pk.shape[0]):
                        acc = PSDAccumulator(axis_policy='userbin', userbin_edges=ps.userbin_edges, userbin_reducer=ps.userbin_reducer, allow_empty_bins=ps.allow_empty_bins, empty_bin_fill='nan' if ps.empty_bin_policy=='nan' else 'zero')
                        acc.update(freq, [pk[ci].tolist()])
                        out = acc.finalize(to_db=(ps.scale_outputs in {'db','both'}))
                        rows.append(out['power']); uf = out['freq']
                    self.results.append({'type':'spectral_matrix_1d','representative':'pca','pca_basis_id':basis_id,'matrix':rows,'freq':uf,'spectral_axis':'userbin','meta':{**md,'pca_basis_mode':pca.basis_mode,'pca_reference_checkpoint_epoch':key.reference_checkpoint_epoch,'pca_reference_split':key.reference_split,'pca_reference_scope':key.reference_scope,'pca_reference_probe_family':key.reference_probe_family,'n_components':k,'sign_convention':key.sign_convention}})
                else:
                    self.results.append({'type':'spectral_matrix_1d','representative':'pca','pca_basis_id':basis_id,'matrix':pk.tolist(),'freq':freq,'spectral_axis':'exact','meta':{**md,'pca_basis_mode':pca.basis_mode,'pca_reference_checkpoint_epoch':key.reference_checkpoint_epoch,'pca_reference_split':key.reference_split,'pca_reference_scope':key.reference_scope,'pca_reference_probe_family':key.reference_probe_family,'n_components':k,'sign_convention':key.sign_convention}})
                self.results.append({'type':'pca_basis','pca_basis_id':basis_id,'basis':basis.components.tolist(),'mean':basis.mean.squeeze(0).tolist(),'row_count':basis.row_count,'n_components':basis.n_components,'reference_checkpoint_epoch':key.reference_checkpoint_epoch,'reference_split':key.reference_split,'reference_scope':key.reference_scope,'reference_probe_family':key.reference_probe_family,'layer_index':key.layer_index,'layer_name':key.layer_name,'signal_kind':key.signal_kind,'series':key.series,'centering':key.centering,'sign_convention':key.sign_convention,'explained_variance':basis.explained_variance.tolist(),'explained_variance_ratio':basis.explained_variance_ratio.tolist(),'basis_artifact_path':basis.artifact_path,'meta':rec.metadata})

    def run_fft2d(self, map_records: list[SignalMapRecord]):
        f2=self.spec.fft2d
        for rec in map_records:
            ex=fft2d_exact(rec.maps, centering=f2.centering, window_time=f2.window_time, window_row=f2.window_row, to_db=(f2.scale_outputs in {'db','both'}))
            if f2.spectral_axis=='exact':
                self.results.append({'type':'spectral_matrix_2d','matrix':ex.matrix.tolist(),'row_axis':ex.row_axis,'col_axis':ex.col_axis,'spectral_axis':'exact','row_axis_semantics':f2.row_axis.row_axis_semantics,'meta':rec.metadata})
            else:
                ub=fft2d_userbin(ex, userbin_axes=f2.userbin.userbin_axes, reducer=f2.userbin_reducer, time_edges=f2.userbin.time_bin_edges, row_edges=f2.userbin.row_bin_edges, allow_empty=f2.allow_empty_bins, empty_policy=f2.empty_bin_policy, row_axis_semantics=f2.row_axis.row_axis_semantics)
                self.results.append({'type':'spectral_matrix_2d','matrix':ub.matrix.tolist(),'row_axis':ub.row_axis,'col_axis':ub.col_axis,'spectral_axis':'userbin','userbin_axes':f2.userbin.userbin_axes,'row_axis_semantics':f2.row_axis.row_axis_semantics,'meta':rec.metadata})

    def finalize(self):
        return self.results
