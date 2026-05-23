from __future__ import annotations
from pathlib import Path
import csv

class SummaryWriter:
    def __init__(self, out_dir: str):
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)

    def _analysis_manifest_row(self, r):
        m=r.get("meta",{})
        return {"artifact_type":"analysis_manifest","run_id":m.get("run_id"),"checkpoint_epoch":m.get("checkpoint_epoch"),"split":m.get("split"),"scope":m.get("scope"),"probe_family":m.get("probe_family"),"analysis_method": r.get("analysis_method","psd"),"representative":r.get("representative"),"spectral_axis":r.get("spectral_axis", r.get("axis_policy")),"output_dir":str(self.out),"artifact_path":"","status":"ok","reason":None}

    def write_results(self, results):
        curve=[]; mat=[]; pca=[]; mat2d=[]; rowax=[]; colax=[]; manifest=[]
        for r in results:
            meta=r.get('meta',{})
            common={'run_id':meta.get('run_id'),'checkpoint_epoch':meta.get('checkpoint_epoch'),'split':meta.get('split'),'scope':meta.get('scope'),'probe_family':meta.get('probe_family'),'layer_name':meta.get('layer_name'),'layer_index':meta.get('layer_index'),'signal_kind':meta.get('signal_kind'),'series':meta.get('series'),'status':'ok','reason':None}
            if r['type']=='spectral_curve':
                for i,(f,v) in enumerate(zip(r['freq'], r['power'])):
                    manifest.append(self._analysis_manifest_row({**r,'analysis_method':'psd'})); curve.append({'artifact_type':'spectral_curve','representative':r['representative'],'component_id':r.get('component_id'),'pca_basis_id':r.get('pca_basis_id'),'frequency_index':i,'frequency':f,'value':v,'spectral_axis':r['axis_policy'], **common})
            elif r['type']=='spectral_matrix_1d':
                for ri,row in enumerate(r['matrix']):
                    for fi,v in enumerate(row):
                        manifest.append(self._analysis_manifest_row({**r,'analysis_method':'psd'})); mat.append({'artifact_type':'spectral_matrix_1d','representative':r['representative'],'component_id':ri if r['representative']=='pca' else None,'pca_basis_id':r.get('pca_basis_id'),'row_index':ri,'frequency_index':fi,'value':v,'spectral_axis':r['spectral_axis'], **common})
            elif r['type']=='pca_basis':
                ev=r.get('explained_variance',[]); evr=r.get('explained_variance_ratio',[])
                for i in range(len(ev)):
                    pca.append({'artifact_type':'pca_basis','run_id':meta.get('run_id'),'pca_basis_id':r['pca_basis_id'],'reference_checkpoint_epoch':r.get('reference_checkpoint_epoch'),'reference_split':r.get('reference_split'),'reference_scope':r.get('reference_scope'),'reference_probe_family':r.get('reference_probe_family'),'layer_index':r.get('layer_index'),'layer_name':r.get('layer_name'),'signal_kind':r.get('signal_kind'),'series':r.get('series'),'component_id':i,'explained_variance':ev[i],'explained_variance_ratio':evr[i] if i < len(evr) else None,'basis_artifact_path':r.get('basis_artifact_path')})
            elif r['type']=='spectral_matrix_2d':
                matrix_id=r.get('matrix_id','m0')
                for ri,row in enumerate(r['matrix']):
                    rec={'artifact_type':'spectral_matrix_2d','matrix_id':matrix_id,'row_index':ri,'row_value':r['row_axis'][ri],'spectral_axis':r['spectral_axis'],'row_axis_semantics':r.get('row_axis_semantics'),'userbin_axes':r.get('userbin_axes'), **common}
                    for ci,v in enumerate(row): rec[f'time_freq_{ci:06d}']=v
                    manifest.append(self._analysis_manifest_row({**r,'analysis_method':'fft2d'})); mat2d.append(rec)
                for ri,rv in enumerate(r['row_axis']): rowax.append({'artifact_type':'spectral_matrix_2d_row_axis','matrix_id':matrix_id,'row_index':ri,'row_value':rv,'row_axis_semantics':r.get('row_axis_semantics')})
                for ci,cv in enumerate(r['col_axis']): colax.append({'artifact_type':'spectral_matrix_2d_column_axis','matrix_id':matrix_id,'column_index':ci,'column_value':cv})
        for name,rows in [('spectral_curve.csv',curve),('spectral_matrix_1d.csv',mat),('pca_basis.csv',pca),('spectral_matrix_2d.csv',mat2d),('spectral_matrix_2d_row_axis.csv',rowax),('spectral_matrix_2d_column_axis.csv',colax),('analysis_manifest.csv',manifest)]:
            p=self.out/name
            if rows:
                with p.open('w', newline='') as f:
                    w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
