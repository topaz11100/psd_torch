from __future__ import annotations
from pathlib import Path
import csv


def matrix_distance(a, b, metric='centered_l2', diff_axis='time_frequency'):
    import math
    import torch
    A=torch.tensor(a); B=torch.tensor(b)
    if metric=='centered_l2':
        da=(A-A.mean()).reshape(-1); db=(B-B.mean()).reshape(-1)
        return float(torch.sqrt(((da-db)**2).sum()))
    if metric=='diff_l2':
        if diff_axis=='time_frequency':
            da=A[:,1:]-A[:,:-1]; db=B[:,1:]-B[:,:-1]
        elif diff_axis=='row_frequency':
            da=A[1:,:]-A[:-1,:]; db=B[1:,:]-B[:-1,:]
        else:
            da=torch.cat([(A[:,1:]-A[:,:-1]).reshape(-1),(A[1:,:]-A[:-1,:]).reshape(-1)])
            db=torch.cat([(B[:,1:]-B[:,:-1]).reshape(-1),(B[1:,:]-B[:-1,:]).reshape(-1)])
            return float(torch.sqrt(((da-db)**2).sum()))
        return float(torch.sqrt(((da-db)**2).sum()))
    raise ValueError('unsupported metric')

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
                    manifest.append(self._analysis_manifest_row({**r,'analysis_method':'psd'})); curve.append({'artifact_type':'spectral_curve','representative':r['representative'],'pca_basis_id':r.get('pca_basis_id'),'frequency_index':i,'frequency':f,'value':v,'spectral_axis':r['axis_policy'], **common})
            elif r['type']=='spectral_matrix_1d':
                for ri,row in enumerate(r['matrix']):
                    for fi,v in enumerate(row):
                        manifest.append(self._analysis_manifest_row({**r,'analysis_method':'psd'})); mat.append({'artifact_type':'spectral_matrix_1d','representative':r['representative'],'row_index':ri,'frequency_index':fi,'value':v,'spectral_axis':r['spectral_axis'], **common})
            elif r['type']=='pca_basis':
                pca.append({'artifact_type':'pca_basis','pca_basis_id':r['pca_basis_id'],'n_rows':len(r['basis']),'n_cols':len(r['basis'][0]) if r['basis'] else 0})
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
