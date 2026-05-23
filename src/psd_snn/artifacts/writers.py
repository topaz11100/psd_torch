from __future__ import annotations
from pathlib import Path
import csv
from psd_snn.analysis.common.status import AnalysisFailure


COMMON_META_KEYS = [
    'artifact_type','run_id','checkpoint_epoch','checkpoint_id','split','scope','probe_family','label',
    'probe_manifest_id','exclusion_family','exclusion_scope','layer_index','layer_name','signal_kind','series',
    'scenario','constraint_hash','status','reason'
]


def _common(meta: dict, artifact_type: str, status: str = 'ok', reason: str | None = None) -> dict:
    return {
        'artifact_type': artifact_type,
        'run_id': meta.get('run_id'),
        'checkpoint_epoch': meta.get('checkpoint_epoch'),
        'checkpoint_id': meta.get('checkpoint_id'),
        'split': meta.get('split'),
        'scope': meta.get('scope'),
        'probe_family': meta.get('probe_family'),
        'label': meta.get('label'),
        'probe_manifest_id': meta.get('probe_manifest_id'),
        'exclusion_family': meta.get('exclusion_family'),
        'exclusion_scope': meta.get('exclusion_scope'),
        'layer_index': meta.get('layer_index'),
        'layer_name': meta.get('layer_name'),
        'signal_kind': meta.get('signal_kind'),
        'series': meta.get('series'),
        'scenario': meta.get('scenario'),
        'constraint_hash': meta.get('constraint_hash'),
        'status': status,
        'reason': reason,
    }


class SummaryWriter:
    def __init__(self, out_dir: str):
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)

    def _analysis_manifest_row(self, r, artifact_path: str = '', status='ok', reason=None):
        m = r.get('meta', {})
        row = _common(m, 'analysis_manifest', status=status, reason=reason)
        row.update({
            'analysis_method': r.get('analysis_method', 'psd'),
            'representative': r.get('representative'),
            'spectral_axis': r.get('spectral_axis', r.get('axis_policy')),
            'output_dir': str(self.out),
            'artifact_path': artifact_path,
        })
        return row

    def write_failure(self, failure: AnalysisFailure):
        p = self.out / 'analysis_manifest.csv'
        row = dict(_common(failure.to_manifest_row(), 'analysis_manifest', status=failure.status, reason=failure.reason))
        row.update({
            'analysis_method': None, 'representative': None, 'spectral_axis': None,
            'output_dir': str(self.out), 'artifact_path': failure.artifact_path or '',
            'error_type': failure.error_type, 'error_message': failure.error_message,
        })
        self._append_rows(p, [row])
        return p

    def _append_rows(self, path: Path, rows: list[dict]):
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        if path.exists():
            with path.open() as f:
                reader = csv.DictReader(f)
                old = list(reader)
                if old:
                    fieldnames = list(dict.fromkeys(list(old[0].keys()) + fieldnames))
                    rows = old + rows
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(rows)

    def write_results(self, results):
        curve=[]; mat=[]; pca=[]; mat2d=[]; rowax=[]; colax=[]; manifest=[]
        for r in results:
            meta=r.get('meta',{})
            if r['type']=='spectral_curve':
                for i,(f,v) in enumerate(zip(r['freq'], r['power'])):
                    curve.append({**_common(meta,'spectral_curve'),'representative':r['representative'],'component_id':r.get('component_id'),'pca_basis_id':r.get('pca_basis_id'),'frequency_index':i,'frequency':f,'value':v,'spectral_axis':r['axis_policy']})
                manifest.append(self._analysis_manifest_row({**r,'analysis_method':'psd'}, artifact_path='spectral_curve.csv'))
            elif r['type']=='spectral_matrix_1d':
                for ri,row in enumerate(r['matrix']):
                    for fi,v in enumerate(row):
                        mat.append({**_common(meta,'spectral_matrix_1d'),'representative':r['representative'],'component_id':ri if r['representative']=='pca' else None,'pca_basis_id':r.get('pca_basis_id'),'row_index':ri,'frequency_index':fi,'value':v,'spectral_axis':r['spectral_axis']})
                manifest.append(self._analysis_manifest_row({**r,'analysis_method':'psd'}, artifact_path='spectral_matrix_1d.csv'))
            elif r['type']=='pca_basis':
                ev=r.get('explained_variance',[]); evr=r.get('explained_variance_ratio',[])
                for i in range(len(ev)):
                    pca.append({**_common(meta,'pca_basis'),'pca_basis_id':r['pca_basis_id'],'reference_checkpoint_epoch':r.get('reference_checkpoint_epoch'),'reference_checkpoint_id':r.get('reference_checkpoint_id'),'reference_split':r.get('reference_split'),'reference_scope':r.get('reference_scope'),'reference_probe_family':r.get('reference_probe_family'),'component_id':i,'explained_variance':ev[i],'explained_variance_ratio':evr[i] if i < len(evr) else None,'basis_artifact_path':r.get('basis_artifact_path')})
                manifest.append(self._analysis_manifest_row({**r,'analysis_method':'psd'}, artifact_path='pca_basis.csv'))
            elif r['type']=='spectral_matrix_2d':
                matrix_id=r.get('matrix_id','m0')
                for ri,row in enumerate(r['matrix']):
                    rec={**_common(meta,'spectral_matrix_2d'),'matrix_id':matrix_id,'row_index':ri,'row_value':r['row_axis'][ri],'spectral_axis':r['spectral_axis'],'row_axis_semantics':r.get('row_axis_semantics'),'userbin_axes':r.get('userbin_axes')}
                    for ci,v in enumerate(row): rec[f'time_freq_{ci:06d}']=v
                    mat2d.append(rec)
                for ri,rv in enumerate(r['row_axis']): rowax.append({**_common(meta,'spectral_matrix_2d_row_axis'),'matrix_id':matrix_id,'row_index':ri,'row_value':rv,'row_axis_semantics':r.get('row_axis_semantics')})
                for ci,cv in enumerate(r['col_axis']): colax.append({**_common(meta,'spectral_matrix_2d_column_axis'),'matrix_id':matrix_id,'column_index':ci,'column_value':cv})
                manifest.append(self._analysis_manifest_row({**r,'analysis_method':'fft2d'}, artifact_path='spectral_matrix_2d.csv'))
        for name,rows in [('spectral_curve.csv',curve),('spectral_matrix_1d.csv',mat),('pca_basis.csv',pca),('spectral_matrix_2d.csv',mat2d),('spectral_matrix_2d_row_axis.csv',rowax),('spectral_matrix_2d_column_axis.csv',colax),('analysis_manifest.csv',manifest)]:
            p=self.out/name
            if rows:
                with p.open('w', newline='') as f:
                    w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
