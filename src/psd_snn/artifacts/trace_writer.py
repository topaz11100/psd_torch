from __future__ import annotations
from pathlib import Path
import csv, torch


class TraceArtifactWriter:
    def __init__(self, out_dir: str):
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.rows=[]

    def write_records(self, records, chunk_size=32):
        for r in records:
            status = getattr(r,'status',None)
            reason = getattr(r,'reason',None)
            if getattr(r,'series',None) != 'spike' or getattr(r,'tensor',None) is None:
                self.rows.append({'artifact_type':'trace_manifest','run_id':getattr(r,'run_id',None),'checkpoint_epoch':getattr(r,'checkpoint_epoch',None),'checkpoint_id':getattr(r,'checkpoint_id',None),'split':getattr(r,'split',None),'scope':getattr(r,'scope',None),'probe_family':getattr(r,'probe_family',None),'label':getattr(r,'label',None),'probe_manifest_id':getattr(r,'probe_manifest_id',None),'exclusion_family':getattr(r,'exclusion_family',None),'exclusion_scope':getattr(r,'exclusion_scope',None),'layer_index':getattr(r,'layer_index',None),'layer_name':getattr(r,'layer_name',None),'signal_kind':getattr(r,'signal_kind',None),'series':getattr(r,'series',None),'chunk_id':None,'path':'','layout':'B,T,*','shape':'','dtype':'','compression':'none','sample_start':None,'sample_count':None,'time_length':None,'status':status or 'unavailable_series','reason':reason or 'series is unavailable'})
                continue
            x = r.tensor
            if x.ndim < 3:
                continue
            for s0 in range(0, x.shape[0], chunk_size):
                ch = x[s0:s0+chunk_size]
                p = self.out / f"trace_{r.layer_index}_{r.series}_{s0}.pt"
                save_tensor = ch.to(torch.uint8) if r.series == 'spike' else ch
                torch.save(save_tensor, p)
                self.rows.append({'artifact_type':'trace_manifest','run_id':getattr(r,'run_id',None),'checkpoint_epoch':getattr(r,'checkpoint_epoch',None),'checkpoint_id':getattr(r,'checkpoint_id',None),'split':getattr(r,'split',None),'scope':getattr(r,'scope',None),'probe_family':getattr(r,'probe_family',None),'label':getattr(r,'label',None),'probe_manifest_id':getattr(r,'probe_manifest_id',None),'exclusion_family':getattr(r,'exclusion_family',None),'exclusion_scope':getattr(r,'exclusion_scope',None),'layer_index':r.layer_index,'layer_name':r.layer_name,'signal_kind':r.signal_kind,'series':r.series,'chunk_id':f'{r.layer_index}_{r.series}_{s0}','path':str(p),'layout':'B,T,*','shape':str(tuple(ch.shape)),'dtype':'uint8' if r.series=='spike' else str(ch.dtype),'compression':'none','sample_start':s0,'sample_count':ch.shape[0],'time_length':ch.shape[1],'status':'ok','reason':None})

    def write_manifest(self):
        mp = self.out/'trace_manifest.csv'
        if not self.rows: return mp
        with mp.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys())); w.writeheader(); w.writerows(self.rows)
        return mp
