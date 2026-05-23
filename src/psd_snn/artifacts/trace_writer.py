from __future__ import annotations
from pathlib import Path
import csv, torch

class TraceArtifactWriter:
    def __init__(self, out_dir: str):
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.rows=[]
    def write_records(self, records, chunk_size=32):
        for r in records:
            if r.series != 'spike' or r.tensor is None: continue
            x = r.tensor
            if x.ndim < 3: continue
            for s0 in range(0, x.shape[0], chunk_size):
                ch = x[s0:s0+chunk_size]
                p = self.out / f"trace_{r.layer_index}_{r.series}_{s0}.pt"
                torch.save(ch.to(torch.uint8), p)
                self.rows.append({'artifact_type':'trace_manifest','run_id':getattr(r,'run_id',None),'checkpoint_epoch':getattr(r,'checkpoint_epoch',None),'split':getattr(r,'split',None),'scope':getattr(r,'scope',None),'probe_family':getattr(r,'probe_family',None),'probe_manifest_id':getattr(r,'probe_manifest_id',None),'exclusion_family':getattr(r,'exclusion_family',None),'layer_index':r.layer_index,'layer_name':r.layer_name,'series':r.series,'path':str(p),'shape':str(tuple(ch.shape)),'dtype':'uint8','layout':'B,T,*','compression':'none','sample_start':s0,'sample_count':ch.shape[0],'time_length':ch.shape[1]})
    def write_manifest(self):
        mp = self.out/'trace_manifest.csv'
        if not self.rows: return mp
        with mp.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys())); w.writeheader(); w.writerows(self.rows)
        return mp
