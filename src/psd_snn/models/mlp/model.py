from __future__ import annotations
from torch import nn
from spikingjelly.activation_based import functional
from psd_snn.analysis.trace.adapter import LayerTraceRecord
from psd_snn.models.readout.final import final_if_from_membrane, final_mem_from_membrane

class MLPStackModel(nn.Module):
    def __init__(self, blocks, output_block, readout_kind='final_mem'):
        super().__init__(); self.blocks=nn.ModuleList(blocks); self.output_block=output_block; self.readout_kind=readout_kind
        self.scenario='none'; self.constraint_hash=None; self.constraint_spec={}
    def reset_spiking_state(self): functional.reset_net(self)
    def layer_group_ids(self): return [b.layer_group_ids for b in self.blocks]
    def layer_constraint_metadata(self):
        return [{'layer_index':i,'feedforward_mask_applied':b.has_feedforward_mask,'recurrent_mask_applied':b.has_recurrent_mask,'layer_group_ids':b.layer_group_ids} for i,b in enumerate(self.blocks)]
    def forward(self, x_btf, capture_trace=False, probe_family='unknown', label='na'):
        traces=[]; h=x_btf
        for li,block in enumerate(self.blocks):
            h,tr=block(h,capture_trace=capture_trace)
            if capture_trace:
                for series,ten in tr.items():
                    summary={'alpha_bounds_applied': getattr(block.cell,'_a_lo',None) is not None,'frequency_bounds_applied': getattr(block.cell,'_f_lo',None) is not None,'damping_bounds_applied': getattr(block.cell,'_d_lo',None) is not None,'threshold_bounds_applied': getattr(block.cell,'_t_lo',None) is not None}
                    rec=LayerTraceRecord(li,f'hidden_{li}','hidden',series,'layer',probe_family,label,ten,self.scenario,self.constraint_hash,block.layer_group_ids,block.has_feedforward_mask,block.has_recurrent_mask)
                    rec.cell_bounds_summary=summary
                    traces.append(rec)
        _,out_tr=self.output_block(h,capture_trace=True)
        logits = final_if_from_membrane(out_tr['membrane_pre']) if self.readout_kind=='final_if' else final_mem_from_membrane(out_tr['membrane_post'])
        out_tr['logits']=logits
        if capture_trace:
            for series,ten in out_tr.items():
                traces.append(LayerTraceRecord(len(self.blocks),'output','output',series,'layer',probe_family,label,ten if ten.ndim>=3 else ten.unsqueeze(1),self.scenario,self.constraint_hash,None,False,False))
            return logits,traces
        return logits
