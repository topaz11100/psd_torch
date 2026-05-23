from __future__ import annotations
from .base import require_torch, require_spikingjelly, CellStepTrace, TraceableMemoryNode
from psd_snn.models.constraints.bounds import bounded_sigmoid, inverse_bounded_sigmoid, materialize_feature_bounds
_torch=require_torch(); _nn=_torch.nn; _base=require_spikingjelly('IFCell')
class IFCell(_base.MemoryModule, TraceableMemoryNode):
    def __init__(self, features, threshold=1.0, threshold_trainable=False, reset_mode='soft', reset_value=0.0, threshold_bounds=None, group_ids=None):
        super().__init__(); self.reset_mode=reset_mode; self.reset_value=reset_value; self.group_ids=group_ids
        self.register_memory('v', None)
        lo,hi=(None,None)
        if threshold_bounds is not None: lo,hi=materialize_feature_bounds((threshold_bounds.lower,threshold_bounds.upper),features)
        self._th_lo,self._th_hi=lo,hi
        if threshold_trainable:
            raw = inverse_bounded_sigmoid(_torch.full((features,),threshold), lo, hi) if lo is not None else _torch.full((features,),threshold)
            self.threshold_raw=_nn.Parameter(raw)
            self.register_parameter('threshold', None)
        else:
            th = _torch.full((features,),threshold)
            if lo is not None and (_torch.any(th<lo) or _torch.any(th>hi)): raise ValueError('fixed threshold outside bounds')
            self.register_parameter('threshold_raw', None); self.threshold=_nn.Parameter(th, requires_grad=False)
    @property
    def effective_threshold(self):
        if self.threshold_raw is None: return self.threshold
        if self._th_lo is None: return self.threshold_raw
        return bounded_sigmoid(self.threshold_raw,self._th_lo,self._th_hi)
    def single_step_forward(self,input_current):
        if self.v is None: self.v=_torch.zeros_like(input_current)
        th=self.effective_threshold.to(input_current.device,input_current.dtype)
        membrane_pre=self.v+input_current; decision=membrane_pre-th; spike=(decision>=0).to(membrane_pre.dtype)
        membrane_post = _torch.where(spike>0,_torch.full_like(membrane_pre,self.reset_value),membrane_pre) if self.reset_mode=='hard' else (membrane_pre-th*spike if self.reset_mode=='soft' else membrane_pre)
        self.v=membrane_post; return CellStepTrace(input_current,membrane_pre,decision,spike,membrane_post)
    def forward_sequence(self,seq,capture_trace=True):
        tr={k:[] for k in self.state_trace_names()}; out=[]
        for t in range(seq.shape[1]): st=self.single_step_forward(seq[:,t,:]); out.append(st.spike); [tr[k].append(getattr(st,k)) for k in tr] if capture_trace else None
        y=_torch.stack(out,1); return (y,{k:_torch.stack(v,1) for k,v in tr.items()}) if capture_trace else (y,None)
    def analysis_parameter_vectors(self):
        th=self.effective_threshold.detach(); return {'threshold': {'name':'threshold','role':'firing_threshold','values':th,'unit':'membrane','trainable':self.threshold_raw is not None,'lower_bound':self._th_lo,'upper_bound':self._th_hi,'group_ids':self.group_ids}}
