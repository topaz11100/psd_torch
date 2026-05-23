from __future__ import annotations
from .base import require_torch, require_spikingjelly, CellStepTrace, TraceableMemoryNode
from psd_snn.models.constraints.bounds import bounded_sigmoid, inverse_bounded_sigmoid, materialize_feature_bounds
_torch=require_torch(); _nn=_torch.nn; _base=require_spikingjelly('LIFCell')
class LIFCell(_base.MemoryModule, TraceableMemoryNode):
    def __init__(self, features, alpha=0.9, alpha_trainable=False, threshold=1.0, threshold_trainable=False, reset_mode='soft', alpha_bounds=None, threshold_bounds=None, group_ids=None):
        super().__init__(); self.reset_mode=reset_mode; self.group_ids=group_ids; self.register_memory('v',None)
        alo,ahi=(None,None)
        if alpha_bounds is not None: alo,ahi=materialize_feature_bounds((alpha_bounds.lower,alpha_bounds.upper),features)
        self._a_lo,self._a_hi=alo,ahi
        if alpha_trainable:
            raw=inverse_bounded_sigmoid(_torch.full((features,),alpha),alo,ahi) if alo is not None else _torch.logit(_torch.full((features,),min(max(alpha,1e-4),1-1e-4)))
            self.alpha_raw=_nn.Parameter(raw)
        else:
            a=_torch.full((features,),alpha); 
            if alo is not None and (_torch.any(a<alo) or _torch.any(a>ahi)): raise ValueError('fixed alpha outside bounds')
            self.alpha_raw=_nn.Parameter(a.requires_grad_(False))
        tlo,thi=(None,None)
        if threshold_bounds is not None: tlo,thi=materialize_feature_bounds((threshold_bounds.lower,threshold_bounds.upper),features)
        self._t_lo,self._t_hi=tlo,thi
        if threshold_trainable:
            tr=inverse_bounded_sigmoid(_torch.full((features,),threshold),tlo,thi) if tlo is not None else _torch.full((features,),threshold)
            self.threshold_raw=_nn.Parameter(tr); self.register_parameter('threshold',None)
        else:
            th=_torch.full((features,),threshold); 
            if tlo is not None and (_torch.any(th<tlo) or _torch.any(th>thi)): raise ValueError('fixed threshold outside bounds')
            self.threshold=_nn.Parameter(th,requires_grad=False); self.register_parameter('threshold_raw',None)
    @property
    def alpha(self):
        return bounded_sigmoid(self.alpha_raw,self._a_lo,self._a_hi) if self._a_lo is not None else (_torch.sigmoid(self.alpha_raw) if self.alpha_raw.requires_grad else self.alpha_raw)
    @property
    def effective_threshold(self):
        if self.threshold_raw is None: return self.threshold
        return bounded_sigmoid(self.threshold_raw,self._t_lo,self._t_hi) if self._t_lo is not None else self.threshold_raw
    def single_step_forward(self,input_current):
        if self.v is None: self.v=_torch.zeros_like(input_current)
        a=self.alpha.to(input_current.device,input_current.dtype); th=self.effective_threshold.to(input_current.device,input_current.dtype)
        mp=a*self.v+input_current; d=mp-th; s=(d>=0).to(mp.dtype)
        mpost = _torch.where(s>0,_torch.zeros_like(mp),mp) if self.reset_mode=='hard' else (mp-th*s if self.reset_mode=='soft' else mp)
        self.v=mpost; return CellStepTrace(input_current,mp,d,s,mpost)
    def forward_sequence(self,seq,capture_trace=True):
        tr={k:[] for k in self.state_trace_names()}; out=[]
        for t in range(seq.shape[1]): st=self.single_step_forward(seq[:,t,:]); out.append(st.spike); [tr[k].append(getattr(st,k)) for k in tr] if capture_trace else None
        y=_torch.stack(out,1); return (y,{k:_torch.stack(v,1) for k,v in tr.items()}) if capture_trace else (y,None)
    def analysis_parameter_vectors(self):
        a=self.alpha.detach(); tau=(-1.0/_torch.log(a.clamp(1e-6,1-1e-6))).detach(); th=self.effective_threshold.detach()
        return {'membrane_decay_alpha':{'name':'membrane_decay_alpha','role':'lif_membrane_decay','values':a,'unit':'dimensionless','trainable':self.alpha_raw.requires_grad,'lower_bound':self._a_lo,'upper_bound':self._a_hi,'group_ids':self.group_ids},'membrane_time_constant_tau_step':{'name':'membrane_time_constant_tau_step','role':'time_constant','values':tau,'unit':'step','trainable':self.alpha_raw.requires_grad,'lower_bound':None,'upper_bound':None,'group_ids':self.group_ids},'threshold':{'name':'threshold','role':'firing_threshold','values':th,'unit':'membrane','trainable':self.threshold_raw is not None,'lower_bound':self._t_lo,'upper_bound':self._t_hi,'group_ids':self.group_ids}}
