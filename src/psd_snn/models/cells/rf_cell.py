from __future__ import annotations
from .base import require_torch, require_spikingjelly, CellStepTrace, TraceableMemoryNode
from psd_snn.models.constraints.bounds import bounded_sigmoid, inverse_bounded_sigmoid, positive_parameter, inverse_positive_parameter, materialize_feature_bounds
_torch=require_torch(); _nn=_torch.nn; _base=require_spikingjelly('RFCell')
class RFCell(_base.MemoryModule, TraceableMemoryNode):
    def __init__(self, features, omega=1.0, damping=0.1, threshold=1.0, threshold_trainable=False, reset_mode='threshold_only', scale_factor=0.5, frequency_bounds=None, damping_bounds=None, threshold_bounds=None, group_ids=None, dt: float=1.0):
        super().__init__(); self.reset_mode=reset_mode; self.scale_factor=scale_factor; self.group_ids=group_ids; self.dt=dt
        self.register_memory('x',None); self.register_memory('y',None)
        flo,fhi=(None,None)
        if frequency_bounds is not None: flo,fhi=materialize_feature_bounds((frequency_bounds.lower,frequency_bounds.upper),features)
        dlo,dhi=(None,None)
        if damping_bounds is not None: dlo,dhi=materialize_feature_bounds((damping_bounds.lower,damping_bounds.upper),features)
        self._f_lo,self._f_hi=flo,fhi; self._d_lo,self._d_hi=dlo,dhi
        freq_init=_torch.full((features,),omega/(2*_torch.pi))
        self.frequency_raw=_nn.Parameter(inverse_bounded_sigmoid(freq_init,flo,fhi) if flo is not None else inverse_positive_parameter(freq_init), requires_grad=True)
        damp_init=_torch.full((features,),damping)
        self.damping_raw=_nn.Parameter(inverse_bounded_sigmoid(damp_init,dlo,dhi) if dlo is not None else inverse_positive_parameter(damp_init), requires_grad=True)
        tlo,thi=(None,None)
        if threshold_bounds is not None: tlo,thi=materialize_feature_bounds((threshold_bounds.lower,threshold_bounds.upper),features)
        self._t_lo,self._t_hi=tlo,thi
        if threshold_trainable:
            self.threshold_raw=_nn.Parameter(inverse_bounded_sigmoid(_torch.full((features,),threshold),tlo,thi) if tlo is not None else _torch.full((features,),threshold)); self.register_parameter('threshold',None)
        else:
            th=_torch.full((features,),threshold); 
            if tlo is not None and (_torch.any(th<tlo) or _torch.any(th>thi)): raise ValueError('fixed threshold outside bounds')
            self.threshold=_nn.Parameter(th,requires_grad=False); self.register_parameter('threshold_raw',None)
    def state_trace_names(self): return ['input_current','membrane_pre','decision','spike','membrane_post','rf_real_pre','rf_imag_pre','rf_real_post','rf_imag_post']
    @property
    def resonant_frequency_cyc_per_step(self): return bounded_sigmoid(self.frequency_raw,self._f_lo,self._f_hi) if self._f_lo is not None else positive_parameter(self.frequency_raw)
    @property
    def omega_rad_per_step(self): return 2*_torch.pi*self.resonant_frequency_cyc_per_step
    @property
    def damping_magnitude(self): return bounded_sigmoid(self.damping_raw,self._d_lo,self._d_hi) if self._d_lo is not None else positive_parameter(self.damping_raw)
    @property
    def decay_radius(self): return _torch.exp(-self.damping_magnitude*self.dt)
    @property
    def effective_threshold(self): return self.threshold if self.threshold_raw is None else (bounded_sigmoid(self.threshold_raw,self._t_lo,self._t_hi) if self._t_lo is not None else self.threshold_raw)
    def single_step_forward(self,input_current):
        if self.x is None: self.x=_torch.zeros_like(input_current); self.y=_torch.zeros_like(input_current)
        c,s=_torch.cos(self.omega_rad_per_step),_torch.sin(self.omega_rad_per_step); r=self.decay_radius
        rp=r*(self.x*c-self.y*s)+input_current; ip=r*(self.x*s+self.y*c); th=self.effective_threshold.to(rp.device,rp.dtype)
        d=rp-th; sp=(d>=0).to(rp.dtype)
        if self.reset_mode=='hard_state': ro,io=_torch.where(sp>0,_torch.zeros_like(rp),rp),_torch.where(sp>0,_torch.zeros_like(ip),ip)
        elif self.reset_mode=='hard_real': ro,io=_torch.where(sp>0,_torch.zeros_like(rp),rp),ip
        elif self.reset_mode=='soft_real': ro,io=rp-th*sp,ip
        elif self.reset_mode=='scale_state': ro,io=rp*(1-self.scale_factor*sp),ip*(1-self.scale_factor*sp)
        elif self.reset_mode in {'threshold_only','none'}: ro,io=rp,ip
        else: raise ValueError('invalid RF reset_mode')
        self.x,self.y=ro,io
        return CellStepTrace(input_current,rp,d,sp,ro,rp,ip,ro,io)
    def forward_sequence(self,seq,capture_trace=True):
        tr={k:[] for k in self.state_trace_names()}; out=[]
        for t in range(seq.shape[1]): st=self.single_step_forward(seq[:,t,:]); out.append(st.spike); [tr[k].append(getattr(st,k)) for k in tr] if capture_trace else None
        y=_torch.stack(out,1); return (y,{k:_torch.stack(v,1) for k,v in tr.items()}) if capture_trace else (y,None)
    def analysis_parameter_vectors(self):
        return {'resonant_frequency_cyc_per_step':{'name':'resonant_frequency_cyc_per_step','role':'frequency','values':self.resonant_frequency_cyc_per_step.detach(),'unit':'cycle_per_step','trainable':self.frequency_raw.requires_grad,'lower_bound':self._f_lo,'upper_bound':self._f_hi,'group_ids':self.group_ids},'omega_rad_per_step':{'name':'omega_rad_per_step','role':'omega','values':self.omega_rad_per_step.detach(),'unit':'rad_per_step','trainable':self.frequency_raw.requires_grad,'lower_bound':None,'upper_bound':None,'group_ids':self.group_ids},'damping_magnitude':{'name':'damping_magnitude','role':'damping','values':self.damping_magnitude.detach(),'unit':'per_step','trainable':self.damping_raw.requires_grad,'lower_bound':self._d_lo,'upper_bound':self._d_hi,'group_ids':self.group_ids},'decay_radius':{'name':'decay_radius','role':'discrete_pole_radius','values':self.decay_radius.detach(),'unit':'ratio','trainable':self.damping_raw.requires_grad,'lower_bound':None,'upper_bound':None,'group_ids':self.group_ids},'threshold':{'name':'threshold','role':'firing_threshold','values':self.effective_threshold.detach(),'unit':'membrane','trainable':self.threshold_raw is not None,'lower_bound':self._t_lo,'upper_bound':self._t_hi,'group_ids':self.group_ids}}
