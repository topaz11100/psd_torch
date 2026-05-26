from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from src.neurons._common import surrogate_spike
from src.neurons.LIF_neuron import _positive_threshold_init

class IFLayer(nn.Module):
    def __init__(self,input_size:int,output_size:int,*,recurrent:bool=False,v_threshold:float=1.0,trainable_threshold:bool=False,reset_mode:str='soft_reset',input_mask:torch.Tensor|None=None,recurrent_mask:torch.Tensor|None=None,emit_spike:bool=True,reset_enabled:bool=True)->None:
        super().__init__()
        if reset_mode not in {'soft_reset','hard_reset'}:
            raise ValueError("reset_mode must be 'soft_reset' or 'hard_reset'.")
        self.input_size=int(input_size); self.output_size=int(output_size); self.recurrent=bool(recurrent)
        self.reset_mode=reset_mode; self.emit_spike=bool(emit_spike); self.reset_enabled=bool(reset_enabled)
        self.trainable_threshold=bool(trainable_threshold); self.threshold_eps=1.0e-6
        self.input_weight=nn.Parameter(torch.empty(self.output_size,self.input_size))
        if self.recurrent: self.recurrent_weight=nn.Parameter(torch.empty(self.output_size,self.output_size))
        else: self.register_parameter('recurrent_weight',None)
        if input_mask is None: input_mask=torch.ones(self.output_size,self.input_size,dtype=torch.float32)
        if self.recurrent and recurrent_mask is None: recurrent_mask=torch.ones(self.output_size,self.output_size,dtype=torch.float32)
        self.register_buffer('input_mask',input_mask.to(dtype=torch.float32))
        self.register_buffer('recurrent_mask',None if recurrent_mask is None else recurrent_mask.to(dtype=torch.float32))
        threshold_init=torch.full((self.output_size,),float(v_threshold),dtype=torch.float32)
        if self.trainable_threshold: self.v_threshold_param=nn.Parameter(_positive_threshold_init(v_threshold,self.output_size,eps=self.threshold_eps))
        else:
            self.register_buffer('v_threshold_buffer',threshold_init); self.register_parameter('v_threshold_param',None)
        self.reset_parameters()

    def reset_parameters(self)->None:
        nn.init.kaiming_uniform_(self.input_weight, a=math.sqrt(5.0))
        if self.recurrent_weight is not None: nn.init.orthogonal_(self.recurrent_weight)
    def effective_threshold(self)->torch.Tensor:
        if self.v_threshold_param is not None: return F.softplus(self.v_threshold_param)+float(self.threshold_eps)
        return self.v_threshold_buffer
    def effective_input_weight(self)->torch.Tensor: return self.input_weight*self.input_mask
    def effective_recurrent_weight(self)->torch.Tensor|None:
        if self.recurrent_weight is None: return None
        return self.recurrent_weight if self.recurrent_mask is None else self.recurrent_weight*self.recurrent_mask
    def _reset_state(self,batch_size:int,device:torch.device,dtype:torch.dtype):
        m=torch.zeros(batch_size,self.output_size,device=device,dtype=dtype); return m,torch.zeros_like(m)
    def forward(self,input_sequence:torch.Tensor,*,return_traces:bool=False):
        self._last_layer_input=None
        b,t,_=input_sequence.shape
        weight=self.effective_input_weight(); rec=self.effective_recurrent_weight(); thr=self.effective_threshold().to(device=input_sequence.device,dtype=input_sequence.dtype)
        membrane,prev=self._reset_state(b,input_sequence.device,input_sequence.dtype)
        membrane_steps=[] if return_traces else None; layer_input_steps=[] if return_traces else None; spikes=[]
        inp=torch.matmul(input_sequence,weight.t())
        for i in range(t):
            current=inp[:,i,:]
            if rec is not None: current=current+prev@rec.t()
            membrane_pre=membrane+current
            signal=membrane_pre-thr.unsqueeze(0)
            spike=surrogate_spike(signal) if self.emit_spike else torch.zeros_like(signal)
            if self.reset_enabled:
                membrane=membrane_pre-thr.unsqueeze(0)*spike if self.reset_mode=='soft_reset' else membrane_pre*(1.0-spike)
            else: membrane=membrane_pre
            if return_traces:
                layer_input_steps.append(current); membrane_steps.append(signal)
            spikes.append(spike); prev=spike
        self._last_layer_input=torch.stack(layer_input_steps,dim=1) if return_traces else None
        return (torch.stack(membrane_steps,dim=1) if return_traces else None), torch.stack(spikes,dim=1)
    def filter_stats_vectors(self)->dict[str,torch.Tensor]: return {}

try:
    from src.neurons.spikingjelly_compat import install_spikingjelly_contract as _install_spikingjelly_contract
    _install_spikingjelly_contract(IFLayer)
except Exception:  # pragma: no cover - defensive import fallback
    pass

__all__=['IFLayer']
