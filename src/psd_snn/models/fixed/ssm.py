from __future__ import annotations
import torch
from torch import nn
from psd_snn.analysis.trace.adapter import LayerTraceRecord


class FixedSSMModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 16):
        super().__init__()
        self.inp = nn.Linear(input_dim, hidden_dim)
        self.decay = nn.Parameter(torch.zeros(hidden_dim))
        self.mix = nn.Parameter(torch.zeros(hidden_dim))
        self.head = nn.Linear(hidden_dim, num_classes)
        self.topology_kind = 'ssm'; self.input_layout='B,T,F'; self.trace_layout='B,T,*'
        self.implementation_detail = 'diagonal_ssm'

    def forward(self, x_btf, capture_trace=False, probe_family='unknown', label='na'):
        b,t,_ = x_btf.shape
        state = torch.zeros(b, self.inp.out_features, device=x_btf.device, dtype=x_btf.dtype)
        xs = self.inp(x_btf)
        states=[]
        a = torch.sigmoid(self.decay).view(1,-1)
        m = torch.tanh(self.mix).view(1,-1)
        for i in range(t):
            state = a*state + (1-a)*(xs[:,i,:] + m*state)
            states.append(state)
        h = torch.stack(states, dim=1)
        logit_t = self.head(h)
        logits = logit_t[:, -1, :]
        if not capture_trace:
            return logits
        tr=[LayerTraceRecord(0,'ssm_state','hidden','state','layer',probe_family,label,h), LayerTraceRecord(1,'ssm_logits','output','logits','layer',probe_family,label,logit_t)]
        for t0 in tr:
            t0.topology_kind='ssm'; t0.implementation_detail='diagonal_ssm'
        return logits, tr
