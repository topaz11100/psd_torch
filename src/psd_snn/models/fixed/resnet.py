from __future__ import annotations
import torch
from torch import nn
from psd_snn.analysis.trace.adapter import LayerTraceRecord


class Block(nn.Module):
    def __init__(self, c):
        super().__init__(); self.c1=nn.Conv2d(c,c,3,padding=1); self.c2=nn.Conv2d(c,c,3,padding=1); self.a=nn.ReLU()
    def forward(self,x):
        y=self.a(self.c1(x)); y=self.c2(y); return self.a(x+y)


class FixedResNetModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__(); self.stem=nn.Conv2d(1,8,3,padding=1); self.b=Block(8); self.pool=nn.AdaptiveAvgPool2d((1,1)); self.head=nn.Linear(8,num_classes); self.topology_kind='resnet'; self.input_layout='B,T,C,H,W'; self.trace_layout='B,T,*'
    def forward(self, x_btchw, capture_trace=False, probe_family='unknown', label='na'):
        b,t,c,h,w=x_btchw.shape
        x=self.stem(x_btchw.reshape(b*t,c,h,w)); z=self.b(x).reshape(b,t,8,h,w)
        flat=self.pool(z.reshape(b*t,8,h,w)).reshape(b,t,8)
        logit_t=self.head(flat); logits=logit_t[:,-1,:]
        if not capture_trace: return logits
        tr=[LayerTraceRecord(0,'resnet_block','hidden','activation','layer',probe_family,label,z), LayerTraceRecord(1,'resnet_logits','output','logits','layer',probe_family,label,logit_t)]
        for t0 in tr: t0.topology_kind='resnet'
        return logits,tr
