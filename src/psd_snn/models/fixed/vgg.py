from __future__ import annotations
import torch
from torch import nn
from psd_snn.analysis.trace.adapter import LayerTraceRecord


class FixedVGGModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1,8,3,padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(8,16,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)))
        self.head = nn.Linear(16, num_classes)
        self.topology_kind='vgg'; self.input_layout='B,T,C,H,W'; self.trace_layout='B,T,*'

    def forward(self, x_btchw, capture_trace=False, probe_family='unknown', label='na'):
        b,t,c,h,w = x_btchw.shape
        z = self.features(x_btchw.reshape(b*t,c,h,w)).reshape(b,t,16,1,1)
        flat = z.reshape(b,t,16)
        logit_t = self.head(flat)
        logits = logit_t[:,-1,:]
        if not capture_trace: return logits
        tr=[LayerTraceRecord(0,'vgg_activation','hidden','activation','layer',probe_family,label,z), LayerTraceRecord(1,'vgg_logits','output','logits','layer',probe_family,label,logit_t)]
        for t0 in tr: t0.topology_kind='vgg'
        return logits,tr
