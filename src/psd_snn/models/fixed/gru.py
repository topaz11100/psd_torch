from __future__ import annotations
import torch
from torch import nn
from psd_snn.analysis.trace.adapter import LayerTraceRecord


class FixedGRUModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 16):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_classes)
        self.topology_kind = 'gru'; self.input_layout='B,T,F'; self.trace_layout='B,T,*'

    def forward(self, x_btf, capture_trace=False, probe_family='unknown', label='na'):
        h, _ = self.gru(x_btf)
        logit_t = self.head(h)
        logits = logit_t[:, -1, :]
        if not capture_trace:
            return logits
        tr = [
            LayerTraceRecord(0,'gru_hidden','hidden','hidden_state','layer',probe_family,label,h),
            LayerTraceRecord(1,'gru_logits','output','logits','layer',probe_family,label,logit_t),
        ]
        for t in tr:
            t.topology_kind = 'gru'
        return logits, tr
