from __future__ import annotations
import torch
from torch import nn
from psd_snn.analysis.trace.adapter import LayerTraceRecord


class FixedSpikeTransformerModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 32, n_heads: int = 4, n_layers: int = 1):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        enc = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Linear(hidden_dim, num_classes)
        self.topology_kind = 'spike_transformer'
        self.implementation_detail = 'thresholded_transformer'
        self.input_layout = 'B,T,F'
        self.trace_layout = 'B,T,*'
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers

    def forward(self, x_btf, capture_trace=False, probe_family='unknown', label='na'):
        h = self.in_proj(x_btf)
        token_state = self.encoder(h)
        spike_activation = (token_state > 0).to(token_state.dtype)
        logit_t = self.head(token_state)
        logits = logit_t[:, -1, :]
        if not capture_trace:
            return logits
        traces = [
            LayerTraceRecord(0, 'spike_transformer_hidden', 'hidden', 'hidden_state', 'layer', probe_family, label, token_state),
            LayerTraceRecord(1, 'spike_transformer_activation', 'hidden', 'spike_activation', 'layer', probe_family, label, spike_activation),
            LayerTraceRecord(2, 'spike_transformer_logits', 'output', 'logits', 'layer', probe_family, label, logit_t),
        ]
        for tr in traces:
            tr.topology_kind = self.topology_kind
            tr.implementation_detail = self.implementation_detail
            tr.input_layout = self.input_layout
            tr.trace_layout = self.trace_layout
            tr.num_classes = self.num_classes
            tr.hidden_dim = self.hidden_dim
            tr.n_heads = self.n_heads
            tr.n_layers = self.n_layers
        return logits, traces
