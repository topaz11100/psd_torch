from __future__ import annotations
import torch
from torch import nn

class DenseTemporalBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, cell: nn.Module, recurrent: bool = False, feedforward_mask=None, recurrent_mask=None, layer_group_ids=None):
        super().__init__()
        self.ff = nn.Linear(in_dim, out_dim)
        self.cell = cell
        self.recurrent = recurrent
        self.layer_group_ids = layer_group_ids
        self.register_buffer('feedforward_mask', feedforward_mask if feedforward_mask is not None else None)
        if recurrent:
            self.recurrent_weight = nn.Parameter(torch.zeros(out_dim, out_dim))
            self.register_buffer('recurrent_mask', recurrent_mask if recurrent_mask is not None else None)
        else:
            self.register_parameter('recurrent_weight', None)
            self.register_buffer('recurrent_mask', None)

    @property
    def has_feedforward_mask(self): return self.feedforward_mask is not None
    @property
    def has_recurrent_mask(self): return self.recurrent_mask is not None

    def forward(self, x_btf, capture_trace=False):
        b, t, _ = x_btf.shape
        w = self.ff.weight if self.feedforward_mask is None else (self.ff.weight * self.feedforward_mask.to(self.ff.weight.device))
        ff_seq = torch.einsum('btf,of->bto', x_btf, w) + self.ff.bias
        if self.recurrent:
            seq = []
            s_prev = torch.zeros(b, self.ff.out_features, device=x_btf.device)
            rw = self.recurrent_weight if self.recurrent_mask is None else (self.recurrent_weight * self.recurrent_mask.to(self.recurrent_weight.device))
            for ti in range(t):
                cur = ff_seq[:, ti, :] + s_prev @ rw.T
                st = self.cell.single_step_forward(cur)
                s_prev = st.spike
                seq.append(cur)
            ff_seq = torch.stack(seq, dim=1)
            self.cell.reset()
        y, tr = self.cell.forward_sequence(ff_seq, capture_trace=capture_trace)
        return y, tr
