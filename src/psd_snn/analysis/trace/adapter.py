from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass
class LayerTraceRecord:
    layer_index: int
    layer_name: str
    signal_kind: str
    series: str
    scope: str
    probe_family: str
    label: str
    tensor: Any
    scenario: str = 'none'
    constraint_hash: str | None = None
    layer_group_ids: list[int] | None = None
    feedforward_mask_applied: bool = False
    recurrent_mask_applied: bool = False

class TraceAdapter:
    def __init__(self, model): self.model = model
    def run_with_trace(self, x_btf, probe_family='unknown', label='na'):
        from spikingjelly.activation_based import functional
        functional.reset_net(self.model)
        logits, traces = self.model(x_btf, capture_trace=True, probe_family=probe_family, label=label)
        functional.reset_net(self.model)
        return logits, traces
