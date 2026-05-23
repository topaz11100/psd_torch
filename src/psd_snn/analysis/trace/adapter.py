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
    def run_with_trace(self, x_btf, probe_family='unknown', label='na', context: TraceContext | None = None):
        from spikingjelly.activation_based import functional
        functional.reset_net(self.model)
        logits, traces = self.model(x_btf, capture_trace=True, probe_family=probe_family, label=label)
        if context is not None:
            traces = [_inject_context(t, context) for t in traces]
        functional.reset_net(self.model)
        return logits, traces


@dataclass
class TraceContext:
    run_id: str
    checkpoint_epoch: int | None
    split: str
    scope: str
    probe_family: str
    checkpoint_id: str | None = None
    label: str | None = None
    sample_indices: list[int] | None = None
    labels: list[int] | None = None
    probe_manifest_id: str | None = None
    exclusion_family: str | None = None
    exclusion_scope: str | None = None
    scenario: str | None = None
    constraint_hash: str | None = None

class TraceCollectionRequest:...


def _inject_context(record, ctx: TraceContext):
    for k in ['run_id','checkpoint_epoch','checkpoint_id','split','scope','probe_family','label','sample_indices','labels','probe_manifest_id','exclusion_family','exclusion_scope','scenario','constraint_hash']:
        setattr(record, k, getattr(ctx, k, None))
    return record

