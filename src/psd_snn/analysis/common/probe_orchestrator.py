from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from psd_snn.analysis.probe.builder import build_probe_indices


@dataclass
class ProbeRequest:
    split: str
    probe_family: str
    label: int | None = None
    sample_count: int | None = None
    seed: int = 0
    exclusion_family: str | None = None


@dataclass
class ProbeBatch:
    inputs: Any
    targets: Any | None
    sample_indices: list[int]
    labels: list[int] | None
    split: str
    scope: str
    probe_family: str
    batch_index: int


def build_probe_batches(dataset: dict[str, Any], request: ProbeRequest, *, batch_size: int = 32) -> list[ProbeBatch]:
    import torch
    labels = dataset[f"{request.split}_labels"]
    inputs = dataset[f"{request.split}_inputs"]
    target_labels = [request.label] if request.label is not None else None
    manifest = build_probe_indices(labels=labels, family=request.probe_family, sample_count=request.sample_count or len(labels), seed=request.seed, target_labels=target_labels)
    scope = f"{request.split}_{request.probe_family}" + (f"_label={request.label}" if request.label is not None else "")
    idx = manifest.selected_indices
    batches = []
    for bi, start in enumerate(range(0, len(idx), batch_size)):
        chunk = idx[start:start + batch_size]
        x = torch.as_tensor([inputs[i] for i in chunk])
        y = torch.as_tensor([labels[i] for i in chunk])
        batches.append(ProbeBatch(x, y, chunk, [int(v) for v in y.tolist()], request.split, scope, request.probe_family, bi))
    return batches
