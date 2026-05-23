from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from psd_snn.analysis.probe.builder import build_probe_indices, ProbeManifest, CANONICAL_PROBE_FAMILIES


@dataclass
class ProbeRequest:
    split: str
    probe_family: str
    label: int | None = None
    labels: list[int] | None = None
    sample_count: int | None = None
    seed: int = 0
    batch_size: int | None = None
    exclusion_family: str | None = None
    exclusion_request: 'ProbeRequest' | None = None
    exclusion_sample_count: int | None = None
    exclusion_seed: int | None = None


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
    probe_manifest_id: str | None = None
    probe_metadata: dict[str, Any] | None = None


class ProbeOrchestrator:
    def __init__(self, dataset: dict[str, Any]):
        self.dataset = dataset

    def _resolve_exclusion_indices(self, request: ProbeRequest, labels: list[int]) -> tuple[set[int], str | None, str | None]:
        if request.exclusion_request is not None:
            ex = request.exclusion_request
            if request.exclusion_family and request.exclusion_family != ex.probe_family:
                raise ValueError('exclusion_family mismatch with exclusion_request.probe_family')
            ex_manifest = self.build_manifest(ex)
            return set(ex_manifest.selected_indices), ex_manifest.probe_family, ex_manifest.scope
        if not request.exclusion_family:
            return set(), None, None
        if request.exclusion_family not in {'balanced_global', 'distributed_set', 'label_set'}:
            raise ValueError('unsupported exclusion_family')
        ex_req = ProbeRequest(split=request.split, probe_family=request.exclusion_family, seed=request.exclusion_seed if request.exclusion_seed is not None else request.seed, sample_count=request.exclusion_sample_count if request.exclusion_sample_count is not None else request.sample_count, label=request.label, labels=request.labels)
        if request.exclusion_family == 'label_set' and not (request.label is not None or request.labels):
            raise ValueError('label_set exclusion_family requires label or labels')
        ex_manifest = self.build_manifest(ex_req)
        return set(ex_manifest.selected_indices), ex_manifest.probe_family, ex_manifest.scope

    def build_manifest(self, request: ProbeRequest) -> ProbeManifest:
        if request.probe_family not in CANONICAL_PROBE_FAMILIES:
            raise ValueError('unsupported probe family')
        labels = self.dataset[f"{request.split}_labels"]
        target_labels = request.labels or ([request.label] if request.label is not None else None)
        exclusion_indices, ex_family, ex_scope = self._resolve_exclusion_indices(request, labels)
        manifest = build_probe_indices(labels=labels, family=request.probe_family, sample_count=request.sample_count or len(labels), seed=request.seed, target_labels=target_labels, exclusion_indices=exclusion_indices, split=request.split, exclusion_family=ex_family)
        manifest.exclusion_scope = ex_scope
        return manifest

    def iter_batches(self, request: ProbeRequest, batch_size: int = 32) -> list[ProbeBatch]:
        try:
            import torch
        except Exception:
            torch = None
        manifest = self.build_manifest(request)
        labels = self.dataset[f"{request.split}_labels"]
        inputs = self.dataset[f"{request.split}_inputs"]
        idx = manifest.selected_indices
        batches = []
        bs = request.batch_size or batch_size
        for bi, start in enumerate(range(0, len(idx), bs)):
            chunk = idx[start:start + bs]
            if torch is None:
                x = [inputs[i] for i in chunk]
                y_vals = [int(labels[i]) for i in chunk]
                y = y_vals
                lbls = y_vals
            else:
                x = torch.as_tensor([inputs[i] for i in chunk])
                y = torch.as_tensor([labels[i] for i in chunk])
                lbls = [int(v) for v in y.tolist()]
            batches.append(ProbeBatch(x, y, chunk, lbls, request.split, manifest.scope, request.probe_family, bi, probe_manifest_id=manifest.probe_manifest_id, probe_metadata={'exclusion_family': manifest.exclusion_family, 'exclusion_scope': manifest.exclusion_scope, 'class_counts': manifest.class_counts, 'quotas': manifest.quotas, 'seed': manifest.seed}))
        return batches


def build_probe_batches(dataset: dict[str, Any], request: ProbeRequest, *, batch_size: int = 32) -> list[ProbeBatch]:
    return ProbeOrchestrator(dataset).iter_batches(request, batch_size=batch_size)
