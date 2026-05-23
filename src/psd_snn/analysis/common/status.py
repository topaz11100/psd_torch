from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Literal

StatusLiteral = Literal[
    'ok','checkpoint_load_failed','model_restore_failed','unsupported_topology','state_dict_missing_keys',
    'state_dict_unexpected_keys','probe_build_failed','no_trace_records','unavailable_series','signal_map_failed',
    'pca_basis_missing','pca_basis_incompatible','distance_incompatible','writer_failed'
]

@dataclass
class AnalysisStatus:
    status: StatusLiteral = 'ok'
    reason: str | None = None

@dataclass
class AnalysisFailure:
    status: StatusLiteral
    reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    run_id: str | None = None
    checkpoint_epoch: int | None = None
    checkpoint_id: str | None = None
    split: str | None = None
    scope: str | None = None
    probe_family: str | None = None
    probe_manifest_id: str | None = None
    layer_index: int | None = None
    layer_name: str | None = None
    signal_kind: str | None = None
    series: str | None = None
    artifact_type: str | None = 'analysis_manifest'
    artifact_path: str | None = None

    def to_manifest_row(self) -> dict:
        return asdict(self)
