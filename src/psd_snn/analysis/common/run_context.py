from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class RunContext:
    run_id: str
    output_dir: str
    device: str = "cpu"
    seed: int = 0
    dataset_token: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    config_hash: str | None = None
    model_hash: str | None = None
    constraint_hash: str | None = None


@dataclass
class AnalysisRunManifest:
    run_id: str
    checkpoint_epoch: int | None
    split: str
    scope: str
    probe_family: str
    analysis_method: str
    representative: str | None
    spectral_axis: str
    status: str
    reason: str | None = None
    artifact_paths: list[str] = field(default_factory=list)
