from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from psd_snn.config.specs import _from_dict
from psd_snn.models.factory import build_model
from psd_snn.models.fixed.factory import UnsupportedTopologyError


@dataclass
class ModelRestoreResult:
    model: Any | None
    model_spec: Any | None
    topology_spec: Any | None
    cell_spec: Any | None
    readout_spec: Any | None
    constraint_hash: str | None
    checkpoint_epoch: int | None
    restore_status: str
    reason: str | None = None
    missing_keys: list[str] | None = None
    unexpected_keys: list[str] | None = None


def restore_model_from_bundle(bundle, *, device: str = "cpu", strict_load: bool = True) -> ModelRestoreResult:
    try:
        cfg_dict = bundle.payload.get("config")
        if cfg_dict is None:
            raise ValueError("checkpoint missing config")
        cfg = _from_dict(cfg_dict)
        try:
            model = build_model(cfg.model)
        except UnsupportedTopologyError as ue:
            return ModelRestoreResult(None, cfg.model, cfg.model.topology, cfg.model.cell, cfg.model.readout, bundle.constraint_hash, bundle.checkpoint_epoch, 'unsupported_topology', reason=str(ue))
        load_res = model.load_state_dict(bundle.state_dict, strict=strict_load)
        model.eval()
        model.to(device)
        return ModelRestoreResult(
            model=model,
            model_spec=cfg.model,
            topology_spec=cfg.model.topology,
            cell_spec=cfg.model.cell,
            readout_spec=cfg.model.readout,
            constraint_hash=bundle.constraint_hash,
            checkpoint_epoch=bundle.checkpoint_epoch,
            restore_status='ok',
            missing_keys=list(load_res.missing_keys),
            unexpected_keys=list(load_res.unexpected_keys),
        )
    except Exception as exc:
        return ModelRestoreResult(None, None, None, None, None, bundle.constraint_hash, bundle.checkpoint_epoch, 'model_restore_failed', reason=str(exc))
