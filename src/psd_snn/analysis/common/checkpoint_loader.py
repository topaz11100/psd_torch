from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class CheckpointRef:
    path: str
    checkpoint_epoch: int | None = None
    checkpoint_id: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class CheckpointBundle:
    checkpoint_ref: CheckpointRef
    payload: dict[str, Any]
    model_metadata: dict[str, Any]
    state_dict: dict[str, Any]
    checkpoint_epoch: int | None
    config_hash: str | None = None
    constraint_hash: str | None = None


def load_checkpoint_bundle(ref: CheckpointRef) -> CheckpointBundle:
    import torch

    payload = torch.load(ref.path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dict")
    state_dict = payload.get("state_dict")
    metadata = payload.get("metadata", {})
    if state_dict is None:
        raise ValueError("checkpoint payload missing state_dict")
    model_md = metadata.get("model", metadata)
    ckpt_epoch = payload.get("checkpoint_epoch", ref.checkpoint_epoch)
    return CheckpointBundle(
        checkpoint_ref=ref,
        payload=payload,
        model_metadata=model_md,
        state_dict=state_dict,
        checkpoint_epoch=ckpt_epoch,
        config_hash=metadata.get("config_hash"),
        constraint_hash=metadata.get("constraint_hash"),
    )
