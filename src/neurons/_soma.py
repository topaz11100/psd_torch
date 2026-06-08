"""Shared soma-threshold/reset helpers for project-defined neuron layers.

The helpers deliberately affect only soma-level membrane variables.  Dendritic
branch states remain untouched, which keeps variable-branch statistics and branch
filters independent of readout threshold/reset policy.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.neurons._common import positive_threshold_raw_init

_SUPPORTED_RESET_MODES = {"soft_reset", "hard_reset", "no_reset"}


def init_soma_threshold_reset(
    module: nn.Module,
    *,
    output_dim: int,
    v_threshold: float,
    trainable_threshold: bool = False,
    reset_mode: str = "soft_reset",
    emit_spike: bool = True,
    reset_enabled: bool = True,
    eps: float = 1.0e-6,
) -> None:
    """Attach the project-standard soma threshold/reset contract to ``module``.

    ``reset_mode`` is a soma-only policy:
      - ``soft_reset``: subtract the threshold after a soma spike,
      - ``hard_reset``: zero the soma membrane after a soma spike,
      - ``no_reset``: leave the soma membrane unreset.

    Branch states are never modified by this helper.
    """

    mode = str(reset_mode).strip().lower().replace("-", "_")
    mode = {
        "soft": "soft_reset",
        "hard": "hard_reset",
        "none": "no_reset",
        "no": "no_reset",
        "off": "no_reset",
    }.get(mode, mode)
    if mode not in _SUPPORTED_RESET_MODES:
        allowed = ", ".join(sorted(_SUPPORTED_RESET_MODES | {"soft", "hard", "none"}))
        raise ValueError(f"reset_mode must be one of {{{allowed}}}, got {reset_mode!r}.")
    if float(v_threshold) <= 0.0:
        raise ValueError("v_threshold must be positive.")

    module.v_th = float(v_threshold)  # backward-compatible metadata alias.
    module.reset_mode = mode
    module.emit_spike = bool(emit_spike)
    module.reset_enabled = bool(reset_enabled) and mode != "no_reset"
    # Threshold parameters only affect soma spikes/resets.  Membrane-only output
    # readouts pass emit_spike=False for the output layer, so a trainable output
    # threshold would be a DDP-unused parameter.  Keep the user-requested metadata
    # separate from the effective trainability and freeze the threshold whenever
    # the layer is configured not to emit spikes.
    module.requested_trainable_threshold = bool(trainable_threshold)
    module.trainable_threshold = bool(trainable_threshold) and bool(emit_spike)
    module.threshold_eps = float(eps)

    if bool(module.trainable_threshold):
        module.v_threshold_param = nn.Parameter(
            positive_threshold_raw_init(float(v_threshold), int(output_dim), eps=float(eps))
        )
    else:
        module.register_buffer(
            "v_threshold_buffer",
            torch.full((int(output_dim),), float(v_threshold), dtype=torch.float32),
        )
        module.register_parameter("v_threshold_param", None)


def effective_soma_threshold(module: Any) -> torch.Tensor:
    """Return the positive soma threshold vector for a layer using the contract."""

    param = getattr(module, "v_threshold_param", None)
    if param is not None:
        return F.softplus(param) + float(getattr(module, "threshold_eps", 1.0e-6))
    buffer = getattr(module, "v_threshold_buffer", None)
    if buffer is None:
        # Legacy fallback for old checkpoints/classes that only carried v_th.
        output_dim = int(getattr(module, "output_dim"))
        return torch.full((output_dim,), float(getattr(module, "v_th", 1.0)), dtype=torch.float32)
    return buffer


def apply_soma_reset(module: Any, membrane_pre: torch.Tensor, spike: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    """Apply soma-only reset to ``membrane_pre`` using the module's reset contract."""

    if not bool(getattr(module, "reset_enabled", True)):
        return membrane_pre
    mode = str(getattr(module, "reset_mode", "soft_reset"))
    if mode == "hard_reset":
        return membrane_pre * (1.0 - spike)
    if mode == "soft_reset":
        return membrane_pre - threshold * spike
    if mode == "no_reset":
        return membrane_pre
    raise ValueError(f"Unsupported soma reset mode: {mode!r}.")


def soma_contract_stat_vectors(module: Any, *, dtype: torch.dtype, device: torch.device) -> dict[str, torch.Tensor]:
    """Return per-soma tensor flags for filter/stat export."""

    output_dim = int(getattr(module, "output_dim"))
    threshold = effective_soma_threshold(module).to(device=device, dtype=dtype)
    return {
        "threshold": threshold,
        "v_threshold": threshold,
        "soma_reset_enabled": torch.full((output_dim,), 1.0 if bool(getattr(module, "reset_enabled", True)) else 0.0, device=device, dtype=dtype),
        "soma_hard_reset": torch.full((output_dim,), 1.0 if str(getattr(module, "reset_mode", "soft_reset")) == "hard_reset" else 0.0, device=device, dtype=dtype),
        "soma_trainable_threshold": torch.full((output_dim,), 1.0 if bool(getattr(module, "trainable_threshold", False)) else 0.0, device=device, dtype=dtype),
        "soma_requested_trainable_threshold": torch.full((output_dim,), 1.0 if bool(getattr(module, "requested_trainable_threshold", getattr(module, "trainable_threshold", False))) else 0.0, device=device, dtype=dtype),
    }


__all__ = [
    "apply_soma_reset",
    "effective_soma_threshold",
    "init_soma_threshold_reset",
    "soma_contract_stat_vectors",
]
