"""Compatibility helpers for SpikingJelly-based SNN plumbing.

Project-specific neurons keep their exact IF/LIF/RF/DH/D-RF equations and trace
contract, but the shared spike/reset plumbing is anchored to SpikingJelly when
the package is available. The fallback path preserves importability in minimal
test environments where SpikingJelly is not installed.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

try:  # pragma: no cover - optional dependency in lightweight CI
    from spikingjelly.activation_based import functional as sj_functional
    from spikingjelly.activation_based import surrogate as sj_surrogate
except Exception:  # pragma: no cover
    sj_functional = None
    sj_surrogate = None


SPIKINGJELLY_AVAILABLE = sj_surrogate is not None

_SIGMOID_SURROGATE = None
if sj_surrogate is not None:  # pragma: no cover - depends on runtime package
    try:
        _SIGMOID_SURROGATE = sj_surrogate.Sigmoid(alpha=10.0, spiking=True)
    except TypeError:
        _SIGMOID_SURROGATE = sj_surrogate.Sigmoid(alpha=10.0)


def spikingjelly_surrogate_spike(input_tensor: torch.Tensor, *, slope: float = 10.0) -> torch.Tensor | None:
    """Return SpikingJelly's sigmoid surrogate output when available."""

    if _SIGMOID_SURROGATE is None or float(slope) != 10.0:
        return None
    return _SIGMOID_SURROGATE(input_tensor)


def surrogate_backend_name() -> str:
    """Return the active surrogate backend name for tests and audit metadata."""

    if _SIGMOID_SURROGATE is not None:
        return 'spikingjelly.activation_based.surrogate.Sigmoid'
    return 'fallback.FastSigmoidSpike'


def reset_spikingjelly_state(module: nn.Module) -> bool:
    """Reset a module recursively through SpikingJelly when available.

    Returns True only when SpikingJelly's reset function was present and ran
    without raising. Callers may fall back to project-local reset hooks when this
    returns False.
    """

    if sj_functional is None:
        return False
    try:  # pragma: no cover - depends on optional package internals
        sj_functional.reset_net(module)
        return True
    except Exception:
        return False


def _default_reset_state(self: nn.Module) -> None:
    """Project-local reset hook compatible with SpikingJelly reset calls."""

    if hasattr(self, '_last_layer_input'):
        self._last_layer_input = None


def install_spikingjelly_contract(cls: type[nn.Module]) -> type[nn.Module]:
    """Install a small SpikingJelly-style contract on a project neuron class.

    The helper avoids invasive base-class rewrites, so it is safe for thin
    wrappers around released code. It provides the public attributes expected by
    SpikingJelly multi-step modules and ensures each instance has ``step_mode``.
    """

    if getattr(cls, '_psd_spikingjelly_contract_installed', False):
        return cls

    original_init = cls.__init__

    def wrapped_init(self: nn.Module, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if not hasattr(self, 'step_mode'):
            self.step_mode = 'm'
        if not hasattr(self, 'supported_step_mode'):
            self.supported_step_mode = ('m',)
        self.spikingjelly_backend = 'spikingjelly' if SPIKINGJELLY_AVAILABLE else 'torch_fallback'

    cls.__init__ = wrapped_init  # type: ignore[method-assign]
    if not hasattr(cls, 'reset_state'):
        cls.reset_state = _default_reset_state  # type: ignore[attr-defined]
    cls._psd_spikingjelly_contract_installed = True  # type: ignore[attr-defined]
    return cls


__all__ = [
    'SPIKINGJELLY_AVAILABLE',
    'install_spikingjelly_contract',
    'reset_spikingjelly_state',
    'spikingjelly_surrogate_spike',
    'surrogate_backend_name',
]
