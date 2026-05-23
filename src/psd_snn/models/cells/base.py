from __future__ import annotations
from dataclasses import dataclass
from typing import Any


def require_torch():
    try:
        import torch  # noqa
        return torch
    except Exception as e:
        raise ImportError('torch is required for psd_snn spiking cells') from e


def require_spikingjelly(cell_name: str = 'unknown'):
    try:
        from spikingjelly.activation_based import base  # noqa
        return base
    except Exception as e:
        raise ImportError(f'spikingjelly is required to create official cell `{cell_name}`. Please install spikingjelly.') from e


@dataclass
class CellStepTrace:
    input_current: Any
    membrane_pre: Any
    decision: Any
    spike: Any
    membrane_post: Any
    rf_real_pre: Any = None
    rf_imag_pre: Any = None
    rf_real_post: Any = None
    rf_imag_post: Any = None


class TraceableMemoryNode:
    emits_spike: bool = True

    def state_trace_names(self):
        return ['input_current', 'membrane_pre', 'decision', 'spike', 'membrane_post']
