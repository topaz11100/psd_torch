"""Precision and matmul-policy helpers for training/inference entrypoints."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Iterator
import os

import torch


_AMP_OFF_TOKENS = {'', '0', 'false', 'no', 'none', 'null', 'off'}
_AMP_BF16_SAFE_TOKENS = {'on', 'bf16_safe'}


def normalize_amp_mode(value: Any) -> str:
    """Return internal AMP mode for the public ``off``/``on`` switch."""

    token = str('off' if value is None else value).strip().lower()
    if token in _AMP_OFF_TOKENS:
        return 'off'
    if token in _AMP_BF16_SAFE_TOKENS:
        return 'bf16_safe'
    raise ValueError('amp must be either off or on. amp=on maps to bf16_safe.')


def configure_tf32(*, enabled: bool = True) -> dict[str, Any]:
    """Configure FP32 matmul/conv internals to use TF32 where CUDA supports it."""

    enabled = bool(enabled)
    status: dict[str, Any] = {'requested': enabled}
    try:
        torch.set_float32_matmul_precision('high' if enabled else 'highest')
        status['float32_matmul_precision'] = 'high' if enabled else 'highest'
    except Exception as exc:  # pragma: no cover - version dependent
        status['float32_matmul_precision_error'] = f'{type(exc).__name__}: {exc}'
    try:
        torch.backends.cuda.matmul.allow_tf32 = enabled
        status['cuda_matmul_allow_tf32'] = enabled
    except Exception as exc:  # pragma: no cover - cpu/version dependent
        status['cuda_matmul_allow_tf32_error'] = f'{type(exc).__name__}: {exc}'
    try:
        torch.backends.cudnn.allow_tf32 = enabled
        status['cudnn_allow_tf32'] = enabled
    except Exception as exc:  # pragma: no cover - cpu/version dependent
        status['cudnn_allow_tf32_error'] = f'{type(exc).__name__}: {exc}'
    # PyTorch 2.9+ exposes string precision controls.  Set them when present but
    # keep the older boolean flags above for compatibility with earlier releases.
    try:
        torch.backends.cuda.matmul.fp32_precision = 'tf32' if enabled else 'ieee'
        status['cuda_matmul_fp32_precision'] = 'tf32' if enabled else 'ieee'
    except Exception:
        pass
    try:
        torch.backends.cudnn.conv.fp32_precision = 'tf32' if enabled else 'ieee'
        status['cudnn_conv_fp32_precision'] = 'tf32' if enabled else 'ieee'
    except Exception:
        pass
    return status


@contextmanager
def amp_autocast_context(*, amp_mode: Any, device: torch.device | str | None) -> Iterator[None]:
    """Forward-only BF16-safe autocast context.

    Only ``bf16_safe`` is supported.  The context sets a private environment flag
    so project neuron layers can keep recurrent membrane/state tensors in FP32
    while allowing large GEMM/Conv kernels to run under CUDA BF16 autocast.
    """

    mode = normalize_amp_mode(amp_mode)
    device_type = str(getattr(device, 'type', device) or 'cpu')
    active = bool(mode == 'bf16_safe' and device_type == 'cuda')
    previous = os.environ.get('PSD_AMP_BF16_SAFE')
    if active:
        os.environ['PSD_AMP_BF16_SAFE'] = '1'
    try:
        ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if active else nullcontext()
        with ctx:
            yield
    finally:
        if previous is None:
            os.environ.pop('PSD_AMP_BF16_SAFE', None)
        else:
            os.environ['PSD_AMP_BF16_SAFE'] = previous


__all__ = ['amp_autocast_context', 'configure_tf32', 'normalize_amp_mode']
