"""CUDA-only runtime helpers."""

from __future__ import annotations

import torch

from src.util.precision import configure_tf32


def require_cuda_device(gpu_index: int | None = None) -> torch.device:
    """Return one CUDA device or fail fast.

    The project intentionally assumes CUDA-only execution. CPU fallback paths are
    not part of the official implementation scope.
    """

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by the official project implementation, but no CUDA device is available.')
    if gpu_index is None:
        gpu_index = 0
    gpu_index = int(gpu_index)
    if gpu_index < 0 or gpu_index >= torch.cuda.device_count():
        raise ValueError(f'Invalid CUDA device index {gpu_index}; available count is {torch.cuda.device_count()}.')
    torch.cuda.set_device(gpu_index)
    configure_tf32(enabled=True)
    return torch.device(f'cuda:{gpu_index}')


__all__ = ['require_cuda_device']
