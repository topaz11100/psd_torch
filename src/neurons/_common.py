"""Shared low-level helpers for project neuron layers."""

from __future__ import annotations

import os

import numpy as np
import torch

from src.neurons.spikingjelly_compat import (
    SPIKINGJELLY_AVAILABLE,
    install_spikingjelly_contract,
    reset_spikingjelly_state,
    spikingjelly_surrogate_spike,
    surrogate_backend_name,
)



def _truthy_env(name: str) -> bool:
    return str(os.environ.get(name, '')).strip().lower() in {'1', 'true', 'yes', 'on'}


def use_preallocated_sequence_buffers() -> bool:
    """Sequence outputs are always written into preallocated tensors."""

    return True


def sequence_buffer_mode() -> str:
    """Human-readable fixed sequence-output buffering policy."""

    return 'prealloc'


def sequence_backend_name() -> str:
    """Human-readable sequence backend label for metadata/logging."""

    return 'compiled_sequence_prealloc'

def stack_time_sequence(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Stack a list of timestep tensors into the project-standard time axis."""

    if not tensors:
        raise ValueError('Cannot stack an empty timestep sequence.')
    return torch.stack(tensors, dim=1).contiguous()


def amp_bf16_safe_enabled() -> bool:
    """Return whether the project BF16-safe autocast policy is active."""

    return _truthy_env('PSD_AMP_BF16_SAFE')


def sequence_state_dtype(reference: torch.Tensor) -> torch.dtype:
    """Choose the recurrent state dtype for a sequence tensor."""

    if (
        amp_bf16_safe_enabled()
        and isinstance(reference, torch.Tensor)
        and reference.is_floating_point()
        and reference.device.type == 'cuda'
    ):
        return torch.float32
    return reference.dtype


def to_sequence_state_dtype(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    dtype = sequence_state_dtype(reference)
    return tensor if tensor.dtype == dtype else tensor.to(dtype=dtype)


class FastSigmoidSpike(torch.autograd.Function):
    """Simple surrogate spike used by project-defined layers."""

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, slope: float = 10.0) -> torch.Tensor:
        """Run the forward pass."""
        ctx.save_for_backward(input_tensor)
        ctx.slope = float(slope)
        return (input_tensor > 0).to(dtype=input_tensor.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Run the backward pass."""
        (input_tensor,) = ctx.saved_tensors
        slope = ctx.slope
        grad = grad_output / (1.0 + slope * input_tensor.abs()).pow(2)
        return grad, None


def _torch_compiler_is_compiling() -> bool:
    compiler = getattr(torch, 'compiler', None)
    is_compiling = getattr(compiler, 'is_compiling', None) if compiler is not None else None
    if callable(is_compiling):
        try:
            return bool(is_compiling())
        except Exception:
            return False
    # Do not touch ``torch._dynamo`` from eager timestep code.  On some hosts the
    # lazy import pulls in SymPy and makes even compile=false smoke paths stall.
    # Modern PyTorch exposes the supported compile-state probe via
    # ``torch.compiler.is_compiling``; if it is absent, stay on the eager
    # surrogate instead of importing Dynamo in the hot path.
    return False


def compile_safe_surrogate_spike(input_tensor: torch.Tensor, slope: float = 10.0) -> torch.Tensor:
    """Pure-Torch straight-through spike used inside TorchDynamo graphs."""

    hard = (input_tensor > 0).to(dtype=input_tensor.dtype)
    soft = torch.sigmoid(float(slope) * input_tensor)
    return hard.detach() - soft.detach() + soft


def surrogate_spike(input_tensor: torch.Tensor, slope: float = 10.0) -> torch.Tensor:
    """Return a surrogate spike with a TorchDynamo-safe path under compile."""

    if _torch_compiler_is_compiling():
        return compile_safe_surrogate_spike(input_tensor, slope=float(slope))
    sj_value = spikingjelly_surrogate_spike(input_tensor, slope=float(slope))
    if sj_value is not None:
        return sj_value
    return FastSigmoidSpike.apply(input_tensor, slope)


def trim_open_interval(left: float, right: float, *, epsilon: float = 1.0e-4) -> tuple[float, float]:
    """Trim a possibly closed interval into a numerically stable open interval."""

    left = float(left)
    right = float(right)
    if right < left:
        raise ValueError(f'Invalid interval [{left}, {right}].')
    if np.isclose(left, right):
        return left, right
    width = right - left
    delta = min(float(epsilon), 0.25 * width)
    return left + delta, right - delta


def logit(x: torch.Tensor | float) -> torch.Tensor:
    """Handle ``logit`` for the ``_common`` module."""

    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(float(x), dtype=torch.float32)
    x = torch.clamp(x, min=1.0e-6, max=1.0 - 1.0e-6)
    return torch.log(x) - torch.log1p(-x)


__all__ = [
    'FastSigmoidSpike',
    'amp_bf16_safe_enabled',
    'compile_safe_surrogate_spike',
    'SPIKINGJELLY_AVAILABLE',
    'install_spikingjelly_contract',
    'logit',
    'reset_spikingjelly_state',
    'sequence_backend_name',
    'sequence_buffer_mode',
    'sequence_state_dtype',
    'stack_time_sequence',
    'surrogate_spike',
    'surrogate_backend_name',
    'to_sequence_state_dtype',
    'trim_open_interval',
    'use_preallocated_sequence_buffers',
]
