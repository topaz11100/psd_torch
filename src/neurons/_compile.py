"""Small helpers for regional ``torch.compile`` inside neuron modules."""
from __future__ import annotations
from typing import Any, Callable
import warnings
import torch
from torch import nn

def compile_kwargs_label(compile_kwargs: dict[str, Any] | None) -> str:
    kwargs = dict(compile_kwargs or {})
    return ','.join(f'{key}={value}' for key, value in sorted(kwargs.items())) if kwargs else 'default'

def module_device_type(module: nn.Module) -> str | None:
    for tensor_iter_name in ('parameters', 'buffers'):
        tensor_iter = getattr(module, tensor_iter_name, None)
        if callable(tensor_iter):
            try:
                for tensor in tensor_iter():
                    return str(tensor.device.type)
            except Exception:
                return None
    return 'cpu'

def compile_callable(fn: Callable[..., Any], *, compile_kwargs: dict[str, Any] | None = None, label: str = 'region') -> tuple[Callable[..., Any] | None, bool, str]:
    compile_fn = getattr(torch, 'compile', None)
    if compile_fn is None:
        return None, False, 'torch.compile_unavailable'
    kwargs = dict(compile_kwargs or {})
    try:
        try:
            compiled = compile_fn(fn, **kwargs)
        except TypeError:
            if kwargs:
                raise
            compiled = compile_fn(fn)
    except Exception as exc:  # pragma: no cover
        return None, False, f'{label}_compile_construction_failed:{type(exc).__name__}: {exc}'
    return compiled, True, f'torch.compile_{label}({compile_kwargs_label(kwargs)})'

def disable_compiled_runtime(module: nn.Module, *, label: str, exc: BaseException) -> None:
    error = f'{type(exc).__name__}: {exc}'
    setattr(module, f'_{label}_compiled_runtime_disabled', True)
    setattr(module, f'_{label}_compiled_runtime_error', error)
    warnings.warn(
        f'[torch.compile] regional {label} runtime fallback activated for '
        f'{module.__class__.__name__}: {error}',
        RuntimeWarning,
        stacklevel=2,
    )
