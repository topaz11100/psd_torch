from __future__ import annotations
import torch


def validate_bounds(lower, upper, *, name: str, strict: bool = True):
    l = torch.as_tensor(lower, dtype=torch.float32)
    u = torch.as_tensor(upper, dtype=torch.float32)
    ok = torch.all(l < u) if strict else torch.all(l <= u)
    if not bool(ok):
        raise ValueError(f'invalid bounds for {name}: lower must be < upper')
    return l, u


def bounded_sigmoid(raw, lower, upper, eps: float = 1e-6):
    l, u = validate_bounds(lower, upper, name='bounded_sigmoid')
    return l.to(raw.device, raw.dtype) + (u - l).to(raw.device, raw.dtype) * torch.sigmoid(raw).clamp(eps, 1 - eps)


def inverse_bounded_sigmoid(value, lower, upper, eps: float = 1e-6):
    l, u = validate_bounds(lower, upper, name='inverse_bounded_sigmoid')
    x = (value - l.to(value.device, value.dtype)) / (u - l).to(value.device, value.dtype)
    if torch.any((x <= 0) | (x >= 1)):
        raise ValueError('init value outside bounds')
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


def positive_parameter(raw, eps: float = 1e-8):
    return torch.exp(raw) + eps


def inverse_positive_parameter(value, eps: float = 1e-8):
    if torch.any(torch.as_tensor(value) <= 0):
        raise ValueError('value must be positive')
    return torch.log(torch.as_tensor(value) - eps)


def materialize_feature_bounds(bounds, feature_dim: int, device=None, dtype=None):
    if bounds is None:
        return None, None
    if isinstance(bounds, (tuple, list)) and len(bounds) == 2 and all(isinstance(v, (int, float)) for v in bounds):
        lo = torch.full((feature_dim,), float(bounds[0]), device=device, dtype=dtype or torch.float32)
        hi = torch.full((feature_dim,), float(bounds[1]), device=device, dtype=dtype or torch.float32)
        return lo, hi
    lo, hi = zip(*bounds)
    lo = torch.as_tensor(lo, device=device, dtype=dtype or torch.float32)
    hi = torch.as_tensor(hi, device=device, dtype=dtype or torch.float32)
    if lo.numel() == 1:
        lo = lo.repeat(feature_dim); hi = hi.repeat(feature_dim)
    if lo.numel() != feature_dim:
        raise ValueError('bounds shape mismatch with feature_dim')
    validate_bounds(lo, hi, name='feature_bounds')
    return lo, hi
