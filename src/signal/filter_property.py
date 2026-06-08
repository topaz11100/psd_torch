"""Parameter-only filter-property analysis for proposed dendritic neurons.

The functions in this module do not run data through a model.  They analyse the
linear path from the neuron input/branch input to ``soma_input`` from learned
parameters only.  Returned tensors are one value per output neuron so existing
layer/model aggregation code can compute layer-level and model-level summaries.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch

FILTER_CLASS_CODE = {
    'lp': 0.0,
    'bp': 1.0,
    'hp': 2.0,
    'mixed': 3.0,
}


def normalized_frequency_grid(n_freq: int = 256, *, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return exact-analysis normalized frequencies in [0, 0.5]."""

    n = max(2, int(n_freq))
    return torch.linspace(0.0, 0.5, n, device=device, dtype=dtype)


def _as_float_tensor(value: torch.Tensor | float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(float(value), device=device, dtype=dtype)


def _safe_normalize_response(amplitude: torch.Tensor) -> torch.Tensor:
    return amplitude / amplitude.amax(dim=-1, keepdim=True).clamp_min(1.0e-12)


def summarize_frequency_response(freq: torch.Tensor, amplitude: torch.Tensor) -> dict[str, torch.Tensor]:
    """Summarize one frequency response per neuron.

    Args:
        freq: ``(F,)`` normalized frequency grid in cycles/sample.
        amplitude: ``(N,F)`` non-negative response magnitude.

    Returns:
        Numeric one-dimensional vectors: peak frequency, -3 dB passband edges,
        bandwidth, DC/Nyquist ratios, and a compact numeric filter class code.
    """

    if freq.ndim != 1:
        raise ValueError(f'freq must be rank-1, got {tuple(freq.shape)}.')
    if amplitude.ndim != 2:
        raise ValueError(f'amplitude must have shape (N,F), got {tuple(amplitude.shape)}.')
    if int(amplitude.shape[1]) != int(freq.numel()):
        raise ValueError('amplitude frequency axis must match freq length.')

    amp = torch.nan_to_num(amplitude.real.abs(), nan=0.0, posinf=0.0, neginf=0.0)
    amp_n = _safe_normalize_response(amp)
    peak_idx = torch.argmax(amp_n, dim=-1)
    f_peak = freq[peak_idx]
    r0 = amp_n[:, 0]
    rpi = amp_n[:, -1]
    threshold = torch.as_tensor(1.0 / math.sqrt(2.0), device=amp_n.device, dtype=amp_n.dtype)

    f_low = torch.empty_like(f_peak)
    f_high = torch.empty_like(f_peak)
    class_code = torch.empty_like(f_peak)
    n_neurons = int(amp_n.shape[0])
    last_freq = freq[-1].to(dtype=amp_n.dtype)
    for idx in range(n_neurons):
        above = amp_n[idx] >= threshold
        p = int(peak_idx[idx].item())
        left = p
        while left > 0 and bool(above[left - 1].item()):
            left -= 1
        right = p
        last = int(above.numel()) - 1
        while right < last and bool(above[right + 1].item()):
            right += 1
        f_low[idx] = freq[left]
        f_high[idx] = freq[right]

        touches_dc = left == 0
        touches_nyq = right == last
        if touches_dc and not touches_nyq:
            code = FILTER_CLASS_CODE['lp']
        elif touches_nyq and not touches_dc:
            code = FILTER_CLASS_CODE['hp']
        elif (not touches_dc) and (not touches_nyq):
            code = FILTER_CLASS_CODE['bp']
        else:
            # Flat/all-pass or multi-passband responses fall here.
            code = FILTER_CLASS_CODE['mixed']
        class_code[idx] = float(code)

    return {
        'f_peak': f_peak,
        'f_low_3db': f_low,
        'f_high_3db': f_high,
        'bw_3db': (f_high - f_low).clamp_min(0.0),
        'dc_ratio': r0,
        'nyquist_ratio': rpi,
        'filter_class_code': class_code,
    }


def branch_structure_vectors(*, s: torch.Tensor, d_int: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return per-neuron branch-count and utilization vectors."""

    s_vec = s.detach().reshape(-1).to(dtype=torch.float32)
    d_vec = d_int.detach().reshape(-1).to(dtype=torch.float32)
    if mask.ndim != 2:
        raise ValueError(f'mask must have shape (N,D), got {tuple(mask.shape)}.')
    branch = max(1, int(mask.shape[1]))
    mass = mask.detach().to(dtype=torch.float32).sum(dim=1)
    return {
        's_value': s_vec,
        'active_branch_count': d_vec,
        'branch_mask_mass': mass,
        'branch_utilization': d_vec / float(branch),
        'branch_mass_fraction': mass / float(branch),
    }


def ema_mixture_filter_vectors(
    *,
    alpha: torch.Tensor,
    mask: torch.Tensor,
    s: torch.Tensor,
    mix_weight: torch.Tensor | None = None,
    n_freq: int = 256,
) -> dict[str, torch.Tensor]:
    """Analyse weighted EMA branch mixtures.

    Each branch uses ``H_d(z)=(1-alpha_d)/(1-alpha_d z^-1)`` and the neuron
    response is ``sum_d mask_d * mix_weight_d * H_d / s``.  ``mix_weight=None``
    corresponds to equal soma mixing.
    """

    if alpha.ndim != 2 or mask.ndim != 2 or alpha.shape != mask.shape:
        raise ValueError(f'alpha and mask must both be (N,D), got {tuple(alpha.shape)} and {tuple(mask.shape)}.')
    device = alpha.device
    dtype = alpha.dtype if alpha.is_floating_point() else torch.float32
    a = alpha.to(dtype=dtype).clamp(1.0e-6, 1.0 - 1.0e-6)
    m = mask.to(device=device, dtype=dtype)
    if mix_weight is None:
        w = torch.ones_like(a)
    else:
        if mix_weight.shape != alpha.shape:
            raise ValueError('mix_weight must match alpha shape.')
        w = mix_weight.to(device=device, dtype=dtype)
    s_vec = s.to(device=device, dtype=dtype).reshape(-1).clamp_min(1.0e-6)
    freq = normalized_frequency_grid(n_freq, device=device, dtype=dtype)
    omega = (2.0 * math.pi * freq).to(dtype=dtype)
    z_inv = torch.exp(-1j * omega.to(dtype=torch.complex64 if dtype == torch.float32 else torch.complex128))
    ac = a.to(dtype=z_inv.dtype).unsqueeze(-1)
    gain = ((1.0 - a) * m * w).to(dtype=z_inv.dtype).unsqueeze(-1)
    response = (gain / (1.0 - ac * z_inv.view(1, 1, -1))).sum(dim=1) / s_vec.to(dtype=z_inv.dtype).unsqueeze(-1)
    return summarize_frequency_response(freq, response.abs().to(dtype=dtype))



def complex_pole_mixture_filter_vectors(
    *,
    pole_radius: torch.Tensor,
    pole_angle: torch.Tensor,
    mask: torch.Tensor,
    s: torch.Tensor,
    input_gain_real: torch.Tensor | None = None,
    input_gain_imag: torch.Tensor | None = None,
    n_freq: int = 256,
) -> dict[str, torch.Tensor]:
    """Analyse a mixture of discrete complex first-order IIR branches.

    Each branch is defined directly in the discrete domain as

        z[t] = a z[t-1] + R x[t],   a = rho * exp(j * phi),

    where ``rho`` is the per-sample pole radius and ``phi`` is the pole angle in
    radians/sample.  The response is evaluated on the unit circle as
    ``R / (1 - a z^-1)``; no continuous-time discretisation map is used.
    """

    if pole_radius.ndim != 2 or pole_angle.ndim != 2 or mask.ndim != 2:
        raise ValueError('pole_radius, pole_angle, and mask must all have shape (N,D).')
    if pole_radius.shape != pole_angle.shape or pole_radius.shape != mask.shape:
        raise ValueError('pole_radius, pole_angle, and mask must have identical shape.')
    device = pole_radius.device
    dtype = pole_radius.dtype if pole_radius.is_floating_point() else torch.float32
    rho = pole_radius.to(dtype=dtype).clamp_min(0.0)
    phi = pole_angle.to(device=device, dtype=dtype)
    m = mask.to(device=device, dtype=dtype)
    if input_gain_real is None:
        gr = torch.ones_like(rho)
    else:
        if input_gain_real.shape != rho.shape:
            raise ValueError('input_gain_real must match pole_radius shape.')
        gr = input_gain_real.to(device=device, dtype=dtype)
    if input_gain_imag is None:
        gi = torch.zeros_like(rho)
    else:
        if input_gain_imag.shape != rho.shape:
            raise ValueError('input_gain_imag must match pole_radius shape.')
        gi = input_gain_imag.to(device=device, dtype=dtype)
    s_vec = s.to(device=device, dtype=dtype).reshape(-1).clamp_min(1.0e-6)
    freq = normalized_frequency_grid(n_freq, device=device, dtype=dtype)
    omega_grid = (2.0 * math.pi * freq).to(dtype=dtype)
    complex_dtype = torch.complex64 if dtype in {torch.float16, torch.bfloat16, torch.float32} else torch.complex128
    pole = torch.complex(rho * torch.cos(phi), rho * torch.sin(phi)).to(dtype=complex_dtype).unsqueeze(-1)
    gain = torch.complex(gr, gi).to(dtype=complex_dtype).unsqueeze(-1)
    z_inv = torch.exp(-1j * omega_grid.to(dtype=complex_dtype)).view(1, 1, -1)
    response = ((m.to(dtype=complex_dtype).unsqueeze(-1) * gain) / (1.0 - pole * z_inv)).sum(dim=1)
    response = response / s_vec.to(dtype=complex_dtype).unsqueeze(-1)
    return summarize_frequency_response(freq, response.abs().to(dtype=dtype))


def drf_discrete_filter_vectors(
    *,
    pole_radius: torch.Tensor,
    pole_angle: torch.Tensor,
    mask: torch.Tensor,
    s: torch.Tensor,
    input_gain_real: torch.Tensor | None = None,
    input_gain_imag: torch.Tensor | None = None,
    n_freq: int = 256,
) -> dict[str, torch.Tensor]:
    """Compatibility alias for direct discrete D-RF filter analysis."""

    return complex_pole_mixture_filter_vectors(
        pole_radius=pole_radius,
        pole_angle=pole_angle,
        mask=mask,
        s=s,
        input_gain_real=input_gain_real,
        input_gain_imag=input_gain_imag,
        n_freq=n_freq,
    )


def drf_filter_vectors(
    *,
    tau: torch.Tensor,
    omega: torch.Tensor,
    mask: torch.Tensor,
    s: torch.Tensor,
    delta: float = 1.0,
    n_freq: int = 256,
    kernel_length: int = 256,
) -> dict[str, torch.Tensor]:
    """Backward-compatible alias using direct-discrete pole semantics.

    ``tau`` is interpreted as a sample-domain memory constant and ``omega`` as
    the pole angle in radians/sample.  The implied pole radius is
    ``rho = exp(-delta / tau)``.  No continuous-time input-integration coefficient
    is reconstructed; the direct discrete branch gain defaults to one on the
    real channel.
    """

    del kernel_length
    if tau.ndim != 2 or omega.ndim != 2 or mask.ndim != 2 or tau.shape != omega.shape or tau.shape != mask.shape:
        raise ValueError('tau, omega, and mask must all have shape (N,D).')
    dtype = tau.dtype if tau.is_floating_point() else torch.float32
    t = tau.to(dtype=dtype).clamp_min(1.0e-6)
    radius = torch.exp(-float(delta) / t)
    return complex_pole_mixture_filter_vectors(
        pole_radius=radius,
        pole_angle=omega.to(device=tau.device, dtype=dtype),
        mask=mask,
        s=s,
        input_gain_real=torch.ones_like(radius),
        input_gain_imag=torch.zeros_like(radius),
        n_freq=n_freq,
    )

def detach_filter_map(payload: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Detach all tensors in one stats mapping."""

    return {str(k): v.detach() if isinstance(v, torch.Tensor) else torch.as_tensor(v) for k, v in payload.items()}
