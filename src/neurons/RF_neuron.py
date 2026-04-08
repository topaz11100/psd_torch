from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.neurons.surrogate import SpikeFn


_RF_INIT_TRIM_EPS = 1.0e-4
_RF_FREE_F_CYC_INIT_RANGE = (0.0, 0.5)
_RF_FREE_B_ABS_INIT_RANGE = (0.1, 1.0)


def _trim_uniform_support(low: torch.Tensor, high: torch.Tensor, *, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    width = (high - low).clamp_min(1.0e-12)
    margin = torch.minimum(torch.full_like(width, float(eps)), width * 0.25)
    return low + margin, high - margin


def _softplus_inverse(x: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x)
    return x + torch.log(-torch.expm1(-x))


class RFDenseLayer(nn.Module):
    """Vanilla resonate-and-fire dense layer with exact ZOH discretization only.

    Notes
    -----
    - The project no longer exposes an Euler integration branch.
    - RF clip bounds are specified in normalized frequency units where Nyquist is
      0.5 cycles/sample. The layer converts those bounds internally to angular
      frequency before applying the constraint to ``omega``.
    - Project-level RF statistics are saved as the direct per-step decay factor
      ``rho = exp(b * dt)`` and the normalized resonance frequency
      ``f_cyc_per_sample = omega * dt / (2*pi)``.
    - Unlike the other origin-backed paper neurons, vanilla RF intentionally uses
      the project-standard exact-ZOH discretization because the upper PSD
      experiment specification fixes exact ZOH as the official RF path.
    - Intrinsic RF dynamics are initialized by bounded uniform sampling in the
      physical parameter domain. Free variants sample normalized resonance
      frequency over the valid Nyquist band and damping magnitude over a bounded
      default range. Clipped variants sample the initial normalized frequency
      uniformly inside each neuron's assigned clip interval; damping magnitude
      keeps the same bounded-uniform policy because the public RF clip CLI
      constrains frequency only.

    Project note
    ------------
    ``recurrent=True`` adds a project-side recurrent spike projection while
    keeping the exact-ZOH RF state update unchanged.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dt: float = 1.0,
        threshold: float = 1.0,
        reset_mode: str = "no_reset",
        bias: bool = True,
        spike_fn: Optional[SpikeFn] = None,
        b_init: float = -0.12,
        omega_init: float = 0.45 * math.pi,
        clip_f_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        input_group_mask: Optional[torch.Tensor] = None,
        recurrent: bool = False,
        recurrent_group_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.dt = float(dt)
        self.threshold = float(threshold)
        mode = str(reset_mode).strip().lower()
        if mode not in ("no_reset", "none", "soft_reset", "soft"):
            raise ValueError(f"reset_mode must be one of no_reset/soft_reset, got {reset_mode}")
        self.reset_mode = "soft_reset" if mode in ("soft_reset", "soft") else "no_reset"
        self.spike_fn = spike_fn or SpikeFn(name="fast_sigmoid", lens=0.5, gamma=0.5)
        self.spiking_enabled = True
        self.recurrent = bool(recurrent)
        _ = float(b_init)  # legacy compatibility only; dynamics use bounded-uniform initialization.
        _ = float(omega_init)

        self.fc = nn.Linear(self.input_dim, self.output_dim, bias=bias)
        if self.recurrent:
            self.recurrent_fc = nn.Linear(self.output_dim, self.output_dim, bias=False)
        else:
            self.recurrent_fc = None

        if clip_f_bounds is None:
            self.register_buffer("clip_f_low", torch.zeros(self.output_dim, dtype=torch.float32), persistent=True)
            self.register_buffer("clip_f_high", torch.zeros(self.output_dim, dtype=torch.float32), persistent=True)
            self._clip_enabled = False
        else:
            low, high = clip_f_bounds
            low_t = torch.as_tensor(low, dtype=torch.float32).reshape(self.output_dim)
            high_t = torch.as_tensor(high, dtype=torch.float32).reshape(self.output_dim)
            if torch.any(high_t <= low_t):
                raise ValueError("clip_f_bounds must satisfy high > low for every neuron")
            self.register_buffer("clip_f_low", low_t, persistent=True)
            self.register_buffer("clip_f_high", high_t, persistent=True)
            self._clip_enabled = True

        if input_group_mask is None:
            self.register_buffer(
                "input_group_mask",
                torch.ones(self.output_dim, self.input_dim, dtype=torch.float32),
                persistent=True,
            )
            self._mask_enabled = False
        else:
            mask = torch.as_tensor(input_group_mask, dtype=torch.float32)
            if tuple(mask.shape) != (self.output_dim, self.input_dim):
                raise ValueError(
                    f"input_group_mask must be {(self.output_dim, self.input_dim)}, got {tuple(mask.shape)}"
                )
            self.register_buffer("input_group_mask", mask, persistent=True)
            self._mask_enabled = True

        if recurrent_group_mask is None:
            self.register_buffer("recurrent_group_mask", torch.ones(self.output_dim, self.output_dim, dtype=torch.float32), persistent=True)
            self._recurrent_mask_enabled = False
        else:
            mask = torch.as_tensor(recurrent_group_mask, dtype=torch.float32)
            if tuple(mask.shape) != (self.output_dim, self.output_dim):
                raise ValueError(
                    f"recurrent_group_mask must be {(self.output_dim, self.output_dim)}, got {tuple(mask.shape)}"
                )
            self.register_buffer("recurrent_group_mask", mask, persistent=True)
            self._recurrent_mask_enabled = True

        self.raw_b = nn.Parameter(torch.empty(self.output_dim, dtype=torch.float32))
        self.raw_omega = nn.Parameter(torch.empty(self.output_dim, dtype=torch.float32))
        self.reset_dynamics_parameters()

        self.x: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None
        self.spk: Optional[torch.Tensor] = None

    @classmethod
    def dynamics_init_metadata(cls) -> Dict[str, object]:
        return {
            "policy": "bounded_uniform_physical_domain",
            "free_f_cyc_per_sample_range": [float(_RF_FREE_F_CYC_INIT_RANGE[0]), float(_RF_FREE_F_CYC_INIT_RANGE[1])],
            "free_b_abs_range": [float(_RF_FREE_B_ABS_INIT_RANGE[0]), float(_RF_FREE_B_ABS_INIT_RANGE[1])],
            "clip_f_range_source": "assigned_frequency_clip_interval",
            "clip_b_abs_range": [float(_RF_FREE_B_ABS_INIT_RANGE[0]), float(_RF_FREE_B_ABS_INIT_RANGE[1])],
            "trim_epsilon": float(_RF_INIT_TRIM_EPS),
        }

    def reset_dynamics_parameters(self) -> None:
        with torch.no_grad():
            b_low = torch.full_like(self.raw_b, float(_RF_FREE_B_ABS_INIT_RANGE[0]))
            b_high = torch.full_like(self.raw_b, float(_RF_FREE_B_ABS_INIT_RANGE[1]))
            b_low, b_high = _trim_uniform_support(b_low, b_high, eps=_RF_INIT_TRIM_EPS)
            b_abs = b_low + (b_high - b_low) * torch.rand_like(b_low)
            self.raw_b.copy_(_softplus_inverse(b_abs.clamp_min(_RF_INIT_TRIM_EPS)))

            if self._clip_enabled:
                low = self.clip_f_low.to(dtype=self.raw_omega.dtype, device=self.raw_omega.device).clamp(0.0, 0.5)
                high = self.clip_f_high.to(dtype=self.raw_omega.dtype, device=self.raw_omega.device).clamp(0.0, 0.5)
                low, high = _trim_uniform_support(low, high, eps=_RF_INIT_TRIM_EPS)
                f_cyc = low + (high - low) * torch.rand_like(low)
                width = (self.clip_f_high - self.clip_f_low).to(dtype=self.raw_omega.dtype, device=self.raw_omega.device).clamp_min(1.0e-12)
                ratio = ((f_cyc - self.clip_f_low.to(dtype=self.raw_omega.dtype, device=self.raw_omega.device)) / width).clamp(
                    _RF_INIT_TRIM_EPS,
                    1.0 - _RF_INIT_TRIM_EPS,
                )
                self.raw_omega.copy_(torch.logit(ratio))
            else:
                low = torch.full_like(self.raw_omega, float(_RF_FREE_F_CYC_INIT_RANGE[0]))
                high = torch.full_like(self.raw_omega, float(_RF_FREE_F_CYC_INIT_RANGE[1]))
                low, high = _trim_uniform_support(low, high, eps=_RF_INIT_TRIM_EPS)
                f_cyc = low + (high - low) * torch.rand_like(low)
                omega = (2.0 * math.pi * f_cyc) / max(self.dt, 1.0e-12)
                self.raw_omega.copy_(_softplus_inverse(omega.clamp_min(_RF_INIT_TRIM_EPS)))

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.x = torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype)
        self.y = torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype)
        self.spk = torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype)

    def set_spiking_enabled(self, enabled: bool) -> None:
        self.spiking_enabled = bool(enabled)
        if not self.spiking_enabled and self.spk is not None:
            self.spk.zero_()

    def b(self) -> torch.Tensor:
        return -F.softplus(self.raw_b)

    def omega(self) -> torch.Tensor:
        if not self._clip_enabled:
            return F.softplus(self.raw_omega)
        width = (self.clip_f_high - self.clip_f_low).clamp_min(1e-6)
        f = self.clip_f_low + width * torch.sigmoid(self.raw_omega)
        return (2.0 * math.pi * f) / max(self.dt, 1e-12)

    def rho(self) -> torch.Tensor:
        return torch.exp(self.b() * self.dt)

    def f_cyc_per_sample(self) -> torch.Tensor:
        return (self.omega() * self.dt) / (2.0 * math.pi)

    def f_omega(self) -> torch.Tensor:
        return self.f_cyc_per_sample()

    def effective_weight(self) -> torch.Tensor:
        w = self.fc.weight
        if self._mask_enabled:
            w = w * self.input_group_mask.to(device=w.device, dtype=w.dtype)
        return w

    def effective_recurrent_weight(self) -> Optional[torch.Tensor]:
        if self.recurrent_fc is None:
            return None
        w = self.recurrent_fc.weight
        if self._recurrent_mask_enabled:
            w = w * self.recurrent_group_mask.to(device=w.device, dtype=w.dtype)
        return w

    def _zoh_step(self, I_t: torch.Tensor, *, b: torch.Tensor, omega: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rho = torch.exp(b * self.dt)
        phi = omega * self.dt
        c = torch.cos(phi)
        s = torch.sin(phi)
        den = (b * b + omega * omega).clamp_min(1e-6)
        beta_x = (b * (rho * c - 1.0) + omega * rho * s) / den
        beta_y = (b * rho * s - omega * (rho * c - 1.0)) / den
        x_new = rho * (c * self.x - s * self.y) + beta_x * I_t
        y_new = rho * (s * self.x + c * self.y) + beta_y * I_t
        return x_new, y_new

    def forward_step(self, x_t: torch.Tensor, record: bool = False):
        if self.x is None or self.y is None or self.spk is None:
            self.reset_state(x_t.shape[0], x_t.device, x_t.dtype)

        I_t = F.linear(x_t, self.effective_weight(), self.fc.bias)
        recurrent_i = None
        recurrent_w = self.effective_recurrent_weight()
        if recurrent_w is not None and self.spk is not None:
            recurrent_i = F.linear(self.spk, recurrent_w, None)
            I_t = I_t + recurrent_i

        b = self.b().to(dtype=x_t.dtype, device=x_t.device)
        omega = self.omega().to(dtype=x_t.dtype, device=x_t.device)
        x_new, y_new = self._zoh_step(I_t, b=b, omega=omega)

        if self.spiking_enabled:
            spk = self.spike_fn(x_new - self.threshold)
        else:
            spk = torch.zeros_like(x_new)

        if self.spiking_enabled and self.reset_mode == "soft_reset":
            x_post = x_new - self.threshold * spk
        else:
            x_post = x_new

        self.x = x_post
        self.y = y_new
        self.spk = spk

        if not record:
            return spk

        signals = {
            "dendrite_input": I_t,
            "dendrite_state": x_new,
            "soma_input": x_new,
            "soma_state": x_new,
            "output": spk,
            "state_y": y_new,
        }
        if recurrent_i is not None:
            signals["recurrent_input"] = recurrent_i
        return spk, signals

    def forward_sequence(self, x_seq: torch.Tensor, record: bool | Sequence[str] = False):
        B, T, _ = x_seq.shape
        self.reset_state(B, x_seq.device, x_seq.dtype)

        if record is False:
            record_keys = None
        elif record is True:
            record_keys = ("dendrite_input", "dendrite_state", "soma_input", "soma_state", "output", "state_y")
        else:
            record_keys = tuple(record)

        if record_keys is None:
            out_list = []
            for t in range(T):
                out_list.append(self.forward_step(x_seq[:, t], record=False))
            return torch.stack(out_list, dim=1)

        out_list = []
        rec_lists: Dict[str, list[torch.Tensor]] = {k: [] for k in record_keys}
        for t in range(T):
            y, sig = self.forward_step(x_seq[:, t], record=True)
            out_list.append(y)
            for k in record_keys:
                if k not in sig:
                    raise KeyError(f"Unknown record key: {k!r}")
                rec_lists[k].append(sig[k])
        out_seq = torch.stack(out_list, dim=1)
        rec = {k: torch.stack(rec_lists[k], dim=1) for k in record_keys}
        return out_seq, rec

    def get_timing_params(self) -> Dict[str, torch.Tensor]:
        return {
            "rho": self.rho().detach(),
            "f_cyc_per_sample": self.f_cyc_per_sample().detach(),
        }

    def get_structure_params(self) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        if self._clip_enabled:
            out["clip_f_low"] = self.clip_f_low.detach()
            out["clip_f_high"] = self.clip_f_high.detach()
        if self._mask_enabled:
            out["input_group_mask"] = self.input_group_mask.detach()
        if self.recurrent_fc is not None and self._recurrent_mask_enabled:
            out["recurrent_group_mask"] = self.recurrent_group_mask.detach()
        return out

    def active_param_count(self) -> int:
        active = int(self.raw_b.numel()) + int(self.raw_omega.numel())
        w = self.fc.weight
        if self._mask_enabled:
            mask = self.input_group_mask.to(device=w.device, dtype=torch.bool)
            active += int(mask.sum().item())
        else:
            active += int(w.numel())
        if self.fc.bias is not None and self.fc.bias.requires_grad:
            active += int(self.fc.bias.numel())
        if self.recurrent_fc is not None:
            rw = self.recurrent_fc.weight
            if self._recurrent_mask_enabled:
                mask = self.recurrent_group_mask.to(device=rw.device, dtype=torch.bool)
                active += int(mask.sum().item())
            else:
                active += int(rw.numel())
        return int(active)
