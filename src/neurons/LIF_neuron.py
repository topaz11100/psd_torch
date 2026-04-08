from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.surrogate import SpikeFn


_LIF_INIT_TRIM_EPS = 1.0e-4
_LIF_FREE_ALPHA_INIT_RANGE = (0.0, 1.0)


def _trim_uniform_support(low: torch.Tensor, high: torch.Tensor, *, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    width = (high - low).clamp_min(1.0e-12)
    margin = torch.minimum(torch.full_like(width, float(eps)), width * 0.25)
    return low + margin, high - margin


class LIFDenseLayer(nn.Module):
    """Baseline dense LIF layer with learnable decay and fixed subtractive reset.

    The layer dynamics are intentionally restricted to

        u[t] = alpha * u[t-1] + I[t] - v_th * o[t-1]

    so the public interface does not expose an arbitrary reset level. This keeps
    the implementation aligned with the project's PSD-analysis LIF variants.

    Project note
    ------------
    ``recurrent=True`` enables a project-side recurrent adapter. The intrinsic
    LIF state update remains the same; we only add a learnable projection from
    the previous output spike vector back into the current input current. This
    keeps the recurrent path compatible with the clip/structure variants without
    changing the core LIF equation itself.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        alpha: float = 0.9,
        v_th: float = 1.0,
        bias: bool = True,
        spike_fn: Optional[SpikeFn] = None,
        learnable_threshold: bool = False,
        alpha_clip_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        input_group_mask: Optional[torch.Tensor] = None,
        recurrent: bool = False,
        recurrent_group_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.spike_fn = spike_fn or SpikeFn(name="mg", lens=0.5, gamma=0.5)
        self.spiking_enabled = True
        self.recurrent = bool(recurrent)
        _ = float(alpha)  # legacy compatibility only; dynamics use bounded-uniform initialization.

        self.learnable_threshold = bool(learnable_threshold)
        self.v_th = float(v_th)
        if self.learnable_threshold:
            self.v_th_raw = nn.Parameter(torch.full((self.output_dim,), float(v_th)))

        self.fc = nn.Linear(self.input_dim, self.output_dim, bias=bias)
        if self.recurrent:
            # The recurrent adapter is a pure project addition. The previous
            # output spike vector is mapped back to the current input current.
            self.recurrent_fc = nn.Linear(self.output_dim, self.output_dim, bias=False)
        else:
            self.recurrent_fc = None

        if alpha_clip_bounds is None:
            self.register_buffer("clip_alpha_low", torch.zeros(self.output_dim, dtype=torch.float32), persistent=True)
            self.register_buffer("clip_alpha_high", torch.zeros(self.output_dim, dtype=torch.float32), persistent=True)
            self._alpha_clip_enabled = False
        else:
            low, high = alpha_clip_bounds
            low_t = torch.as_tensor(low, dtype=torch.float32).reshape(self.output_dim)
            high_t = torch.as_tensor(high, dtype=torch.float32).reshape(self.output_dim)
            if torch.any(high_t <= low_t):
                raise ValueError("alpha_clip_bounds must satisfy high > low for every neuron")
            self.register_buffer("clip_alpha_low", low_t, persistent=True)
            self.register_buffer("clip_alpha_high", high_t, persistent=True)
            self._alpha_clip_enabled = True

        if input_group_mask is None:
            self.register_buffer("input_group_mask", torch.ones(self.output_dim, self.input_dim, dtype=torch.float32), persistent=True)
            self._mask_enabled = False
        else:
            mask = torch.as_tensor(input_group_mask, dtype=torch.float32)
            if tuple(mask.shape) != (self.output_dim, self.input_dim):
                raise ValueError(f"input_group_mask must be {(self.output_dim, self.input_dim)}, got {tuple(mask.shape)}")
            self.register_buffer("input_group_mask", mask, persistent=True)
            self._mask_enabled = True

        if recurrent_group_mask is None:
            self.register_buffer("recurrent_group_mask", torch.ones(self.output_dim, self.output_dim, dtype=torch.float32), persistent=True)
            self._recurrent_mask_enabled = False
        else:
            mask = torch.as_tensor(recurrent_group_mask, dtype=torch.float32)
            if tuple(mask.shape) != (self.output_dim, self.output_dim):
                raise ValueError(f"recurrent_group_mask must be {(self.output_dim, self.output_dim)}, got {tuple(mask.shape)}")
            self.register_buffer("recurrent_group_mask", mask, persistent=True)
            self._recurrent_mask_enabled = True

        self.alpha_raw = nn.Parameter(torch.empty(self.output_dim, dtype=torch.float32))
        self.reset_dynamics_parameters()

        self.mem: Optional[torch.Tensor] = None
        self.spk: Optional[torch.Tensor] = None

    @classmethod
    def dynamics_init_metadata(cls) -> Dict[str, object]:
        return {
            "policy": "bounded_uniform_physical_domain",
            "free_alpha_range": [float(_LIF_FREE_ALPHA_INIT_RANGE[0]), float(_LIF_FREE_ALPHA_INIT_RANGE[1])],
            "clip_alpha_range_source": "assigned_alpha_clip_interval",
            "trim_epsilon": float(_LIF_INIT_TRIM_EPS),
        }

    def reset_dynamics_parameters(self) -> None:
        with torch.no_grad():
            if self._alpha_clip_enabled:
                low = self.clip_alpha_low.to(dtype=self.alpha_raw.dtype, device=self.alpha_raw.device).clamp(0.0, 1.0)
                high = self.clip_alpha_high.to(dtype=self.alpha_raw.dtype, device=self.alpha_raw.device).clamp(0.0, 1.0)
            else:
                low = torch.full_like(self.alpha_raw, float(_LIF_FREE_ALPHA_INIT_RANGE[0]))
                high = torch.full_like(self.alpha_raw, float(_LIF_FREE_ALPHA_INIT_RANGE[1]))
            low, high = _trim_uniform_support(low, high, eps=_LIF_INIT_TRIM_EPS)
            alpha = low + (high - low) * torch.rand_like(low)
            self.alpha_raw.copy_(torch.logit(alpha.clamp(_LIF_INIT_TRIM_EPS, 1.0 - _LIF_INIT_TRIM_EPS)))

    def alpha(self) -> torch.Tensor:
        if not self._alpha_clip_enabled:
            return torch.sigmoid(self.alpha_raw)
        width = (self.clip_alpha_high - self.clip_alpha_low).clamp_min(1e-6)
        return self.clip_alpha_low + width * torch.sigmoid(self.alpha_raw)

    def threshold(self) -> torch.Tensor:
        if self.learnable_threshold:
            return self.v_th_raw
        return torch.full((self.output_dim,), float(self.v_th), device=self.alpha_raw.device, dtype=self.alpha_raw.dtype)

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

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.mem = torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype)
        self.spk = torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype)

    def set_spiking_enabled(self, enabled: bool) -> None:
        self.spiking_enabled = bool(enabled)
        if not self.spiking_enabled and self.spk is not None:
            self.spk.zero_()

    def forward_step(self, x_t: torch.Tensor, record: bool = False):
        if self.mem is None or self.spk is None:
            self.reset_state(x_t.shape[0], x_t.device, x_t.dtype)

        i_t = F.linear(x_t, self.effective_weight(), self.fc.bias)
        recurrent_i = None
        recurrent_w = self.effective_recurrent_weight()
        if recurrent_w is not None and self.spk is not None:
            recurrent_i = F.linear(self.spk, recurrent_w, None)
            i_t = i_t + recurrent_i

        alpha = self.alpha().to(device=x_t.device, dtype=x_t.dtype).unsqueeze(0)
        th = self.threshold().to(device=x_t.device, dtype=x_t.dtype).unsqueeze(0)
        mem_pre = self.mem * alpha + i_t - th * self.spk
        if self.spiking_enabled:
            spk = self.spike_fn(mem_pre - th)
        else:
            spk = torch.zeros_like(mem_pre)
        self.mem = mem_pre
        self.spk = spk

        if not record:
            return spk

        signals = {
            "dendrite_input": i_t,
            "dendrite_state": mem_pre,
            "soma_input": i_t,
            "soma_state": mem_pre,
            "output": spk,
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
            record_keys = ("dendrite_input", "dendrite_state", "soma_input", "soma_state", "output")
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
        return {"alpha": self.alpha().detach()}

    def get_structure_params(self) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        if self._alpha_clip_enabled:
            out["clip_alpha_low"] = self.clip_alpha_low.detach()
            out["clip_alpha_high"] = self.clip_alpha_high.detach()
        if self._mask_enabled:
            out["input_group_mask"] = self.input_group_mask.detach()
        if self.recurrent_fc is not None and self._recurrent_mask_enabled:
            out["recurrent_group_mask"] = self.recurrent_group_mask.detach()
        return out

    def active_param_count(self) -> int:
        active = int(self.alpha_raw.numel())
        w = self.fc.weight
        if self._mask_enabled:
            mask = self.input_group_mask.to(device=w.device, dtype=torch.bool)
            active += int(mask.sum().item())
        else:
            active += int(w.numel())
        if self.fc.bias is not None and self.fc.bias.requires_grad:
            active += int(self.fc.bias.numel())
        if self.learnable_threshold:
            active += int(self.v_th_raw.numel())
        if self.recurrent_fc is not None:
            rw = self.recurrent_fc.weight
            if self._recurrent_mask_enabled:
                mask = self.recurrent_group_mask.to(device=rw.device, dtype=torch.bool)
                active += int(mask.sum().item())
            else:
                active += int(rw.numel())
        return int(active)
