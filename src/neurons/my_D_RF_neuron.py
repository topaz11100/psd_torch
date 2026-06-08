from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.neurons._common import SpikeFn
from src.neurons._soma import apply_soma_reset, effective_soma_threshold, init_soma_threshold_reset, soma_contract_stat_vectors
from src.signal.filter_property import branch_structure_vectors, drf_discrete_filter_vectors


def _soft_branch_mask_from_s(s: torch.Tensor, branch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mask, D_int) for soft dendrite masking."""
    if s.ndim != 1:
        raise ValueError(f"s must be 1D (got shape={tuple(s.shape)})")
    D = int(branch)
    s_c = s.clamp_min(0.0).clamp_max(float(D))
    k = torch.floor(s_c).to(torch.int64)
    frac = (s_c - k.to(s_c.dtype)).clamp(0.0, 1.0)
    idx = torch.arange(D, device=s.device, dtype=torch.int64).view(1, D)
    k2 = k.view(-1, 1)
    mask = (idx < k2).to(s.dtype)
    mask = mask + (idx == k2).to(s.dtype) * frac.view(-1, 1) * (k2 < D).to(s.dtype)
    D_int = (k + (frac > 0).to(torch.int64)).clamp(min=1, max=D)
    return mask, D_int


def _hard_branch_mask_from_s(s: torch.Tensor, branch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mask_hard, D_hard) for hard dendrite masking."""
    if s.ndim != 1:
        raise ValueError(f"s must be 1D (got shape={tuple(s.shape)})")
    D = int(branch)
    s_c = s.clamp_min(0.0).clamp_max(float(D))
    D_hard = torch.floor(s_c + 0.5).to(torch.int64).clamp(min=1, max=D)
    idx = torch.arange(D, device=s.device, dtype=torch.int64).view(1, D)
    mask_hard = (idx < D_hard.view(-1, 1)).to(s.dtype)
    return mask_hard, D_hard


def _inverse_sigmoid_scalar(value: float) -> float:
    v = min(max(float(value), 1.0e-6), 1.0 - 1.0e-6)
    return float(math.log(v / (1.0 - v)))


def _softplus_inverse_scalar(value: float) -> float:
    v = max(float(value), 1.0e-6)
    return float(math.log(math.expm1(v)))


def _drf_orthogonality_loss_weighted(pole_radius: torch.Tensor, pole_angle: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weighted orthogonality loss for direct discrete resonator branches.

    The overlap is computed from the stable-pole inner-product proxy for
    ``a_i = rho_i exp(j phi_i)``.  Radii above one are clipped in this auxiliary
    penalty so the loss remains finite when the model is configured as a
    finite-horizon amplifier.
    """
    if pole_radius.ndim != 2 or pole_angle.ndim != 2 or w.ndim != 2 or pole_radius.shape != pole_angle.shape or pole_radius.shape != w.shape:
        raise ValueError("pole_radius, pole_angle, w must all be (N,D) with the same shape")
    _, D = pole_radius.shape
    if int(D) <= 1:
        return pole_radius.new_zeros(())
    r = pole_radius.clamp_min(0.0).clamp_max(1.0 - 1.0e-6)
    ri = r.unsqueeze(2)
    rj = r.unsqueeze(1)
    pi = pole_angle.unsqueeze(2)
    pj = pole_angle.unsqueeze(1)
    m = ri * rj
    denom = 1.0 + m * m - 2.0 * m * torch.cos(pi - pj)
    denom = denom.clamp_min(1.0e-6)
    num = (1.0 - ri * ri) * (1.0 - rj * rj)
    cos2 = (num / denom) * (w.unsqueeze(2) * w.unsqueeze(1))
    return torch.triu(cos2, diagonal=1).sum(dim=(1, 2)).mean()


class MyDRFDenseLayer(nn.Module):
    """Proposed direct-discrete D-RF neuron.

    Each dendritic branch is a learnable complex first-order IIR branch,

        z_d[t+1] = a_d z_d[t] + R_d I[t+1],
        a_d = rho_d exp(j phi_d),

    implemented with real states ``u=Re(z)`` and ``v=Im(z)``.  The branch count
    ``s`` selects a soft/hard prefix of the fixed branch bank.  The layer exposes
    pole-radius, pole-angle, branch-count, and filter-response statistics for PSD
    analysis.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        branch: int = 8,
        S_min: float = 1.0,
        S_max: Optional[float] = None,
        th_len: int = 4,
        delta: float = 1.0,
        v_pre: float = 1.0,
        bias: bool = True,
        spike_fn: Optional[SpikeFn] = None,
        tau_min: float = 1e-3,
        s_init: Optional[float] = None,
        pole_radius_constrained: bool = True,
        pole_radius_max: float = 0.9999,
        trainable_threshold: bool = False,
        reset_mode: str = "soft_reset",
        emit_spike: bool = True,
        reset_enabled: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.branch = int(branch)
        self.th_len = int(th_len)
        self.delta = float(delta)  # retained as metadata; direct-discrete dynamics use one sample per step.
        init_soma_threshold_reset(
            self,
            output_dim=self.output_dim,
            v_threshold=float(v_pre),
            trainable_threshold=bool(trainable_threshold),
            reset_mode=reset_mode,
            emit_spike=bool(emit_spike),
            reset_enabled=bool(reset_enabled),
        )
        self.v_pre = float(v_pre)  # backward-compatible base-threshold alias.
        self.tau_min = float(tau_min)
        self.pole_radius_constrained = bool(pole_radius_constrained)
        self.pole_radius_max = float(pole_radius_max)
        if not math.isfinite(self.pole_radius_max) or self.pole_radius_max <= 0.0:
            raise ValueError("pole_radius_max must be positive and finite")
        if self.pole_radius_constrained and self.pole_radius_max >= 1.0:
            raise ValueError("pole_radius_max must be < 1 when pole_radius_constrained=True")

        if S_max is None:
            S_max = float(self.branch)
        self.S_min = float(S_min)
        self.S_max = float(S_max)
        if not (1 <= self.branch):
            raise ValueError(f"branch must be >= 1 (got {self.branch})")
        if not (1.0 <= self.S_min <= self.S_max):
            raise ValueError(f"Require 1 <= S_min <= S_max (got S_min={self.S_min}, S_max={self.S_max})")

        self.spike_fn = spike_fn or SpikeFn(name="mg", lens=0.5, gamma=0.5)
        self.fc = nn.Linear(self.input_dim, self.output_dim, bias=bias)

        # Direct discrete branch pole and input gain.
        radius_init = min(0.8, self.pole_radius_max - 1.0e-4) if self.pole_radius_constrained else 0.8
        angle_init = min(1.0, math.pi - 1.0e-4)
        if self.pole_radius_constrained:
            radius_raw_init = _inverse_sigmoid_scalar(radius_init / self.pole_radius_max)
        else:
            radius_raw_init = _softplus_inverse_scalar(radius_init)
        angle_raw_init = _inverse_sigmoid_scalar(angle_init / math.pi)
        self.radius_raw = nn.Parameter(torch.full((self.output_dim, self.branch), radius_raw_init, dtype=torch.float32))
        self.angle_raw = nn.Parameter(torch.full((self.output_dim, self.branch), angle_raw_init, dtype=torch.float32))
        self.input_gain_real = nn.Parameter(torch.ones(self.output_dim, self.branch, dtype=torch.float32))
        self.input_gain_imag = nn.Parameter(torch.zeros(self.output_dim, self.branch, dtype=torch.float32))

        # Adaptive threshold kernel a_k > 0.  This is a soma-local refractory
        # surrogate and is trainable only when soma spikes and reset-history are
        # actually active.  For membrane-only output readouts or reset=none it
        # would not participate in the loss; freezing it prevents DDP unused-
        # parameter failures without changing the computed membrane trajectory.
        self.a_raw = nn.Parameter(torch.zeros(self.th_len), requires_grad=bool(self.emit_spike and self.reset_enabled))

        if s_init is None:
            s_init = float(self.S_min)
        s0 = torch.full((self.output_dim,), float(s_init), dtype=torch.float32)
        if abs(self.S_max - self.S_min) < 1e-12:
            s_norm = torch.zeros_like(s0)
        else:
            s_norm = (s0 - float(self.S_min)) / float(self.S_max - self.S_min)
        eps = 1e-2
        s_norm = s_norm.clamp(eps, 1.0 - eps)
        self.s_raw = nn.Parameter(torch.log(s_norm / (1.0 - s_norm)).clone().detach())

        self.register_buffer("_hard_enabled", torch.zeros((), dtype=torch.uint8), persistent=True)
        self.register_buffer("_s_hard", torch.full((self.output_dim,), float(self.S_min), dtype=torch.float32), persistent=True)
        self.register_buffer("_ste_enabled", torch.zeros((), dtype=torch.uint8), persistent=False)

        self.u: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None
        self.soma_mem: Optional[torch.Tensor] = None
        self.p_hist: Optional[torch.Tensor] = None

    def s(self) -> torch.Tensor:
        if int(self._hard_enabled.item()) == 1:
            return self._s_hard.to(device=self.s_raw.device, dtype=self.s_raw.dtype)
        if abs(self.S_max - self.S_min) < 1e-12:
            return torch.full((self.output_dim,), float(self.S_max), device=self.s_raw.device, dtype=self.s_raw.dtype)
        return self.S_min + (self.S_max - self.S_min) * torch.sigmoid(self.s_raw)

    def d_int(self) -> torch.Tensor:
        if int(self._hard_enabled.item()) == 1:
            return self._s_hard.to(device=self.s_raw.device, dtype=torch.int64).clamp(min=1, max=int(self.branch))
        if int(self._ste_enabled.item()) == 1:
            _, D_hard = _hard_branch_mask_from_s(self.s().to(torch.float32), self.branch)
            return D_hard
        _, D_int = _soft_branch_mask_from_s(self.s(), self.branch)
        return D_int

    def soft_mask(self, dtype: torch.dtype) -> torch.Tensor:
        if int(self._hard_enabled.item()) == 1:
            D_int = self.d_int().to(torch.int64)
            idx = torch.arange(int(self.branch), device=D_int.device, dtype=torch.int64).view(1, int(self.branch))
            return (idx < D_int.view(-1, 1)).to(dtype)
        s = self.s().to(dtype)
        mask_soft, _ = _soft_branch_mask_from_s(s, self.branch)
        if int(self._ste_enabled.item()) == 1:
            mask_hard, _ = _hard_branch_mask_from_s(s, self.branch)
            return mask_soft + (mask_hard - mask_soft).detach()
        return mask_soft

    @torch.no_grad()
    def enable_ste(self, enabled: bool) -> None:
        if int(self._hard_enabled.item()) == 1:
            self._ste_enabled.fill_(0)
            return
        self._ste_enabled.fill_(1 if bool(enabled) else 0)

    @torch.no_grad()
    def harden_branches(self) -> None:
        s_val = self.s().detach().to(torch.float32)
        D = float(self.branch)
        s_hard = torch.floor(torch.minimum(s_val, torch.tensor(D, device=s_val.device, dtype=s_val.dtype)) + 0.5).clamp(min=1.0, max=D)
        self._s_hard.copy_(s_hard.to(self._s_hard.dtype))
        self._hard_enabled.fill_(1)
        self._ste_enabled.fill_(0)
        self.s_raw.requires_grad_(False)
        if self.u is not None:
            m = self.soft_mask(self.u.dtype).to(self.u.device).unsqueeze(0)
            self.u.mul_(m)
        if self.v is not None:
            m = self.soft_mask(self.v.dtype).to(self.v.device).unsqueeze(0)
            self.v.mul_(m)

    def pole_radius(self) -> torch.Tensor:
        if self.pole_radius_constrained:
            return self.pole_radius_max * torch.sigmoid(self.radius_raw)
        return F.softplus(self.radius_raw)

    def pole_angle(self) -> torch.Tensor:
        return math.pi * torch.sigmoid(self.angle_raw)

    def input_gain(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_gain_real, self.input_gain_imag

    # Backward-compatible aliases for old analysis code.  These are sample-domain
    # quantities, not continuous-time ODE parameters.
    def tau(self) -> torch.Tensor:
        radius = self.pole_radius().clamp(min=1.0e-6, max=1.0 - 1.0e-6)
        return -1.0 / torch.log(radius)

    def omega(self) -> torch.Tensor:
        return self.pole_angle()

    def a_kernel(self) -> torch.Tensor:
        return F.softplus(self.a_raw)

    def effective_threshold(self) -> torch.Tensor:
        """Return the soma base-threshold vector before adaptive-threshold terms."""
        return effective_soma_threshold(self)

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.u = torch.zeros(batch_size, self.output_dim, self.branch, device=device, dtype=dtype)
        self.v = torch.zeros(batch_size, self.output_dim, self.branch, device=device, dtype=dtype)
        self.soma_mem = torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype)
        self.p_hist = torch.zeros(batch_size, self.output_dim, self.th_len, device=device, dtype=dtype)

    def _compute_pole_input(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        radius = self.pole_radius()
        angle = self.pole_angle()
        return radius * torch.cos(angle), radius * torch.sin(angle), self.input_gain_real, self.input_gain_imag

    def forward_step(self, x_t: torch.Tensor, record: bool = False):
        if self.u is None or self.v is None or self.soma_mem is None or self.p_hist is None:
            self.reset_state(x_t.shape[0], x_t.device, x_t.dtype)
        mask = self.soft_mask(x_t.dtype).to(x_t.device).unsqueeze(0)
        I_t = self.fc(x_t)
        rho_r, rho_i, gam_r, gam_i = self._compute_pole_input()
        rho_r = rho_r.to(device=x_t.device, dtype=x_t.dtype)
        rho_i = rho_i.to(device=x_t.device, dtype=x_t.dtype)
        gam_r = gam_r.to(device=x_t.device, dtype=x_t.dtype)
        gam_i = gam_i.to(device=x_t.device, dtype=x_t.dtype)
        I_b = I_t.unsqueeze(-1)
        u_new = rho_r.unsqueeze(0) * self.u - rho_i.unsqueeze(0) * self.v + gam_r.unsqueeze(0) * I_b
        v_new = rho_i.unsqueeze(0) * self.u + rho_r.unsqueeze(0) * self.v + gam_i.unsqueeze(0) * I_b
        self.u = u_new * mask
        self.v = v_new * mask
        s_val = self.s().to(x_t.dtype).unsqueeze(0)
        H_t = self.u.sum(dim=2) / s_val.clamp_min(1e-6)
        base_threshold = self.effective_threshold().to(device=x_t.device, dtype=x_t.dtype).unsqueeze(0)
        p_raw = self.spike_fn(H_t - base_threshold)
        update_history = bool(getattr(self, "reset_enabled", True)) and bool(getattr(self, "emit_spike", True))
        p_for_history = p_raw if update_history else torch.zeros_like(p_raw)
        if update_history:
            a = self.a_kernel().to(device=x_t.device, dtype=x_t.dtype)
            adaptive_threshold = (self.p_hist * a.view(1, 1, -1)).sum(dim=2)
        else:
            adaptive_threshold = torch.zeros_like(base_threshold).expand_as(H_t)
        V_th = base_threshold + adaptive_threshold
        soma_signal = H_t - V_th
        raw_spk = self.spike_fn(soma_signal)
        spk = raw_spk if bool(getattr(self, "emit_spike", True)) else torch.zeros_like(raw_spk)
        # Soma-local reset: this records/holds the post-reset soma state but does
        # not alter dendritic resonator states u/v.
        self.soma_mem = apply_soma_reset(self, H_t, spk, V_th)
        self.p_hist = torch.cat([p_for_history.unsqueeze(-1), self.p_hist[:, :, :-1]], dim=2)
        if not record:
            return spk
        signals = {
            "dendrite_input": I_t.unsqueeze(-1).expand(-1, -1, self.branch) * mask,
            "dendrite_state": self.u,
            "soma_input": H_t,
            "soma_state": H_t,
            "soma_signal": soma_signal,
            "adaptive_threshold": adaptive_threshold,
            "soma_state_post_reset": self.soma_mem,
            "output": spk,
        }
        return spk, signals

    def forward_sequence(self, x_seq: torch.Tensor, record: bool | Sequence[str] = False):
        B, T, _ = x_seq.shape
        self.reset_state(B, x_seq.device, x_seq.dtype)
        if record is False:
            record_keys = None
        elif record is True:
            record_keys = ("dendrite_input", "dendrite_state", "soma_input", "soma_state", "soma_signal", "adaptive_threshold", "soma_state_post_reset", "output")
        else:
            record_keys = tuple(record)
        if record_keys is None:
            return torch.stack([self.forward_step(x_seq[:, t], record=False) for t in range(T)], dim=1)
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

    def regularization_loss(self, lambda_ortho: float = 0.0, lambda_s: float = 0.0) -> torch.Tensor:
        loss = self.fc.weight.new_zeros(())
        if lambda_s != 0.0:
            loss = loss + float(lambda_s) * self.s().mean()
        if lambda_ortho != 0.0:
            radius = self.pole_radius()
            angle = self.pole_angle()
            w = self.soft_mask(radius.dtype).to(radius.device)
            loss = loss + float(lambda_ortho) * _drf_orthogonality_loss_weighted(radius, angle, w)
        return loss

    def get_timing_params(self) -> Dict[str, torch.Tensor]:
        return {
            "pole_radius": self.pole_radius().detach().cpu().flatten(),
            "pole_angle": self.pole_angle().detach().cpu().flatten(),
            "input_gain_real": self.input_gain_real.detach().cpu().flatten(),
            "input_gain_imag": self.input_gain_imag.detach().cpu().flatten(),
            "sample_time_constant": self.tau().detach().cpu().flatten(),
        }

    def get_structure_params(self) -> Dict[str, torch.Tensor]:
        return {"s": self.s().detach().cpu(), "D_int": self.d_int().detach().cpu()}

    def forward(self, input_sequence: torch.Tensor, *, return_traces: bool = False) -> tuple[torch.Tensor | None, torch.Tensor]:
        self._last_layer_input = None
        if input_sequence.ndim != 3:
            raise ValueError(f"Expected shape (B,T,C), got {tuple(input_sequence.shape)}")
        if return_traces:
            spike_seq, rec = self.forward_sequence(input_sequence, record=True)
            self._last_layer_input = rec["soma_input"].contiguous()
            self._last_filter_records = {key: value.contiguous() for key, value in rec.items()}
            return rec["soma_state"].contiguous(), spike_seq.contiguous()
        spike_seq = self.forward_sequence(input_sequence, record=False)
        return None, spike_seq.contiguous()

    def filter_stats_vectors(self) -> Dict[str, torch.Tensor]:
        radius = self.pole_radius()
        angle = self.pole_angle()
        gain_r, gain_i = self.input_gain()
        mask = self.soft_mask(radius.dtype).to(radius.device)
        s_value = self.s().to(device=radius.device, dtype=radius.dtype)
        center_frequency = (angle / (2.0 * math.pi)).clamp(0.0, 0.5)
        sample_tau = self.tau()
        mask_sum = mask.sum(dim=1).clamp_min(1.0)
        radius_mean = (radius * mask).sum(dim=1) / mask_sum
        angle_mean = (angle * mask).sum(dim=1) / mask_sum
        vectors: Dict[str, torch.Tensor] = {
            "pole_radius": radius.reshape(-1),
            "damping": (-torch.log(torch.clamp(radius, min=1.0e-12))).reshape(-1),
            "sample_decay_factor": radius.reshape(-1),
            "pole_angle": angle.reshape(-1),
            "pole_real": (radius * torch.cos(angle)).reshape(-1),
            "pole_imag": (radius * torch.sin(angle)).reshape(-1),
            "center_frequency": center_frequency.reshape(-1),
            "input_gain_real": gain_r.reshape(-1),
            "input_gain_imag": gain_i.reshape(-1),
            "sample_time_constant": sample_tau.reshape(-1),
            "stability_margin": (1.0 - radius).reshape(-1),
            "stability_excess": torch.relu(radius - 1.0).reshape(-1),
            "adaptive_threshold_kernel_sum": self.a_kernel().sum().reshape(1).repeat(self.output_dim),
            "pole_radius_mean": radius_mean,
            "pole_angle_mean": angle_mean,
            "pole_radius_std": torch.sqrt((((radius - radius_mean.unsqueeze(1)) ** 2) * mask).sum(dim=1) / mask_sum),
            "pole_angle_std": torch.sqrt((((angle - angle_mean.unsqueeze(1)) ** 2) * mask).sum(dim=1) / mask_sum),
            # Backward-compatible aliases.
            "tau": sample_tau.reshape(-1),
            "omega": angle.reshape(-1),
        }
        vectors.update(soma_contract_stat_vectors(self, dtype=radius.dtype, device=radius.device))
        vectors.update(branch_structure_vectors(s=s_value, d_int=self.d_int(), mask=mask))
        vectors.update(
            drf_discrete_filter_vectors(
                pole_radius=radius,
                pole_angle=angle,
                input_gain_real=gain_r,
                input_gain_imag=gain_i,
                mask=mask,
                s=s_value,
            )
        )
        return {key: value.detach() for key, value in vectors.items()}

    def active_param_count(self) -> int:
        d_int = self.d_int().to(torch.int64)
        active_branches = int(d_int.sum().item())
        active_syn = sum(int(p.numel()) for p in self.fc.parameters())
        active_branch_dyn = 4 * active_branches  # radius, angle, real gain, imaginary gain.
        active_a = self.th_len
        active_s = self.output_dim
        active_threshold = self.output_dim if self.v_threshold_param is not None else 0
        return int(active_syn + active_branch_dyn + active_a + active_s + active_threshold)
