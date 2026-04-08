from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from src.neurons.surrogate import SpikeFn
from src.neurons._origin_imports import load_drf_newlayer_module


class DRFDenseLayer(nn.Module):
    """D-RF layer backed by the released BiRF implementation.

    The released D-RF repository exposes its neuron through
    ``Origin/.../models/layers.py`` (`BiRFKernel` / `BiRFModel`) rather than as
    a standalone dense cell. This wrapper keeps the released kernel, parameter
    initialization, and spike nonlinearity intact, and only adds the minimum
    project adapter pieces:

      - a feed-forward projection ``fc`` so the layer fits the common hidden-
        layer interface ``(B, T, input_dim) -> (B, T, output_dim)``
      - optional signal-record dictionaries for PSD analysis
      - a no-spike branch for the project's ``final_membrane`` output readout

    Fairness / reproducibility note
    -------------------------------
    The default sequence path is intentionally tied as closely as possible to
    the released author code. For plain sequence forwarding and for the output-
    layer membrane/spike record path, this wrapper reuses the author FFT-based
    `BiRFModel.forward` computation almost verbatim. Only the richer hidden-
    state recording path falls back to a step-wise adapter, because the released
    repository does not expose per-step dendritic state tensors directly.
    """

    _ALL_KEYS = ("dendrite_input", "dendrite_state", "state_v", "pre_hist", "V_th", "soma_input", "soma_state", "output")
    _EXACT_SEQUENCE_KEYS = {"soma_input", "soma_state", "output", "pre_hist", "V_th"}

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        branch: int = 4,
        th_len: int = 4,
        v_pre: float = 1.0,
        bias: bool = True,
        spike_fn: Optional[SpikeFn] = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.branch = int(branch)
        self.th_len = int(th_len)
        self.v_pre = float(v_pre)
        self.spike_fn = spike_fn  # kept for signature compatibility only
        self.spiking_enabled = True

        self.fc = nn.Linear(self.input_dim, self.output_dim, bias=bias)

        origin = load_drf_newlayer_module()
        self._origin = origin
        self.core = origin.BiRFModel(d_model=self.output_dim, d_state=self.branch)
        self.register_parameter("_alpha_th_raw_stub", nn.Parameter(torch.zeros(self.th_len), requires_grad=False))

        self.u: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None
        self.pre_hist: Optional[torch.Tensor] = None
        self.V_th: Optional[torch.Tensor] = None
        self.spk: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Public aliases expected by project utilities
    # ------------------------------------------------------------------
    @property
    def tau_raw(self):
        return self.core.kernel.log_A_real

    @property
    def omega_raw(self):
        return self.core.kernel.A_imag

    @property
    def C(self):
        return self.core.kernel.C

    @property
    def alpha_th_raw(self):
        return self._alpha_th_raw_stub

    # ------------------------------------------------------------------
    # Parameter views
    # ------------------------------------------------------------------
    def tau(self) -> torch.Tensor:
        return torch.exp(-self.core.kernel.log_A_real)

    def omega(self) -> torch.Tensor:
        return self.core.kernel.A_imag

    def alpha_th(self) -> torch.Tensor:
        return torch.zeros(self.th_len, device=self.core.kernel.log_dt.device, dtype=self.core.kernel.log_dt.dtype)

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.u = torch.zeros(batch_size, self.output_dim, self.branch, device=device, dtype=dtype)
        self.v = torch.zeros(batch_size, self.output_dim, self.branch, device=device, dtype=dtype)
        self.pre_hist = torch.zeros(batch_size, self.output_dim, self.th_len, device=device, dtype=dtype)
        self.V_th = torch.full((batch_size, self.output_dim), float(self.v_pre), device=device, dtype=dtype)
        self.spk = torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype)

    def set_spiking_enabled(self, enabled: bool) -> None:
        self.spiking_enabled = bool(enabled)
        if not self.spiking_enabled and self.spk is not None:
            self.spk.zero_()

    # ------------------------------------------------------------------
    # Exact released sequence path (author code aligned)
    # ------------------------------------------------------------------
    def _exact_membrane_sequence(self, projected_seq: torch.Tensor) -> torch.Tensor:
        """Return membrane sequence using the released BiRF FFT path.

        ``projected_seq`` has project shape ``(B, T, H)``. The body below keeps
        the released `BiRFModel.forward` computation almost line-for-line, with
        only axis conversion added to match the project's dense-layer API.
        """
        u = projected_seq.transpose(1, 2).contiguous()  # (B, H, T)
        L = u.size(-1)
        k = self.core.kernel(L=L)  # (H, L)
        k_f = torch.fft.rfft(k, n=2 * L)  # (H, L_fft)
        u_f = torch.fft.rfft(u, n=2 * L)  # (B, H, L_fft)
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]  # (B, H, L)
        y = y + self.core.D.unsqueeze(dim=-1) * u
        return y.transpose(1, 2).contiguous()  # (B, T, H)

    def _exact_output_sequence(self, projected_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        membrane_seq = self._exact_membrane_sequence(projected_seq)
        if self.spiking_enabled:
            out_seq = self.core.act1(membrane_seq - 1.0)
        else:
            out_seq = torch.zeros_like(membrane_seq)
        return out_seq, membrane_seq

    def _can_use_exact_sequence_path(self, record_keys: Optional[Sequence[str]]) -> bool:
        if record_keys is None:
            return True
        return set(str(k) for k in record_keys).issubset(self._EXACT_SEQUENCE_KEYS)

    # ------------------------------------------------------------------
    # Step-wise adapter path (only for rich hidden-state recording)
    # ------------------------------------------------------------------
    def _transition_params(self, *, device: torch.device, dtype: torch.dtype):
        dt = torch.exp(self.core.kernel.log_dt).to(device=device, dtype=dtype).unsqueeze(-1)  # (H, 1)
        A_real = (-torch.exp(self.core.kernel.log_A_real)).to(device=device, dtype=dtype)
        A_imag = self.core.kernel.A_imag.to(device=device, dtype=dtype)
        A = torch.complex(A_real, A_imag)
        dtA = A * dt
        rho = torch.exp(dtA)
        gamma = (torch.exp(dtA) - 1.0) / A
        C = torch.view_as_complex(self.core.kernel.C).to(device=device)
        D = self.core.D.to(device=device, dtype=dtype)
        return rho, gamma, C, D

    def forward_step(self, x_t: torch.Tensor, record: bool = False):
        if self.u is None or self.v is None or self.pre_hist is None or self.V_th is None or self.spk is None:
            self.reset_state(x_t.shape[0], x_t.device, x_t.dtype)

        in_sum = self.fc(x_t)
        rho, gamma, C, D = self._transition_params(device=x_t.device, dtype=x_t.dtype)

        z_prev = torch.complex(self.u, self.v)
        complex_input = in_sum.unsqueeze(-1).to(dtype=z_prev.real.dtype)
        z_new = rho.unsqueeze(0) * z_prev + gamma.unsqueeze(0) * torch.complex(complex_input, torch.zeros_like(complex_input))
        self.u = z_new.real.to(dtype=x_t.dtype)
        self.v = z_new.imag.to(dtype=x_t.dtype)

        membrane = (C.unsqueeze(0) * z_new).sum(dim=2).real.to(dtype=x_t.dtype) + D.unsqueeze(0) * in_sum
        self.V_th = torch.full_like(membrane, float(self.v_pre))

        if self.spiking_enabled:
            spk = self.core.act1(membrane - 1.0)
        else:
            spk = torch.zeros_like(membrane)
        self.spk = spk

        # The released BiRF path has no adaptive-threshold history buffer. Keep a
        # zero placeholder so the project-wide recording schema remains valid.
        if self.pre_hist is not None:
            self.pre_hist.zero_()

        if not record:
            return spk

        signals = {
            "dendrite_input": in_sum.unsqueeze(-1).expand(-1, -1, self.branch),
            "dendrite_state": self.u,
            "state_v": self.v,
            "pre_hist": self.pre_hist,
            "V_th": self.V_th,
            "soma_input": membrane,
            "soma_state": membrane,
            "output": spk,
        }
        return spk, signals

    def forward_sequence(self, x_seq: torch.Tensor, record: bool | Sequence[str] = False):
        bsz, steps, _ = x_seq.shape

        if record is False:
            record_keys = None
        elif record is True:
            record_keys = self._ALL_KEYS
        else:
            record_keys = tuple(record)

        if self._can_use_exact_sequence_path(record_keys):
            self.reset_state(int(bsz), x_seq.device, x_seq.dtype)
            projected_seq = self.fc(x_seq)
            out_seq, membrane_seq = self._exact_output_sequence(projected_seq)
            self.spk = out_seq[:, -1].contiguous()
            self.V_th = torch.full_like(membrane_seq[:, -1], float(self.v_pre))
            if self.pre_hist is not None:
                self.pre_hist.zero_()

            if record_keys is None:
                return out_seq

            rec: Dict[str, torch.Tensor] = {}
            for key in record_keys:
                if key == "output":
                    rec[key] = out_seq
                elif key in ("soma_input", "soma_state"):
                    rec[key] = membrane_seq
                elif key == "V_th":
                    rec[key] = torch.full_like(membrane_seq, float(self.v_pre))
                elif key == "pre_hist":
                    rec[key] = torch.zeros(bsz, steps, self.output_dim, self.th_len, device=x_seq.device, dtype=x_seq.dtype)
                else:
                    raise KeyError(f"Unknown record key: {key!r}")
            return out_seq, rec

        self.reset_state(int(bsz), x_seq.device, x_seq.dtype)
        out_list = []
        rec_lists: Dict[str, list[torch.Tensor]] = {k: [] for k in record_keys}
        for t in range(int(steps)):
            y, sig = self.forward_step(x_seq[:, t], record=True)
            out_list.append(y)
            for k in record_keys:
                if k not in sig:
                    raise KeyError(f"Unknown record key: {k!r}")
                rec_lists[k].append(sig[k])
        out_seq = torch.stack(out_list, dim=1)
        rec = {k: torch.stack(v, dim=1) for k, v in rec_lists.items()}
        return out_seq, rec

    def get_timing_params(self) -> Dict[str, torch.Tensor]:
        return {
            "tau": self.tau().detach().cpu().flatten(),
            "omega": self.omega().detach().cpu().flatten(),
            "alpha_th": self.alpha_th().detach().cpu().flatten(),
        }

    def active_param_count(self) -> int:
        return sum(int(p.numel()) for p in self.parameters() if p.requires_grad)
