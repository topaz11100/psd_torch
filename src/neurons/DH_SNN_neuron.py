from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from src.neurons.surrogate import SpikeFn
from src.neurons._origin_imports import load_dh_spike_dense


class DHSNNDenseLayer(nn.Module):
    """DH-SNN dense layer backed by the released author implementation.

    The wrapped dynamics come from:
      Origin/Temporal dendritic heterogeneity.../SHD/SNN_layers/spike_dense.py

    This wrapper only adds the project-facing layer API:
      - ``forward_sequence`` returning ``(B, T, N)``
      - optional project recurrent adapter on top of the released dense branch
      - signal recording dictionaries for PSD analysis
      - ``spiking_enabled`` handling for the output-layer ``final_membrane`` mode

    The branch masking, parameter initialization, random state initialization,
    and dendrite/soma update equations are delegated to the released code.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        branch: int = 4,
        v_th: float = 0.5,
        dt: float = 1.0,
        bias: bool = True,
        test_sparsity: bool = False,
        sparsity: float = 0.5,
        mask_share: int = 1,
        spike_fn: Optional[SpikeFn] = None,
        recurrent: bool = False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.branch = int(branch)
        self.v_th = float(v_th)
        self.dt = float(dt)
        self.test_sparsity = bool(test_sparsity)
        self.sparsity = float(sparsity)
        self.mask_share = int(mask_share)
        self.spike_fn = spike_fn or SpikeFn(name="mg", lens=0.5, gamma=0.5)
        self.spiking_enabled = True
        self.recurrent = bool(recurrent)

        origin = load_dh_spike_dense()
        self._origin = origin
        # Released author module. The project wrapper leaves the internal dense
        # branch untouched and, when recurrent mode is enabled, adds only an
        # external spike-to-dendrite projection term.
        self.core = origin.spike_dense_test_denri_wotanh_R(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tau_minitializer="uniform",
            low_m=0,
            high_m=4,
            tau_ninitializer="uniform",
            low_n=0,
            high_n=4,
            vth=self.v_th,
            dt=self.dt,
            branch=self.branch,
            device="cpu",
            bias=bias,
            test_sparsity=self.test_sparsity,
            sparsity=self.sparsity,
            mask_share=self.mask_share,
        )
        if self.recurrent:
            self.recurrent_fc = nn.Linear(self.output_dim, self.output_dim * self.branch, bias=False)
        else:
            self.recurrent_fc = None
        self._prev_spike: Optional[torch.Tensor] = None

    @property
    def fc(self) -> nn.Linear:
        return self.core.dense

    @property
    def tau_m(self) -> torch.nn.Parameter:
        return self.core.tau_m

    @property
    def tau_n(self) -> torch.nn.Parameter:
        return self.core.tau_n

    @property
    def mask(self) -> torch.Tensor:
        return self.core.mask

    def _sync_runtime_device(self, device: torch.device, dtype: torch.dtype) -> None:
        self.core.device = device
        self.core.v_th = torch.ones(1, device=device, dtype=dtype) * float(self.v_th)
        if torch.is_tensor(getattr(self.core, "mask", None)):
            self.core.mask = self.core.mask.to(device=device, dtype=dtype)
        if torch.is_tensor(getattr(self.core, "mem", None)):
            self.core.mem = self.core.mem.to(device=device, dtype=dtype)
        if torch.is_tensor(getattr(self.core, "spike", None)):
            self.core.spike = self.core.spike.to(device=device, dtype=dtype)
        if torch.is_tensor(getattr(self.core, "d_input", None)):
            self.core.d_input = self.core.d_input.to(device=device, dtype=dtype)

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.core.device = device
        self.core.set_neuron_state(int(batch_size))
        self._sync_runtime_device(device, dtype)
        self._prev_spike = torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype)

    def set_spiking_enabled(self, enabled: bool) -> None:
        self.spiking_enabled = bool(enabled)
        if not self.spiking_enabled and torch.is_tensor(getattr(self.core, "spike", None)):
            self.core.spike.zero_()

    def forward_step(self, x_t: torch.Tensor, record: bool = False):
        if not torch.is_tensor(getattr(self.core, "mem", None)):
            self.reset_state(x_t.shape[0], x_t.device, x_t.dtype)
        else:
            self._sync_runtime_device(x_t.device, x_t.dtype)

        # The released training path keeps the hard branch mask by mutating the
        # dense weights. We mirror that behaviour at each step.
        self.core.apply_mask()

        padding = torch.zeros(x_t.size(0), self.core.pad, device=x_t.device, dtype=x_t.dtype)
        k_input = torch.cat((x_t.float(), padding), dim=1)

        beta = torch.sigmoid(self.core.tau_n).to(device=x_t.device, dtype=x_t.dtype)
        d_proj = self.core.dense(k_input).reshape(-1, self.output_dim, self.branch)
        recurrent_i = None
        if self.recurrent_fc is not None and self._prev_spike is not None:
            recurrent_i = self.recurrent_fc(self._prev_spike).reshape(-1, self.output_dim, self.branch)
            d_proj = d_proj + recurrent_i
        self.core.d_input = beta * self.core.d_input + (1.0 - beta) * d_proj
        soma_in = self.core.d_input.sum(dim=2, keepdim=False)

        if self.spiking_enabled:
            self.core.mem, self.core.spike = self._origin.mem_update_pra(
                soma_in,
                self.core.mem,
                self.core.spike,
                self.core.v_th,
                self.core.tau_m,
                self.core.dt,
                device=self.core.device,
            )
        else:
            self.core.mem = self._origin.output_Neuron_pra(
                soma_in,
                self.core.mem,
                self.core.tau_m,
                self.core.dt,
                device=self.core.device,
            )
            self.core.spike = torch.zeros_like(self.core.mem)
        self._prev_spike = self.core.spike

        if not record:
            return self.core.spike

        signals = {
            "dendrite_input": d_proj,
            "dendrite_state": self.core.d_input,
            "soma_input": soma_in,
            "soma_state": self.core.mem,
            "output": self.core.spike,
        }
        if recurrent_i is not None:
            signals["recurrent_input"] = recurrent_i
        return self.core.spike, signals

    def forward_sequence(self, x_seq: torch.Tensor, record: bool | Sequence[str] = False):
        bsz, steps, _ = x_seq.shape
        self.reset_state(int(bsz), x_seq.device, x_seq.dtype)

        if record is False:
            record_keys = None
        elif record is True:
            record_keys = ("dendrite_input", "dendrite_state", "soma_input", "soma_state", "output")
        else:
            record_keys = tuple(record)

        if record_keys is None:
            out_list = []
            for t in range(int(steps)):
                out_list.append(self.forward_step(x_seq[:, t], record=False))
            return torch.stack(out_list, dim=1)

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
            "tau_n": self.tau_n.detach().cpu().flatten(),
            "tau_m": self.tau_m.detach().cpu().flatten(),
        }

    def active_param_count(self) -> int:
        return sum(int(p.numel()) for p in self.parameters() if p.requires_grad)
