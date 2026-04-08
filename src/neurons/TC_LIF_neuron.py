from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from src.neurons.surrogate import SpikeFn
from src.neurons._origin_imports import OriginSurrogateAdapter, load_tclif_module
from src.neurons.sequence_adapter import normalize_record_keys, rollout_sequence


class TCLIFDenseLayer(nn.Module):
    """TC-LIF dense layer backed by the released author ``TCLIFNode``.

    The released node defines the two-compartment state update from:
    - Paper: TC-LIF A Two-Compartment Spiking Neuron Model for Long-Term Sequential Modelling
    - Code: Origin/TC-LIF .../SHD-SSC/spiking_neuron/TCLIF.py

    This wrapper only
    adds the feed-forward projection, optional project recurrent adapter,
    sequence API, recording hooks, and the project's ``spiking_enabled`` switch
    for ``final_membrane`` output readout.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        v_th: float = 1.0,
        gamma: float = 0.5,
        bias: bool = True,
        spike_fn: Optional[SpikeFn] = None,
        recurrent: bool = False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.v_th = float(v_th)
        self.gamma = float(gamma)
        self.spike_fn = spike_fn or SpikeFn(name="mg", lens=0.5, gamma=0.5)
        self.spiking_enabled = True
        self.recurrent = bool(recurrent)

        self.fc = nn.Linear(self.input_dim, self.output_dim, bias=bias)
        if self.recurrent:
            self.recurrent_fc = nn.Linear(self.output_dim, self.output_dim, bias=False)
        else:
            self.recurrent_fc = None
        self._prev_spk: Optional[torch.Tensor] = None

        origin = load_tclif_module()
        self._origin = origin
        # Author node from the released TC-LIF repository. We keep the node
        # logic intact and only wrap it with project IO / recording helpers.
        self.node = origin.TCLIFNode(
            v_threshold=self.v_th,
            v_reset=0.0,
            surrogate_function=OriginSurrogateAdapter(self.spike_fn),
            detach_reset=False,
            hard_reset=False,
            step_mode="s",
            k=2,
            decay_factor=torch.full([1, 2], 0.0, dtype=torch.float32),
            gamma=self.gamma,
        )

    @property
    def decay_factor(self) -> torch.nn.Parameter:
        return self.node.decay_factor

    @property
    def v1(self):
        return self.node.names["v1"]

    @property
    def v2(self):
        return self.node.names["v2"]

    @property
    def spk(self):
        return self._prev_spk

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.node.reset()
        self.node.v = 0.0
        self.node.names["v1"] = 0.0
        self.node.names["v2"] = 0.0
        self._prev_spk = torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype)

    def set_spiking_enabled(self, enabled: bool) -> None:
        self.spiking_enabled = bool(enabled)

    def _step_node(self, i_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.node.v_float_to_tensor(i_t)
        self.node.neuronal_charge(i_t)
        v1_pre = self.node.names["v1"]
        v2_pre = self.node.names["v2"]
        if self.spiking_enabled:
            spk = self.node.neuronal_fire()
            self.node.neuronal_reset(spk)
        else:
            spk = torch.zeros_like(v2_pre)
        return spk, v1_pre, v2_pre

    def forward_step(self, x_t: torch.Tensor, record: bool = False):
        if self._prev_spk is None:
            self.reset_state(x_t.shape[0], x_t.device, x_t.dtype)
        i_t = self.fc(x_t)
        recurrent_i = None
        if self.recurrent_fc is not None and self._prev_spk is not None:
            recurrent_i = self.recurrent_fc(self._prev_spk)
            i_t = i_t + recurrent_i
        spk, v1_pre, v2_pre = self._step_node(i_t)
        self._prev_spk = spk

        if not record:
            return spk

        signals = {
            "dendrite_input": i_t,
            "dendrite_state": v1_pre,
            "soma_input": v2_pre,
            "soma_state": v2_pre,
            "output": spk,
        }
        if recurrent_i is not None:
            signals["recurrent_input"] = recurrent_i
        return spk, signals

    def forward_sequence(self, x_seq: torch.Tensor, record: bool | Sequence[str] = False):
        bsz, steps, _ = x_seq.shape
        self.reset_state(int(bsz), x_seq.device, x_seq.dtype)

        _ = steps
        record_keys = normalize_record_keys(
            record,
            ("dendrite_input", "dendrite_state", "soma_input", "soma_state", "output"),
        )
        return rollout_sequence(x_seq, step_fn=self.forward_step, record_keys=record_keys)

    def get_timing_params(self) -> Dict[str, torch.Tensor]:
        df = torch.sigmoid(self.decay_factor.detach().cpu().view(-1))
        return {
            "decay_factor_0": df[0].repeat(self.output_dim),
            "decay_factor_1": df[1].repeat(self.output_dim),
        }

    def active_param_count(self) -> int:
        return sum(int(p.numel()) for p in self.parameters() if p.requires_grad)
