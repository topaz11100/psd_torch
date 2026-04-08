from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from src.common.surrogate import SpikeFn
from src.neurons._origin_imports import OriginSurrogateAdapter, load_tslif_module


class TSLIFDenseLayer(nn.Module):
    """TS-LIF dense layer backed by the released author ``TSLIFNode``.

    The original repository couples ``TSLIFNode`` to a larger forecasting stack,
    so this wrapper only adds the feed-forward projection, optional project
    recurrent adapter, sequence API, recording hooks, and the project's
    ``spiking_enabled`` switch for ``final_membrane`` output readout.

    One small adaptation is necessary to make the released node generic over
    layer width: the repository hard-codes ``alpha_s`` and ``alpha_l`` to width
    128, so the wrapper reinitializes those two author parameters to
    ``(1, output_dim)`` while leaving the node logic unchanged.
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
        self.spike_fn = spike_fn or SpikeFn(name='mg', lens=0.5, gamma=0.5)
        self.spiking_enabled = True
        self.recurrent = bool(recurrent)

        self.fc = nn.Linear(self.input_dim, self.output_dim, bias=bias)
        if self.recurrent:
            self.recurrent_fc = nn.Linear(self.output_dim, self.output_dim, bias=False)
        else:
            self.recurrent_fc = None
        self._prev_spk: Optional[torch.Tensor] = None

        origin = load_tslif_module()
        self._origin = origin
        self.node = origin.TSLIFNode(
            v_threshold=self.v_th,
            v_reset=0.0,
            surrogate_function=OriginSurrogateAdapter(self.spike_fn),
            detach_reset=False,
            hard_reset=False,
            step_mode='s',
            k=2,
            decay_factor=torch.tensor([0.8, 0.2, 0.3, 0.7], dtype=torch.float32),
            gamma=self.gamma,
        )
        # The released implementation hard-codes width 128 for these two gain
        # parameters. Recreate them at the requested layer width so the author
        # node can be reused as a generic dense layer.
        self.node.alpha_s = nn.Parameter(torch.randn(1, self.output_dim, dtype=torch.float32))
        self.node.alpha_l = nn.Parameter(torch.randn(1, self.output_dim, dtype=torch.float32))

    @property
    def decay_factor(self) -> torch.nn.Parameter:
        return self.node.decay_factor

    @property
    def kk(self) -> torch.nn.Parameter:
        return self.node.kk

    @property
    def yy(self) -> torch.nn.Parameter:
        return self.node.yy

    @property
    def alpha_s(self) -> torch.nn.Parameter:
        return self.node.alpha_s

    @property
    def alpha_l(self) -> torch.nn.Parameter:
        return self.node.alpha_l

    @property
    def v1(self):
        return self.node.names['v1']

    @property
    def v2(self):
        return self.node.names['v2']

    @property
    def spk(self):
        return self._prev_spk

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.node.reset()
        self.node.v = 0.0
        self.node.v_s = 0.0
        self.node.names['v1'] = 0.0
        self.node.names['v2'] = 0.0
        self._prev_spk = torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype)

    def set_spiking_enabled(self, enabled: bool) -> None:
        self.spiking_enabled = bool(enabled)

    def _step_node(self, i_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.node.v_float_to_tensor(i_t)
        self.node.neuronal_charge(i_t)
        v1_pre = self.node.names['v1']
        v2_pre = self.node.names['v2']
        if self.spiking_enabled:
            s_s, s_l = self.node.sl_neuronal_fire()
            spk = self.node.alpha_s * s_s + self.node.alpha_l * s_l
            self.node.neuronal_reset(s_s, s_l)
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
            'dendrite_input': i_t,
            'dendrite_state': v1_pre,
            'soma_input': v2_pre,
            'soma_state': v2_pre,
            'output': spk,
        }
        if recurrent_i is not None:
            signals['recurrent_input'] = recurrent_i
        return spk, signals

    def forward_sequence(self, x_seq: torch.Tensor, record: bool | Sequence[str] = False):
        bsz, steps, _ = x_seq.shape
        self.reset_state(int(bsz), x_seq.device, x_seq.dtype)

        if record is False:
            record_keys = None
        elif record is True:
            record_keys = ('dendrite_input', 'dendrite_state', 'soma_input', 'soma_state', 'output')
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
                    raise KeyError(f'Unknown record key: {k!r}')
                rec_lists[k].append(sig[k])
        out_seq = torch.stack(out_list, dim=1)
        rec = {k: torch.stack(v, dim=1) for k, v in rec_lists.items()}
        return out_seq, rec

    def get_timing_params(self) -> Dict[str, torch.Tensor]:
        decay = self.decay_factor.detach().cpu().view(-1)
        kappa = torch.sigmoid((self.alpha_s - self.alpha_l).detach().cpu()).reshape(-1)
        return {
            'alpha1': decay[0].repeat(self.output_dim),
            'alpha2': decay[2].repeat(self.output_dim),
            'beta1': decay[1].repeat(self.output_dim),
            'beta2': decay[3].repeat(self.output_dim),
            'gamma1': self.yy.detach().cpu().view(-1).repeat(self.output_dim),
            'gamma2': self.kk.detach().cpu().view(-1).repeat(self.output_dim),
            'kappa': kappa,
        }

    def active_param_count(self) -> int:
        return sum(int(p.numel()) for p in self.parameters() if p.requires_grad)
