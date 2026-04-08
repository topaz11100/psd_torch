from __future__ import annotations

"""First-spike timing loss aligned with the released First-spike codebase.

The implementation follows the released code in
`Origin/First-spike coding promotes accurate and efficient spiking neural
networks for discrete events with rich temporal structures`, adapted to this
project's `(B, T, C)` output-record convention.
"""

from dataclasses import dataclass
from typing import Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.readout.readout import earliest_spike_exact_logits


@dataclass
class FirstSpikeAnalysis:
    output_times: torch.Tensor
    firing_rate: torch.Tensor
    decision_scores: torch.Tensor
    prediction_scores: torch.Tensor
    pred: torch.Tensor


class _Time2FSTFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output_times: torch.Tensor, t_inf: int):
        first_times, indices = torch.min(output_times, dim=-1)
        ctx.t_inf = float(t_inf)
        ctx.save_for_backward(output_times, indices)
        return first_times

    @staticmethod
    def backward(ctx, propagated_time_error: torch.Tensor):
        output_times, indices = ctx.saved_tensors
        dead_mask = (output_times >= float(ctx.t_inf)).all(dim=-1, keepdim=False)
        time_gradient = torch.zeros_like(output_times)
        all_time_gradient = propagated_time_error.unsqueeze(-1).repeat(1, 1, output_times.size(-1))
        time_gradient = time_gradient.scatter_(2, indices.unsqueeze(-1), propagated_time_error.unsqueeze(-1))
        time_gradient[dead_mask] = all_time_gradient[dead_mask]
        return time_gradient, None


class _Spike2TimeFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output_spikes: torch.Tensor, output_potentials: torch.Tensor, D: int = 16, A: float = 200.0):
        # Origin convention: tensors are (B, N, T)
        if output_spikes.dim() != 3:
            raise ValueError(f'output_spikes must be (B,N,T), got {tuple(output_spikes.shape)}')
        if output_potentials.shape != output_spikes.shape:
            raise ValueError(
                f'output_potentials must match output_spikes shape, got {tuple(output_potentials.shape)} '
                f'vs {tuple(output_spikes.shape)}'
            )
        B, N, T = output_spikes.shape
        device = output_spikes.device
        dtype = output_spikes.dtype
        time_steps = torch.arange(1, T + 1, device=device, dtype=dtype).view(1, 1, T)

        # Dead-neuron ranking follows the released code: neurons with larger peak
        # membrane potential get smaller post-horizon times, and within each neuron
        # larger membrane states at later steps receive smaller secondary offsets.
        max_u, _ = output_potentials.max(dim=-1)
        _, indices_neuron = torch.sort(max_u, dim=-1, descending=True)

        M_neuron = torch.arange(1.0, float(N) + 1.0, device=device, dtype=dtype).view(1, N).repeat(B, 1)
        X_neuron = torch.zeros_like(M_neuron).scatter_(1, indices_neuron, M_neuron)
        X_neuron = X_neuron.unsqueeze(-1).repeat(1, 1, T)

        _, indices_time = torch.sort(output_potentials, dim=-1, descending=True)
        M_time = (torch.arange(T, device=device, dtype=dtype) * 0.01).view(1, 1, T).repeat(B, N, 1)
        X_time = X_neuron.scatter_add_(2, indices_time, M_time)

        output_times = time_steps * output_spikes + (float(T) + X_time) * (1.0 - output_spikes)

        sigma = max(int(T) // max(int(D), 1), 1)
        length = min(int(sigma * 6 + 1), int(2 * (T // 2) + 1))
        if length < 1:
            length = 1
        if length % 2 == 0:
            length += 1
        coords = torch.arange(length, device=device, dtype=dtype) - float(length - 1) / 2.0
        kernel_1d = torch.exp(-0.5 * (coords / float(max(sigma, 1))) ** 2)
        kernel = kernel_1d.view(1, 1, length).repeat(N, 1, 1)

        ctx.kernel = kernel
        ctx.kernel_padding = length // 2
        ctx.kernel_groups = int(N)
        ctx.A = float(A)
        return output_times

    @staticmethod
    def backward(ctx, propagated_time_error: torch.Tensor):
        grad = -float(ctx.A) * propagated_time_error
        spike_gradient = F.conv1d(
            grad,
            ctx.kernel,
            padding=int(ctx.kernel_padding),
            groups=int(ctx.kernel_groups),
        )
        return spike_gradient, None, None, None


class Spike2FirstTime(nn.Module):
    def __init__(self, *, D: int = 16, A: float = 200.0):
        super().__init__()
        self.D = int(D)
        self.A = float(A)

    def forward(self, output_spikes_btc: torch.Tensor, output_potentials_btc: torch.Tensor) -> torch.Tensor:
        if output_spikes_btc.dim() != 3:
            raise ValueError(f'output_spikes_btc must be (B,T,C), got {tuple(output_spikes_btc.shape)}')
        if output_potentials_btc.shape != output_spikes_btc.shape:
            raise ValueError(
                f'output_potentials_btc must match output_spikes_btc shape, got '
                f'{tuple(output_potentials_btc.shape)} vs {tuple(output_spikes_btc.shape)}'
            )
        output_spikes = output_spikes_btc.transpose(1, 2).contiguous().to(torch.float32)
        output_potentials = output_potentials_btc.transpose(1, 2).contiguous().to(torch.float32)
        output_times = _Spike2TimeFunc.apply(output_spikes, output_potentials, int(self.D), float(self.A))
        return _Time2FSTFunc.apply(output_times, int(output_spikes.shape[-1]) + 1)


class FirstSpikeLoss(nn.Module):
    """First-spike timing loss for output-layer spike/membrane records.

    The loss follows the released First-spike code:
      - convert output spikes + membrane trajectories to first-spike times,
      - optimize a softmax over negative first-spike times,
      - add the dead-neuron regularizer during training.
    """

    requires_output_record = True

    def __init__(
        self,
        *,
        num_classes: int,
        step: int,
        alpha_fs: float = 0.2,
        D: int = 16,
        A: float = 200.0,
        lambda_treg: float = 0.01,
        beta_treg: float = 0.02,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.step = int(step)
        self.alpha_fs = float(alpha_fs)
        self.lambda_treg = float(lambda_treg)
        self.beta_treg = float(beta_treg)
        self.time_encoder = Spike2FirstTime(D=int(D), A=float(A))

    @staticmethod
    def _extract_record_tensor(out_rec: Mapping[str, torch.Tensor], primary: str, *fallbacks: str) -> torch.Tensor:
        for key in (str(primary),) + tuple(str(v) for v in fallbacks):
            value = out_rec.get(key)
            if torch.is_tensor(value):
                return value
        raise KeyError(f'missing output record key {primary!r} (fallbacks={fallbacks!r})')

    def analyze_output_record(self, out_rec: Mapping[str, torch.Tensor]) -> FirstSpikeAnalysis:
        spike_seq = self._extract_record_tensor(out_rec, 'output', 'spike', 'spk').to(torch.float32)
        soma_seq = self._extract_record_tensor(out_rec, 'soma_state', 'membrane').to(torch.float32)
        if spike_seq.dim() != 3:
            raise ValueError(f'output spike record must be (B,T,C), got {tuple(spike_seq.shape)}')
        if soma_seq.shape != spike_seq.shape:
            raise ValueError(
                f'output membrane record must match spike record shape, got {tuple(soma_seq.shape)} '
                f'vs {tuple(spike_seq.shape)}'
            )
        output_times = self.time_encoder(spike_seq, soma_seq)
        firing_rate = spike_seq.mean(dim=1)
        decision_scores = -output_times + output_times.min(dim=1, keepdim=True).values
        # Accuracy / probe-accuracy must follow the exact same earliest-spike
        # decision rule as the public readout path, including same-time membrane
        # tie-breaking and the silent-sample last-membrane fallback. The released
        # timing loss still uses ``decision_scores`` above; only the prediction
        # rule is aligned here.
        prediction_scores = earliest_spike_exact_logits(spike_seq, soma_seq)
        pred = prediction_scores.argmax(dim=1)
        return FirstSpikeAnalysis(
            output_times=output_times,
            firing_rate=firing_rate,
            decision_scores=decision_scores,
            prediction_scores=prediction_scores,
            pred=pred,
        )

    def loss_from_analysis(self, analysis: FirstSpikeAnalysis, target: torch.Tensor) -> torch.Tensor:
        y = target.to(torch.long).view(-1)
        target_oh = F.one_hot(y, num_classes=int(self.num_classes)).to(torch.float32)
        logits = float(self.alpha_fs) * analysis.decision_scores
        cls_loss = -(target_oh * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()

        if not bool(self.training):
            return cls_loss

        labeled_time = analysis.output_times.gather(1, y.view(-1, 1)).clone()
        labeled_time[labeled_time <= float(self.step)] = 0.0
        regularisation = float(self.lambda_treg) * (torch.exp(float(self.beta_treg) * labeled_time) - 1.0).mean()
        return cls_loss + regularisation

    def prediction_scores_from_analysis(self, analysis: FirstSpikeAnalysis) -> torch.Tensor:
        return analysis.prediction_scores

    def predictions_from_analysis(self, analysis: FirstSpikeAnalysis) -> torch.Tensor:
        return analysis.pred

    def forward_from_output_record(self, out_rec: Mapping[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        analysis = self.analyze_output_record(out_rec)
        return self.loss_from_analysis(analysis, target)

    def forward(self, outputs, target: torch.Tensor):
        if isinstance(outputs, Mapping):
            return self.forward_from_output_record(outputs, target)
        raise TypeError('FirstSpikeLoss expects an output-record mapping. Use forward_from_output_record(...).')
