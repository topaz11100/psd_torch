"""First-spike timing readout/loss with a compile-friendly default path.

The adapter still loads the released modules for compatibility/reference
bookkeeping, but the runtime path avoids custom autograd graph breaks.  It keeps
released first-time semantics in forward and reproduces the released
Spike2Time/Time2FST surrogate backward by using a straight-through tensor
surrogate instead of Python indexing inside ``autograd.Function.backward``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model._origin_first_spike import load_first_spike_modules
from src.neurons._compile import compile_callable


_DEFAULT_LOSS_PARAMS = {
    'loss_mode': 'first_time',
    'FS': {'D': 16, 'A': 200, 'alpha': 0.2},
    'FR': {'alpha': 1.0},
    'treg': {'lambda': 0.01, 'beta': 0.02},
}


@dataclass
class FirstSpikeAnalysis:
    """Structured result of one first-spike output-record analysis.

    ``firing_rate`` keeps the second tensor expected by the released ``LossFn``
    interface. ``scores`` is exposed as a computed convenience view that matches
    the readout specification.
    """

    first_times: torch.Tensor
    firing_rate: torch.Tensor

    @property
    def scores(self) -> torch.Tensor:
        """Return unscaled first-time logits; earlier first spikes are better."""

        return -self.first_times + self.first_times.min(dim=1, keepdim=True).values


def _gaussian_kernel_1d(*, sequence_length: int, d_param: int, num_classes: int) -> torch.Tensor:
    """Return the released Gaussian assignment kernel as ``(C,1,W)`` tensor.

    The released code uses ``sigma = T // D`` and an odd window length.  Very
    small smoke-test sequences can make ``sigma == 0``; in that degenerate case a
    length-1 identity kernel preserves a finite, well-defined surrogate path.
    """

    time_steps = int(sequence_length)
    d_param = max(1, int(d_param))
    sigma = int(time_steps // d_param)
    if sigma <= 0:
        base = torch.ones(1, dtype=torch.float32)
    else:
        length = min(int(sigma * 6 + 1), int(2 * (time_steps // 2) + 1))
        length = max(1, int(length))
        if length % 2 == 0:
            length += 1
        x = torch.arange(length, dtype=torch.float32) - 0.5 * float(length - 1)
        base = torch.exp(-0.5 * (x / float(sigma)).square())
    return base.reshape(1, 1, -1).repeat(int(num_classes), 1, 1).contiguous()


class FirstSpikeLossAdapter(nn.Module):
    """Project-facing wrapper for the released First-spike coding path.

    The default runtime path is tensor-only and compatible with ``torch.compile``.
    Set ``PSD_FIRST_SPIKE_ORIGIN_RUNTIME=1`` to force the exact released Python
    module path for debugging/reference checks.
    """

    def __init__(self, *, num_classes: int, sequence_length: int, device: torch.device | str, loss_params: dict[str, Any] | None = None) -> None:
        super().__init__()
        loss_params = dict(_DEFAULT_LOSS_PARAMS if loss_params is None else loss_params)
        time_mod, loss_mod = load_first_spike_modules()
        self.device_name = str(device)
        self.num_classes = int(num_classes)
        self.sequence_length = int(sequence_length)
        self.d_param = int(loss_params['FS']['D'])
        self.a_param = float(loss_params['FS']['A'])
        self.alpha_fs = float(loss_params['FS']['alpha'])
        self.lambda_treg_train = float(loss_params['treg']['lambda'])
        self.beta_treg_train = float(loss_params['treg']['beta'])
        self.spike_to_time = time_mod.Spike2Time(loss_params, device=self.device_name)
        self.train_loss = loss_mod.LossFn(loss_params, self.num_classes, step=self.sequence_length, mode='train')
        self.eval_loss = loss_mod.LossFn(loss_params, self.num_classes, step=self.sequence_length, mode='test')
        self.register_buffer(
            'time_error_kernel',
            _gaussian_kernel_1d(sequence_length=self.sequence_length, d_param=self.d_param, num_classes=self.num_classes),
            persistent=False,
        )
        self._compiled_analyze = None
        self._compiled_train_loss = None
        self._compiled_eval_loss = None
        self._compiled_runtime_disabled = False
        self._compiled_runtime_error = None
        self._compile_policy = 'eager_tensor_first_spike'

    def enable_compiled_forward(self, **compile_kwargs: Any) -> tuple[bool, str]:
        """Compile tensor-only first-spike analyze/loss kernels when requested."""

        kwargs = dict(compile_kwargs or {})
        analyze, analyze_applied, analyze_policy = compile_callable(
            self._analyze_tensors,
            compile_kwargs=kwargs,
            label='first_spike_readout_analyze',
        )
        train_loss, train_applied, train_policy = compile_callable(
            self._train_loss_tensor,
            compile_kwargs=kwargs,
            label='first_spike_readout_train_loss',
        )
        eval_loss, eval_applied, eval_policy = compile_callable(
            self._eval_loss_tensor,
            compile_kwargs=kwargs,
            label='first_spike_readout_eval_loss',
        )
        if analyze_applied:
            self._compiled_analyze = analyze
        if train_applied:
            self._compiled_train_loss = train_loss
        if eval_applied:
            self._compiled_eval_loss = eval_loss
        applied = bool(analyze_applied and train_applied and eval_applied)
        self._compiled_runtime_disabled = False
        self._compiled_runtime_error = None
        self._compile_policy = 'first_spike_compile[' + ';'.join([analyze_policy, train_policy, eval_policy]) + ']'
        return applied, self._compile_policy

    def _origin_runtime_requested(self) -> bool:
        import os

        return str(os.environ.get('PSD_FIRST_SPIKE_ORIGIN_RUNTIME', '')).strip().lower() in {'1', 'true', 'yes', 'on'}

    def _origin_analyze(self, output_membrane: torch.Tensor, output_spike: torch.Tensor) -> FirstSpikeAnalysis:
        spikes_bct = output_spike.transpose(1, 2).contiguous()
        membrane_bct = output_membrane.transpose(1, 2).contiguous()
        origin_potentials = torch.zeros_like(membrane_bct)
        origin_potentials[..., 1:] = membrane_bct[..., :-1]
        first_times = self.spike_to_time(spikes_bct, origin_potentials)
        firing_rate = spikes_bct.mean(dim=-1)
        return FirstSpikeAnalysis(first_times=first_times, firing_rate=firing_rate)

    def _output_times(self, spikes_bct: torch.Tensor, potentials_bct: torch.Tensor) -> torch.Tensor:
        if spikes_bct.ndim != 3 or potentials_bct.ndim != 3:
            raise ValueError('first_spike output tensors must have shape (B,C,T) after transpose.')
        batch_size, num_classes, time_steps = [int(v) for v in spikes_bct.shape]
        if num_classes != self.num_classes:
            raise ValueError(f'first_spike expected {self.num_classes} classes, got {num_classes}.')
        device = spikes_bct.device
        dtype = spikes_bct.dtype
        time_axis = torch.arange(1, time_steps + 1, device=device, dtype=dtype).view(1, 1, time_steps)
        max_u = potentials_bct.max(dim=-1).values
        indices_neuron = torch.sort(max_u, dim=-1, descending=True).indices
        neuron_rank = torch.arange(1, num_classes + 1, device=device, dtype=dtype).view(1, num_classes).expand(batch_size, num_classes)
        x_neuron = torch.zeros_like(neuron_rank).scatter(1, indices_neuron, neuron_rank)
        indices_time = torch.sort(potentials_bct, dim=-1, descending=True).indices
        time_rank = (torch.arange(time_steps, device=device, dtype=dtype) * 0.01).view(1, 1, time_steps).expand(batch_size, num_classes, time_steps)
        base = x_neuron.unsqueeze(-1).expand(batch_size, num_classes, time_steps)
        x_time = base.scatter_add(2, indices_time, time_rank)
        return time_axis * spikes_bct + (float(time_steps) + x_time) * (1.0 - spikes_bct)

    def _compile_friendly_first_times(self, spikes_bct: torch.Tensor, potentials_bct: torch.Tensor) -> torch.Tensor:
        output_times = self._output_times(spikes_bct, potentials_bct)
        first_times_forward, indices = output_times.min(dim=-1)
        batch_size, num_classes, time_steps = [int(v) for v in output_times.shape]
        selector = F.one_hot(indices.to(dtype=torch.long), num_classes=time_steps).to(dtype=spikes_bct.dtype)
        dead_mask = (output_times >= float(time_steps + 1)).all(dim=-1, keepdim=True)
        time_basis = torch.where(dead_mask, torch.ones_like(selector), selector).detach()
        kernel = self.time_error_kernel.to(device=spikes_bct.device, dtype=spikes_bct.dtype)
        padding = int(kernel.shape[-1] // 2)
        coeff = F.conv1d(-float(self.a_param) * time_basis, kernel, padding=padding, groups=num_classes).detach()
        surrogate = (spikes_bct * coeff).sum(dim=-1)
        return first_times_forward.detach() + surrogate - surrogate.detach()

    def _analyze_tensors(self, output_membrane: torch.Tensor, output_spike: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if output_membrane.ndim != 3 or output_spike.ndim != 3:
            raise ValueError('first_spike analysis expects output_membrane and output_spike with shape (B,T,C).')
        spikes_bct = output_spike.transpose(1, 2).contiguous()
        membrane_bct = output_membrane.transpose(1, 2).contiguous()
        origin_potentials = torch.zeros_like(membrane_bct)
        origin_potentials[..., 1:] = membrane_bct[..., :-1]
        first_times = self._compile_friendly_first_times(spikes_bct, origin_potentials)
        firing_rate = spikes_bct.mean(dim=-1)
        return first_times, firing_rate

    def _loss_tensor(self, first_times: torch.Tensor, target: torch.Tensor, *, training: bool) -> torch.Tensor:
        target = target.to(dtype=torch.long)
        scores = float(self.alpha_fs) * (-first_times + first_times.min(dim=1, keepdim=True).values)
        loss = F.cross_entropy(scores.float(), target)
        if bool(training) and float(self.lambda_treg_train) != 0.0:
            labeled_time = torch.gather(first_times, dim=1, index=target.view(-1, 1))
            late = torch.where(labeled_time <= float(self.sequence_length), torch.zeros_like(labeled_time), labeled_time)
            regularization = float(self.lambda_treg_train) * (torch.exp(float(self.beta_treg_train) * late) - 1.0).mean()
            loss = loss + regularization
        return loss

    def _train_loss_tensor(self, first_times: torch.Tensor, firing_rate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        del firing_rate
        return self._loss_tensor(first_times, target, training=True)

    def _eval_loss_tensor(self, first_times: torch.Tensor, firing_rate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        del firing_rate
        return self._loss_tensor(first_times, target, training=False)

    def analyze(self, output_membrane: torch.Tensor, output_spike: torch.Tensor) -> FirstSpikeAnalysis:
        """Convert output records into compile-friendly released-style first times."""

        if self._origin_runtime_requested():
            return self._origin_analyze(output_membrane, output_spike)
        if self._compiled_analyze is not None and not bool(self._compiled_runtime_disabled):
            try:
                first_times, firing_rate = self._compiled_analyze(output_membrane, output_spike)
                return FirstSpikeAnalysis(first_times=first_times, firing_rate=firing_rate)
            except Exception as exc:  # pragma: no cover - backend dependent
                self._compiled_runtime_disabled = True
                self._compiled_runtime_error = f'{type(exc).__name__}: {exc}'
                warnings.warn('[first_spike] compiled analyze fallback activated: ' + self._compiled_runtime_error, RuntimeWarning, stacklevel=2)
        first_times, firing_rate = self._analyze_tensors(output_membrane, output_spike)
        return FirstSpikeAnalysis(first_times=first_times, firing_rate=firing_rate)

    def predictions_from_analysis(self, analysis: FirstSpikeAnalysis) -> torch.Tensor:
        return analysis.first_times.argmin(dim=1)

    def one_hot(self, target: torch.Tensor) -> torch.Tensor:
        return F.one_hot(target.to(dtype=torch.long), num_classes=self.num_classes).to(dtype=torch.float32)

    def _origin_loss_from_analysis(self, analysis: FirstSpikeAnalysis, target: torch.Tensor, *, training: bool) -> torch.Tensor:
        target_one_hot = self.one_hot(target).to(device=analysis.first_times.device)
        loss_module = self.train_loss if training else self.eval_loss
        return loss_module((analysis.first_times, analysis.firing_rate), target_one_hot, loss_mode='first_time')

    def loss_from_analysis(self, analysis: FirstSpikeAnalysis, target: torch.Tensor, *, training: bool) -> torch.Tensor:
        if self._origin_runtime_requested():
            return self._origin_loss_from_analysis(analysis, target, training=training)
        compiled = self._compiled_train_loss if bool(training) else self._compiled_eval_loss
        if compiled is not None and not bool(self._compiled_runtime_disabled):
            try:
                return compiled(analysis.first_times, analysis.firing_rate, target)
            except Exception as exc:  # pragma: no cover - backend dependent
                self._compiled_runtime_disabled = True
                self._compiled_runtime_error = f'{type(exc).__name__}: {exc}'
                warnings.warn('[first_spike] compiled loss fallback activated: ' + self._compiled_runtime_error, RuntimeWarning, stacklevel=2)
        return self._train_loss_tensor(analysis.first_times, analysis.firing_rate, target) if bool(training) else self._eval_loss_tensor(analysis.first_times, analysis.firing_rate, target)


__all__ = ['FirstSpikeAnalysis', 'FirstSpikeLossAdapter']
