"""Model-construction utilities for PSD analysis."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from src.model.arch_spec import (
    ConvLayerSpec,
    DenseLayerSpec,
    LayerSpec,
    ResidualBlockSpec,
    arch_hidden_sizes,
    arch_spec_payload,
    infer_output_sequence_length,
    resolve_arch_spec,
    serialize_arch_spec,
)
from src.model.constraints import ConstraintConfig, LayerConstraint, layer_constraint_for_hidden_index, resolve_constraint_plan
from src.model.model_registry import ModelSpec, canonicalize_model_token
from src.neurons.DH_SNN_neuron import DHSNNLayer
from src.neurons.D_RF_neuron import DRFLayer
from src.neurons.IF_neuron import IFLayer
from src.neurons.LIF_neuron import LIFLayer
from src.neurons.RF_neuron import RFLayer
from src.neurons.TC_LIF_neuron import TCLIFLayer
from src.neurons.TS_LIF_neuron import TSLIFLayer
from src.neurons._common import surrogate_spike
from src.neurons.cnn2d import CNN2DLIFLayer, CNN2DRFLayer


@dataclass
class LayerRecord:
    """Recorded sequence outputs for one layer."""

    layer_name: str
    membrane: torch.Tensor
    spike: torch.Tensor
    layer_input: torch.Tensor | None = None


@dataclass
class ForwardResult:
    """Full forward-pass product used by training and PSD analysis."""

    hidden_records: list[LayerRecord]
    output_record: LayerRecord
    input_record: torch.Tensor | None = None


@dataclass
class LayerMeta:
    """Static metadata for one built layer."""

    name: str
    size: int
    is_output: bool



def _time_distributed_2d(module: nn.Module, input_sequence: torch.Tensor) -> torch.Tensor:
    """Apply one 2-D module to each frame of a ``(B,T,C,H,W)`` tensor."""

    if input_sequence.ndim != 5:
        raise ValueError(f'Expected shape (B,T,C,H,W), got {tuple(input_sequence.shape)}')
    batch_size, time_steps, channels, height, width = [int(v) for v in input_sequence.shape]
    flattened = input_sequence.reshape(batch_size * time_steps, channels, height, width)
    output = module(flattened)
    out_channels, out_height, out_width = [int(v) for v in output.shape[1:]]
    return output.reshape(batch_size, time_steps, out_channels, out_height, out_width).contiguous()


def _squeeze_unit_spatial(record_tensor: torch.Tensor) -> torch.Tensor:
    """Convert ``(B,T,C,1,1)`` output tensors to ``(B,T,C)`` for readout code."""

    if record_tensor.ndim != 5 or int(record_tensor.shape[-1]) != 1 or int(record_tensor.shape[-2]) != 1:
        raise ValueError(f'Expected output tensor shape (B,T,C,1,1), got {tuple(record_tensor.shape)}')
    return record_tensor[..., 0, 0].contiguous()


def _output_layer_model_spec(spec: ModelSpec) -> ModelSpec:
    """Return the official output-layer spec with hidden recurrence disabled."""

    if spec.family in {'lif', 'rf', 'tc_lif', 'ts_lif', 'dh_snn'} and bool(spec.recurrent):
        return replace(spec, recurrent=False)
    return spec


class TimeDistributedConv2DBN(nn.Module):
    """Time-distributed Conv2d plus optional BatchNorm2d without a spiking activation."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        kernel_size: int,
        stride: int,
        padding: int,
        batch_norm: bool,
        bias: bool,
    ) -> None:
        """Initialize one analog 2-D convolution path used inside ResNet BasicBlock."""
        super().__init__()
        self.conv = nn.Conv2d(
            int(input_size),
            int(output_size),
            kernel_size=int(kernel_size),
            stride=int(stride),
            padding=int(padding),
            bias=bool(bias),
        )
        self.bn = nn.BatchNorm2d(int(output_size)) if bool(batch_norm) else None

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """Return the analog current sequence after conv/BN."""
        output = _time_distributed_2d(self.conv, input_sequence)
        if self.bn is not None:
            output = _time_distributed_2d(self.bn, output)
        return output


def _make_identity_activation_layer(layer: nn.Module) -> nn.Module:
    """Freeze a 1x1 CNN spiking layer into an identity-current activation."""

    conv = getattr(layer, 'conv', None)
    if conv is None or not isinstance(conv, nn.Conv2d):
        raise TypeError('Identity activation helper expects a CNN2D layer with a Conv2d member.')
    if int(conv.in_channels) != int(conv.out_channels) or tuple(conv.kernel_size) != (1, 1):
        raise ValueError('Identity activation requires a square 1x1 Conv2d.')
    with torch.no_grad():
        conv.weight.zero_()
        for index in range(int(conv.out_channels)):
            conv.weight[index, index, 0, 0] = 1.0
        if conv.bias is not None:
            conv.bias.zero_()
    conv.weight.requires_grad_(False)
    if conv.bias is not None:
        conv.bias.requires_grad_(False)
    return layer


class CNN2DResidualBlock(nn.Module):
    """Canonical ResNet-18 BasicBlock with spiking neurons replacing ReLU sites."""

    def __init__(
        self,
        spec: ModelSpec,
        *,
        input_size: int,
        block_spec: ResidualBlockSpec,
        v_th: float,
    ) -> None:
        """Initialize one ResNet-18 BasicBlock."""
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(block_spec.out_channels)
        self.kernel_size = int(block_spec.kernel_size)
        self.stride = int(block_spec.stride)
        self.padding = int(block_spec.padding)
        self.batch_norm = bool(block_spec.batch_norm)
        first_conv = ConvLayerSpec(
            out_channels=self.output_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            batch_norm=self.batch_norm,
            bias=False,
        )
        self.layer1 = _build_conv2d_family_layer(
            spec,
            input_size=self.input_size,
            layer_spec=first_conv,
            v_th=v_th,
            output_overrides={},
        )
        self.conv2 = TimeDistributedConv2DBN(
            self.output_size,
            self.output_size,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            batch_norm=self.batch_norm,
            bias=False,
        )
        activation_spec = ConvLayerSpec(out_channels=self.output_size, kernel_size=1, stride=1, padding=0, batch_norm=False, bias=False)
        self.residual_activation = _make_identity_activation_layer(
            _build_conv2d_family_layer(
                spec,
                input_size=self.output_size,
                layer_spec=activation_spec,
                v_th=v_th,
                output_overrides={},
            )
        )
        self.skip_projection = None
        self.skip_bn = None
        if self.input_size != self.output_size or self.stride != 1:
            self.skip_projection = nn.Conv2d(self.input_size, self.output_size, kernel_size=1, stride=self.stride, padding=0, bias=False)
            self.skip_bn = nn.BatchNorm2d(self.output_size) if self.batch_norm else None
        self._last_layer_input = None
        self._last_trace_records: list[LayerRecord] = []

    def _skip_path(self, input_sequence: torch.Tensor) -> torch.Tensor:
        if self.skip_projection is None:
            return input_sequence
        output = _time_distributed_2d(self.skip_projection, input_sequence)
        if self.skip_bn is not None:
            output = _time_distributed_2d(self.skip_bn, output)
        return output

    def forward(self, input_sequence: torch.Tensor, *, return_traces: bool = False) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Run one residual BasicBlock forward pass.

        The second spiking site receives the complete residual sum
        ``conv2(spike1) + shortcut(input)``.  Therefore the recorded
        ``residual_add`` layer_input/membrane/spike all correspond to the
        post-shortcut residual-add signal, not to the branch-only conv2 path.
        """
        self._last_layer_input = None
        self._last_trace_records = []
        mem1, spike1 = self.layer1(input_sequence, return_traces=return_traces)
        branch_current = self.conv2(spike1)
        shortcut_current = self._skip_path(input_sequence)
        residual_current = branch_current + shortcut_current
        residual_membrane, residual_spike = self.residual_activation(residual_current, return_traces=return_traces)
        if return_traces:
            layer1_input = getattr(self.layer1, '_last_layer_input', None)
            residual_input = getattr(self.residual_activation, '_last_layer_input', None)
            if mem1 is None or layer1_input is None:
                raise RuntimeError('ResNet BasicBlock first spiking neuron did not expose complete traces.')
            if residual_membrane is None or residual_input is None:
                raise RuntimeError('ResNet BasicBlock residual spiking neuron did not expose complete traces.')
            self._last_layer_input = residual_current
            self._last_trace_records = [
                LayerRecord(layer_name='conv1', membrane=mem1, spike=spike1, layer_input=layer1_input),
                LayerRecord(layer_name='residual_add', membrane=residual_membrane, spike=residual_spike, layer_input=residual_current),
            ]
        return residual_membrane, residual_spike

    def trace_records(self, base_name: str) -> list[LayerRecord]:
        """Return PSD records for both spiking-neuron sites in this BasicBlock."""

        records: list[LayerRecord] = []
        for record in self._last_trace_records:
            records.append(
                LayerRecord(
                    layer_name=f'{base_name}_{record.layer_name}',
                    membrane=record.membrane,
                    spike=record.spike,
                    layer_input=record.layer_input,
                )
            )
        return records

    def filter_stats_vectors(self) -> dict[str, torch.Tensor]:
        """Return concatenated filter-stat vectors from internal neuron layers."""
        merged: dict[str, list[torch.Tensor]] = {}
        for layer in (self.layer1, self.residual_activation):
            if not hasattr(layer, 'filter_stats_vectors'):
                continue
            for key, value in layer.filter_stats_vectors().items():
                merged.setdefault(str(key), []).append(value.detach().reshape(-1))
        return {key: torch.cat(values, dim=0) for key, values in merged.items() if values}


class FixedCNN2DClassifier(nn.Module):
    """Fixed VGG11/ResNet18 CNN-SNN classifier with project readout head."""

    def __init__(
        self,
        *,
        spec: ModelSpec,
        input_dim: int,
        sequence_length: int,
        input_shape: Sequence[int],
        output_sequence_length: int,
        num_classes: int,
        hidden_layers: list[nn.Module],
        output_layer: nn.Module,
        layer_meta: list[LayerMeta],
        pool_after: list[nn.Module | None],
        extra_metadata: dict[str, Any],
    ) -> None:
        super().__init__()
        self.spec = spec
        self.input_dim = int(input_dim)
        self.sequence_length = int(sequence_length)
        self.input_shape = tuple(int(v) for v in input_shape)
        self.output_sequence_length = int(output_sequence_length)
        self.num_classes = int(num_classes)
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.output_layer = output_layer
        self.layer_meta = list(layer_meta)
        self.pool_after = nn.ModuleList([nn.Identity() if pool is None else pool for pool in pool_after])
        self.pool_after_enabled = [pool is not None for pool in pool_after]
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.extra_metadata = dict(extra_metadata)

    def iter_named_layers(self) -> Iterable[tuple[str, nn.Module]]:
        for meta, layer in zip(self.layer_meta[:-1], self.hidden_layers):
            if isinstance(layer, CNN2DResidualBlock):
                yield f'{meta.name}_conv1', layer.layer1
                yield f'{meta.name}_residual_add', layer.residual_activation
            else:
                yield meta.name, layer
        yield self.layer_meta[-1].name, self.output_layer

    def iter_named_hidden_layers(self) -> Iterable[tuple[str, nn.Module]]:
        for meta, layer in zip(self.layer_meta[:-1], self.hidden_layers):
            if isinstance(layer, CNN2DResidualBlock):
                yield f'{meta.name}_conv1', layer.layer1
                yield f'{meta.name}_residual_add', layer.residual_activation
            else:
                yield meta.name, layer

    def model_metadata(self) -> dict[str, Any]:
        payload = {
            'raw_model_token': self.spec.raw_token,
            'canonical_model_token': self.spec.canonical_token,
            'family': self.spec.family,
            'recurrent': self.spec.recurrent,
            'branch': self.spec.branch,
            'input_dim': self.input_dim,
            'sequence_length': self.sequence_length,
            'output_sequence_length': self.output_sequence_length,
            'num_classes': self.num_classes,
            'cnn_input_shape': list(self.input_shape),
        }
        payload.update(self.extra_metadata)
        return payload

    def _expected_cnn_shape(self) -> tuple[int, int | None, int, int, int]:
        if len(self.input_shape) == 3:
            channels, height, width = [int(v) for v in self.input_shape]
            return channels, None, channels, height, width
        if len(self.input_shape) == 4:
            time_steps, channels, height, width = [int(v) for v in self.input_shape]
            return channels, time_steps, channels, height, width
        raise ValueError(f'CNN input_shape must be rank 3 or 4, got {self.input_shape!r}.')

    def _prepare_input(self, inputs: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(inputs)
        expected_channels, expected_time, channels, height, width = self._expected_cnn_shape()
        if tensor.ndim != 5:
            raise ValueError(
                'CNN models require prepared input shape (B,T,C,H,W). '
                f'Flattened or rank-{tensor.ndim} input is not accepted for CNN models; got shape {tuple(tensor.shape)}.'
            )
        if int(tensor.shape[2]) != expected_channels:
            raise ValueError(f'Expected frame channels={expected_channels}, got shape {tuple(tensor.shape)}.')
        if expected_time is not None and int(tensor.shape[1]) != expected_time:
            raise ValueError(f'Expected temporal frames={expected_time}, got shape {tuple(tensor.shape)}.')
        if int(tensor.shape[3]) != height or int(tensor.shape[4]) != width:
            raise ValueError(f'Expected spatial shape ({height},{width}), got {tuple(tensor.shape[-2:])}.')

        if self.spec.backbone == 'vgg11':
            pad_h = max(0, 32 - int(tensor.shape[-2]))
            pad_w = max(0, 32 - int(tensor.shape[-1]))
            if pad_h > 0 or pad_w > 0:
                tensor = F.pad(tensor, (0, pad_w, 0, pad_h))
        return tensor.contiguous()

    def forward(self, input_sequence: torch.Tensor, *, capture_hidden: bool = False) -> ForwardResult:
        hidden_records: list[LayerRecord] = []
        prepared_input = self._prepare_input(input_sequence)
        current = prepared_input
        for index, (meta, layer) in enumerate(zip(self.layer_meta[:-1], self.hidden_layers)):
            membrane, spike = layer(current, return_traces=capture_hidden)
            current = spike
            record_membrane = membrane
            record_layer_input = getattr(layer, '_last_layer_input', None) if capture_hidden else None
            if self.pool_after_enabled[index]:
                current = _time_distributed_2d(self.pool_after[index], current)
            if capture_hidden:
                trace_records = getattr(layer, 'trace_records', None)
                if callable(trace_records):
                    block_records = trace_records(meta.name)
                    if not block_records:
                        raise RuntimeError(f'Layer {meta.name} did not expose BasicBlock trace records under capture_hidden=True.')
                    hidden_records.extend(block_records)
                else:
                    if record_membrane is None:
                        raise RuntimeError(f'Layer {meta.name} did not return membrane traces under capture_hidden=True.')
                    if record_layer_input is None:
                        raise RuntimeError(f'Layer {meta.name} did not expose exact layer_input traces under capture_hidden=True.')
                    hidden_records.append(LayerRecord(layer_name=meta.name, membrane=record_membrane, spike=current, layer_input=record_layer_input))
        current = _time_distributed_2d(self.global_pool, current)
        output_membrane_5d, output_spike_5d = self.output_layer(current, return_traces=True)
        if output_membrane_5d is None:
            raise RuntimeError('Output layer must always return membrane traces.')
        output_layer_input_5d = getattr(self.output_layer, '_last_layer_input', None)
        if output_layer_input_5d is None:
            raise RuntimeError('Output layer did not expose exact layer_input traces under return_traces=True.')
        output_record = LayerRecord(
            layer_name=self.layer_meta[-1].name,
            membrane=_squeeze_unit_spatial(output_membrane_5d),
            spike=_squeeze_unit_spatial(output_spike_5d),
            layer_input=_squeeze_unit_spatial(output_layer_input_5d),
        )
        return ForwardResult(hidden_records=hidden_records, output_record=output_record, input_record=prepared_input)

class SNNClassifier(nn.Module):
    """Plain stacked SNN classifier with an output neuron layer and no head."""

    def __init__(
        self,
        *,
        spec: ModelSpec,
        input_dim: int,
        sequence_length: int,
        output_sequence_length: int,
        num_classes: int,
        hidden_layers: list[nn.Module],
        output_layer: nn.Module,
        layer_meta: list[LayerMeta],
        extra_metadata: dict[str, Any],
    ) -> None:
        super().__init__()
        self.spec = spec
        self.input_dim = int(input_dim)
        self.sequence_length = int(sequence_length)
        self.output_sequence_length = int(output_sequence_length)
        self.num_classes = int(num_classes)
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.output_layer = output_layer
        self.layer_meta = list(layer_meta)
        self.extra_metadata = dict(extra_metadata)

    def iter_named_layers(self) -> Iterable[tuple[str, nn.Module]]:
        for meta, layer in zip(self.layer_meta[:-1], self.hidden_layers):
            yield meta.name, layer
        yield self.layer_meta[-1].name, self.output_layer

    def iter_named_hidden_layers(self) -> Iterable[tuple[str, nn.Module]]:
        for meta, layer in zip(self.layer_meta[:-1], self.hidden_layers):
            yield meta.name, layer

    def model_metadata(self) -> dict[str, Any]:
        payload = {
            'raw_model_token': self.spec.raw_token,
            'canonical_model_token': self.spec.canonical_token,
            'family': self.spec.family,
            'recurrent': self.spec.recurrent,
            'branch': self.spec.branch,
            'input_dim': self.input_dim,
            'sequence_length': self.sequence_length,
            'output_sequence_length': self.output_sequence_length,
            'num_classes': self.num_classes,
        }
        payload.update(self.extra_metadata)
        return payload

    def forward(self, input_sequence: torch.Tensor, *, capture_hidden: bool = False) -> ForwardResult:
        if input_sequence.ndim != 3:
            raise ValueError(f'Expected shape (B,T,C), got {tuple(input_sequence.shape)}')
        hidden_records: list[LayerRecord] = []
        current = input_sequence
        for meta, layer in zip(self.layer_meta[:-1], self.hidden_layers):
            membrane, spike = layer(current, return_traces=capture_hidden)
            current = spike
            if capture_hidden:
                if membrane is None:
                    raise RuntimeError(f'Layer {meta.name} did not return membrane traces under capture_hidden=True.')
                layer_input = getattr(layer, '_last_layer_input', None)
                if layer_input is None:
                    raise RuntimeError(f'Layer {meta.name} did not expose exact layer_input traces under capture_hidden=True.')
                hidden_records.append(LayerRecord(layer_name=meta.name, membrane=membrane, spike=spike, layer_input=layer_input))
        output_membrane, output_spike = self.output_layer(current, return_traces=True)
        if output_membrane is None:
            raise RuntimeError('Output layer must always return membrane traces.')
        output_layer_input = getattr(self.output_layer, '_last_layer_input', None)
        if output_layer_input is None:
            raise RuntimeError('Output layer did not expose exact layer_input traces under return_traces=True.')
        output_record = LayerRecord(layer_name=self.layer_meta[-1].name, membrane=output_membrane, spike=output_spike, layer_input=output_layer_input)
        return ForwardResult(hidden_records=hidden_records, output_record=output_record, input_record=input_sequence)


class SpikGRUCellBlock(nn.Module):
    """One vanilla SpikGRU recurrent block with a single update gate."""

    def __init__(self, input_dim: int, hidden_size: int, *, v_th: float = 1.0) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.v_threshold = float(v_th)
        self.input_to_candidate = nn.Linear(self.input_dim, self.hidden_size)
        self.hidden_to_candidate = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.input_to_gate = nn.Linear(self.input_dim, self.hidden_size)
        self.hidden_to_gate = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.alpha = nn.Parameter(torch.full((self.hidden_size,), 0.9))

    def initial_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        current_state = torch.zeros_like(hidden)
        previous_spike = torch.zeros_like(hidden)
        return hidden, current_state, previous_spike

    def step(
        self,
        *,
        gate_input_t: torch.Tensor,
        candidate_input_t: torch.Tensor,
        hidden: torch.Tensor,
        current_state: torch.Tensor,
        previous_spike: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_t = torch.sigmoid(gate_input_t + self.hidden_to_gate(previous_spike))
        drive_t = candidate_input_t + self.hidden_to_candidate(previous_spike)

    def clamp_projected_parameters(self) -> None:
        with torch.no_grad():
            self.alpha.clamp_(0.0, 1.0)


class SpikGRUClassifier(nn.Module):
    """Vanilla two-layer 2x128 SpikeGRU classifier with PSD trace exposure."""

    def __init__(
        self,
        *,
        spec: ModelSpec,
        input_dim: int,
        sequence_length: int,
        num_classes: int,
        hidden_size: int = 128,
        v_th: float = 1.0,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.input_dim = int(input_dim)
        self.sequence_length = int(sequence_length)
        self.output_sequence_length = int(sequence_length)
        self.num_classes = int(num_classes)
        self.hidden_size = int(hidden_size)
        self.v_threshold = float(v_th)
        self.layer_01 = SpikGRUCellBlock(self.input_dim, self.hidden_size, v_th=v_th)
        self.layer_02 = SpikGRUCellBlock(self.hidden_size, self.hidden_size, v_th=v_th)
        self.readout = nn.Linear(self.hidden_size, self.num_classes)

    def iter_named_layers(self) -> Iterable[tuple[str, nn.Module]]:
        yield 'layer_01', self.layer_01
        yield 'layer_02', self.layer_02
        yield 'output', self.readout

    def iter_named_hidden_layers(self) -> Iterable[tuple[str, nn.Module]]:
        yield 'layer_01', self.layer_01
        yield 'layer_02', self.layer_02

    def model_metadata(self) -> dict[str, Any]:
        return {
            'raw_model_token': self.spec.raw_token,
            'canonical_model_token': self.spec.canonical_token,
            'family': self.spec.family,
            'analysis_role': 'auxiliary_diversity',
            'main_analysis_target': 'dense_snn',
            'model_profile': 'spikegru',
            'paper_title': 'Investigating Current-based and Gating Approaches for Accurate and Energy-efficient Spiking Recurrent Neural Networks',
            'paper_topology': '2x128',
            'recurrent_layers': 2,
            'gate_count': 1,
            'recurrent': True,
            'branch': None,
            'input_dim': self.input_dim,
            'sequence_length': self.sequence_length,
            'output_sequence_length': self.output_sequence_length,
            'num_classes': self.num_classes,
            'hidden_size': self.hidden_size,
            'hidden_spec': '2x128',
            'arch_spec': 'spikegru(layers=2,hidden=128)',
            'arch_layers': [{'kind': 'spikegru', 'hidden_size': self.hidden_size}, {'kind': 'spikegru', 'hidden_size': self.hidden_size}],
            'v_th': self.v_threshold,
            'alpha_init': 0.9,
            'alpha_clamp': [0.0, 1.0],
            'spike_reset_term': 'membrane_update_minus_v_th_times_previous_spike',
            'readout': 'non-spiking integrating readout membrane trace',
            'loss_reference': 'max-over-time cross entropy',
            'dvs_lip_frontend_included': False,
            'two_gate_backend_included': False,
            'bidirectional_backend_included': False,
            'signed_activation_ablation_included': False,
            'structure_variation': 'none',
            'trace_families': ['x_probe', 'x_layer', 'i_current', 'z_gate', 'y_mem', 'y_spike', 'readout_mem'],
        }

    def clamp_projected_parameters(self) -> None:
        self.layer_01.clamp_projected_parameters()
        self.layer_02.clamp_projected_parameters()

    def _run_block(self, block: SpikGRUCellBlock, block_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, time_steps, _input_dim = [int(v) for v in block_input.shape]
        hidden, current_state, previous_spike = block.initial_state(batch_size, device=block_input.device, dtype=block_input.dtype)
        mem_steps: list[torch.Tensor] = []
        spike_steps: list[torch.Tensor] = []
        current_steps: list[torch.Tensor] = []
        gate_steps: list[torch.Tensor] = []
        gate_input_sequence = block.input_to_gate(block_input)
        candidate_input_sequence = block.input_to_candidate(block_input)
        for t in range(time_steps):
            mem_t, i_t, spike_t, z_t, previous_spike = block.step(
                gate_input_t=gate_input_sequence[:, t, :],
                candidate_input_t=candidate_input_sequence[:, t, :],
                hidden=hidden,
                current_state=current_state,
                previous_spike=previous_spike,
            )
            hidden = mem_t
            current_state = i_t
            mem_steps.append(mem_t)
            spike_steps.append(spike_t)
            current_steps.append(i_t)
            gate_steps.append(z_t)
        return (
            torch.stack(mem_steps, dim=1).contiguous(),
            torch.stack(spike_steps, dim=1).contiguous(),
            torch.stack(current_steps, dim=1).contiguous(),
            torch.stack(gate_steps, dim=1).contiguous(),
        )

    def forward(self, input_sequence: torch.Tensor, *, capture_hidden: bool = False) -> ForwardResult:
        if input_sequence.ndim != 3:
            raise ValueError(f'SpikeGRU expects non-image input shape (B,T,C), got {tuple(input_sequence.shape)}')
        batch_size, time_steps, input_dim = [int(v) for v in input_sequence.shape]
        if input_dim != self.input_dim:
            raise ValueError(f'SpikeGRU expected input_dim={self.input_dim}, got {input_dim}.')
        layer1_mem, layer1_spike, layer1_current, layer1_gate = self._run_block(self.layer_01, input_sequence)
        layer2_mem, layer2_spike, layer2_current, layer2_gate = self._run_block(self.layer_02, layer1_spike)
        readout_drive_seq = self.readout(layer2_spike)
        readout_mem_seq = torch.cumsum(readout_drive_seq, dim=1).contiguous()
        readout_spike_placeholder = torch.zeros_like(readout_mem_seq)

        hidden_records: list[LayerRecord] = []
        if capture_hidden:
            layer1_record = LayerRecord(layer_name='layer_01', membrane=layer1_mem, spike=layer1_spike, layer_input=input_sequence)
            setattr(layer1_record, 'x_layer', input_sequence)
            setattr(layer1_record, 'i_current', layer1_current)
            setattr(layer1_record, 'z_gate', layer1_gate)
            setattr(layer1_record, 'y_mem', layer1_mem)
            setattr(layer1_record, 'y_spike', layer1_spike)
            layer2_record = LayerRecord(layer_name='layer_02', membrane=layer2_mem, spike=layer2_spike, layer_input=layer1_spike)
            setattr(layer2_record, 'x_layer', layer1_spike)
            setattr(layer2_record, 'i_current', layer2_current)
            setattr(layer2_record, 'z_gate', layer2_gate)
            setattr(layer2_record, 'y_mem', layer2_mem)
            setattr(layer2_record, 'y_spike', layer2_spike)
            hidden_records = [layer1_record, layer2_record]

        output_record = LayerRecord(layer_name='output', membrane=readout_mem_seq, spike=readout_spike_placeholder, layer_input=layer2_spike)
        setattr(output_record, 'readout_mem', readout_mem_seq)
        return ForwardResult(hidden_records=hidden_records, output_record=output_record, input_record=input_sequence)


def _resolved_model_reset_mode(spec: ModelSpec, *, family: str) -> str:
    mode = spec.reset_mode
    if mode is None:
        raise ValueError(f'Model family {family!r} requires an explicit reset mode.')
    if family in {'lif', 'if', 'cnn_lif'} and mode == 'no_reset':
        raise ValueError('LIF-family models do not support no_reset suffix.')
    return str(mode)


def _resolved_reset_enabled(spec: ModelSpec, *, output_overrides: dict[str, Any]) -> bool:
    enabled = bool(output_overrides.get('reset_enabled', True))
    if spec.reset_mode == 'no_reset':
        return False
    return enabled


def _build_dense_family_layer(
    spec: ModelSpec,
    *,
    input_size: int,
    output_size: int,
    v_th: float,
    output_overrides: dict[str, Any],
    layer_constraint: LayerConstraint | None = None,
) -> nn.Module:
    if spec.family == 'if':
        lc = layer_constraint if layer_constraint is not None else LayerConstraint()
        return IFLayer(
            input_size,
            output_size,
            recurrent=spec.recurrent,
            v_threshold=v_th,
            trainable_threshold=bool(spec.trainable_threshold),
            input_mask=lc.input_mask,
            recurrent_mask=lc.recurrent_mask,
            reset_mode=_resolved_model_reset_mode(spec, family='if'),
            emit_spike=output_overrides.get('emit_spike', True),
            reset_enabled=_resolved_reset_enabled(spec, output_overrides=output_overrides),
        )
    if spec.family == 'lif':
        lc = layer_constraint if layer_constraint is not None else LayerConstraint()
        return LIFLayer(
            input_size,
            output_size,
            recurrent=spec.recurrent,
            v_threshold=v_th,
            trainable_threshold=bool(spec.trainable_threshold),
            reset_mode=_resolved_model_reset_mode(spec, family='lif'),
            alpha_bounds=lc.lif_alpha_bounds,
            input_mask=lc.input_mask,
            recurrent_mask=lc.recurrent_mask,
            emit_spike=output_overrides.get('emit_spike', True),
            reset_enabled=_resolved_reset_enabled(spec, output_overrides=output_overrides),
        )
    if spec.family == 'rf':
        lc = layer_constraint if layer_constraint is not None else LayerConstraint()
        return RFLayer(
            input_size,
            output_size,
            recurrent=spec.recurrent,
            v_threshold=v_th,
            trainable_threshold=bool(spec.trainable_threshold),
            frequency_bounds=lc.rf_frequency_bounds,
            input_mask=lc.input_mask,
            recurrent_mask=lc.recurrent_mask,
            reset_mode=_resolved_model_reset_mode(spec, family='rf'),
            emit_spike=output_overrides.get('emit_spike', True),
            reset_enabled=_resolved_reset_enabled(spec, output_overrides=output_overrides),
        )
    if spec.family == 'tc_lif':
        return TCLIFLayer(input_size, output_size, recurrent=spec.recurrent, v_threshold=v_th, **output_overrides)
    if spec.family == 'ts_lif':
        return TSLIFLayer(input_size, output_size, recurrent=spec.recurrent, v_threshold=v_th, **output_overrides)
    if spec.family == 'dh_snn':
        return DHSNNLayer(input_size, output_size, recurrent=spec.recurrent, branch=int(spec.branch or 4), v_threshold=v_th, **output_overrides)
    if spec.family == 'd_rf':
        if spec.recurrent:
            raise ValueError('d_rf does not support recurrent suffix in the official spec.')
        return DRFLayer(
            input_size,
            output_size,
            branch=int(spec.branch or 4),
            v_threshold=v_th,
            emit_spike=output_overrides.get('emit_spike', True),
            reset_enabled=output_overrides.get('reset_enabled', True),
        )
    raise ValueError(f'Dense hidden layers are unsupported for model family {spec.family!r}.')


def _build_conv2d_family_layer(
    spec: ModelSpec,
    *,
    input_size: int,
    layer_spec: ConvLayerSpec,
    v_th: float,
    output_overrides: dict[str, Any],
    layer_constraint: LayerConstraint | None = None,
) -> nn.Module:
    """Build one 2-D convolutional spiking layer for canonical CNN backbones."""

    if spec.family == 'cnn_lif':
        cnn_lif_overrides = dict(output_overrides)
        cnn_lif_overrides['reset_enabled'] = _resolved_reset_enabled(spec, output_overrides=output_overrides)
        return CNN2DLIFLayer(
            input_size,
            int(layer_spec.out_channels),
            kernel_size=int(layer_spec.kernel_size),
            stride=int(layer_spec.stride),
            padding=int(layer_spec.padding),
            v_threshold=v_th,
            trainable_threshold=bool(spec.trainable_threshold),
            reset_mode=_resolved_model_reset_mode(spec, family='cnn_lif'),
            batch_norm=bool(layer_spec.batch_norm),
            bias=bool(layer_spec.bias),
            **cnn_lif_overrides,
        )
    if spec.family == 'cnn_rf':
        cnn_rf_overrides = dict(output_overrides)
        cnn_rf_overrides['reset_enabled'] = _resolved_reset_enabled(spec, output_overrides=output_overrides)
        return CNN2DRFLayer(
            input_size,
            int(layer_spec.out_channels),
            kernel_size=int(layer_spec.kernel_size),
            stride=int(layer_spec.stride),
            padding=int(layer_spec.padding),
            v_threshold=v_th,
            trainable_threshold=bool(spec.trainable_threshold),
            reset_mode=_resolved_model_reset_mode(spec, family='cnn_rf'),
            batch_norm=bool(layer_spec.batch_norm),
            bias=bool(layer_spec.bias),
            **cnn_rf_overrides,
        )
    raise ValueError(f'2-D conv hidden layers are unsupported for model family {spec.family!r}.')


def build_layer_from_spec(
    spec: ModelSpec,
    *,
    input_size: int,
    layer_spec: LayerSpec,
    v_th: float,
    output_overrides: dict[str, Any] | None = None,
    layer_constraint: LayerConstraint | None = None,
) -> nn.Module:
    output_overrides = {} if output_overrides is None else dict(output_overrides)
    if isinstance(layer_spec, DenseLayerSpec):
        return _build_dense_family_layer(
            spec,
            input_size=input_size,
            output_size=int(layer_spec.width),
            v_th=v_th,
            output_overrides=output_overrides,
            layer_constraint=layer_constraint,
        )
    if isinstance(layer_spec, ResidualBlockSpec):
        if output_overrides:
            raise ValueError('Residual hidden blocks do not support output-layer overrides.')
        if spec.family in {'cnn_lif', 'cnn_rf'}:
            return CNN2DResidualBlock(spec, input_size=input_size, block_spec=layer_spec, v_th=v_th)
        raise ValueError(f'Residual blocks are unsupported for model family {spec.family!r}.')
    if spec.family in {'cnn_lif', 'cnn_rf'}:
        return _build_conv2d_family_layer(
            spec,
            input_size=input_size,
            layer_spec=layer_spec,
            v_th=v_th,
            output_overrides=output_overrides,
            layer_constraint=layer_constraint,
        )
    raise ValueError(f'Convolutional layers are unsupported for model family {spec.family!r}.')



def _resolve_cnn_input_shape(
    *,
    input_shape: Sequence[int] | None,
    input_dim: int,
    sequence_length: int,
) -> tuple[int, ...]:
    """Resolve rank-3 static or rank-4 frame CNN input shape."""

    if input_shape is not None:
        shape = tuple(int(v) for v in input_shape)
        if len(shape) not in {3, 4}:
            raise ValueError(f'CNN input_shape must be [C,H,W] or [T,C,H,W], got {shape}.')
        if any(v <= 0 for v in shape):
            raise ValueError(f'CNN input_shape entries must be positive, got {shape}.')
        return shape
    input_dim = int(input_dim)
    sequence_length = int(sequence_length)
    if input_dim in {1, 3}:
        side = int(math.isqrt(sequence_length))
        if side * side == sequence_length:
            return (input_dim, side, side)
    raise ValueError(
        'Fixed VGG11/ResNet18 CNN backbones require original_shape metadata. '
        'Pass input_shape=[C,H,W] for static images or [T,C,H,W] for frame sequences.'
    )


def _cnn_temporal_output_length(input_shape: Sequence[int]) -> int:
    """Return the explicit temporal length after a 2-D fixed CNN backbone."""

    shape = tuple(int(v) for v in input_shape)
    if len(shape) == 3:
        return 1
    if len(shape) == 4:
        return int(shape[0])
    raise ValueError(f'CNN input_shape must be [C,H,W] or [T,C,H,W], got {shape}.')


def _build_fixed_cnn2d_classifier(
    *,
    spec: ModelSpec,
    input_dim: int,
    sequence_length: int,
    input_shape: Sequence[int] | None,
    num_classes: int,
    layer_specs: Sequence[LayerSpec],
    output_layer_overrides: dict[str, Any] | None,
    v_th: float,
) -> FixedCNN2DClassifier:
    """Build one canonical VGG11/ResNet18 CNN-SNN classifier."""

    resolved_input_shape = _resolve_cnn_input_shape(
        input_shape=input_shape,
        input_dim=int(input_dim),
        sequence_length=int(sequence_length),
    )
    input_channels = int(resolved_input_shape[0] if len(resolved_input_shape) == 3 else resolved_input_shape[1])

    dense_hidden_widths = [int(layer.width) for layer in resolved_layer_specs if isinstance(layer, DenseLayerSpec)]
    if len(dense_hidden_widths) != len(resolved_layer_specs):
        if constraint_config is not None and str(getattr(constraint_config, 'mode', 'none')).strip().lower() not in {'none', ''}:
            raise ValueError('constraint_mode is supported only for dense hidden layers in v1 (no conv/residual arch).')
    constraint_plan = resolve_constraint_plan(spec, dense_hidden_widths, constraint_config)

    hidden_layers: list[nn.Module] = []
    layer_meta: list[LayerMeta] = []
    pool_after: list[nn.Module | None] = []
    prev_size = input_channels
    conv_count = 0
    block_count = 0
    for layer_spec in layer_specs:
        if isinstance(layer_spec, ConvLayerSpec):
            conv_count += 1
            layer = _build_conv2d_family_layer(
                spec,
                input_size=prev_size,
                layer_spec=layer_spec,
                v_th=v_th,
                output_overrides={},
            layer_constraint=layer_constraint,
            )
            hidden_layers.append(layer)
            layer_meta.append(LayerMeta(name=f'{spec.backbone}_conv_{conv_count:02d}', size=int(layer_spec.out_channels), is_output=False))
            if bool(layer_spec.pool_after):
                pool_after.append(
                    nn.MaxPool2d(
                        kernel_size=int(layer_spec.pool_kernel_size),
                        stride=int(layer_spec.pool_stride),
                        padding=int(layer_spec.pool_padding),
                        ceil_mode=bool(layer_spec.pool_ceil_mode),
                    )
                )
            else:
                pool_after.append(None)
            prev_size = int(layer_spec.out_channels)
        elif isinstance(layer_spec, ResidualBlockSpec):
            block_count += 1
            layer = CNN2DResidualBlock(spec, input_size=prev_size, block_spec=layer_spec, v_th=v_th)
            hidden_layers.append(layer)
            layer_meta.append(LayerMeta(name=f'{spec.backbone}_block_{block_count:02d}', size=int(layer_spec.out_channels), is_output=False))
            pool_after.append(None)
            prev_size = int(layer_spec.out_channels)
        else:
            raise ValueError(f'Fixed CNN backbones do not support dense layer spec {layer_spec!r}.')

    output_layer_spec = ConvLayerSpec(out_channels=int(num_classes), kernel_size=1, stride=1, padding=0, bias=True)
    output_layer = _build_conv2d_family_layer(
        spec,
        input_size=prev_size,
        layer_spec=output_layer_spec,
        v_th=v_th,
        output_overrides={} if output_layer_overrides is None else dict(output_layer_overrides),
    )
    layer_meta.append(LayerMeta(name='output', size=int(num_classes), is_output=True))
    extra_metadata: dict[str, Any] = {
        'rf_reset_mode': spec.reset_mode if spec.family == 'cnn_rf' else None,
        'lif_reset_mode': spec.reset_mode if spec.family == 'cnn_lif' else None,
        'rf_trainable_threshold': bool(spec.trainable_threshold) if spec.family == 'cnn_rf' else None,
        'lif_trainable_threshold': bool(spec.trainable_threshold) if spec.family == 'cnn_lif' else None,
        'v_th': float(v_th),
        'backbone': spec.backbone,
        'backbone_structure': 'VGG-11' if spec.backbone == 'vgg11' else 'ResNet-18',
        'hidden_spec': serialize_arch_spec(layer_specs),
        'arch_spec': serialize_arch_spec(layer_specs),
        'arch_layers': arch_spec_payload(layer_specs),
        'hidden_sizes': arch_hidden_sizes(layer_specs),
        'cnn_head': 'adaptive_avg_pool_2d_plus_1x1_spiking_output',
    }
    return FixedCNN2DClassifier(
        spec=spec,
        input_dim=int(input_dim),
        sequence_length=int(sequence_length),
        input_shape=resolved_input_shape,
        output_sequence_length=_cnn_temporal_output_length(resolved_input_shape),
        num_classes=int(num_classes),
        hidden_layers=hidden_layers,
        output_layer=output_layer,
        layer_meta=layer_meta,
        pool_after=pool_after,
        extra_metadata=extra_metadata,
    )



def _build_author_backbone_classifier(
    *,
    spec: ModelSpec,
    input_dim: int,
    sequence_length: int,
    num_classes: int,
    input_shape: Sequence[int] | None,
    device_note: str = '',
) -> SNNClassifier:
    """Build one official author-source auxiliary diversity wrapper."""

    del device_note
    if spec.family == 'spikformer':
        from src.model.author_adapter_spikformer import build_spikformer_author_classifier

        return build_spikformer_author_classifier(
            spec=spec,
            input_dim=int(input_dim),
            sequence_length=int(sequence_length),
            num_classes=int(num_classes),
            input_shape=input_shape,
            layer_record_cls=LayerRecord,
            forward_result_cls=ForwardResult,
        )
    if spec.family == 'spikingssm':
        from src.model.author_adapter_state_space import build_state_space_author_classifier

        return build_state_space_author_classifier(
            spec=spec,
            input_dim=int(input_dim),
            sequence_length=int(sequence_length),
            num_classes=int(num_classes),
            input_shape=input_shape,
            layer_record_cls=LayerRecord,
            forward_result_cls=ForwardResult,
        )
    raise ValueError(f'Unsupported author-source auxiliary profile: {spec.family!r}')


def build_snn_classifier(
    *,
    model_token: str | ModelSpec,
    input_dim: int,
    sequence_length: int,
    num_classes: int,
    input_shape: Sequence[int] | None = None,
    hidden_sizes: Sequence[int] | None = None,
    arch_spec: str | Sequence[str] | None = None,
    layer_specs: Sequence[LayerSpec] | None = None,
    output_layer_overrides: dict[str, Any] | None = None,
    v_th: float = 1.0,
    constraint_config: ConstraintConfig | None = None,
) -> SNNClassifier:
    """Build one complete hidden-layer stack plus output neuron layer."""

    spec = canonicalize_model_token(model_token) if isinstance(model_token, str) else model_token
    if constraint_config is not None:
        constraint_mode = str(getattr(constraint_config, 'mode', 'none')).strip().lower()
        if constraint_mode not in {'', 'none'} and spec.family not in {'lif', 'rf'}:
            raise ValueError(
                f'constraint_mode={constraint_mode!r} is supported only for dense lif/rf families; got family={spec.family!r}.'
            )
    if spec.family == 'spikegru':
        if input_shape is not None:
            raise ValueError('spikegru is specified as a non-image [B,T,C] model and does not accept input_shape metadata.')
        return SpikGRUClassifier(
            spec=spec,
            input_dim=int(input_dim),
            sequence_length=int(sequence_length),
            num_classes=int(num_classes),
            hidden_size=128,
            v_th=float(v_th),
        )

    if spec.family in {'spikingssm', 'spikformer'}:
        return _build_author_backbone_classifier(
            spec=spec,
            input_dim=int(input_dim),
            sequence_length=int(sequence_length),
            num_classes=int(num_classes),
            input_shape=input_shape,
        )
    
    if spec.family in {"cnn_lif", "cnn_rf"} and spec.backbone in {"vgg11", "resnet18"}:
        from src.model.snnbench_adapter import build_snnbench_cnn_classifier

        return build_snnbench_cnn_classifier(
            spec=spec,
            input_dim=int(input_dim),
            sequence_length=int(sequence_length),
            num_classes=int(num_classes),
            input_shape=input_shape,
            v_th=float(v_th),
            layer_record_cls=LayerRecord,
            forward_result_cls=ForwardResult,
        )

    if layer_specs is None:
        resolved_layer_specs = resolve_arch_spec(
            model_spec=spec,
            arch_spec_text=arch_spec,
            hidden_sizes=None if hidden_sizes is None else [int(v) for v in hidden_sizes],
        )
    else:
        resolved_layer_specs = list(layer_specs)

    if spec.family in {'cnn_lif', 'cnn_rf'}:
        return _build_fixed_cnn2d_classifier(
            spec=spec,
            input_dim=int(input_dim),
            sequence_length=int(sequence_length),
            input_shape=input_shape,
            num_classes=int(num_classes),
            layer_specs=resolved_layer_specs,
            output_layer_overrides=output_layer_overrides,
            v_th=float(v_th),
        )


    dense_hidden_widths = [int(layer.width) for layer in resolved_layer_specs if isinstance(layer, DenseLayerSpec)]
    if len(dense_hidden_widths) != len(resolved_layer_specs):
        if constraint_config is not None and str(getattr(constraint_config, 'mode', 'none')).strip().lower() not in {'none', ''}:
            raise ValueError('constraint_mode is supported only for dense hidden layers in v1 (no conv/residual arch).')
    constraint_plan = resolve_constraint_plan(spec, dense_hidden_widths, constraint_config)

    hidden_layers: list[nn.Module] = []
    layer_meta: list[LayerMeta] = []
    prev_size = int(input_dim)
    current_sequence_length = int(sequence_length)
    for hidden_index, layer_spec in enumerate(resolved_layer_specs, start=1):
        layer_constraint = None
        if isinstance(layer_spec, DenseLayerSpec):
            layer_constraint = layer_constraint_for_hidden_index(constraint_plan, hidden_index=hidden_index - 1, input_size=prev_size, output_size=int(layer_spec.width), recurrent=bool(spec.recurrent))
        layer = build_layer_from_spec(
            spec,
            input_size=prev_size,
            layer_spec=layer_spec,
            v_th=v_th,
            output_overrides={},
            layer_constraint=layer_constraint,
        )
        hidden_layers.append(layer)
        hidden_size = int(layer_spec.width) if isinstance(layer_spec, DenseLayerSpec) else int(layer_spec.out_channels)
        layer_meta.append(LayerMeta(name=f'layer_{hidden_index:02d}', size=hidden_size, is_output=False))
        prev_size = hidden_size
        if isinstance(layer_spec, (ConvLayerSpec, ResidualBlockSpec)):
            current_sequence_length = ((current_sequence_length + 2 * int(layer_spec.padding) - int(layer_spec.kernel_size)) // int(layer_spec.stride)) + 1
            if current_sequence_length <= 0:
                raise ValueError(
                    'Resolved conv arch_spec collapses the sequence axis to non-positive length before the output layer; '
                    f'got length={current_sequence_length} at hidden layer {hidden_index}. '
                )

    output_layer_spec: LayerSpec
    if spec.family in {'cnn_lif', 'cnn_rf'}:
        output_layer_spec = ConvLayerSpec(out_channels=int(num_classes), kernel_size=1, stride=1, padding=0)
    else:
        output_layer_spec = DenseLayerSpec(width=int(num_classes))
    output_spec = _output_layer_model_spec(spec)
    output_layer = build_layer_from_spec(
        output_spec,
        input_size=prev_size,
        layer_spec=output_layer_spec,
        v_th=v_th,
        output_overrides=output_layer_overrides,
    )
    layer_meta.append(LayerMeta(name='output', size=int(num_classes), is_output=True))

    extra_metadata: dict[str, Any] = {
        'rf_reset_mode': spec.reset_mode if spec.family in {'rf', 'cnn_rf'} else None,
        'lif_reset_mode': spec.reset_mode if spec.family in {'lif', 'cnn_lif'} else None,
        'rf_trainable_threshold': bool(spec.trainable_threshold) if spec.family in {'rf', 'cnn_rf'} else None,
        'lif_trainable_threshold': bool(spec.trainable_threshold) if spec.family in {'lif', 'cnn_lif'} else None,
        'v_th': float(v_th),
        'backbone': spec.backbone,
        'hidden_spec': serialize_arch_spec(resolved_layer_specs),
        'arch_spec': serialize_arch_spec(resolved_layer_specs),
        'arch_layers': arch_spec_payload(resolved_layer_specs),
        'hidden_sizes': arch_hidden_sizes(resolved_layer_specs),
        'constraint_metadata': constraint_plan.metadata,
    }

    return SNNClassifier(
        spec=spec,
        input_dim=input_dim,
        sequence_length=int(sequence_length),
        output_sequence_length=infer_output_sequence_length(int(sequence_length), list(resolved_layer_specs) + [output_layer_spec]),
        num_classes=num_classes,
        hidden_layers=hidden_layers,
        output_layer=output_layer,
        layer_meta=layer_meta,
        extra_metadata=extra_metadata,
    )


__all__ = [
    'ForwardResult',
    'LayerMeta',
    'LayerRecord',
    'SNNClassifier',
    'SpikGRUClassifier',
    'build_layer_from_spec',
    'build_snn_classifier',
]
try:
    from src.patch_overlays.runtime_patch import patch_snn_builder as _patch_snn_builder
    _patch_snn_builder(globals())
except Exception:
    pass
