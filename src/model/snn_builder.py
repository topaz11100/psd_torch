"""Model-construction utilities for PSD analysis."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import os
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
from src.neurons._common import logit, sequence_backend_name, sequence_buffer_mode, sequence_state_dtype, surrogate_spike, to_sequence_state_dtype
from src.neurons._compile import compile_callable, disable_compiled_runtime
from src.neurons.cnn2d import CNN2DLIFLayer, CNN2DRFLayer


def _truthy_env(name: str) -> bool:
    return str(os.environ.get(name, '')).strip().lower() in {'1', 'true', 'yes', 'on'}


def _configure_default_torch_cpu_threads() -> None:
    """Bound default CPU parallelism unless the caller already configured it.

    Some CI/shared hosts expose a large CPU count.  Letting PyTorch and BLAS use
    all visible CPUs made small SNN/CNN smoke paths stall after origin-wrapper
    imports.  Users can override with ``PSD_TORCH_CPU_THREADS`` or their normal
    BLAS thread environment variables.
    """

    if _truthy_env('PSD_DISABLE_DEFAULT_TORCH_THREAD_CAP'):
        return
    explicit = os.environ.get('PSD_TORCH_CPU_THREADS')
    thread_env_keys = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
    if explicit is None and any(os.environ.get(key) for key in thread_env_keys):
        return
    try:
        requested = int(explicit) if explicit not in {None, ''} else min(os.cpu_count() or 1, 8)
    except Exception:
        requested = min(os.cpu_count() or 1, 8)
    num_threads = max(1, requested)
    for key in thread_env_keys:
        os.environ.setdefault(key, str(num_threads))
    try:
        torch.set_num_threads(num_threads)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(max(1, min(num_threads, max(1, num_threads // 2))))
    except Exception:
        pass


_configure_default_torch_cpu_threads()


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
    if flattened.device.type == 'cuda':
        flattened = flattened.contiguous(memory_format=torch.channels_last)
    output = module(flattened)
    out_channels, out_height, out_width = [int(v) for v in output.shape[1:]]
    return output.reshape(batch_size, time_steps, out_channels, out_height, out_width).contiguous()


def _squeeze_unit_spatial(record_tensor: torch.Tensor) -> torch.Tensor:
    """Convert ``(B,T,C,1,1)`` output tensors to ``(B,T,C)`` for readout code."""

    if record_tensor.ndim != 5 or int(record_tensor.shape[-1]) != 1 or int(record_tensor.shape[-2]) != 1:
        raise ValueError(f'Expected output tensor shape (B,T,C,1,1), got {tuple(record_tensor.shape)}')
    return record_tensor[..., 0, 0].contiguous()


def _compile_child_regions(children: Iterable[tuple[str, nn.Module]], compile_kwargs: dict[str, Any]) -> tuple[int, list[str]]:
    applied_count = 0
    policies: list[str] = []
    seen: set[int] = set()
    for name, child in children:
        if id(child) in seen:
            continue
        seen.add(id(child))
        hook = getattr(child, 'enable_compiled_forward', None)
        if not callable(hook):
            continue
        applied, policy = hook(**compile_kwargs)
        policies.append(f'{name}:{policy}')
        if applied:
            applied_count += 1
    return applied_count, policies


def _sew_merge(branch_spike: torch.Tensor, shortcut_spike: torch.Tensor, cnf: str = 'ADD') -> torch.Tensor:
    """Apply the SEW residual merge rule on spike tensors.

    This imports only the residual semantics from ``reference/SNNs/sew_resnet.py``.
    The input geometry and neuron implementation remain project-owned.
    """

    token = str(cnf).upper()
    if token == 'ADD':
        return branch_spike + shortcut_spike
    if token == 'AND':
        return branch_spike * shortcut_spike
    if token == 'IAND':
        return (1.0 - branch_spike) * shortcut_spike
    raise ValueError(f'Unsupported SEW connect function: {cnf!r}')


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
    """Input-agnostic SEW-ResNet BasicBlock implemented with project-native layers."""

    sew_connect_function = 'ADD'

    def __init__(self, spec: ModelSpec, *, input_size: int, block_spec: ResidualBlockSpec, v_th: float) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(block_spec.out_channels)
        self.kernel_size = int(block_spec.kernel_size)
        self.stride = int(block_spec.stride)
        self.padding = int(block_spec.padding)
        self.batch_norm = bool(block_spec.batch_norm)
        self.layer1 = _build_conv2d_family_layer(
            spec,
            input_size=self.input_size,
            layer_spec=ConvLayerSpec(out_channels=self.output_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, batch_norm=self.batch_norm, bias=False),
            v_th=v_th,
            output_overrides={},
        )
        self.layer2 = _build_conv2d_family_layer(
            spec,
            input_size=self.output_size,
            layer_spec=ConvLayerSpec(out_channels=self.output_size, kernel_size=self.kernel_size, stride=1, padding=self.padding, batch_norm=self.batch_norm, bias=False),
            v_th=v_th,
            output_overrides={},
        )
        self.skip_projection = None
        self.skip_bn = None
        self.skip_activation = None
        if self.input_size != self.output_size or self.stride != 1:
            self.skip_projection = nn.Conv2d(self.input_size, self.output_size, kernel_size=1, stride=self.stride, padding=0, bias=False)
            self.skip_bn = nn.BatchNorm2d(self.output_size) if self.batch_norm else None
            self.skip_activation = _make_identity_activation_layer(
                _build_conv2d_family_layer(
                    spec,
                    input_size=self.output_size,
                    layer_spec=ConvLayerSpec(out_channels=self.output_size, kernel_size=1, stride=1, padding=0, batch_norm=False, bias=False),
                    v_th=v_th,
                    output_overrides={},
                )
            )
        self._last_layer_input = None
        self._last_trace_records: list[LayerRecord] = []

    def enable_compiled_forward(self, **compile_kwargs: Any) -> tuple[bool, str]:
        children: list[tuple[str, nn.Module]] = [('layer1', self.layer1), ('layer2', self.layer2)]
        if self.skip_activation is not None:
            children.append(('skip_activation', self.skip_activation))
        applied_count, policies = _compile_child_regions(children, dict(compile_kwargs or {}))
        if applied_count > 0:
            return True, 'regional_compile[' + ';'.join(policies) + ']'
        return False, 'regional_compile_no_child_regions[' + ';'.join(policies) + ']'

    def _skip_path(self, input_sequence: torch.Tensor, *, return_traces: bool) -> torch.Tensor:
        if self.skip_projection is None:
            return input_sequence
        output = _time_distributed_2d(self.skip_projection, input_sequence)
        if self.skip_bn is not None:
            output = _time_distributed_2d(self.skip_bn, output)
        if self.skip_activation is None:
            return output
        _skip_membrane, skip_spike = self.skip_activation(output, return_traces=return_traces)
        return skip_spike

    @staticmethod
    def _sew_merge(branch_spike: torch.Tensor, shortcut_spike: torch.Tensor, cnf: str = 'ADD') -> torch.Tensor:
        token = str(cnf).upper()
        if token == 'ADD':
            return branch_spike + shortcut_spike
        if token == 'AND':
            return branch_spike * shortcut_spike
        if token == 'IAND':
            return (1.0 - branch_spike) * shortcut_spike
        raise ValueError(f'Unsupported SEW connect function: {cnf!r}')

    def forward(self, input_sequence: torch.Tensor, *, return_traces: bool = False) -> tuple[torch.Tensor | None, torch.Tensor]:
        self._last_layer_input = None
        self._last_trace_records = []
        mem1, spike1 = self.layer1(input_sequence, return_traces=return_traces)
        mem2, branch_spike = self.layer2(spike1, return_traces=return_traces)
        shortcut_spike = self._skip_path(input_sequence, return_traces=return_traces)
        merged_spike = self._sew_merge(branch_spike, shortcut_spike, self.sew_connect_function)
        if return_traces:
            layer1_input = getattr(self.layer1, '_last_layer_input', None)
            layer2_input = getattr(self.layer2, '_last_layer_input', None)
            if mem1 is None or layer1_input is None:
                raise RuntimeError('SEW-ResNet BasicBlock first spiking site did not expose complete traces.')
            if mem2 is None or layer2_input is None:
                raise RuntimeError('SEW-ResNet BasicBlock second spiking site did not expose complete traces.')
            self._last_layer_input = layer2_input
            self._last_trace_records = [
                LayerRecord(layer_name='conv1', membrane=mem1, spike=spike1, layer_input=layer1_input),
                LayerRecord(layer_name='residual_add', membrane=mem2, spike=merged_spike, layer_input=layer2_input),
            ]
        return mem2, merged_spike

    def trace_records(self, base_name: str) -> list[LayerRecord]:
        return [
            LayerRecord(layer_name=f'{base_name}_{record.layer_name}', membrane=record.membrane, spike=record.spike, layer_input=record.layer_input)
            for record in self._last_trace_records
        ]

    def filter_stats_vectors(self) -> dict[str, torch.Tensor]:
        merged: dict[str, list[torch.Tensor]] = {}
        for layer in (self.layer1, self.layer2, self.skip_activation):
            if layer is None or not hasattr(layer, 'filter_stats_vectors'):
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
        self.compile_granularity = 'sequence_regions'
        self._compiled_core_forward = None
        self._compiled_core_disabled = False
        self._compiled_core_error = None
        self._compiled_core_policy = 'eager'

    def enable_compiled_forward(self, **compile_kwargs: Any) -> tuple[bool, str]:
        kwargs = dict(compile_kwargs or {})
        child_pairs = list(self.iter_named_layers())
        child_applied, child_policies = _compile_child_regions(child_pairs, kwargs)
        if _truthy_env('PSD_ENABLE_CNN_CORE_COMPILE'):
            compiled, core_applied, core_policy = compile_callable(self._forward_core_no_trace, compile_kwargs=kwargs, label='cnn_core')
            if core_applied:
                self._compiled_core_forward = compiled
                self._compiled_core_disabled = False
                self._compiled_core_error = None
                self._compiled_core_policy = core_policy
            else:
                self._compiled_core_forward = None
                self._compiled_core_error = core_policy
                self._compiled_core_policy = core_policy
        else:
            core_applied = False
            self._compiled_core_forward = None
            self._compiled_core_error = 'cnn_core_compile_skipped_outer_loop_guard'
            core_policy = self._compiled_core_error
            self._compiled_core_policy = core_policy
        policies = []
        if child_policies:
            policies.append('children=' + ';'.join(child_policies))
        policies.append('core=' + core_policy)
        self.extra_metadata['compile_granularity'] = 'cnn_sequence_regions' if not core_applied else 'cnn_core_plus_sequence_regions'
        self.extra_metadata['compile_child_region_count'] = int(child_applied)
        self.extra_metadata['compile_core_policy'] = self._compiled_core_policy
        self.extra_metadata['sequence_backend'] = sequence_backend_name()
        self.extra_metadata['sequence_buffer_mode'] = sequence_buffer_mode()
        return bool(core_applied or child_applied > 0), 'regional_cnn_compile[' + '|'.join(policies) + ']'

    def iter_named_layers(self) -> Iterable[tuple[str, nn.Module]]:
        for meta, layer in zip(self.layer_meta[:-1], self.hidden_layers):
            if isinstance(layer, CNN2DResidualBlock):
                yield f'{meta.name}_conv1', layer.layer1
                yield f'{meta.name}_residual_add', layer.layer2
            else:
                yield meta.name, layer
        yield self.layer_meta[-1].name, self.output_layer

    def iter_named_hidden_layers(self) -> Iterable[tuple[str, nn.Module]]:
        for meta, layer in zip(self.layer_meta[:-1], self.hidden_layers):
            if isinstance(layer, CNN2DResidualBlock):
                yield f'{meta.name}_conv1', layer.layer1
                yield f'{meta.name}_residual_add', layer.layer2
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
        expected_channels, expected_time, _channels, height, width = self._expected_cnn_shape()
        if tensor.ndim == 4 and expected_time is None:
            tensor = tensor.unsqueeze(1)
            expected_time = 1
        elif tensor.ndim != 5:
            raise ValueError(
                'CNN models require prepared image input shape (B,T,C,H,W); '
                'rank-4 (B,C,H,W) is accepted only for static rank-3 image metadata. '
                f'Got shape {tuple(tensor.shape)}.'
            )
        if int(tensor.shape[2]) != expected_channels:
            raise ValueError(f'Expected frame channels={expected_channels}, got shape {tuple(tensor.shape)}.')
        if expected_time is not None and int(tensor.shape[1]) != expected_time:
            raise ValueError(f'Expected temporal frames={expected_time}, got shape {tuple(tensor.shape)}.')
        if int(tensor.shape[3]) != height or int(tensor.shape[4]) != width:
            raise ValueError(f'Expected spatial shape ({height},{width}), got {tuple(tensor.shape[-2:])}.')
        return tensor.contiguous()

    def _forward_core_no_trace(self, prepared_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current = prepared_input
        for index, layer in enumerate(self.hidden_layers):
            _membrane, spike = layer(current, return_traces=False)
            current = spike
            if self.pool_after_enabled[index]:
                current = _time_distributed_2d(self.pool_after[index], current)
        current = _time_distributed_2d(self.global_pool, current)
        output_membrane_5d, output_spike_5d = self.output_layer(current, return_traces=True)
        if output_membrane_5d is None:
            raise RuntimeError('Output layer must always return membrane traces.')
        output_layer_input_5d = getattr(self.output_layer, '_last_layer_input', None)
        if output_layer_input_5d is None:
            raise RuntimeError('Output layer did not expose exact layer_input traces under return_traces=True.')
        return output_membrane_5d, output_spike_5d, output_layer_input_5d

    def forward(self, input_sequence: torch.Tensor, *, capture_hidden: bool = False) -> ForwardResult:
        hidden_records: list[LayerRecord] = []
        prepared_input = self._prepare_input(input_sequence)
        if not bool(capture_hidden):
            core_forward = self._compiled_core_forward if (self._compiled_core_forward is not None and not self._compiled_core_disabled) else self._forward_core_no_trace
            try:
                output_membrane_5d, output_spike_5d, output_layer_input_5d = core_forward(prepared_input)
            except Exception as exc:  # pragma: no cover - backend dependent fallback
                if core_forward is self._forward_core_no_trace:
                    raise
                self._compiled_core_disabled = True
                self._compiled_core_error = f'{type(exc).__name__}: {exc}'
                output_membrane_5d, output_spike_5d, output_layer_input_5d = self._forward_core_no_trace(prepared_input)
            output_record = LayerRecord(
                layer_name=self.layer_meta[-1].name,
                membrane=_squeeze_unit_spatial(output_membrane_5d),
                spike=_squeeze_unit_spatial(output_spike_5d),
                layer_input=_squeeze_unit_spatial(output_layer_input_5d),
            )
            return ForwardResult(hidden_records=hidden_records, output_record=output_record, input_record=prepared_input)

        current = prepared_input
        for index, (meta, layer) in enumerate(zip(self.layer_meta[:-1], self.hidden_layers)):
            membrane, spike = layer(current, return_traces=True)
            current = spike
            record_membrane = membrane
            record_layer_input = getattr(layer, '_last_layer_input', None)
            if self.pool_after_enabled[index]:
                current = _time_distributed_2d(self.pool_after[index], current)
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
        self.compile_granularity = 'sequence_regions'

    def enable_compiled_forward(self, **compile_kwargs: Any) -> tuple[bool, str]:
        applied_count, policies = _compile_child_regions(list(self.iter_named_layers()), dict(compile_kwargs or {}))
        self.extra_metadata['compile_granularity'] = 'sequence_regions'
        self.extra_metadata['compile_child_region_count'] = int(applied_count)
        self.extra_metadata['compile_child_policies'] = list(policies)
        self.extra_metadata['sequence_backend'] = sequence_backend_name()
        self.extra_metadata['sequence_buffer_mode'] = sequence_buffer_mode()
        if applied_count > 0:
            return True, 'regional_sequence_compile[' + ';'.join(policies) + ']'
        return False, 'regional_sequence_compile_no_child_regions[' + ';'.join(policies) + ']'

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


def _spikgru_sequence_impl(
    gate_input_sequence: torch.Tensor,
    candidate_input_sequence: torch.Tensor,
    hidden: torch.Tensor,
    current_state: torch.Tensor,
    previous_spike: torch.Tensor,
    alpha: torch.Tensor,
    hidden_to_gate_weight: torch.Tensor,
    hidden_to_candidate_weight: torch.Tensor,
    hidden_to_candidate_bias: torch.Tensor | None,
    threshold: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, time_steps, hidden_size = gate_input_sequence.shape
    mem_seq = gate_input_sequence.new_empty((batch_size, time_steps, hidden_size))
    spike_seq = gate_input_sequence.new_empty((batch_size, time_steps, hidden_size))
    current_seq = gate_input_sequence.new_empty((batch_size, time_steps, hidden_size))
    gate_seq = gate_input_sequence.new_empty((batch_size, time_steps, hidden_size))
    alpha_view = alpha.unsqueeze(0)
    for t in range(time_steps):
        gate_input_t = gate_input_sequence[:, t, :]
        candidate_input_t = candidate_input_sequence[:, t, :]
        z_t = torch.sigmoid(gate_input_t + F.linear(previous_spike, hidden_to_gate_weight, None))
        drive_t = candidate_input_t + F.linear(previous_spike, hidden_to_candidate_weight, hidden_to_candidate_bias)
        i_t = alpha_view * current_state + drive_t
        mem_t = z_t * hidden + (1.0 - z_t) * i_t - threshold * previous_spike
        spike_t = surrogate_spike(mem_t - threshold)
        hidden = mem_t
        current_state = i_t
        previous_spike = spike_t
        mem_seq[:, t, :] = mem_t
        spike_seq[:, t, :] = spike_t
        current_seq[:, t, :] = i_t
        gate_seq[:, t, :] = z_t
    return mem_seq, spike_seq, current_seq, gate_seq


class SpikGRUCellBlock(nn.Module):
    """One vanilla SpikGRU recurrent block with a compiled sequence region."""

    compile_granularity = 'sequence'

    def __init__(self, input_dim: int, hidden_size: int, *, v_th: float = 1.0) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.v_threshold = float(v_th)
        self.input_to_candidate = nn.Linear(self.input_dim, self.hidden_size)
        self.hidden_to_candidate = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.input_to_gate = nn.Linear(self.input_dim, self.hidden_size)
        self.hidden_to_gate = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.alpha_raw = nn.Parameter(torch.full((self.hidden_size,), float(logit(0.9).item())))
        self._compiled_sequence = None
        self._compiled_sequence_policy = 'eager'
        self._sequence_compiled_runtime_disabled = False
        self._sequence_compiled_runtime_error = None

    def enable_compiled_forward(self, **compile_kwargs: Any) -> tuple[bool, str]:
        compiled, applied, policy = compile_callable(_spikgru_sequence_impl, compile_kwargs=compile_kwargs, label='spikgru_sequence')
        if applied:
            self._compiled_sequence = compiled
            self._compiled_sequence_policy = policy
            self._sequence_compiled_runtime_disabled = False
            self._sequence_compiled_runtime_error = None
        return applied, policy

    def initial_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        current_state = torch.zeros_like(hidden)
        previous_spike = torch.zeros_like(hidden)
        return hidden, current_state, previous_spike

    def run_sequence(
        self,
        gate_input_sequence: torch.Tensor,
        candidate_input_sequence: torch.Tensor,
        hidden: torch.Tensor,
        current_state: torch.Tensor,
        previous_spike: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = gate_input_sequence.dtype
        device = gate_input_sequence.device
        alpha = self.effective_alpha().to(device=device, dtype=dtype)
        threshold = torch.as_tensor(self.v_threshold, device=device, dtype=dtype)
        gate_weight = self.hidden_to_gate.weight.to(device=device, dtype=dtype)
        candidate_weight = self.hidden_to_candidate.weight.to(device=device, dtype=dtype)
        candidate_bias = None if self.hidden_to_candidate.bias is None else self.hidden_to_candidate.bias.to(device=device, dtype=dtype)
        fn = self._compiled_sequence
        if fn is not None and not bool(self._sequence_compiled_runtime_disabled):
            try:
                return fn(gate_input_sequence, candidate_input_sequence, hidden, current_state, previous_spike, alpha, gate_weight, candidate_weight, candidate_bias, threshold)
            except Exception as exc:
                disable_compiled_runtime(self, label='sequence', exc=exc)
        return _spikgru_sequence_impl(gate_input_sequence, candidate_input_sequence, hidden, current_state, previous_spike, alpha, gate_weight, candidate_weight, candidate_bias, threshold)

    def step(
        self,
        *,
        gate_input_t: torch.Tensor,
        candidate_input_t: torch.Tensor,
        hidden: torch.Tensor,
        current_state: torch.Tensor,
        previous_spike: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Eager single-step reference retained for parity tests and debugging."""

        alpha = torch.sigmoid(self.alpha_raw).to(device=gate_input_t.device, dtype=gate_input_t.dtype).unsqueeze(0)
        z_t = torch.sigmoid(gate_input_t + self.hidden_to_gate(previous_spike))
        drive_t = candidate_input_t + self.hidden_to_candidate(previous_spike)
        i_t = alpha * current_state + drive_t
        mem_t = z_t * hidden + (1.0 - z_t) * i_t - float(self.v_threshold) * previous_spike
        spike_t = surrogate_spike(mem_t - float(self.v_threshold))
        return mem_t, i_t, spike_t, z_t, spike_t

    def effective_alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_raw)

    def filter_stats_vectors(self) -> dict[str, torch.Tensor]:
        device = self.alpha_raw.device
        return {
            'alpha': self.effective_alpha().detach(),
            'v_threshold': torch.full((self.hidden_size,), float(self.v_threshold), device=device, dtype=torch.float32),
        }

    def clamp_projected_parameters(self) -> None:
        # Alpha is structurally constrained to (0, 1) through ``sigmoid(alpha_raw)``.
        return None


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
        self.compile_granularity = 'sequence_regions'
        self.extra_metadata: dict[str, Any] = {}

    def enable_compiled_forward(self, **compile_kwargs: Any) -> tuple[bool, str]:
        applied_count, policies = _compile_child_regions([('layer_01', self.layer_01), ('layer_02', self.layer_02)], dict(compile_kwargs or {}))
        self.extra_metadata['compile_granularity'] = 'sequence_regions'
        self.extra_metadata['compile_child_region_count'] = int(applied_count)
        self.extra_metadata['compile_child_policies'] = list(policies)
        self.extra_metadata['sequence_backend'] = sequence_backend_name()
        self.extra_metadata['sequence_buffer_mode'] = sequence_buffer_mode()
        if applied_count > 0:
            return True, 'regional_spikgru_sequence_compile[' + ';'.join(policies) + ']'
        return False, 'regional_spikgru_sequence_compile_no_child_regions[' + ';'.join(policies) + ']'

    def iter_named_layers(self) -> Iterable[tuple[str, nn.Module]]:
        yield 'layer_01', self.layer_01
        yield 'layer_02', self.layer_02
        yield 'output', self.readout

    def iter_named_hidden_layers(self) -> Iterable[tuple[str, nn.Module]]:
        yield 'layer_01', self.layer_01
        yield 'layer_02', self.layer_02

    def model_metadata(self) -> dict[str, Any]:
        payload = {
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
            'alpha_constraint': 'structured_sigmoid_raw_parameter',
            'candidate_recurrent_bias': True,
            'origin_formula_contract': 'tempZ=sigmoid(wz(x)+uz(prev_spike)); tempcurrent=alpha*tempcurrent+wi(x)+ui(prev_spike); temp=tempZ*temp+(1-tempZ)*tempcurrent-v_th*prev_spike',
            'spike_reset_term': 'membrane_update_minus_v_th_times_previous_spike',
            'readout': 'non-spiking integrating readout membrane trace',
            'loss_reference': 'max-over-time cross entropy',
            'dvs_lip_frontend_included': False,
            'two_gate_backend_included': False,
            'bidirectional_backend_included': False,
            'signed_activation_ablation_included': False,
            'structure_variation': 'alpha_raw_sigmoid_unit_interval',
            'trace_families': ['x_probe', 'x_layer', 'i_current', 'z_gate', 'y_mem', 'y_spike', 'readout_mem'],
        }
        payload.update(getattr(self, 'extra_metadata', {}))
        return payload

    def clamp_projected_parameters(self) -> None:
        self.layer_01.clamp_projected_parameters()
        self.layer_02.clamp_projected_parameters()

    def _run_block(self, block: SpikGRUCellBlock, block_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, _time_steps, _input_dim = [int(v) for v in block_input.shape]
        state_dtype = sequence_state_dtype(block_input)
        hidden, current_state, previous_spike = block.initial_state(batch_size, device=block_input.device, dtype=state_dtype)
        gate_input_sequence = to_sequence_state_dtype(block.input_to_gate(block_input), block_input)
        candidate_input_sequence = to_sequence_state_dtype(block.input_to_candidate(block_input), block_input)
        mem_seq, spike_seq, current_seq, gate_seq = block.run_sequence(
            gate_input_sequence,
            candidate_input_sequence,
            hidden,
            current_state,
            previous_spike,
        )
        return mem_seq.contiguous(), spike_seq.contiguous(), current_seq.contiguous(), gate_seq.contiguous()


    def forward(self, input_sequence: torch.Tensor, *, capture_hidden: bool = False) -> ForwardResult:
        if input_sequence.ndim != 3:
            raise ValueError(f'SpikeGRU expects non-image input shape (B,T,C), got {tuple(input_sequence.shape)}')
        batch_size, time_steps, input_dim = [int(v) for v in input_sequence.shape]
        if input_dim != self.input_dim:
            raise ValueError(f'SpikeGRU expected input_dim={self.input_dim}, got {input_dim}.')
        layer1_mem, layer1_spike, layer1_current, layer1_gate = self._run_block(self.layer_01, input_sequence)
        layer2_mem, layer2_spike, layer2_current, layer2_gate = self._run_block(self.layer_02, layer1_spike)
        readout_drive_seq = to_sequence_state_dtype(self.readout(layer2_spike), layer2_spike)
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




def _fixed_filter_value_for_family(spec: ModelSpec, *, family: str) -> float | None:
    mode = str(getattr(spec, 'filter_mode', 'train') or 'train').strip().lower()
    value = getattr(spec, 'filter_value', None)
    if mode == 'train':
        return None
    if mode != 'fixed' or value is None:
        raise ValueError(f'Invalid filter setting mode={mode!r}, value={value!r}.')
    if family not in {'lif', 'rf', 'cnn_lif', 'cnn_rf'}:
        raise ValueError(f'Fixed filter values are supported only for lif/rf/cnn_lif/cnn_rf families, got {family!r}.')
    return float(value)

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
            filter_value=_fixed_filter_value_for_family(spec, family='lif'),
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
            filter_value=_fixed_filter_value_for_family(spec, family='rf'),
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
            filter_value=_fixed_filter_value_for_family(spec, family='cnn_lif'),
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
            filter_value=_fixed_filter_value_for_family(spec, family='cnn_rf'),
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
        'backbone_structure': 'VGG-11 topology' if spec.backbone == 'vgg11' else 'SEW-ResNet-18 topology; first BasicBlock consumes prep_data channels directly',
        'reference_backbone_contract': 'topology_only_from_reference_SNNs; input geometry comes from prep_data',
        'sew_resnet_connect_function': 'ADD' if spec.backbone == 'resnet18' else None,
        'spiking_backend': 'project_native_torch',
        'cnn_lif_backend': 'torch' if spec.family == 'cnn_lif' else None,
        'cnn_rf_backend': 'torch' if spec.family == 'cnn_rf' else None,
        'resnet_input_projection': 'none_first_basicblock_consumes_prepared_frame_channels' if spec.backbone == 'resnet18' else None,
        'vgg_input_policy': 'first_vgg_conv_consumes_prepared_frame_channels_directly' if spec.backbone == 'vgg11' else None,
        'hidden_spec': serialize_arch_spec(layer_specs),
        'arch_spec': serialize_arch_spec(layer_specs),
        'arch_layers': arch_spec_payload(layer_specs),
        'hidden_sizes': arch_hidden_sizes(layer_specs),
        'cnn_head': 'adaptive_avg_pool_2d_plus_1x1_spiking_output',
        'cnn_backend': 'project_pure_torch_compile_target',
        'reference_topology_only': True,
        'spikingjelly_runtime_backend': False,
        'input_contract': 'prepared prep_data image frames, rank (B,T,C,H,W); no reference input front-end is imported',
        'input_policy': 'prepared_data_shape_driven_no_reference_input_frontend',
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
        scenario_mode = str(getattr(constraint_config, 'mode', 'none')).strip().lower()
        if scenario_mode not in {'', 'none'} and spec.family not in {'if', 'lif', 'rf'}:
            raise ValueError(
                f'scenario_mode={scenario_mode!r} is supported only for dense if/lif/rf families; got family={spec.family!r}.'
            )
    if spec.family == 'spikegru':
        classifier = SpikGRUClassifier(
            spec=spec,
            input_dim=int(input_dim),
            sequence_length=int(sequence_length),
            num_classes=int(num_classes),
            hidden_size=128,
            v_th=float(v_th),
        )
        if input_shape is not None:
            classifier.extra_metadata['input_shape_metadata_ignored_for_sequence_model'] = [int(v) for v in input_shape]
            classifier.extra_metadata['runtime_input_contract'] = 'expects selected training view as [B,T,C]; image datasets use flattened sequence_input/flatten_input views for SpikeGRU.'
        return classifier

    if spec.family in {'spikingssm', 'spikformer'}:
        return _build_author_backbone_classifier(
            spec=spec,
            input_dim=int(input_dim),
            sequence_length=int(sequence_length),
            num_classes=int(num_classes),
            input_shape=input_shape,
        )

    # VGG-11/SEW-ResNet18 tokens are implemented by the project-native
    # pure-Torch CNN2D builder.  The reference code contributes only topology
    # and SEW residual semantics; no reference input front-end is imported.

    if layer_specs is None:
        # Fixed CNN families use topology resolved from arch_spec/backbone.
        # Project-wide dense hidden_sizes are intentionally ignored here.
        resolved_hidden_sizes = None if spec.family in {'cnn_lif', 'cnn_rf'} else (None if hidden_sizes is None else [int(v) for v in hidden_sizes])
        resolved_layer_specs = resolve_arch_spec(
            model_spec=spec,
            arch_spec_text=arch_spec,
            hidden_sizes=resolved_hidden_sizes,
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

    hidden_widths_for_constraints = [
        int(layer_spec.width) for layer_spec in resolved_layer_specs if isinstance(layer_spec, DenseLayerSpec)
    ]
    constraint_plan = resolve_constraint_plan(spec, hidden_widths_for_constraints, constraint_config)

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
