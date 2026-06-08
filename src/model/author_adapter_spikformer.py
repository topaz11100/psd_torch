"""Spikformer author-source adapter for the auxiliary diversity profile."""

from __future__ import annotations

import importlib.util
import sys
import types
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.neurons._compile import compile_callable, disable_compiled_runtime


_SPIKFORMER_STUBS_INSTALLED = False


def _install_spikformer_dependency_stubs() -> None:
    """Install minimal stubs only when optional author-code dependencies are absent."""

    global _SPIKFORMER_STUBS_INSTALLED
    installed = False

    try:
        import spikingjelly.clock_driven.neuron  # type: ignore  # noqa: F401
    except Exception:
        sj = sys.modules.setdefault('spikingjelly', types.ModuleType('spikingjelly'))
        clock_driven = sys.modules.setdefault('spikingjelly.clock_driven', types.ModuleType('spikingjelly.clock_driven'))
        neuron_mod = types.ModuleType('spikingjelly.clock_driven.neuron')
        layer_mod = types.ModuleType('spikingjelly.clock_driven.layer')
        surrogate_mod = types.ModuleType('spikingjelly.clock_driven.surrogate')

        class MultiStepLIFNode(nn.Module):
            def __init__(self, tau: float = 2.0, v_threshold: float = 1.0, detach_reset: bool = True, backend: str | None = None, **_kwargs: Any) -> None:
                super().__init__()
                self.tau = float(tau)
                self.v_threshold = float(v_threshold)
                self.detach_reset = bool(detach_reset)
                self.backend = 'torch_stub' if backend is None else f'torch_stub_for_{backend}'

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if x.ndim == 0:
                    return (x > float(self.v_threshold)).to(dtype=x.dtype)
                time_steps = int(x.shape[0])
                mem = torch.zeros_like(x[0])
                outs = []
                alpha = 1.0 - 1.0 / max(float(self.tau), 1.0)
                for index in range(time_steps):
                    mem = alpha * mem + x[index]
                    spike = (mem > float(self.v_threshold)).to(dtype=x.dtype)
                    reset = spike.detach() if self.detach_reset else spike
                    mem = mem - reset * float(self.v_threshold)
                    outs.append(spike)
                return torch.stack(outs, dim=0)

        neuron_mod.MultiStepLIFNode = MultiStepLIFNode
        clock_driven.neuron = neuron_mod
        clock_driven.layer = layer_mod
        clock_driven.surrogate = surrogate_mod
        sj.clock_driven = clock_driven
        sys.modules['spikingjelly.clock_driven.neuron'] = neuron_mod
        sys.modules['spikingjelly.clock_driven.layer'] = layer_mod
        sys.modules['spikingjelly.clock_driven.surrogate'] = surrogate_mod
        installed = True

    try:
        import timm.models.layers  # type: ignore  # noqa: F401
        import timm.models.registry  # type: ignore  # noqa: F401
        import timm.models.vision_transformer  # type: ignore  # noqa: F401
    except Exception:
        timm_mod = sys.modules.setdefault('timm', types.ModuleType('timm'))
        models_mod = sys.modules.setdefault('timm.models', types.ModuleType('timm.models'))
        layers_mod = types.ModuleType('timm.models.layers')
        registry_mod = types.ModuleType('timm.models.registry')
        vit_mod = types.ModuleType('timm.models.vision_transformer')

        def to_2tuple(value: Any) -> tuple[Any, Any]:
            if isinstance(value, tuple):
                return value
            return (value, value)

        def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0) -> torch.Tensor:
            return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

        class DropPath(nn.Module):
            def __init__(self, drop_prob: float = 0.0) -> None:
                super().__init__()
                self.drop_prob = float(drop_prob)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        def register_model(fn=None):
            def decorator(func):
                return func
            return decorator(fn) if fn is not None else decorator

        def _cfg(**kwargs: Any) -> dict[str, Any]:
            return dict(kwargs)

        layers_mod.to_2tuple = to_2tuple
        layers_mod.trunc_normal_ = trunc_normal_
        layers_mod.DropPath = DropPath
        registry_mod.register_model = register_model
        vit_mod._cfg = _cfg
        models_mod.layers = layers_mod
        models_mod.registry = registry_mod
        models_mod.vision_transformer = vit_mod
        timm_mod.models = models_mod
        sys.modules['timm.models.layers'] = layers_mod
        sys.modules['timm.models.registry'] = registry_mod
        sys.modules['timm.models.vision_transformer'] = vit_mod
        installed = True

    try:
        import einops.layers.torch  # type: ignore  # noqa: F401
    except Exception:
        einops_mod = sys.modules.setdefault('einops', types.ModuleType('einops'))
        layers_pkg = sys.modules.setdefault('einops.layers', types.ModuleType('einops.layers'))
        torch_layers_mod = types.ModuleType('einops.layers.torch')

        class Rearrange(nn.Module):
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        torch_layers_mod.Rearrange = Rearrange
        layers_pkg.torch = torch_layers_mod
        einops_mod.layers = layers_pkg
        sys.modules['einops.layers.torch'] = torch_layers_mod
        installed = True

    _SPIKFORMER_STUBS_INSTALLED = _SPIKFORMER_STUBS_INSTALLED or installed



def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _origin_root() -> Path:
    return _project_root() / 'origin'


def _load_origin_spikformer_class() -> type[nn.Module]:
    source_path = _origin_root() / 'spikformer' / 'cifar10dvs' / 'model.py'
    if not source_path.exists():
        raise RuntimeError(f'Official Spikformer source is missing: {source_path}')
    module_spec = importlib.util.spec_from_file_location('_psd_origin_spikformer_cifar10dvs_model', source_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f'Could not create import spec for official Spikformer source: {source_path}')
    _install_spikformer_dependency_stubs()
    module = importlib.util.module_from_spec(module_spec)
    try:
        module_spec.loader.exec_module(module)
    except Exception as exc:
        raise RuntimeError(
            'Could not import origin/spikformer/cifar10dvs/model.py. '
            'Install the author-code dependencies used by the official Spikformer profile.'
        ) from exc
    cls = getattr(module, 'Spikformer', None)
    if cls is None:
        raise RuntimeError('Official Spikformer source did not expose class Spikformer.')
    return cls


def _adapt_to_cifar10dvs_frames(input_sequence: torch.Tensor) -> torch.Tensor:
    """Adapt supported prepared inputs to the fixed [B,16,2,128,128] author profile."""

    x = input_sequence.to(dtype=torch.float32)
    if x.ndim == 5:
        frames = x
    elif x.ndim == 4:
        frames = x.unsqueeze(1).repeat(1, 16, 1, 1, 1)
    elif x.ndim == 3:
        batch, time_steps, features = [int(v) for v in x.shape]
        flat = x.reshape(batch * time_steps, features)
        target_features = 2 * 128 * 128
        if features < target_features:
            flat = F.pad(flat, (0, target_features - features))
        else:
            flat = flat[:, :target_features]
        frames = flat.reshape(batch, time_steps, 2, 128, 128)
    else:
        raise ValueError(f'Spikformer adapter expects rank 3, 4, or 5 input, got {tuple(x.shape)}.')

    if int(frames.shape[2]) == 1:
        frames = frames.repeat(1, 1, 2, 1, 1)
    elif int(frames.shape[2]) > 2:
        frames = frames[:, :, :2]
    elif int(frames.shape[2]) < 1:
        raise ValueError('Spikformer adapter received zero input channels.')

    batch, time_steps, channels, height, width = [int(v) for v in frames.shape]
    if time_steps != 16:
        flattened = frames.permute(0, 2, 3, 4, 1).reshape(batch * channels * height * width, 1, time_steps)
        resized = F.interpolate(flattened, size=16, mode='nearest')
        frames = resized.reshape(batch, channels, height, width, 16).permute(0, 4, 1, 2, 3).contiguous()
    if height != 128 or width != 128:
        frames_2d = frames.reshape(batch * 16, int(frames.shape[2]), int(frames.shape[3]), int(frames.shape[4]))
        frames_2d = F.interpolate(frames_2d, size=(128, 128), mode='nearest')
        frames = frames_2d.reshape(batch, 16, int(frames_2d.shape[1]), 128, 128).contiguous()
    return frames


class SpikformerAuthorClassifier(nn.Module):
    """Wrapper around origin/spikformer/cifar10dvs/model.py::Spikformer."""

    def __init__(
        self,
        *,
        spec: Any,
        input_dim: int,
        sequence_length: int,
        num_classes: int,
        input_shape: Sequence[int] | None,
        layer_record_cls: type[Any],
        forward_result_cls: type[Any],
    ) -> None:
        super().__init__()
        self.spec = spec
        self.input_dim = int(input_dim)
        self.sequence_length = int(sequence_length)
        self.output_sequence_length = 1
        self.num_classes = int(num_classes)
        self.input_shape = None if input_shape is None else tuple(int(v) for v in input_shape)
        self._layer_record_cls = layer_record_cls
        self._forward_result_cls = forward_result_cls
        origin_cls = _load_origin_spikformer_class()
        self.spikformer_dependency_backend = 'fallback_stubs' if _SPIKFORMER_STUBS_INSTALLED else 'author_dependencies'
        self.source_model = origin_cls(
            patch_size=16,
            embed_dims=256,
            num_heads=16,
            mlp_ratios=4,
            in_channels=2,
            num_classes=int(num_classes),
            qkv_bias=False,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=2,
            sr_ratios=1,
        )
        self._spikformer_block_outputs: list[torch.Tensor] = []
        self._capture_hooks_enabled = False
        self._compiled_source_forward = None
        self.extra_metadata: dict[str, Any] = {
            'compile_granularity': 'spikformer_source_forward_region',
            'compile_child_region_count': 0,
        }
        for block_index, block in enumerate(getattr(self.source_model, 'block', []), start=1):
            block.register_forward_hook(self._make_block_hook(block_index))

    def _make_block_hook(self, block_index: int):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if bool(getattr(self, '_capture_hooks_enabled', False)) and isinstance(output, torch.Tensor):
                self._spikformer_block_outputs.append(output)
        return hook

    def _source_forward_no_trace(self, frames: torch.Tensor) -> torch.Tensor:
        return self.source_model(frames)

    def enable_compiled_forward(self, **compile_kwargs: Any) -> tuple[bool, str]:
        compiled, applied, policy = compile_callable(
            self._source_forward_no_trace,
            compile_kwargs=dict(compile_kwargs or {}),
            label='spikformer_source',
        )
        self._compiled_source_forward = compiled if applied else None
        self.extra_metadata['compile_granularity'] = 'spikformer_source_forward_region'
        self.extra_metadata['compile_child_region_count'] = 1 if applied else 0
        self.extra_metadata['compile_child_policies'] = [policy]
        self.extra_metadata['compile_capture_policy'] = 'compiled_no_trace_forward_for_training; eager_source_forward_when_capture_hidden=True'
        if applied:
            return True, f'regional_spikformer_source_compile[{policy}]'
        return False, f'regional_spikformer_source_compile_not_applied[{policy}]'

    def _run_source_model(self, frames: torch.Tensor, *, capture_hidden: bool) -> torch.Tensor:
        if capture_hidden:
            return self.source_model(frames)
        compiled = getattr(self, '_compiled_source_forward', None)
        disabled = bool(getattr(self, '_spikformer_source_compiled_runtime_disabled', False))
        if compiled is not None and not disabled:
            try:
                return compiled(frames)
            except Exception as exc:
                disable_compiled_runtime(self, label='spikformer_source', exc=exc)
        return self.source_model(frames)

    @staticmethod
    def _block_output_to_batch_time_feature(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 4:
            # origin Block returns (T,B,C,N). PSD analysis expects (B,T,features).
            return tensor.permute(1, 0, 2, 3).reshape(int(tensor.shape[1]), int(tensor.shape[0]), -1).contiguous()
        if tensor.ndim == 3:
            return tensor.permute(1, 0, 2).contiguous()
        if tensor.ndim == 2:
            return tensor.unsqueeze(1)
        raise ValueError(f'Unsupported Spikformer block output shape: {tuple(tensor.shape)}')

    def iter_named_layers(self):
        for block_index, block in enumerate(getattr(self.source_model, 'block', []), start=1):
            yield f'layer_{block_index:02d}', block
        yield 'spikformer_source', self.source_model

    def iter_named_hidden_layers(self):
        return ((f'layer_{index:02d}', block) for index, block in enumerate(getattr(self.source_model, 'block', []), start=1))


    def model_metadata(self) -> dict[str, Any]:
        dependency_backend = self.spikformer_dependency_backend
        paper_exact_runtime = dependency_backend == 'author_dependencies'
        payload = {
            'raw_model_token': self.spec.raw_token,
            'canonical_model_token': self.spec.canonical_token,
            'family': self.spec.family,
            'analysis_role': 'auxiliary_diversity',
            'main_analysis_target': 'dense_snn',
            'model_profile': 'spikformer',
            'paper_experiment': 'cifar10_dvs_neuromorphic_classification',
            'source_code_path': 'origin/spikformer/cifar10dvs/model.py',
            'source_factory_name': 'spikformer',
            'source_class_name': 'Spikformer',
            'source_train_entrypoint': 'origin/spikformer/cifar10dvs/train.py',
            'paper_setting': '2-256',
            'model_size': '2-256',
            'patch_size': 16,
            'in_channels': 2,
            'img_size': '128x128',
            'embed_dims': 256,
            'depths': 2,
            'num_heads': 16,
            'mlp_ratios': 4,
            'sr_ratios': 1,
            'time_steps': 16,
            'optimizer_family': 'Adam',
            'structure_variation': 'none' if paper_exact_runtime else 'dependency_stub_runtime_smoke_only',
            'paper_definition_compliance': 'author_source_with_real_dependencies' if paper_exact_runtime else 'author_source_imported_with_dependency_stubs; runtime is not paper-exact',
            'adapter_output_layout': '[B,T,2,128,128]',
            'source_input_layout': '[B,T,2,128,128]',
            'internal_input_layout': '[T,B,2,128,128]',
            'official_capture_family': ['x_probe'],
            'static_or_non_dvs_adapter': 'two_channel_frame_construction_when_needed',
            'dependency_backend': self.spikformer_dependency_backend,
            'compile_runtime_disabled': bool(getattr(self, '_spikformer_source_compiled_runtime_disabled', False)),
            'compile_runtime_error': getattr(self, '_spikformer_source_compiled_runtime_error', None),
        }
        payload.update(getattr(self, 'extra_metadata', {}))
        return payload

    def forward(self, input_sequence: torch.Tensor, *, capture_hidden: bool = False):
        frames = _adapt_to_cifar10dvs_frames(input_sequence)
        self._spikformer_block_outputs = []
        self._capture_hooks_enabled = bool(capture_hidden)
        try:
            logits = self._run_source_model(frames, capture_hidden=bool(capture_hidden))
        finally:
            self._capture_hooks_enabled = False
        hidden_records: list[Any] = []
        if capture_hidden:
            for block_index, tensor in enumerate(self._spikformer_block_outputs, start=1):
                feature = self._block_output_to_batch_time_feature(tensor).to(device=logits.device, dtype=logits.dtype)
                record = self._layer_record_cls(layer_name=f'layer_{block_index:02d}', membrane=feature, spike=feature, layer_input=feature)
                setattr(record, 'signal_kind', 'feature')
                setattr(record, 'series', 'block_output')
                hidden_records.append(record)
        membrane = logits.unsqueeze(1).contiguous()
        spike = (membrane > 0).to(dtype=membrane.dtype)
        output_record = self._layer_record_cls(layer_name='output', membrane=membrane, spike=spike, layer_input=membrane)
        return self._forward_result_cls(hidden_records=hidden_records, output_record=output_record, input_record=frames)


def build_spikformer_author_classifier(**kwargs: Any) -> SpikformerAuthorClassifier:
    return SpikformerAuthorClassifier(**kwargs)
