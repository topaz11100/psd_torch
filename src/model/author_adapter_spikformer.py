"""Spikformer author-source adapter for the auxiliary diversity profile."""

from __future__ import annotations

import importlib.util
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_origin_spikformer_class() -> type[nn.Module]:
    source_path = _project_root() / 'Origin' / 'spikformer' / 'cifar10dvs' / 'model.py'
    if not source_path.exists():
        raise RuntimeError(f'Official Spikformer source is missing: {source_path}')
    module_spec = importlib.util.spec_from_file_location('_psd_origin_spikformer_cifar10dvs_model', source_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f'Could not create import spec for official Spikformer source: {source_path}')
    module = importlib.util.module_from_spec(module_spec)
    try:
        module_spec.loader.exec_module(module)
    except Exception as exc:
        raise RuntimeError(
            'Could not import Origin/spikformer/cifar10dvs/model.py. '
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
    """Wrapper around Origin/spikformer/cifar10dvs/model.py::Spikformer."""

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
        for block_index, block in enumerate(getattr(self.source_model, 'block', []), start=1):
            block.register_forward_hook(self._make_block_hook(block_index))

    def _make_block_hook(self, block_index: int):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if isinstance(output, torch.Tensor):
                self._spikformer_block_outputs.append(output)
        return hook

    @staticmethod
    def _block_output_to_batch_time_feature(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 4:
            # Origin Block returns (T,B,C,N). PSD analysis expects (B,T,features).
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
        return {
            'raw_model_token': self.spec.raw_token,
            'canonical_model_token': self.spec.canonical_token,
            'family': self.spec.family,
            'analysis_role': 'auxiliary_diversity',
            'main_analysis_target': 'dense_snn',
            'model_profile': 'spikformer',
            'paper_experiment': 'cifar10_dvs_neuromorphic_classification',
            'source_code_path': 'Origin/spikformer/cifar10dvs/model.py',
            'source_factory_name': 'spikformer',
            'source_class_name': 'Spikformer',
            'source_train_entrypoint': 'Origin/spikformer/cifar10dvs/train.py',
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
            'structure_variation': 'none',
            'adapter_output_layout': '[B,T,2,128,128]',
            'source_input_layout': '[B,T,2,128,128]',
            'internal_input_layout': '[T,B,2,128,128]',
            'official_capture_family': ['x_probe'],
            'static_or_non_dvs_adapter': 'two_channel_frame_construction_when_needed',
        }

    def forward(self, input_sequence: torch.Tensor, *, capture_hidden: bool = False):
        frames = _adapt_to_cifar10dvs_frames(input_sequence)
        self._spikformer_block_outputs = []
        logits = self.source_model(frames)
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
