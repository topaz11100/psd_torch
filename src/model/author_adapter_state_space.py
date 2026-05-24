"""SpikingSSM author-source adapter for the auxiliary diversity profile."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_origin_spiking_ssm_class() -> type[nn.Module]:
    source_path = _project_root() / 'Origin' / 'state_space_sd4' / 'models' / 'spike' / 'ss4d.py'
    if not source_path.exists():
        raise RuntimeError(f'Official SpikingSSM source is missing: {source_path}')
    module_spec = importlib.util.spec_from_file_location('_psd_origin_state_space_ss4d', source_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f'Could not create import spec for official SpikingSSM source: {source_path}')
    module = importlib.util.module_from_spec(module_spec)
    try:
        module_spec.loader.exec_module(module)
    except Exception as exc:
        raise RuntimeError(
            'Could not import Origin/state_space_sd4/models/spike/ss4d.py. '
            'Install the author-code and external S4 dependencies used by the official SpikingSSM profile.'
        ) from exc
    cls = getattr(module, 'SpikingSSM', None)
    if cls is None:
        raise RuntimeError('Official SpikingSSM source did not expose class SpikingSSM.')
    return cls


class SpikingSSMAuthorClassifier(nn.Module):
    """Wrapper around Origin/state_space_sd4/models/spike/ss4d.py::SpikingSSM."""

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
        del input_shape
        self.spec = spec
        self.input_dim = int(input_dim)
        self.sequence_length = int(sequence_length)
        self.output_sequence_length = int(sequence_length)
        self.num_classes = int(num_classes)
        self.d_model = 400
        self.d_state = 64
        self._layer_record_cls = layer_record_cls
        self._forward_result_cls = forward_result_cls
        origin_cls = _load_origin_spiking_ssm_class()
        self.input_adapter = nn.Linear(self.input_dim, self.d_model)
        self.source_model = origin_cls(
            d_model=self.d_model,
            n_layers=2,
            dropout=0.1,
            prenorm=False,
            layer={'d_state': self.d_state, 'lr': 0.001},
        )
        self.classifier_head = nn.Linear(self.d_model, self.num_classes)

    def iter_named_layers(self):
        yield 'input_adapter', self.input_adapter
        yield 'spikingssm_source', self.source_model
        yield 'output', self.classifier_head

    def iter_named_hidden_layers(self):
        yield 'spikingssm_source', self.source_model


    def model_metadata(self) -> dict[str, Any]:
        return {
            'raw_model_token': self.spec.raw_token,
            'canonical_model_token': self.spec.canonical_token,
            'family': self.spec.family,
            'analysis_role': 'auxiliary_diversity',
            'main_analysis_target': 'dense_snn',
            'model_profile': 'spikingssm',
            'paper_experiment': 'sequential_mnist_classification',
            'source_code_path': 'Origin/state_space_sd4/models/spike/ss4d.py',
            'source_neuron_path': 'Origin/state_space_sd4/src/models/spike/neuron.py',
            'source_config_note': 'configuration metadata is supplied by root JSON config',
            'source_train_entrypoint': 'Origin/state_space_sd4/train.py',
            'source_class_name': 'SpikingSSM',
            'origin_block_name': 'SS4D',
            'n_layers': 2,
            'd_model': self.d_model,
            'd_state': self.d_state,
            'bidirectional': False,
            'prenorm': False,
            'dropout': 0.1,
            'optimizer_family': 'Adam',
            'structure_variation': 'none',
            'adapter_is_model_variation': False,
            'adapter_output_layout': '[B,L,400]',
            'source_input_layout': '[B,L,D]',
            'internal_input_layout': '[B,D,L]',
            'trace_families': ['x_probe', 'z_front', 'x_layer', 'i_state', 'z_state', 'y_mem', 'y_spike'],
        }

    def forward(self, input_sequence: torch.Tensor, *, capture_hidden: bool = False):
        if input_sequence.ndim != 3:
            raise ValueError(f'SpikingSSM expects [B,L,D] input after dataset adaptation, got {tuple(input_sequence.shape)}.')
        projected = self.input_adapter(input_sequence.to(dtype=torch.float32))
        source_output, _state = self.source_model(projected)
        logits_seq = self.classifier_head(source_output)
        output_spike = (logits_seq > 0).to(dtype=logits_seq.dtype)
        hidden_records = []
        if capture_hidden:
            hidden_records.append(
                self._layer_record_cls(
                    layer_name='spikingssm_source',
                    membrane=source_output,
                    spike=torch.zeros_like(source_output),
                    layer_input=projected,
                )
            )
        output_record = self._layer_record_cls(layer_name='output', membrane=logits_seq, spike=output_spike, layer_input=source_output)
        return self._forward_result_cls(hidden_records=hidden_records, output_record=output_record, input_record=input_sequence)


def build_state_space_author_classifier(**kwargs: Any) -> SpikingSSMAuthorClassifier:
    return SpikingSSMAuthorClassifier(**kwargs)
