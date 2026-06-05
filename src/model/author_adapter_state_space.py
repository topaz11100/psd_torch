"""SpikingSSM author-source adapter for the auxiliary diversity profile."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn


_STATE_SPACE_IMPORT_SHIMS: list[str] = []


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_package(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = module
    if not hasattr(module, '__path__'):
        module.__path__ = []  # type: ignore[attr-defined]
    return module  # type: ignore[return-value]


def _load_origin_module(module_name: str, file_path: Path) -> types.ModuleType:
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing  # type: ignore[return-value]
    if not file_path.exists():
        raise RuntimeError(f'Official SpikingSSM dependency source is missing: {file_path}')
    module_spec = importlib.util.spec_from_file_location(module_name, file_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f'Could not create import spec for official SpikingSSM dependency: {file_path}')
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    return module  # type: ignore[return-value]


def _install_spiking_ssm_origin_import_shims() -> None:
    """Expose checked-in SpikingSSM origin modules under their released import names.

    The checked-in ``ss4d.py`` uses the released absolute imports
    ``src.models.*``. The project package is also named ``src``, so the adapter
    installs only the missing submodules instead of editing the author source.
    """

    global _STATE_SPACE_IMPORT_SHIMS
    origin_root = _project_root() / 'Origin' / 'state_space_sd4'

    try:
        import src as project_src  # type: ignore
    except Exception as exc:  # pragma: no cover - package import failure is fatal.
        raise RuntimeError('Could not import project package src before installing SpikingSSM origin shims.') from exc

    models_pkg = _ensure_package('src.models')
    setattr(project_src, 'models', models_pkg)

    # The released ss4d.py depends on DropoutNd from src.models.nn, but that file
    # is not included in the checked-in origin snapshot. This shim preserves the
    # expected API and applies dropout tied over the temporal axis for [B,C,L]
    # tensors, which is the convention used by the S4/SpikingSSM block.
    if 'src.models.nn' not in sys.modules:
        nn_mod = types.ModuleType('src.models.nn')

        class DropoutNd(nn.Module):
            def __init__(self, p: float = 0.0, tie: bool = True, transposed: bool = True) -> None:
                super().__init__()
                if p < 0.0 or p > 1.0:
                    raise ValueError(f'DropoutNd probability has to be between 0 and 1, got {p}.')
                self.p = float(p)
                self.tie = bool(tie)
                self.transposed = bool(transposed)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if not self.training or self.p == 0.0:
                    return x
                if self.p == 1.0:
                    return torch.zeros_like(x)
                shape = list(x.shape)
                if self.tie and len(shape) >= 3:
                    # ss4d.py runs in transposed [B,C,L] layout, so share a mask
                    # across the sequence dimension as the released S4 DropoutNd
                    # family does.
                    shape[-1] = 1
                mask = x.new_empty(shape).bernoulli_(1.0 - self.p).div_(1.0 - self.p)
                return x * mask

        nn_mod.DropoutNd = DropoutNd
        sys.modules['src.models.nn'] = nn_mod
        setattr(models_pkg, 'nn', nn_mod)
        _STATE_SPACE_IMPORT_SHIMS.append('src.models.nn.DropoutNd')
    else:
        setattr(models_pkg, 'nn', sys.modules['src.models.nn'])

    spike_pkg = _ensure_package('src.models.spike')
    setattr(models_pkg, 'spike', spike_pkg)
    surrogate_mod = _load_origin_module(
        'src.models.spike.surrogate',
        origin_root / 'src' / 'models' / 'spike' / 'surrogate.py',
    )
    setattr(spike_pkg, 'surrogate', surrogate_mod)
    neuron_mod = _load_origin_module(
        'src.models.spike.neuron',
        origin_root / 'src' / 'models' / 'spike' / 'neuron.py',
    )
    setattr(spike_pkg, 'neuron', neuron_mod)

    sequence_pkg = _ensure_package('src.models.sequence')
    kernels_pkg = _ensure_package('src.models.sequence.kernels')
    setattr(models_pkg, 'sequence', sequence_pkg)
    setattr(sequence_pkg, 'kernels', kernels_pkg)

    # ss4d.py imports SSMKernelDiag even though the current profile uses
    # trainable_B=False and therefore S4DKernel. Keep this import resolvable but
    # fail loudly if a future config asks for the missing trainable-B backend.
    if 'src.models.sequence.kernels.ssm' not in sys.modules:
        ssm_mod = types.ModuleType('src.models.sequence.kernels.ssm')

        class SSMKernelDiag(nn.Module):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__()
                raise RuntimeError(
                    'SpikingSSM trainable_B=True requires the official SSMKernelDiag source, '
                    'which is not present in the checked-in origin snapshot.'
                )

        ssm_mod.SSMKernelDiag = SSMKernelDiag
        sys.modules['src.models.sequence.kernels.ssm'] = ssm_mod
        setattr(kernels_pkg, 'ssm', ssm_mod)
        _STATE_SPACE_IMPORT_SHIMS.append('src.models.sequence.kernels.ssm.SSMKernelDiag_import_stub')
    else:
        setattr(kernels_pkg, 'ssm', sys.modules['src.models.sequence.kernels.ssm'])


def _load_origin_spiking_ssm_class() -> type[nn.Module]:
    source_path = _project_root() / 'Origin' / 'state_space_sd4' / 'models' / 'spike' / 'ss4d.py'
    if not source_path.exists():
        raise RuntimeError(f'Official SpikingSSM source is missing: {source_path}')
    module_spec = importlib.util.spec_from_file_location('_psd_origin_state_space_ss4d', source_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f'Could not create import spec for official SpikingSSM source: {source_path}')
    _install_spiking_ssm_origin_import_shims()
    module = importlib.util.module_from_spec(module_spec)
    try:
        module_spec.loader.exec_module(module)
    except Exception as exc:
        raise RuntimeError(
            'Could not import Origin/state_space_sd4/models/spike/ss4d.py. '
            'Install the complete author-code dependencies used by the official SpikingSSM profile.'
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
            'source_config_note': 'configuration metadata is supplied by root YAML config',
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
            'structure_variation': 'origin_core_with_project_input_head_adapters',
            'adapter_is_model_variation': False,
            'paper_definition_scope': 'checked-in origin SpikingSSM/SS4D core with project input and classifier adapters',
            'origin_import_shims': list(dict.fromkeys(_STATE_SPACE_IMPORT_SHIMS)),
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
