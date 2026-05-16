"""Hidden-layer and fixed-CNN backbone resolution for official experiment CLIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.model.model_registry import ModelSpec


@dataclass(frozen=True)
class DenseLayerSpec:
    """One dense hidden layer width specification."""

    width: int
    kind: str = 'dense'


@dataclass(frozen=True)
class ConvLayerSpec:
    """One fixed-backbone 2-D convolutional hidden layer specification."""

    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    pool_after: bool = False
    pool_kernel_size: int = 2
    pool_stride: int = 2
    pool_padding: int = 0
    pool_ceil_mode: bool = False
    batch_norm: bool = False
    bias: bool = False
    kind: str = 'conv'


@dataclass(frozen=True)
class ResidualBlockSpec:
    """One ResNet-18 BasicBlock specification."""

    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    batch_norm: bool = True
    kind: str = 'residual_block'


LayerSpec = DenseLayerSpec | ConvLayerSpec | ResidualBlockSpec


_CNN_FAMILIES = {'cnn_lif', 'cnn_rf'}
_NULL_HIDDEN_SPEC_TOKENS = {'', '-', 'default', 'none', 'null'}


_FIXED_CNN_BACKBONES: dict[str, tuple[LayerSpec, ...]] = {
    # Official CNN experiments use canonical 2-D VGG-11/ResNet-18 backbones and
    # only swap the spiking neuron family attached to convolutional currents.
    'vgg11': (
        ConvLayerSpec(out_channels=64, kernel_size=3, stride=1, padding=1, pool_after=True, bias=True),
        ConvLayerSpec(out_channels=128, kernel_size=3, stride=1, padding=1, pool_after=True, bias=True),
        ConvLayerSpec(out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
        ConvLayerSpec(out_channels=256, kernel_size=3, stride=1, padding=1, pool_after=True, bias=True),
        ConvLayerSpec(out_channels=512, kernel_size=3, stride=1, padding=1, bias=True),
        ConvLayerSpec(out_channels=512, kernel_size=3, stride=1, padding=1, pool_after=True, bias=True),
        ConvLayerSpec(out_channels=512, kernel_size=3, stride=1, padding=1, bias=True),
        ConvLayerSpec(out_channels=512, kernel_size=3, stride=1, padding=1, pool_after=True, bias=True),
    ),
    'resnet18': (
        ConvLayerSpec(out_channels=64, kernel_size=7, stride=2, padding=3, pool_after=True, pool_kernel_size=3, pool_stride=2, pool_padding=1, batch_norm=True, bias=False),
        ResidualBlockSpec(out_channels=64, kernel_size=3, stride=1, padding=1),
        ResidualBlockSpec(out_channels=64, kernel_size=3, stride=1, padding=1),
        ResidualBlockSpec(out_channels=128, kernel_size=3, stride=2, padding=1),
        ResidualBlockSpec(out_channels=128, kernel_size=3, stride=1, padding=1),
        ResidualBlockSpec(out_channels=256, kernel_size=3, stride=2, padding=1),
        ResidualBlockSpec(out_channels=256, kernel_size=3, stride=1, padding=1),
        ResidualBlockSpec(out_channels=512, kernel_size=3, stride=2, padding=1),
        ResidualBlockSpec(out_channels=512, kernel_size=3, stride=1, padding=1),
    ),
}


def _validate_positive(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f'{name} must be positive; got {value}.')
    return value


def _is_null_hidden_spec(value: str | Sequence[str] | None) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _NULL_HIDDEN_SPEC_TOKENS
    tokens = [str(token).strip().lower() for token in value if str(token).strip() != '']
    return not tokens or (len(tokens) == 1 and tokens[0] in _NULL_HIDDEN_SPEC_TOKENS)


def parse_hidden_spec_tokens(tokens: Sequence[str]) -> list[DenseLayerSpec]:
    """Parse one tokenized dense hidden spec into per-layer widths."""

    resolved: list[DenseLayerSpec] = []
    for raw_token in tokens:
        token = str(raw_token).strip()
        if token == '':
            continue
        if not token.isdigit():
            raise ValueError(
                'Dense hidden specs must be comma-delimited positive integers, e.g. 64,32,20. '
                f'Got {raw_token!r}.'
            )
        resolved.append(DenseLayerSpec(width=_validate_positive('hidden width', int(token))))
    if not resolved:
        raise ValueError('hidden_spec must contain at least one dense layer width.')
    return resolved


def parse_hidden_spec(spec_text: str | Sequence[str] | None) -> list[DenseLayerSpec] | None:
    """Parse one comma-delimited dense hidden spec string or token list."""

    if _is_null_hidden_spec(spec_text):
        return None
    if isinstance(spec_text, str):
        tokens = [token.strip() for token in spec_text.split(',') if token.strip() != '']
    else:
        tokens = [str(token).strip() for token in spec_text if str(token).strip() != '']
    if not tokens:
        return None
    return parse_hidden_spec_tokens(tokens)


def default_arch_spec_from_hidden_sizes(model_spec: ModelSpec, hidden_sizes: Sequence[int]) -> list[LayerSpec]:
    """Lift dense hidden-size lists into layer specs."""

    if model_spec.family in _CNN_FAMILIES:
        raise ValueError('Fixed CNN backbones do not accept dense hidden-size lists; use hidden_spec=- in dataset slots.')
    widths = [_validate_positive('hidden width', int(width)) for width in hidden_sizes]
    if not widths:
        raise ValueError('At least one hidden layer width is required.')
    return [DenseLayerSpec(width=width) for width in widths]


def fixed_cnn_backbone_specs(backbone: str | None) -> list[LayerSpec]:
    """Return a copy of the fixed convolutional stack selected by a CNN model token."""

    name = '' if backbone is None else str(backbone).strip().lower()
    if name not in _FIXED_CNN_BACKBONES:
        allowed = ', '.join(sorted(_FIXED_CNN_BACKBONES))
        raise ValueError(f'Unsupported fixed CNN backbone {backbone!r}. Allowed: {allowed}.')
    return list(_FIXED_CNN_BACKBONES[name])


def resolve_arch_spec(
    *,
    model_spec: ModelSpec,
    arch_spec_text: str | Sequence[str] | None,
    hidden_sizes: Sequence[int] | None,
) -> list[LayerSpec]:
    """Resolve the effective hidden layer spec list from dataset-slot text or defaults."""

    if model_spec.family in _CNN_FAMILIES:
        if not _is_null_hidden_spec(arch_spec_text):
            raise ValueError(
                f'Model token {model_spec.raw_token!r} selects fixed CNN backbone {model_spec.backbone!r}; '
                'the dataset-slot hidden_spec field must be empty/default/- for fixed-CNN runs.'
            )
        if hidden_sizes not in (None, [], ()):  # caller should normally avoid passing dataset defaults here
            raise ValueError('Fixed CNN backbones do not consume dense hidden sizes.')
        return fixed_cnn_backbone_specs(model_spec.backbone)

    explicit = parse_hidden_spec(arch_spec_text)
    if explicit is None:
        if hidden_sizes is None:
            raise ValueError('One of hidden_spec or hidden_sizes must be provided for dense SNN model families.')
        explicit = default_arch_spec_from_hidden_sizes(model_spec, hidden_sizes)
    validate_arch_spec_for_model(model_spec, explicit)
    return explicit


def validate_arch_spec_for_model(model_spec: ModelSpec, layer_specs: Sequence[LayerSpec]) -> None:
    """Ensure the chosen layer specs match the model-family semantics."""

    if not layer_specs:
        raise ValueError('At least one hidden layer spec is required.')
    if model_spec.family in _CNN_FAMILIES:
        if any(isinstance(spec, DenseLayerSpec) for spec in layer_specs):
            raise ValueError(f'Model family {model_spec.family!r} requires a fixed CNN backbone, not dense hidden widths.')
        return
    if any(not isinstance(spec, DenseLayerSpec) for spec in layer_specs):
        raise ValueError(f'Model family {model_spec.family!r} requires dense hidden width specs.')


def arch_hidden_sizes(layer_specs: Sequence[LayerSpec]) -> list[int]:
    """Return one width/channel list for configs/manifests from resolved layer specs."""

    sizes: list[int] = []
    for spec in layer_specs:
        if isinstance(spec, DenseLayerSpec):
            sizes.append(int(spec.width))
        else:
            sizes.append(int(spec.out_channels))
    return sizes


def serialize_arch_spec(layer_specs: Sequence[LayerSpec]) -> str:
    """Serialize resolved specs into a compact manifest string."""

    tokens: list[str] = []
    for spec in layer_specs:
        if isinstance(spec, DenseLayerSpec):
            tokens.append(str(int(spec.width)))
        elif isinstance(spec, ResidualBlockSpec):
            tokens.append(
                'basicblock('
                f'channels={int(spec.out_channels)},'
                f'kernel={int(spec.kernel_size)},'
                f'stride={int(spec.stride)},'
                f'padding={int(spec.padding)}'
                ')'
            )
        else:
            tokens.append(
                'conv2d('
                f'channels={int(spec.out_channels)},'
                f'kernel={int(spec.kernel_size)},'
                f'stride={int(spec.stride)},'
                f'padding={int(spec.padding)},'
                f'pool_after={bool(spec.pool_after)}'
                ')'
            )
    return ','.join(tokens)


def arch_spec_payload(layer_specs: Sequence[LayerSpec]) -> list[dict[str, int | str | bool]]:
    """Return one JSON-serializable layer-spec payload."""

    payload: list[dict[str, int | str | bool]] = []
    for spec in layer_specs:
        if isinstance(spec, DenseLayerSpec):
            payload.append({'kind': 'dense', 'width': int(spec.width)})
        elif isinstance(spec, ResidualBlockSpec):
            payload.append(
                {
                    'kind': 'basic_block',
                    'out_channels': int(spec.out_channels),
                    'kernel_size': int(spec.kernel_size),
                    'stride': int(spec.stride),
                    'padding': int(spec.padding),
                    'batch_norm': bool(spec.batch_norm),
                }
            )
        else:
            payload.append(
                {
                    'kind': 'conv2d',
                    'out_channels': int(spec.out_channels),
                    'kernel_size': int(spec.kernel_size),
                    'stride': int(spec.stride),
                    'padding': int(spec.padding),
                    'pool_after': bool(spec.pool_after),
                    'pool_kernel_size': int(spec.pool_kernel_size),
                    'pool_stride': int(spec.pool_stride),
                    'pool_padding': int(spec.pool_padding),
                    'pool_ceil_mode': bool(spec.pool_ceil_mode),
                    'batch_norm': bool(spec.batch_norm),
                    'bias': bool(spec.bias),
                }
            )
    return payload


def infer_output_sequence_length(input_length: int, layer_specs: Sequence[LayerSpec]) -> int:
    """Infer the final time-axis length after dense or fixed-CNN layer specs."""

    length = _validate_positive('input sequence length', int(input_length))
    for spec in layer_specs:
        if isinstance(spec, DenseLayerSpec):
            continue
        # 2-D fixed CNN backbones keep the explicit temporal axis. Static images
        # are handled by the CNN builder as one frame, not as raster time.
        continue
    return int(length)


__all__ = [
    'ConvLayerSpec',
    'DenseLayerSpec',
    'LayerSpec',
    'ResidualBlockSpec',
    'arch_hidden_sizes',
    'arch_spec_payload',
    'default_arch_spec_from_hidden_sizes',
    'fixed_cnn_backbone_specs',
    'infer_output_sequence_length',
    'parse_hidden_spec',
    'parse_hidden_spec_tokens',
    'resolve_arch_spec',
    'serialize_arch_spec',
    'validate_arch_spec_for_model',
]
