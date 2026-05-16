from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import functional, neuron, surrogate

from src.neurons._common import surrogate_spike, trim_open_interval


class TDBatchNorm(nn.Module):
    def __init__(self, num_features, init_threshold=1.0, momentum=0.1, epsilon=1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon

        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        self.register_buffer("threshold", torch.ones(num_features) * init_threshold)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.gamma / self.threshold
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            weight,
            self.beta,
            self.training,
            self.momentum,
            self.epsilon,
        )


def _resolve_lif_v_reset(reset_mode: str | None):
    if reset_mode == "hard_reset":
        return 0.0
    if reset_mode in (None, "soft_reset"):
        return None
    if reset_mode == "no_reset":
        raise ValueError("snn-bench LIFNode does not support no_reset.")
    raise ValueError(f"Unsupported LIF reset_mode: {reset_mode!r}")


def make_snnbench_lif_node(
    *,
    channels: int | None = None,
    v_threshold: float = 1.0,
    reset_mode: str | None = None,
    **_kwargs,
):
    """Build the original snn-bench LIF activation node.

    ``channels`` is accepted for interface compatibility with RF nodes.
    """

    del channels
    node = neuron.LIFNode(
        tau=2.0,
        v_threshold=float(v_threshold),
        v_reset=_resolve_lif_v_reset(reset_mode),
        surrogate_function=surrogate.ATan(),
    )
    setattr(node, "is_snnbench_spiking_node", True)
    setattr(node, "snnbench_neuron_tag", "lif")
    return node


def _positive_threshold_init(v_threshold: float, size: int, *, eps: float) -> torch.Tensor:
    value = max(float(v_threshold) - float(eps), float(eps))
    raw = math.log(math.expm1(value))
    return torch.full((int(size),), float(raw), dtype=torch.float32)


class SNNBenchRFNode(nn.Module):
    """Single-step RF activation node for snn-bench CNN backbones.

    The snn-bench models already contain Conv/BN/Linear modules.  Therefore this
    node contains only the RF dynamics that follow the current-producing module.
    """

    is_snnbench_spiking_node = True
    snnbench_neuron_tag = "rf"

    def __init__(
        self,
        *,
        channels: int,
        v_threshold: float = 1.0,
        reset_mode: str | None = None,
        trainable_threshold: bool = False,
        emit_spike: bool = True,
        reset_enabled: bool = True,
        damping_magnitude_bounds: tuple[float, float] = (0.1, 1.0),
        **_kwargs,
    ) -> None:
        super().__init__()
        if reset_mode is None:
            reset_mode = "soft_reset"
        if reset_mode not in {"soft_reset", "hard_reset", "no_reset"}:
            raise ValueError(f"Unsupported RF reset_mode: {reset_mode!r}")

        self.channels = int(channels)
        self.trainable_threshold = bool(trainable_threshold)
        self.threshold_eps = 1.0e-6
        self.emit_spike = bool(emit_spike)
        self.reset_mode = str(reset_mode)
        self.reset_enabled = bool(reset_enabled) and self.reset_mode != "no_reset"
        self.damping_lower = float(damping_magnitude_bounds[0])
        self.damping_upper = float(damping_magnitude_bounds[1])

        self.register_buffer("freq_lower", torch.zeros(self.channels, dtype=torch.float32))
        self.register_buffer("freq_upper", torch.full((self.channels,), 0.5, dtype=torch.float32))
        self.freq_raw = nn.Parameter(torch.empty(self.channels))
        self.damping_raw = nn.Parameter(torch.empty(self.channels))

        threshold_init = torch.full((self.channels,), float(v_threshold), dtype=torch.float32)
        if self.trainable_threshold:
            self.v_threshold_param = nn.Parameter(
                _positive_threshold_init(v_threshold, self.channels, eps=self.threshold_eps)
            )
        else:
            self.register_buffer("v_threshold_buffer", threshold_init)
            self.register_parameter("v_threshold_param", None)

        self.reset_parameters()
        self.reset()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            freq = torch.empty_like(self.freq_lower)
            damping = torch.empty_like(self.damping_raw)
            for index in range(freq.numel()):
                left, right = trim_open_interval(float(self.freq_lower[index]), float(self.freq_upper[index]))
                freq[index] = 0.5 * (left + right) if right <= left else float(torch.empty(1).uniform_(left, right).item())

                dleft, dright = trim_open_interval(self.damping_lower, self.damping_upper)
                damping[index] = 0.5 * (dleft + dright) if dright <= dleft else float(torch.empty(1).uniform_(dleft, dright).item())

            freq01 = torch.clamp(freq / 0.5, min=1.0e-6, max=1.0 - 1.0e-6)
            damp01 = torch.clamp(
                (damping - self.damping_lower) / (self.damping_upper - self.damping_lower),
                min=1.0e-6,
                max=1.0 - 1.0e-6,
            )
            self.freq_raw.copy_(torch.log(freq01) - torch.log1p(-freq01))
            self.damping_raw.copy_(torch.log(damp01) - torch.log1p(-damp01))

    def reset(self) -> None:
        self.x_post = None
        self.y_post = None
        self.v = None

    def effective_frequency(self) -> torch.Tensor:
        sigma = torch.sigmoid(self.freq_raw)
        return self.freq_lower + (self.freq_upper - self.freq_lower) * sigma

    def effective_damping_magnitude(self) -> torch.Tensor:
        sigma = torch.sigmoid(self.damping_raw)
        return self.damping_lower + (self.damping_upper - self.damping_lower) * sigma

    def effective_b(self) -> torch.Tensor:
        return -self.effective_damping_magnitude()

    def effective_omega(self) -> torch.Tensor:
        return 2.0 * math.pi * self.effective_frequency()

    def effective_threshold(self) -> torch.Tensor:
        if self.v_threshold_param is not None:
            return F.softplus(self.v_threshold_param) + float(self.threshold_eps)
        return self.v_threshold_buffer

    def rho(self) -> torch.Tensor:
        return torch.exp(self.effective_b())

    def f_cyc_per_sample(self) -> torch.Tensor:
        return self.effective_frequency()

    def _channel_view(self, current: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        view_shape = [1, self.channels] + [1] * (current.ndim - 2)
        return value.to(device=current.device, dtype=current.dtype).view(*view_shape)

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        if current.ndim < 2:
            raise ValueError(f"SNNBenchRFNode expects at least rank-2 input, got {tuple(current.shape)}")
        if int(current.shape[1]) != self.channels:
            raise ValueError(f"SNNBenchRFNode expected channels={self.channels}, got shape {tuple(current.shape)}")

        if self.x_post is None or tuple(self.x_post.shape) != tuple(current.shape):
            self.x_post = torch.zeros_like(current)
            self.y_post = torch.zeros_like(current)

        b = self.effective_b().to(device=current.device, dtype=current.dtype)
        omega = self.effective_omega().to(device=current.device, dtype=current.dtype)
        rho = torch.exp(b)
        cos_phi = torch.cos(omega)
        sin_phi = torch.sin(omega)

        complex_den = torch.complex(b, omega)
        alpha_complex = torch.exp(torch.complex(b, omega))
        beta_complex = (alpha_complex - 1.0) / complex_den

        rho_view = self._channel_view(current, rho)
        cos_view = self._channel_view(current, cos_phi)
        sin_view = self._channel_view(current, sin_phi)
        beta_x_view = self._channel_view(current, beta_complex.real)
        beta_y_view = self._channel_view(current, beta_complex.imag)
        threshold_view = self._channel_view(current, self.effective_threshold())

        x_pre = rho_view * (cos_view * self.x_post - sin_view * self.y_post) + beta_x_view * current
        y_pre = rho_view * (sin_view * self.x_post + cos_view * self.y_post) + beta_y_view * current
        membrane_signal = x_pre - threshold_view
        spike = surrogate_spike(membrane_signal) if self.emit_spike else torch.zeros_like(membrane_signal)

        if self.reset_enabled:
            if self.reset_mode == "soft_reset":
                self.x_post = x_pre - threshold_view * spike
                self.y_post = y_pre
            else:
                keep = 1.0 - spike
                self.x_post = x_pre * keep
                self.y_post = y_pre * keep
        else:
            self.x_post = x_pre
            self.y_post = y_pre

        record_raw_membrane = (not self.emit_spike) and (not self.reset_enabled)
        self.v = x_pre if record_raw_membrane else membrane_signal
        return spike

    def filter_stats_vectors(self) -> dict[str, torch.Tensor]:
        return {
            "damping": self.effective_damping_magnitude().detach(),
            "center_frequency": self.f_cyc_per_sample().detach(),
        }


def make_snnbench_rf_node(
    *,
    channels: int,
    v_threshold: float = 1.0,
    reset_mode: str | None = None,
    trainable_threshold: bool = False,
    **kwargs,
):
    return SNNBenchRFNode(
        channels=int(channels),
        v_threshold=float(v_threshold),
        reset_mode=reset_mode,
        trainable_threshold=bool(trainable_threshold),
        **kwargs,
    )

def get_vgg_cfg(depth):
    if depth == 7:
        return [64, "M", 128, "M", 256, "M"], 256
    elif depth == 11:
        return [64, "M", 128, "M", 256, 256, "M"], 256
    elif depth == 15:
        return [64, 64, "M", 128, 128, "M", 256, 256, 256, "M"], 256
    else:
        raise ValueError("vgg_depth must be one of {7,11,15}")


class VGG_SNN(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        depth=11,
        bias=True,
        v_threshold: float = 1.0,
        reset_mode: str | None = None,
        single_step_neuron=None,
        trainable_threshold: bool = False,
    ):
        super().__init__()

        if single_step_neuron is None:
            single_step_neuron = make_snnbench_lif_node

        try:
            functional.set_backend(self, backend="cupy")
        except Exception:
            pass

        cfg, last_c = get_vgg_cfg(depth)
        self.features, first_conv, first_lif, pool_count = self._make_layers(
            cfg,
            in_channels,
            bias,
            v_threshold=v_threshold,
            reset_mode=reset_mode,
            single_step_neuron=single_step_neuron,
            trainable_threshold=trainable_threshold,
        )

        self.first_conv = first_conv
        self.first_lif = first_lif

        self.fixpool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(last_c * 4 * 4, 512, bias=bias),
            single_step_neuron(
                channels=512,
                v_threshold=float(v_threshold),
                reset_mode=reset_mode,
                trainable_threshold=bool(trainable_threshold),
            ),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes, bias=bias),
            single_step_neuron(
                channels=int(num_classes),
                v_threshold=float(v_threshold),
                reset_mode=reset_mode,
                trainable_threshold=bool(trainable_threshold),
            ),
        )

    def _make_layers(
        self,
        cfg,
        in_ch,
        bias,
        *,
        v_threshold: float,
        reset_mode: str | None = None,
        single_step_neuron=None,
        trainable_threshold: bool = False,
    ):
        if single_step_neuron is None:
            single_step_neuron = make_snnbench_lif_node

        layers = []
        first_conv = None
        first_lif = None
        pool_count = 0
        cur_ch = in_ch

        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                pool_count += 1
            else:
                conv = nn.Conv2d(cur_ch, v, kernel_size=3, padding=1, bias=bias)
                bn = TDBatchNorm(v)
                sn = single_step_neuron(
                    channels=int(v),
                    v_threshold=float(v_threshold),
                    reset_mode=reset_mode,
                    trainable_threshold=bool(trainable_threshold),
                )
                layers += [conv, bn, sn]

                if first_conv is None:
                    first_conv = conv
                    first_lif = sn

                cur_ch = v

        return nn.Sequential(*layers), first_conv, first_lif, pool_count

    def forward(self, x):
        x = self.features(x)
        x = self.fixpool(x)
        x = self.classifier(x)
        return x
    
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        single_step_neuron=None,
        **kwargs,
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")

        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn1 = single_step_neuron(channels=int(planes), **kwargs)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.sn2 = single_step_neuron(channels=int(planes), **kwargs)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.sn2(out)

        return out
    
class SpikingResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=200,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        single_step_neuron=None,
        **kwargs,
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if single_step_neuron is None:
            single_step_neuron = make_snnbench_lif_node

        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple")

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = single_step_neuron(channels=int(self.inplanes), **kwargs)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], single_step_neuron=single_step_neuron, **kwargs)
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            single_step_neuron=single_step_neuron,
            **kwargs,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            single_step_neuron=single_step_neuron,
            **kwargs,
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            single_step_neuron=single_step_neuron,
            **kwargs,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, TDBatchNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
        dilate=False,
        single_step_neuron=None,
        **kwargs,
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                single_step_neuron,
                **kwargs,
            )
        )

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    single_step_neuron=single_step_neuron,
                    **kwargs,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def spiking_resnet18(
    pretrained=False,
    progress=True,
    norm_layer=None,
    single_step_neuron=None,
    num_classes=10,
    **kwargs,
):
    if pretrained:
        raise ValueError("pretrained=True is not supported in the local snn-bench adapter.")

    return SpikingResNet(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        norm_layer=norm_layer,
        single_step_neuron=single_step_neuron,
        **kwargs,
    )

