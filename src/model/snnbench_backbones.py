from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from src.neurons._common import surrogate_spike, trim_open_interval



class SafeMaxPool2d(nn.MaxPool2d):
    """Max-pool marker that never collapses prepared image tensors to zero area."""

    def _output_dim(self, value: int, axis: int) -> int:
        kernel = _pair(self.kernel_size)[axis]
        stride = _pair(self.stride if self.stride is not None else self.kernel_size)[axis]
        padding = _pair(self.padding)[axis]
        dilation = _pair(self.dilation)[axis]
        numerator = value + 2 * padding - dilation * (kernel - 1) - 1
        if self.ceil_mode:
            return math.floor((numerator + stride - 1) / stride + 1)
        return math.floor(numerator / stride + 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim >= 4:
            height = int(input.shape[-2])
            width = int(input.shape[-1])
            if self._output_dim(height, 0) <= 0 or self._output_dim(width, 1) <= 0:
                return input
        return super().forward(input)


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
        training = bool(self.training)
        if training and x.ndim >= 2:
            samples_per_channel = x.numel() // max(1, int(x.shape[1]))
            if samples_per_channel <= 1:
                # PyTorch BatchNorm cannot estimate variance from a singleton
                # channel slice. Fall back to the running-stat path so VGG/ResNet
                # topology remains usable for tiny smoke batches and 1x1 maps.
                training = False
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            weight,
            self.beta,
            training,
            self.momentum,
            self.epsilon,
        )


def _resolve_lif_v_reset(reset_mode: str | None):
    if reset_mode == "hard_reset":
        return 0.0
    if reset_mode in (None, "soft_reset"):
        return None
    if reset_mode == "no_reset":
        raise ValueError("SpikingJelly LIFNode does not support no_reset.")
    raise ValueError(f"Unsupported LIF reset_mode: {reset_mode!r}")


class TorchSingleStepLIFNode(nn.Module):
    """Single-step LIF node implemented in plain PyTorch for CNN backbones.

    The reference files provide topology only. Neuron state and compile
    behaviour are project-owned so prepared static images and event frames such
    as DVS-Gesture128 can be consumed without fixed reference-input assumptions.
    """

    is_snnbench_spiking_node = True
    snnbench_neuron_tag = "lif"

    def __init__(
        self,
        *,
        channels: int | None = None,
        v_threshold: float = 1.0,
        reset_mode: str | None = None,
        tau: float = 2.0,
        trainable_threshold: bool = False,
        emit_spike: bool = True,
        reset_enabled: bool = True,
        **_kwargs,
    ) -> None:
        super().__init__()
        if reset_mode is None:
            reset_mode = "soft_reset"
        if reset_mode not in {"soft_reset", "hard_reset"}:
            raise ValueError(f"TorchSingleStepLIFNode supports soft/hard reset, got {reset_mode!r}.")
        self.channels = None if channels is None else int(channels)
        self.v_threshold = float(v_threshold)
        self.reset_mode = str(reset_mode)
        self.tau = float(tau)
        self.alpha = float(1.0 - 1.0 / max(self.tau, 1.0e-6))
        self.trainable_threshold = bool(trainable_threshold)
        self.emit_spike = bool(emit_spike)
        self.reset_enabled = bool(reset_enabled)
        self.threshold_eps = 1.0e-6
        if self.channels is not None and self.trainable_threshold:
            self.v_threshold_param = nn.Parameter(
                _positive_threshold_init(self.v_threshold, self.channels, eps=self.threshold_eps)
            )
        else:
            self.register_parameter("v_threshold_param", None)
        self.reset()

    def reset(self) -> None:
        self.mem = None
        self.v = None

    def effective_threshold(self, current: torch.Tensor) -> torch.Tensor:
        if self.v_threshold_param is None:
            return torch.as_tensor(self.v_threshold, device=current.device, dtype=current.dtype)
        threshold = F.softplus(self.v_threshold_param) + float(self.threshold_eps)
        view_shape = [1, int(threshold.numel())] + [1] * (current.ndim - 2)
        return threshold.to(device=current.device, dtype=current.dtype).view(*view_shape)

    def threshold_stats_vector(self) -> torch.Tensor:
        if self.v_threshold_param is not None:
            return (F.softplus(self.v_threshold_param) + float(self.threshold_eps)).detach()
        device = self.v_threshold_param.device if self.v_threshold_param is not None else torch.device('cpu')
        count = 1 if self.channels is None else int(self.channels)
        return torch.full((count,), float(self.v_threshold), device=device, dtype=torch.float32)

    def filter_stats_vectors(self) -> dict[str, torch.Tensor]:
        return {
            "alpha": torch.full((1 if self.channels is None else int(self.channels),), float(self.alpha), dtype=torch.float32),
            "v_threshold": self.threshold_stats_vector(),
        }

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        if current.ndim < 2:
            raise ValueError(f"TorchSingleStepLIFNode expects rank >= 2 input, got {tuple(current.shape)}")
        if self.channels is not None and int(current.shape[1]) != int(self.channels):
            raise ValueError(f"TorchSingleStepLIFNode expected channels={self.channels}, got shape {tuple(current.shape)}")
        if self.mem is None or tuple(self.mem.shape) != tuple(current.shape):
            self.mem = torch.zeros_like(current)
        threshold = self.effective_threshold(current)
        mem_pre = self.alpha * self.mem + current
        membrane_signal = mem_pre - threshold
        spike = surrogate_spike(membrane_signal) if self.emit_spike else torch.zeros_like(membrane_signal)
        if self.reset_enabled:
            if self.reset_mode == "soft_reset":
                self.mem = mem_pre - threshold * spike
            else:
                self.mem = mem_pre * (1.0 - spike)
        else:
            self.mem = mem_pre
        self.v = membrane_signal
        return spike


def make_spikingjelly_lif_node(
    *,
    channels: int | None = None,
    v_threshold: float = 1.0,
    reset_mode: str | None = None,
    **_kwargs,
):
    """Build the legacy SpikingJelly LIF node when explicitly requested."""

    del channels
    try:
        from spikingjelly.activation_based import neuron, surrogate
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PSD_SNNBENCH_LIF_BACKEND=spikingjelly requires SpikingJelly to be installed.") from exc
    node = neuron.LIFNode(
        tau=2.0,
        v_threshold=float(v_threshold),
        v_reset=_resolve_lif_v_reset(reset_mode),
        surrogate_function=surrogate.ATan(),
    )
    setattr(node, "is_snnbench_spiking_node", True)
    setattr(node, "snnbench_neuron_tag", "lif")
    setattr(node, "psd_neuron_backend", "spikingjelly")
    return node


_SPIKINGJELLY_COMPILE_PROBE_CACHE: bool | None = None


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "")
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _spikingjelly_available() -> bool:
    try:
        from spikingjelly.activation_based import neuron as _neuron, surrogate as _surrogate  # noqa: F401
    except Exception:
        return False
    return True


def _spikingjelly_lif_compile_probe() -> bool:
    """Return whether the legacy SpikingJelly LIF node survives torch.compile.

    The probe is opt-in because even backend="eager" still invokes Dynamo and
    can add unpredictable cold-start latency in CPU/CI jobs.  The default path
    uses the project-owned pure-Torch node; set
    ``PSD_SNNBENCH_ENABLE_SPIKINGJELLY_COMPILE_PROBE=1`` to re-enable the probe.
    """

    global _SPIKINGJELLY_COMPILE_PROBE_CACHE
    if _SPIKINGJELLY_COMPILE_PROBE_CACHE is not None:
        return bool(_SPIKINGJELLY_COMPILE_PROBE_CACHE)
    if getattr(torch, "compile", None) is None:
        _SPIKINGJELLY_COMPILE_PROBE_CACHE = False
        return False
    if _env_truthy("PSD_SNNBENCH_SKIP_SPIKINGJELLY_COMPILE_PROBE") or not _env_truthy("PSD_SNNBENCH_ENABLE_SPIKINGJELLY_COMPILE_PROBE"):
        _SPIKINGJELLY_COMPILE_PROBE_CACHE = False
        return False
    try:
        node = make_spikingjelly_lif_node(channels=4, v_threshold=1.0, reset_mode="soft_reset")
        compiled = torch.compile(node, backend="eager")
        with torch.no_grad():
            compiled(torch.zeros(1, 4))
            compiled(torch.ones(1, 4))
    except Exception:
        _SPIKINGJELLY_COMPILE_PROBE_CACHE = False
    else:
        _SPIKINGJELLY_COMPILE_PROBE_CACHE = True
    return bool(_SPIKINGJELLY_COMPILE_PROBE_CACHE)


def _should_use_spikingjelly_lif_auto() -> bool:
    # The default CNN path uses the project-owned pure-Torch node. The legacy
    # SpikingJelly node is selected automatically only when the explicit opt-in
    # compile probe succeeds.
    return _spikingjelly_lif_compile_probe()


def make_snnbench_lif_node(
    *,
    channels: int | None = None,
    v_threshold: float = 1.0,
    reset_mode: str | None = None,
    trainable_threshold: bool = False,
    **kwargs,
):
    """Build the CNN LIF node.

    The default ``auto`` backend uses the project-owned pure-Torch LIF node.
    Setting ``PSD_SNNBENCH_ENABLE_SPIKINGJELLY_COMPILE_PROBE=1`` allows auto to
    preserve the SpikingJelly node when a tiny compile probe succeeds.
    ``PSD_SNNBENCH_LIF_BACKEND=torch`` forces the pure-Torch path;
    ``PSD_SNNBENCH_LIF_BACKEND=spikingjelly`` explicitly requires SpikingJelly.
    """

    backend = os.environ.get("PSD_SNNBENCH_LIF_BACKEND", "auto").strip().lower()
    if backend in {"spikingjelly", "sj"}:
        return make_spikingjelly_lif_node(
            channels=channels,
            v_threshold=float(v_threshold),
            reset_mode=reset_mode,
            **kwargs,
        )
    if backend in {"auto", "", "default"} and _should_use_spikingjelly_lif_auto():
        return make_spikingjelly_lif_node(
            channels=channels,
            v_threshold=float(v_threshold),
            reset_mode=reset_mode,
            **kwargs,
        )
    node = TorchSingleStepLIFNode(
        channels=channels,
        v_threshold=float(v_threshold),
        reset_mode=reset_mode,
        trainable_threshold=bool(trainable_threshold),
        **kwargs,
    )
    setattr(node, "psd_neuron_backend", "torch")
    return node


def _positive_threshold_init(v_threshold: float, size: int, *, eps: float) -> torch.Tensor:
    value = max(float(v_threshold) - float(eps), float(eps))
    raw = math.log(math.expm1(value))
    return torch.full((int(size),), float(raw), dtype=torch.float32)


class SNNBenchRFNode(nn.Module):
    """Single-step RF activation node for reference-topology CNN backbones.

    The project reference-topology models already contain Conv/BN/Linear modules. Therefore this
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
            "v_threshold": self.effective_threshold().detach(),
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
    """Return the reference VGG-11 topology channel schedule.

    The VGG-11 schedule follows ``reference/SNNs/vgg_snn.py``
    ``VGG_CFGS[11] = [1, M, 2, M, 4, 4, M, 8, 8, M, 8, 8, M]``
    with ``base_channels=64`` and ``max_channels=512``.
    """
    if depth == 7:
        return [64, "M", 128, "M", 256, "M", 512, "M"], 512
    elif depth == 11:
        return [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"], 512
    elif depth == 15:
        return [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, "M"], 512
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


        cfg, last_c = get_vgg_cfg(depth)
        self.features, pool_count = self._make_layers(
            cfg,
            in_channels,
            bias,
            v_threshold=v_threshold,
            reset_mode=reset_mode,
            single_step_neuron=single_step_neuron,
            trainable_threshold=trainable_threshold,
        )

        self.fixpool = nn.AdaptiveAvgPool2d((1, 1))

        # Reference VGG-11 classifier: fc -> spike -> dropout -> fc -> spike ->
        # dropout -> fc -> spike.  The final spike node is kept because this
        # project records temporal output membranes/spikes for PSD analysis.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(last_c, 512, bias=bias),
            single_step_neuron(
                channels=512,
                v_threshold=float(v_threshold),
                reset_mode=reset_mode,
                trainable_threshold=bool(trainable_threshold),
            ),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512, bias=bias),
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
        pool_count = 0
        cur_ch = in_ch

        for v in cfg:
            if v == "M":
                layers += [SafeMaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
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

                cur_ch = v

        return nn.Sequential(*layers), pool_count

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


def sew_function(a: torch.Tensor, s: torch.Tensor, cnf: str) -> torch.Tensor:
    """Spike-element-wise residual merge used by SEW-ResNet.

    ``ADD`` is the fixed project default, matching the requested
    reference/SNNs SEW-ResNet18 structure.
    """
    token = str(cnf).upper()
    if token == "ADD":
        return a + s
    if token == "AND":
        return a * s
    if token == "IAND":
        return (1.0 - a) * s
    raise ValueError(f"Unsupported SEW connect function: {cnf!r}")



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
        cnf: str = "ADD",
        **kwargs,
    ):
        super().__init__()
        self.cnf = str(cnf).upper()

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
        self.downsample_sn = single_step_neuron(channels=int(planes * self.expansion), **kwargs) if downsample is not None else None
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            if self.downsample_sn is not None:
                identity = self.downsample_sn(identity)

        return sew_function(out, identity, self.cnf)
    
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
        cnf: str = "ADD",
        in_channels: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.cnf = str(cnf).upper()
        self.input_channels = int(in_channels)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if single_step_neuron is None:
            single_step_neuron = make_snnbench_lif_node

        # reference/SNNs contributes only the ResNet-18 BasicBlock topology and
        # SEW residual merge. Prepared frame channels are consumed directly by
        # the first BasicBlock; no separate input projection branch is created.
        self._norm_layer = norm_layer
        self.inplanes = self.input_channels
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple")

        self.groups = groups
        self.base_width = width_per_group

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
                cnf=self.cnf,
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
                    cnf=self.cnf,
                    **kwargs,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
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
    in_channels: int = 3,
    **kwargs,
):
    if pretrained:
        raise ValueError("pretrained=True is not supported in the local reference-topology adapter.")

    return SpikingResNet(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        norm_layer=norm_layer,
        single_step_neuron=single_step_neuron,
        in_channels=int(in_channels),
        **kwargs,
    )
