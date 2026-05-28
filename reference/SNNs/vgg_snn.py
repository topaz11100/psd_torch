#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VGG-SNN classifier for CIFAR-10/100, CIFAR10-DVS, CIFAR100-DVS/I2E-CIFAR100,
Caltech-101, N-Caltech101, and DVS128-Gesture.

Framework
---------
- PyTorch + SpikingJelly activation_based API.
- Direct SNN training with standard BPTT or temporal truncated BPTT.
- No Poisson encoder is used.

Input convention
----------------
- Static datasets (CIFAR-10/100, Caltech-101): image tensor [N, C, H, W] is repeated
  directly for T timesteps, giving [T, N, C, H, W].
- Event datasets (CIFAR10-DVS, CIFAR100-DVS/I2E-CIFAR100, N-Caltech101, DVS128-Gesture): event frames are
  loaded as [N, T, 2, H, W] and transposed to [T, N, 2, H, W]. Frames are
  optionally resized and binarized. This is event-frame integration, not Poisson
  encoding.

Examples
--------
# Fast syntax/shape/gradient check without dataset download
python snn_vgg_multidataset_tbptt.py --mode sanity --device cpu --vgg-depth 7 --base-channels 16 --T 4

# CIFAR-10, VGG11-SNN, standard BPTT
python snn_vgg_multidataset_tbptt.py --dataset cifar10 --mode train_test --vgg-depth 11 --T 10 --data-dir ./data --out-dir ./runs/cifar10_vgg11

# CIFAR-100, VGG11-SNN, standard BPTT
python snn_vgg_multidataset_tbptt.py --dataset cifar100 --mode train_test --vgg-depth 11 --T 10 --data-dir ./data --out-dir ./runs/cifar100_vgg11

# CIFAR10-DVS, TEBN split if supported by SpikingJelly, TBPTT with k=10
python snn_vgg_multidataset_tbptt.py --dataset cifar10dvs --mode train_test --T 100 --tbptt --tbptt-k 10 --dvs-size 48

# DVS128-Gesture, direct Dropbox download of DvsGesture.tar.gz then SpikingJelly preprocessing
python snn_vgg_multidataset_tbptt.py --dataset dvsgesture --mode train_test --T 60 --tbptt --tbptt-k 10 --dvs-size 64

# Caltech-101, background class excluded by default
python snn_vgg_multidataset_tbptt.py --dataset caltech101 --mode train_test --image-size 128 --T 10

# N-Caltech101, Mendeley archive download then SpikingJelly preprocessing
python snn_vgg_multidataset_tbptt.py --dataset ncaltech101 --mode train_test --T 60 --tbptt --tbptt-k 10 --dvs-size 48

# CIFAR100-DVS / I2E-CIFAR100, Hugging Face auto-download through datasets.load_dataset
python snn_vgg_multidataset_tbptt.py --dataset cifar100dvs --mode train_test --T 10 --dvs-size 128 --data-dir ./data --out-dir ./runs/i2e_cifar100_vgg11
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Compatibility patch for old SpikingJelly CIFAR10-DVS converter with NumPy >= 1.24.
if "bool" not in np.__dict__:
    np.bool = np.bool_

if "int" not in np.__dict__:
    np.int = int

if "float" not in np.__dict__:
    np.float = float

if "complex" not in np.__dict__:
    np.complex = complex

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch.utils.data.distributed import DistributedSampler

try:
    from spikingjelly.activation_based import functional, layer, neuron, surrogate
except Exception as exc:  # pragma: no cover - gives a clear runtime message
    raise RuntimeError(
        "This script requires SpikingJelly. Install it with `pip install spikingjelly`."
    ) from exc


# Dropbox direct link used by snnTorch for the original IBM DVS Gesture archive.
# SpikingJelly's official IBM Box link is not code-downloadable without login in
# some environments, so this link is used only to obtain the same archive and then
# SpikingJelly's own extraction/conversion routine is used.
DVSGESTURE_DROPBOX_URL = "https://www.dropbox.com/s/cct5kyilhtsliup/DvsGesture.tar.gz?dl=1"
DVSGESTURE_ARCHIVE_MD5 = "8a5c71fb11e24e5ca5b11866ca6c00a1"

# Direct Mendeley URL used by Tonic for N-Caltech101. SpikingJelly exposes
# converters for the dataset but marks its original web page as non-downloadable;
# this URL points to the same Caltech101.zip archive expected by SpikingJelly.
NCALTECH101_MENDELEY_URL = (
    "https://data.mendeley.com/public-files/datasets/cy6cvx3ryv/files/"
    "36b5c52a-b49d-4853-addb-a836a8883e49/file_downloaded"
)
NCALTECH101_ARCHIVE_MD5 = "66201824eabb0239c7ab992480b50ba3"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    input_shape: Tuple[int, int, int]
    num_classes: int
    time_steps: int
    is_event_dataset: bool
    description: str = ""

    @property
    def input_dim(self) -> int:
        c, h, w = self.input_shape
        return c * h * w


class EventFrameTransform:
    """Convert SpikingJelly frame arrays to tensors and match VGG input size.

    Expected sample shape is [T, C, H, W]. If `binarize=True`, the transformed
    tensor indicates event occurrence at each pixel/polarity/time bin. This is
    the common event-image representation used in direct SNN training papers and
    is not Poisson spike generation.
    """

    def __init__(
        self,
        size: Optional[Tuple[int, int]] = None,
        target_frames: Optional[int] = None,
        binarize: bool = True,
        normalize: bool = False,
        resize_mode: str = "nearest",
        temporal_resize_mode: str = "nearest",
    ):
        self.size = size
        self.target_frames = int(target_frames) if target_frames is not None and int(target_frames) > 0 else None
        self.binarize = bool(binarize)
        self.normalize = bool(normalize)
        self.resize_mode = resize_mode
        self.temporal_resize_mode = temporal_resize_mode

    def __call__(self, x) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Event frames must have shape [T,C,H,W], got {tuple(x.shape)}")

        if self.size is not None and tuple(x.shape[-2:]) != tuple(self.size):
            t, c, h, w = x.shape
            flat = x.reshape(t * c, 1, h, w)
            if self.resize_mode == "nearest":
                flat = F.interpolate(flat, size=self.size, mode="nearest")
            else:
                flat = F.interpolate(flat, size=self.size, mode=self.resize_mode, align_corners=False)
            x = flat.reshape(t, c, self.size[0], self.size[1])

        if self.target_frames is not None and x.shape[0] != self.target_frames:
            # Resize/crop the temporal axis to match the requested SNN simulation length.
            # Input is [T,C,H,W]; interpolate as [1, C*H*W, T].
            t, c, h, w = x.shape
            seq = x.permute(1, 2, 3, 0).reshape(1, c * h * w, t)
            if self.temporal_resize_mode == "nearest":
                seq = F.interpolate(seq, size=self.target_frames, mode="nearest")
            else:
                seq = F.interpolate(seq, size=self.target_frames, mode="linear", align_corners=False)
            x = seq.reshape(c, h, w, self.target_frames).permute(3, 0, 1, 2).contiguous()

        if self.binarize:
            x = (x > 0).to(torch.float32)
        elif self.normalize:
            denom = x.flatten(1).amax(dim=1).clamp_min(1.0)
            x = x / denom.view(-1, 1, 1, 1)
        return x


class TargetRemapSubset(Dataset):
    """Subset wrapper that can remap targets while preserving base transforms."""

    def __init__(self, dataset: Dataset, indices: Sequence[int], target_map: Optional[Dict[int, int]] = None):
        self.dataset = dataset
        self.indices = list(int(i) for i in indices)
        self.target_map = target_map

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        x, y = self.dataset[self.indices[idx]]
        if self.target_map is not None:
            y = self.target_map[int(y)]
        return x, y




def unpack_i2e_event_data(item, use_io: bool = True) -> torch.Tensor:
    """Decode the packed I2E event tensor format to [T,C,H,W].

    The I2E Hugging Face dataset stores a uint8 binary blob whose first four
    uint16 values encode the tensor shape (T, C, H, W). The remaining bytes are
    bit-packed binary event values.
    """
    import io
    import numpy as np

    payload = item["data"] if isinstance(item, dict) else item
    if use_io:
        with io.BytesIO(payload) as f:
            raw_data = np.load(f)
    else:
        raw_data = np.load(payload)
    header_size = 4 * 2
    shape_header = raw_data[:header_size].view(np.uint16)
    original_shape = tuple(int(v) for v in shape_header)
    packed_body = raw_data[header_size:]
    unpacked = np.unpackbits(packed_body)
    num_elements = int(np.prod(original_shape))
    event_flat = unpacked[:num_elements]
    event_data = event_flat.reshape(original_shape).astype(np.float32).copy()
    return torch.from_numpy(event_data)


class I2EEventDataset(Dataset):
    """Hugging Face I2E event dataset wrapper.

    Provides samples as [T,2,H,W] event frames and integer labels. The loader is
    used for the CIFAR100-DVS/I2E-CIFAR100 option and downloads/caches data
    through `datasets.load_dataset` when the cache is absent.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        config_name: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        hf_endpoint: str = "",
    ):
        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = hf_endpoint
        try:
            from datasets import load_dataset
        except Exception as exc:
            raise RuntimeError(
                "CIFAR100-DVS/I2E-CIFAR100 loading requires the Hugging Face `datasets` package. "
                "Install it with `pip install datasets`."
            ) from exc
        self.ds = load_dataset(
            "UESTC-BICS/I2E",
            config_name,
            split=split,
            cache_dir=str(cache_dir),
            keep_in_memory=False,
        )
        self.transform = transform
        self.target_transform = target_transform
        try:
            self.targets = [int(v) for v in self.ds["label"]]
        except Exception:
            self.targets = None

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        item = self.ds[idx]
        event = unpack_i2e_event_data(item)
        label = int(item["label"])
        if self.transform is not None:
            event = self.transform(event)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return event, label

class MultiStepAdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size: int | Tuple[int, int]):
        super().__init__()
        self.output_size = output_size
        # Always consumes multi-step tensors [T,N,C,H,W].

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [T,N,C,H,W], got {tuple(x.shape)}")
        t, n = x.shape[:2]
        y = F.adaptive_avg_pool2d(x.flatten(0, 1), self.output_size)
        return y.view(t, n, *y.shape[1:])


class MultiStepFlatten(nn.Module):
    def __init__(self):
        super().__init__()
        # Always consumes multi-step tensors.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(start_dim=2)


class MultiStepBatchNorm2d(nn.BatchNorm2d):
    """Standard BN over T*N samples for [T,N,C,H,W]."""

    def __init__(self, num_features: int, **kwargs):
        super().__init__(num_features, **kwargs)
        # Always consumes multi-step tensors [T,N,C,H,W].

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [T,N,C,H,W], got {tuple(x.shape)}")
        t, n = x.shape[:2]
        y = super().forward(x.flatten(0, 1))
        return y.view(t, n, *y.shape[1:])


class MultiStepBatchNorm1d(nn.BatchNorm1d):
    """Standard BN over T*N samples for [T,N,C]."""

    def __init__(self, num_features: int, **kwargs):
        super().__init__(num_features, **kwargs)
        # Always consumes multi-step tensors [T,N,C].

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [T,N,C], got {tuple(x.shape)}")
        t, n = x.shape[:2]
        y = super().forward(x.flatten(0, 1))
        return y.view(t, n, -1)


class ThresholdDependentBatchNorm2d(nn.Module):
    """tdBN for [T,N,C,H,W].

    Implements y = gamma * alpha * Vth * (x - mean) / sqrt(var + eps) + beta,
    where mean/var are estimated over T, N, H, W for each channel.
    """

    def __init__(
        self,
        num_features: int,
        alpha: float = 1.0,
        v_threshold: float = 1.0,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.alpha = float(alpha)
        self.v_threshold = float(v_threshold)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)
        self.track_running_stats = bool(track_running_stats)
        # Always consumes multi-step tensors [T,N,C,H,W].

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"tdBN2d expects [T,N,C,H,W], got {tuple(x.shape)}")
        if x.shape[2] != self.num_features:
            raise ValueError(f"Expected C={self.num_features}, got C={x.shape[2]}")

        dtype = x.dtype
        xf = x.float()
        if self.training or not self.track_running_stats:
            mean = xf.mean(dim=(0, 1, 3, 4))
            var = xf.var(dim=(0, 1, 3, 4), unbiased=False)
            if self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked.add_(1)
                    self.running_mean.mul_(1.0 - self.momentum).add_(self.momentum * mean.detach())
                    self.running_var.mul_(1.0 - self.momentum).add_(self.momentum * var.detach())
        else:
            mean = self.running_mean
            var = self.running_var

        view = (1, 1, -1, 1, 1)
        y = (xf - mean.view(view)) / torch.sqrt(var.view(view) + self.eps)
        y = y * (self.alpha * self.v_threshold)
        if self.affine:
            y = y * self.weight.float().view(view) + self.bias.float().view(view)
        return y.to(dtype=dtype)

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, alpha={self.alpha}, "
            f"v_threshold={self.v_threshold}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}"
        )


class ThresholdDependentBatchNorm1d(nn.Module):
    """tdBN for [T,N,C]."""

    def __init__(
        self,
        num_features: int,
        alpha: float = 1.0,
        v_threshold: float = 1.0,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.alpha = float(alpha)
        self.v_threshold = float(v_threshold)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)
        self.track_running_stats = bool(track_running_stats)
        # Always consumes multi-step tensors [T,N,C].

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"tdBN1d expects [T,N,C], got {tuple(x.shape)}")
        if x.shape[2] != self.num_features:
            raise ValueError(f"Expected C={self.num_features}, got C={x.shape[2]}")

        dtype = x.dtype
        xf = x.float()
        if self.training or not self.track_running_stats:
            mean = xf.mean(dim=(0, 1))
            var = xf.var(dim=(0, 1), unbiased=False)
            if self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked.add_(1)
                    self.running_mean.mul_(1.0 - self.momentum).add_(self.momentum * mean.detach())
                    self.running_var.mul_(1.0 - self.momentum).add_(self.momentum * var.detach())
        else:
            mean = self.running_mean
            var = self.running_var

        view = (1, 1, -1)
        y = (xf - mean.view(view)) / torch.sqrt(var.view(view) + self.eps)
        y = y * (self.alpha * self.v_threshold)
        if self.affine:
            y = y * self.weight.float().view(view) + self.bias.float().view(view)
        return y.to(dtype=dtype)

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, alpha={self.alpha}, "
            f"v_threshold={self.v_threshold}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}"
        )


# Conv-layer configurations. Depth counts trainable layers: conv layers + 3 FC layers.
VGG_CFGS: Dict[int, List[int | str]] = {
    7: [1, "M", 2, "M", 4, "M", 8, "M"],
    9: [1, 1, "M", 2, "M", 4, 4, "M", 8, "M"],
    11: [1, "M", 2, "M", 4, 4, "M", 8, 8, "M", 8, 8, "M"],
    13: [1, 1, "M", 2, 2, "M", 4, 4, "M", 8, 8, "M", 8, 8, "M"],
    15: [1, 1, "M", 2, 2, "M", 4, 4, 4, "M", 8, 8, 8, "M", 8, 8, "M"],
}


class VGGSNN(nn.Module):
    """VGG-style convolutional SNN with tdBN/BN/none normalization.

    Input:  [T, N, C, H, W]
    Output: [T, N, num_classes]
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_classes: int,
        vgg_depth: int = 11,
        base_channels: int = 64,
        max_channels: int = 512,
        fc_dim: int = 512,
        pool_output_size: int = 1,
        pool_type: str = "avg",
        norm: str = "tdbn",
        tdbn_alpha: float = 1.0,
        tdbn_eps: float = 1e-5,
        tdbn_momentum: float = 0.1,
        dropout: float = 0.0,
        conv_dropout: float = 0.0,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        detach_reset: bool = True,
        surrogate_name: str = "atan",
        neuron_type: str = "lif",
        readout: str = "linear",
    ):
        super().__init__()
        if vgg_depth not in VGG_CFGS:
            raise ValueError(f"vgg_depth must be one of {sorted(VGG_CFGS)}")
        if pool_type not in {"avg", "max"}:
            raise ValueError("pool_type must be 'avg' or 'max'")
        if norm not in {"tdbn", "bn", "none"}:
            raise ValueError("norm must be one of: tdbn, bn, none")
        if readout not in {"linear", "spike"}:
            raise ValueError("readout must be 'linear' or 'spike'")

        self.input_shape = tuple(int(v) for v in input_shape)
        self.num_classes = int(num_classes)
        self.vgg_depth = int(vgg_depth)
        self.base_channels = int(base_channels)
        self.max_channels = int(max_channels)
        self.fc_dim = int(fc_dim)
        self.pool_output_size = int(pool_output_size)
        self.pool_type = pool_type
        self.norm = norm
        self.readout = readout
        self.v_threshold = float(v_threshold)

        surr = build_surrogate(surrogate_name)
        in_channels = self.input_shape[0]
        modules: List[nn.Module] = []
        last_channels = in_channels

        def scaled_channels(mult: int) -> int:
            return int(max(1, min(self.max_channels, self.base_channels * int(mult))))

        for item in VGG_CFGS[vgg_depth]:
            if item == "M":
                if pool_type == "avg":
                    modules.append(layer.AvgPool2d(kernel_size=2, stride=2))
                else:
                    modules.append(layer.MaxPool2d(kernel_size=2, stride=2))
                continue

            out_channels = scaled_channels(int(item))
            modules.append(layer.Conv2d(last_channels, out_channels, kernel_size=3, padding=1, bias=(norm == "none")))
            modules.append(build_norm2d(norm, out_channels, tdbn_alpha, v_threshold, tdbn_eps, tdbn_momentum))
            modules.append(build_spiking_neuron(neuron_type, tau, v_threshold, v_reset, detach_reset, surr))
            if conv_dropout > 0:
                modules.append(layer.Dropout(conv_dropout))
            last_channels = out_channels

        self.features = nn.Sequential(*modules)
        self.avgpool = MultiStepAdaptiveAvgPool2d((pool_output_size, pool_output_size))
        self.flatten = MultiStepFlatten()

        feature_dim = last_channels * pool_output_size * pool_output_size
        classifier: List[nn.Module] = [
            layer.Linear(feature_dim, fc_dim),
            build_norm1d(norm, fc_dim, tdbn_alpha, v_threshold, tdbn_eps, tdbn_momentum),
            build_spiking_neuron(neuron_type, tau, v_threshold, v_reset, detach_reset, surr),
        ]
        if dropout > 0:
            classifier.append(layer.Dropout(dropout))
        classifier.extend(
            [
                layer.Linear(fc_dim, fc_dim),
                build_norm1d(norm, fc_dim, tdbn_alpha, v_threshold, tdbn_eps, tdbn_momentum),
                build_spiking_neuron(neuron_type, tau, v_threshold, v_reset, detach_reset, surr),
            ]
        )
        if dropout > 0:
            classifier.append(layer.Dropout(dropout))
        classifier.append(layer.Linear(fc_dim, num_classes))
        if readout == "spike":
            classifier.append(build_spiking_neuron(neuron_type, tau, v_threshold, v_reset, detach_reset, surr))
        self.classifier = nn.Sequential(*classifier)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, layer.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Linear, layer.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.ndim != 5:
            raise ValueError(f"VGGSNN expects [T,N,C,H,W], got {tuple(x_seq.shape)}")
        x = self.features(x_seq)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def build_surrogate(name: str):
    name = name.lower()
    if name == "atan":
        return surrogate.ATan()
    if name == "sigmoid":
        return surrogate.Sigmoid()
    if name == "fast_sigmoid":
        if hasattr(surrogate, "FastSigmoid"):
            return surrogate.FastSigmoid()
        return surrogate.Sigmoid()
    if name == "piecewise_quadratic":
        return surrogate.PiecewiseQuadratic()
    raise ValueError(f"Unsupported surrogate function: {name}")


def build_spiking_neuron(
    neuron_type: str,
    tau: float,
    v_threshold: float,
    v_reset: Optional[float],
    detach_reset: bool,
    surr: Callable,
) -> nn.Module:
    neuron_type = neuron_type.lower()
    if neuron_type == "lif":
        return neuron.LIFNode(
            tau=tau,
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_function=surr,
            detach_reset=detach_reset,
            step_mode="m",
        )
    if neuron_type == "if":
        return neuron.IFNode(
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_function=surr,
            detach_reset=detach_reset,
            step_mode="m",
        )
    if neuron_type == "plif":
        return neuron.ParametricLIFNode(
            init_tau=tau,
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_function=surr,
            detach_reset=detach_reset,
            step_mode="m",
        )
    raise ValueError("neuron_type must be one of: lif, if, plif")


def build_norm2d(norm: str, c: int, alpha: float, v_threshold: float, eps: float, momentum: float) -> nn.Module:
    if norm == "tdbn":
        return ThresholdDependentBatchNorm2d(c, alpha=alpha, v_threshold=v_threshold, eps=eps, momentum=momentum)
    if norm == "bn":
        return MultiStepBatchNorm2d(c, eps=eps, momentum=momentum)
    return nn.Identity()


def build_norm1d(norm: str, c: int, alpha: float, v_threshold: float, eps: float, momentum: float) -> nn.Module:
    if norm == "tdbn":
        return ThresholdDependentBatchNorm1d(c, alpha=alpha, v_threshold=v_threshold, eps=eps, momentum=momentum)
    if norm == "bn":
        return MultiStepBatchNorm1d(c, eps=eps, momentum=momentum)
    return nn.Identity()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)



def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed_mode(args: argparse.Namespace) -> None:
    args.rank = int(os.environ.get("RANK", "0"))
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.local_rank = int(os.environ.get("LOCAL_RANK", getattr(args, "local_rank", 0)))
    args.distributed = bool(args.ddp and args.world_size > 1)
    if not args.distributed:
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=backend, init_method="env://")


def cleanup_distributed() -> None:
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def model_without_ddp(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def wrap_model_for_ddp(model: nn.Module, args: argparse.Namespace, device: torch.device) -> nn.Module:
    if not getattr(args, "distributed", False):
        return model
    if device.type == "cuda":
        return DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    return DistributedDataParallel(model)


def reduce_epoch_totals(
    total_loss: float,
    total_correct: int,
    total_seen: int,
    seconds: float,
    device: torch.device,
) -> Tuple[float, int, int, float]:
    if not is_dist_avail_and_initialized():
        return total_loss, total_correct, total_seen, seconds
    stats = torch.tensor(
        [float(total_loss), float(total_correct), float(total_seen)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    time_stats = torch.tensor([float(seconds)], dtype=torch.float64, device=device)
    dist.all_reduce(time_stats, op=dist.ReduceOp.MAX)
    return float(stats[0].item()), int(stats[1].item()), int(stats[2].item()), float(time_stats[0].item())


def build_datasets_ddp_safe(args: argparse.Namespace):
    if not is_dist_avail_and_initialized():
        return build_datasets(args)
    if is_main_process():
        result = build_datasets(args)
        dist.barrier()
    else:
        dist.barrier()
        result = build_datasets(args)
    dist.barrier()
    return result


def canonical_dataset_name(name: str) -> str:
    """Normalize user-facing dataset aliases to internal names."""
    key = str(name).lower().replace("-", "").replace("_", "")
    aliases = {
        "dvscifar10": "cifar10dvs",
        "dvs10cifar": "cifar10dvs",
        "cifar10dvs": "cifar10dvs",
        "dvscifar100": "cifar100dvs",
        "dvs100cifar": "cifar100dvs",
        "cifar100dvs": "cifar100dvs",
        "i2ecifar100": "cifar100dvs",
        "ncaltech": "ncaltech101",
        "ncaltech101": "ncaltech101",
        "dvs128gesture": "dvsgesture",
        "dvsgesture": "dvsgesture",
    }
    return aliases.get(key, key)


def _has_npz(root: Path) -> bool:
    try:
        return root.exists() and any(root.rglob("*.npz"))
    except OSError:
        return False


def _download_resources(resource_url_md5: Sequence[Tuple[str, str, str]], download_root: Path) -> None:
    from torchvision.datasets.utils import check_integrity, download_url

    download_root.mkdir(parents=True, exist_ok=True)
    for filename, url, md5 in resource_url_md5:
        fpath = download_root / filename
        if not check_integrity(str(fpath), md5):
            if fpath.exists():
                fpath.unlink()
            print(f"[download] {filename}")
            download_url(url=url, root=str(download_root), filename=filename, md5=md5)
        else:
            print(f"[download] found valid archive: {fpath}")


def ensure_cifar10dvs_prepared(root: Path, download: bool = True) -> None:
    """Ensure CIFAR10-DVS raw events_np exists for SpikingJelly."""
    raw_root = root / "events_np"
    if _has_npz(raw_root):
        return
    if not download:
        raise FileNotFoundError(
            f"CIFAR10-DVS events_np not found under {raw_root}. Use --download or prepare manually."
        )

    try:
        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
        from torchvision.datasets.utils import extract_archive
    except Exception as exc:
        raise RuntimeError("CIFAR10-DVS preparation needs SpikingJelly datasets and torchvision.") from exc

    root.mkdir(parents=True, exist_ok=True)
    download_root = root / "download"
    extract_root = root / "extract"
    if raw_root.exists() and not _has_npz(raw_root):
        shutil.rmtree(raw_root)
    if extract_root.exists() and not any(extract_root.iterdir()):
        shutil.rmtree(extract_root)

    _download_resources(CIFAR10DVS.resource_url_md5(), download_root)

    if not extract_root.exists() or not any(extract_root.iterdir()):
        extract_root.mkdir(parents=True, exist_ok=True)
        print(f"[CIFAR10-DVS] extracting archives to {extract_root}")
        if hasattr(CIFAR10DVS, "extract_downloaded_files"):
            CIFAR10DVS.extract_downloaded_files(download_root, extract_root)
        else:
            for archive in download_root.iterdir():
                extract_archive(str(archive), str(extract_root))

    raw_root.mkdir(parents=True, exist_ok=True)
    print(f"[CIFAR10-DVS] creating events_np under {raw_root}")
    if hasattr(CIFAR10DVS, "create_raw_from_extracted"):
        CIFAR10DVS.create_raw_from_extracted(extract_root, raw_root)
    elif hasattr(CIFAR10DVS, "create_events_np_files"):
        CIFAR10DVS.create_events_np_files(str(extract_root), str(raw_root))
    else:
        raise RuntimeError("This SpikingJelly version does not expose CIFAR10-DVS raw conversion helpers.")

    if not _has_npz(raw_root):
        raise RuntimeError(f"CIFAR10-DVS preprocessing finished but no npz file was found under {raw_root}")


def ensure_dvsgesture_prepared(root: Path, download: bool = True) -> None:
    """Ensure DVS128-Gesture events_np exists for SpikingJelly.

    SpikingJelly marks the IBM Box source as non-downloadable in some versions.
    This function downloads the same DvsGesture.tar.gz through a direct Dropbox
    mirror used by snnTorch, then calls SpikingJelly's official converter.
    """
    raw_root = root / "events_np"
    if _has_npz(raw_root / "train") and _has_npz(raw_root / "test"):
        return
    if not download:
        raise FileNotFoundError(
            f"DVS128-Gesture events_np not found under {raw_root}. Use --download or place DvsGesture.tar.gz manually."
        )

    try:
        from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
        from torchvision.datasets.utils import check_integrity, download_url, extract_archive
    except Exception as exc:
        raise RuntimeError("DVS128-Gesture preparation needs SpikingJelly datasets and torchvision.") from exc

    root.mkdir(parents=True, exist_ok=True)
    download_root = root / "download"
    extract_root = root / "extract"
    download_root.mkdir(parents=True, exist_ok=True)
    archive = download_root / "DvsGesture.tar.gz"

    if not check_integrity(str(archive), DVSGESTURE_ARCHIVE_MD5):
        if archive.exists():
            archive.unlink()
        print(f"[DVS-Gesture] downloading DvsGesture.tar.gz to {download_root}")
        try:
            download_url(
                url=DVSGESTURE_DROPBOX_URL,
                root=str(download_root),
                filename="DvsGesture.tar.gz",
                md5=DVSGESTURE_ARCHIVE_MD5,
            )
        except Exception as exc:
            raise RuntimeError(
                "Automatic DVS128-Gesture download failed. Place DvsGesture.tar.gz in "
                f"{download_root} and rerun, or check the Dropbox/IBM Box dataset link."
            ) from exc
    else:
        print(f"[DVS-Gesture] found valid archive: {archive}")

    if not (extract_root / "DvsGesture").exists():
        if extract_root.exists():
            shutil.rmtree(extract_root)
        extract_root.mkdir(parents=True, exist_ok=True)
        print(f"[DVS-Gesture] extracting archive to {extract_root}")
        if hasattr(DVS128Gesture, "extract_downloaded_files"):
            DVS128Gesture.extract_downloaded_files(download_root, extract_root)
        else:
            extract_archive(str(archive), str(extract_root))

    if raw_root.exists() and not (_has_npz(raw_root / "train") and _has_npz(raw_root / "test")):
        shutil.rmtree(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    print(f"[DVS-Gesture] creating events_np under {raw_root}")
    if hasattr(DVS128Gesture, "create_raw_from_extracted"):
        DVS128Gesture.create_raw_from_extracted(extract_root, raw_root)
    elif hasattr(DVS128Gesture, "create_events_np_files"):
        DVS128Gesture.create_events_np_files(str(extract_root), str(raw_root))
    else:
        raise RuntimeError("This SpikingJelly version does not expose DVS-Gesture raw conversion helpers.")

    if not (_has_npz(raw_root / "train") and _has_npz(raw_root / "test")):
        raise RuntimeError(f"DVS-Gesture preprocessing finished but events_np is incomplete: {raw_root}")


def ensure_ncaltech101_prepared(root: Path, download: bool = True) -> None:
    """Ensure N-Caltech101 events_np exists for SpikingJelly.

    SpikingJelly documents NCaltech101 as non-downloadable because its
    resource_url_md5 points to a dataset web page rather than direct file URLs.
    This helper downloads the Mendeley archive used by Tonic, extracts it, and
    then delegates event-to-npz conversion to SpikingJelly. The final expected
    tree is root/events_np/<class_name>/*.npz.
    """
    raw_root = root / "events_np"
    if _has_npz(raw_root):
        return

    try:
        from spikingjelly.datasets.n_caltech101 import NCaltech101
        from torchvision.datasets.utils import check_integrity, download_url, extract_archive
    except Exception as exc:
        raise RuntimeError("N-Caltech101 preparation needs SpikingJelly datasets and torchvision.") from exc

    root.mkdir(parents=True, exist_ok=True)
    download_root = root / "download"
    extract_root = root / "extract"
    download_root.mkdir(parents=True, exist_ok=True)

    archive = download_root / "Caltech101.zip"
    alt_archive = download_root / "N-Caltech101-archive.zip"
    valid_archive: Optional[Path] = None
    for candidate in (archive, alt_archive):
        if check_integrity(str(candidate), NCALTECH101_ARCHIVE_MD5):
            valid_archive = candidate
            break

    extract_ready = (extract_root / "Caltech101").is_dir() and any((extract_root / "Caltech101").iterdir())
    if not extract_ready and valid_archive is None:
        if not download:
            raise FileNotFoundError(
                f"N-Caltech101 archive was not found under {download_root}. "
                "Use --download, or manually place Caltech101.zip / N-Caltech101-archive.zip there."
            )
        if archive.exists():
            archive.unlink()
        print(f"[N-Caltech101] downloading Caltech101.zip to {download_root}")
        download_url(
            url=NCALTECH101_MENDELEY_URL,
            root=str(download_root),
            filename="Caltech101.zip",
            md5=NCALTECH101_ARCHIVE_MD5,
        )
        valid_archive = archive
    elif valid_archive is not None:
        print(f"[N-Caltech101] found valid archive: {valid_archive}")

    if not extract_ready:
        if extract_root.exists():
            shutil.rmtree(extract_root)
        extract_root.mkdir(parents=True, exist_ok=True)
        print(f"[N-Caltech101] extracting archive to {extract_root}")
        # SpikingJelly's extractor assumes Caltech101.zip in download_root.
        # Direct extraction is more robust when the archive was manually placed
        # under either supported filename.
        extract_archive(str(valid_archive), str(extract_root))
    else:
        print(f"[N-Caltech101] existing extract directory found: {extract_root / 'Caltech101'}")

    if raw_root.exists() and not _has_npz(raw_root):
        shutil.rmtree(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    print(f"[N-Caltech101] creating events_np under {raw_root}")
    if hasattr(NCaltech101, "create_raw_from_extracted"):
        NCaltech101.create_raw_from_extracted(extract_root, raw_root)
    elif hasattr(NCaltech101, "create_events_np_files"):
        NCaltech101.create_events_np_files(str(extract_root), str(raw_root))
    else:
        raise RuntimeError("This SpikingJelly version does not expose N-Caltech101 raw conversion helpers.")

    if not _has_npz(raw_root):
        raise RuntimeError(f"N-Caltech101 preprocessing finished but no npz file was found under {raw_root}")


def _label_key(value):
    if isinstance(value, torch.Tensor):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value


def _coerce_labels_to_int(values: Sequence) -> List[int]:
    normalized = [_label_key(v) for v in values]
    try:
        return [int(v) for v in normalized]
    except Exception:
        # Some event datasets expose class names as labels. Map them to stable
        # alphabetical IDs so stratified splitting still works.
        classes = {str(v): i for i, v in enumerate(sorted(set(str(v) for v in normalized)))}
        return [classes[str(v)] for v in normalized]


def get_targets(dataset: Dataset) -> List[int]:
    if isinstance(dataset, TargetRemapSubset):
        base_targets = get_targets(dataset.dataset)
        labels = [int(base_targets[i]) for i in dataset.indices]
        if dataset.target_map is not None:
            labels = [int(dataset.target_map[y]) for y in labels]
        return labels
    if isinstance(dataset, Subset):
        base_targets = get_targets(dataset.dataset)
        return [int(base_targets[i]) for i in dataset.indices]
    for attr in ("targets", "labels", "y"):
        if hasattr(dataset, attr):
            value = getattr(dataset, attr)
            if value is not None:
                return _coerce_labels_to_int(list(value))
    # Fallback: expensive but robust for smaller datasets.
    labels = []
    old_transform = getattr(dataset, "transform", None)
    try:
        if hasattr(dataset, "transform"):
            dataset.transform = None
        for i in range(len(dataset)):
            _, y = dataset[i]
            labels.append(y)
    finally:
        if hasattr(dataset, "transform"):
            dataset.transform = old_transform
    return _coerce_labels_to_int(labels)


def stratified_split_indices(
    labels: Sequence[int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1)")
    if val_ratio < 0.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("val_ratio must satisfy train_ratio + val_ratio < 1")

    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        by_class.setdefault(int(y), []).append(i)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for _, idxs in sorted(by_class.items()):
        idxs = idxs[:]
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = int(round(n * val_ratio))
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train + n_val])
        test_idx.extend(idxs[n_train + n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def split_train_val_indices(labels: Sequence[int], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    if val_ratio <= 0:
        return list(range(len(labels))), []
    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        by_class.setdefault(int(y), []).append(i)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for _, idxs in sorted(by_class.items()):
        idxs = idxs[:]
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_ratio))) if len(idxs) > 1 else 0
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def limit_dataset(dataset: Optional[Dataset], limit: int) -> Optional[Dataset]:
    if dataset is None or limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, list(range(limit)))


def resolve_auto_time_steps(args: argparse.Namespace) -> int:
    if args.T > 0:
        return args.T
    # Paper-informed defaults, user-overridable. Static CIFAR follows TBPTT paper's T=10;
    # event datasets use relatively long integrated event-frame sequences.
    dataset_name = canonical_dataset_name(args.dataset)
    defaults = {
        "cifar10": 10,
        "cifar100": 10,
        "caltech101": 10,
        "cifar10dvs": 100,
        "cifar100dvs": 10,
        "ncaltech101": 60,
        "dvsgesture": 60,
    }
    return defaults[dataset_name]


def build_static_transforms(args: argparse.Namespace, dataset_name: str, train: bool):
    from torchvision import transforms

    if dataset_name in {"cifar10", "cifar100"}:
        ops: List[Callable] = []
        if train and args.augment_static:
            ops.extend([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
        ops.append(transforms.ToTensor())
        if args.normalize_static:
            if dataset_name == "cifar10":
                ops.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)))
            else:
                ops.append(transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)))
        return transforms.Compose(ops)

    if dataset_name == "caltech101":
        if train and args.augment_static:
            crop = transforms.RandomResizedCrop(args.image_size, scale=(0.7, 1.0))
            flip = transforms.RandomHorizontalFlip()
            resize_ops: List[Callable] = [crop, flip]
        else:
            resize_ops = [transforms.Resize(args.image_size + 16), transforms.CenterCrop(args.image_size)]
        ops = resize_ops + [transforms.Lambda(lambda img: img.convert("RGB")), transforms.ToTensor()]
        if args.normalize_static:
            ops.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        return transforms.Compose(ops)

    raise ValueError(dataset_name)


def build_datasets(args: argparse.Namespace) -> Tuple[Optional[Dataset], Optional[Dataset], Dataset, DatasetSpec]:
    data_dir = Path(args.data_dir).expanduser().resolve()
    args.T = resolve_auto_time_steps(args)
    dataset_name = canonical_dataset_name(args.dataset)
    args.dataset = dataset_name

    if dataset_name in {"cifar10", "cifar100"}:
        from torchvision import datasets

        cls = datasets.CIFAR10 if dataset_name == "cifar10" else datasets.CIFAR100
        num_classes = 10 if dataset_name == "cifar10" else 100
        train_tf = build_static_transforms(args, dataset_name, train=True)
        eval_tf = build_static_transforms(args, dataset_name, train=False)
        train_base = cls(root=str(data_dir), train=True, transform=train_tf, download=args.download)
        val_base = cls(root=str(data_dir), train=True, transform=eval_tf, download=args.download)
        test_set = cls(root=str(data_dir), train=False, transform=eval_tf, download=args.download)
        train_idx, val_idx = split_train_val_indices(train_base.targets, args.val_ratio, args.seed)
        train_set = Subset(train_base, train_idx)
        val_set = Subset(val_base, val_idx) if val_idx else None
        spec = DatasetSpec(dataset_name, (3, 32, 32), num_classes, args.T, False, "static RGB image repeated for T steps")

    elif dataset_name == "caltech101":
        from torchvision import datasets

        train_tf = build_static_transforms(args, "caltech101", train=True)
        eval_tf = build_static_transforms(args, "caltech101", train=False)
        base_for_labels = datasets.Caltech101(root=str(data_dir), target_type="category", transform=None, download=args.download)
        train_base = datasets.Caltech101(root=str(data_dir), target_type="category", transform=train_tf, download=False)
        eval_base = datasets.Caltech101(root=str(data_dir), target_type="category", transform=eval_tf, download=False)

        labels_raw = get_targets(base_for_labels)
        categories = getattr(base_for_labels, "categories", None)
        keep_indices = list(range(len(labels_raw)))
        target_map: Optional[Dict[int, int]] = None
        if args.caltech_exclude_background and categories is not None:
            background_ids = [i for i, name in enumerate(categories) if str(name).lower() == "background_google"]
            if background_ids:
                bg = background_ids[0]
                keep_indices = [i for i, y in enumerate(labels_raw) if int(y) != bg]
                kept_labels = sorted(set(int(labels_raw[i]) for i in keep_indices))
                target_map = {old: new for new, old in enumerate(kept_labels)}
        labels = [target_map[int(labels_raw[i])] if target_map else int(labels_raw[i]) for i in keep_indices]
        train_rel, val_rel, test_rel = stratified_split_indices(labels, args.caltech_train_ratio, args.val_ratio, args.seed)
        train_idx = [keep_indices[i] for i in train_rel]
        val_idx = [keep_indices[i] for i in val_rel]
        test_idx = [keep_indices[i] for i in test_rel]
        train_set = TargetRemapSubset(train_base, train_idx, target_map)
        val_set = TargetRemapSubset(eval_base, val_idx, target_map) if val_idx else None
        test_set = TargetRemapSubset(eval_base, test_idx, target_map)
        num_classes = len(set(labels))
        spec = DatasetSpec(
            "caltech101",
            (3, args.image_size, args.image_size),
            num_classes,
            args.T,
            False,
            "static RGB image resized/cropped and repeated for T steps",
        )

    elif dataset_name == "cifar10dvs":
        try:
            from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
            try:
                from spikingjelly.datasets.cifar10_dvs import CIFAR10DVSTEBNSplit
            except Exception:
                CIFAR10DVSTEBNSplit = None
        except Exception as exc:
            raise RuntimeError("CIFAR10-DVS loading requires spikingjelly.datasets.") from exc

        root = data_dir / "CIFAR10-DVS"
        if args.prepare_event_data:
            ensure_cifar10dvs_prepared(root, download=args.download)
        size = (args.dvs_size, args.dvs_size) if args.dvs_size > 0 else None
        transform = EventFrameTransform(
            size=size,
            target_frames=args.T,
            binarize=args.event_binarize,
            normalize=args.event_normalize,
            resize_mode=args.event_resize_mode,
        )
        if args.cifar10dvs_split == "tebn" and CIFAR10DVSTEBNSplit is not None:
            ds_cls = CIFAR10DVSTEBNSplit
            train_full = ds_cls(
                root=str(root),
                train=True,
                data_type="frame",
                frames_number=args.T,
                split_by=args.event_split_by,
                transform=transform,
            )
            test_set = ds_cls(
                root=str(root),
                train=False,
                data_type="frame",
                frames_number=args.T,
                split_by=args.event_split_by,
                transform=transform,
            )
        else:
            full = CIFAR10DVS(
                root=str(root),
                data_type="frame",
                frames_number=args.T,
                split_by=args.event_split_by,
                transform=transform,
            )
            labels = get_targets(full)
            train_rel, val_rel, test_rel = stratified_split_indices(labels, args.event_train_ratio, args.val_ratio, args.seed)
            train_full = Subset(full, train_rel + val_rel)  # split val below on the train part for common flow
            test_set = Subset(full, test_rel)
        labels_train = get_targets(train_full)
        train_idx, val_idx = split_train_val_indices(labels_train, args.val_ratio, args.seed)
        train_set = Subset(train_full, train_idx)
        val_set = Subset(train_full, val_idx) if val_idx else None
        final_size = args.dvs_size if args.dvs_size > 0 else 128
        spec = DatasetSpec("cifar10dvs", (2, final_size, final_size), 10, args.T, True, "event frames [T,2,H,W]")

    elif dataset_name == "ncaltech101":
        try:
            from spikingjelly.datasets.n_caltech101 import NCaltech101
        except Exception as exc:
            raise RuntimeError("N-Caltech101 loading requires spikingjelly.datasets.") from exc

        root = data_dir / "N-Caltech101"
        if args.prepare_event_data:
            ensure_ncaltech101_prepared(root, download=args.download)
        size = (args.dvs_size, args.dvs_size) if args.dvs_size > 0 else None
        transform = EventFrameTransform(
            size=size,
            target_frames=args.T,
            binarize=args.event_binarize,
            normalize=args.event_normalize,
            resize_mode=args.event_resize_mode,
        )
        full = NCaltech101(
            root=str(root),
            data_type="frame",
            frames_number=args.T,
            split_by=args.event_split_by,
            transform=transform,
        )
        labels = get_targets(full)
        train_rel, val_rel, test_rel = stratified_split_indices(labels, args.event_train_ratio, args.val_ratio, args.seed)
        train_set = Subset(full, train_rel)
        val_set = Subset(full, val_rel) if val_rel else None
        test_set = Subset(full, test_rel)
        event_input_shape = (2, args.dvs_size, args.dvs_size) if args.dvs_size > 0 else (2, 180, 240)
        spec = DatasetSpec(
            "ncaltech101",
            event_input_shape,
            len(set(labels)),
            args.T,
            True,
            "N-Caltech101 event frames [T,2,H,W], resized after SpikingJelly integration",
        )

    elif dataset_name == "cifar100dvs":
        if args.dvscifar100_source != "i2e":
            raise ValueError("Currently --dataset cifar100dvs supports --dvscifar100-source i2e.")
        hf_cache = data_dir / "hf_i2e_cache"
        size = (args.dvs_size, args.dvs_size) if args.dvs_size > 0 else (args.i2e_size, args.i2e_size)
        transform = EventFrameTransform(
            size=size,
            target_frames=args.T,
            binarize=args.event_binarize,
            normalize=args.event_normalize,
            resize_mode=args.event_resize_mode,
        )
        train_full = I2EEventDataset(
            cache_dir=hf_cache,
            config_name=args.i2e_cifar100_config,
            split="train",
            transform=transform,
            hf_endpoint=args.hf_endpoint,
        )
        test_set = I2EEventDataset(
            cache_dir=hf_cache,
            config_name=args.i2e_cifar100_config,
            split=args.i2e_validation_split,
            transform=transform,
            hf_endpoint=args.hf_endpoint,
        )
        labels_train = get_targets(train_full)
        train_idx, val_idx = split_train_val_indices(labels_train, args.val_ratio, args.seed)
        train_set = Subset(train_full, train_idx)
        val_set = Subset(train_full, val_idx) if val_idx else None
        final_size = args.dvs_size if args.dvs_size > 0 else args.i2e_size
        spec = DatasetSpec(
            "cifar100dvs",
            (2, final_size, final_size),
            100,
            args.T,
            True,
            "I2E-CIFAR100 event frames [T,2,H,W] from Hugging Face, not Poisson encoded",
        )

    elif dataset_name == "dvsgesture":
        try:
            from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
        except Exception as exc:
            raise RuntimeError("DVS128-Gesture loading requires spikingjelly.datasets.") from exc

        root = data_dir / "DVS128Gesture"
        if args.prepare_event_data:
            ensure_dvsgesture_prepared(root, download=args.download)
        size = (args.dvs_size, args.dvs_size) if args.dvs_size > 0 else None
        transform = EventFrameTransform(
            size=size,
            target_frames=args.T,
            binarize=args.event_binarize,
            normalize=args.event_normalize,
            resize_mode=args.event_resize_mode,
        )
        train_full = DVS128Gesture(
            root=str(root),
            train=True,
            data_type="frame",
            frames_number=args.T,
            split_by=args.event_split_by,
            transform=transform,
        )
        test_set = DVS128Gesture(
            root=str(root),
            train=False,
            data_type="frame",
            frames_number=args.T,
            split_by=args.event_split_by,
            transform=transform,
        )
        labels_train = get_targets(train_full)
        train_idx, val_idx = split_train_val_indices(labels_train, args.val_ratio, args.seed)
        train_set = Subset(train_full, train_idx)
        val_set = Subset(train_full, val_idx) if val_idx else None
        final_size = args.dvs_size if args.dvs_size > 0 else 128
        spec = DatasetSpec("dvsgesture", (2, final_size, final_size), 11, args.T, True, "event frames [T,2,H,W]")

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    if args.mode == "test":
        train_set = None
        val_set = None

    train_set = limit_dataset(train_set, args.limit_train) if train_set is not None else None
    val_set = limit_dataset(val_set, args.limit_val) if val_set is not None else None
    test_set = limit_dataset(test_set, args.limit_test)
    return train_set, val_set, test_set, spec


def build_loader(dataset: Optional[Dataset], batch_size: int, shuffle: bool, args: argparse.Namespace) -> Optional[DataLoader]:
    if dataset is None:
        return None
    sampler = DistributedSampler(dataset, shuffle=shuffle) if getattr(args, "distributed", False) else None
    kwargs = dict(
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )
    if args.num_workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **kwargs)


def make_time_sequence(x: torch.Tensor, spec: DatasetSpec) -> torch.Tensor:
    """Return [T,N,C,H,W] for both static and event datasets."""
    if spec.is_event_dataset:
        if x.ndim != 5:
            raise ValueError(f"Event frames must have shape [N,T,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] != spec.time_steps:
            raise ValueError(f"Expected T={spec.time_steps}, got input T={x.shape[1]}")
        return x.transpose(0, 1).contiguous()

    if x.ndim != 4:
        raise ValueError(f"Static images must have shape [N,C,H,W], got {tuple(x.shape)}")
    return x.unsqueeze(0).expand(spec.time_steps, *x.shape).contiguous()


def configure_snn_backend(model: nn.Module, args: argparse.Namespace, device: torch.device) -> str:
    functional.set_step_mode(model, "m")
    backend = "torch"
    if args.cupy:
        if device.type != "cuda":
            warnings.warn("--cupy was requested but the selected device is not CUDA. Falling back to torch backend.")
        else:
            try:
                import cupy  # noqa: F401
                functional.set_backend(model, "cupy", instance=neuron.LIFNode)
                if hasattr(neuron, "ParametricLIFNode"):
                    functional.set_backend(model, "cupy", instance=neuron.ParametricLIFNode)
                backend = "cupy"
            except Exception as exc:
                warnings.warn(f"CuPy backend could not be enabled ({exc}). Falling back to torch backend.")
    return backend


def model_config_from_args(args: argparse.Namespace, spec: DatasetSpec) -> Dict:
    return {
        "input_shape": list(spec.input_shape),
        "num_classes": spec.num_classes,
        "vgg_depth": args.vgg_depth,
        "base_channels": args.base_channels,
        "max_channels": args.max_channels,
        "fc_dim": args.fc_dim,
        "pool_output_size": args.pool_output_size,
        "pool_type": args.pool_type,
        "norm": args.norm,
        "tdbn_alpha": args.tdbn_alpha,
        "tdbn_eps": args.tdbn_eps,
        "tdbn_momentum": args.tdbn_momentum,
        "dropout": args.dropout,
        "conv_dropout": args.conv_dropout,
        "tau": args.tau,
        "v_threshold": args.v_threshold,
        "v_reset": None if args.v_reset.lower() == "none" else float(args.v_reset),
        "detach_reset": args.detach_reset,
        "surrogate_name": args.surrogate,
        "neuron_type": args.neuron,
        "readout": args.readout,
    }


def build_model_from_config(config: Dict) -> VGGSNN:
    return VGGSNN(
        input_shape=tuple(int(v) for v in config["input_shape"]),
        num_classes=int(config["num_classes"]),
        vgg_depth=int(config["vgg_depth"]),
        base_channels=int(config["base_channels"]),
        max_channels=int(config["max_channels"]),
        fc_dim=int(config["fc_dim"]),
        pool_output_size=int(config["pool_output_size"]),
        pool_type=str(config["pool_type"]),
        norm=str(config["norm"]),
        tdbn_alpha=float(config["tdbn_alpha"]),
        tdbn_eps=float(config["tdbn_eps"]),
        tdbn_momentum=float(config["tdbn_momentum"]),
        dropout=float(config["dropout"]),
        conv_dropout=float(config["conv_dropout"]),
        tau=float(config["tau"]),
        v_threshold=float(config["v_threshold"]),
        v_reset=config["v_reset"],
        detach_reset=bool(config["detach_reset"]),
        surrogate_name=str(config["surrogate_name"]),
        neuron_type=str(config["neuron_type"]),
        readout=str(config["readout"]),
    )


def checkpoint_path(args: argparse.Namespace) -> Path:
    if args.ckpt:
        return Path(args.ckpt).expanduser().resolve()
    return Path(args.out_dir).expanduser().resolve() / args.save_name


def load_checkpoint(path: Path, device: torch.device) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def maybe_apply_checkpoint_args(args: argparse.Namespace, ckpt: Optional[Dict]) -> None:
    if ckpt is None or args.ignore_ckpt_config:
        return
    ds_cfg = ckpt.get("dataset_spec")
    if isinstance(ds_cfg, dict):
        args.dataset = str(ds_cfg.get("name", args.dataset))
        args.T = int(ds_cfg.get("time_steps", args.T))
        input_shape = ds_cfg.get("input_shape")
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 3:
            if canonical_dataset_name(args.dataset) in {"caltech101"}:
                args.image_size = int(input_shape[-1])
            if canonical_dataset_name(args.dataset) in {"cifar10dvs", "cifar100dvs", "ncaltech101", "dvsgesture"}:
                # If H and W differ (native N-Caltech101), keep --dvs-size=0 so the loader uses native dimensions.
                args.dvs_size = int(input_shape[-1]) if int(input_shape[-1]) == int(input_shape[-2]) else 0


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    args: argparse.Namespace,
    spec: DatasetSpec,
    model_config: Dict,
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    if not is_main_process():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = model_without_ddp(model)
    payload = {
        "epoch": epoch,
        "model_state": model_to_save.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "dataset_spec": asdict(spec),
        "model_config": model_config,
        "args": vars(args),
        "metrics": metrics,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(payload, path)


@contextmanager
def autocast_context(device: torch.device, enabled: bool):
    enabled = bool(enabled and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        with torch.amp.autocast(device_type=device.type, enabled=enabled):
            yield
    else:  # pragma: no cover - for older PyTorch
        with torch.cuda.amp.autocast(enabled=enabled):
            yield


def build_grad_scaler(device: torch.device, enabled: bool):
    enabled = bool(enabled and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device.type, enabled=enabled)
        except TypeError:  # pragma: no cover - version compatibility
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)  # pragma: no cover


def optimizer_step(
    loss: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    grad_clip: float,
) -> None:
    scaler.scale(loss).backward()
    if grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    criterion: nn.Module,
    device: torch.device,
    spec: DatasetSpec,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    t0 = time.time()

    use_tbptt = args.tbptt
    chunk_k = args.tbptt_k if use_tbptt else spec.time_steps
    if chunk_k <= 0:
        raise ValueError("--tbptt-k must be positive")
    chunk_k = min(chunk_k, spec.time_steps)

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long()
        x_seq = make_time_sequence(x, spec)
        batch_size = y.numel()

        functional.reset_net(model)
        if not use_tbptt:
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.amp):
                out_seq = model(x_seq)
                logits = out_seq.mean(dim=0)
                loss = criterion(logits, y)
            optimizer_step(loss, model, optimizer, scaler, args.grad_clip)
            batch_loss = float(loss.detach())
            pred = logits.detach().argmax(dim=1)
        else:
            # Temporally truncated BPTT: update after every k-step chunk, then detach states.
            logits_sum = None
            weighted_loss_sum = 0.0
            total_steps = 0
            for start in range(0, spec.time_steps, chunk_k):
                end = min(start + chunk_k, spec.time_steps)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, args.amp):
                    out_chunk = model(x_seq[start:end])
                    logits_chunk = out_chunk.mean(dim=0)
                    loss = criterion(logits_chunk, y)
                optimizer_step(loss, model, optimizer, scaler, args.grad_clip)

                weight = end - start
                weighted_loss_sum += float(loss.detach()) * weight
                total_steps += weight
                detached_logits = logits_chunk.detach() * weight
                logits_sum = detached_logits if logits_sum is None else logits_sum + detached_logits

                # Identical temporal truncation principle to the uploaded MLP code:
                # hidden states are carried forward but their computation graph is cut.
                functional.detach_net(model)

            batch_loss = weighted_loss_sum / max(total_steps, 1)
            pred = (logits_sum / max(total_steps, 1)).argmax(dim=1)

        functional.reset_net(model)
        total_loss += batch_loss * batch_size
        total_correct += int((pred == y).sum())
        total_seen += batch_size

        if is_main_process() and args.log_interval > 0 and (batch_idx + 1) % args.log_interval == 0:
            print(
                f"  batch {batch_idx + 1:05d}/{len(loader):05d} "
                f"loss={total_loss / max(total_seen, 1):.4f} "
                f"acc={total_correct / max(total_seen, 1):.4f}"
            )

    seconds = time.time() - t0
    total_loss, total_correct, total_seen, seconds = reduce_epoch_totals(
        total_loss, total_correct, total_seen, seconds, device
    )
    return {
        "loss": total_loss / max(total_seen, 1),
        "acc": total_correct / max(total_seen, 1),
        "seconds": seconds,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    spec: DatasetSpec,
    args: argparse.Namespace,
    name: str = "eval",
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    t0 = time.time()
    chunk = args.eval_chunk_size if args.eval_chunk_size > 0 else spec.time_steps
    chunk = min(chunk, spec.time_steps)

    for x, y in loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long()
        x_seq = make_time_sequence(x, spec)
        functional.reset_net(model)

        logits_sum = None
        total_steps = 0
        for start in range(0, spec.time_steps, chunk):
            end = min(start + chunk, spec.time_steps)
            out_chunk = model(x_seq[start:end])
            weight = end - start
            part = out_chunk.sum(dim=0)
            logits_sum = part if logits_sum is None else logits_sum + part
            total_steps += weight
        logits = logits_sum / max(total_steps, 1)
        loss = criterion(logits, y)
        pred = logits.argmax(dim=1)
        functional.reset_net(model)

        batch_size = y.numel()
        total_loss += float(loss.detach()) * batch_size
        total_correct += int((pred == y).sum())
        total_seen += batch_size

    seconds = time.time() - t0
    total_loss, total_correct, total_seen, seconds = reduce_epoch_totals(
        total_loss, total_correct, total_seen, seconds, device
    )
    metrics = {
        "loss": total_loss / max(total_seen, 1),
        "acc": total_correct / max(total_seen, 1),
        "seconds": seconds,
    }
    if is_main_process():
        print(f"[{name}] loss={metrics['loss']:.4f} acc={metrics['acc']:.4f} time={metrics['seconds']:.1f}s")
    return metrics


def build_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer):
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    if args.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    raise ValueError(args.lr_scheduler)


def run_train(args: argparse.Namespace, device: torch.device) -> Tuple[Path, Dict[str, float]]:
    train_set, val_set, _, spec = build_datasets_ddp_safe(args)
    if train_set is None:
        raise RuntimeError("Training mode requires a training dataset.")

    train_loader = build_loader(train_set, args.batch_size, True, args)
    val_loader = build_loader(val_set, args.batch_size, False, args) if val_set is not None else None

    model_config = model_config_from_args(args, spec)
    model = build_model_from_config(model_config).to(device)
    backend = configure_snn_backend(model, args, device)
    if args.compile:
        try:
            model = torch.compile(model)  # type: ignore[assignment]
        except Exception as exc:
            warnings.warn(f"torch.compile failed: {exc}. Continuing without compilation.")

    model = wrap_model_for_ddp(model, args, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(args, optimizer)
    criterion = nn.CrossEntropyLoss()
    scaler = build_grad_scaler(device, args.amp)
    best_path = checkpoint_path(args)

    if is_main_process():
        print("=" * 88)
        print(f"Dataset       : {spec.name}  input_shape={spec.input_shape}  classes={spec.num_classes}  T={spec.time_steps}")
        print(f"Input         : {spec.description}; no Poisson encoder")
        print(f"Model         : VGG{args.vgg_depth}-SNN base_channels={args.base_channels} max_channels={args.max_channels} norm={args.norm}")
        print(f"Training      : Adam + CE  lr={args.lr} batch_size={args.batch_size} epochs={args.epochs}")
        print(f"BPTT          : {'TBPTT' if args.tbptt else 'standard BPTT'} k={args.tbptt_k if args.tbptt else spec.time_steps}")
        print(f"Backend       : {backend} amp={args.amp and device.type == 'cuda'} device={device}")
        print(f"Best ckpt     : {best_path}")
        print("=" * 88)

    best_score = -math.inf
    best_metrics: Dict[str, float] = {}
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        if isinstance(getattr(train_loader, "sampler", None), DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        if is_main_process():
            print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, spec, args)
        if is_main_process():
            print(
                f"[train] loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f} "
                f"time={train_metrics['seconds']:.1f}s"
            )

        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, criterion, device, spec, args, name="val")
        else:
            val_metrics = train_metrics

        if scheduler is not None:
            scheduler.step()
            if is_main_process():
                print(f"[lr] {optimizer.param_groups[0]['lr']:.6g}")

        monitor = val_metrics["acc"] if args.monitor == "acc" else -val_metrics["loss"]
        monitor_display = val_metrics[args.monitor]
        improved = monitor > (best_score + args.min_delta)
        if improved:
            best_score = monitor
            best_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
            best_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            save_checkpoint(best_path, model, optimizer, scheduler, args, spec, model_config, epoch, best_metrics)
            if is_main_process():
                print(f"[checkpoint] saved best model at epoch={epoch}, {args.monitor}={monitor_display:.6f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if is_main_process():
                print(
                    f"[early-stop] no improvement: {epochs_without_improvement}/"
                    f"{args.patience if args.patience > 0 else 'disabled'}"
                )

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            if is_main_process():
                print(f"[early-stop] stopped at epoch {epoch}")
            break

    if is_main_process():
        config_path = best_path.with_suffix(".json")
        config_payload = {
            "dataset_spec": asdict(spec),
            "model_config": model_config,
            "best_metrics": best_metrics,
            "checkpoint": str(best_path),
        }
        config_path.write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return best_path, best_metrics


def run_test(args: argparse.Namespace, device: torch.device) -> Dict[str, float]:
    ckpt_path = checkpoint_path(args)
    ckpt = load_checkpoint(ckpt_path, device)
    maybe_apply_checkpoint_args(args, ckpt)
    _, _, test_set, spec = build_datasets_ddp_safe(args)

    model_config = ckpt.get("model_config")
    if not isinstance(model_config, dict):
        model_config = model_config_from_args(args, spec)
    model = build_model_from_config(model_config).to(device)
    configure_snn_backend(model, args, device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    test_loader = build_loader(test_set, args.batch_size, False, args)
    criterion = nn.CrossEntropyLoss()
    if is_main_process():
        print(f"[test] loaded checkpoint: {ckpt_path}")
    return evaluate(model, test_loader, criterion, device, spec, args, name="test")


def run_sanity(args: argparse.Namespace, device: torch.device) -> None:
    """Fast random-data forward/backward check. No dataset download is used."""
    print("[sanity] Running random-data checks. No dataset download is used.")
    original_T = args.T
    args.T = args.T if args.T > 0 else 4
    dvs_shape_48 = (2, args.dvs_size if args.dvs_size > 0 else 48, args.dvs_size if args.dvs_size > 0 else 48)
    specs = [
        DatasetSpec("cifar10", (3, 32, 32), 10, args.T, False),
        DatasetSpec("cifar100", (3, 32, 32), 100, args.T, False),
        DatasetSpec("caltech101", (3, args.image_size, args.image_size), 101, args.T, False),
        DatasetSpec("cifar10dvs", dvs_shape_48, 10, args.T, True),
        DatasetSpec("cifar100dvs", dvs_shape_48, 100, args.T, True),
        DatasetSpec("ncaltech101", dvs_shape_48, 101, args.T, True),
        DatasetSpec("dvsgesture", (2, args.dvs_size if args.dvs_size > 0 else 64, args.dvs_size if args.dvs_size > 0 else 64), 11, args.T, True),
    ]

    for spec in specs:
        model_config = model_config_from_args(args, spec)
        model = build_model_from_config(model_config).to(device)
        configure_snn_backend(model, args, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        scaler = build_grad_scaler(device, args.amp)
        functional.reset_net(model)
        model.train()

        b = args.sanity_batch_size
        if spec.is_event_dataset:
            x = torch.rand(b, spec.time_steps, *spec.input_shape, device=device)
            x = (x > 0.95).float()
        else:
            x = torch.rand(b, *spec.input_shape, device=device)
        y = torch.randint(0, spec.num_classes, (b,), device=device)
        x_seq = make_time_sequence(x, spec)

        if args.tbptt:
            k = min(args.tbptt_k, spec.time_steps)
            for start in range(0, spec.time_steps, k):
                end = min(start + k, spec.time_steps)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, args.amp):
                    logits = model(x_seq[start:end]).mean(0)
                    loss = criterion(logits, y)
                optimizer_step(loss, model, optimizer, scaler, args.grad_clip)
                functional.detach_net(model)
        else:
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.amp):
                logits = model(x_seq).mean(0)
                loss = criterion(logits, y)
            optimizer_step(loss, model, optimizer, scaler, args.grad_clip)
        functional.reset_net(model)
        print(f"[sanity:{spec.name}] ok  input={spec.input_shape} classes={spec.num_classes}")
    args.T = original_T


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="VGG-SNN for CIFAR-10/100, CIFAR10-DVS, CIFAR100-DVS/I2E-CIFAR100, Caltech-101, N-Caltech101, and DVS-Gesture with BPTT or TBPTT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Run mode / data
    p.add_argument("--mode", choices=["train", "test", "train_test", "sanity"], default="train_test")
    p.add_argument(
        "--dataset",
        choices=[
            "cifar10", "cifar100",
            "cifar10dvs", "dvs-cifar10", "dvscifar10",
            "cifar100dvs", "dvs-cifar100", "dvscifar100", "i2e-cifar100",
            "caltech101", "ncaltech101", "n-caltech101",
            "dvsgesture", "dvs128gesture",
        ],
        default="dvsgesture",
    )
    p.add_argument("--data-dir", type=str, default="/home/leehyunjong/PycharmProjects/Machine_Learning/SNN/TA_BPTT/Motivation/data")
    p.add_argument("--out-dir", type=str, default="./runs/snn_vgg")
    p.add_argument("--save-name", type=str, default="best_model.pt")
    p.add_argument("--ckpt", type=str, default="", help="Checkpoint path for test mode. Defaults to out-dir/save-name.")
    p.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--prepare-event-data", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ignore-ckpt-config", action="store_true")

    # Core hyperparameters
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)    # 32 for DVS
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--T", type=int, default=16, help="0 selects a dataset-specific default timestep count.")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lr-scheduler", choices=["none", "cosine", "step"], default="none")
    p.add_argument("--lr-step-size", type=int, default=20)
    p.add_argument("--lr-gamma", type=float, default=0.5)

    # VGG architecture
    p.add_argument("--vgg-depth", type=int, choices=[7, 9, 11, 13, 15], default=11)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--max-channels", type=int, default=512)
    p.add_argument("--fc-dim", type=int, default=512)
    p.add_argument("--pool-output-size", type=int, default=1)
    p.add_argument("--pool-type", choices=["avg", "max"], default="avg")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--conv-dropout", type=float, default=0.0)
    p.add_argument("--readout", choices=["linear", "spike"], default="linear")

    # tdBN / normalization
    p.add_argument("--norm", choices=["tdbn", "bn", "none"], default="tdbn")
    p.add_argument("--tdbn-alpha", type=float, default=1.0)
    p.add_argument("--tdbn-eps", type=float, default=1e-5)
    p.add_argument("--tdbn-momentum", type=float, default=0.1)

    # SNN neuron hyperparameters
    p.add_argument("--neuron", choices=["lif", "if", "plif"], default="lif")
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--v-threshold", type=float, default=1.0)
    p.add_argument("--v-reset", type=str, default="0.0", help="Use 'None' for soft reset.")
    p.add_argument("--detach-reset", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--surrogate", choices=["atan", "sigmoid", "fast_sigmoid", "piecewise_quadratic"], default="atan")

    # BPTT / TBPTT
    p.add_argument("--tbptt", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--tbptt-k", type=int, default=1, help="Temporal truncation length k.")
    p.add_argument("--eval-chunk-size", type=int, default=0, help="0 means evaluate all T steps at once.")

    # Static data preprocessing
    p.add_argument("--image-size", type=int, default=128, help="Caltech-101 image size. CIFAR-10/100 remain 32x32.")
    p.add_argument("--augment-static", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--normalize-static", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--caltech-exclude-background", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--caltech-train-ratio", type=float, default=0.7)

    # Event data preprocessing
    p.add_argument("--dvs-size", type=int, default=0, help="0 keeps native event-frame resolution; otherwise resize to dvs-size x dvs-size.")
    p.add_argument("--event-split-by", choices=["time", "number"], default="time")
    p.add_argument("--event-binarize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--event-normalize", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--event-resize-mode", choices=["nearest", "bilinear"], default="nearest")
    p.add_argument("--cifar10dvs-split", choices=["tebn", "random"], default="tebn")
    p.add_argument("--event-train-ratio", type=float, default=0.8, help="Used only for event datasets without official train/test split.")
    p.add_argument("--dvscifar100-source", choices=["i2e"], default="i2e", help="CIFAR100-DVS backend. i2e uses UESTC-BICS/I2E I2E-CIFAR100 from Hugging Face.")
    p.add_argument("--i2e-cifar100-config", type=str, default="I2E-CIFAR100")
    p.add_argument("--i2e-validation-split", type=str, default="validation")
    p.add_argument("--i2e-size", type=int, default=128, help="Native I2E-CIFAR100 spatial resolution used when --dvs-size=0.")
    p.add_argument("--hf-endpoint", type=str, default="", help="Optional Hugging Face endpoint/mirror, e.g. https://hf-mirror.com.")

    # Early stopping / validation
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=10, help="0 disables early stopping.")
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--monitor", choices=["acc", "loss"], default="acc")

    # Device / speed
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--cupy", action=argparse.BooleanOptionalAction, default=True,
                   help="Use SpikingJelly CuPy backend for LIF/PLIF nodes when CUDA is available.")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable automatic mixed precision on CUDA.")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False,
                   help="Try torch.compile(model). Disabled by default.")
    p.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ddp", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable DistributedDataParallel when launched with torchrun.")
    p.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=0,
                   help=argparse.SUPPRESS)

    # Logging / reproducibility / debug limits
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--limit-train", type=int, default=0, help="Limit train subset. 0 disables.")
    p.add_argument("--limit-val", type=int, default=0, help="Limit validation subset. 0 disables.")
    p.add_argument("--limit-test", type=int, default=0, help="Limit test subset. 0 disables.")
    p.add_argument("--sanity-batch-size", type=int, default=4)
    return p


def main() -> None:
    args = make_argparser().parse_args()
    init_distributed_mode(args)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.T < 0:
        raise ValueError("--T must be non-negative; use 0 for dataset-specific default")
    if args.tbptt and args.tbptt_k <= 0:
        raise ValueError("--tbptt-k must be positive")
    if args.dvs_size < 0:
        raise ValueError("--dvs-size must be non-negative")

    set_seed(args.seed)
    device = resolve_device(args.device)
    if getattr(args, "distributed", False) and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.local_rank}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
    else:
        args.amp = False
        args.cupy = False
        args.pin_memory = False

    Path(args.out_dir).expanduser().resolve().mkdir(parents=True, exist_ok=True)

    if args.mode == "sanity":
        run_sanity(args, device)
        cleanup_distributed()
        return

    if args.mode in {"train", "train_test"}:
        best_path, best_metrics = run_train(args, device)
        if is_main_process():
            print(f"\nBest checkpoint: {best_path}")
            if best_metrics:
                print(json.dumps(best_metrics, indent=2))
        if is_dist_avail_and_initialized():
            dist.barrier()

    if args.mode in {"test", "train_test"}:
        if args.mode == "train_test" and not args.ckpt:
            args.ckpt = str(Path(args.out_dir).expanduser().resolve() / args.save_name)
        test_metrics = run_test(args, device)
        if is_main_process():
            print("\nTest metrics:")
            print(json.dumps(test_metrics, indent=2))

    cleanup_distributed()


if __name__ == "__main__":
    main()
