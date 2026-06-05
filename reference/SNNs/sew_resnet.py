#!/usr/bin/env python3
"""
SEW-ResNet practice code with SpikingJelly.

Supported datasets:
  - CIFAR-10 / CIFAR-100: torchvision download=True
  - Tiny-ImageNet-200: optional download from the Stanford CS231n archive, with a
    custom validation loader for val_annotations.txt
  - ImageNet-1k: expects an ImageFolder-compatible root with train/ and val/

Typical commands:
  python snn_sew_resnet.py --dataset cifar10 --data-path ./data \
      --depth 18 --epochs 200 --batch-size 128 --timesteps 4 --backend cupy --amp

  torchrun --nproc_per_node=8 snn_sew_resnet.py --dataset imagenet \
      --data-path /path/to/imagenet --depth 50 --epochs 320 --batch-size 32 \
      --timesteps 4 --backend cupy --amp --channels-last
"""

from __future__ import annotations

import argparse
import json
import yaml
import math
import os
import random
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_and_extract_archive

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from spikingjelly.activation_based import functional, layer, neuron, surrogate


# -----------------------------
# Utilities
# -----------------------------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEPTH_TO_LAYERS: Dict[int, Tuple[type, List[int]]] = {
    18: (None, [2, 2, 2, 2]),   # block type is patched after class definitions
    34: (None, [3, 4, 6, 3]),
    50: (None, [3, 4, 6, 3]),
    101: (None, [3, 4, 23, 3]),
    152: (None, [3, 8, 36, 3]),
}


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process(args: argparse.Namespace) -> bool:
    return getattr(args, "rank", 0) == 0


def setup_distributed(args: argparse.Namespace) -> None:
    """Initialize DDP when launched by torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        args.distributed = True
        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend=args.dist_backend, init_method="env://")
        dist.barrier()
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.distributed = False


def cleanup_distributed() -> None:
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


def seed_everything(seed: int, rank: int = 0) -> None:
    seed = seed + rank
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(args: argparse.Namespace, msg: str) -> None:
    if is_main_process(args):
        print(msg, flush=True)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def reset_snn(model: nn.Module) -> None:
    functional.reset_net(unwrap_model(model))


def get_device(args: argparse.Namespace) -> torch.device:
    if args.device == "auto":
        return torch.device("cuda", args.local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if args.device.startswith("cuda") and args.distributed:
        return torch.device("cuda", args.local_rank)
    return torch.device(args.device)


def build_surrogate(name: str, alpha: float) -> nn.Module:
    name = name.lower()
    if name == "atan":
        return surrogate.ATan(alpha=alpha)
    if name == "sigmoid":
        return surrogate.Sigmoid(alpha=alpha)
    raise ValueError(f"Unsupported surrogate: {name}")


def get_neuron_ctor_and_kwargs(args: argparse.Namespace):
    surr = build_surrogate(args.surrogate, args.surrogate_alpha)
    common = dict(
        v_threshold=args.v_threshold,
        surrogate_function=surr,
        detach_reset=args.detach_reset,
    )
    name = args.neuron.upper()
    if name == "IF":
        return neuron.IFNode, common
    if name == "LIF":
        kwargs = dict(common)
        kwargs.update(tau=args.tau, decay_input=args.decay_input)
        return neuron.LIFNode, kwargs
    if name == "PLIF":
        kwargs = dict(common)
        # Current SpikingJelly versions use init_tau for ParametricLIFNode.
        kwargs.update(init_tau=args.tau, decay_input=args.decay_input)
        return neuron.ParametricLIFNode, kwargs
    raise ValueError(f"Unsupported neuron: {args.neuron}")


def set_spikingjelly_fast_mode(model: nn.Module, args: argparse.Namespace, device: torch.device) -> None:
    """Use multi-step simulation, and optionally cupy/triton neuron backend."""
    functional.set_step_mode(model, "m")
    if args.backend == "torch":
        return
    if device.type != "cuda":
        raise RuntimeError(f"backend={args.backend} requires CUDA; current device is {device}.")
    # SpikingJelly exposes backend on neuron modules. Set available neuron types.
    for cls_name in ("IFNode", "LIFNode", "ParametricLIFNode"):
        cls = getattr(neuron, cls_name, None)
        if cls is None:
            continue
        try:
            functional.set_backend(model, args.backend, instance=cls)
        except Exception as exc:
            print(f"[warning] failed to set backend={args.backend} for {cls_name}: {exc}", file=sys.stderr)


def make_time_sequence(images: torch.Tensor, timesteps: int) -> torch.Tensor:
    """Direct static-image encoding: repeat the same image for T time-steps.

    The first Conv-BN-SN path receives analog image intensities at every time-step,
    which is the common direct training setup for static image SNNs.
    """
    return images.unsqueeze(0).repeat(timesteps, 1, 1, 1, 1)


# -----------------------------
# SEW-ResNet implementation
# -----------------------------

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Module:
    return layer.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Module:
    return layer.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def sew_function(a: torch.Tensor, s: torch.Tensor, cnf: str) -> torch.Tensor:
    """Spike-element-wise residual function.

    a: residual branch spike A^l[t]
    s: shortcut spike S^l[t]
    """
    cnf = cnf.upper()
    if cnf == "ADD":
        return a + s
    if cnf == "AND":
        return a * s
    if cnf == "IAND":
        return (1.0 - a) * s
    raise ValueError(f"Unsupported SEW connect function: {cnf}")


class SEWBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[type] = None,
        cnf: str = "ADD",
        spiking_neuron: Optional[type] = None,
        neuron_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("SEWBasicBlock supports only groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 is not supported in SEWBasicBlock")
        if spiking_neuron is None:
            spiking_neuron = neuron.IFNode
        neuron_kwargs = neuron_kwargs or {}

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn1 = spiking_neuron(**deepcopy(neuron_kwargs))
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.sn2 = spiking_neuron(**deepcopy(neuron_kwargs))
        self.downsample = downsample
        self.downsample_sn = spiking_neuron(**deepcopy(neuron_kwargs)) if downsample is not None else None
        self.cnf = cnf.upper()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)

        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))

        return sew_function(out, identity, self.cnf)


class SEWBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[type] = None,
        cnf: str = "ADD",
        spiking_neuron: Optional[type] = None,
        neuron_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if spiking_neuron is None:
            spiking_neuron = neuron.IFNode
        neuron_kwargs = neuron_kwargs or {}

        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.sn1 = spiking_neuron(**deepcopy(neuron_kwargs))
        # Torchvision/SpikingJelly ResNet-v1.5 style: stride in the 3x3 conv.
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.sn2 = spiking_neuron(**deepcopy(neuron_kwargs))
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.sn3 = spiking_neuron(**deepcopy(neuron_kwargs))
        self.downsample = downsample
        self.downsample_sn = spiking_neuron(**deepcopy(neuron_kwargs)) if downsample is not None else None
        self.cnf = cnf.upper()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.sn3(out)

        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))

        return sew_function(out, identity, self.cnf)


class SEWResNet(nn.Module):
    def __init__(
        self,
        block: type,
        layers_cfg: Sequence[int],
        num_classes: int,
        stem: str,
        cnf: str,
        spiking_neuron: type,
        neuron_kwargs: dict,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[Sequence[bool]] = None,
        norm_layer: Optional[type] = None,
        and_identity_bias: float = 1.0,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.stem = stem
        self.cnf = cnf.upper()
        self.num_classes = num_classes

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation must be None or a 3-element sequence")

        if stem == "imagenet":
            self.conv1 = layer.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif stem == "cifar":
            self.conv1 = layer.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()
        else:
            raise ValueError(f"Unsupported stem: {stem}")

        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = spiking_neuron(**deepcopy(neuron_kwargs))

        self.layer1 = self._make_layer(block, 64, layers_cfg[0], cnf=cnf, spiking_neuron=spiking_neuron, neuron_kwargs=neuron_kwargs)
        self.layer2 = self._make_layer(
            block,
            128,
            layers_cfg[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            cnf=cnf,
            spiking_neuron=spiking_neuron,
            neuron_kwargs=neuron_kwargs,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers_cfg[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            cnf=cnf,
            spiking_neuron=spiking_neuron,
            neuron_kwargs=neuron_kwargs,
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers_cfg[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            cnf=cnf,
            spiking_neuron=spiking_neuron,
            neuron_kwargs=neuron_kwargs,
        )
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.fc = layer.Linear(512 * block.expansion, num_classes)

        self._init_weights()
        if zero_init_residual:
            self._zero_init_residual(and_identity_bias=and_identity_bias)

    def _make_layer(
        self,
        block: type,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        cnf: str = "ADD",
        spiking_neuron: Optional[type] = None,
        neuron_kwargs: Optional[dict] = None,
    ) -> nn.Sequential:
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

        layers_list = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                cnf,
                spiking_neuron,
                neuron_kwargs,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers_list.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    cnf=cnf,
                    spiking_neuron=spiking_neuron,
                    neuron_kwargs=neuron_kwargs,
                )
            )
        return nn.Sequential(*layers_list)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, layer.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, layer.BatchNorm2d, nn.GroupNorm)):
                if getattr(m, "weight", None) is not None:
                    nn.init.constant_(m.weight, 1)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear, layer.Linear)):
                nn.init.normal_(m.weight, 0, 0.01)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)

    def _zero_init_residual(self, and_identity_bias: float = 1.0) -> None:
        # For ADD/IAND, A=0 yields identity. For AND, A=1 is needed.
        for m in self.modules():
            target_bn = None
            if isinstance(m, SEWBottleneck):
                target_bn = m.bn3
            elif isinstance(m, SEWBasicBlock):
                target_bn = m.bn2
            if target_bn is not None:
                nn.init.constant_(target_bn.weight, 0)
                if self.cnf == "AND":
                    nn.init.constant_(target_bn.bias, and_identity_bias)
                else:
                    nn.init.constant_(target_bn.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # multi-step shape: [T, N, C, 1, 1] -> [T, N, C]
        if x.dim() == 5:
            x = torch.flatten(x, 2)
        else:
            x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resolve_stem(dataset_name: str, stem_arg: str) -> str:
    if stem_arg != "auto":
        return stem_arg
    # CIFAR and Tiny-ImageNet are small-resolution datasets; the 3x3 no-maxpool stem
    # avoids overly aggressive early downsampling.
    if dataset_name in {"cifar10", "cifar100", "tiny-imagenet"}:
        return "cifar"
    return "imagenet"


def build_model(args: argparse.Namespace, num_classes: int) -> SEWResNet:
    if args.depth not in DEPTH_TO_LAYERS:
        raise ValueError(f"Unsupported depth={args.depth}; choose one of {sorted(DEPTH_TO_LAYERS)}")
    block = SEWBasicBlock if args.depth in (18, 34) else SEWBottleneck
    _, layers_cfg = DEPTH_TO_LAYERS[args.depth]
    spiking_neuron, neuron_kwargs = get_neuron_ctor_and_kwargs(args)
    stem = resolve_stem(args.dataset, args.stem)
    model = SEWResNet(
        block=block,
        layers_cfg=layers_cfg,
        num_classes=num_classes,
        stem=stem,
        cnf=args.connect_f,
        spiking_neuron=spiking_neuron,
        neuron_kwargs=neuron_kwargs,
        zero_init_residual=args.zero_init_residual,
        and_identity_bias=args.v_threshold,
    )
    return model


# -----------------------------
# Datasets
# -----------------------------

class TinyImageNetValDataset(Dataset):
    def __init__(self, root: Path, class_to_idx: Dict[str, int], transform=None) -> None:
        self.root = Path(root)
        self.transform = transform
        ann_file = self.root / "val" / "val_annotations.txt"
        image_dir = self.root / "val" / "images"
        if not ann_file.is_file():
            raise FileNotFoundError(f"Missing Tiny-ImageNet annotation file: {ann_file}")
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing Tiny-ImageNet val image dir: {image_dir}")

        self.samples: List[Tuple[Path, int]] = []
        with ann_file.open("r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                filename, wnid = parts[0], parts[1]
                if wnid not in class_to_idx:
                    continue
                self.samples.append((image_dir / filename, class_to_idx[wnid]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        img = default_loader(str(path))
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def maybe_download_tiny_imagenet(data_path: Path, args: argparse.Namespace) -> Path:
    """Return a path containing tiny-imagenet-200/ files."""
    data_path = Path(data_path)
    if (data_path / "wnids.txt").is_file() and (data_path / "train").is_dir():
        return data_path
    tiny_root = data_path / "tiny-imagenet-200"
    if (tiny_root / "wnids.txt").is_file() and (tiny_root / "train").is_dir():
        return tiny_root

    if not args.download:
        raise FileNotFoundError(
            f"Tiny-ImageNet was not found under {data_path}. Expected either {data_path}/wnids.txt "
            f"or {data_path}/tiny-imagenet-200/wnids.txt. Re-run with --download to fetch it."
        )

    data_path.mkdir(parents=True, exist_ok=True)
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    log(args, f"Downloading Tiny-ImageNet-200 from {url} into {data_path} ...")
    download_and_extract_archive(url, download_root=str(data_path), filename="tiny-imagenet-200.zip")
    if not tiny_root.is_dir():
        raise RuntimeError(f"Tiny-ImageNet download finished, but {tiny_root} was not found.")
    return tiny_root


def tiny_imagenet_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(image_size + 8),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def imagenet_transforms(image_size: int, val_resize: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(val_resize),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def cifar_transforms(dataset_name: str) -> Tuple[transforms.Compose, transforms.Compose]:
    if dataset_name == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    else:
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf


def build_datasets(args: argparse.Namespace) -> Tuple[Dataset, Dataset, int]:
    # Only rank 0 downloads; other ranks wait until data exist.
    if args.distributed and not is_main_process(args):
        dist.barrier()

    root = Path(args.data_path)
    ds = args.dataset
    if ds in {"cifar10", "cifar100"}:
        train_tf, val_tf = cifar_transforms(ds)
        dataset_cls = datasets.CIFAR10 if ds == "cifar10" else datasets.CIFAR100
        num_classes = 10 if ds == "cifar10" else 100
        train_set = dataset_cls(root=str(root), train=True, transform=train_tf, download=args.download and is_main_process(args))
        val_set = dataset_cls(root=str(root), train=False, transform=val_tf, download=args.download and is_main_process(args))
    elif ds == "tiny-imagenet":
        tiny_root = maybe_download_tiny_imagenet(root, args)
        train_tf, val_tf = tiny_imagenet_transforms(args.image_size)
        train_set = datasets.ImageFolder(str(tiny_root / "train"), transform=train_tf)
        val_set = TinyImageNetValDataset(tiny_root, train_set.class_to_idx, transform=val_tf)
        num_classes = 200
    elif ds == "imagenet":
        train_dir = root / "train"
        val_dir = root / "val"
        if not train_dir.is_dir() or not val_dir.is_dir():
            raise FileNotFoundError(
                f"ImageNet-1k path must contain train/ and val/ class subdirectories. Got: {root}. "
                "The script intentionally does not download ImageNet."
            )
        train_tf, val_tf = imagenet_transforms(args.image_size, args.val_resize)
        train_set = datasets.ImageFolder(str(train_dir), transform=train_tf)
        val_set = datasets.ImageFolder(str(val_dir), transform=val_tf)
        num_classes = 1000
    else:
        raise ValueError(f"Unsupported dataset: {ds}")

    if args.distributed and is_main_process(args):
        dist.barrier()
    return train_set, val_set, num_classes


def build_loaders(args: argparse.Namespace, train_set: Dataset, val_set: Dataset) -> Tuple[DataLoader, DataLoader]:
    train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True) if args.distributed else None
    val_sampler = DistributedSampler(val_set, shuffle=False, drop_last=False) if args.distributed else None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
        drop_last=False,
        persistent_workers=args.workers > 0,
    )
    return train_loader, val_loader


# -----------------------------
# Checkpoints and pretrained weights
# -----------------------------

def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        while k.startswith("module."):
            k = k[len("module."):]
        out[k] = v
    return out


def extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "net", "network"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError("Checkpoint does not contain a state_dict-like object")


def compatible_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())
    cleaned = strip_module_prefix(state_dict)
    selected: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []

    for k, v in cleaned.items():
        candidates = [k]
        if k.endswith(".weight"):
            candidates.append(k[:-len(".weight")] + ".parametrizations.weight.original")
        if ".parametrizations.weight.original" in k:
            candidates.append(k.replace(".parametrizations.weight.original", ".weight"))

        matched = False
        for ck in candidates:
            if ck in model_keys and tuple(model_sd[ck].shape) == tuple(v.shape):
                selected[ck] = v
                matched = True
                break
        if not matched:
            skipped.append(k)
    return selected, skipped


def load_model_weights(model: nn.Module, path: Path, strict: bool = False, map_location="cpu") -> None:
    checkpoint = torch.load(str(path), map_location=map_location)
    state_dict = extract_state_dict(checkpoint)
    if strict:
        model.load_state_dict(strip_module_prefix(state_dict), strict=True)
        return
    selected, skipped = compatible_state_dict(model, state_dict)
    missing, unexpected = model.load_state_dict(selected, strict=False)
    print(
        f"Loaded weights from {path}. used={len(selected)}, skipped={len(skipped)}, "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )
    if skipped:
        print("First skipped keys:", skipped[:10])


def load_torchvision_ann_pretrained(model: SEWResNet, args: argparse.Namespace, num_classes: int) -> None:
    """Load torchvision ANN ResNet weights into matching Conv/BN/FC layers.

    This is useful as a fine-tuning initialization, but it is not the paper author's
    directly trained SNN checkpoint.
    """
    import torchvision.models as tv_models

    name = f"resnet{args.depth}"
    if not hasattr(tv_models, name):
        raise ValueError(f"torchvision does not expose {name}")

    weights_name = f"ResNet{args.depth}_Weights"
    weights_enum = getattr(tv_models, weights_name, None)
    weights = weights_enum.DEFAULT if weights_enum is not None else "DEFAULT"
    ann = getattr(tv_models, name)(weights=weights)
    sd = ann.state_dict()

    # CIFAR/Tiny stems have a different conv1 shape; non-1000-class heads have different fc.
    filtered = {}
    model_sd = model.state_dict()
    for k, v in sd.items():
        if k.startswith("fc.") and num_classes != 1000:
            continue
        if k == "conv1.weight" and model.stem != "imagenet":
            continue
        if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape):
            filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(
        f"Loaded torchvision ANN pretrained {name}: used={len(filtered)}, "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )


def download_author_checkpoint(args: argparse.Namespace) -> Path:
    """Download an author's checkpoint from the Figshare article if available.

    The original GitHub README points to Figshare article 14752998. This helper
    uses the public Figshare API at runtime and selects a file by simple keyword
    matching. Use --author-file to force an exact or substring match.
    """
    import requests

    article_id = "14752998"
    api_url = f"https://api.figshare.com/v2/articles/{article_id}"
    response = requests.get(api_url, timeout=30)
    response.raise_for_status()
    article = response.json()
    files = article.get("files", [])
    if not files:
        raise RuntimeError("Figshare API returned no files for article 14752998")

    def score_file(file_info: dict) -> int:
        name = file_info.get("name", "").lower()
        if args.author_file:
            return 10_000 if args.author_file.lower() in name else -10_000
        score = 0
        for token, weight in [
            ("sew", 10),
            (f"resnet{args.depth}", 30),
            (f"resnet-{args.depth}", 30),
            (str(args.depth), 5),
            (args.connect_f.lower(), 5),
            (args.dataset.replace("-", ""), 10),
            (args.dataset, 10),
            ("imagenet", 8 if args.dataset == "imagenet" else 0),
        ]:
            if weight and token and token in name:
                score += weight
        return score

    ranked = sorted(files, key=score_file, reverse=True)
    best = ranked[0]
    if score_file(best) < 0:
        available = [f.get("name", "<unnamed>") for f in files]
        raise RuntimeError(f"No Figshare file matched --author-file={args.author_file}. Available: {available}")

    name = best.get("name", f"author_checkpoint_{article_id}.pth")
    download_url = best.get("download_url")
    if not download_url:
        raise RuntimeError(f"Figshare file {name} has no download_url")

    ckpt_dir = Path(args.output_dir) / "author_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    target = ckpt_dir / name
    if target.exists() and target.stat().st_size > 0:
        return target

    print(f"Downloading author checkpoint: {name}")
    with requests.get(download_url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with target.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return target


def save_checkpoint(
    args: argparse.Namespace,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_acc1: float,
    filename: str,
) -> None:
    if not is_main_process(args):
        return
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_acc1": best_acc1,
        "args": vars(args),
    }
    torch.save(state, str(out_dir / filename))


def load_training_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler,
    scaler,
    path: Path,
    strict: bool = False,
    map_location="cpu",
) -> Tuple[int, float]:
    checkpoint = torch.load(str(path), map_location=map_location)
    state_dict = extract_state_dict(checkpoint)
    if strict:
        unwrap_model(model).load_state_dict(strip_module_prefix(state_dict), strict=True)
    else:
        selected, skipped = compatible_state_dict(unwrap_model(model), state_dict)
        unwrap_model(model).load_state_dict(selected, strict=False)
        print(f"Resumed model weights from {path}: used={len(selected)}, skipped={len(skipped)}")

    start_epoch = int(checkpoint.get("epoch", -1)) + 1 if isinstance(checkpoint, dict) else 0
    best_acc1 = float(checkpoint.get("best_acc1", 0.0)) if isinstance(checkpoint, dict) else 0.0

    if isinstance(checkpoint, dict) and optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if isinstance(checkpoint, dict) and scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if isinstance(checkpoint, dict) and scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return start_epoch, best_acc1


# -----------------------------
# Optional quantization
# -----------------------------

class SymmetricWeightFakeQuant(nn.Module):
    """Per-output-channel symmetric fake quantization with STE.

    This is a training-time fake-quantization option for experiments. It does not
    make CUDA inference integer-only; it constrains the learned weights as if they
    were quantized.
    """
    def __init__(self, bits: int = 8, eps: float = 1e-8) -> None:
        super().__init__()
        if bits < 2 or bits > 31:
            raise ValueError("bits must be in [2, 31]")
        self.bits = bits
        self.eps = eps
        self.qmax = 2 ** (bits - 1) - 1
        self.qmin = -2 ** (bits - 1)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        if w.numel() == 0:
            return w
        if w.ndim >= 2:
            reduce_dims = tuple(range(1, w.ndim))
            scale = w.detach().abs().amax(dim=reduce_dims, keepdim=True).clamp_min(self.eps) / self.qmax
        else:
            scale = w.detach().abs().amax().clamp_min(self.eps) / self.qmax
        w_int = torch.clamp(torch.round(w / scale), self.qmin, self.qmax)
        w_q = w_int * scale
        return w + (w_q - w).detach()


def apply_weight_fake_quant(model: nn.Module, bits: int) -> None:
    if bits >= 32:
        return
    count = 0
    for module in model.modules():
        if hasattr(module, "weight") and isinstance(getattr(module, "weight"), torch.Tensor):
            w = module.weight
            if w.ndim in (2, 4) and not parametrize.is_parametrized(module, "weight"):
                parametrize.register_parametrization(module, "weight", SymmetricWeightFakeQuant(bits))
                count += 1
    print(f"Applied {bits}-bit symmetric fake quantization to {count} weight tensors.")


def apply_dynamic_quant_for_eval(model: nn.Module, args: argparse.Namespace) -> nn.Module:
    if args.quantize_eval == "none":
        return model
    if args.quantize_eval != "dynamic":
        raise ValueError(f"Unsupported quantize-eval: {args.quantize_eval}")
    if args.distributed:
        raise RuntimeError("--quantize-eval dynamic is intended for single-process CPU evaluation.")
    # Dynamic quantization is CPU-oriented and mainly affects Linear layers.
    model_cpu = model.cpu()
    try:
        from torch.ao.quantization import quantize_dynamic
    except Exception:  # pragma: no cover
        from torch.quantization import quantize_dynamic
    qmodel = quantize_dynamic(model_cpu, {nn.Linear, layer.Linear}, dtype=torch.qint8)
    return qmodel


# -----------------------------
# Optimizer, scheduler, metrics
# -----------------------------

def build_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    if args.opt == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    if args.opt == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError(f"Unsupported optimizer: {args.opt}")


def build_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer):
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    if args.lr_scheduler == "cosine":
        min_lr_ratio = args.min_lr / args.lr if args.lr > 0 else 0.0

        def lr_lambda(epoch: int) -> float:
            if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
                return float(epoch + 1) / float(args.warmup_epochs)
            denom = max(1, args.epochs - args.warmup_epochs)
            progress = (epoch - args.warmup_epochs) / denom
            progress = min(max(progress, 0.0), 1.0)
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    raise ValueError(f"Unsupported lr scheduler: {args.lr_scheduler}")


def make_grad_scaler(args: argparse.Namespace, device: torch.device):
    enabled = args.amp and device.type == "cuda" and args.amp_dtype == "float16"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:  # pragma: no cover
        return torch.cuda.amp.GradScaler(enabled=enabled)


def amp_context(args: argparse.Namespace, device: torch.device):
    if not args.amp:
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> List[torch.Tensor]:
    maxk = min(max(topk), output.size(1))
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    res = []
    for k in topk:
        k = min(k, output.size(1))
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k)
    return res


def reduce_stats(device: torch.device, loss_sum: float, correct1: float, correct5: float, total: int) -> Tuple[float, float, float, int]:
    stats = torch.tensor([loss_sum, correct1, correct5, float(total)], device=device)
    if is_dist_avail_and_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return float(stats[0].item()), float(stats[1].item()), float(stats[2].item()), int(stats[3].item())


def train_one_epoch(
    args: argparse.Namespace,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    train_loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    if isinstance(train_loader.sampler, DistributedSampler):
        train_loader.sampler.set_epoch(epoch)

    optimizer.zero_grad(set_to_none=True)
    local_loss_sum = 0.0
    local_correct1 = 0.0
    local_correct5 = 0.0
    local_total = 0

    iterator: Iterable = train_loader
    if tqdm is not None and is_main_process(args):
        iterator = tqdm(train_loader, desc=f"epoch {epoch} train", dynamic_ncols=True)

    for step, (images, target) in enumerate(iterator):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if args.channels_last and images.dim() == 4:
            images = images.contiguous(memory_format=torch.channels_last)
        images_seq = make_time_sequence(images, args.timesteps)

        with amp_context(args, device):
            out_seq = model(images_seq)
            logits = out_seq.mean(0) if out_seq.dim() == 3 else out_seq
            loss = criterion(logits, target)
            loss_for_backward = loss / args.grad_accum_steps

        scaler.scale(loss_for_backward).backward()

        if (step + 1) % args.grad_accum_steps == 0:
            if args.clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            acc1, acc5 = accuracy(logits.float(), target, topk=(1, 5))
            batch = target.size(0)
            local_loss_sum += float(loss.item()) * batch
            local_correct1 += float(acc1.item())
            local_correct5 += float(acc5.item())
            local_total += batch

        reset_snn(model)

    # Flush remaining accumulated gradients if len(loader) is not divisible by grad_accum_steps.
    if len(train_loader) % args.grad_accum_steps != 0:
        if args.clip_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    loss_sum, correct1, correct5, total = reduce_stats(device, local_loss_sum, local_correct1, local_correct5, local_total)
    return {
        "loss": loss_sum / max(1, total),
        "acc1": 100.0 * correct1 / max(1, total),
        "acc5": 100.0 * correct5 / max(1, total),
    }


@torch.no_grad()
def evaluate(
    args: argparse.Namespace,
    model: nn.Module,
    criterion: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    local_loss_sum = 0.0
    local_correct1 = 0.0
    local_correct5 = 0.0
    local_total = 0

    iterator: Iterable = val_loader
    if tqdm is not None and is_main_process(args):
        iterator = tqdm(val_loader, desc="eval", dynamic_ncols=True)

    for images, target in iterator:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if args.channels_last and images.dim() == 4:
            images = images.contiguous(memory_format=torch.channels_last)
        images_seq = make_time_sequence(images, args.timesteps)

        with amp_context(args, device):
            out_seq = model(images_seq)
            logits = out_seq.mean(0) if out_seq.dim() == 3 else out_seq
            loss = criterion(logits, target)

        acc1, acc5 = accuracy(logits.float(), target, topk=(1, 5))
        batch = target.size(0)
        local_loss_sum += float(loss.item()) * batch
        local_correct1 += float(acc1.item())
        local_correct5 += float(acc5.item())
        local_total += batch
        reset_snn(model)

    loss_sum, correct1, correct5, total = reduce_stats(device, local_loss_sum, local_correct1, local_correct5, local_total)
    return {
        "loss": loss_sum / max(1, total),
        "acc1": 100.0 * correct1 / max(1, total),
        "acc5": 100.0 * correct5 / max(1, total),
    }


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate SEW-ResNet with SpikingJelly")

    # Dataset/model
    parser.add_argument("--dataset", default="cifar100", choices=["cifar10", "cifar100", "tiny-imagenet", "imagenet"])
    parser.add_argument("--data-path", default="./data", type=str)
    parser.add_argument("--download", dest="download", action="store_true", default=True, help="Download CIFAR/Tiny-ImageNet if missing. ImageNet is never downloaded. Default: True")
    parser.add_argument("--no-download", dest="download", action="store_false", help="Disable automatic download for CIFAR/Tiny-ImageNet")
    parser.add_argument("--depth", default=18, type=int, choices=[18, 34, 50, 101, 152])
    parser.add_argument("--stem", default="auto", choices=["auto", "cifar", "imagenet"])
    parser.add_argument("--connect-f", default="ADD", choices=["ADD", "AND", "IAND"], help="SEW element-wise function g")
    parser.add_argument("--timesteps", "-T", default=4, type=int)
    parser.add_argument("--image-size", default=None, type=int, help="Default: 32 for CIFAR, 64 for Tiny, 224 for ImageNet")
    parser.add_argument("--val-resize", default=None, type=int, help="Default: image_size + 32 for ImageNet, image_size + 8 for Tiny")

    # Neuron/surrogate
    parser.add_argument("--neuron", default="IF", choices=["IF", "LIF", "PLIF"])
    parser.add_argument("--v-threshold", default=1.0, type=float)
    parser.add_argument("--tau", default=2.0, type=float)
    parser.add_argument("--decay-input", action="store_true")
    parser.add_argument("--detach-reset", dest="detach_reset", action="store_true", default=True)
    parser.add_argument("--no-detach-reset", dest="detach_reset", action="store_false")
    parser.add_argument("--surrogate", default="atan", choices=["atan", "sigmoid"])
    parser.add_argument("--surrogate-alpha", default=2.0, type=float)
    parser.add_argument("--zero-init-residual", action="store_true")

    # Training
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch-size", "-b", default=128, type=int)
    parser.add_argument("--workers", "-j", default=4, type=int)
    parser.add_argument("--opt", default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--min-lr", default=0.0, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--lr-scheduler", default="cosine", choices=["cosine", "step", "none"])
    parser.add_argument("--warmup-epochs", default=0, type=int)
    parser.add_argument("--lr-step-size", default=80, type=int)
    parser.add_argument("--lr-gamma", default=0.1, type=float)
    parser.add_argument("--label-smoothing", default=0.0, type=float)
    parser.add_argument("--grad-accum-steps", default=1, type=int)
    parser.add_argument("--clip-grad-norm", default=0.0, type=float)

    # Acceleration / inference options
    parser.add_argument("--device", default="auto", type=str)
    parser.add_argument("--backend", default="torch", choices=["torch", "cupy", "triton"], help="SpikingJelly neuron backend")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Try torch.compile on the model")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--weight-quant-bits", default=32, type=int, help="<=31 enables STE fake weight quantization")
    parser.add_argument("--quantize-eval", default="none", choices=["none", "dynamic"], help="Dynamic int8 Linear-only CPU eval")

    # Pretrained/checkpoint
    parser.add_argument("--ann-pretrained", action="store_true", help="Initialize Conv/BN/FC from torchvision ANN ResNet weights where shapes match")
    parser.add_argument("--author-pretrained", action="store_true", help="Try to download a SEW-ResNet checkpoint from the authors' Figshare article")
    parser.add_argument("--author-file", default=None, type=str, help="Substring/exact hint for selecting a Figshare file")
    parser.add_argument("--resume", default=None, type=str, help="Checkpoint to resume or evaluate")
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--output-dir", default="./outputs/sew_resnet", type=str)
    parser.add_argument("--save-every", default=0, type=int, help="Save every N epochs; 0 disables periodic saves")

    # DDP/reproducibility
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dist-backend", default="nccl", type=str)

    args = parser.parse_args()

    if args.image_size is None:
        args.image_size = 32 if args.dataset in {"cifar10", "cifar100"} else 64 if args.dataset == "tiny-imagenet" else 224
    if args.val_resize is None:
        args.val_resize = args.image_size + 8 if args.dataset == "tiny-imagenet" else args.image_size + 32
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    setup_distributed(args)
    seed_everything(args.seed, args.rank)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = get_device(args)
    log(args, f"Using device={device}, distributed={args.distributed}, world_size={args.world_size}")

    train_set, val_set, num_classes = build_datasets(args)
    train_loader, val_loader = build_loaders(args, train_set, val_set)
    log(args, f"Dataset={args.dataset}, train={len(train_set)}, val={len(val_set)}, classes={num_classes}")

    model = build_model(args, num_classes=num_classes)
    set_spikingjelly_fast_mode(model, args, device)

    if args.ann_pretrained:
        load_torchvision_ann_pretrained(model, args, num_classes)

    if args.author_pretrained:
        if args.distributed and not is_main_process(args):
            dist.barrier()
            author_path = None
        else:
            author_path = download_author_checkpoint(args)
            log(args, f"Author checkpoint selected: {author_path}")
            if args.distributed:
                dist.barrier()
        if args.distributed and not is_main_process(args):
            # Find the file downloaded by rank 0.
            ckpt_dir = Path(args.output_dir) / "author_checkpoints"
            candidates = sorted(ckpt_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not candidates:
                raise RuntimeError("Rank 0 downloaded no author checkpoint visible to this rank.")
            author_path = candidates[0]
        load_model_weights(model, Path(author_path), strict=args.strict_load)

    if args.weight_quant_bits < 32:
        apply_weight_fake_quant(model, args.weight_quant_bits)

    model.to(device)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)

    if args.compile:
        try:
            model = torch.compile(model)
            log(args, "torch.compile enabled")
        except Exception as exc:
            log(args, f"torch.compile failed and was skipped: {exc}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    scaler = make_grad_scaler(args, device)

    start_epoch = 0
    best_acc1 = 0.0
    if args.resume is not None:
        start_epoch, best_acc1 = load_training_checkpoint(
            model,
            optimizer=None if args.eval else optimizer,
            scheduler=None if args.eval else scheduler,
            scaler=None if args.eval else scaler,
            path=Path(args.resume),
            strict=args.strict_load,
            map_location="cpu",
        )
        log(args, f"Loaded --resume {args.resume}; start_epoch={start_epoch}, best_acc1={best_acc1:.2f}")

    if args.eval and args.quantize_eval != "none":
        model = apply_dynamic_quant_for_eval(model, args)
        device = torch.device("cpu")
        criterion = criterion.cpu()

    if args.distributed and not args.eval:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank] if device.type == "cuda" else None)

    if args.eval:
        stats = evaluate(args, model, criterion, val_loader, device)
        log(args, f"Eval: loss={stats['loss']:.4f}, acc1={stats['acc1']:.2f}, acc5={stats['acc5']:.2f}")
        cleanup_distributed()
        return

    out_dir = Path(args.output_dir)
    if is_main_process(args):
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "args.yaml").open("w") as f:
            yaml.safe_dump(vars(args), f, sort_keys=False, allow_unicode=True)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_stats = train_one_epoch(args, model, criterion, optimizer, scaler, train_loader, device, epoch)
        val_stats = evaluate(args, model, criterion, val_loader, device)
        if scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        is_best = val_stats["acc1"] > best_acc1
        best_acc1 = max(best_acc1, val_stats["acc1"])
        elapsed = time.time() - t0

        log(
            args,
            f"Epoch {epoch:03d}: lr={lr:.6g}, "
            f"train_loss={train_stats['loss']:.4f}, train_acc1={train_stats['acc1']:.2f}, "
            f"val_loss={val_stats['loss']:.4f}, val_acc1={val_stats['acc1']:.2f}, "
            f"val_acc5={val_stats['acc5']:.2f}, best_acc1={best_acc1:.2f}, time={elapsed:.1f}s",
        )

        save_checkpoint(args, model, optimizer, scheduler, scaler, epoch, best_acc1, "last.pth")
        if is_best:
            save_checkpoint(args, model, optimizer, scheduler, scaler, epoch, best_acc1, "best.pth")
        if args.save_every and (epoch + 1) % args.save_every == 0:
            save_checkpoint(args, model, optimizer, scheduler, scaler, epoch, best_acc1, f"epoch_{epoch:03d}.pth")

    cleanup_distributed()


if __name__ == "__main__":
    main()
