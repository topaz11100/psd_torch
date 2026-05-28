#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-dataset SEW-ResNet SNN training/testing script.

Core design
-----------
- Uses `sew_resnet.py` as the SEW-ResNet backend and mirrors its default model/training settings.
- PyTorch + SpikingJelly activation_based API.
- Direct SNN input: no Poisson encoder.
- Static images are repeated over T steps: [N,C,H,W] -> [T,N,C,H,W].
- Event datasets are integrated/loaded as event frames: [N,T,2,H,W] -> [T,N,2,H,W].
- Standard BPTT or temporally truncated BPTT (TBPTT) with user-defined k.
- SGD + CrossEntropyLoss by default, matching `sew_resnet.py`.
- Early stopping and best-checkpoint save/load in one file.

Supported datasets
------------------
Static:
  cifar10, cifar100, caltech101, tiny-imagenet
Event / event-frame:
  cifar10dvs / dvs-cifar10, cifar100dvs / dvs-cifar100 / i2e-cifar100,
  ncaltech101 / n-caltech101, dvsgesture / dvs128gesture / dvs-gesture,
  es-tiny-imagenet

Notes
-----
1. CIFAR100-DVS is implemented through the public I2E-CIFAR100 Hugging Face
   dataset interface. Install `datasets` when using it.
2. ES-Tiny-ImageNet has two practical modes:
   - `--estiny-source generated`: downloads Tiny-ImageNet and generates an
     ODG-like 8-step event-frame sequence online from each image.
   - `--estiny-source folder`: expects pre-generated NPZ files containing event
     frames; this is useful if you already have an exact ES-Tiny-ImageNet dump.
3. Some neuromorphic datasets can be large and require long preprocessing.

Examples
--------
# Fast random tensor shape/gradient check. No dataset is downloaded.
python sew_resnet_snn.py --mode sanity --device cpu --T 2 --depth 18 --sanity-datasets cifar10,cifar10dvs,estinyimagenet

# CIFAR-10, SEW-ResNet-18, standard BPTT
python sew_resnet_snn.py --dataset cifar10 --mode train_test --depth 18 --T 10 --data-dir ./data --out-dir ./runs/cifar10_sew18

# DVS-CIFAR10, SEW-ResNet-18, TBPTT k=10
python sew_resnet_snn.py --dataset dvs-cifar10 --mode train_test --depth 18 --T 100 --tbptt --tbptt-k 10 --dvs-size 48

# N-Caltech101, TBPTT k=10
python sew_resnet_snn.py --dataset n-caltech101 --mode train_test --depth 18 --T 60 --tbptt --tbptt-k 10 --dvs-size 48

# DVS-Gesture, TBPTT k=10
python sew_resnet_snn.py --dataset dvs-gesture --mode train_test --depth 18 --T 60 --tbptt --tbptt-k 10 --dvs-size 64

# Tiny-ImageNet, SEW-ResNet-34
python sew_resnet_snn.py --dataset tiny-imagenet --mode train_test --depth 34 --T 8 --image-size 64

# ES-Tiny-ImageNet generated from Tiny-ImageNet with 8 event frames
python sew_resnet_snn.py --dataset es-tiny-imagenet --mode train_test --depth 18 --T 8 --tbptt --tbptt-k 4 --estiny-source generated
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import shutil
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
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
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler


# -----------------------------------------------------------------------------
# Compatibility patch and backend import
# -----------------------------------------------------------------------------

_TORCHVISION_NMS_STUB_LIB = None


def ensure_torchvision_nms_stub(force: bool = False) -> None:
    """Register a torchvision::nms schema only when it is explicitly needed.

    Defining this schema unconditionally can collide with a healthy torchvision
    build and abort Python at native-op registration time. The safe path is:
    first try the normal import, and only enable this fallback when the known
    `operator torchvision::nms does not exist` error appears.
    """
    if not force and os.environ.get("SEW_USE_TORCHVISION_NMS_STUB", "0") != "1":
        return
    global _TORCHVISION_NMS_STUB_LIB
    try:
        _TORCHVISION_NMS_STUB_LIB = torch.library.Library("torchvision", "DEF")
        _TORCHVISION_NMS_STUB_LIB.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
    except Exception:
        # Already defined, or this PyTorch build does not need/support the patch.
        pass


try:
    from spikingjelly.activation_based import functional, layer, neuron, surrogate
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script requires SpikingJelly. Install it with `pip install spikingjelly`.") from exc


def _clear_torchvision_modules() -> None:
    for name in list(sys.modules):
        if name == "torchvision" or name.startswith("torchvision."):
            del sys.modules[name]


def import_sew_backend():
    """Import `sew_resnet.py` and recover from the known missing-NMS torchvision issue."""
    try:
        return importlib.import_module("sew_resnet")
    except Exception as exc:  # pragma: no cover
        msg = str(exc)
        if "torchvision::nms" in msg or "operator torchvision::nms does not exist" in msg:
            _clear_torchvision_modules()
            ensure_torchvision_nms_stub(force=True)
            try:
                return importlib.import_module("sew_resnet")
            except Exception as exc2:
                raise RuntimeError(
                    "Could not import `sew_resnet.py` even after applying the optional torchvision::nms "
                    "compatibility stub. Check that `sew_resnet.py`, torch, torchvision, and SpikingJelly "
                    "are installed in the same environment."
                ) from exc2
        raise RuntimeError(
            "Could not import the SEW-ResNet backend `sew_resnet.py`. Place this file in the same directory "
            "as `sew_resnet.py`, or add that directory to PYTHONPATH."
        ) from exc


sew_backend = import_sew_backend()


def backend_depth_choices() -> List[int]:
    depth_cfg = getattr(sew_backend, "DEPTH_TO_LAYERS", None)
    if isinstance(depth_cfg, dict) and depth_cfg:
        return sorted(int(k) for k in depth_cfg.keys())
    return [18, 34, 50, 101, 152]


# -----------------------------------------------------------------------------
# Constants and dataset metadata
# -----------------------------------------------------------------------------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Mirrors used only when the official SpikingJelly metadata is not directly
# downloadable in automated scripts.
DVSGESTURE_DROPBOX_URL = "https://www.dropbox.com/s/cct5kyilhtsliup/DvsGesture.tar.gz?dl=1"
DVSGESTURE_ARCHIVE_MD5 = "8a5c71fb11e24e5ca5b11866ca6c00a1"
NCALTECH101_MENDELEY_URL = (
    "https://data.mendeley.com/public-files/datasets/cy6cvx3ryv/files/"
    "36b5c52a-b49d-4853-addb-a836a8883e49/file_downloaded"
)
NCALTECH101_ARCHIVE_MD5 = "66201824eabb0239c7ab992480b50ba3"
TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    input_shape: Tuple[int, int, int]
    num_classes: int
    time_steps: int
    is_event_dataset: bool
    description: str = ""


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------


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
    key = str(name).lower().replace("-", "").replace("_", "")
    aliases = {
        "cifar10": "cifar10",
        "cifar100": "cifar100",
        "caltech101": "caltech101",
        "tinyimagenet": "tinyimagenet",
        "tinyimagenet200": "tinyimagenet",
        "dvscifar10": "cifar10dvs",
        "cifar10dvs": "cifar10dvs",
        "dvs10cifar": "cifar10dvs",
        "dvscifar100": "cifar100dvs",
        "cifar100dvs": "cifar100dvs",
        "i2ecifar100": "cifar100dvs",
        "dvs100cifar": "cifar100dvs",
        "ncaltech": "ncaltech101",
        "ncaltech101": "ncaltech101",
        "ncaltehc101": "ncaltech101",  # common typo in prompts
        "dvsgesture": "dvsgesture",
        "dvs128gesture": "dvsgesture",
        "estinyimagenet": "estinyimagenet",
        "tinyesimagenet": "estinyimagenet",
        "estinyimagenet200": "estinyimagenet",
    }
    return aliases.get(key, key)


def _has_npz(root: Path) -> bool:
    try:
        return root.exists() and any(root.rglob("*.npz"))
    except OSError:
        return False


def _download_resources(resource_url_md5: Sequence[Tuple[str, str, str]], download_root: Path) -> None:
    ensure_torchvision_nms_stub()
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
            print(f"[download] valid archive already exists: {fpath}")


def limit_dataset(dataset: Optional[Dataset], limit: int) -> Optional[Dataset]:
    if dataset is None or limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, list(range(limit)))


class TargetRemapSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: Sequence[int], target_map: Optional[Dict[int, int]] = None):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]
        self.target_map = target_map

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        x, y = self.dataset[self.indices[idx]]
        y = int(y)
        if self.target_map is not None:
            y = int(self.target_map[y])
        return x, y


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
    labels: List[int] = []
    old_transform = getattr(dataset, "transform", None)
    try:
        if hasattr(dataset, "transform"):
            dataset.transform = None
        for i in range(len(dataset)):
            _, y = dataset[i]
            labels.append(int(y))
    finally:
        if hasattr(dataset, "transform"):
            dataset.transform = old_transform
    return labels


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


# -----------------------------------------------------------------------------
# Event frame transforms and I2E dataset
# -----------------------------------------------------------------------------


class EventFrameTransform:
    """Convert event frame arrays to [T,C,H,W] float tensors.

    If binarize=True, positive counts are converted to event occurrence. This is
    event-frame integration, not Poisson encoding.
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


def unpack_i2e_event_data(item, use_io: bool = True) -> torch.Tensor:
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
    """Hugging Face I2E dataset wrapper for CIFAR100-DVS/I2E-CIFAR100."""

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
                "CIFAR100-DVS/I2E-CIFAR100 requires the Hugging Face `datasets` package. "
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


# -----------------------------------------------------------------------------
# Tiny-ImageNet and ES-Tiny-ImageNet helpers
# -----------------------------------------------------------------------------


class TinyImageNetValDataset(Dataset):
    def __init__(self, root: Path, class_to_idx: Dict[str, int], transform=None) -> None:
        from torchvision.datasets.folder import default_loader

        self.root = Path(root)
        self.transform = transform
        self.default_loader = default_loader
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
                if wnid in class_to_idx:
                    self.samples.append((image_dir / filename, class_to_idx[wnid]))
        self.targets = [y for _, y in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        img = self.default_loader(str(path))
        if self.transform is not None:
            img = self.transform(img)
        return img, int(target)


def maybe_download_tiny_imagenet(data_dir: Path, download: bool = True) -> Path:
    ensure_torchvision_nms_stub()
    from torchvision.datasets.utils import download_and_extract_archive

    data_dir = Path(data_dir)
    if (data_dir / "wnids.txt").is_file() and (data_dir / "train").is_dir():
        return data_dir
    tiny_root = data_dir / "tiny-imagenet-200"
    if (tiny_root / "wnids.txt").is_file() and (tiny_root / "train").is_dir():
        return tiny_root
    if not download:
        raise FileNotFoundError(
            f"Tiny-ImageNet was not found under {data_dir}. Re-run with --download."
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Tiny-ImageNet] downloading/extracting from {TINY_IMAGENET_URL}")
    download_and_extract_archive(TINY_IMAGENET_URL, download_root=str(data_dir), filename="tiny-imagenet-200.zip")
    if not tiny_root.is_dir():
        raise RuntimeError(f"Tiny-ImageNet download finished but {tiny_root} was not found.")
    return tiny_root


def static_transforms_for(args: argparse.Namespace, dataset_name: str, train: bool):
    ensure_torchvision_nms_stub()
    from torchvision import transforms

    if dataset_name in {"cifar10", "cifar100"}:
        ops: List[Callable] = []
        if train and args.augment_static:
            ops.extend([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
        ops.append(transforms.ToTensor())
        if args.normalize_static:
            if dataset_name == "cifar10":
                ops.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
            else:
                ops.append(transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD))
        return transforms.Compose(ops)

    if dataset_name == "caltech101":
        if train and args.augment_static:
            resize_ops: List[Callable] = [
                transforms.RandomResizedCrop(args.image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            resize_ops = [transforms.Resize(args.image_size + 16), transforms.CenterCrop(args.image_size)]
        ops = resize_ops + [transforms.Lambda(lambda img: img.convert("RGB")), transforms.ToTensor()]
        if args.normalize_static:
            ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
        return transforms.Compose(ops)

    if dataset_name == "tinyimagenet":
        if train and args.augment_static:
            ops = [
                transforms.RandomResizedCrop(args.image_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        else:
            ops = [transforms.Resize(args.image_size + 8), transforms.CenterCrop(args.image_size), transforms.ToTensor()]
        if args.normalize_static:
            ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
        return transforms.Compose(ops)

    if dataset_name == "estinyimagenet_base":
        # No normalization before event generation; the event converter works on [0,1] intensities.
        if train and args.augment_static:
            return transforms.Compose([
                transforms.RandomResizedCrop(args.image_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.ToTensor(),
            ])
        return transforms.Compose([
            transforms.Resize(args.image_size + 8),
            transforms.CenterCrop(args.image_size),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
        ])

    raise ValueError(dataset_name)


class ODGStyleEventGenerator:
    """Lightweight ODG-like converter: image -> [T,2,H,W] event frames.

    This is designed for ES-Tiny-ImageNet experiments when an exact pre-generated
    ES-Tiny-ImageNet folder is not available. It uses image motion and signed
    intensity changes to create positive/negative event channels.
    """

    def __init__(
        self,
        frames: int = 8,
        threshold: float = 0.08,
        motion_pixels: int = 1,
        binarize: bool = True,
    ):
        self.frames = int(frames)
        self.threshold = float(threshold)
        self.motion_pixels = int(motion_pixels)
        self.binarize = bool(binarize)
        base = [(0, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0)]
        if self.frames <= len(base):
            self.offsets = base[:self.frames]
        else:
            self.offsets = [base[i % len(base)] for i in range(self.frames)]

    @staticmethod
    def _shift(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        # x: [1,H,W]. Zero-padded integer shift.
        _, h, w = x.shape
        out = torch.zeros_like(x)
        y_src0 = max(0, -dy)
        y_src1 = min(h, h - dy) if dy >= 0 else h
        x_src0 = max(0, -dx)
        x_src1 = min(w, w - dx) if dx >= 0 else w
        y_dst0 = max(0, dy)
        y_dst1 = y_dst0 + max(0, y_src1 - y_src0)
        x_dst0 = max(0, dx)
        x_dst1 = x_dst0 + max(0, x_src1 - x_src0)
        if y_dst1 > y_dst0 and x_dst1 > x_dst0:
            out[:, y_dst0:y_dst1, x_dst0:x_dst1] = x[:, y_src0:y_src1, x_src0:x_src1]
        return out

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if img.ndim != 3:
            raise ValueError(f"Expected image [C,H,W], got {tuple(img.shape)}")
        if img.shape[0] == 1:
            gray = img
        else:
            gray = 0.2989 * img[0:1] + 0.5870 * img[1:2] + 0.1140 * img[2:3]
        frames: List[torch.Tensor] = []
        prev = self._shift(gray, self.offsets[0][0] * self.motion_pixels, self.offsets[0][1] * self.motion_pixels)
        for t, (dy, dx) in enumerate(self.offsets):
            cur = self._shift(gray, dy * self.motion_pixels, dx * self.motion_pixels)
            diff = cur - prev if t > 0 else torch.zeros_like(cur)
            pos = diff.clamp_min(0.0)
            neg = (-diff).clamp_min(0.0)
            if self.binarize:
                pos = (pos > self.threshold).float()
                neg = (neg > self.threshold).float()
            else:
                pos = (pos / max(self.threshold, 1e-6)).clamp(0, 1)
                neg = (neg / max(self.threshold, 1e-6)).clamp(0, 1)
            frames.append(torch.cat([pos, neg], dim=0))
            prev = cur
        return torch.stack(frames, dim=0).contiguous()


class ESTinyGeneratedDataset(Dataset):
    def __init__(self, image_dataset: Dataset, frames: int, threshold: float, motion_pixels: int, binarize: bool):
        self.image_dataset = image_dataset
        self.generator = ODGStyleEventGenerator(frames, threshold, motion_pixels, binarize)
        self.targets = get_targets(image_dataset)

    def __len__(self) -> int:
        return len(self.image_dataset)

    def __getitem__(self, idx: int):
        img, y = self.image_dataset[idx]
        if not isinstance(img, torch.Tensor):
            raise TypeError("ESTinyGeneratedDataset expects base transforms to return torch.Tensor images")
        return self.generator(img), int(y)


class NPZEventFolder(Dataset):
    """Folder dataset for pre-generated ES-Tiny-ImageNet NPZ files.

    Expected layout:
        root/train/<class_name>/*.npz
        root/val/<class_name>/*.npz
    Each npz must contain either `frames`, `x`, or a single unnamed array with
    shape [T,2,H,W].
    """

    def __init__(self, root: Path, split: str, transform: Optional[Callable] = None):
        import numpy as np  # noqa: F401  # imported to fail early if unavailable

        self.root = Path(root) / split
        self.transform = transform
        if not self.root.is_dir():
            raise FileNotFoundError(f"Expected NPZ event folder: {self.root}")
        classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        if not classes:
            raise RuntimeError(f"No class subdirectories found under {self.root}")
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples: List[Tuple[Path, int]] = []
        for c in classes:
            for path in sorted((self.root / c).rglob("*.npz")):
                self.samples.append((path, self.class_to_idx[c]))
        if not self.samples:
            raise RuntimeError(f"No npz files found under {self.root}")
        self.targets = [y for _, y in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import numpy as np

        path, y = self.samples[idx]
        with np.load(path) as data:
            if "frames" in data:
                arr = data["frames"]
            elif "x" in data:
                arr = data["x"]
            else:
                arr = data[data.files[0]]
        x = torch.as_tensor(arr, dtype=torch.float32)
        if self.transform is not None:
            x = self.transform(x)
        return x, int(y)


# -----------------------------------------------------------------------------
# Neuromorphic dataset preparation
# -----------------------------------------------------------------------------


def ensure_cifar10dvs_prepared(root: Path, download: bool = True) -> None:
    raw_root = root / "events_np"
    if _has_npz(raw_root):
        return
    if not download:
        raise FileNotFoundError(f"CIFAR10-DVS events_np not found under {raw_root}. Use --download.")
    try:
        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
        from torchvision.datasets.utils import extract_archive
    except Exception as exc:
        raise RuntimeError("CIFAR10-DVS preparation requires SpikingJelly datasets and torchvision.") from exc

    root.mkdir(parents=True, exist_ok=True)
    download_root = root / "download"
    extract_root = root / "extract"
    _download_resources(CIFAR10DVS.resource_url_md5(), download_root)

    if not extract_root.exists() or not any(extract_root.iterdir()):
        extract_root.mkdir(parents=True, exist_ok=True)
        print(f"[CIFAR10-DVS] extracting archives to {extract_root}")
        if hasattr(CIFAR10DVS, "extract_downloaded_files"):
            CIFAR10DVS.extract_downloaded_files(download_root, extract_root)
        else:
            for archive in download_root.iterdir():
                extract_archive(str(archive), str(extract_root))

    if raw_root.exists() and not _has_npz(raw_root):
        shutil.rmtree(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    print(f"[CIFAR10-DVS] creating events_np under {raw_root}")
    if hasattr(CIFAR10DVS, "create_raw_from_extracted"):
        CIFAR10DVS.create_raw_from_extracted(extract_root, raw_root)
    elif hasattr(CIFAR10DVS, "create_events_np_files"):
        CIFAR10DVS.create_events_np_files(str(extract_root), str(raw_root))
    else:
        raise RuntimeError("This SpikingJelly version does not expose CIFAR10-DVS conversion helpers.")
    if not _has_npz(raw_root):
        raise RuntimeError(f"CIFAR10-DVS preprocessing finished but no npz file was found under {raw_root}")


def ensure_dvsgesture_prepared(root: Path, download: bool = True) -> None:
    raw_root = root / "events_np"
    if _has_npz(raw_root / "train") and _has_npz(raw_root / "test"):
        return
    if not download:
        raise FileNotFoundError(f"DVS128-Gesture events_np not found under {raw_root}. Use --download.")
    try:
        from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
        from torchvision.datasets.utils import check_integrity, download_url, extract_archive
    except Exception as exc:
        raise RuntimeError("DVS128-Gesture preparation requires SpikingJelly datasets and torchvision.") from exc

    root.mkdir(parents=True, exist_ok=True)
    download_root = root / "download"
    extract_root = root / "extract"
    download_root.mkdir(parents=True, exist_ok=True)
    archive = download_root / "DvsGesture.tar.gz"
    if not check_integrity(str(archive), DVSGESTURE_ARCHIVE_MD5):
        if archive.exists():
            archive.unlink()
        print(f"[DVS-Gesture] downloading DvsGesture.tar.gz to {download_root}")
        download_url(DVSGESTURE_DROPBOX_URL, str(download_root), filename="DvsGesture.tar.gz", md5=DVSGESTURE_ARCHIVE_MD5)
    else:
        print(f"[DVS-Gesture] valid archive already exists: {archive}")

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
        raise RuntimeError("This SpikingJelly version does not expose DVS-Gesture conversion helpers.")
    if not (_has_npz(raw_root / "train") and _has_npz(raw_root / "test")):
        raise RuntimeError(f"DVS-Gesture preprocessing finished but events_np is incomplete: {raw_root}")


def ensure_ncaltech101_prepared(root: Path, download: bool = True) -> None:
    raw_root = root / "events_np"
    if _has_npz(raw_root):
        return
    try:
        from spikingjelly.datasets.n_caltech101 import NCaltech101
        from torchvision.datasets.utils import check_integrity, download_url, extract_archive
    except Exception as exc:
        raise RuntimeError("N-Caltech101 preparation requires SpikingJelly datasets and torchvision.") from exc

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
            raise FileNotFoundError(f"N-Caltech101 archive was not found under {download_root}. Use --download.")
        if archive.exists():
            archive.unlink()
        print(f"[N-Caltech101] downloading Caltech101.zip to {download_root}")
        download_url(NCALTECH101_MENDELEY_URL, str(download_root), filename="Caltech101.zip", md5=NCALTECH101_ARCHIVE_MD5)
        valid_archive = archive
    elif valid_archive is not None:
        print(f"[N-Caltech101] valid archive already exists: {valid_archive}")

    if not extract_ready:
        if extract_root.exists():
            shutil.rmtree(extract_root)
        extract_root.mkdir(parents=True, exist_ok=True)
        print(f"[N-Caltech101] extracting archive to {extract_root}")
        extract_archive(str(valid_archive), str(extract_root))

    if raw_root.exists() and not _has_npz(raw_root):
        shutil.rmtree(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    print(f"[N-Caltech101] creating events_np under {raw_root}")
    if hasattr(NCaltech101, "create_raw_from_extracted"):
        NCaltech101.create_raw_from_extracted(extract_root, raw_root)
    elif hasattr(NCaltech101, "create_events_np_files"):
        NCaltech101.create_events_np_files(str(extract_root), str(raw_root))
    else:
        raise RuntimeError("This SpikingJelly version does not expose N-Caltech101 conversion helpers.")
    if not _has_npz(raw_root):
        raise RuntimeError(f"N-Caltech101 preprocessing finished but no npz file was found under {raw_root}")


# -----------------------------------------------------------------------------
# Dataset construction
# -----------------------------------------------------------------------------


def resolve_auto_time_steps(args: argparse.Namespace) -> int:
    if args.T > 0:
        return int(args.T)
    defaults = {
        "cifar10": 10,
        "cifar100": 10,
        "caltech101": 10,
        "tinyimagenet": 8,
        "cifar10dvs": 100,
        "cifar100dvs": 10,
        "ncaltech101": 60,
        "dvsgesture": 60,
        "estinyimagenet": 8,
    }
    return defaults[canonical_dataset_name(args.dataset)]


def build_datasets(args: argparse.Namespace) -> Tuple[Optional[Dataset], Optional[Dataset], Dataset, DatasetSpec]:
    ensure_torchvision_nms_stub()
    data_dir = Path(args.data_dir).expanduser().resolve()
    args.dataset = canonical_dataset_name(args.dataset)
    args.T = resolve_auto_time_steps(args)
    ds = args.dataset

    if ds in {"cifar10", "cifar100"}:
        from torchvision import datasets

        cls = datasets.CIFAR10 if ds == "cifar10" else datasets.CIFAR100
        num_classes = 10 if ds == "cifar10" else 100
        train_tf = static_transforms_for(args, ds, train=True)
        eval_tf = static_transforms_for(args, ds, train=False)
        train_base = cls(root=str(data_dir), train=True, transform=train_tf, download=args.download)
        val_base = cls(root=str(data_dir), train=True, transform=eval_tf, download=args.download)
        test_set = cls(root=str(data_dir), train=False, transform=eval_tf, download=args.download)
        train_idx, val_idx = split_train_val_indices(train_base.targets, args.val_ratio, args.seed)
        train_set: Optional[Dataset] = Subset(train_base, train_idx)
        val_set: Optional[Dataset] = Subset(val_base, val_idx) if val_idx else None
        spec = DatasetSpec(ds, (3, 32, 32), num_classes, args.T, False, "static RGB image repeated for T steps")

    elif ds == "caltech101":
        from torchvision import datasets

        train_tf = static_transforms_for(args, "caltech101", train=True)
        eval_tf = static_transforms_for(args, "caltech101", train=False)
        base_for_labels = datasets.Caltech101(root=str(data_dir), target_type="category", transform=None, download=args.download)
        train_base = datasets.Caltech101(root=str(data_dir), target_type="category", transform=train_tf, download=False)
        eval_base = datasets.Caltech101(root=str(data_dir), target_type="category", transform=eval_tf, download=False)
        labels_raw = get_targets(base_for_labels)
        categories = getattr(base_for_labels, "categories", None)
        keep_indices = list(range(len(labels_raw)))
        target_map: Optional[Dict[int, int]] = None
        if args.caltech_exclude_background and categories is not None:
            bg_ids = [i for i, name in enumerate(categories) if str(name).lower() == "background_google"]
            if bg_ids:
                bg = bg_ids[0]
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
        spec = DatasetSpec("caltech101", (3, args.image_size, args.image_size), len(set(labels)), args.T, False, "static RGB image repeated for T steps")

    elif ds == "tinyimagenet":
        from torchvision import datasets

        tiny_root = maybe_download_tiny_imagenet(data_dir, args.download)
        train_tf = static_transforms_for(args, "tinyimagenet", train=True)
        eval_tf = static_transforms_for(args, "tinyimagenet", train=False)
        train_full = datasets.ImageFolder(str(tiny_root / "train"), transform=train_tf)
        val_base_for_split = datasets.ImageFolder(str(tiny_root / "train"), transform=eval_tf)
        test_set = TinyImageNetValDataset(tiny_root, train_full.class_to_idx, transform=eval_tf)
        train_idx, val_idx = split_train_val_indices(train_full.targets, args.val_ratio, args.seed)
        train_set = Subset(train_full, train_idx)
        val_set = Subset(val_base_for_split, val_idx) if val_idx else None
        spec = DatasetSpec("tinyimagenet", (3, args.image_size, args.image_size), 200, args.T, False, "Tiny-ImageNet RGB image repeated for T steps")

    elif ds == "estinyimagenet":
        if args.estiny_source == "generated":
            from torchvision import datasets

            tiny_root = maybe_download_tiny_imagenet(data_dir, args.download)
            train_tf = static_transforms_for(args, "estinyimagenet_base", train=True)
            eval_tf = static_transforms_for(args, "estinyimagenet_base", train=False)
            train_img = datasets.ImageFolder(str(tiny_root / "train"), transform=train_tf)
            val_img = datasets.ImageFolder(str(tiny_root / "train"), transform=eval_tf)
            test_img = TinyImageNetValDataset(tiny_root, train_img.class_to_idx, transform=eval_tf)
            train_idx, val_idx = split_train_val_indices(train_img.targets, args.val_ratio, args.seed)
            train_set = Subset(
                ESTinyGeneratedDataset(train_img, args.T, args.estiny_threshold, args.estiny_motion_pixels, args.event_binarize),
                train_idx,
            )
            val_set = Subset(
                ESTinyGeneratedDataset(val_img, args.T, args.estiny_threshold, args.estiny_motion_pixels, args.event_binarize),
                val_idx,
            ) if val_idx else None
            test_set = ESTinyGeneratedDataset(test_img, args.T, args.estiny_threshold, args.estiny_motion_pixels, args.event_binarize)
            spec = DatasetSpec(
                "estinyimagenet",
                (2, args.image_size, args.image_size),
                200,
                args.T,
                True,
                "Tiny-ImageNet converted online to ODG-like ES event frames [T,2,H,W]",
            )
        else:
            root = Path(args.estiny_folder).expanduser().resolve() if args.estiny_folder else data_dir / "ES-Tiny-ImageNet"
            size = (args.dvs_size, args.dvs_size) if args.dvs_size > 0 else None
            transform = EventFrameTransform(size=size, target_frames=args.T, binarize=args.event_binarize, normalize=args.event_normalize, resize_mode=args.event_resize_mode)
            train_full = NPZEventFolder(root, "train", transform=transform)
            test_set = NPZEventFolder(root, "val", transform=transform)
            train_idx, val_idx = split_train_val_indices(train_full.targets, args.val_ratio, args.seed)
            train_set = Subset(train_full, train_idx)
            val_set = Subset(train_full, val_idx) if val_idx else None
            spatial = args.dvs_size if args.dvs_size > 0 else args.image_size
            spec = DatasetSpec("estinyimagenet", (2, spatial, spatial), len(set(train_full.targets)), args.T, True, "pre-generated ES-Tiny-ImageNet NPZ event frames [T,2,H,W]")

    elif ds == "cifar10dvs":
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
        transform = EventFrameTransform(size=size, target_frames=args.T, binarize=args.event_binarize, normalize=args.event_normalize, resize_mode=args.event_resize_mode)
        if args.cifar10dvs_split == "tebn" and CIFAR10DVSTEBNSplit is not None:
            train_full = CIFAR10DVSTEBNSplit(root=str(root), train=True, data_type="frame", frames_number=args.T, split_by=args.event_split_by, transform=transform)
            test_set = CIFAR10DVSTEBNSplit(root=str(root), train=False, data_type="frame", frames_number=args.T, split_by=args.event_split_by, transform=transform)
        else:
            full = CIFAR10DVS(root=str(root), data_type="frame", frames_number=args.T, split_by=args.event_split_by, transform=transform)
            labels = get_targets(full)
            train_rel, val_rel, test_rel = stratified_split_indices(labels, args.event_train_ratio, args.val_ratio, args.seed)
            train_full = Subset(full, train_rel + val_rel)
            test_set = Subset(full, test_rel)
        labels_train = get_targets(train_full)
        train_idx, val_idx = split_train_val_indices(labels_train, args.val_ratio, args.seed)
        train_set = Subset(train_full, train_idx)
        val_set = Subset(train_full, val_idx) if val_idx else None
        spatial = args.dvs_size if args.dvs_size > 0 else 128
        spec = DatasetSpec("cifar10dvs", (2, spatial, spatial), 10, args.T, True, "CIFAR10-DVS event frames [T,2,H,W]")

    elif ds == "cifar100dvs":
        hf_cache = data_dir / "hf_i2e_cache"
        size = (args.dvs_size, args.dvs_size) if args.dvs_size > 0 else (args.i2e_size, args.i2e_size)
        transform = EventFrameTransform(size=size, target_frames=args.T, binarize=args.event_binarize, normalize=args.event_normalize, resize_mode=args.event_resize_mode)
        train_full = I2EEventDataset(hf_cache, args.i2e_cifar100_config, split="train", transform=transform, hf_endpoint=args.hf_endpoint)
        test_set = I2EEventDataset(hf_cache, args.i2e_cifar100_config, split=args.i2e_validation_split, transform=transform, hf_endpoint=args.hf_endpoint)
        train_idx, val_idx = split_train_val_indices(get_targets(train_full), args.val_ratio, args.seed)
        train_set = Subset(train_full, train_idx)
        val_set = Subset(train_full, val_idx) if val_idx else None
        spatial = args.dvs_size if args.dvs_size > 0 else args.i2e_size
        spec = DatasetSpec("cifar100dvs", (2, spatial, spatial), 100, args.T, True, "I2E-CIFAR100 event frames [T,2,H,W]")

    elif ds == "ncaltech101":
        try:
            from spikingjelly.datasets.n_caltech101 import NCaltech101
        except Exception as exc:
            raise RuntimeError("N-Caltech101 loading requires spikingjelly.datasets.") from exc
        root = data_dir / "N-Caltech101"
        if args.prepare_event_data:
            ensure_ncaltech101_prepared(root, download=args.download)
        size = (args.dvs_size, args.dvs_size) if args.dvs_size > 0 else None
        transform = EventFrameTransform(size=size, target_frames=args.T, binarize=args.event_binarize, normalize=args.event_normalize, resize_mode=args.event_resize_mode)
        full = NCaltech101(root=str(root), data_type="frame", frames_number=args.T, split_by=args.event_split_by, transform=transform)
        labels = get_targets(full)
        train_rel, val_rel, test_rel = stratified_split_indices(labels, args.event_train_ratio, args.val_ratio, args.seed)
        train_set = Subset(full, train_rel)
        val_set = Subset(full, val_rel) if val_rel else None
        test_set = Subset(full, test_rel)
        input_shape = (2, args.dvs_size, args.dvs_size) if args.dvs_size > 0 else (2, 180, 240)
        spec = DatasetSpec("ncaltech101", input_shape, len(set(labels)), args.T, True, "N-Caltech101 event frames [T,2,H,W]")

    elif ds == "dvsgesture":
        try:
            from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
        except Exception as exc:
            raise RuntimeError("DVS128-Gesture loading requires spikingjelly.datasets.") from exc
        root = data_dir / "DVS128Gesture"
        if args.prepare_event_data:
            ensure_dvsgesture_prepared(root, download=args.download)
        size = (args.dvs_size, args.dvs_size) if args.dvs_size > 0 else None
        transform = EventFrameTransform(size=size, target_frames=args.T, binarize=args.event_binarize, normalize=args.event_normalize, resize_mode=args.event_resize_mode)
        train_full = DVS128Gesture(root=str(root), train=True, data_type="frame", frames_number=args.T, split_by=args.event_split_by, transform=transform)
        test_set = DVS128Gesture(root=str(root), train=False, data_type="frame", frames_number=args.T, split_by=args.event_split_by, transform=transform)
        train_idx, val_idx = split_train_val_indices(get_targets(train_full), args.val_ratio, args.seed)
        train_set = Subset(train_full, train_idx)
        val_set = Subset(train_full, val_idx) if val_idx else None
        spatial = args.dvs_size if args.dvs_size > 0 else 128
        spec = DatasetSpec("dvsgesture", (2, spatial, spatial), 11, args.T, True, "DVS128-Gesture event frames [T,2,H,W]")

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    if args.mode == "test":
        train_set = None
        val_set = None

    train_set = limit_dataset(train_set, args.limit_train) if train_set is not None else None
    val_set = limit_dataset(val_set, args.limit_val) if val_set is not None else None
    test_set = limit_dataset(test_set, args.limit_test)  # type: ignore[arg-type]
    return train_set, val_set, test_set, spec  # type: ignore[return-value]


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


# -----------------------------------------------------------------------------
# Model construction through sew_resnet.py
# -----------------------------------------------------------------------------


def backend_dataset_name_for_model(spec_name: str) -> str:
    """Map this multi-dataset script's names to the names expected by sew_resnet.py."""
    mapping = {
        "tinyimagenet": "tiny-imagenet",
        "estinyimagenet": "tiny-imagenet",
        "cifar10dvs": "cifar10",
        "cifar100dvs": "cifar100",
        "ncaltech101": "imagenet",
        "dvsgesture": "imagenet",
        "caltech101": "imagenet",
    }
    return mapping.get(spec_name, spec_name)


def resolve_stem_for_dataset(spec: DatasetSpec, args: argparse.Namespace) -> str:
    """Resolve the SEW-ResNet stem while preserving sew_resnet.py defaults.

    For datasets that `sew_resnet.py` already knows, its own `resolve_stem` is
    used. For extra event/static datasets, the same small-resolution principle is
    applied: CIFAR-style 3x3/no-maxpool stem for <=96px square inputs, ImageNet
    7x7/maxpool stem otherwise.
    """
    if args.stem != "auto":
        return args.stem
    backend_name = backend_dataset_name_for_model(spec.name)
    if hasattr(sew_backend, "resolve_stem") and spec.name in {"cifar10", "cifar100", "tinyimagenet"}:
        resolved = sew_backend.resolve_stem(backend_name, "auto")
        if resolved in {"cifar", "imagenet"}:
            return resolved
    _, h, w = spec.input_shape
    return "cifar" if max(h, w) <= 96 else "imagenet"


def patch_first_conv_input_channels(model: nn.Module, in_channels: int) -> None:
    if not hasattr(model, "conv1"):
        raise RuntimeError("The imported SEW-ResNet model has no conv1 attribute to patch.")
    old = model.conv1
    old_in = int(getattr(old, "in_channels", 3))
    if old_in == in_channels:
        return
    new_conv = layer.Conv2d(
        in_channels,
        int(old.out_channels),
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=(old.bias is not None),
    )
    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
    if new_conv.bias is not None:
        nn.init.zeros_(new_conv.bias)
    model.conv1 = new_conv


def model_config_from_args(args: argparse.Namespace, spec: DatasetSpec) -> Dict:
    depth_choices = backend_depth_choices()
    if int(args.depth) not in depth_choices:
        raise ValueError(f"Unsupported depth={args.depth}; sew_resnet.py supports {depth_choices}")
    return {
        "depth": int(args.depth),
        "dataset": backend_dataset_name_for_model(spec.name),
        "external_dataset": spec.name,
        "num_classes": int(spec.num_classes),
        "input_channels": int(spec.input_shape[0]),
        "input_shape": list(spec.input_shape),
        "stem": resolve_stem_for_dataset(spec, args),
        "connect_f": str(args.connect_f),
        "neuron": str(args.neuron),
        "v_threshold": float(args.v_threshold),
        "tau": float(args.tau),
        "decay_input": bool(args.decay_input),
        "detach_reset": bool(args.detach_reset),
        "surrogate": str(args.surrogate),
        "surrogate_alpha": float(args.surrogate_alpha),
        "zero_init_residual": bool(args.zero_init_residual),
    }


def build_model_from_config(config: Dict) -> nn.Module:
    # Keep the model construction delegated to sew_resnet.py. This preserves its
    # block selection, depth-to-layer mapping, neuron setup, SEW connection
    # function, initialization, and residual-zero-init behavior.
    model_args = SimpleNamespace(
        depth=int(config["depth"]),
        dataset=str(config.get("dataset", "cifar10")),
        stem=str(config.get("stem", "auto")),
        connect_f=str(config.get("connect_f", "ADD")),
        neuron=str(config.get("neuron", "IF")),
        v_threshold=float(config.get("v_threshold", 1.0)),
        tau=float(config.get("tau", 2.0)),
        decay_input=bool(config.get("decay_input", False)),
        detach_reset=bool(config.get("detach_reset", True)),
        surrogate=str(config.get("surrogate", "atan")),
        surrogate_alpha=float(config.get("surrogate_alpha", 2.0)),
        zero_init_residual=bool(config.get("zero_init_residual", False)),
    )
    model = sew_backend.build_model(model_args, num_classes=int(config["num_classes"]))
    patch_first_conv_input_channels(model, int(config.get("input_channels", 3)))
    return model


def configure_snn_backend(model: nn.Module, args: argparse.Namespace, device: torch.device) -> str:
    functional.set_step_mode(model, "m")
    backend = args.backend
    if backend == "torch":
        return "torch"
    if device.type != "cuda":
        warnings.warn(f"--backend {backend} requires CUDA. Falling back to torch backend.")
        return "torch"
    if backend == "cupy":
        try:
            import cupy  # noqa: F401
        except Exception as exc:
            warnings.warn(f"CuPy is not available ({exc}). Falling back to torch backend.")
            return "torch"
    for cls_name in ("IFNode", "LIFNode", "ParametricLIFNode"):
        cls = getattr(neuron, cls_name, None)
        if cls is None:
            continue
        try:
            functional.set_backend(model, backend, instance=cls)
        except Exception as exc:
            warnings.warn(f"Failed to set backend={backend} for {cls_name}: {exc}")
    return backend


def make_time_sequence(x: torch.Tensor, spec: DatasetSpec) -> torch.Tensor:
    if spec.is_event_dataset:
        if x.ndim != 5:
            raise ValueError(f"Event frames must have shape [N,T,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] != spec.time_steps:
            raise ValueError(f"Expected T={spec.time_steps}, got input T={x.shape[1]}")
        return x.transpose(0, 1).contiguous()
    if x.ndim != 4:
        raise ValueError(f"Static images must have shape [N,C,H,W], got {tuple(x.shape)}")
    return x.unsqueeze(0).expand(spec.time_steps, *x.shape).contiguous()


# -----------------------------------------------------------------------------
# Optimizer, AMP, checkpointing
# -----------------------------------------------------------------------------


@contextmanager
def autocast_context(device: torch.device, enabled: bool, dtype_name: str = "float16"):
    enabled = bool(enabled and device.type == "cuda")
    dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
        yield


def build_grad_scaler(device: torch.device, enabled: bool, dtype_name: str = "float16"):
    enabled = bool(enabled and device.type == "cuda" and dtype_name == "float16")
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except TypeError:  # older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def optimizer_step(loss: torch.Tensor, model: nn.Module, optimizer: torch.optim.Optimizer, scaler, grad_clip: float) -> None:
    scaler.scale(loss).backward()
    if grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    """Build optimizer with sew_resnet.py defaults: SGD, lr=0.1, momentum=0.9."""
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
    if args.opt == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError(f"Unsupported optimizer: {args.opt}")


def build_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer):
    """Use the same scheduler semantics as sew_resnet.py when possible."""
    if hasattr(sew_backend, "build_scheduler") and args.lr_scheduler in {"none", "step", "cosine"}:
        try:
            return sew_backend.build_scheduler(args, optimizer)
        except Exception as exc:
            warnings.warn(f"sew_resnet.py build_scheduler failed ({exc}); falling back to local scheduler.")
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
    raise ValueError(args.lr_scheduler)


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
    model_state = model_to_save.state_dict()
    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict() if scheduler is not None else None
    payload = {
        "epoch": int(epoch),
        "model_state": model_state,
        "model": model_state,  # compatibility with sew_resnet.py checkpoints
        "optimizer_state": optimizer_state,
        "optimizer": optimizer_state,
        "scheduler_state": scheduler_state,
        "scheduler": scheduler_state,
        "best_acc1": float(metrics.get("val_acc", metrics.get("train_acc", 0.0))) * 100.0,
        "dataset_spec": asdict(spec),
        "model_config": model_config,
        "args": vars(args),
        "metrics": metrics,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(payload, path)



def extract_model_state_from_checkpoint(ckpt: Dict) -> Dict[str, torch.Tensor]:
    if not isinstance(ckpt, dict):
        raise TypeError("Checkpoint must be a dictionary.")
    for key in ("model_state", "model", "state_dict", "net", "network"):
        value = ckpt.get(key)
        if isinstance(value, dict):
            return value
    # Last resort: allow a raw state_dict checkpoint.
    if ckpt and all(isinstance(k, str) for k in ckpt.keys()):
        return ckpt  # type: ignore[return-value]
    raise KeyError("Checkpoint does not contain a model state dict.")


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if hasattr(sew_backend, "strip_module_prefix"):
        try:
            return sew_backend.strip_module_prefix(state_dict)
        except Exception:
            pass
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        while k.startswith("module."):
            k = k[len("module."):]
        while k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        out[k] = v
    return out


def load_model_state_compatible(model: nn.Module, state_dict: Dict[str, torch.Tensor], strict: bool = False) -> None:
    if strict:
        model.load_state_dict(strip_module_prefix(state_dict), strict=True)
        return
    if hasattr(sew_backend, "compatible_state_dict"):
        selected, skipped = sew_backend.compatible_state_dict(model, state_dict)
        missing, unexpected = model.load_state_dict(selected, strict=False)
        print(f"[load] used={len(selected)} skipped={len(skipped)} missing={len(missing)} unexpected={len(unexpected)}")
        if skipped:
            print(f"[load] first skipped keys: {skipped[:10]}")
        return
    model.load_state_dict(strip_module_prefix(state_dict), strict=False)


def maybe_load_initial_weights(model: nn.Module, args: argparse.Namespace, spec: DatasetSpec, device: torch.device) -> int:
    """Load optional ANN/author/pretrained/resume model weights before training.

    Returns the epoch index to start from when `--resume` is a training checkpoint.
    """
    start_epoch = 1
    if args.ann_pretrained:
        if hasattr(sew_backend, "load_torchvision_ann_pretrained"):
            sew_backend.load_torchvision_ann_pretrained(model, args, spec.num_classes)
        else:
            warnings.warn("sew_resnet.py does not expose load_torchvision_ann_pretrained; skipping.")
    if args.author_pretrained:
        if hasattr(sew_backend, "download_author_checkpoint"):
            args.output_dir = args.out_dir
            ckpt_path = sew_backend.download_author_checkpoint(args)
            ckpt = load_checkpoint(Path(ckpt_path), device)
            load_model_state_compatible(model, extract_model_state_from_checkpoint(ckpt), strict=args.strict_load)
        else:
            warnings.warn("sew_resnet.py does not expose download_author_checkpoint; skipping.")
    if args.pretrained:
        ckpt = load_checkpoint(Path(args.pretrained).expanduser().resolve(), device)
        load_model_state_compatible(model, extract_model_state_from_checkpoint(ckpt), strict=args.strict_load)
    if args.resume:
        ckpt = load_checkpoint(Path(args.resume).expanduser().resolve(), device)
        load_model_state_compatible(model, extract_model_state_from_checkpoint(ckpt), strict=args.strict_load)
        if isinstance(ckpt, dict) and "epoch" in ckpt:
            start_epoch = int(ckpt.get("epoch", 0)) + 1
    return start_epoch


def maybe_resume_optimizer_scheduler(optimizer: torch.optim.Optimizer, scheduler, scaler, args: argparse.Namespace, device: torch.device) -> None:
    if not args.resume or args.no_resume_optimizer:
        return
    ckpt = load_checkpoint(Path(args.resume).expanduser().resolve(), device)
    opt_state = ckpt.get("optimizer_state", ckpt.get("optimizer")) if isinstance(ckpt, dict) else None
    sch_state = ckpt.get("scheduler_state", ckpt.get("scheduler")) if isinstance(ckpt, dict) else None
    scaler_state = ckpt.get("scaler_state", ckpt.get("scaler")) if isinstance(ckpt, dict) else None
    if opt_state is not None:
        optimizer.load_state_dict(opt_state)
    if scheduler is not None and sch_state is not None:
        scheduler.load_state_dict(sch_state)
    if scaler is not None and scaler_state is not None:
        try:
            scaler.load_state_dict(scaler_state)
        except Exception:
            pass


def maybe_apply_checkpoint_args(args: argparse.Namespace, ckpt: Optional[Dict]) -> None:
    if ckpt is None or args.ignore_ckpt_config:
        return
    ds_cfg = ckpt.get("dataset_spec") if isinstance(ckpt, dict) else None
    if isinstance(ds_cfg, dict):
        args.dataset = str(ds_cfg.get("name", args.dataset))
        args.T = int(ds_cfg.get("time_steps", args.T))
        input_shape = ds_cfg.get("input_shape")
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 3:
            if canonical_dataset_name(args.dataset) in {"caltech101", "tinyimagenet", "estinyimagenet"}:
                args.image_size = int(input_shape[-1])
            if canonical_dataset_name(args.dataset) in {"cifar10dvs", "cifar100dvs", "ncaltech101", "dvsgesture"}:
                args.dvs_size = int(input_shape[-1]) if int(input_shape[-1]) == int(input_shape[-2]) else 0
    model_cfg = ckpt.get("model_config") if isinstance(ckpt, dict) else None
    if isinstance(model_cfg, dict):
        for key in ("depth", "connect_f", "neuron", "v_threshold", "tau", "decay_input", "detach_reset", "surrogate", "surrogate_alpha", "zero_init_residual"):
            if key in model_cfg:
                # argparse names use hyphen, but Namespace attributes use underscores.
                setattr(args, key, model_cfg[key])


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------


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
    chunk_k = spec.time_steps if not args.tbptt else min(args.tbptt_k, spec.time_steps)
    if chunk_k <= 0:
        raise ValueError("--tbptt-k must be positive")

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long()
        if args.channels_last and x.ndim == 4:
            x = x.contiguous(memory_format=torch.channels_last)
        x_seq = make_time_sequence(x, spec)
        batch_size = y.numel()

        functional.reset_net(model)
        if not args.tbptt:
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.amp, args.amp_dtype):
                out_seq = model(x_seq)
                logits = out_seq.mean(dim=0) if out_seq.ndim == 3 else out_seq
                loss = criterion(logits, y)
            optimizer_step(loss, model, optimizer, scaler, args.grad_clip)
            batch_loss = float(loss.detach())
            pred = logits.detach().argmax(dim=1)
        else:
            # Temporally truncated BPTT: update after each k-step interval and
            # carry hidden states forward with their computation graph detached.
            logits_sum = None
            weighted_loss_sum = 0.0
            total_steps = 0
            for start in range(0, spec.time_steps, chunk_k):
                end = min(start + chunk_k, spec.time_steps)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, args.amp, args.amp_dtype):
                    out_chunk = model(x_seq[start:end])
                    logits_chunk = out_chunk.mean(dim=0) if out_chunk.ndim == 3 else out_chunk
                    loss = criterion(logits_chunk, y)
                optimizer_step(loss, model, optimizer, scaler, args.grad_clip)
                weight = end - start
                weighted_loss_sum += float(loss.detach()) * weight
                total_steps += weight
                part = logits_chunk.detach() * weight
                logits_sum = part if logits_sum is None else logits_sum + part
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
    return {"loss": total_loss / max(total_seen, 1), "acc": total_correct / max(total_seen, 1), "seconds": seconds}


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
        if args.channels_last and x.ndim == 4:
            x = x.contiguous(memory_format=torch.channels_last)
        x_seq = make_time_sequence(x, spec)
        functional.reset_net(model)
        logits_sum = None
        total_steps = 0
        for start in range(0, spec.time_steps, chunk):
            end = min(start + chunk, spec.time_steps)
            out_chunk = model(x_seq[start:end])
            part = out_chunk.sum(dim=0) if out_chunk.ndim == 3 else out_chunk * (end - start)
            logits_sum = part if logits_sum is None else logits_sum + part
            total_steps += end - start
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
    metrics = {"loss": total_loss / max(total_seen, 1), "acc": total_correct / max(total_seen, 1), "seconds": seconds}
    if is_main_process():
        print(f"[{name}] loss={metrics['loss']:.4f} acc={metrics['acc']:.4f} time={metrics['seconds']:.1f}s")
    return metrics


def run_train(args: argparse.Namespace, device: torch.device) -> Tuple[Path, Dict[str, float]]:
    train_set, val_set, _, spec = build_datasets_ddp_safe(args)
    if train_set is None:
        raise RuntimeError("Training mode requires a training dataset.")
    train_loader = build_loader(train_set, args.batch_size, True, args)
    val_loader = build_loader(val_set, args.batch_size, False, args) if val_set is not None else None

    model_config = model_config_from_args(args, spec)
    model = build_model_from_config(model_config).to(device)
    start_epoch = maybe_load_initial_weights(model, args, spec, device)
    if args.weight_quant_bits < 32:
        if hasattr(sew_backend, "apply_weight_fake_quant"):
            sew_backend.apply_weight_fake_quant(model, args.weight_quant_bits)
        else:
            warnings.warn("sew_resnet.py does not expose apply_weight_fake_quant; skipping fake quantization.")
    backend = configure_snn_backend(model, args, device)
    if args.channels_last:
        try:
            model.to(memory_format=torch.channels_last)
        except Exception:
            pass
    if args.compile:
        try:
            model = torch.compile(model)  # type: ignore[assignment]
        except Exception as exc:
            warnings.warn(f"torch.compile failed: {exc}. Continuing without compilation.")

    model = wrap_model_for_ddp(model, args, device)

    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = build_grad_scaler(device, args.amp, args.amp_dtype)
    maybe_resume_optimizer_scheduler(optimizer, scheduler, scaler, args, device)
    best_path = checkpoint_path(args)

    if is_main_process():
        print("=" * 96)
        print(f"Dataset       : {spec.name} input_shape={spec.input_shape} classes={spec.num_classes} T={spec.time_steps}")
        print(f"Input         : {spec.description}; no Poisson encoder")
        print(f"Model         : SEW-ResNet-{args.depth} stem={model_config['stem']} connect_f={args.connect_f} backend_dataset={model_config['dataset']}")
        print(f"Training      : {args.opt.upper()} + CE lr={args.lr} momentum={getattr(args, 'momentum', 0.0)} weight_decay={args.weight_decay} batch_size={args.batch_size} epochs={args.epochs}")
        print(f"Scheduler     : {args.lr_scheduler} min_lr={args.min_lr} warmup_epochs={args.warmup_epochs}")
        print(f"BPTT          : {'TBPTT' if args.tbptt else 'standard BPTT'} k={args.tbptt_k if args.tbptt else spec.time_steps}")
        print(f"Backend       : {backend} amp={args.amp and device.type == 'cuda'} device={device}")
        print(f"Best ckpt     : {best_path}")
        print("=" * 96)

    best_score = -math.inf
    best_metrics: Dict[str, float] = {}
    epochs_without_improvement = 0

    for epoch in range(start_epoch, args.epochs + 1):
        if isinstance(getattr(train_loader, "sampler", None), DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        if is_main_process():
            print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, spec, args)  # type: ignore[arg-type]
        if is_main_process():
            print(f"[train] loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f} time={train_metrics['seconds']:.1f}s")
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
                print(f"[early-stop] no improvement: {epochs_without_improvement}/{args.patience if args.patience > 0 else 'disabled'}")
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            if is_main_process():
                print(f"[early-stop] stopped at epoch {epoch}")
            break

    if is_main_process():
        config_path = best_path.with_suffix(".json")
        config_path.write_text(json.dumps({"dataset_spec": asdict(spec), "model_config": model_config, "best_metrics": best_metrics, "checkpoint": str(best_path)}, indent=2, ensure_ascii=False), encoding="utf-8")
    return best_path, best_metrics


def run_test(args: argparse.Namespace, device: torch.device) -> Dict[str, float]:
    ckpt_path = checkpoint_path(args)
    ckpt = load_checkpoint(ckpt_path, device)
    maybe_apply_checkpoint_args(args, ckpt)
    _, _, test_set, spec = build_datasets_ddp_safe(args)
    model_config = ckpt.get("model_config") if isinstance(ckpt, dict) else None
    if not isinstance(model_config, dict):
        model_config = model_config_from_args(args, spec)
    model = build_model_from_config(model_config).to(device)
    configure_snn_backend(model, args, device)
    load_model_state_compatible(model, extract_model_state_from_checkpoint(ckpt), strict=True)
    if args.quantize_eval != "none":
        if hasattr(sew_backend, "apply_dynamic_quant_for_eval"):
            model = sew_backend.apply_dynamic_quant_for_eval(model, args)
            device = torch.device("cpu")
        else:
            warnings.warn("sew_resnet.py does not expose apply_dynamic_quant_for_eval; skipping quantized eval.")
    test_loader = build_loader(test_set, args.batch_size, False, args)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if is_main_process():
        print(f"[test] loaded checkpoint: {ckpt_path}")
    return evaluate(model, test_loader, criterion, device, spec, args, name="test")  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# Sanity check
# -----------------------------------------------------------------------------


def run_sanity(args: argparse.Namespace, device: torch.device) -> None:
    print("[sanity] Running random-data forward/backward checks. No dataset download is used.")
    args.T = args.T if args.T > 0 else 2
    selected = [canonical_dataset_name(x.strip()) for x in args.sanity_datasets.split(",") if x.strip()]
    specs_all = {
        "cifar10": DatasetSpec("cifar10", (3, 32, 32), 10, args.T, False),
        "cifar100": DatasetSpec("cifar100", (3, 32, 32), 100, args.T, False),
        "caltech101": DatasetSpec("caltech101", (3, args.image_size, args.image_size), 101, args.T, False),
        "tinyimagenet": DatasetSpec("tinyimagenet", (3, 64, 64), 200, args.T, False),
        "cifar10dvs": DatasetSpec("cifar10dvs", (2, 32, 32), 10, args.T, True),
        "cifar100dvs": DatasetSpec("cifar100dvs", (2, 32, 32), 100, args.T, True),
        "ncaltech101": DatasetSpec("ncaltech101", (2, 32, 32), 101, args.T, True),
        "dvsgesture": DatasetSpec("dvsgesture", (2, 32, 32), 11, args.T, True),
        "estinyimagenet": DatasetSpec("estinyimagenet", (2, 32, 32), 200, args.T, True),
    }
    if not selected or selected == ["all"]:
        selected = list(specs_all.keys())
    for name in selected:
        if name not in specs_all:
            raise ValueError(f"Unknown sanity dataset alias: {name}")

    for name in selected:
        spec = specs_all[name]
        # Use CIFAR stem in sanity for small random tensors to avoid 7x7/maxpool over-downsampling.
        saved_stem = args.stem
        args.stem = "cifar"
        model_config = model_config_from_args(args, spec)
        args.stem = saved_stem
        model = build_model_from_config(model_config).to(device)
        configure_snn_backend(model, args, device)
        optimizer = build_optimizer(args, model)
        criterion = nn.CrossEntropyLoss()
        scaler = build_grad_scaler(device, args.amp, args.amp_dtype)
        model.train()
        functional.reset_net(model)

        b = args.sanity_batch_size
        if spec.is_event_dataset:
            x = (torch.rand(b, spec.time_steps, *spec.input_shape, device=device) > 0.95).float()
        else:
            x = torch.rand(b, *spec.input_shape, device=device)
        y = torch.randint(0, spec.num_classes, (b,), device=device)
        x_seq = make_time_sequence(x, spec)
        if args.tbptt:
            k = min(args.tbptt_k, spec.time_steps)
            for start in range(0, spec.time_steps, k):
                end = min(start + k, spec.time_steps)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, args.amp, args.amp_dtype):
                    logits = model(x_seq[start:end]).mean(0)
                    loss = criterion(logits, y)
                optimizer_step(loss, model, optimizer, scaler, args.grad_clip)
                functional.detach_net(model)
        else:
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.amp, args.amp_dtype):
                logits = model(x_seq).mean(0)
                loss = criterion(logits, y)
            optimizer_step(loss, model, optimizer, scaler, args.grad_clip)
        functional.reset_net(model)
        print(f"[sanity:{name}] ok input={spec.input_shape} classes={spec.num_classes} T={spec.time_steps}")


# -----------------------------------------------------------------------------
# Argparse
# -----------------------------------------------------------------------------


def make_argparser() -> argparse.ArgumentParser:
    depth_choices = backend_depth_choices()
    p = argparse.ArgumentParser(
        description="SEW-ResNet SNN multi-dataset trainer using sew_resnet.py as the model backend.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--mode", choices=["train", "test", "train_test", "sanity"], default="train_test")
    p.add_argument(
        "--dataset",
        default="ncaltech101",
        choices=[
            "cifar10", "cifar100", "caltech101", "tiny-imagenet", "tinyimagenet",
            "cifar10dvs", "dvs-cifar10", "dvscifar10",
            "cifar100dvs", "dvs-cifar100", "dvscifar100", "i2e-cifar100",
            "ncaltech101", "n-caltech101", "n-caltehc-101",
            "dvsgesture", "dvs128gesture", "dvs-gesture",
            "es-tiny-imagenet", "estinyimagenet", "tiny-es-imagenet",
        ],
    )
    p.add_argument("--data-dir", "--data-path", dest="data_dir", type=str, default="/home/leehyunjong/PycharmProjects/Machine_Learning/SNN/TA_BPTT/Motivation/data")
    p.add_argument("--out-dir", "--output-dir", dest="out_dir", type=str, default="./outputs/sew_resnet_snn")
    p.add_argument("--save-name", type=str, default="best_model.pt")
    p.add_argument("--ckpt", type=str, default="", help="Checkpoint path for test mode. Defaults to out-dir/save-name.")
    p.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--prepare-event-data", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ignore-ckpt-config", action="store_true")

    # Core training hyperparameters: defaults mirror sew_resnet.py.
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", "-b", type=int, default=32)
    p.add_argument("--T", "--timesteps", "-T", dest="T", type=int, default=16, help="0 selects a dataset-specific default T.")
    p.add_argument("--opt", "--optimizer", dest="opt", choices=["sgd", "adamw", "adam"], default="sgd")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--nesterov", action="store_true")
    p.add_argument("--lr-scheduler", choices=["none", "cosine", "step"], default="cosine")
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--lr-step-size", type=int, default=80)
    p.add_argument("--lr-gamma", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--grad-clip", "--clip-grad-norm", dest="grad_clip", type=float, default=0.0)

    # SEW-ResNet architecture settings delegated to sew_resnet.py.
    p.add_argument("--depth", type=int, choices=depth_choices, default=18)
    p.add_argument("--stem", choices=["auto", "cifar", "imagenet"], default="auto")
    p.add_argument("--connect-f", choices=["ADD", "AND", "IAND"], default="ADD")
    p.add_argument("--zero-init-residual", action="store_true")

    # SNN neuron/surrogate settings delegated to sew_resnet.py.
    p.add_argument("--neuron", choices=["IF", "LIF", "PLIF"], default="IF")
    p.add_argument("--v-threshold", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--decay-input", action="store_true")
    p.add_argument("--detach-reset", dest="detach_reset", action="store_true", default=True)
    p.add_argument("--no-detach-reset", dest="detach_reset", action="store_false")
    p.add_argument("--surrogate", choices=["atan", "sigmoid"], default="atan")
    p.add_argument("--surrogate-alpha", type=float, default=2.0)

    # BPTT / temporal truncated BPTT.
    p.add_argument("--tbptt", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--tbptt-k", type=int, default=1)
    p.add_argument("--eval-chunk-size", type=int, default=0, help="0 means evaluate all T steps at once.")

    # Static preprocessing.
    p.add_argument("--image-size", type=int, default=None, help="Default: 32 for CIFAR, 64 for Tiny/ES-Tiny, 128 for Caltech.")
    p.add_argument("--augment-static", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument( "--normalize-static", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--caltech-exclude-background", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--caltech-train-ratio", type=float, default=0.7)

    # Event preprocessing.
    p.add_argument("--dvs-size", type=int, default=0, help="0 keeps native resolution; otherwise resize to dvs-size x dvs-size.")
    p.add_argument("--event-split-by", choices=["time", "number"], default="time")
    p.add_argument("--event-binarize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--event-normalize", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--event-resize-mode", choices=["nearest", "bilinear"], default="nearest")
    p.add_argument("--cifar10dvs-split", choices=["tebn", "random"], default="tebn")
    p.add_argument("--event-train-ratio", type=float, default=0.8, help="Used for event datasets without an official train/test split.")
    p.add_argument("--i2e-cifar100-config", type=str, default="I2E-CIFAR100")
    p.add_argument("--i2e-validation-split", type=str, default="validation")
    p.add_argument("--i2e-size", type=int, default=128)
    p.add_argument("--hf-endpoint", type=str, default="")

    # ES-Tiny-ImageNet.
    p.add_argument("--estiny-source", choices=["generated", "folder"], default="generated")
    p.add_argument("--estiny-folder", type=str, default="", help="Root for --estiny-source folder, containing train/ and val/ class folders of NPZ frames.")
    p.add_argument("--estiny-threshold", type=float, default=0.08)
    p.add_argument("--estiny-motion-pixels", type=int, default=1)

    # Early stopping.
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=10, help="0 disables early stopping.")
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--monitor", choices=["acc", "loss"], default="acc")

    # Device/speed options from sew_resnet.py plus PyCharm-friendly aliases.
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--workers", "-j", "--num-workers", dest="num_workers", type=int, default=4)
    p.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=True)
    p.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    p.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--backend", choices=["torch", "cupy", "triton"], default="cupy")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable automatic mixed precision on CUDA.")
    p.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--channels-last", action="store_true")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False,
                   help="Try torch.compile(model). Disabled by default.")
    p.add_argument("--allow-tf32", action="store_true")
    p.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ddp", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable DistributedDataParallel when launched with torchrun.")
    p.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=0,
                   help=argparse.SUPPRESS)
    p.add_argument("--torch-num-threads", type=int, default=0, help="Set torch CPU intra-op threads. 0 keeps PyTorch default; use 1 for quick CPU sanity checks.")

    # Pretraining/checkpoint/quantization options exposed by sew_resnet.py.
    p.add_argument("--ann-pretrained", action="store_true", help="Initialize matching Conv/BN/FC layers from torchvision ANN ResNet weights.")
    p.add_argument("--author-pretrained", action="store_true", help="Try to download a SEW-ResNet author checkpoint through sew_resnet.py.")
    p.add_argument("--author-file", default=None, type=str, help="Substring/exact hint for selecting an author checkpoint file.")
    p.add_argument("--pretrained", default="", type=str, help="Load model weights from a checkpoint before training.")
    p.add_argument("--resume", default="", type=str, help="Resume model weights, and optimizer/scheduler unless --no-resume-optimizer is set.")
    p.add_argument("--no-resume-optimizer", action="store_true")
    p.add_argument("--strict-load", action="store_true")
    p.add_argument("--weight-quant-bits", default=32, type=int, help="<=31 enables sew_resnet.py STE fake weight quantization if available.")
    p.add_argument("--quantize-eval", default="none", choices=["none", "dynamic"], help="Dynamic int8 Linear-only CPU eval if supported by sew_resnet.py.")

    # Logging/repro/debug.
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument("--limit-val", type=int, default=0)
    p.add_argument("--limit-test", type=int, default=0)
    p.add_argument("--sanity-batch-size", type=int, default=1)
    p.add_argument("--sanity-datasets", type=str, default="cifar10,cifar100,cifar10dvs,cifar100dvs,ncaltech101,dvsgesture,estinyimagenet")
    return p



def main() -> None:
    args = make_argparser().parse_args()
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    args.dataset = canonical_dataset_name(args.dataset)
    # Aliases expected by utilities imported from sew_resnet.py.
    args.data_path = args.data_dir
    args.output_dir = args.out_dir
    args.workers = args.num_workers
    args.timesteps = args.T
    args.clip_grad_norm = args.grad_clip
    init_distributed_mode(args)
    if args.image_size is None:
        if args.dataset in {"cifar10", "cifar100", "cifar10dvs", "cifar100dvs", "dvsgesture", "ncaltech101"}:
            args.image_size = 32
        elif args.dataset in {"tinyimagenet", "estinyimagenet"}:
            args.image_size = 64
        elif args.dataset == "caltech101":
            args.image_size = 128
        else:
            args.image_size = 64
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.T < 0:
        raise ValueError("--T must be non-negative; use 0 for dataset-specific default")
    if args.tbptt and args.tbptt_k <= 0:
        raise ValueError("--tbptt-k must be positive")
    if args.dvs_size < 0:
        raise ValueError("--dvs-size must be non-negative")
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    set_seed(args.seed)
    device = resolve_device(args.device)
    if getattr(args, "distributed", False) and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.local_rank}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
    else:
        args.amp = False
        args.pin_memory = False
        if args.backend != "torch":
            args.backend = "torch"

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
