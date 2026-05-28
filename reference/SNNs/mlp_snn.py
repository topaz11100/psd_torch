#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLP-SNN classifier for MNIST / Fashion-MNIST / N-MNIST using PyTorch + SpikingJelly.

Key features
------------
1) Direct input to the SNN. No Poisson encoder is used.
   - MNIST/Fashion-MNIST: [N, 1, 28, 28] static image is repeated for T time-steps.
   - N-MNIST: SpikingJelly converts events to frame tensors [N, T, 2, 34, 34].
2) Standard BPTT or temporally truncated BPTT.
   - --tbptt disabled: one optimizer update after the whole T-step forward pass.
   - --tbptt enabled : one optimizer update after each k-step chunk and hidden states are detached.
3) Adam + CrossEntropy loss.
4) Optional CuPy backend, AMP, torch.compile, early stopping, best-checkpoint saving.
5) One file supports train, test, train_test, and sanity-check modes.

Examples
--------
# MNIST, standard BPTT
python snn_mlp_mnist_tbptt.py --dataset mnist --mode train_test --data-dir ./data --out-dir ./runs/mnist

# Fashion-MNIST, truncated BPTT with k=5
python snn_mlp_mnist_tbptt.py --dataset fmnist --mode train_test --tbptt --tbptt-k 5 --T 20

# N-MNIST, event frames, automatic download/preprocess through torchvision utils + SpikingJelly processing
python snn_mlp_mnist_tbptt.py --dataset nmnist --mode train_test --T 20 --tbptt --tbptt-k 5 --data-dir ./data

# CUDA speed options
python snn_mlp_mnist_tbptt.py --dataset nmnist --device cuda:0 --cupy --amp --num-workers 8 --pin-memory
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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch.utils.data.distributed import DistributedSampler

try:
    from spikingjelly.activation_based import functional, layer, neuron, surrogate
except Exception as exc:  # pragma: no cover - gives a clear runtime message
    raise RuntimeError(
        "This script requires SpikingJelly. Install it with `pip install spikingjelly`."
    ) from exc


# Direct Mendeley URLs used by common neuromorphic data loaders.
# The files are the same archives expected by SpikingJelly's NMNIST converter.
NMNIST_RESOURCES = [
    (
        "Train.zip",
        "https://data.mendeley.com/public-files/datasets/468j46mzdv/files/"
        "39c25547-014b-4137-a934-9d29fa53c7a0/file_downloaded",
        "20959b8e626244a1b502305a9e6e2031",
    ),
    (
        "Test.zip",
        "https://data.mendeley.com/public-files/datasets/468j46mzdv/files/"
        "05a4d654-7e03-4c15-bdfa-9bb2bcbea494/file_downloaded",
        "69ca8762b2fe404d9b9bad1103e97832",
    ),
]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    input_shape: Tuple[int, int, int]
    num_classes: int
    time_steps: int
    is_event_dataset: bool

    @property
    def input_dim(self) -> int:
        c, h, w = self.input_shape
        return c * h * w


class EventFrameToTensor:
    """Convert SpikingJelly event frames from numpy arrays to float tensors.

    For event-frame datasets, binarization is the default because the paper-style
    event image representation uses two polarity channels and indicates event
    occurrence at each pixel location. This is not Poisson encoding.
    """

    def __init__(self, binarize: bool = True, normalize: bool = False):
        self.binarize = binarize
        self.normalize = normalize

    def __call__(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        if self.binarize:
            x = (x > 0).to(torch.float32)
        elif self.normalize:
            denom = x.flatten(1).amax(dim=1).clamp_min(1.0)
            view_shape = [x.shape[0]] + [1] * (x.ndim - 1)
            x = x / denom.view(*view_shape)
        return x


class SimpleMLPSNN(nn.Module):
    """Simple multi-layer perceptron SNN.

    Input shape for multi-step mode:
      [T, N, C, H, W] or [T, N, D]
    Output shape:
      [T, N, num_classes]

    Hidden layers use LIF neurons. The readout can be either:
      - linear: final analog logits, stable with CE loss (default)
      - spike : final LIF spike rates are used as CE logits
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 10,
        hidden_dims: Sequence[int] = (512, 256),
        dropout: float = 0.0,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        detach_reset: bool = True,
        readout: str = "linear",
        surrogate_name: str = "atan",
    ):
        super().__init__()
        if readout not in {"linear", "spike"}:
            raise ValueError("readout must be either 'linear' or 'spike'")
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.hidden_dims = tuple(int(v) for v in hidden_dims)
        self.readout = readout

        surr = build_surrogate(surrogate_name)
        modules: List[nn.Module] = []
        in_features = self.input_dim
        for hidden in self.hidden_dims:
            modules.append(layer.Linear(in_features, hidden, bias=True))
            modules.append(
                neuron.LIFNode(
                    tau=tau,
                    v_threshold=v_threshold,
                    v_reset=v_reset,
                    surrogate_function=surr,
                    detach_reset=detach_reset,
                    step_mode="m",
                )
            )
            if dropout > 0:
                modules.append(layer.Dropout(dropout))
            in_features = hidden

        modules.append(layer.Linear(in_features, self.num_classes, bias=True))
        if readout == "spike":
            modules.append(
                neuron.LIFNode(
                    tau=tau,
                    v_threshold=v_threshold,
                    v_reset=v_reset,
                    surrogate_function=surr,
                    detach_reset=detach_reset,
                    step_mode="m",
                )
            )
        self.net = nn.Sequential(*modules)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.ndim > 3:
            x_seq = x_seq.flatten(start_dim=2)
        return self.net(x_seq)


def build_surrogate(name: str):
    name = name.lower()
    if name == "atan":
        return surrogate.ATan()
    if name == "sigmoid":
        return surrogate.Sigmoid()
    if name == "fast_sigmoid":
        # Not all SpikingJelly versions expose FastSigmoid.
        if hasattr(surrogate, "FastSigmoid"):
            return surrogate.FastSigmoid()
        return surrogate.Sigmoid()
    raise ValueError(f"Unsupported surrogate function: {name}")


def parse_hidden_dims(value: str) -> Tuple[int, ...]:
    dims = tuple(int(v.strip()) for v in value.split(",") if v.strip())
    if not dims:
        raise argparse.ArgumentTypeError("--hidden-dims must contain at least one integer, e.g. 512,256")
    if any(v <= 0 for v in dims):
        raise argparse.ArgumentTypeError("all hidden dimensions must be positive")
    return dims


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


def _is_nmnist_events_np_ready(raw_root: Path) -> bool:
    """Return True when SpikingJelly-ready N-MNIST npz files exist."""
    train_dir = raw_root / "train"
    test_dir = raw_root / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        return False
    try:
        return any(train_dir.rglob("*.npz")) and any(test_dir.rglob("*.npz"))
    except OSError:
        return False


def _is_nmnist_extract_ready(extract_root: Path) -> bool:
    """Return True when extracted Train/Test binary trees appear to exist."""
    train_dir = extract_root / "Train"
    test_dir = extract_root / "Test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        return False
    try:
        return any(train_dir.rglob("*.bin")) and any(test_dir.rglob("*.bin"))
    except OSError:
        return False


def _manual_create_nmnist_events_np(extract_root: Path, raw_root: Path) -> None:
    """Version-independent fallback for converting N-MNIST .bin files to events_np.

    Different SpikingJelly releases expose different helper names for this
    conversion. This fallback mirrors the official converter: each .bin event file
    is decoded with SpikingJelly's ATIS loader and saved as an .npz file with
    keys t, x, y, p under events_np/{train,test}/{class}/.
    """
    import multiprocessing
    from concurrent.futures import ThreadPoolExecutor

    try:
        from spikingjelly.datasets import utils as sj_utils
        load_atis_bin = sj_utils.load_ATIS_bin
        np_savez = sj_utils.np_savez
    except Exception:
        from spikingjelly import datasets as sjds
        load_atis_bin = sjds.load_ATIS_bin
        np_savez = getattr(sjds, "np_savez", None)
        if np_savez is None:
            import numpy as np

            def np_savez(file_name, **kwargs):
                np.savez(file_name, **kwargs)

    def convert_one(source_file: Path, target_file: Path) -> None:
        events = load_atis_bin(str(source_file))
        np_savez(
            str(target_file),
            t=events["t"],
            x=events["x"],
            y=events["y"],
            p=events["p"],
        )

    max_workers = max(1, min(multiprocessing.cpu_count(), 8))
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for split_src, split_dst in [("Train", "train"), ("Test", "test")]:
            source_dir = extract_root / split_src
            target_dir = raw_root / split_dst
            target_dir.mkdir(parents=True, exist_ok=True)
            for class_dir in sorted(source_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                np_dir = target_dir / class_dir.name
                np_dir.mkdir(parents=True, exist_ok=True)
                for bin_file in sorted(class_dir.glob("*.bin")):
                    target_file = np_dir / f"{bin_file.stem}.npz"
                    if target_file.exists():
                        continue
                    futures.append(pool.submit(convert_one, bin_file, target_file))
        for future in futures:
            future.result()


def ensure_nmnist_prepared(root: Path, download: bool = True) -> None:
    """Prepare N-MNIST for SpikingJelly if root/events_np is absent.

    SpikingJelly versions differ in the name of the N-MNIST raw-event conversion
    helper:
      - newer/development versions: create_raw_from_extracted(...)
      - PyPI stable 0.0.0.0.14: create_events_np_files(...)

    This function supports both names and falls back to a manual converter if
    neither helper is available. The final expected tree is:
      root/events_np/train/<class>/*.npz
      root/events_np/test/<class>/*.npz
    """
    raw_root = root / "events_np"
    if _is_nmnist_events_np_ready(raw_root):
        return
    if raw_root.exists() and not _is_nmnist_events_np_ready(raw_root):
        print(f"[N-MNIST] Incomplete events_np found; rebuilding: {raw_root}")
        shutil.rmtree(raw_root)

    if not download:
        raise FileNotFoundError(
            f"N-MNIST events_np was not found under {raw_root}. "
            "Use --download, or manually prepare events_np/train and events_np/test."
        )

    # Lazy imports avoid importing torchvision unless N-MNIST is requested.
    try:
        from spikingjelly.datasets.n_mnist import NMNIST
        from torchvision.datasets.utils import check_integrity, download_url, extract_archive
    except Exception as exc:
        raise RuntimeError(
            "N-MNIST automatic download/preprocess needs both SpikingJelly datasets and torchvision. "
            "Install compatible PyTorch/torchvision/SpikingJelly versions."
        ) from exc

    root.mkdir(parents=True, exist_ok=True)
    download_root = root / "download"
    extract_root = root / "extract"
    download_root.mkdir(parents=True, exist_ok=True)

    for filename, url, md5 in NMNIST_RESOURCES:
        file_path = download_root / filename
        if not check_integrity(str(file_path), md5):
            if file_path.exists():
                file_path.unlink()
            print(f"[N-MNIST] Downloading {filename} to {download_root}")
            download_url(url=url, root=str(download_root), filename=filename, md5=md5)
        else:
            print(f"[N-MNIST] Found valid archive: {file_path}")

    if not _is_nmnist_extract_ready(extract_root):
        if extract_root.exists():
            print(f"[N-MNIST] Incomplete extract directory found; rebuilding: {extract_root}")
            shutil.rmtree(extract_root)
        extract_root.mkdir(parents=True, exist_ok=True)
        print(f"[N-MNIST] Extracting archives to {extract_root}")
        if hasattr(NMNIST, "extract_downloaded_files"):
            NMNIST.extract_downloaded_files(download_root, extract_root)
        else:
            for archive_name, _, _ in NMNIST_RESOURCES:
                extract_archive(str(download_root / archive_name), str(extract_root))
    else:
        print(f"[N-MNIST] Existing extract directory found: {extract_root}")

    raw_root.mkdir(parents=True, exist_ok=True)
    print(f"[N-MNIST] Creating SpikingJelly events_np under {raw_root}")

    if hasattr(NMNIST, "create_raw_from_extracted"):
        # Newer SpikingJelly development API.
        NMNIST.create_raw_from_extracted(extract_root, raw_root)
    elif hasattr(NMNIST, "create_events_np_files"):
        # SpikingJelly PyPI stable 0.0.0.0.14 API.
        NMNIST.create_events_np_files(str(extract_root), str(raw_root))
    else:
        # Last-resort compatibility path.
        _manual_create_nmnist_events_np(extract_root, raw_root)

    if not _is_nmnist_events_np_ready(raw_root):
        raise RuntimeError(
            f"N-MNIST preprocessing finished but events_np is still incomplete: {raw_root}"
        )

def build_datasets(args: argparse.Namespace) -> Tuple[Optional[Dataset], Optional[Dataset], Dataset, DatasetSpec]:
    data_dir = Path(args.data_dir).expanduser().resolve()
    dataset_name = args.dataset.lower()

    if dataset_name in {"mnist", "fmnist"}:
        # Lazy import avoids top-level torchvision dependency during sanity checks.
        try:
            from torchvision import datasets, transforms
        except Exception as exc:
            raise RuntimeError(
                "MNIST/Fashion-MNIST loading needs torchvision. Install a compatible "
                "PyTorch/torchvision pair."
            ) from exc

        transform = transforms.ToTensor()  # direct pixel intensity in [0, 1]
        cls = datasets.MNIST if dataset_name == "mnist" else datasets.FashionMNIST
        full_train = cls(root=str(data_dir), train=True, transform=transform, download=args.download)
        test_set = cls(root=str(data_dir), train=False, transform=transform, download=args.download)
        spec = DatasetSpec(
            name=dataset_name,
            input_shape=(1, 28, 28),
            num_classes=10,
            time_steps=args.T,
            is_event_dataset=False,
        )
    elif dataset_name == "nmnist":
        try:
            from spikingjelly.datasets.n_mnist import NMNIST
        except Exception as exc:
            raise RuntimeError(
                "N-MNIST loading needs spikingjelly.datasets and torchvision. "
                "Install compatible PyTorch/torchvision/SpikingJelly versions."
            ) from exc

        root = data_dir / "N-MNIST"
        ensure_nmnist_prepared(root, download=args.download)
        transform = EventFrameToTensor(binarize=args.nmnist_binarize, normalize=args.nmnist_normalize)
        full_train = NMNIST(
            root=str(root),
            train=True,
            data_type="frame",
            frames_number=args.T,
            split_by=args.nmnist_split_by,
            transform=transform,
        )
        test_set = NMNIST(
            root=str(root),
            train=False,
            data_type="frame",
            frames_number=args.T,
            split_by=args.nmnist_split_by,
            transform=transform,
        )
        spec = DatasetSpec(
            name=dataset_name,
            input_shape=(2, 34, 34),
            num_classes=10,
            time_steps=args.T,
            is_event_dataset=True,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    full_train = limit_dataset(full_train, args.limit_train_total)
    test_set = limit_dataset(test_set, args.limit_test)

    if args.mode == "test":
        return None, None, test_set, spec

    val_ratio = float(args.val_ratio)
    if val_ratio <= 0:
        train_set = full_train
        val_set = None
    else:
        val_size = max(1, int(len(full_train) * val_ratio))
        train_size = len(full_train) - val_size
        if train_size <= 0:
            raise ValueError("Validation split is too large; no training samples remain.")
        generator = torch.Generator().manual_seed(args.seed)
        train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)

    train_set = limit_dataset(train_set, args.limit_train)
    val_set = limit_dataset(val_set, args.limit_val) if val_set is not None else None
    return train_set, val_set, test_set, spec


def limit_dataset(dataset: Optional[Dataset], limit: int) -> Optional[Dataset]:
    if dataset is None or limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, list(range(limit)))


def build_loader(
    dataset: Optional[Dataset],
    batch_size: int,
    shuffle: bool,
    args: argparse.Namespace,
) -> Optional[DataLoader]:
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
    """Return a tensor with shape [T, N, C, H, W] or [T, N, D]."""
    if spec.is_event_dataset:
        # Expected from DataLoader: [N, T, C, H, W]
        if x.ndim != 5:
            raise ValueError(f"N-MNIST frames must have shape [N,T,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] != spec.time_steps:
            raise ValueError(f"Expected T={spec.time_steps}, got input T={x.shape[1]}")
        return x.transpose(0, 1).contiguous()

    # Static frame dataset: [N, C, H, W]. Repeat the original image directly.
    if x.ndim == 3:
        x = x.unsqueeze(1)
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
                backend = "cupy"
            except Exception as exc:
                warnings.warn(f"CuPy backend could not be enabled ({exc}). Falling back to torch backend.")
    return backend


def build_model_from_args(args: argparse.Namespace, spec: DatasetSpec) -> SimpleMLPSNN:
    return SimpleMLPSNN(
        input_dim=spec.input_dim,
        num_classes=spec.num_classes,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        tau=args.tau,
        v_threshold=args.v_threshold,
        v_reset=None if args.v_reset.lower() == "none" else float(args.v_reset),
        detach_reset=args.detach_reset,
        readout=args.readout,
        surrogate_name=args.surrogate,
    )


def build_model_from_config(model_config: Dict) -> SimpleMLPSNN:
    return SimpleMLPSNN(
        input_dim=int(model_config["input_dim"]),
        num_classes=int(model_config["num_classes"]),
        hidden_dims=tuple(int(v) for v in model_config["hidden_dims"]),
        dropout=float(model_config["dropout"]),
        tau=float(model_config["tau"]),
        v_threshold=float(model_config["v_threshold"]),
        v_reset=model_config["v_reset"],
        detach_reset=bool(model_config["detach_reset"]),
        readout=str(model_config["readout"]),
        surrogate_name=str(model_config["surrogate"]),
    )


def model_config_from_args(args: argparse.Namespace, spec: DatasetSpec) -> Dict:
    return {
        "input_dim": spec.input_dim,
        "num_classes": spec.num_classes,
        "hidden_dims": list(args.hidden_dims),
        "dropout": args.dropout,
        "tau": args.tau,
        "v_threshold": args.v_threshold,
        "v_reset": None if args.v_reset.lower() == "none" else float(args.v_reset),
        "detach_reset": args.detach_reset,
        "readout": args.readout,
        "surrogate": args.surrogate,
    }


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
    """For test mode, use checkpoint metadata so N-MNIST T matches saved frames."""
    if ckpt is None or args.ignore_ckpt_config:
        return
    ds_cfg = ckpt.get("dataset_spec")
    if isinstance(ds_cfg, dict):
        args.T = int(ds_cfg.get("time_steps", args.T))
        args.dataset = str(ds_cfg.get("name", args.dataset))


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
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
            # Temporally truncated BPTT: update after every k-step chunk.
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

                # This is the actual temporal truncation. Hidden states carry forward,
                # but their computation graph is cut at chunk boundaries.
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


def run_train(args: argparse.Namespace, device: torch.device) -> Tuple[Path, Dict[str, float]]:
    train_set, val_set, test_set, spec = build_datasets_ddp_safe(args)
    if train_set is None:
        raise RuntimeError("Training mode requires a training dataset.")

    train_loader = build_loader(train_set, args.batch_size, True, args)
    val_loader = build_loader(val_set, args.batch_size, False, args) if val_set is not None else None

    model = build_model_from_args(args, spec).to(device)
    backend = configure_snn_backend(model, args, device)
    if args.compile:
        try:
            model = torch.compile(model)  # type: ignore[assignment]
        except Exception as exc:
            warnings.warn(f"torch.compile failed: {exc}. Continuing without compilation.")

    model = wrap_model_for_ddp(model, args, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = build_grad_scaler(device, args.amp)
    model_config = model_config_from_args(args, spec)
    best_path = checkpoint_path(args)

    if is_main_process():
        print("=" * 80)
        print(f"Dataset       : {spec.name}  input_shape={spec.input_shape}  T={spec.time_steps}")
        print(f"Model         : hidden_dims={args.hidden_dims} readout={args.readout}")
        print(f"Training      : lr={args.lr} batch_size={args.batch_size} epochs={args.epochs}")
        print(f"BPTT          : {'TBPTT' if args.tbptt else 'standard BPTT'} k={args.tbptt_k if args.tbptt else spec.time_steps}")
        print(f"Backend       : {backend} amp={args.amp and device.type == 'cuda'} device={device}")
        print(f"Best ckpt     : {best_path}")
        print("=" * 80)

    best_score = -math.inf
    best_metrics: Dict[str, float] = {}
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        if isinstance(getattr(train_loader, "sampler", None), DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        if is_main_process():
            print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device, spec, args
        )
        if is_main_process():
            print(
                f"[train] loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f} "
                f"time={train_metrics['seconds']:.1f}s"
            )

        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, criterion, device, spec, args, name="val")
        else:
            val_metrics = train_metrics

        monitor = val_metrics["acc"] if args.monitor == "acc" else -val_metrics["loss"]
        monitor_display = val_metrics[args.monitor]
        improved = monitor > (best_score + args.min_delta)
        if improved:
            best_score = monitor
            best_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
            best_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            save_checkpoint(best_path, model, optimizer, args, spec, model_config, epoch, best_metrics)
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

    # Save a compact config JSON next to the checkpoint for convenient PyCharm inspection.
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
    """Fast no-download check for forward/backward/TBPTT logic."""
    print("[sanity] Running random-data checks. No dataset download is used.")
    for name, spec in [
        ("mnist", DatasetSpec("mnist", (1, 28, 28), 10, args.T, False)),
        ("nmnist", DatasetSpec("nmnist", (2, 34, 34), 10, args.T, True)),
    ]:
        model = build_model_from_args(args, spec).to(device)
        configure_snn_backend(model, args, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        scaler = build_grad_scaler(device, args.amp)
        model.train()
        functional.reset_net(model)

        if spec.is_event_dataset:
            x = torch.rand(args.sanity_batch_size, args.T, *spec.input_shape, device=device)
            x = (x > 0.95).float()
        else:
            x = torch.rand(args.sanity_batch_size, *spec.input_shape, device=device)
        y = torch.randint(0, spec.num_classes, (args.sanity_batch_size,), device=device)
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
        print(f"[sanity:{name}] ok")


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MLP-SNN for MNIST/Fashion-MNIST/N-MNIST with standard BPTT or truncated BPTT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Run mode / data
    p.add_argument("--mode", choices=["train", "test", "train_test", "sanity"], default="train_test")
    p.add_argument("--dataset", choices=["mnist", "fmnist", "nmnist"], default="mnist")
    p.add_argument("--data-dir", type=str, default="/home/leehyunjong/PycharmProjects/Machine_Learning/SNN/TA_BPTT/Motivation/data")
    p.add_argument("--out-dir", type=str, default="./runs/snn_mlp")
    p.add_argument("--save-name", type=str, default="best_model.pt")
    p.add_argument("--ckpt", type=str, default="", help="Checkpoint path for test mode. Defaults to out-dir/save-name.")
    p.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ignore-ckpt-config", action="store_true")

    # Core hyperparameters
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--T", type=int, default=8, help="Number of SNN time steps / N-MNIST integrated frames.")
    p.add_argument("--hidden-dims", type=parse_hidden_dims, default=parse_hidden_dims("512,256"))
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # SNN neuron hyperparameters
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--v-threshold", type=float, default=1.0)
    p.add_argument("--v-reset", type=str, default="0.0", help="Use 'None' for soft reset.")
    p.add_argument("--detach-reset", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--surrogate", choices=["atan", "sigmoid", "fast_sigmoid"], default="atan")
    p.add_argument("--readout", choices=["linear", "spike"], default="linear")

    # BPTT / TBPTT
    p.add_argument("--tbptt", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tbptt-k", type=int, default=2, help="Temporal truncation length k.")
    p.add_argument("--eval-chunk-size", type=int, default=0, help="0 means evaluate all T steps at once.")

    # N-MNIST frame conversion
    p.add_argument("--nmnist-split-by", choices=["number", "time"], default="number")
    p.add_argument("--nmnist-binarize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--nmnist-normalize", action=argparse.BooleanOptionalAction, default=False)

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
    p.add_argument("--ddp", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable DistributedDataParallel when launched with torchrun.")
    p.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=0,
                   help=argparse.SUPPRESS)

    # Logging / reproducibility / debug limits
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--limit-train-total", type=int, default=0, help="Limit full train set before val split. 0 disables.")
    p.add_argument("--limit-train", type=int, default=0, help="Limit train subset after val split. 0 disables.")
    p.add_argument("--limit-val", type=int, default=0, help="Limit validation subset. 0 disables.")
    p.add_argument("--limit-test", type=int, default=0, help="Limit test subset. 0 disables.")
    p.add_argument("--sanity-batch-size", type=int, default=8)
    return p


def main() -> None:
    args = make_argparser().parse_args()
    init_distributed_mode(args)
    if args.T <= 0:
        raise ValueError("--T must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
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
