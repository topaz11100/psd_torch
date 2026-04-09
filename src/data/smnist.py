"""s-MNIST dataset adapter used by dataset_psd and psd_analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class SMNISTConfig:
    """Configuration for s-MNIST loading.

    If torchvision is available and `use_torchvision=True`, real MNIST is loaded
    and converted to s-MNIST (28x28 -> 784x1, permutation disabled).
    Otherwise deterministic synthetic fallback is used.
    """

    train_size: int = 512
    test_size: int = 128
    seq_len: int = 784
    num_classes: int = 10
    seed: int = 42
    use_torchvision: bool = False
    data_root: str = "./data"


class SequenceMNISTDataset(Dataset):
    """Deterministic in-memory sequence dataset with shape (T, 1)."""

    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


def _build_synthetic_split(size: int, seq_len: int, num_classes: int, seed: int) -> SequenceMNISTDataset:
    g = torch.Generator().manual_seed(seed)
    inputs = torch.rand(size, seq_len, 1, generator=g)
    targets = torch.randint(0, num_classes, (size,), generator=g)
    return SequenceMNISTDataset(inputs=inputs, targets=targets)


def _try_build_torchvision_split(train: bool, data_root: str) -> Optional[SequenceMNISTDataset]:
    try:
        from torchvision import datasets, transforms
    except Exception:
        return None

    ds = datasets.MNIST(root=data_root, train=train, transform=transforms.ToTensor(), download=True)
    images = ds.data.to(torch.float32) / 255.0  # [N, 28, 28]
    inputs = images.view(images.shape[0], 784, 1)
    targets = ds.targets.to(torch.long)
    return SequenceMNISTDataset(inputs=inputs, targets=targets)


def build_smnist_datasets(config: SMNISTConfig) -> Tuple[Dataset, Dataset]:
    """Build train/test datasets with s-MNIST compatible shape."""

    if config.use_torchvision:
        train = _try_build_torchvision_split(train=True, data_root=config.data_root)
        test = _try_build_torchvision_split(train=False, data_root=config.data_root)
        if train is not None and test is not None:
            return train, test

    train = _build_synthetic_split(
        size=config.train_size,
        seq_len=config.seq_len,
        num_classes=config.num_classes,
        seed=config.seed,
    )
    test = _build_synthetic_split(
        size=config.test_size,
        seq_len=config.seq_len,
        num_classes=config.num_classes,
        seed=config.seed + 1,
    )
    return train, test
