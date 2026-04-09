"""Dataset registry and canonical dataset token resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch.utils.data import Dataset

from .smnist import SMNISTConfig, build_smnist_datasets


DATASET_ALIASES: Dict[str, str] = {
    "s-mnist": "s-mnist",
    "smnist": "s-mnist",
    "s_mnist": "s-mnist",
}


@dataclass
class DatasetBundle:
    """Container for train/test datasets and metadata."""

    name: str
    train: Dataset
    test: Dataset
    num_classes: int
    input_channels: int
    input_length: int


def resolve_dataset_token(token: str) -> str:
    """Resolve a user token to canonical dataset name."""

    key = token.strip().lower()
    if key not in DATASET_ALIASES:
        raise ValueError(f"Unsupported dataset token: {token}")
    return DATASET_ALIASES[key]


def extract_labels(dataset: Dataset) -> torch.Tensor:
    """Extract integer labels from dataset in deterministic order."""

    if hasattr(dataset, "targets"):
        values = getattr(dataset, "targets")
        if isinstance(values, torch.Tensor):
            return values.to(torch.long)
        return torch.as_tensor(values, dtype=torch.long)

    labels = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        labels.append(int(y))
    return torch.tensor(labels, dtype=torch.long)


def build_dataset_bundle(dataset: str, seed: int = 42, use_torchvision: bool = False) -> DatasetBundle:
    """Create canonical dataset bundle used by analysis pipelines."""

    canonical = resolve_dataset_token(dataset)
    if canonical != "s-mnist":
        raise ValueError(f"Unsupported canonical dataset: {canonical}")

    cfg = SMNISTConfig(seed=seed, use_torchvision=use_torchvision)
    train, test = build_smnist_datasets(cfg)
    return DatasetBundle(
        name=canonical,
        train=train,
        test=test,
        num_classes=cfg.num_classes,
        input_channels=1,
        input_length=cfg.seq_len,
    )
