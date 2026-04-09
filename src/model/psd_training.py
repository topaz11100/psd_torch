"""Training helpers for psd_analysis run loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.readout.readout import apply_readout


@dataclass
class EpochMetric:
    """Per-epoch train/test accuracy summary."""

    epoch: int
    train_accuracy: float
    test_accuracy: float


def _evaluate(model: nn.Module, loader: DataLoader, readout_mode: str, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            spikes, mem = model(x, final_membrane_disable_spike=(readout_mode == "final_membrane"))
            if getattr(criterion, "requires_output_record", False):
                analysis = criterion.analyze_output_record(spikes, mem)
                pred = criterion.predictions_from_analysis(analysis)
            else:
                logits = apply_readout(readout_mode, spikes, mem)
                pred = torch.argmax(logits, dim=-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return float(correct) / float(max(total, 1))


def train_for_psd(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, readout_mode: str, epochs: int, criterion: nn.Module, device: torch.device) -> List[EpochMetric]:
    """Run simple training loop and return epoch metrics."""

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    history: List[EpochMetric] = []
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            spikes, mem = model(x, final_membrane_disable_spike=(readout_mode == "final_membrane"))
            if getattr(criterion, "requires_output_record", False):
                loss = criterion(spikes, mem, y)
            else:
                logits = apply_readout(readout_mode, spikes, mem)
                loss = ce(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc = _evaluate(model, train_loader, readout_mode, criterion, device)
        test_acc = _evaluate(model, test_loader, readout_mode, criterion, device)
        history.append(EpochMetric(epoch=epoch, train_accuracy=train_acc, test_accuracy=test_acc))
    return history
