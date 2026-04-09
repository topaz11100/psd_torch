"""SCNN model builders for future DVS experiments.

By design, these models are not integrated into current PSD experiment drivers.
"""

from __future__ import annotations

import torch
from torch import nn

from src.neurons.cnn_lif_neuron import ConvLIF2d
from src.neurons.cnn_rf_neuron import ConvRF2d


class SCNNLIFClassifier(nn.Module):
    """Minimal spiking CNN classifier using ConvLIF blocks."""

    def __init__(self, in_channels: int, num_classes: int, hidden_channels: int = 32):
        super().__init__()
        self.block1 = ConvLIF2d(in_channels, hidden_channels)
        self.block2 = ConvLIF2d(hidden_channels, hidden_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(hidden_channels, num_classes)

    def forward(self, x_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward sequence input x_seq:(B,T,C,H,W), returns (spikes, membranes)."""

        b, t, _, _, _ = x_seq.shape
        v1 = v2 = None
        out_spikes, out_mems = [], []
        for i in range(t):
            s1, v1 = self.block1(x_seq[:, i], v1)
            s2, v2 = self.block2(s1, v2)
            feat = self.pool(v2).view(b, -1)
            mem = self.head(feat)
            spike = (mem > 1.0).to(mem.dtype)
            out_spikes.append(spike)
            out_mems.append(mem)
        return torch.stack(out_spikes, dim=1), torch.stack(out_mems, dim=1)


class SCNNRFClassifier(nn.Module):
    """Minimal spiking CNN classifier using ConvRF blocks."""

    def __init__(self, in_channels: int, num_classes: int, hidden_channels: int = 32):
        super().__init__()
        self.block1 = ConvRF2d(in_channels, hidden_channels)
        self.block2 = ConvRF2d(hidden_channels, hidden_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(hidden_channels, num_classes)

    def forward(self, x_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward sequence input x_seq:(B,T,C,H,W), returns (spikes, membranes)."""

        b, t, _, _, _ = x_seq.shape
        st1 = st2 = None
        out_spikes, out_mems = [], []
        for i in range(t):
            s1, st1 = self.block1(x_seq[:, i], st1)
            s2, st2 = self.block2(s1, st2)
            feat = self.pool(st2[0]).view(b, -1)
            mem = self.head(feat)
            spike = (mem > 1.0).to(mem.dtype)
            out_spikes.append(spike)
            out_mems.append(mem)
        return torch.stack(out_spikes, dim=1), torch.stack(out_mems, dim=1)


def build_scnn_classifier(kind: str, in_channels: int, num_classes: int, hidden_channels: int = 32) -> nn.Module:
    """Factory for reserved SCNN models (`cnn_lif`, `cnn_rf`)."""

    key = kind.strip().lower()
    if key == "cnn_lif":
        return SCNNLIFClassifier(in_channels=in_channels, num_classes=num_classes, hidden_channels=hidden_channels)
    if key == "cnn_rf":
        return SCNNRFClassifier(in_channels=in_channels, num_classes=num_classes, hidden_channels=hidden_channels)
    raise ValueError(f"Unsupported SCNN kind: {kind}")
