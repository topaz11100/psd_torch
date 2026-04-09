"""SNN builder utilities for neuron-layer-only classifier heads."""

from __future__ import annotations

import torch
from torch import nn

from src.neurons.vanilla_lif import VanillaLIFCell
from src.neurons.vanilla_rf import VanillaRFCell


class SimpleSNNClassifier(nn.Module):
    """Minimal neuron-layer classifier with output membrane/spike records."""

    def __init__(self, input_channels: int, hidden_size: int, num_classes: int, mode: str = "lif"):
        super().__init__()
        self.mode = mode
        self.in_proj = nn.Linear(input_channels, hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_classes)
        if mode == "lif":
            self.cell = VanillaLIFCell(hidden_size)
        else:
            self.cell = VanillaRFCell(hidden_size)

    def forward(self, x: torch.Tensor, final_membrane_disable_spike: bool = False):
        """Run sequence input x:(B,T,C) and return spikes/membranes of output layer."""

        b, t, _ = x.shape
        h = torch.zeros(b, self.in_proj.out_features, device=x.device)
        if self.mode == "lif":
            rf_state = None
        else:
            rf_state = (torch.zeros_like(h), torch.zeros_like(h))
        spikes = []
        mems = []
        for i in range(t):
            xh = self.in_proj(x[:, i, :])
            if self.mode == "lif":
                s, h = self.cell(xh, h)
            else:
                s, rf_state = self.cell(xh, rf_state)
                h = rf_state[0]
            out_mem = self.out_proj(h)
            out_spike = torch.zeros_like(out_mem) if final_membrane_disable_spike else (out_mem > 1.0).to(out_mem.dtype)
            spikes.append(out_spike)
            mems.append(out_mem)
        return torch.stack(spikes, dim=1), torch.stack(mems, dim=1)


def build_snn_classifier(input_channels: int, hidden_size: int, num_classes: int, mode: str = "lif") -> nn.Module:
    """Factory for classifier models."""

    return SimpleSNNClassifier(input_channels=input_channels, hidden_size=hidden_size, num_classes=num_classes, mode=mode)
