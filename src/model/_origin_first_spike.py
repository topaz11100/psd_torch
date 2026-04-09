"""Loader for released first-spike timing and loss modules."""

from __future__ import annotations

from src.neurons._origin_imports import load_module_from_path

TIME_ENCODING_MODULE = load_module_from_path(
    "origin_first_spike_time_encoding",
    "Origin/First-spike coding promotes accurate and efficient spiking neural networks for discrete events with rich temporal structures/superspike/src/time_encoding.py",
)
LOSS_MODULE = load_module_from_path(
    "origin_first_spike_loss",
    "Origin/First-spike coding promotes accurate and efficient spiking neural networks for discrete events with rich temporal structures/utils/loss.py",
)
