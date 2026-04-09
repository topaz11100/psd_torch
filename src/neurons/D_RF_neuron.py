"""Thin wrappers around released D-RF origin code."""

from __future__ import annotations

from ._origin_imports import load_module_from_path

_mod = load_module_from_path(
    "origin_drf_layers",
    "Origin/Dendritic Resonate-and-Fire Neuron for Effective and Efficient Long Sequence Modeling/models/layers.py",
)

BiRFKernel = _mod.BiRFKernel
BiRFModel = _mod.BiRFModel
LIFModel = _mod.LIFModel
