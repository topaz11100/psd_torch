"""Thin wrappers around released TC-LIF origin code."""

from __future__ import annotations

from ._origin_imports import load_module_from_path

_mod = load_module_from_path(
    "origin_tc_lif",
    "Origin/TC-LIF A Two-Compartment Spiking Neuron Model for Long-Term Sequential Modelling/SHD-SSC/spiking_neuron/TCLIF.py",
)

TCLIFNode = _mod.TCLIFNode
