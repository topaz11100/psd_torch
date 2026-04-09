"""Thin wrappers around released DH-SNN origin code."""

from __future__ import annotations

from ._origin_imports import load_module_from_path

_mod_dense = load_module_from_path(
    "origin_dh_snn_spike_dense",
    "Origin/Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics/SHD/SNN_layers/spike_dense.py",
)
_mod_rnn = load_module_from_path(
    "origin_dh_snn_spike_rnn",
    "Origin/Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics/SHD/SNN_layers/spike_rnn.py",
)

DHSpikeDense = _mod_dense.spike_dense_test_denri_wotanh_R
DHSpikeRNN = _mod_rnn.spike_rnn_test_denri_wotanh_R
DHOriginDense = _mod_dense.spike_dense_test_origin
DHOriginDenseNoReset = _mod_dense.spike_dense_test_origin_noreset
DHOriginRNN = _mod_rnn.spike_rnn_test_origin
DHOriginRNNNoReset = _mod_rnn.spike_rnn_test_origin_noreset
