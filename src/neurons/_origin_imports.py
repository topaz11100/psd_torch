"""Stable import helpers for origin-code thin wrappers.

The user explicitly requested that origin-backed models keep author code
rather than ad-hoc reimplementations. These helpers therefore load the
released origin modules directly from ``origin/`` and expose them to the
wrapper modules.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from src.neurons._common import surrogate_spike


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ORIGIN_ROOT = _PROJECT_ROOT / 'origin'
_ORIGIN_NEURON_ROOT = _ORIGIN_ROOT / 'neuron_model'


def _load_module(module_name: str, file_path: Path, *, extra_sys_paths: list[Path] | None = None) -> types.ModuleType:
    """Internal helper that load module."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    if not file_path.exists():
        raise FileNotFoundError(f'Could not locate origin module: {file_path}')
    extra_sys_paths = extra_sys_paths or []
    for path in reversed(extra_sys_paths):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not create module spec for {file_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_package_stub(module_name: str, package_path: Path) -> types.ModuleType:
    """Register a lightweight package stub without executing package ``__init__``.

    Some origin repositories wire broad package-level imports that depend on
    training-time utilities unavailable in this project. For thin wrappers we
    only need a narrow subset of those packages, so we pre-register package
    modules with the correct ``__path__`` and then load the required leaf
    modules directly.
    """

    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    package = types.ModuleType(module_name)
    package.__path__ = [str(package_path)]  # type: ignore[attr-defined]
    package.__package__ = module_name
    sys.modules[module_name] = package
    return package


def _ensure_snntorch_stub() -> None:
    """Internal helper that ensure snntorch stub."""
    if 'snntorch' in sys.modules:
        return
    snntorch_module = types.ModuleType('snntorch')
    surrogate_module = types.ModuleType('snntorch.surrogate')

    class _AtanSurrogate:
        """Internal class for ``AtanSurrogate`` in the ``_origin_imports`` module."""
        def __call__(self, x, *args, **kwargs):
            """Call the object like a function."""
            return surrogate_spike(x)

    def atan(alpha: float = 2.0):  # noqa: ARG001 - compatibility signature
        """Handle ``atan`` for the ``_origin_imports`` module."""
        return _AtanSurrogate()

    surrogate_module.atan = atan
    snntorch_module.surrogate = surrogate_module
    sys.modules['snntorch'] = snntorch_module
    sys.modules['snntorch.surrogate'] = surrogate_module


def _first_existing_dir(*candidates: Path) -> Path:
    """Return the first existing origin directory among canonical/fallback paths."""

    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = ', '.join(str(path) for path in candidates)
    raise FileNotFoundError(f'Could not locate any supported origin directory: {joined}')


def load_dh_snn_modules() -> tuple[types.ModuleType, types.ModuleType, types.ModuleType]:
    """Load DH-SNN origin modules from the released s-MNIST implementation."""

    smnist_root = _first_existing_dir(
        _ORIGIN_NEURON_ROOT / 'DH-SNN' / 's-mnist',
        _ORIGIN_NEURON_ROOT / 'Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics' / 's-mnist',
    )
    extra = [smnist_root]
    neuron = _load_module('origin_dh_snn_spike_neuron', smnist_root / 'SNN_layers' / 'spike_neuron.py', extra_sys_paths=extra)
    dense = _load_module('origin_dh_snn_spike_dense', smnist_root / 'SNN_layers' / 'spike_dense.py', extra_sys_paths=extra)
    rnn = _load_module('origin_dh_snn_spike_rnn', smnist_root / 'SNN_layers' / 'spike_rnn.py', extra_sys_paths=extra)
    return neuron, dense, rnn


def load_tc_lif_module() -> types.ModuleType:
    """Load the TC-LIF author module from the checked-in origin tree."""

    smnist_root = _first_existing_dir(
        _ORIGIN_NEURON_ROOT / 'TC-LIF' / 'MNIST',
        _ORIGIN_NEURON_ROOT / 'TC-LIF A Two-Compartment Spiking Neuron Model for Long-Term Sequential Modelling' / 'MNIST',
    )
    return _load_module('origin_tc_lif', smnist_root / 'spiking_neuron' / 'TCLIF.py', extra_sys_paths=[smnist_root])


def load_ts_lif_module() -> types.ModuleType:
    """Load the TS-LIF author module from the checked-in origin tree."""

    _ensure_snntorch_stub()
    ts_root = _first_existing_dir(
        _ORIGIN_NEURON_ROOT / 'TS-LIF',
        _ORIGIN_NEURON_ROOT / 'TS-LIF A TEMPORAL SEGMENT SPIKING NEURON NETWORK FOR TIME SERIES FORECASTING',
    )
    seq_root = ts_root / 'SeqSNN'
    network_root = seq_root / 'network'
    snn_root = network_root / 'snn'

    _ensure_package_stub('SeqSNN', seq_root)
    _ensure_package_stub('SeqSNN.network', network_root)
    snn_pkg = _ensure_package_stub('SeqSNN.network.snn', snn_root)
    base_mod = _load_module('SeqSNN.network.snn.TSLIF_base', snn_root / 'TSLIF_base.py', extra_sys_paths=[ts_root])
    setattr(snn_pkg, 'TSLIF_base', base_mod)
    return _load_module(
        'origin_ts_lif',
        snn_root / 'TSLIF.py',
        extra_sys_paths=[ts_root],
    )


def load_d_rf_module() -> types.ModuleType:
    """Load the D-RF author module from the checked-in origin tree."""

    drf_root = _first_existing_dir(
        _ORIGIN_NEURON_ROOT / 'D-RF',
        _ORIGIN_NEURON_ROOT / 'Dendritic Resonate-and-Fire Neuron for Effective and Efficient Long Sequence Modeling',
    )
    return _load_module('origin_d_rf_layers', drf_root / 'models' / 'layers.py', extra_sys_paths=[drf_root])


__all__ = [
    'load_d_rf_module',
    'load_dh_snn_modules',
    'load_tc_lif_module',
    'load_ts_lif_module',
]
