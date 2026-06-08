"""origin first-spike module loader.

This file centralizes the path handling for the released First-spike coding
implementation so the rest of the project can keep using a thin wrapper.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from torch import nn


_ROOT = Path(__file__).resolve().parents[2]
_ORIGIN_ROOT = _ROOT / 'origin' / 'readout' / 'first_spike'
_TIME_PATH = _ORIGIN_ROOT / 'superspike' / 'src' / 'time_encoding.py'
_LOSS_PATH = _ORIGIN_ROOT / 'utils' / 'loss.py'


def _load_module(module_name: str, file_path: Path) -> ModuleType:
    """Internal helper that load module."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    if not file_path.exists():
        raise FileNotFoundError(f'Could not locate origin module: {file_path}')
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f'Failed to create import spec for {file_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _patch_origin_loss_module(loss_mod: ModuleType) -> None:
    """Repair the released ``LossFn`` superclass init without changing its formulas.

    The released file uses ``super(LossFn).__init__()`` and therefore skips
    ``nn.Module.__init__``. The wrapper keeps the author implementation intact
    and only adds the missing module initialization step before delegating to
    the original ``__init__`` body.
    """

    loss_cls = getattr(loss_mod, 'LossFn', None)
    if not isinstance(loss_cls, type):
        raise AttributeError('Released first-spike loss module does not define LossFn.')
    if getattr(loss_cls, '_psd_super_init_patched', False):
        return

    original_init = loss_cls.__init__

    def patched_init(self, *args, **kwargs):
        """Internal compatibility shim for the released ``LossFn`` class."""
        nn.Module.__init__(self)
        original_init(self, *args, **kwargs)

    loss_cls.__init__ = patched_init
    loss_cls._psd_super_init_patched = True


def load_first_spike_modules() -> tuple[ModuleType, ModuleType]:
    """Load the released time-encoding and loss modules lazily.

    The origin ``time_encoding.py`` expects ``scipy.signal.gaussian``. Recent
    SciPy versions may expose it under ``scipy.signal.windows.gaussian`` only,
    so the wrapper patches that compatibility alias when needed.
    """

    time_mod = _load_module('origin_first_spike_time_encoding', _TIME_PATH)
    loss_mod = _load_module('origin_first_spike_loss', _LOSS_PATH)

    if hasattr(time_mod, 'signal') and not hasattr(time_mod.signal, 'gaussian'):
        windows = getattr(time_mod.signal, 'windows', None)
        if windows is not None and hasattr(windows, 'gaussian'):
            time_mod.signal.gaussian = windows.gaussian
    _patch_origin_loss_module(loss_mod)
    return time_mod, loss_mod


__all__ = ['load_first_spike_modules']
