from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _prepend_once(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def _ensure_namespace_package(name: str, path: Path) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        module.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = module
        return module
    if not hasattr(module, '__path__'):
        module.__path__ = [str(path)]  # type: ignore[attr-defined]
        return module
    if str(path) not in module.__path__:  # type: ignore[attr-defined]
        module.__path__.append(str(path))  # type: ignore[attr-defined]
    return module


def _ensure_snntorch_stub() -> None:
    if 'snntorch' in sys.modules and 'snntorch.surrogate' in sys.modules:
        return
    try:
        importlib.import_module('snntorch.surrogate')
        return
    except ModuleNotFoundError:
        pass

    surrogate_mod = ModuleType('snntorch.surrogate')

    def _atan(alpha: float = 2.0):
        class _Atan:
            def __call__(self, x, *args, **kwargs):
                return (x > 0).to(dtype=x.dtype)

        return _Atan()

    surrogate_mod.atan = _atan  # type: ignore[attr-defined]
    snntorch_mod = ModuleType('snntorch')
    snntorch_mod.surrogate = surrogate_mod  # type: ignore[attr-defined]
    sys.modules['snntorch'] = snntorch_mod
    sys.modules['snntorch.surrogate'] = surrogate_mod


_DRF_MODULE_NAME = '_psd_origin_drf_layers'
_TSLIF_MODULE_NAME = 'SeqSNN.network.snn.TSLIF'
_TSLIF_BASE_MODULE_NAME = 'SeqSNN.network.snn.TSLIF_base'


def load_dh_spike_dense() -> ModuleType:
    root = _PROJECT_ROOT / 'Origin' / 'Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics' / 'SHD'
    _prepend_once(root)
    importlib.invalidate_caches()
    return importlib.import_module('SNN_layers.spike_dense')



def load_tclif_module() -> ModuleType:
    root = _PROJECT_ROOT / 'Origin' / 'TC-LIF A Two-Compartment Spiking Neuron Model for Long-Term Sequential Modelling' / 'SHD-SSC'
    _prepend_once(root)
    importlib.invalidate_caches()
    return importlib.import_module('spiking_neuron.TCLIF')



def load_tslif_module() -> ModuleType:
    if _TSLIF_MODULE_NAME in sys.modules:
        return sys.modules[_TSLIF_MODULE_NAME]

    root = _PROJECT_ROOT / 'Origin' / 'TS-LIF A TEMPORAL SEGMENT SPIKING NEURON NETWORK FOR TIME SERIES FORECASTING' / 'SeqSNN'
    snn_root = root / 'network' / 'snn'
    _ensure_snntorch_stub()
    _ensure_namespace_package('SeqSNN', root)
    _ensure_namespace_package('SeqSNN.network', root / 'network')
    snn_pkg = _ensure_namespace_package('SeqSNN.network.snn', snn_root)

    if _TSLIF_BASE_MODULE_NAME not in sys.modules:
        base_spec = importlib.util.spec_from_file_location(_TSLIF_BASE_MODULE_NAME, snn_root / 'TSLIF_base.py')
        if base_spec is None or base_spec.loader is None:
            raise ImportError(f'unable to load TS-LIF base module from {snn_root / "TSLIF_base.py"}')
        base_module = importlib.util.module_from_spec(base_spec)
        sys.modules[_TSLIF_BASE_MODULE_NAME] = base_module
        base_spec.loader.exec_module(base_module)
        setattr(snn_pkg, 'TSLIF_base', base_module)
    else:
        setattr(snn_pkg, 'TSLIF_base', sys.modules[_TSLIF_BASE_MODULE_NAME])

    spec = importlib.util.spec_from_file_location(_TSLIF_MODULE_NAME, snn_root / 'TSLIF.py')
    if spec is None or spec.loader is None:
        raise ImportError(f'unable to load TS-LIF origin module from {snn_root / "TSLIF.py"}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[_TSLIF_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module



def load_drf_newlayer_module() -> ModuleType:
    if _DRF_MODULE_NAME in sys.modules:
        return sys.modules[_DRF_MODULE_NAME]

    path = _PROJECT_ROOT / 'Origin' / 'Dendritic Resonate-and-Fire Neuron for Effective and Efficient Long Sequence Modeling' / 'models' / 'layers.py'
    spec = importlib.util.spec_from_file_location(_DRF_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'unable to load D-RF origin module from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[_DRF_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


class OriginSurrogateAdapter:
    """Adapter so project spike surrogates can be passed into origin nodes.

    Some released nodes call the surrogate with extra scalar arguments. The
    project spike function only needs the membrane tensor, so we ignore any
    additional positional/keyword arguments.
    """

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x, *args, **kwargs):
        return self.fn(x)
