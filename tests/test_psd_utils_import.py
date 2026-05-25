import importlib
import py_compile
from pathlib import Path


def test_psd_utils_importable():
    module = importlib.import_module('src.signal.psd_utils')
    assert module is not None


def test_psd_utils_known_symbols_are_callable_when_present():
    module = importlib.import_module('src.signal.psd_utils')
    for name in [
        'scalar_representative_maps',
        'compute_fixed_pca_basis',
        'apply_fixed_pca_basis',
        'pca_dim_from_cli_vector',
        'auto_spectral_matrix_from_mode_maps',
    ]:
        if hasattr(module, name):
            assert callable(getattr(module, name))


def test_psd_utils_py_compile_clean():
    path = Path('src/signal/psd_utils.py')
    py_compile.compile(str(path), doraise=True)
