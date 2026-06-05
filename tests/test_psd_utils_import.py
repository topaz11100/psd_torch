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


def test_signal_window_none_disables_taper():
    import torch
    from src.signal.psd_utils import exact_periodogram_from_maps, temporal_window, normalize_signal_window

    assert normalize_signal_window(False) == 'none'
    assert normalize_signal_window(True) == 'hann'
    assert torch.allclose(temporal_window(8, signal_window='none'), torch.ones(8, dtype=temporal_window(8, signal_window='none').dtype))
    hann = temporal_window(8, signal_window='hann')
    assert not torch.allclose(hann, torch.ones(8, dtype=hann.dtype))

    maps = torch.arange(2 * 3 * 16, dtype=torch.float32).reshape(2, 3, 16)
    freqs_h, psd_h = exact_periodogram_from_maps(maps, signal_window='hann')
    freqs_n, psd_n = exact_periodogram_from_maps(maps, signal_window='none')
    assert torch.allclose(freqs_h, freqs_n)
    assert psd_h.shape == psd_n.shape
    assert not torch.allclose(psd_h, psd_n)
