import pytest

torch = pytest.importorskip("torch")

from src.element_fft import _component_matrix, _element_fft_matrix


def test_element_fft_raw_constant_signal_dc_component():
    maps = torch.ones(2, 3, 4)
    fft = _element_fft_matrix(maps, variant='raw')
    assert fft.shape == (3, 3)
    assert torch.allclose(fft[:, 0].real, torch.full((3,), 4.0))
    assert torch.allclose(fft[:, 1:].abs(), torch.zeros(3, 2))


def test_element_fft_centered_constant_signal_is_zero():
    maps = torch.ones(2, 3, 4)
    fft = _element_fft_matrix(maps, variant='centered')
    assert torch.allclose(fft.abs(), torch.zeros_like(fft.abs()))


def test_element_fft_component_magnitude_and_phase():
    matrix = torch.tensor([[1 + 0j, 0 + 1j]])
    magnitude = _component_matrix(matrix, component='magnitude')
    phase = _component_matrix(matrix, component='phase')
    assert torch.allclose(magnitude, torch.ones_like(magnitude))
    assert torch.allclose(phase[:, 0], torch.zeros(1))
    assert torch.allclose(phase[:, 1], torch.full((1,), torch.pi / 2))
