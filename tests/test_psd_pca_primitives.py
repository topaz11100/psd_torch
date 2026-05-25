import pytest

torch = pytest.importorskip('torch')

from src.signal.psd_utils import (
    apply_fixed_pca_basis,
    auto_spectral_matrix_from_mode_maps,
    compute_fixed_pca_basis,
    pca_dim_from_cli_vector,
    scalar_representative_maps,
)


def test_scalar_representative_maps_shapes_and_values():
    maps = torch.tensor(
        [
            [[1.0, 3.0], [5.0, 7.0], [9.0, 11.0]],
            [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]],
        ]
    )
    mean_maps = scalar_representative_maps(maps, reducer='mean')
    median_maps = scalar_representative_maps(maps, reducer='median')
    assert mean_maps.shape == (2, 1, 2)
    assert median_maps.shape == (2, 1, 2)
    assert torch.allclose(mean_maps[0, 0], torch.tensor([5.0, 7.0]))
    assert torch.allclose(median_maps[1, 0], torch.tensor([6.0, 8.0]))


def test_pca_dim_from_cli_vector_default_tail_and_clamp():
    assert pca_dim_from_cli_vector([], 0, 3) == 3
    assert pca_dim_from_cli_vector(None, 2, 10) == 4
    assert pca_dim_from_cli_vector([8, 2], 0, 5) == 5
    assert pca_dim_from_cli_vector([8, 2], 5, 5) == 2
    assert pca_dim_from_cli_vector([0], 0, 5) == 1


def test_compute_and_apply_fixed_pca_basis_shapes_clamp_and_dtype_device():
    maps = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    basis, centroid = compute_fixed_pca_basis(maps, target_dim=99)
    assert basis.shape == (3, 3)
    assert centroid.shape == (3,)
    assert basis.requires_grad is False
    assert centroid.requires_grad is False

    projected = apply_fixed_pca_basis(maps, basis.double(), centroid.double())
    assert projected.shape == (2, 3, 4)
    assert projected.dtype == maps.dtype
    assert projected.device == maps.device


def test_compute_fixed_pca_basis_observation_clamp():
    maps = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]])  # (1,5,1), obs=1
    basis, centroid = compute_fixed_pca_basis(maps, target_dim=4)
    assert basis.shape == (5, 1)
    assert centroid.shape == (5,)


def test_auto_spectral_matrix_shape_hermitian_and_diagonal_reality():
    mode_maps = torch.tensor(
        [
            [[1.0, -1.0, 1.0, -1.0], [0.0, 1.0, 0.0, 1.0]],
            [[-1.0, 1.0, -1.0, 1.0], [1.0, 0.0, 1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    freqs, matrix = auto_spectral_matrix_from_mode_maps(mode_maps)
    assert matrix.shape == (freqs.numel(), 2, 2)
    assert torch.allclose(matrix, matrix.conj().transpose(1, 2), atol=1e-6, rtol=1e-5)
    diag = torch.diagonal(matrix, dim1=1, dim2=2)
    assert torch.allclose(diag.imag, torch.zeros_like(diag.imag), atol=1e-6, rtol=0.0)
    assert torch.all(diag.real >= -1e-8)
