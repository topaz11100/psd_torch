import pytest
import torch

from src.signal.psd_utils import (
    apply_fixed_pca_basis,
    auto_spectral_matrix_from_mode_maps,
    compute_fixed_pca_basis,
    pca_dim_from_cli_vector,
    scalar_representative_maps,
)


def test_scalar_representative_maps_mean_median_and_errors():
    x = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5)
    mean = scalar_representative_maps(x, reducer='mean')
    med = scalar_representative_maps(x, reducer='median')
    assert mean.shape == (2, 1, 5)
    assert med.shape == (2, 1, 5)
    assert torch.allclose(mean, x.mean(dim=1, keepdim=True))
    assert torch.allclose(med, x.median(dim=1, keepdim=True).values)
    with pytest.raises(ValueError):
        scalar_representative_maps(x, reducer='x')
    with pytest.raises(ValueError):
        scalar_representative_maps(torch.zeros(2, 3), reducer='mean')


def test_pca_dim_from_cli_vector_contract():
    assert pca_dim_from_cli_vector(None, 0, 10) == 4
    assert pca_dim_from_cli_vector([], 0, 3) == 3
    assert pca_dim_from_cli_vector([2, 5], 0, 10) == 2
    assert pca_dim_from_cli_vector([2, 5], 3, 10) == 5
    assert pca_dim_from_cli_vector([100], 0, 8) == 8
    assert pca_dim_from_cli_vector([0], 0, 8) == 1
    assert pca_dim_from_cli_vector([-3], 0, 8) == 1
    with pytest.raises(ValueError):
        pca_dim_from_cli_vector([2], -1, 8)
    with pytest.raises(ValueError):
        pca_dim_from_cli_vector([2], 0, 0)


def test_compute_fixed_pca_basis_shapes_and_nograd():
    ref = torch.randn(2, 4, 5, dtype=torch.float32)
    basis, centroid = compute_fixed_pca_basis(ref, 2)
    assert basis.shape == (4, 2)
    assert centroid.shape == (4,)
    assert basis.requires_grad is False
    assert centroid.requires_grad is False

    b2, _ = compute_fixed_pca_basis(ref, 100)
    assert b2.shape[1] <= 4

    with pytest.raises(ValueError):
        compute_fixed_pca_basis(torch.randn(2, 4), 2)


def test_apply_fixed_pca_basis_projection_and_grad():
    signal = torch.randn(2, 4, 5, dtype=torch.float32, requires_grad=True)
    basis, centroid = compute_fixed_pca_basis(signal.detach(), 2)
    out = apply_fixed_pca_basis(signal, basis, centroid)
    assert out.shape == (2, 2, 5)

    # manual projection
    obs = signal.permute(0, 2, 1).reshape(-1, 4)
    manual = (obs - centroid.view(1, -1)) @ basis
    manual = manual.reshape(2, 5, 2).permute(0, 2, 1).contiguous()
    assert torch.allclose(out, manual, atol=1e-5)

    out.sum().backward()
    assert signal.grad is not None
    assert basis.grad is None
    assert centroid.grad is None

    with pytest.raises(ValueError):
        apply_fixed_pca_basis(torch.randn(2, 4), basis, centroid)
    with pytest.raises(ValueError):
        apply_fixed_pca_basis(torch.randn(2, 5, 4), basis, centroid)


def test_auto_spectral_matrix_from_mode_maps_contract():
    x = torch.randn(2, 3, 8, dtype=torch.float32)
    freqs, m = auto_spectral_matrix_from_mode_maps(x)
    assert m.shape == (5, 3, 3)
    assert freqs.shape[0] == 5
    for f in range(m.shape[0]):
        assert torch.allclose(m[f], m[f].conj().transpose(0, 1), atol=1e-5)
    diag = torch.diagonal(m, dim1=1, dim2=2)
    assert torch.isfinite(diag.real).all()
    assert torch.all(diag.real >= -1e-6)
    assert torch.allclose(diag.imag, torch.zeros_like(diag.imag), atol=1e-5)

    z = torch.zeros(2, 3, 8)
    _fz, mz = auto_spectral_matrix_from_mode_maps(z)
    assert torch.allclose(mz, torch.zeros_like(mz), atol=1e-7)

    with pytest.raises(ValueError):
        auto_spectral_matrix_from_mode_maps(torch.randn(2, 3))
    with pytest.raises(ValueError):
        auto_spectral_matrix_from_mode_maps(torch.randn(2, 0, 8))
    with pytest.raises(ValueError):
        auto_spectral_matrix_from_mode_maps(torch.randn(2, 3, 1))
