import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('spikingjelly')

from psd_snn.models.cells.rf_cell import RFCell


def test_rf_matches_matrix_exp_no_spike():
    cell = RFCell(features=1, omega=0.7, damping=0.2, threshold=999.0, reset_mode='none')
    x = torch.tensor([[0.3]])
    st = cell.single_step_forward(x)
    omega = cell.omega.detach()[0]
    damping = cell.damping_magnitude.detach()[0]
    A = torch.tensor([[-damping, -omega],[omega, -damping]], dtype=x.dtype)
    expA = torch.linalg.matrix_exp(A)
    z0 = torch.tensor([0.0,0.0], dtype=x.dtype)
    b = torch.tensor([x[0,0],0.0], dtype=x.dtype)
    z1 = expA @ z0 + b
    assert torch.allclose(st.rf_real_pre[0,0], z1[0], atol=1e-5)


def test_rf_reset_modes_and_shapes():
    b,t,f = 2,4,3
    u = torch.ones(b,t,f)
    for mode in ['threshold_only','hard_state','hard_real','soft_real','scale_state','none']:
        c = RFCell(features=f, threshold=0.1, reset_mode=mode)
        y,tr = c.forward_sequence(u, capture_trace=True)
        assert y.shape == (b,t,f)
        assert tr['rf_real_pre'].shape == (b,t,f)
        assert tr['rf_imag_post'].shape == (b,t,f)
    c = RFCell(features=1, damping=0.5)
    assert torch.all(c.decay_radius < 1.0)
    assert torch.all(c.omega > 0)
