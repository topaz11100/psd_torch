import pytest

torch = pytest.importorskip('torch')
from psd_snn.models.constraints.bounds import bounded_sigmoid, inverse_bounded_sigmoid, validate_bounds

def test_bounded_roundtrip_and_range():
    raw = torch.randn(4, requires_grad=True)
    y = bounded_sigmoid(raw, 0.2, 0.8)
    assert torch.all((y >= 0.2) & (y <= 0.8))
    rec = inverse_bounded_sigmoid(y.detach(), 0.2, 0.8)
    y2 = bounded_sigmoid(rec, 0.2, 0.8)
    assert torch.allclose(y.detach(), y2, atol=1e-5)
    y.sum().backward(); assert raw.grad is not None

def test_invalid_bounds():
    with pytest.raises(ValueError): validate_bounds(1.0, 0.5, name='x')
