import pytest

torch = pytest.importorskip('torch')
from psd_snn.analysis.signal.fft2d import fft2d_exact, fft2d_userbin


def test_fft2d_exact_shape():
    x=torch.randn(2,8,16)
    out=fft2d_exact(x)
    assert out.matrix.shape == (8,9)
    assert len(out.row_axis)==8 and len(out.col_axis)==9

def test_fft2d_userbin_axes():
    x=torch.randn(2,8,16)
    ex=fft2d_exact(x)
    t=fft2d_userbin(ex,'time_frequency',time_edges=[0,0.2,0.5],row_axis_semantics='unordered')
    assert t.matrix.shape[1]==2
    with pytest.raises(ValueError):
        fft2d_userbin(ex,'row_frequency',row_edges=[-0.5,0,0.5],row_axis_semantics='unordered')
    r=fft2d_userbin(ex,'row_frequency',row_edges=[-0.5,0,0.5],row_axis_semantics='group_ordered')
    assert r.matrix.shape[0]==2
