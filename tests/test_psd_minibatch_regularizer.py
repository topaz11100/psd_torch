import pytest

torch = pytest.importorskip('torch')

from src.model.snn_builder import LayerRecord
from src.model.psd_minibatch_regularizer import compute_fixed_pca_reference_bank, compute_minibatch_psd_regularizer


def _mk_records():
    b,t,c = 4,8,3
    inp = torch.randn(b,t,c)
    h1s = torch.randn(b,t,c)
    h1m = torch.randn(b,t,c)
    h2s = torch.randn(b,t,c)
    h2m = torch.randn(b,t,c)
    recs=[LayerRecord('l1',h1m,h1s), LayerRecord('l2',h2m,h2s)]
    return inp,recs


def test_zero_lambda_returns_zero_safe():
    inp,recs=_mk_records()
    out=compute_minibatch_psd_regularizer(inp,recs,'raw','spike',0.0,0.0,0.0,None)
    assert float(out.total)==0.0


def test_pca_lambda_without_bank_raises():
    inp,recs=_mk_records()
    with pytest.raises(ValueError):
        compute_minibatch_psd_regularizer(inp,recs,'raw','spike',0.0,0.1,0.0,None)


def test_fixed_basis_no_grad_and_finite_losses():
    inp,recs=_mk_records()
    bank=compute_fixed_pca_reference_bank(inp,recs,'spike',[2])
    for v in bank.values():
        assert not v.x_basis.requires_grad and not v.y_basis.requires_grad
    out=compute_minibatch_psd_regularizer(inp,recs,'centered','spike',0.1,0.2,0.3,bank)
    assert torch.isfinite(out.total)
    assert torch.isfinite(out.rep_1d)
    assert torch.isfinite(out.pca_1d)
    assert torch.isfinite(out.pca_mimo)
