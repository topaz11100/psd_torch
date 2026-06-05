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
    out=compute_minibatch_psd_regularizer(inp,recs,'raw','spike')
    assert float(out.total)==0.0


def test_pca_lambda_without_bank_raises():
    inp,recs=_mk_records()
    with pytest.raises(ValueError):
        compute_minibatch_psd_regularizer(inp,recs,'raw','spike',lambda_pca_input=0.1)


def test_fixed_basis_no_grad_and_finite_losses():
    inp,recs=_mk_records()
    bank={'input': compute_fixed_pca_reference_bank(inp,recs,'spike',[2], relation='input')}
    for v in bank['input'].values():
        assert not v.x_basis.requires_grad and not v.y_basis.requires_grad
    out=compute_minibatch_psd_regularizer(
        inp,recs,'centered','spike',
        lambda_rep_input=0.1,
        lambda_pca_input=0.2,
        pca_reference_banks=bank,
    )
    assert torch.isfinite(out.total)
    assert torch.isfinite(out.rep_1d)
    assert torch.isfinite(out.pca_1d)
    assert torch.isfinite(out.pca_mimo)


def test_rep_input_and_adjacent_can_be_enabled_together():
    inp, recs = _mk_records()
    out_input = compute_minibatch_psd_regularizer(
        inp, recs, 'centered', 'spike',
        lambda_rep_input=0.1,
        lambda_rep_adjacent=0.0,
    )
    out_adjacent = compute_minibatch_psd_regularizer(
        inp, recs, 'centered', 'spike',
        lambda_rep_input=0.0,
        lambda_rep_adjacent=0.2,
    )
    out_both = compute_minibatch_psd_regularizer(
        inp, recs, 'centered', 'spike',
        lambda_rep_input=0.1,
        lambda_rep_adjacent=0.2,
    )
    assert torch.allclose(out_both.total, out_input.total + out_adjacent.total, atol=1e-5, rtol=1e-5)
    assert torch.allclose(out_both.rep_input, out_input.rep_1d, atol=1e-5, rtol=1e-5)
    assert torch.allclose(out_both.rep_adjacent, out_adjacent.rep_1d, atol=1e-5, rtol=1e-5)


def test_pca_dim_controls_scalar_vs_mimo_mode():
    inp, recs = _mk_records()
    bank_1d = {'input': compute_fixed_pca_reference_bank(inp, recs, 'spike', [1], relation='input')}
    out_1d = compute_minibatch_psd_regularizer(
        inp, recs, 'centered', 'spike',
        lambda_pca_input=0.2,
        pca_reference_banks=bank_1d,
    )
    assert float(out_1d.pca_mimo.detach()) == 0.0
    assert out_1d.metadata['relation_metadata']['input']['pca_mode_by_layer'] == {'l1': '1d', 'l2': '1d'}

    bank_mimo = {'input': compute_fixed_pca_reference_bank(inp, recs, 'spike', [2], relation='input')}
    out_mimo = compute_minibatch_psd_regularizer(
        inp, recs, 'centered', 'spike',
        lambda_pca_input=0.2,
        pca_reference_banks=bank_mimo,
    )
    assert float(out_mimo.pca_1d.detach()) == 0.0
    assert out_mimo.metadata['relation_metadata']['input']['pca_mode_by_layer'] == {'l1': 'mimo', 'l2': 'mimo'}


def test_userbin_curve_space_is_available_for_rep_and_pca1():
    inp, recs = _mk_records()
    bank = {'input': compute_fixed_pca_reference_bank(inp, recs, 'spike', [1], variant='centered', relation='input')}
    out = compute_minibatch_psd_regularizer(
        inp,
        recs,
        'centered',
        'spike',
        lambda_rep_input=0.1,
        lambda_pca_input=0.2,
        pca_reference_banks=bank,
        curve_space='userbin',
        userbin_edges=[0.0, 0.25, 0.5],
        userbin_reducer='mean',
        curve_scale='db',
    )
    assert torch.isfinite(out.total)
    assert out.metadata['curve_space'] == 'userbin'
    assert out.metadata['curve_scale'] == 'db'
