import pytest

torch = pytest.importorskip('torch')

from psd_snn.analysis.signal.pca_basis_store import PCABasisStore, PCABasisKey, PCAFitRequest, PCAApplyRequest


def _key(**kw):
    base = dict(reference_checkpoint_epoch=1, reference_checkpoint_id='c1', reference_split='test', reference_scope='test_balanced_global', reference_probe_family='balanced_global', layer_index=0, layer_name='layer0', signal_kind='hidden', series='spike', n_components=2, row_count=4, centering=True)
    base.update(kw)
    return PCABasisKey(**base)


def test_fit_apply_save_load_smoke(tmp_path):
    maps = torch.randn(5,4,8)
    store = PCABasisStore(str(tmp_path))
    rec = store.fit(PCAFitRequest(maps=maps, key=_key(), created_from={'run_id':'r'}))
    p = store.save_tensor_artifact(rec)
    assert rec.basis_id
    assert p.endswith('.pt')
    proj, got = store.apply(PCAApplyRequest(maps=maps, key=_key()))
    assert proj.shape == (5,2,8)
    assert got.basis_id == rec.basis_id


def test_missing_and_mismatch_errors():
    maps = torch.randn(2,4,4)
    store = PCABasisStore()
    with pytest.raises(ValueError, match='pca_basis_missing'):
        store.apply(PCAApplyRequest(maps=maps, key=_key()))
    rec = store.fit(PCAFitRequest(maps=maps, key=_key(), created_from={}))
    with pytest.raises(ValueError, match='row_count mismatch'):
        store.apply(PCAApplyRequest(maps=torch.randn(2,3,4), key=_key(row_count=4)))
    with pytest.raises(ValueError, match='n_components > row_count'):
        store.fit(PCAFitRequest(maps=maps, key=_key(n_components=5, row_count=4), created_from={}))
