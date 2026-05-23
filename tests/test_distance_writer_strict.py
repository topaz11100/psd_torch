from psd_snn.artifacts.distance_writer import build_distance_row


def mk_curve(axis='exact', rep='mean', pca_basis_id=None):
    return {
        'type':'spectral_curve',
        'power':[1.0,2.0,3.0],
        'identity':{
            'artifact_type':'spectral_curve','spectral_axis':axis,'scale':'raw','centering':'raw','window':'none',
            'representative':rep,'analysis_method':'psd','pca_basis_id':pca_basis_id,
            'userbin_axes':None,'userbin_reducer':None,'row_axis_semantics':None,'fftshift_row':None,
            'row_bin_edges':None,'column_bin_edges':None,
        },
        'meta':{'run_id':'r','checkpoint_epoch':1,'split':'test','scope':'s','probe_family':'label_single','layer_name':'l','series':'spike'}
    }


def test_exact_userbin_mismatch_error():
    a=mk_curve('exact'); b=mk_curve('userbin')
    try:
        build_distance_row(a,b,'centered_l2')
        assert False
    except ValueError as e:
        assert 'exact/userbin mismatch' in str(e)


def test_pca_basis_required_and_match():
    a=mk_curve(rep='pca', pca_basis_id='x')
    b=mk_curve(rep='pca', pca_basis_id='x')
    r=build_distance_row(a,b,'centered_l2')
    assert r['status']=='ok'
    c=mk_curve(rep='pca', pca_basis_id='y')
    try:
        build_distance_row(a,c,'centered_l2')
        assert False
    except ValueError as e:
        assert 'different pca_basis_id' in str(e)


def test_invalid_metric_error():
    a=mk_curve(); b=mk_curve()
    try:
        build_distance_row(a,b,'cosine')
        assert False
    except ValueError:
        pass
