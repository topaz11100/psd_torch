from psd_snn.analysis.spectral.accumulator import PSDAccumulator
from psd_snn.analysis.spectral.distance import distance_between
import pytest

def test_exact_len_and_raw_state_then_db_finalize():
    acc = PSDAccumulator(axis_policy='exact')
    f=[0,1,2]
    acc.update(f, [[1,10,100],[3,30,300]])
    assert acc._sum[0] == 4.0
    out = acc.finalize(to_db=False)
    assert len(out['freq']) == 3 and out['power'][0] == 2.0
    out_db = acc.finalize(to_db=True)
    assert out_db['power'][0] != out['power'][0]

def test_userbin_mean_median_empty_and_allow_empty():
    f=[0,1,2,3]
    a = PSDAccumulator(axis_policy='userbin', userbin_edges=[0,2,4], userbin_reducer='mean')
    a.update(f, [[1,3,5,7]])
    assert a.finalize()['power'] == [2.0,6.0]
    b = PSDAccumulator(axis_policy='userbin', userbin_edges=[0,2,4], userbin_reducer='median')
    b.update(f, [[1,9,5,7]])
    assert b.finalize()['power'] == [5.0,6.0]
    c = PSDAccumulator(axis_policy='userbin', userbin_edges=[10,11]); c.update(f, [[1,2,3,4]])
    with pytest.raises(ValueError): c.finalize()
    d = PSDAccumulator(axis_policy='userbin', userbin_edges=[10,11], allow_empty_bins=True, empty_bin_fill='zero'); d.update(f, [[1,2,3,4]])
    assert d.finalize()['power'] == [0.0]

def test_distance_metrics_and_axis_mismatch():
    ex1={'axis_policy':'exact','power':[2,3,4]}; ex2={'axis_policy':'exact','power':[12,13,14]}
    assert distance_between(ex1, ex2, 'centered_l2') == 0.0
    assert distance_between({'axis_policy':'exact','power':[1,2,4]}, {'axis_policy':'exact','power':[1,3,6]}, 'diff_l2') > 0
    with pytest.raises(ValueError): distance_between({'axis_policy':'exact','power':[1]}, {'axis_policy':'userbin','power':[1]}, 'centered_l2')
