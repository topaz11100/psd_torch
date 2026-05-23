from psd_snn.analysis.signal_map.emitter import bt_to_srt, btchw_to_bflat_t

def test_bt_to_srt_shape():
    x = [[[0 for _ in range(4)] for _ in range(3)] for _ in range(2)]
    y = bt_to_srt(x)
    assert len(y) == 2 and len(y[0]) == 4 and len(y[0][0]) == 3

def test_btchw_shape():
    x = [[[[[1]] for _ in range(2)] for _ in range(3)] for _ in range(2)]
    y = btchw_to_bflat_t(x)
    assert len(y) == 2 and len(y[0]) == 2 and len(y[0][0]) == 3
