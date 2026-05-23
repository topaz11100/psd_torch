from psd_snn.analysis.probe.builder import build_probe_indices

def test_label_set_and_label_single():
    labels = [0,0,1,1,2,2]
    m = build_probe_indices(labels, 'label_set', 2, 1, target_labels=[1,2])
    assert set(m.class_counts).issubset({1,2})
    s = build_probe_indices(labels, 'label_single', 6, 1)
    assert all(v <= 1 for v in s.class_counts.values())

def test_distributed_quota_sum():
    labels = [0]*7 + [1]*3
    m = build_probe_indices(labels, 'distributed_set', 5, 0)
    assert sum(m.quotas.values()) == 5

def test_no_same_label_token():
    import pathlib
    txt = pathlib.Path('src/psd_snn/analysis/probe/builder.py').read_text()
    assert 'same_label' not in txt
