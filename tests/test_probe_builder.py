from psd_snn.analysis.probe.builder import build_probe_indices
from psd_snn.analysis.common.probe_orchestrator import ProbeRequest, ProbeOrchestrator


def test_label_set_and_label_single():
    labels = [0,0,1,1,2,2]
    m = build_probe_indices(labels, 'label_set', 2, 1, target_labels=[1,2], split='test')
    assert set(m.class_counts).issubset({1,2})
    s = build_probe_indices(labels, 'label_single', 6, 1, split='test')
    assert all(v <= 1 for v in s.class_counts.values())


def test_distributed_quota_sum():
    labels = [0]*7 + [1]*3
    m = build_probe_indices(labels, 'distributed_set', 5, 0, split='test')
    assert sum(m.quotas.values()) == 5


def test_label_single_exclusion_family_balanced_global():
    labels = [0,0,1,1,2,2,0,1,2]
    ex = build_probe_indices(labels, 'balanced_global', 3, 3, split='test')
    cur = build_probe_indices(labels, 'label_single', 9, 1, split='test', exclusion_indices=set(ex.selected_indices), exclusion_family='balanced_global')
    assert set(cur.selected_indices).isdisjoint(set(ex.selected_indices))
    assert cur.probe_family == 'label_single'
    assert cur.exclusion_family == 'balanced_global'


def test_probe_orchestrator_manifest_and_batches():
    dataset = {'test_inputs': [[[0.0]]]*9, 'test_labels': [0,0,1,1,2,2,0,1,2]}
    orch = ProbeOrchestrator(dataset)
    req = ProbeRequest(split='test', probe_family='label_single', sample_count=9, seed=1, exclusion_family='balanced_global', exclusion_sample_count=3)
    man = orch.build_manifest(req)
    man2 = orch.build_manifest(req)
    assert man.probe_manifest_id == man2.probe_manifest_id
    batches = orch.iter_batches(req, batch_size=2)
    assert batches and batches[0].probe_family == 'label_single'
    assert batches[0].probe_manifest_id == man.probe_manifest_id


def test_no_same_label_token():
    import pathlib
    txt = pathlib.Path('src/psd_snn/analysis/probe/builder.py').read_text()
    assert 'same_label' not in txt
