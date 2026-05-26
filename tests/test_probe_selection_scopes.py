import importlib.util

import pytest

HAS_TORCH = importlib.util.find_spec('torch') is not None

if HAS_TORCH:
    from src.stat.probe_selection import build_probe_index_bundle, build_probe_scopes
    from src.analysis_matrix_common import iter_matrix_probe_scopes

    class DummyDS:
        def __init__(self):
            self.targets = [i % 3 for i in range(30)]
        def __len__(self):
            return len(self.targets)
        def __getitem__(self, i):
            return [0.0], int(self.targets[i])


def test_probe_selection_collection_smoke():
    if not HAS_TORCH:
        assert True


@pytest.mark.skipif(not HAS_TORCH, reason='torch not installed')
def test_build_probe_scopes_contains_three_families():
    ds = DummyDS()
    b = build_probe_index_bundle(ds, split_name='train', seed=1, same_label_n_per_label=3, balanced_global_n_per_label=3, distribution_global_min_class_n=3)
    scopes = build_probe_scopes(ds, split_name='train', bundle=b)
    fam = {s.probe_family for s in scopes}
    assert {'same_label', 'balanced_global', 'distribution_global'}.issubset(fam)


@pytest.mark.skipif(not HAS_TORCH, reason='torch not installed')
def test_matrix_scopes_include_official_families():
    scopes = iter_matrix_probe_scopes(DummyDS(), split_name='test', seed=3)
    fam = {s.probe_family for s in scopes}
    assert {'same_label', 'balanced_global', 'distribution_global'}.issubset(fam)
