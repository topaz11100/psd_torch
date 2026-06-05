from pathlib import Path

from src.util.config import save_yaml
import csv
import json
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.psd_analysis as pa


def _write_cfg(path: Path, out: Path, pca_1d=True, pca_mimo=True):
    payload = {
        'psd_analysis': {
            'checkpoint': 'x', 'dataset': 'd', 'prep_root': '/p', 'output_root': str(out), 'anal_batch': 1, 'gpu_index': 0,
            'enable_pca_1d': pca_1d, 'enable_pca_mimo': pca_mimo, 'pca_ref_epoch': 1, 'pca_min_train_accuracy': 0.0,
            'run_timestamp': 'TEST_PSD_ANALYSIS',
        }
    }
    save_yaml(path, payload)


class _DummyModel:
    def iter_named_layers(self):
        return [('hidden1', object()), ('hidden2', object())]


class _DummySpec:
    canonical_token = 'lif_soft_fixed'
    family = 'lif'


class _DummyBundle:
    dataset_name = 'd'
    train_dataset = object()
    test_dataset = object()
    training_view_name = 'train'
    manifest_path = Path('/tmp/x.yaml')
    psd_axis_kind = 'temporal'


def _patch_common(monkeypatch):
    monkeypatch.setattr(pa, '_load_runtime_dependencies', lambda: None)
    monkeypatch.setattr(pa, '_resolve_checkpoint_files', lambda _p: ([Path('c1.pt')], []))
    monkeypatch.setattr(pa, '_load_checkpoint', lambda p, map_location='cpu': {'epoch': 1, 'metric_snapshot': {'train_accuracy': 0.9}})
    monkeypatch.setattr(pa, '_require_cuda_device', lambda _i: 'cpu')
    monkeypatch.setattr(pa, '_seed_everything', lambda _s: None)
    monkeypatch.setattr(pa, '_build_model_from_checkpoint', lambda payload, device: (_DummyModel(), None, _DummySpec(), 'none'))
    monkeypatch.setattr(pa, '_resolve_bundle', lambda payload, cli_dataset, cli_prep_root, model_spec=None: _DummyBundle())
    monkeypatch.setattr(pa, '_manifest_dict', lambda _p: {'psd_time_axis': 'last', 'psd_row_axes': ['row'], 'psd_flatten_rule': 'rows', 'psd_logical_shape': {'T': 8}})
    monkeypatch.setattr(pa, '_validate_axis_metadata', lambda manifest, payload: None)
    monkeypatch.setattr(pa, 'dataset_for_view', lambda ds, view: ds, raising=False)
    monkeypatch.setattr(pa, 'tqdm', lambda iterable, **kwargs: iterable, raising=False)
    from src.signal.psd_utils import pca_dim_from_cli_vector, compute_fixed_pca_basis, apply_fixed_pca_basis, auto_spectral_matrix_from_mode_maps, cross_spectral_matrix_from_mode_maps
    monkeypatch.setattr(pa, 'pca_dim_from_cli_vector', pca_dim_from_cli_vector, raising=False)
    monkeypatch.setattr(pa, 'compute_fixed_pca_basis', compute_fixed_pca_basis, raising=False)
    monkeypatch.setattr(pa, 'apply_fixed_pca_basis', apply_fixed_pca_basis, raising=False)
    monkeypatch.setattr(pa, 'auto_spectral_matrix_from_mode_maps', auto_spectral_matrix_from_mode_maps, raising=False)
    monkeypatch.setattr(pa, 'cross_spectral_matrix_from_mode_maps', cross_spectral_matrix_from_mode_maps, raising=False)
    monkeypatch.setattr(pa, 'torch', torch, raising=False)

    maps = {
        ('hidden1', 1, 'hidden', 'layer_input', 'train', 'lif', None): torch.ones(2, 4, 8),
        ('hidden1', 1, 'hidden', 'membrane', 'train', 'lif', None): torch.ones(2, 5, 8),
        ('hidden1', 1, 'hidden', 'spike', 'train', 'lif', None): torch.ones(2, 6, 8),
        ('hidden2', 2, 'hidden', 'spike', 'train', 'lif', None): torch.ones(2, 3, 8),
    }
    monkeypatch.setattr(pa, '_collect_signal_maps', lambda **kwargs: maps)

    def _summary(maps, **kwargs):
        return {'freq': torch.linspace(0, 0.5, 5).numpy(), 'representative': {'raw': {'mean': {'psd_exact': torch.ones(5).numpy()}}}}

    monkeypatch.setattr(pa, 'compute_family_spectral_summary', _summary, raising=False)
    monkeypatch.setattr(pa, 'representative_curve_from_summary', lambda *a, **k: torch.ones(5).numpy(), raising=False)
    monkeypatch.setattr(pa, 'curve_axis_from_summary', lambda *a, **k: torch.linspace(0, 0.5, 5).numpy(), raising=False)
    monkeypatch.setattr(pa, '_summary_curve_rows', lambda **k: ([], []))
    monkeypatch.setattr(pa, '_layer_distance_rows_for_checkpoint', lambda **k: ([], []))
    monkeypatch.setattr(pa, '_layer_dispersion_rows_for_checkpoint', lambda **k: ([], []))
    monkeypatch.setattr(pa, '_pair_rows_for_checkpoint', lambda **k: ([], []))
    monkeypatch.setattr(pa, '_filter_snapshot_rows', lambda **k: ([], {}))


def _read_csv_rows(path: Path):
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def test_pca_reference_schema_and_basis_payload(monkeypatch, tmp_path: Path):
    _patch_common(monkeypatch)
    cfg = tmp_path / 'cfg.yaml'
    out = tmp_path / 'out'
    _write_cfg(cfg, out, pca_1d=True, pca_mimo=True)
    rc = pa.main(['--config', str(cfg)])
    assert rc == 0

    result_root = out / 'psd_analysis_TEST_PSD_ANALYSIS'
    basis_dir = result_root / 'pca_reference' / 'basis'
    basis_files = list(basis_dir.glob('*.pt'))
    assert basis_files
    payload = torch.load(basis_files[0], map_location='cpu')
    assert payload['basis'].device.type == 'cpu'
    assert payload['centroid'].device.type == 'cpu'
    assert payload['basis'].requires_grad is False
    assert payload['centroid'].requires_grad is False

    all_rows = []
    for csv_path in (result_root / 'checkpoint_epoch_000001').rglob('*.csv'):
        all_rows.extend(_read_csv_rows(csv_path))
    cross_rows = [r for r in all_rows if str(r.get('series', '')).startswith('pca_cross_')]
    assert cross_rows
    series_values = {r.get('series', '') for r in cross_rows}
    assert any('layer_input_to_membrane' in s for s in series_values)
    assert any('layer_input_to_spike' in s for s in series_values)
    assert any('adjacent_hidden_output' in s for s in series_values)
    row = cross_rows[0]
    assert row.get('pca_analysis_schema_version') == '1'


def test_pca_disabled_skips_pca_artifacts(monkeypatch, tmp_path: Path):
    _patch_common(monkeypatch)
    cfg = tmp_path / 'cfg.yaml'
    out = tmp_path / 'out'
    _write_cfg(cfg, out, pca_1d=False, pca_mimo=False)
    rc = pa.main(['--config', str(cfg)])
    assert rc == 0
    ckpt_dir = out / 'psd_analysis_TEST_PSD_ANALYSIS' / 'checkpoint_epoch_000001'
    assert not (ckpt_dir / 'pca_mode_traces').exists()
    assert not (ckpt_dir / 'pca_mimo_traces').exists()
    assert not (ckpt_dir / 'pca_cross_traces').exists()
