from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.psd_analysis as pa


def _base_args(parser, *extra):
    return parser.parse_args([
        '--checkpoint', 'a.pt', '--dataset', 'd', '--prep_root', '/p', '--output_root', '/o', '--anal_batch', '1', '--gpu_index', '0', *extra
    ])


def test_bool_like_and_defaults_parser():
    parser = pa.build_arg_parser()
    args = _base_args(parser, '--enable_pca_1d', 'yes', '--enable_pca_mimo', 'off')
    assert pa._parse_bool_like(args.enable_pca_1d, default=False) is True
    assert pa._parse_bool_like(args.enable_pca_mimo, default=True) is False


def test_validate_pca_args_range_and_dim():
    parser = pa.build_arg_parser()
    args = _base_args(parser, '--pca_min_train_accuracy', '0.5', '--pca_dim_per_layer', '2', '4')
    assert pa._validate_pca_args(args, parser) == [2, 4]
    args_bad = _base_args(parser, '--pca_min_train_accuracy', '1.2')
    with pytest.raises(SystemExit):
        pa._validate_pca_args(args_bad, parser)


def test_main_requires_ref_epoch_when_pca_enabled(monkeypatch, tmp_path: Path):
    cfg = tmp_path / 'cfg.json'
    cfg.write_text('{"psd_analysis": {"checkpoint": "x", "dataset": "d", "prep_root": "/p", "output_root": "/o", "anal_batch": 1, "gpu_index": 0, "enable_pca_1d": true, "enable_pca_mimo": false, "pca_ref_epoch": null}}', encoding='utf-8')
    monkeypatch.setattr(pa, '_load_runtime_dependencies', lambda: None)
    with pytest.raises(ValueError, match='pca_ref_epoch must be provided'):
        pa.main(['--config', str(cfg)])


def test_main_ref_accuracy_gate_unavailable(monkeypatch, tmp_path: Path):
    cfg = tmp_path / 'cfg.json'
    cfg.write_text('{"psd_analysis": {"checkpoint": "x", "dataset": "d", "prep_root": "/p", "output_root": "/o", "anal_batch": 1, "gpu_index": 0, "enable_pca_1d": true, "enable_pca_mimo": false, "pca_ref_epoch": 1, "pca_min_train_accuracy": 0.1}}', encoding='utf-8')
    monkeypatch.setattr(pa, '_load_runtime_dependencies', lambda: None)
    monkeypatch.setattr(pa, '_resolve_checkpoint_files', lambda _p: ([Path('c1.pt')], []))
    monkeypatch.setattr(pa, '_load_checkpoint', lambda p, map_location='cpu': {'epoch': 1})
    monkeypatch.setattr(pa, '_require_cuda_device', lambda _i: 'cpu')
    monkeypatch.setattr(pa, '_seed_everything', lambda _s: None)
    monkeypatch.setattr(pa, '_build_model_from_checkpoint', lambda payload, device: (_DummyModel(), None, _DummySpec(), 'none'))
    monkeypatch.setattr(pa, '_resolve_bundle', lambda payload, cli_dataset, cli_prep_root, model_spec=None: _DummyBundle())
    monkeypatch.setattr(pa, '_manifest_dict', lambda _p: {'psd_time_axis': 'last', 'psd_row_axes': ['row'], 'psd_flatten_rule': 'rows', 'psd_logical_shape': {'T': 8}})
    monkeypatch.setattr(pa, '_validate_axis_metadata', lambda manifest, payload: None)
    monkeypatch.setattr(pa, '_collect_signal_maps', lambda **kwargs: {('L1', 1, 'hidden', 'spike', 'train', 'lif', None): torch.ones(2, 4, 8)})
    monkeypatch.setattr(pa, 'tqdm', lambda iterable, **kwargs: iterable, raising=False)
    monkeypatch.setattr(pa, 'dataset_for_view', lambda ds, view: ds, raising=False)
    with pytest.raises(ValueError, match='pca_min_train_accuracy > 0 requires'):
        pa.main(['--config', str(cfg)])


class _DummyModel:
    def iter_named_layers(self):
        return []


class _DummySpec:
    canonical_token = 'lif_soft_fixed'
    family = 'lif'


class _DummyBundle:
    dataset_name = 'd'
    train_dataset = object()
    test_dataset = object()
    training_view_name = 'train'
    manifest_path = Path('/tmp/x.json')
    psd_axis_kind = 'temporal'
