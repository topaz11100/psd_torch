from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.psd_analysis as pa


def test_psd_analysis_parser_accepts_pca_options():
    parser = pa.build_arg_parser()
    args = parser.parse_args([
        '--checkpoint', 'a.pt',
        '--dataset', 's-mnist',
        '--prep_root', '/tmp/prep',
        '--output_root', '/tmp/out',
        '--anal_batch', '8',
        '--gpu_index', '0',
        '--enable_pca_1d', 'true',
        '--enable_pca_mimo', 'false',
        '--pca_ref_epoch', '3',
        '--pca_min_train_accuracy', '0.5',
        '--pca_dim_per_layer', '8', '4',
    ])
    assert args.enable_pca_1d == 'true'
    assert args.enable_pca_mimo == 'false'
    assert args.pca_ref_epoch == 3
    assert args.pca_min_train_accuracy == 0.5
    assert args.pca_dim_per_layer == ['8', '4']


def test_validate_pca_dim_vector_positive_only():
    parser = pa.build_arg_parser()
    args = parser.parse_args([
        '--checkpoint', 'a.pt', '--dataset', 'd', '--prep_root', '/p', '--output_root', '/o', '--anal_batch', '1', '--gpu_index', '0',
        '--pca_dim_per_layer', '4', '2',
    ])
    assert pa._validate_pca_args(args, parser) == [4, 2]
    args_bad = parser.parse_args([
        '--checkpoint', 'a.pt', '--dataset', 'd', '--prep_root', '/p', '--output_root', '/o', '--anal_batch', '1', '--gpu_index', '0',
        '--pca_dim_per_layer', '0',
    ])
    with pytest.raises(ValueError):
        pa._validate_pca_args(args_bad, parser)


def test_pca_ref_epoch_missing_raises_value_error(monkeypatch, tmp_path: Path):
    pytest.importorskip('torch')
    cfg = tmp_path / 'cfg.yaml'
    cfg.write_text('{"psd_analysis": {"checkpoint": "x", "dataset": "d", "prep_root": "/p", "output_root": "/o", "anal_batch": 1, "gpu_index": 0, "pca_ref_epoch": 9}}', encoding='utf-8')
    monkeypatch.setattr(pa, '_load_runtime_dependencies', lambda: None)
    monkeypatch.setattr(pa, '_resolve_checkpoint_files', lambda _p: ([Path('c1.pt'), Path('c2.pt')], []))
    monkeypatch.setattr(pa, '_load_checkpoint', lambda p, map_location='cpu': {'epoch': 1 if str(p) == 'c1.pt' else 2})
    monkeypatch.setattr(pa, '_require_cuda_device', lambda _i: 'cpu')
    with pytest.raises(ValueError, match='pca_ref_epoch=9'):
        pa.main(['--config', str(cfg)])
