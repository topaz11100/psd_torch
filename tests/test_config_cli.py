import json
from pathlib import Path
import sys

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.util.config_cli import load_config_dict, parse_args_with_config


def _make_parser():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='default_ds')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--config', default=None)
    return p


def test_config_default_and_cli_override(tmp_path: Path):
    cfg = tmp_path / 'a.json'
    cfg.write_text(json.dumps({'dataset_psd': {'dataset': 'mnist', 'batch_size': 128}}), encoding='utf-8')
    parser = _make_parser()
    args = parse_args_with_config(parser, argv=['--config', str(cfg), '--batch_size', '64'], stage_key='dataset_psd')
    assert args.dataset == 'mnist'
    assert args.batch_size == 64


def test_unknown_key_raises(tmp_path: Path):
    cfg = tmp_path / 'bad.json'
    cfg.write_text(json.dumps({'dataset': 'mnist', 'unknown_key': 1}), encoding='utf-8')
    parser = _make_parser()
    with pytest.raises(ValueError):
        parse_args_with_config(parser, argv=['--config', str(cfg)])


def test_yaml_blocked(tmp_path: Path):
    cfg = tmp_path / 'bad.yaml'
    cfg.write_text('a: 1\n', encoding='utf-8')
    with pytest.raises(ValueError):
        load_config_dict(str(cfg))


def test_all_config_json_valid():
    for path in sorted(Path('config').glob('*.json')):
        payload = json.loads(path.read_text(encoding='utf-8'))
        assert isinstance(payload, dict)


def test_bash_wrappers_reference_config():
    script_targets = {
        'data_prep': 'src/data_prep.py',
        'dataset_psd': 'src/dataset_psd.py',
        'dataset_fft': 'src/dataset_fft.py',
        'model_training': 'src/model_training.py',
        'psd_analysis': 'src/psd_analysis.py',
        'element_psd': 'src/element_psd.py',
        'fft2d_analysis': 'src/2d_fft_analysis.py',
        'plotting': 'src/plotting.py',
    }
    for name, target in script_targets.items():
        script = Path('bash') / f'{name}.sh'
        assert script.exists()
        text = script.read_text(encoding='utf-8')
        assert f'config/{name}.json' in text
        assert '--config "$CONFIG_PATH"' in text
        assert target in text
        assert Path(target).exists()


def test_no_bbalanced_typo():
    text = Path('src/dataset_fft.py').read_text(encoding='utf-8')
    assert 'bbalanced' not in text
    assert 'balanced_global' in text


def test_model_training_config_only_parses_required(tmp_path: Path):
    from src.model_training import build_arg_parser

    cfg = tmp_path / 'train.json'
    cfg.write_text(json.dumps({'model_training': {
        'dataset': 'mnist', 'prep_root': '/ABS/PATH/TO/prepared', 'model': 'mlp_lif',
        'hidden_spec': '128,128', 'readout_mode': 'temporal_membrane', 'epochs': 1,
        'batch_size': 8, 'lr': 0.001, 'seed': 1, 'checkpoint_root': '/ABS/PATH/TO/ckpt',
        'metric_root': '/ABS/PATH/TO/metric'
    }}), encoding='utf-8')
    parser = build_arg_parser()
    args = parse_args_with_config(parser, argv=['--config', str(cfg)], stage_key='model_training')
    assert args.dataset == 'mnist'
