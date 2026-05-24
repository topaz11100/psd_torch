import json
from pathlib import Path
import sys

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_prep import build_arg_parser, _validate_args
from src.util.config_cli import parse_args_with_config


def test_data_prep_config_only_parses_required(tmp_path: Path):
    cfg = tmp_path / 'prep.json'
    cfg.write_text(json.dumps({'data_prep': {
        'dataset': 'mnist', 'raw_data_root': '/ABS/PATH/TO/raw', 'prep_root': '/ABS/PATH/TO/prep', 'seed': 123,
        'force_overwrite': False, 'download': True, 'max_samples': 10, 'prep_profile': 'project_standard',
        'deap_label_axis': 'valence', 'deap_num_classes': 3, 'shd_dt_ms': 1.0, 'shd_max_time': 1.2, 'ssc_dt_ms': 1.0, 'ssc_max_time': 1.0
    }}), encoding='utf-8')
    parser = build_arg_parser()
    args = parse_args_with_config(parser, argv=['--config', str(cfg)], stage_key='data_prep')
    validated = _validate_args(args)
    assert validated.dataset == 'mnist'
    assert validated.download is True
    assert validated.force_overwrite is False
    assert validated.max_samples == 10


def test_data_prep_option_validation():
    parser = build_arg_parser()
    args = parser.parse_args(['--dataset','mnist','--raw_data_root','/a','--prep_root','/b','--max_samples','1'])
    _validate_args(args)
    args.max_samples = 0
    with pytest.raises(ValueError):
        _validate_args(args)
    args.max_samples = 1
    args.shd_dt_ms = 0.0
    with pytest.raises(ValueError):
        _validate_args(args)


def test_config_readme_mentions_new_options():
    text = Path('config/README.md').read_text(encoding='utf-8')
    for key in ['download', 'max_samples', 'prep_profile', 'deap_label_axis', 'deap_num_classes', 'shd_dt_ms', 'shd_max_time', 'ssc_dt_ms', 'ssc_max_time']:
        assert key in text


def test_no_active_yaml_files():
    roots = [Path('.')]
    found = []
    for path in Path('.').rglob('*'):
        if not path.is_file():
            continue
        if any(part in {'.git', 'old', 'Origin'} for part in path.parts):
            continue
        if path.suffix.lower() in {'.yaml', '.yml'}:
            found.append(str(path))
    assert not found, f'YAML 파일이 남아 있습니다: {found}'
