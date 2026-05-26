from pathlib import Path
import json
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.util.config_cli import parse_args_with_config
import src.model_training as mt
import src.psd_analysis as pa
import src.data_prep as dp
import src.dataset_psd as dpsd
import src.element_psd as epsd
import src.dataset_fft as dfft
import importlib

def test_readme_mentions_no_list_sweep_other_stages():
    text = Path('config/README.md').read_text(encoding='utf-8')
    assert '다른 stage에서는 dataset list를 허용하지 않는다' in text


def test_example_configs_parse_with_stage_parsers():
    model_args = parse_args_with_config(mt.build_arg_parser(), argv=['--config', 'config/model_training.json'], stage_key='model_training')
    assert model_args.psd_reg_variant in ('raw', 'centered')
    assert model_args.psd_reg_output_family in ('spike', 'membrane')
    psd_args = parse_args_with_config(pa.build_arg_parser(), argv=['--config', 'config/psd_analysis.json'], stage_key='psd_analysis')
    assert (psd_args.pca_ref_epoch is None) or (int(psd_args.pca_ref_epoch) >= 1)


def test_readme_choices_match_parser_choices():
    text = Path('config/README.md').read_text(encoding='utf-8')
    assert 'psd_reg_variant' in text and '`raw`, `centered`' in text
    assert 'psd_reg_output_family' in text and '`spike`, `membrane`' in text
    assert 'analysis_distance_metric' in text and '`centered_l2`, `diff_l2`' in text


def test_all_configs_static_schema_rules():
    cfgs = sorted(Path('config').rglob('*.json'))
    assert cfgs
    training_cfgs = []
    for p in cfgs:
        data = json.loads(p.read_text(encoding='utf-8'))
        node = data
        if isinstance(data, dict) and len(data) == 1 and isinstance(next(iter(data.values())), dict):
            node = next(iter(data.values()))
        assert isinstance(node, dict)
        if 'prep_root' in node:
            assert node['prep_root'] == '/home/yongokhan/바탕화면/prep_data'
        txt = p.read_text(encoding='utf-8')
        assert '"constraint_mode"' not in txt
        assert '"band_neuron_ends"' not in txt
        assert '"max_rate"' not in txt
        assert 'lambda_psd_pca_1d' not in txt
        assert 'lambda_psd_pca_mimo' not in txt
        assert not re.search(r'"anal_epoch_list"\s*:\s*\[\s*\n', txt)
        if p.name in {'model_training.json', 'model_training_ddp.json'} or 'ddp_train_scenario' in str(p):
            training_cfgs.append((p, node))
    assert training_cfgs
    for p, node in training_cfgs:
        assert 'scenario_mode' in node, p
        assert node.get('readout_mode') in {'temporal_membrane', 'final_membrane', 'first_spike', 'max_fire', 'spikegru_max_over_time'}, p
        assert 'lambda_psd_pca' in node, p
        assert node.get('compile_model') is True, p
        assert 'tear' in node and int(node['tear']) > 0, p
        assert 'band_edge' in node, p


def test_stage_parser_smoke():
    parse_args_with_config(dp.build_arg_parser(), argv=['--config', 'config/data_prep.json'], stage_key='data_prep')
    parse_args_with_config(dpsd.build_arg_parser(), argv=['--config', 'config/dataset_psd.json'], stage_key='dataset_psd')
    parse_args_with_config(pa.build_arg_parser(), argv=['--config', 'config/psd_analysis.json'], stage_key='psd_analysis')
    parse_args_with_config(epsd.build_arg_parser(), argv=['--config', 'config/element_psd.json'], stage_key='element_psd')
    parse_args_with_config(dfft.build_arg_parser(), argv=['--config', 'config/dataset_fft.json'], stage_key='dataset_fft')
    parse_args_with_config(mt.build_arg_parser(), argv=['--config', 'config/model_training.json'], stage_key='model_training')
    parse_args_with_config(mt.build_arg_parser(), argv=['--config', 'config/model_training_ddp.json'], stage_key='model_training')
