from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.util.config import load_structured
from src.util.config_cli import parse_args_with_config
import src.model_training as mt
import src.psd_analysis as pa
import src.data_prep as dp
import src.dataset_psd as dpsd
import src.element_psd as epsd
import src.element_fft as efft
import src.dataset_fft as dfft
import importlib

def test_readme_mentions_no_list_sweep_other_stages():
    text = Path('config/README.md').read_text(encoding='utf-8')
    assert '다른 stage에서는 dataset list를 허용하지 않는다' in text


def test_example_configs_parse_with_stage_parsers():
    model_args = parse_args_with_config(mt.build_arg_parser(), argv=['--config', 'config/model_training.yaml'], stage_key='model_training')
    assert model_args.signal_curve_centering in ('raw', 'centered')
    assert model_args.signal_curve_space in ('exact', 'userbin')
    assert model_args.psd_reg_output_family in ('spike', 'membrane')
    psd_args = parse_args_with_config(pa.build_arg_parser(), argv=['--config', 'config/psd_analysis.yaml'], stage_key='psd_analysis')
    assert (psd_args.pca_ref_epoch is None) or (int(psd_args.pca_ref_epoch) >= 1)


def test_readme_choices_match_parser_choices():
    text = Path('config/README.md').read_text(encoding='utf-8')
    assert 'signal_curve_centering' in text and '`raw`, `centered`' in text
    assert 'signal_curve_space' in text and '`exact`, `userbin`' in text
    assert 'psd_reg_output_family' in text and '`spike`, `membrane`' in text
    assert 'analysis_distance_metric' in text and '`centered_l2`, `diff_l2`' in text
    assert 'signal_window' in text and '`hann`, `none`' in text


def test_all_configs_static_schema_rules():
    cfgs = sorted(Path('config').rglob('*.yaml'))
    assert cfgs
    training_cfgs = []
    for p in cfgs:
        data = load_structured(p)
        node = data
        stage_name = None
        if isinstance(data, dict) and len(data) == 1 and isinstance(next(iter(data.values())), dict):
            stage_name = next(iter(data.keys()))
            node = next(iter(data.values()))
        assert isinstance(node, dict)
        if 'prep_root' in node:
            assert node['prep_root'] == '/home/yongokhan/workspace/data/prep_data'
        txt = p.read_text(encoding='utf-8')
        assert 'config: null' not in txt and '"config": null' not in txt
        assert 'constraint_mode' not in txt
        assert 'max_rate' not in txt
        forbidden_duplicate_keys = {
            'anal_epoch_list',
            'regularization_lambda1',
            'regularization_lambda2',
            'regularization_signal',
            'analysis_psd_tokens',
        }
        assert forbidden_duplicate_keys.isdisjoint(node.keys()), p
        forbidden_public_training_keys = {
            'output_root',
            'regularization_psd_curve_tokens',
            'lambda_psd_pca_1d',
            'lambda_psd_pca_mimo',
            'lambda_psd_rep_1d',
            'lambda_psd_pca',
            'psd_reg_relation',
            'rf_frequency_clip_edges',
            'lif_alpha_clip_edges',
            'constraint_tear',
            'band_neuron_ends',
        }
        if 'analysis_checkpoint_epochs' in node:
            assert isinstance(node['analysis_checkpoint_epochs'], list), p
            assert re.search(r'(?m)^\s*analysis_checkpoint_epochs:\s*\[[^\n]*\]\s*$', txt), p
        if stage_name in {'model_training', 'common_training_defaults'} or p.name in {'model_training.yaml', 'model_training_ddp.yaml'} or 'ddp_train_scenario' in str(p) or 'neuron_motivation_scenario/train' in str(p):
            training_cfgs.append((p, node))
            assert forbidden_public_training_keys.isdisjoint(node.keys()), p
            assert node.get('signal_window') in {'hann', 'none'}, p
    assert training_cfgs
    for p, node in training_cfgs:
        assert 'scenario_mode' in node, p
        assert node.get('readout_mode') in {'temporal_membrane', 'final_membrane', 'first_spike', 'max_fire', 'spikegru_max_over_time'}, p
        assert 'analysis_checkpoint_epochs' in node, p
        assert 'lambda_psd_pca_input' in node and 'lambda_psd_pca_adjacent' in node, p
        assert node.get('compile') is True, p
        assert node.get('compile_cpu_threads') in {None, 2}, p
        assert node.get('amp') in {'off', 'on'}, p
        assert 'amp_bf16_safe' not in node, p
        forbidden_compile_keys = {
            'compile_model', 'compile_backend', 'compile_mode', 'compile_fullgraph',
            'compile_dynamic', 'compile_train_step', 'compile_eval_step',
            'compile_threads', 'compile_policy', 'compile_stance',
        }
        assert forbidden_compile_keys.isdisjoint(node.keys()), p
        assert 'tear' in node and int(node['tear']) > 0, p
        assert 'band_edge' in node, p
        assert 'output_root' not in node, p
        assert node.get('signal_window') in {'hann', 'none'}, p


def test_stage_parser_smoke():
    parse_args_with_config(dp.build_arg_parser(), argv=['--config', 'config/data_prep.yaml'], stage_key='data_prep')
    parse_args_with_config(dpsd.build_arg_parser(), argv=['--config', 'config/dataset_psd.yaml'], stage_key='dataset_psd')
    parse_args_with_config(pa.build_arg_parser(), argv=['--config', 'config/psd_analysis.yaml'], stage_key='psd_analysis')
    parse_args_with_config(epsd.build_arg_parser(), argv=['--config', 'config/element_psd.yaml'], stage_key='element_psd')
    parse_args_with_config(efft.build_arg_parser(), argv=['--config', 'config/element_fft.yaml'], stage_key='element_fft')
    parse_args_with_config(dfft.build_arg_parser(), argv=['--config', 'config/dataset_fft.yaml'], stage_key='dataset_fft')
    parse_args_with_config(mt.build_arg_parser(), argv=['--config', 'config/model_training.yaml'], stage_key='model_training')
    ddp_args = parse_args_with_config(mt.build_arg_parser(), argv=['--config', 'config/model_training_ddp.yaml'], stage_key='model_training')
    assert ddp_args.compile_cpu_threads == 2
