from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.util.config_cli import parse_args_with_config
import src.model_training as mt
import src.psd_analysis as pa

def test_readme_mentions_no_list_sweep_other_stages():
    text = Path('config/README.md').read_text(encoding='utf-8')
    assert '다른 stage에서는 dataset list를 허용하지 않는다' in text


def test_example_configs_parse_with_stage_parsers():
    model_args = parse_args_with_config(mt.build_arg_parser(), argv=['--config', 'config/model_training.json'], stage_key='model_training')
    assert model_args.psd_reg_variant in ('raw', 'centered')
    assert model_args.psd_reg_output_family in ('spike', 'membrane')
    psd_args = parse_args_with_config(pa.build_arg_parser(), argv=['--config', 'config/psd_analysis.json'], stage_key='psd_analysis')
    assert int(psd_args.pca_ref_epoch) >= 1


def test_readme_choices_match_parser_choices():
    text = Path('config/README.md').read_text(encoding='utf-8')
    assert 'psd_reg_variant' in text and '`raw`, `centered`' in text
    assert 'psd_reg_output_family' in text and '`spike`, `membrane`' in text
    assert 'analysis_distance_metric' in text and '`centered_l2`, `diff_l2`' in text
