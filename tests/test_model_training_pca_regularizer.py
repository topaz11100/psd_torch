from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.model_training as mt


def test_psd_metadata_from_args_and_parser_validation():
    parser = mt.build_arg_parser()
    args = parser.parse_args([
        '--dataset','d','--prep_root','p','--model','lif_soft_fixed','--hidden_spec','8','--readout_mode','temporal_membrane','--epochs','1','--batch_size','2','--lr','0.001','--seed','0','--checkpoint_root','c','--metric_root','m',
        '--lambda_psd_rep_1d','0.1','--lambda_psd_pca','0.5','--psd_reg_variant','raw','--psd_reg_output_family','spike'
    ])
    meta = mt._psd_regularization_metadata_from_args(args)
    assert meta['lambda_psd_rep_1d'] == 0.1
    assert meta['lambda_psd_pca'] == 0.5
    assert meta['psd_reg_variant'] == 'raw'
    assert meta['psd_reg_output_family'] == 'spike'


def test_resume_conflict_psd_regularization_metadata():
    current = {
        'lambda_psd_rep_1d': 0.5,
        'lambda_psd_pca': 0.0,
        'psd_reg_variant': 'raw',
        'psd_reg_output_family': 'spike',
    }
    ck = {
        'lambda_psd_rep_1d': 0.0,
        'lambda_psd_pca': 0.0,
        'psd_reg_variant': 'raw',
        'psd_reg_output_family': 'spike',
    }
    with pytest.raises(ValueError, match='PSD regularization resume mismatch'):
        mt._assert_psd_resume_compatible(current, ck)
