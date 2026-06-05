import pytest

from src.model_training import _normalize_constraint_args, build_arg_parser


def _parse(argv):
    return build_arg_parser().parse_args(argv)


def _base_args():
    return ['--dataset','mnist','--prep_root','/tmp','--neuron_type','lif','--recurrent','false','--reset','soft','--v_th','fixed','1.0','--filter','train','--hidden_spec','8,8','--readout_mode','temporal_membrane','--epochs','1','--batch_size','2','--lr','0.001','--seed','1','--checkpoint_root','/tmp/c','--metric_root','/tmp/m']


def test_scenario_cli_parse_json_edges():
    args = _parse(_base_args() + [
        '--scenario_mode','clip_structure',
        '--alpha_clip_edges','[[[0.0,0.5],[0.5,1.0]],[[0.0,0.5],[0.5,1.0]]]',
        '--band_edge','[null,null]',
        '--tear','2',
    ])
    cfg = _normalize_constraint_args(args)
    assert cfg.mode == 'clipstructure'
    assert cfg.alpha_clip_edges[0][0] == [0.0, 0.5]
    assert cfg.band_edge == [None, None]
    assert cfg.tear == 2


def test_constraint_mode_cli_removed():
    with pytest.raises(SystemExit):
        _parse(_base_args() + ['--constraint_mode','clip'])



def test_legacy_constraint_aliases_removed():
    for alias in ('--lif_alpha_clip_edges', '--rf_frequency_clip_edges', '--constraint_tear'):
        with pytest.raises(SystemExit):
            _parse(_base_args() + [alias, '1'])
