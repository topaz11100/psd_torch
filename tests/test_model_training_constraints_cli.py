import pytest

from src.model_training import _normalize_constraint_args, build_arg_parser


def _parse(argv):
    return build_arg_parser().parse_args(argv)


def test_constraint_cli_parse_and_alias():
    args = _parse(['--dataset','mnist','--prep_root','/tmp','--model','lif_soft_fixed','--hidden_spec','8,8','--readout_mode','temporal_membrane','--epochs','1','--batch_size','2','--lr','0.001','--seed','1','--checkpoint_root','/tmp/c','--metric_root','/tmp/m','--constraint_mode','clip_structure','--lif_alpha_clip_edges','0.0','0.5','1.0','--band_neuron_ends','4','4','--constraint_tear','2'])
    cfg = _normalize_constraint_args(args)
    assert cfg.mode == 'clipstructure'
    assert cfg.alpha_clip_edges == (0.0, 0.5, 1.0)
    assert cfg.tear == 2


def test_alias_conflict_raises():
    args = _parse(['--dataset','mnist','--prep_root','/tmp','--model','lif_soft_fixed','--hidden_spec','8,8','--readout_mode','temporal_membrane','--epochs','1','--batch_size','2','--lr','0.001','--seed','1','--checkpoint_root','/tmp/c','--metric_root','/tmp/m','--w_clip_edges','0.0','0.25','0.5','--rf_frequency_clip_edges','0.0','0.3','0.5'])
    with pytest.raises(ValueError):
        _normalize_constraint_args(args)
