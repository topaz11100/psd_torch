from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.model_training as mt


def test_parser_accepts_split_psd_pca_regularization_options():
    p = mt.build_arg_parser()
    a = p.parse_args([
        '--dataset','d','--prep_root','p','--neuron_type','lif','--recurrent','false','--reset','soft','--v_th','fixed','1.0','--filter','train',
        '--hidden_spec','8','--readout_mode','temporal_membrane','--epochs','1','--batch_size','2','--lr','0.001','--seed','0',
        '--checkpoint_root','c','--metric_root','m',
        '--lambda_psd_rep_input','0.1','--lambda_psd_rep_adjacent','0.2',
        '--lambda_psd_pca_input','0.3','--lambda_psd_pca_adjacent','0.4',
        '--signal_curve_centering','centered','--psd_reg_output_family','membrane','--pca_dim_per_layer','4','2'
    ])
    assert a.lambda_psd_rep_input == 0.1
    assert a.lambda_psd_rep_adjacent == 0.2
    assert a.lambda_psd_pca_input == 0.3
    assert a.lambda_psd_pca_adjacent == 0.4
    assert a.signal_curve_centering == 'centered'
    assert a.psd_reg_output_family == 'membrane'
    assert a.pca_dim_per_layer == ['4','2']
