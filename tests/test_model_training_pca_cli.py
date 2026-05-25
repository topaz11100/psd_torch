from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.model_training as mt

def test_parser_accepts_psd_pca_regularization_options():
    p=mt.build_arg_parser()
    a=p.parse_args(['--dataset','d','--prep_root','p','--model','lif_soft_fixed','--hidden_spec','8','--readout_mode','temporal_membrane','--epochs','1','--batch_size','2','--lr','0.001','--seed','0','--checkpoint_root','c','--metric_root','m','--lambda_psd_rep_1d','0.1','--lambda_psd_pca_1d','0.2','--lambda_psd_pca_mimo','0.3','--psd_reg_variant','centered','--psd_reg_output_family','membrane','--pca_dim_per_layer','4','2'])
    assert a.lambda_psd_rep_1d==0.1
    assert a.lambda_psd_pca_1d==0.2
    assert a.lambda_psd_pca_mimo==0.3
    assert a.psd_reg_variant=='centered'
    assert a.psd_reg_output_family=='membrane'
    assert a.pca_dim_per_layer==['4','2']
