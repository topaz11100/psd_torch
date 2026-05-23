import pytest
torch = pytest.importorskip('torch')
import json, tempfile, os
from psd_snn.config.specs import load_experiment_config
from psd_snn.analysis.signal.runner import SignalAnalysisRunner, SignalMapRecord


def _cfg(rep='mean', axis='exact'):
    d={'signal_analysis':{'psd':{'representative':{'method':rep,'pca':{'n_components':2,'basis_mode':'fit_per_checkpoint'}},'spectral_axis':axis,'userbin_edges':[0.0,0.25,0.5],'userbin_reducer':'mean','scale_outputs':'raw'},'artifact':{'output_dir':'/tmp/psd_out'}}}
    fd,p=tempfile.mkstemp(suffix='.json'); os.close(fd)
    with open(p,'w') as f: json.dump(d,f)
    return p

def test_mean_median_element_pca():
    maps=torch.randn(3,5,32)
    for rep in ['mean','median','element_psd','pca']:
        p=_cfg(rep)
        cfg=load_experiment_config(p)
        r=SignalAnalysisRunner(cfg.signal_analysis)
        r.update_signal_maps([SignalMapRecord(maps=maps, metadata={})])
        out=r.finalize()
        assert len(out)>=1

def test_userbin_path():
    maps=torch.randn(3,4,32)
    p=_cfg('element_psd','userbin')
    cfg=load_experiment_config(p)
    r=SignalAnalysisRunner(cfg.signal_analysis)
    r.update_signal_maps([SignalMapRecord(maps=maps, metadata={})])
    out=r.finalize()
    assert any(x.get('spectral_axis')=='userbin' for x in out if x['type']=='spectral_matrix_1d')
