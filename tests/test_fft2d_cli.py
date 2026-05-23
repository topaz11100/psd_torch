import pytest
pytest.importorskip("torch")
import json, tempfile, os, subprocess, sys

def test_fft2d_cli_smoke():
    cfg={'signal_analysis':{'artifact':{'output_dir':'/tmp/fft2d_out'},'fft2d':{'enabled':True,'spectral_axis':'exact'}}}
    fd,p=tempfile.mkstemp(suffix='.json'); os.close(fd)
    with open(p,'w') as f: json.dump(cfg,f)
    cp=subprocess.run([sys.executable,'-m','psd_snn.cli.analyze_fft2d','--config',p], env={**os.environ,'PYTHONPATH':'src'}, capture_output=True)
    assert cp.returncode==0
