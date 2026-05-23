import json, tempfile, subprocess, sys, os, csv
from pathlib import Path
import pytest

torch = pytest.importorskip('torch')

from psd_snn.config.specs import ExperimentConfig, to_sanitized_dict


def test_train_help():
    cp=subprocess.run([sys.executable,'-m','psd_snn.cli.train','--help'], env={**os.environ,'PYTHONPATH':'src'}, capture_output=True)
    assert cp.returncode==0


def test_train_to_analyze_e2e():
    cfg=ExperimentConfig(); cfg.model.topology.kind='gru'; cfg.model.topology.input_dim=8; cfg.model.topology.output_dim=3
    with tempfile.TemporaryDirectory() as td:
        cpath=Path(td)/'cfg.json'; cpath.write_text(json.dumps(to_sanitized_dict(cfg)))
        out=Path(td)/'train_out'
        cp=subprocess.run([sys.executable,'-m','psd_snn.cli.train','--config',str(cpath),'--output_dir',str(out),'--epochs','1','--synthetic'], env={**os.environ,'PYTHONPATH':'src'}, capture_output=True)
        assert cp.returncode==0
        ckpt=out/'checkpoint_epoch_1.pt'; assert ckpt.exists()
        payload=torch.load(ckpt, map_location='cpu')
        assert 'state_dict' in payload and 'metadata' in payload and 'schema_version' not in str(payload)

        sig_out=Path(td)/'sig'; cfg.signal_analysis.artifact.output_dir=str(sig_out); cpath.write_text(json.dumps(to_sanitized_dict(cfg)))
        cp2=subprocess.run([sys.executable,'-m','psd_snn.cli.analyze_signal','--config',str(cpath),'--mode','checkpoint','--checkpoint',str(ckpt)], env={**os.environ,'PYTHONPATH':'src'}, capture_output=True)
        assert cp2.returncode==0
        assert (sig_out/'analysis_manifest.csv').exists()

        fft_out=Path(td)/'fft'; cfg.signal_analysis.artifact.output_dir=str(fft_out); cfg.signal_analysis.fft2d.enabled=True; cpath.write_text(json.dumps(to_sanitized_dict(cfg)))
        cp3=subprocess.run([sys.executable,'-m','psd_snn.cli.analyze_fft2d','--config',str(cpath),'--mode','checkpoint','--checkpoint',str(ckpt)], env={**os.environ,'PYTHONPATH':'src'}, capture_output=True)
        assert cp3.returncode==0
        assert (fft_out/'spectral_matrix_2d.csv').exists()
