import json, tempfile, os, subprocess, sys, csv
from pathlib import Path
import pytest

torch = pytest.importorskip('torch')

from psd_snn.config.specs import ExperimentConfig, to_sanitized_dict
from psd_snn.models.factory import build_model
from psd_snn.analysis.signal_map.emitter import bt_to_srt
from psd_snn.analysis.common import CheckpointRef, load_checkpoint_bundle
from psd_snn.analysis.common.model_restore import restore_model_from_bundle


def _cfg(kind):
    c=ExperimentConfig(); c.model.topology.kind=kind; c.model.topology.input_dim=8; c.model.topology.output_dim=3
    c.signal_analysis.artifact.output_dir='/tmp/x'
    return c


def _run_cli(mod, cfg, ckpt, out_dir):
    cfg.signal_analysis.artifact.output_dir = out_dir
    with tempfile.TemporaryDirectory() as td:
        cpath=Path(td)/'cfg.json'; cpath.write_text(json.dumps(to_sanitized_dict(cfg)))
        return subprocess.run([sys.executable,'-m',mod,'--config',str(cpath),'--mode','checkpoint','--checkpoint',str(ckpt)], env={**os.environ,'PYTHONPATH':'src'}, capture_output=True)


def _dummy_input(kind):
    return torch.randn(2,4,1,8,8) if kind in {'vgg','resnet'} else torch.randn(2,5,8)


def test_fixed_build_and_trace_smoke_all_supported():
    for k in ['gru','ssm','s4','spike_transformer']:
        m=build_model(_cfg(k).model)
        x=_dummy_input(k)
        y,tr=m(x,capture_trace=True,probe_family='balanced_global',label='na')
        assert y.shape==(2,3)
        maps=bt_to_srt(tr[0].tensor)
        assert maps.shape[0]==2 and maps.shape[2]==x.shape[1]


def test_vgg_resnet_smoke_btchw():
    for k in ['vgg','resnet']:
        m=build_model(_cfg(k).model)
        x=torch.randn(2,4,1,8,8)
        y,tr=m(x,capture_trace=True,probe_family='balanced_global',label='na')
        assert y.shape==(2,3)
        assert tr[0].tensor.ndim in (3,5)


def test_unknown_topology_unsupported_restore_status():
    cfg=_cfg('gru'); cfg.model.topology.kind='unknown'
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'c.pt'
        torch.save({'state_dict':{},'config':to_sanitized_dict(cfg),'checkpoint_epoch':1,'metadata':{}},p)
        b=load_checkpoint_bundle(CheckpointRef(path=str(p)))
        r=restore_model_from_bundle(b)
        assert r.restore_status=='unsupported_topology'


def test_checkpoint_analyze_signal_fixed_topologies_smoke():
    for k in ['gru','ssm','vgg','resnet','spike_transformer']:
        cfg=_cfg(k)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'c.pt'; m=build_model(cfg.model)
            torch.save({'state_dict':m.state_dict(),'config':to_sanitized_dict(cfg),'checkpoint_epoch':1,'metadata':{}},p)
            cp=_run_cli('psd_snn.cli.analyze_signal', cfg, p, f'/tmp/out_sig_{k}')
            assert cp.returncode==0
            am=Path(f'/tmp/out_sig_{k}')/'analysis_manifest.csv'; assert am.exists()


def test_checkpoint_analyze_fft2d_fixed_topologies_smoke_and_sidecars():
    for k in ['gru','ssm','vgg','resnet','spike_transformer']:
        cfg=_cfg(k); cfg.signal_analysis.fft2d.enabled=True
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'c.pt'; m=build_model(cfg.model)
            torch.save({'state_dict':m.state_dict(),'config':to_sanitized_dict(cfg),'checkpoint_epoch':1,'metadata':{}},p)
            cp=_run_cli('psd_snn.cli.analyze_fft2d', cfg, p, f'/tmp/out_fft_{k}')
            assert cp.returncode==0
            root=Path(f'/tmp/out_fft_{k}')
            assert (root/'spectral_matrix_2d.csv').exists()
            assert (root/'spectral_matrix_2d_row_axis.csv').exists()
            assert (root/'spectral_matrix_2d_column_axis.csv').exists()


def test_analyze_fft2d_missing_checkpoint_writes_failure_row():
    cfg=_cfg('gru'); cfg.signal_analysis.artifact.output_dir='/tmp/out_fft_missing'
    with tempfile.TemporaryDirectory() as td:
        cpath=Path(td)/'cfg.json'; cpath.write_text(json.dumps(to_sanitized_dict(cfg)))
        cp=subprocess.run([sys.executable,'-m','psd_snn.cli.analyze_fft2d','--config',str(cpath),'--mode','checkpoint','--checkpoint',str(Path(td)/'missing.pt')], env={**os.environ,'PYTHONPATH':'src'}, capture_output=True)
        assert cp.returncode!=0
        rows=list(csv.DictReader((Path('/tmp/out_fft_missing')/'analysis_manifest.csv').open()))
        assert any(r['status']=='checkpoint_load_failed' for r in rows)
