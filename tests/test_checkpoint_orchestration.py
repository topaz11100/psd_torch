from __future__ import annotations
import tempfile
import pytest


def test_run_context_manifest_dataclasses():
    from psd_snn.analysis.common.run_context import RunContext, AnalysisRunManifest
    rc = RunContext(run_id='r1', output_dir='out')
    mf = AnalysisRunManifest(run_id='r1', checkpoint_epoch=1, split='test', scope='test_balanced_global', probe_family='balanced_global', analysis_method='psd', representative='mean', spectral_axis='exact', status='ok')
    assert rc.run_id == 'r1'
    assert mf.analysis_method == 'psd'


def test_checkpoint_roundtrip_restore_status_ok():
    torch = pytest.importorskip('torch')
    pytest.importorskip('spikingjelly')
    from psd_snn.config.specs import ExperimentConfig, to_sanitized_dict
    from psd_snn.models.mlp.builder import build_mlp_stack_model
    from psd_snn.analysis.common import CheckpointRef, load_checkpoint_bundle, restore_model_from_bundle

    cfg = ExperimentConfig()
    model = build_mlp_stack_model(cfg.model)
    with tempfile.TemporaryDirectory() as td:
        p = f"{td}/ckpt.pt"
        torch.save({'state_dict': model.state_dict(), 'config': to_sanitized_dict(cfg), 'checkpoint_epoch': 3, 'metadata': {'model': {'topology.kind': 'mlp_stack'}}}, p)
        b = load_checkpoint_bundle(CheckpointRef(path=p))
        r = restore_model_from_bundle(b, device='cpu')
        assert r.restore_status == 'ok'
        assert r.checkpoint_epoch == 3
        assert r.model is not None and not r.model.training
