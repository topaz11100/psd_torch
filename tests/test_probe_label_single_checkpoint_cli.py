from __future__ import annotations
import json
from pathlib import Path
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('spikingjelly')

from psd_snn.config.specs import ExperimentConfig, to_sanitized_dict
from psd_snn.models.mlp.builder import build_mlp_stack_model
from psd_snn.cli.analyze_signal import main as analyze_signal_main
from psd_snn.cli.analyze_fft2d import main as analyze_fft2d_main


def _mk_ckpt(path: Path):
    cfg = ExperimentConfig()
    model = build_mlp_stack_model(cfg.model)
    torch.save({'state_dict': model.state_dict(), 'config': to_sanitized_dict(cfg), 'checkpoint_epoch': 5, 'metadata': {'model': {'topology.kind': 'mlp_stack'}}}, path)


def _csv_rows(p: Path):
    import csv
    with p.open() as f:
        return list(csv.DictReader(f))


def test_analyze_signal_checkpoint_label_single(tmp_path: Path):
    ck = tmp_path / 'c.pt'; _mk_ckpt(ck)
    out = tmp_path / 'out'
    cfg = {'signal_analysis': {'artifact': {'output_dir': str(out)}, 'psd': {'representative': {'method':'mean'}}}}
    cp = tmp_path / 'c.json'; cp.write_text(json.dumps(cfg))
    analyze_signal_main(['--config', str(cp), '--mode', 'checkpoint', '--checkpoint', str(ck), '--probe_family', 'label_single', '--sample_count', '8'])
    rows = _csv_rows(out / 'analysis_manifest.csv')
    assert any(r.get('probe_family') == 'label_single' for r in rows)


def test_analyze_fft2d_checkpoint_label_single(tmp_path: Path):
    ck = tmp_path / 'c.pt'; _mk_ckpt(ck)
    out = tmp_path / 'out2'
    cfg = {'signal_analysis': {'artifact': {'output_dir': str(out)}, 'fft2d': {'enabled': True, 'spectral_axis': 'exact'}}}
    cp = tmp_path / 'f.json'; cp.write_text(json.dumps(cfg))
    analyze_fft2d_main(['--config', str(cp), '--mode', 'checkpoint', '--checkpoint', str(ck), '--probe_family', 'label_single', '--sample_count', '8'])
    rows = _csv_rows(out / 'spectral_matrix_2d.csv')
    assert any(r.get('probe_family') == 'label_single' for r in rows)
