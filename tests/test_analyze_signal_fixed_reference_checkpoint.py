from __future__ import annotations
import csv
import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('spikingjelly')

from psd_snn.config.specs import ExperimentConfig, to_sanitized_dict
from psd_snn.models.mlp.builder import build_mlp_stack_model
from psd_snn.cli.analyze_signal import main as analyze_signal_main


def _mk_ckpt(path: Path, epoch: int):
    cfg = ExperimentConfig()
    cfg.model.topology.input_dim = 8
    cfg.model.topology.hidden_widths = [6]
    cfg.model.topology.output_dim = 2
    cfg.model.cell.kind = 'lif'
    cfg.model.readout.kind = 'final_mem'
    model = build_mlp_stack_model(cfg.model)
    torch.save({'state_dict': model.state_dict(), 'config': to_sanitized_dict(cfg), 'checkpoint_epoch': epoch, 'metadata': {'model': {'topology.kind': 'mlp_stack'}}}, path)


def _read_csv(path: Path):
    with path.open() as f:
        return list(csv.DictReader(f))


def test_checkpoint_fixed_reference_pca_e2e(tmp_path: Path):
    ref = tmp_path / 'ref.pt'; tgt = tmp_path / 'tgt.pt'
    _mk_ckpt(ref, 1); _mk_ckpt(tgt, 2)
    out = tmp_path / 'out'
    cfg = {
        'model': {'topology': {'kind': 'mlp_stack', 'input_dim': 8, 'hidden_widths': [6], 'output_dim': 2}, 'cell': {'kind': 'lif'}, 'readout': {'kind': 'final_mem'}},
        'signal_analysis': {
            'artifact': {'output_dir': str(out)},
            'trace_save': {'enabled': True, 'series': ['spike'], 'chunk_size': 4},
            'psd': {'spectral_axis': 'exact', 'representative': {'method': 'pca', 'pca': {'n_components': 2, 'basis_mode': 'fixed_reference', 'reference_checkpoint': 1, 'reference_split': 'test', 'reference_probe_family': 'balanced_global', 'reference_scope': 'test_balanced_global'}}}
        }
    }
    cpath = tmp_path / 'cfg.json'; cpath.write_text(json.dumps(cfg))
    analyze_signal_main(['--config', str(cpath), '--mode', 'checkpoint', '--checkpoint', str(tgt), '--reference_checkpoint', str(ref), '--run_id', 'runfx', '--sample_count', '8', '--batch_size', '4'])

    am = out / 'analysis_manifest.csv'; pb = out / 'pca_basis.csv'; sm = out / 'spectral_matrix_1d.csv'; tm = out / 'trace_manifest.csv'
    assert am.exists() and pb.exists() and sm.exists() and tm.exists()

    am_rows = _read_csv(am)
    assert 'run_id' in am_rows[0] and 'checkpoint_epoch' in am_rows[0] and 'scope' in am_rows[0] and 'probe_family' in am_rows[0]
    assert 'schema_version' not in am_rows[0] and 'csv_v2' not in am_rows[0]

    pb_rows = _read_csv(pb)
    assert 'pca_basis_id' in pb_rows[0] and 'basis_artifact_path' in pb_rows[0] and 'component_id' in pb_rows[0]
    assert 'reference_checkpoint_epoch' in pb_rows[0] and pb_rows[0]['reference_checkpoint_epoch'] == '1'
    basis_path = Path(pb_rows[0]['basis_artifact_path'])
    assert basis_path.exists()
    payload = torch.load(basis_path, map_location='cpu')
    for k in ('basis_id', 'mean', 'components', 'explained_variance', 'explained_variance_ratio'):
        assert k in payload

    sm_rows = _read_csv(sm)
    assert any(r.get('representative') == 'pca' for r in sm_rows)
    assert 'pca_basis_id' in sm_rows[0] and 'component_id' in sm_rows[0]
    assert any(r.get('checkpoint_epoch') == '2' for r in sm_rows)
    assert all(k in sm_rows[0] for k in ('run_id', 'split', 'scope', 'probe_family'))
    pbid = pb_rows[0]['pca_basis_id']
    assert any(r.get('pca_basis_id') == pbid for r in sm_rows)

    tm_rows = _read_csv(tm)
    assert 'sample_start' in tm_rows[0] and 'sample_count' in tm_rows[0] and 'time_length' in tm_rows[0]
    assert Path(tm_rows[0]['path']).exists()
