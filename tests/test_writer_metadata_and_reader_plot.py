import csv
from pathlib import Path
import tempfile
import subprocess, sys, os

import pytest

torch = pytest.importorskip('torch')

from psd_snn.artifacts.writers import SummaryWriter
from psd_snn.artifacts.reader import ArtifactReader
from psd_snn.artifacts.trace_writer import TraceArtifactWriter
from psd_snn.analysis.trace.adapter import LayerTraceRecord


def test_summary_common_metadata_and_manifest_failure_row():
    with tempfile.TemporaryDirectory() as td:
        w = SummaryWriter(td)
        w.write_results([{'type':'spectral_curve','representative':'mean','freq':[0.1],'power':[1.0],'axis_policy':'exact','meta':{'run_id':'r1','checkpoint_epoch':1,'split':'test','scope':'s','probe_family':'balanced_global','probe_manifest_id':'pm1','exclusion_family':'balanced_global','exclusion_scope':'x','layer_index':0,'layer_name':'l0','signal_kind':'hidden','series':'spike'}}])
        rows = list(csv.DictReader((Path(td)/'spectral_curve.csv').open()))
        for k in ['artifact_type','run_id','checkpoint_epoch','split','scope','probe_family','probe_manifest_id','exclusion_family','exclusion_scope','status']:
            assert k in rows[0]
        am = list(csv.DictReader((Path(td)/'analysis_manifest.csv').open()))
        assert am[0]['artifact_type'] == 'analysis_manifest'


def test_trace_manifest_metadata_and_unavailable_series():
    with tempfile.TemporaryDirectory() as td:
        tw = TraceArtifactWriter(td)
        ok = LayerTraceRecord(0,'l0','hidden','spike','sc','balanced_global','na',torch.ones(2,3,4))
        ok.run_id='r1'; ok.checkpoint_epoch=1; ok.split='test'; ok.scope='s'; ok.probe_manifest_id='pm1'; ok.exclusion_family='balanced_global'
        bad = LayerTraceRecord(1,'l1','out','logits','sc','balanced_global','na',None)
        bad.run_id='r1'; bad.checkpoint_epoch=1; bad.split='test'; bad.scope='s'
        tw.write_records([ok,bad], chunk_size=1)
        tw.write_manifest()
        rows = list(csv.DictReader((Path(td)/'trace_manifest.csv').open()))
        assert any(r['status']=='unavailable_series' for r in rows)
        okrow = [r for r in rows if r['series']=='spike'][0]
        assert okrow['probe_manifest_id']=='pm1'
        assert okrow['path'].endswith('.pt')


def test_reader_and_plot_cli_smoke():
    with tempfile.TemporaryDirectory() as td:
        w = SummaryWriter(td)
        w.write_results([{'type':'spectral_curve','representative':'mean','freq':[0.1,0.2],'power':[1.0,2.0],'axis_policy':'exact','meta':{'run_id':'r1'}}, {'type':'spectral_matrix_1d','representative':'element_psd','matrix':[[1.0,2.0]],'spectral_axis':'exact','meta':{'run_id':'r1'}}, {'type':'spectral_matrix_2d','matrix':[[1.0,2.0]],'row_axis':[0.0],'col_axis':[0.1,0.2],'spectral_axis':'exact','meta':{'run_id':'r1'}}])
        r = ArtifactReader(td)
        assert r.read_spectral_curve()
        assert r.read_spectral_matrix_1d()
        assert r.read_spectral_matrix_2d()
        assert r.read_spectral_matrix_2d_axes()
        out = Path(td)/'p.png'
        cp = subprocess.run([sys.executable,'-m','psd_snn.cli.plot_artifacts','--input',td,'--artifact-type','spectral_curve','--output',str(out)], env={**os.environ,'PYTHONPATH':'src'}, capture_output=True)
        assert cp.returncode == 0
        assert out.exists()


def test_plot_cli_pca_and_trace_smoke():
    import tempfile, subprocess, sys, os, torch, csv
    from psd_snn.artifacts.writers import SummaryWriter
    from psd_snn.artifacts.trace_writer import TraceArtifactWriter
    from psd_snn.analysis.trace.adapter import LayerTraceRecord
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        w=SummaryWriter(td)
        w.write_results([{'type':'pca_basis','pca_basis_id':'b1','explained_variance':[1.0],'explained_variance_ratio':[1.0],'basis_artifact_path':'x.pt','meta':{'run_id':'r1'}}])
        out=Path(td)/'pca.png'
        cp=subprocess.run([sys.executable,'-m','psd_snn.cli.plot_artifacts','--input',td,'--artifact-type','pca_basis','--output',str(out)], env={**os.environ,'PYTHONPATH':'src'}, capture_output=True)
        assert cp.returncode==0 and out.exists()
        tw=TraceArtifactWriter(td)
        tr=LayerTraceRecord(0,'l0','hidden','spike','s','balanced_global','na',torch.ones(1,4,5))
        tw.write_records([tr], chunk_size=1); tw.write_manifest()
        out2=Path(td)/'trace.png'
        cp2=subprocess.run([sys.executable,'-m','psd_snn.cli.plot_artifacts','--input',td,'--artifact-type','trace','--output',str(out2)], env={**os.environ,'PYTHONPATH':'src'}, capture_output=True)
        assert cp2.returncode==0 and out2.exists()
