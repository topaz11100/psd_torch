from pathlib import Path
import csv


def test_acceptance_forbidden_and_layout_guards():
    assert not Path('src/reinterpretation').exists()
    txt='\n'.join([p.read_text() for p in Path('src/psd_snn').rglob('*.py')])
    for bad in ['analysis_2d_fft','schema_version','csv_v2','recurrent_source']:
        assert bad not in txt


def test_trace_manifest_required_columns_and_pt_policy():
    p=Path('/tmp/out_sig_gru')/'trace_manifest.csv'
    if not p.exists():
        return
    rows=list(csv.DictReader(p.open()))
    if not rows:
        return
    required=['artifact_type','run_id','checkpoint_epoch','split','scope','probe_family','layer_index','layer_name','series','path','layout','sample_count','time_length','status']
    for k in required:
        assert k in rows[0]
    for r in rows:
        if r.get('path'):
            assert r['path'].endswith('.pt')


def test_analysis_manifest_success_failure_possible():
    ok=Path('/tmp/out_sig_gru')/'analysis_manifest.csv'
    bad=Path('/tmp/out_fft_missing')/'analysis_manifest.csv'
    if ok.exists():
        rows=list(csv.DictReader(ok.open())); assert any(r['status']=='ok' for r in rows)
    if bad.exists():
        rows=list(csv.DictReader(bad.open())); assert any(r['status']!='ok' for r in rows)


def test_distance_strictness_source_present():
    txt=Path('src/psd_snn/artifacts/identity.py').read_text()
    assert 'pca_basis_id' in txt and 'spectral_axis' in txt
