from psd_snn.analysis.common.status import AnalysisStatus, AnalysisFailure


def test_status_ok():
    s = AnalysisStatus()
    assert s.status == 'ok'


def test_failure_status_strings_and_row_fields():
    statuses = [
        'ok','checkpoint_load_failed','model_restore_failed','unsupported_topology','state_dict_missing_keys',
        'state_dict_unexpected_keys','probe_build_failed','no_trace_records','unavailable_series','signal_map_failed',
        'pca_basis_missing','pca_basis_incompatible','distance_incompatible','writer_failed'
    ]
    for st in statuses:
        row = AnalysisFailure(status=st, run_id='r1', split='test').to_manifest_row()
        assert row['status'] == st
        assert 'schema_version' not in row and 'csv_v2' not in row
        assert 'run_id' in row and 'split' in row
