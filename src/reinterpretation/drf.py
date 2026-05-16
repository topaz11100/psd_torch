"""D-RF/BRF SHD reinterpretation case metadata."""

CASE_ID = 'drf'
CASE_SPEC = {
    'experiment_id': CASE_ID,
    'paper_experiment': 'D-RF Section 5.1 Table 1 SHD D-RF vs BRF',
    'paper_result_location': 'Section 5.1 / Table 1',
    'compared_models': ['D-RF', 'BRF'],
    'dataset': 'SHD',
    'prep_profile': 'drf_shd_t250',
    'origin_code_path': 'Origin/neuron_model/D_RF_neuron.py',
    'fixed_setting': {'dataset': 'SHD'},
    'hook_families': ['x_probe', 'x_layer', 'y_mem', 'y_spike', 'i_branch', 'z_branch_real', 'z_branch_imag', 'y_soma_mem'],
}
