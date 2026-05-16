"""DH-SNN/vanilla SFNN SHD reinterpretation case metadata."""

CASE_ID = 'dh_snn'
CASE_SPEC = {
    'experiment_id': CASE_ID,
    'paper_experiment': 'DH-SNN Fig. 3f Fig. 4f Table 1 SHD vanilla SFNN vs DH-SFNN',
    'paper_result_location': 'Fig. 3f / Fig. 4f / Table 1',
    'compared_models': ['vanilla SFNN', 'DH-SFNN'],
    'dataset': 'SHD',
    'prep_profile': 'dh_snn_shd_t1000',
    'origin_code_path': 'Origin/neuron_model/DH_SNN_neuron.py',
    'fixed_setting': {'dataset': 'SHD'},
    'hook_families': ['x_probe', 'x_layer', 'y_mem', 'y_spike', 'i_dend', 'z_dend', 'tau_group', 'y_soma_mem'],
}
