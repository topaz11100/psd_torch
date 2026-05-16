"""Need High Max-Former/MS-QKFormer reinterpretation case metadata."""

CASE_ID = 'need_high'
CASE_SPEC = {
    'experiment_id': CASE_ID,
    'paper_experiment': 'Need High Section 4.1 Table 2 CIFAR10-DVS Max-Former vs MS-QKFormer',
    'paper_result_location': 'Section 4.1 / Table 2',
    'compared_models': ['Max-Former', 'MS-QKFormer'],
    'dataset': 'CIFAR10-DVS',
    'prep_profile': 'need_high_cifar10_dvs_t16',
    'origin_code_path': 'Origin/need-high/',
    'fixed_setting': {'time_steps': 16},
    'hook_families': ['x_probe', 'x_layer', 'y_mem', 'y_spike', 'x_embed', 'q_spike', 'k_spike', 'v_spike', 'attn_out'],
}
