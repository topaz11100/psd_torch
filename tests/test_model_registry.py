import pytest

from src.model.model_registry import canonicalize_model_token


@pytest.mark.parametrize(
    ('token', 'canonical', 'family', 'recurrent', 'branch'),
    [
        ('tc_lif', 'tc_lif', 'tc_lif', False, None),
        ('tc_lif_R', 'tc_lif_R', 'tc_lif', True, None),
        ('tc', 'tc_lif', 'tc_lif', False, None),
        ('tc_R', 'tc_lif_R', 'tc_lif', True, None),
        ('tclif', 'tc_lif', 'tc_lif', False, None),
        ('tclif_R', 'tc_lif_R', 'tc_lif', True, None),
        ('ts_lif', 'ts_lif', 'ts_lif', False, None),
        ('ts_lif_R', 'ts_lif_R', 'ts_lif', True, None),
        ('ts', 'ts_lif', 'ts_lif', False, None),
        ('ts_R', 'ts_lif_R', 'ts_lif', True, None),
        ('tslif', 'ts_lif', 'ts_lif', False, None),
        ('tslif_R', 'ts_lif_R', 'ts_lif', True, None),
        ('dh', 'dh_snn_4', 'dh_snn', False, 4),
        ('dh_snn', 'dh_snn_4', 'dh_snn', False, 4),
        ('dh_8', 'dh_snn_8', 'dh_snn', False, 8),
        ('dh_snn_8', 'dh_snn_8', 'dh_snn', False, 8),
        ('dh_R', 'dh_snn_R_4', 'dh_snn', True, 4),
        ('dh_snn_R', 'dh_snn_R_4', 'dh_snn', True, 4),
        ('dh_R_8', 'dh_snn_R_8', 'dh_snn', True, 8),
        ('dh_snn_R_8', 'dh_snn_R_8', 'dh_snn', True, 8),
        ('d_rf', 'd_rf_4', 'd_rf', False, 4),
        ('d_rf_4', 'd_rf_4', 'd_rf', False, 4),
        ('d_rf_8', 'd_rf_8', 'd_rf', False, 8),
    ],
)
def test_new_token_valid_cases(token, canonical, family, recurrent, branch):
    spec = canonicalize_model_token(token)
    assert spec.canonical_token == canonical
    assert spec.family == family
    assert spec.recurrent is recurrent
    assert spec.branch == branch


@pytest.mark.parametrize(
    'token',
    [
        'tc_lif_soft_fixed',
        'tc_lif_hard_train',
        'tc_lif_4',
        'ts_lif_soft_fixed',
        'ts_lif_hard_train',
        'ts_lif_R_4',
        'dh_snn_soft_fixed',
        'dh_snn_R_soft_fixed',
        'dh_snn_0',
        'dh_snn_R_0',
        'd_rf_0',
        'd_rf_R',
        'd_rf_R_4',
        'd_rf_soft_fixed',
        'd_rf_4_soft_fixed',
    ],
)
def test_new_token_invalid_cases(token):
    with pytest.raises(ValueError):
        canonicalize_model_token(token)


@pytest.mark.parametrize(
    ('token', 'family', 'recurrent', 'reset_mode', 'trainable_threshold'),
    [
        ('lif_soft_fixed', 'lif', False, 'soft_reset', False),
        ('lif_R_soft_fixed', 'lif', True, 'soft_reset', False),
        ('rf_soft_fixed', 'rf', False, 'soft_reset', False),
        ('rf_R_soft_fixed', 'rf', True, 'soft_reset', False),
        ('vgg11_lif_soft_fixed', 'cnn_lif', False, 'soft_reset', False),
        ('resnet18_rf_hard_train', 'cnn_rf', False, 'hard_reset', True),
    ],
)
def test_backward_compatibility_tokens(token, family, recurrent, reset_mode, trainable_threshold):
    spec = canonicalize_model_token(token)
    assert spec.family == family
    assert spec.recurrent is recurrent
    assert spec.reset_mode == reset_mode
    assert spec.trainable_threshold is trainable_threshold
