import argparse
from pathlib import Path

import torch

import src.model_training as mt
from src.model.snn_builder import LayerRecord
from src.model.psd_minibatch_regularizer import compute_minibatch_psd_regularizer


def _args(dim: int) -> argparse.Namespace:
    return argparse.Namespace(
        pca_dim_per_layer=[str(dim)],
        psd_reg_output_family='spike',
        signal_curve_centering='centered',
        ddp=True,
        lambda_psd_pca_input=0.1,
        lambda_psd_pca_adjacent=0.1,
        lambda_psd_rep_input=0.0,
        lambda_psd_rep_adjacent=0.0,
        signal_curve_scale='raw',
    )


def _payload(rank: int, offset: float = 0.0):
    generator = torch.Generator().manual_seed(100 + rank)
    x = torch.randn(2, 8, 3, generator=generator) + offset
    spike = torch.sigmoid(torch.randn(2, 8, 4, generator=generator) + offset)
    return {
        'rank': rank,
        'local_rank': rank,
        'world_size': 2,
        'input': x,
        'hidden_records': [{'layer_index': 0, 'layer_name': 'hidden_0', 'spike': spike}],
        'num_samples': int(x.shape[0]),
    }


def test_global_pca_reference_bank_uses_all_gathered_rank_payloads_dim1_per_relation():
    ctx = mt.DDPContext(True, 0, 0, 2, torch.device('cpu'), True)
    banks = mt._build_global_pca_reference_bank_from_payloads([_payload(1, 10.0), _payload(0, 0.0)], _args(1), ctx)
    assert set(banks.keys()) == {'input', 'adjacent'}
    ref = banks['input']['hidden_0']
    assert ref.dim == 1
    assert ref.metadata['variant'] == 'centered'
    assert ref.metadata['reference_policy'] == 'ddp_all_gather_first_local_batch_build_global_reference_rank0_broadcast_per_relation'
    assert ref.metadata['rank_order'] == [0, 1]
    assert ref.metadata['per_rank_reference_samples'] == [2, 2]
    assert ref.metadata['global_reference_samples'] == 4
    assert ref.metadata['pca_mode'] == '1d'
    assert banks['adjacent']['hidden_0'].metadata['relation'] == 'adjacent'


def test_pca_dim_one_activates_only_1d_regularizer_bucket():
    ctx = mt.DDPContext(False, 0, 0, 1, torch.device('cpu'), True)
    payload = _payload(0, 0.0)
    banks = mt._build_global_pca_reference_bank_from_payloads([payload], _args(1), ctx)
    record = LayerRecord(layer_name='hidden_0', membrane=torch.empty(0), spike=payload['hidden_records'][0]['spike'] * 0.7)
    out = compute_minibatch_psd_regularizer(
        payload['input'],
        [record],
        variant='centered',
        output_family='spike',
        lambda_rep_input=0.0,
        lambda_rep_adjacent=0.0,
        lambda_pca_input=1.0,
        lambda_pca_adjacent=0.0,
        pca_reference_banks=banks,
        curve_scale='raw',
    )
    assert out.metadata['relation_metadata']['input']['pca_mode_by_layer'] == {'hidden_0': '1d'}
    assert float(out.pca_mimo.detach().item()) == 0.0
    assert torch.isfinite(out.pca_1d)


def test_pca_dim_two_activates_only_mimo_regularizer_bucket():
    ctx = mt.DDPContext(False, 0, 0, 1, torch.device('cpu'), True)
    payload = _payload(0, 0.0)
    banks = mt._build_global_pca_reference_bank_from_payloads([payload], _args(2), ctx)
    record = LayerRecord(layer_name='hidden_0', membrane=torch.empty(0), spike=payload['hidden_records'][0]['spike'] * 0.7)
    out = compute_minibatch_psd_regularizer(
        payload['input'],
        [record],
        variant='centered',
        output_family='spike',
        lambda_rep_input=0.0,
        lambda_rep_adjacent=0.0,
        lambda_pca_input=1.0,
        lambda_pca_adjacent=0.0,
        pca_reference_banks=banks,
        curve_scale='raw',
    )
    assert out.metadata['relation_metadata']['input']['pca_mode_by_layer'] == {'hidden_0': 'mimo'}
    assert float(out.pca_1d.detach().item()) == 0.0
    assert torch.isfinite(out.pca_mimo)


def test_compile_cache_env_is_overridden_and_rank_partitioned(tmp_path: Path, monkeypatch):
    root = tmp_path / 'cache' / 'experiment_a' / 'config_x'
    monkeypatch.setenv('PSD_TORCH_COMPILE_CACHE_DIR', str(root))
    monkeypatch.setenv('TORCHINDUCTOR_CACHE_DIR', '/tmp/old_shared_cache')
    status = mt._configure_compile_cache(enabled=True, rank=1)
    expected = root / 'rank1'
    assert status['cache_dir'] == str(expected.resolve())
    assert status['previous_env']['TORCHINDUCTOR_CACHE_DIR'] == '/tmp/old_shared_cache'
    assert status['TORCHINDUCTOR_CACHE_DIR'] == str(expected.resolve())
    assert expected.exists()
