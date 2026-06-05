import json
import subprocess
from pathlib import Path

from src.util.config import save_yaml

import pytest

from tests.test_model_training_smoke import _write_tiny_prepared_bundle


def test_model_training_ddp_smoke(tmp_path: Path):
    torch = pytest.importorskip('torch')
    if (not torch.cuda.is_available()) or torch.cuda.device_count() < 2:
        pytest.skip(f'2-GPU CUDA 환경 없음: cuda={torch.cuda.is_available()} count={torch.cuda.device_count() if torch.cuda.is_available() else 0}')

    prep_root, dataset = _write_tiny_prepared_bundle(tmp_path)
    cfg = {
        'model_training': {
            'dataset': dataset,
            'prep_root': str(prep_root),
            'neuron_type': 'lif',
            'recurrent': False,
            'reset': 'soft',
            'v_th': ['fixed', 1.0],
            'filter': 'train',
            'hidden_spec': '8',
            'readout_mode': 'temporal_membrane',
            'epochs': 1,
            'batch_size': 2,
            'lr': 0.001,
            'num_workers': 0,
            'seed': 0,
            'gpu_index': 0,
            'analysis_checkpoint_epochs': [1],
            'checkpoint_root': str(tmp_path / 'ckpt_ddp'),
            'metric_root': str(tmp_path / 'metric_ddp'),
            'ddp': True,
            'ddp_world_size': 2,
            'batch_size_is_global': True,
            'signal_window': 'hann',
            'compile': False,
            'amp': 'off',
            'run_timestamp': 'TEST_DDP_RUN',
        }
    }
    cfg_path = tmp_path / 'smoke_ddp.yaml'
    save_yaml(cfg_path, cfg)
    proc = subprocess.run(['torchrun', '--standalone', '--nproc_per_node=2', 'src/model_training.py', '--config', str(cfg_path), '--ddp', 'true'], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + '\n' + proc.stderr

    ckpt = tmp_path / 'ckpt_ddp' / 'run_TEST_DDP_RUN' / 'checkpoint_epoch_000001.pt'
    metric = tmp_path / 'metric_ddp' / 'run_TEST_DDP_RUN' / 'training_metrics.csv'
    assert ckpt.exists() and metric.exists()
    assert len(list((tmp_path / 'ckpt_ddp' / 'run_TEST_DDP_RUN').glob('*.pt'))) == 1
    payload = torch.load(ckpt, map_location='cpu', weights_only=False)
    assert not any(str(k).startswith('module.') for k in payload['state_dict'].keys())
