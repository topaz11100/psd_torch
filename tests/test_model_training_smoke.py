import json
import subprocess
import sys
from pathlib import Path

import pytest


def _write_tiny_prepared_bundle(base: Path) -> tuple[Path, str]:
    import numpy as np

    dataset = 's-mnist'
    root = base / 'prepared' / dataset
    root.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype([('sample_index', np.int64), ('label', np.int64), ('input', np.float32, (784, 1))])
    train = np.zeros(6, dtype=dtype)
    test = np.zeros(4, dtype=dtype)
    for i in range(6):
        train[i]['sample_index'] = i
        train[i]['label'] = i % 2
        train[i]['input'] = np.full((784, 1), float(i), dtype=np.float32)
    for i in range(4):
        test[i]['sample_index'] = i
        test[i]['label'] = i % 2
        test[i]['input'] = np.full((784, 1), float(i), dtype=np.float32)
    np.save(root / 'train.npy', train)
    np.save(root / 'test.npy', test)
    manifest = {
        'dataset_name': dataset,
        'storage_format': 'single_structured_npy_v1',
        'split_internal_order_preserved': True,
        'files': {'train': 'train.npy', 'test': 'test.npy'},
        'stored_shape': [784, 1],
        'stored_dtype': 'float32',
        'sequence_length': 784,
        'input_dim': 1,
        'training_view_name': 'model_input',
        'psd_view_name': 'model_input',
        'available_views': ['model_input'],
        'psd_axis_kind': 'sequence',
        'psd_time_axis': 0,
        'psd_row_axes': [1],
        'psd_flatten_rule': 'row_major',
        'psd_logical_shape': [784, 1],
        'default_hidden_sizes': [8],
        'label_dtype': 'int64',
        'sample_index_dtype': 'int64',
        'num_classes': 2,
    }
    (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return base / 'prepared', dataset


def test_model_training_single_smoke(tmp_path: Path):
    pytest.importorskip('torch')
    prep_root, dataset = _write_tiny_prepared_bundle(tmp_path)
    cfg = {
        'model_training': {
            'dataset': dataset,
            'prep_root': str(prep_root),
            'model': 'lif_soft_fixed',
            'hidden_spec': '8',
            'readout_mode': 'temporal_membrane',
            'epochs': 1,
            'batch_size': 2,
            'lr': 0.001,
            'num_workers': 0,
            'seed': 0,
            'gpu_index': 0,
            'anal_epoch_list': [1],
            'checkpoint_root': str(tmp_path / 'ckpt'),
            'metric_root': str(tmp_path / 'metric'),
            'output_root': str(tmp_path / 'out'),
            'ddp': False,
            'ddp_world_size': 2,
            'batch_size_is_global': True,
        }
    }
    cfg_path = tmp_path / 'smoke.json'
    cfg_path.write_text(json.dumps(cfg), encoding='utf-8')
    proc = subprocess.run([sys.executable, 'src/model_training.py', '--config', str(cfg_path)], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + '\n' + proc.stderr
    ckpt = tmp_path / 'ckpt' / 'checkpoint_epoch_000001.pt'
    metric = tmp_path / 'metric' / 'training_metrics.csv'
    assert ckpt.exists()
    assert metric.exists()
    import torch
    payload = torch.load(ckpt, map_location='cpu', weights_only=False)
    assert isinstance(payload, dict) and 'state_dict' in payload
    assert not any(str(k).startswith('module.') for k in payload['state_dict'].keys())
    text = metric.read_text(encoding='utf-8')
    assert ',train,' in text and ',test,' in text
