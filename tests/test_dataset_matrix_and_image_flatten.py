import json
from pathlib import Path

from src.util.config import save_yaml

import numpy as np
import pytest

from src.data.registry import _select_image_training_view
from src.model_training import main as model_training_main


def _write_structured_bundle(base: Path, dataset: str, shape: tuple[int, ...], num_classes: int = 2) -> Path:
    root = base / 'prepared' / dataset
    root.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype([('sample_index', np.int64), ('label', np.int64), ('input', np.float32, shape)])
    for split, count in [('train', 4), ('test', 2)]:
        arr = np.zeros(count, dtype=dtype)
        for index in range(count):
            arr[index]['sample_index'] = index
            arr[index]['label'] = index % num_classes
            arr[index]['input'] = np.full(shape, float(index + 1), dtype=np.float32)
        np.save(root / f'{split}.npy', arr)
    manifest = {
        'dataset_name': dataset,
        'storage_format': 'single_structured_npy_v1',
        'split_internal_order_preserved': True,
        'files': {'train': 'train.npy', 'test': 'test.npy'},
        'stored_shape': list(shape),
        'stored_dtype': 'float32',
        'sequence_length': int(shape[0]),
        'input_dim': int(np.prod(shape[1:])),
        'training_view_name': 'model_input',
        'psd_view_name': 'model_input_psd_view',
        'available_views': ['model_input', 'psd_input', 'model_input_psd_view', 'sequence_input'],
        'psd_axis_kind': 'temporal',
        'psd_time_axis': 2,
        'psd_row_axes': [1],
        'psd_flatten_rule': 'test_time_major',
        'psd_logical_shape': [int(np.prod(shape[1:])), int(shape[0])],
        'default_hidden_sizes': [4],
        'label_dtype': 'int64',
        'sample_index_dtype': 'int64',
        'num_classes': int(num_classes),
    }
    save_yaml(root / 'manifest.yaml', manifest)
    return base / 'prepared'


@pytest.mark.parametrize(
    ('dataset', 'shape'),
    [
        ('ssc', (12, 4)),
        ('s-cifar10', (12, 3)),
        ('deap', (12, 4)),
        ('uci-har', (12, 6)),
    ],
)
def test_requested_dataset_matrix_loads_and_trains_one_tiny_epoch(tmp_path: Path, dataset: str, shape: tuple[int, ...]):
    pytest.importorskip('torch')
    prep_root = _write_structured_bundle(tmp_path / dataset, dataset, shape)
    cfg = {
        'model_training': {
            'dataset': dataset,
            'prep_root': str(prep_root),
            'neuron_type': 'lif',
            'recurrent': False,
            'reset': 'soft',
            'v_th': ['fixed', 1.0],
            'filter': 'train',
            'hidden_spec': '4',
            'readout_mode': 'temporal_membrane',
            'epochs': 1,
            'batch_size': 2,
            'lr': 0.001,
            'num_workers': 0,
            'seed': 0,
            'gpu_index': 0,
            'analysis_checkpoint_epochs': [1],
            'checkpoint_root': str(tmp_path / dataset / 'ckpt'),
            'metric_root': str(tmp_path / dataset / 'metric'),
            'ddp': False,
            'ddp_world_size': 2,
            'batch_size_is_global': True,
            'signal_window': 'hann',
            'compile': False,
            'amp': 'off',
            'run_timestamp': f'TEST_{dataset}',
        }
    }
    cfg_path = tmp_path / f'{dataset}.yaml'
    save_yaml(cfg_path, cfg)
    assert model_training_main(['--config', str(cfg_path)]) == 0
    assert (tmp_path / dataset / 'ckpt' / f'run_TEST_{dataset}' / 'checkpoint_epoch_000001.pt').exists()
    assert (tmp_path / dataset / 'metric' / f'run_TEST_{dataset}' / 'training_metrics.csv').exists()


def test_image_temporal_datasets_select_flattened_view_for_dense_models():
    manifest = {
        'psd_axis_kind': 'image_temporal',
        'available_views': ['model_input', 'original_input', 'flatten_input', 'sequence_input'],
    }
    assert _select_image_training_view(manifest, model_family='lif', default_view='model_input') == 'sequence_input'
    assert _select_image_training_view(manifest, model_family='rf', default_view='model_input') == 'sequence_input'
    assert _select_image_training_view(manifest, model_family='cnn_lif', default_view='model_input') == 'model_input'


def test_image_input_shape_metadata_is_only_passed_to_frame_consuming_models():
    from src.model_training import _bundle_input_shape

    manifest = {
        'psd_axis_kind': 'image_temporal',
        'stored_shape': [4, 2, 16, 16],
        'cnn_input_shape': [4, 2, 16, 16],
    }
    assert _bundle_input_shape(object(), model_family='lif', manifest=manifest) is None
    assert _bundle_input_shape(object(), model_family='spikegru', manifest=manifest) is None
    assert _bundle_input_shape(object(), model_family='cnn_lif', manifest=manifest) == [4, 2, 16, 16]
    assert _bundle_input_shape(object(), model_family='spikformer', manifest=manifest) == [4, 2, 16, 16]
