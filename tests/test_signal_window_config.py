import json
from pathlib import Path

from src.util.config import save_yaml

import torch

from src.signal.psd_utils import exact_periodogram_from_maps, one_sided_scaling, normalize_signal_window
from src.util.config_cli import parse_args_with_config
from src.model_training import build_arg_parser


def test_signal_window_none_uses_rectangular_fft_power():
    x = torch.linspace(-1.0, 1.0, steps=8, dtype=torch.float32).view(1, 1, 8)
    _freqs, psd = exact_periodogram_from_maps(x, signal_window='none')
    x64 = x.to(dtype=psd.dtype)
    spectrum = torch.fft.rfft(x64, dim=-1)
    scale = one_sided_scaling(x.shape[-1], device=x.device, dtype=psd.dtype)
    expected = scale.view(1, 1, -1) * spectrum.abs().square() / float(x.shape[-1])
    assert torch.allclose(psd, expected, atol=1e-6, rtol=1e-6)


def test_signal_window_hann_differs_from_no_window_for_nontrivial_signal():
    x = torch.arange(8, dtype=torch.float32).view(1, 1, 8)
    _freqs, hann_psd = exact_periodogram_from_maps(x, signal_window='hann')
    _freqs, none_psd = exact_periodogram_from_maps(x, signal_window='none')
    assert not torch.allclose(hann_psd, none_psd)


def test_model_training_config_accepts_signal_window_without_output_root(tmp_path: Path):
    cfg = {
        'model_training': {
            'dataset': 'mnist',
            'prep_root': '/home/yongokhan/workspace/data/prep_data',
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
            'checkpoint_root': '/tmp/ckpt',
            'metric_root': '/tmp/metric',
            'signal_window': 'none',
            'compile': False,
            'amp': 'off',
        }
    }
    path = tmp_path / 'cfg.yaml'
    save_yaml(path, cfg)
    args = parse_args_with_config(build_arg_parser(), argv=['--config', str(path)], stage_key='model_training')
    assert normalize_signal_window(args.signal_window) == 'none'
    assert not hasattr(args, 'output_root')
