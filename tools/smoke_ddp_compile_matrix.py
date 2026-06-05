#!/usr/bin/env python3
"""Run or print a focused DDP + torch.compile smoke matrix.

The script intentionally consumes already prepared ``prep_data`` bundles.  It is
meant for the GPU server after data_prep has produced the canonical single-file
bundles.  Use ``--dry-run`` to inspect the generated torchrun commands without
launching training.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.model_registry import canonicalize_model_token
from src.util.config import save_yaml


DEFAULT_CASES: tuple[str, ...] = (
    # User-requested structural compile/DDP matrix.
    'dvs128-gesture:spikeformer:temporal_membrane',
    'cifar-100:spikeformer:temporal_membrane',
    'cifar10-dvs:spikeformer:temporal_membrane',
    'dvs128-gesture:resnet18_lif_soft_fixed:temporal_membrane',
    'cifar-100:resnet18_lif_soft_fixed:temporal_membrane',
    'cifar10-dvs:resnet18_lif_soft_fixed:temporal_membrane',
    'cifar-10:vgg11_lif_soft_fixed:temporal_membrane',
    'cifar10-dvs:vgg11_lif_soft_fixed:temporal_membrane',
    'n-mnist:vgg11_lif_soft_fixed:temporal_membrane',
    's-mnist:spikegru:spikegru_max_over_time',
    's-cifar10:spikegru:spikegru_max_over_time',
    'shd:spikegru:spikegru_max_over_time',
    'ssc:spikegru:spikegru_max_over_time',
)


@dataclass(frozen=True)
class SmokeCase:
    dataset: str
    model: str
    readout_mode: str

    @property
    def name(self) -> str:
        safe = f'{self.dataset}__{self.model}__{self.readout_mode}'
        return ''.join(ch if ch.isalnum() or ch in {'-', '_'} else '_' for ch in safe)


def parse_case(text: str) -> SmokeCase:
    parts = [part.strip() for part in str(text).split(':')]
    if len(parts) not in {2, 3} or not all(parts):
        raise argparse.ArgumentTypeError('case must be DATASET:MODEL[:READOUT_MODE]')
    dataset, model = parts[0], parts[1]
    if len(parts) == 3:
        readout = parts[2]
    else:
        readout = 'spikegru_max_over_time' if model.lower() == 'spikegru' else 'temporal_membrane'
    return SmokeCase(dataset=dataset, model=model, readout_mode=readout)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='DDP + torch.compile smoke matrix for prepared PSD datasets/models.')
    parser.add_argument('--prep-root', required=True, help='Prepared data root containing dataset subdirectories.')
    parser.add_argument('--output-root', default='smoke/ddp_compile_matrix', help='Directory for generated configs, logs, metrics, checkpoints.')
    parser.add_argument('--compile-cache-root', default='cache/torch_compile_smoke', help='Root for per-case torch.compile cache namespaces.')
    parser.add_argument('--experiment-name', default='ddp_compile_matrix', help='Experiment namespace under --compile-cache-root.')
    parser.add_argument('--case', action='append', type=parse_case, help='One DATASET:MODEL[:READOUT_MODE] case. May be repeated.')
    parser.add_argument('--nproc-per-node', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden-spec', default='4')
    parser.add_argument('--compile-cpu-threads', type=int, default=2)
    parser.add_argument('--torchrun', default='torchrun', help='torchrun executable path.')
    parser.add_argument('--dry-run', action='store_true', help='Print commands/config paths without launching.')
    return parser


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _explicit_model_fields(token: str) -> dict[str, object]:
    spec = canonicalize_model_token(token)
    reset = None
    if spec.reset_mode == 'soft_reset':
        reset = 'soft'
    elif spec.reset_mode == 'hard_reset':
        reset = 'hard'
    elif spec.reset_mode == 'no_reset':
        reset = 'none'
    if spec.family in {'if', 'lif', 'rf'}:
        neuron_type = spec.family
    elif spec.family == 'tc_lif':
        neuron_type = 'tc'
    elif spec.family == 'ts_lif':
        neuron_type = 'ts'
    elif spec.family == 'dh_snn':
        neuron_type = f'dh_snn_{int(spec.branch or 4)}'
    elif spec.family == 'd_rf':
        neuron_type = f'd_rf_{int(spec.branch or 4)}'
    elif spec.family == 'spikformer':
        neuron_type = 'spikeformer'
    elif spec.family == 'spikegru':
        neuron_type = 'spikegru'
    elif spec.family == 'spikingssm':
        neuron_type = 'spikingssm'
    elif spec.family in {'cnn_lif', 'cnn_rf'}:
        backbone = str(spec.backbone or '').strip().lower() or 'cnn'
        neuron = 'lif' if spec.family == 'cnn_lif' else 'rf'
        neuron_type = f'{backbone}_{neuron}'
    else:
        neuron_type = spec.family
    return {
        'neuron_type': neuron_type,
        'recurrent': bool(spec.recurrent),
        'reset': reset,
        'v_th': ['train' if bool(spec.trainable_threshold) else 'fixed', 1.0],
        'filter': 'train',
    }


def _write_case_config(case: SmokeCase, *, args: argparse.Namespace, root: Path) -> Path:
    case_root = root / case.name
    config_dir = root / 'configs'
    config_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'model_training': {
            'dataset': case.dataset,
            'prep_root': str(Path(args.prep_root).expanduser().resolve()),
            **_explicit_model_fields(case.model),
            'hidden_spec': str(args.hidden_spec),
            'readout_mode': case.readout_mode,
            'epochs': int(args.epochs),
            'batch_size': int(args.batch_size),
            'lr': float(args.lr),
            'num_workers': int(args.num_workers),
            'seed': int(args.seed),
            'gpu_index': 0,
            'analysis_checkpoint_epochs': [int(args.epochs)],
            'checkpoint_root': str((case_root / 'checkpoints').resolve()),
            'metric_root': str((case_root / 'metrics').resolve()),
            'ddp': True,
            'ddp_world_size': int(args.nproc_per_node),
            'batch_size_is_global': True,
            'signal_curve_space': 'exact',
            'signal_curve_scale': 'raw',
            'signal_curve_centering': 'raw',
            'signal_curve_reducer': 'mean',
            'signal_curve_distance_metric': 'centered_l2',
            'signal_curve_userbin_edges': None,
            'signal_curve_userbin_reducer': 'mean',
            'lambda_psd_rep_input': 0.0,
            'lambda_psd_rep_adjacent': 0.0,
            'lambda_psd_pca_input': 0.0,
            'lambda_psd_pca_adjacent': 0.0,
            'pca_dim_per_layer': [1],
            'signal_window': 'hann',
            'compile': True,
            'compile_cpu_threads': int(args.compile_cpu_threads),
            'amp': 'off',
            'run_timestamp': f'SMOKE_{case.name}',
        }
    }
    path = config_dir / f'{case.name}.yaml'
    save_yaml(path, payload)
    return path


def _command_for_case(case: SmokeCase, *, args: argparse.Namespace, config_path: Path) -> tuple[dict[str, str], list[str]]:
    cache_dir = Path(args.compile_cache_root).expanduser().resolve() / str(args.experiment_name) / case.name
    env = {'PSD_TORCH_COMPILE_CACHE_DIR': str(cache_dir)}
    cmd = [
        str(args.torchrun),
        '--standalone',
        f'--nproc_per_node={int(args.nproc_per_node)}',
        'src/model_training.py',
        '--config',
        str(config_path),
        '--ddp',
        'true',
    ]
    return env, cmd


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    cases = tuple(args.case) if args.case else tuple(parse_case(raw) for raw in DEFAULT_CASES)
    output_root = Path(args.output_root).expanduser().resolve()
    logs_root = output_root / 'logs'
    logs_root.mkdir(parents=True, exist_ok=True)
    project_root = _project_root()

    summary: list[dict[str, object]] = []
    for case in cases:
        config_path = _write_case_config(case, args=args, root=output_root)
        env_delta, cmd = _command_for_case(case, args=args, config_path=config_path)
        log_path = logs_root / f'{case.name}.log'
        item = {
            'case': case.name,
            'dataset': case.dataset,
            **_explicit_model_fields(case.model),
            'readout_mode': case.readout_mode,
            'config': str(config_path),
            'log': str(log_path),
            'env': env_delta,
            'cmd': cmd,
        }
        print(json.dumps(item, ensure_ascii=False), flush=True)
        summary.append(item)
        if args.dry_run:
            continue
        env = os.environ.copy()
        env.update(env_delta)
        with log_path.open('w', encoding='utf-8') as log_file:
            proc = subprocess.run(cmd, cwd=str(project_root), env=env, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            raise SystemExit(f'case {case.name} failed with exit code {proc.returncode}. See {log_path}')

    manifest_path = output_root / 'smoke_matrix_manifest.yaml'
    save_yaml(manifest_path, summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
