from __future__ import annotations

"""CLI entry point for dataset-level PSD reference generation."""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.util.cli_common import normalize_dataset_token, parse_bool01
from src.util.dataset_psd_driver import run_dataset_psd
from src.util.psd_run_configs import DatasetPSDRunConfig


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser for ``dataset_psd``."""
    p = argparse.ArgumentParser(
        description=(
            'Compute train/test input PSD references for a single dataset. '
            'The same deterministic probe-set reference can also be saved here.'
        )
    )
    p.add_argument('--dataset', type=normalize_dataset_token, required=True, help='Single dataset token. Supported canonical names include: s-mnist, dvsgesture, shd, deap, forda.')
    p.add_argument('--data_root', type=str, required=True, help='Absolute external data root containing dataset subdirectories.')
    p.add_argument('--out_root', type=str, required=True, help='Absolute result root.')
    p.add_argument('--gpu', type=int, default=0, help='CUDA device index used for PSD computation.')
    p.add_argument('--device', type=str, default=None, help='Optional explicit torch device string. Defaults to cuda:<gpu>. Explicit cpu is intended for debug / smoke tests; the main target remains CUDA.')
    p.add_argument('--seed', type=int, default=0, help='Optional data-loader seed for deterministic dataset adapters and probe-set selection.')
    p.add_argument('--batch_size', type=int, default=256, help='Split-processing batch size used when aggregating PSD payloads.')
    p.add_argument('--num_workers', type=int, default=4, help='PyTorch DataLoader worker count used by dataset adapters.')
    p.add_argument('--download', type=parse_bool01, default=False, help='Allow automatic dataset download when supported.')
    p.add_argument('--shd_T', type=int, default=250, help='SHD number of time bins. Keep 250 for the standard setup.')
    p.add_argument('--shd_max_time', type=float, default=1.0, help='SHD window length in seconds.')
    p.add_argument('--shd_binning', type=str, default='origin', choices=['origin', 'floor'], help='SHD event-to-bin rule.')
    p.add_argument('--shd_unit_indexing', type=str, default='auto', help='SHD unit-index mode: auto, 0, or 1.')
    p.add_argument('--shd_channel_flip', type=parse_bool01, default=True, help='Reverse SHD channel order.')
    p.add_argument('--shd_align_to_first_event', type=parse_bool01, default=False, help='Shift each SHD sample so its first event starts at t=0.')
    p.add_argument('--shd_use_event_counts', type=parse_bool01, default=False, help='Use SHD event counts instead of binary occupancy inside each time bin.')
    p.add_argument('--dvsgesture_chunk_size', type=int, default=120, help='DVS128 Gesture event chunk length before optional empty padding.')
    p.add_argument('--dvsgesture_empty_size', type=int, default=40, help='DVS128 Gesture zero-padding length appended after the cropped event chunk.')
    p.add_argument('--dvsgesture_dt_ms', type=float, default=10.0, help='DVS128 Gesture temporal bin width in milliseconds.')
    p.add_argument('--dvsgesture_ds', type=int, default=4, help='DVS128 Gesture spatial downsampling factor.')
    p.add_argument('--deap_label_axis', type=int, default=0, help='DEAP label axis. 0=valence, 1=arousal.')
    p.add_argument('--deap_num_classes', type=int, default=3, help='DEAP number of quantized classes. Supported values: 2 or 3.')
    p.add_argument('--psd_window', type=int, default=64, help='Frame length for the exact sliding simple-periodogram spectrogram path.')
    p.add_argument('--psd_overlap', type=int, default=32, help='Frame overlap for the exact sliding simple-periodogram spectrogram path.')
    p.add_argument('--window_fn', type=str, default='hann', choices=['hann', 'hamming', 'blackman'], help='Legacy compatibility argument. Accepted but ignored by the exact PSD path.')
    p.add_argument('--userbin_edges', nargs='*', type=float, default=None, help='Optional userbin frequency edges in cycles/sample for periodogram and spectrogram heatmap aggregation.')
    p.add_argument('--same_label_n_per_label', type=int, default=4, help='Probe-set prefix length per label for the same_label scope.')
    p.add_argument('--balanced_global_n_per_label', type=int, default=4, help='Probe-set prefix length per label for the balanced_global scope.')
    p.add_argument('--probe_plot', type=parse_bool01, default=True, help='Whether to save deterministic probe-set input references under probe_set_reference/.')
    p.add_argument('--max_samples', type=int, default=0, help='Optional per-split cap on the number of samples to analyze. Use 0 for the full train/test sets.')
    p.add_argument('--exp_name', type=str, default=None, help='Optional experiment-name prefix.')
    p.add_argument('--timestamp', type=str, default=None, help='Optional timestamp suffix. If omitted, current Asia/Seoul time is used.')
    return p


def config_from_args(args: argparse.Namespace) -> DatasetPSDRunConfig:
    """Convert parsed CLI arguments into one normalized run-config object."""
    return DatasetPSDRunConfig(
        dataset=str(normalize_dataset_token(str(args.dataset))),
        data_root=str(args.data_root),
        out_root=str(args.out_root),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        download=bool(args.download),
        seed=int(args.seed),
        psd_window=int(args.psd_window),
        psd_overlap=int(args.psd_overlap),
        window_fn=str(args.window_fn),
        userbin_edges=None if args.userbin_edges is None or len(args.userbin_edges) == 0 else [float(v) for v in args.userbin_edges],
        same_label_n_per_label=int(args.same_label_n_per_label),
        balanced_global_n_per_label=int(args.balanced_global_n_per_label),
        probe_plot=bool(args.probe_plot),
        max_samples=None if int(args.max_samples) <= 0 else int(args.max_samples),
        exp_name=None if args.exp_name is None else str(args.exp_name),
        timestamp=None if args.timestamp is None else str(args.timestamp),
        device=str(args.device) if args.device is not None else f'cuda:{int(args.gpu)}',
        shd_T=int(args.shd_T),
        shd_max_time=float(args.shd_max_time),
        shd_binning=str(args.shd_binning),
        shd_unit_indexing=str(args.shd_unit_indexing),
        shd_channel_flip=bool(args.shd_channel_flip),
        shd_align_to_first_event=bool(args.shd_align_to_first_event),
        shd_use_event_counts=bool(args.shd_use_event_counts),
        dvsgesture_chunk_size=int(args.dvsgesture_chunk_size),
        dvsgesture_empty_size=int(args.dvsgesture_empty_size),
        dvsgesture_dt_ms=float(args.dvsgesture_dt_ms),
        dvsgesture_ds=int(args.dvsgesture_ds),
        deap_label_axis=int(args.deap_label_axis),
        deap_num_classes=int(args.deap_num_classes),
    )


def main() -> None:
    """Parse CLI arguments, run the dataset PSD pipeline, and print the run root."""
    config = config_from_args(build_parser().parse_args())
    run_root = run_dataset_psd(**config.to_kwargs())
    print(run_root)


if __name__ == '__main__':
    main()
