from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.datasets import dataset_choices
from src.util.psd_analysis_driver import (
    _parse_model_token_recurrence_and_branch,
    run_psd_analysis,
)
from src.model.psd_model_variants import parse_psd_model_variant, validate_lif_clip_edges, validate_rf_clip_edges
from src.readout.readout import READOUT_CHOICES, normalize_readout_mode


def _parse_bool01(text: str) -> bool:
    key = str(text).strip().lower()
    if key in ('1', 'true', 't', 'yes', 'y'):
        return True
    if key in ('0', 'false', 'f', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError(f'Expected 0/1 or boolean text, got: {text}')


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def _normalize_dataset_token(text: str) -> str:
    key = str(text).strip()
    lowered = key.lower().replace(' ', '').replace('/', '').replace('\\', '')
    for candidate in dataset_choices():
        if lowered == candidate.replace('-', '').replace('_', ''):
            return candidate
    return key


def _normalize_model_token(text: str) -> str:
    key = str(text).strip().lower()
    try:
        # The shared parser validates recurrent suffixes like ``_R`` and
        # ``_R_<int>`` together with the legacy fixed-branch suffix ``_<int>``.
        _parse_model_token_recurrence_and_branch(key)
    except Exception as exc:  # pragma: no cover - argparse surface only
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return key


def _validate_model_specific_args(parser: argparse.ArgumentParser, args: argparse.Namespace, models: list[str]) -> None:
    needs_rf_clip = False
    needs_lif_clip = False
    for model in models:
        base_token, _, _ = _parse_model_token_recurrence_and_branch(str(model))
        variant = parse_psd_model_variant(str(base_token))
        if variant is None or not bool(variant.clip_params):
            continue
        if str(variant.base_model) == 'RF':
            needs_rf_clip = True
        elif str(variant.base_model) == 'LIF':
            needs_lif_clip = True

    if needs_rf_clip:
        if args.w_clip_edges is None or len(args.w_clip_edges) == 0:
            parser.error('Selected model list includes rf_clip/rf_structclip, so --w_clip_edges must be provided.')
        try:
            validate_rf_clip_edges([float(v) for v in args.w_clip_edges])
        except ValueError as exc:
            parser.error(str(exc))

    if needs_lif_clip:
        if args.alpha_clip_edges is None or len(args.alpha_clip_edges) == 0:
            parser.error('Selected model list includes lif_clip/lif_structclip, so --alpha_clip_edges must be provided.')
        try:
            validate_lif_clip_edges([float(v) for v in args.alpha_clip_edges])
        except ValueError as exc:
            parser.error(str(exc))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            'Run PSD analysis for a single dataset and one or more model/readout combinations. '
            'The runner executes model × readout combinations serially.'
        )
    )
    p.add_argument('--dataset', type=_normalize_dataset_token, required=True, help='Single dataset token. Supported canonical names include: s-mnist, dvsgesture, shd, deap, forda.')
    p.add_argument('--model', nargs='+', type=_normalize_model_token, required=True, help='One or more model tokens. Recurrent variants may use suffixes like lif_R, lif_struct_R, dh_snn_R, or dh_snn_R_4.')
    p.add_argument('--readout_mode', nargs='+', type=normalize_readout_mode, default=['final_membrane'], help='One or more readout modes to run serially.')
    p.add_argument('--out_root', type=str, required=True, help='Absolute result root.')
    p.add_argument('--data_root', type=str, required=True, help='Absolute external data root containing dataset subdirectories.')
    p.add_argument('--exp_name', type=str, default=None, help='Optional experiment-name prefix.')
    p.add_argument('--timestamp', type=str, default=None, help='Optional timestamp suffix. If omitted, current Asia/Seoul time is used.')
    p.add_argument('--gpu', type=int, default=0, help='CUDA device index.')
    p.add_argument('--device', type=str, default=None, help='Optional explicit torch device string. Defaults to cuda:<gpu>. Explicit cpu is intended for debug / smoke tests; the main target remains CUDA.')
    p.add_argument('--seed', type=int, default=0, help='Global random seed.')
    p.add_argument('--hidden', nargs='+', type=int, required=True, help='Hidden-layer widths. Example: --hidden 256 256')
    p.add_argument('--epochs', type=int, default=50, help='Number of training epochs. Set 0 to skip training and epoch-wise artifact generation.')
    p.add_argument('--soft_mask_epochs', type=int, default=None, help='Optional soft-mask phase length.')
    p.add_argument('--stabilize_epochs', type=int, default=0, help='Structure stabilization phase length.')
    p.add_argument('--ste_epochs', type=int, default=0, help='STE / hardened phase length.')
    p.add_argument('--batch_size', type=int, default=128, help='Training and evaluation mini-batch size.')
    p.add_argument('--lr', type=float, default=1e-3, help='AdamW learning rate.')
    p.add_argument('--weight_decay', type=float, default=0.0, help='Standard AdamW weight decay.')
    p.add_argument('--weight_decay_dend_soma', type=float, default=None, help='Optional separate weight decay for dendrite/soma parameters.')
    p.add_argument('--S_min', type=float, default=1.0, help='Lower bound of the variable-branch range used by my_* models.')
    p.add_argument('--S_max', type=float, default=8.0, help='Upper bound of the variable-branch range used by my_* models.')
    p.add_argument('--th_len', type=int, default=4, help='Surrogate-gradient window length / threshold span used by common builders.')
    p.add_argument('--v_th', type=float, default=1.0, help='Neuron firing threshold.')
    p.add_argument('--v_pre', type=float, default=1.0, help='Pre-spike scaling / normalization constant used by applicable builders.')
    p.add_argument('--num_workers', type=int, default=4, help='PyTorch DataLoader worker count.')
    p.add_argument('--download', type=_parse_bool01, default=False, help='Allow automatic dataset download when supported.')
    p.add_argument('--shd_T', type=int, default=250, help='SHD number of time bins. Keep 250 for the standard setup.')
    p.add_argument('--shd_max_time', type=float, default=1.0, help='SHD window length in seconds.')
    p.add_argument('--shd_binning', type=str, default='origin', choices=['origin', 'floor'], help='SHD event-to-bin rule.')
    p.add_argument('--shd_unit_indexing', type=str, default='auto', help='SHD unit-index mode: auto, 0, or 1.')
    p.add_argument('--shd_channel_flip', type=_parse_bool01, default=True, help='Reverse SHD channel order.')
    p.add_argument('--shd_align_to_first_event', type=_parse_bool01, default=False, help='Shift each SHD sample so its first event starts at t=0.')
    p.add_argument('--shd_use_event_counts', type=_parse_bool01, default=False, help='Use SHD event counts instead of binary occupancy inside each time bin.')
    p.add_argument('--dvsgesture_chunk_size', type=int, default=120, help='DVS128 Gesture event chunk length before optional empty padding.')
    p.add_argument('--dvsgesture_empty_size', type=int, default=40, help='DVS128 Gesture zero-padding length appended after the cropped event chunk.')
    p.add_argument('--dvsgesture_dt_ms', type=float, default=10.0, help='DVS128 Gesture temporal bin width in milliseconds.')
    p.add_argument('--dvsgesture_ds', type=int, default=4, help='DVS128 Gesture spatial downsampling factor.')
    p.add_argument('--deap_label_axis', type=int, default=0, help='DEAP label axis. 0=valence, 1=arousal.')
    p.add_argument('--deap_num_classes', type=int, default=3, help='DEAP number of quantized classes. Supported values: 2 or 3.')
    p.add_argument('--lambda_ortho', type=float, default=0.0, help='Optional orthogonality regularization weight.')
    p.add_argument('--lambda_s', type=float, default=0.0, help='Optional structure regularization weight.')
    p.add_argument('--same_label_n_per_label', type=int, default=4, help='Probe-set prefix length per label for the same_label scope.')
    p.add_argument('--balanced_global_n_per_label', type=int, default=4, help='Probe-set prefix length per label for the balanced_global scope.')
    p.add_argument('--plot_epoch', '--plot_epochs', nargs='+', type=int, default=None, dest='plot_epochs', help='Optional explicit epoch list for signal-PSD artifact generation.')
    p.add_argument('--psd_window', type=int, default=64, help='Frame length for the exact sliding simple-periodogram spectrogram path.')
    p.add_argument('--psd_overlap', type=int, default=32, help='Frame overlap for the exact sliding simple-periodogram spectrogram path.')
    p.add_argument('--window_fn', type=str, default='hann', choices=['hann', 'hamming', 'blackman'], help='Legacy compatibility argument. Accepted but ignored by the exact PSD path.')
    p.add_argument('--userbin_edges', nargs='*', type=float, default=None, help='Optional userbin frequency edges in cycles/sample for periodogram and spectrogram heatmap aggregation.')
    p.add_argument('--rf_reset_mode', type=str, default='no_reset', choices=['no_reset', 'soft_reset'], help='Vanilla RF reset policy.')
    p.add_argument('--w_clip_edges', nargs='*', type=float, default=None, help='Optional RF clip edges for rf_clip / rf_structclip.')
    p.add_argument('--alpha_clip_edges', nargs='*', type=float, default=None, help='Optional LIF clip edges for lif_clip / lif_structclip.')
    p.add_argument('--band_neuron_ends', nargs='*', type=str, default=None, help='Optional hidden-layer cumulative group-end specifications for structured variants.')
    p.add_argument('--tear', type=int, default=1, help='1-based hidden-layer index where structure / clip rules begin.')
    return p


def _build_exp_name(args, *, dataset: str, model: str, readout_mode: str, num_models: int, num_readouts: int) -> str | None:
    include_model = int(num_models) > 1
    include_readout = int(num_readouts) > 1
    if not include_model and not include_readout:
        return None if args.exp_name is None else str(args.exp_name)
    parts: list[str] = [str(args.exp_name) if args.exp_name is not None else 'psd_analysis']
    if args.exp_name is None:
        parts.append(str(dataset))
    if include_model or args.exp_name is None:
        parts.append(str(model))
    if include_readout:
        parts.append(str(readout_mode))
    return '_'.join(parts)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    dataset = _normalize_dataset_token(str(args.dataset))
    models = _dedupe_preserve_order([_normalize_model_token(str(model)) for model in list(args.model)])
    modes = _dedupe_preserve_order([normalize_readout_mode(str(mode)) for mode in list(args.readout_mode)])
    _validate_model_specific_args(parser, args, models)
    for model in models:
        for mode in modes:
            exp_name = _build_exp_name(
                args,
                dataset=str(dataset),
                model=str(model),
                readout_mode=str(mode),
                num_models=len(models),
                num_readouts=len(modes),
            )
            run_root = run_psd_analysis(
                dataset=str(dataset),
                model=str(model),
                out_root=str(args.out_root),
                data_root=str(args.data_root),
                hidden=[int(h) for h in args.hidden],
                epochs=int(args.epochs),
                soft_mask_epochs=None if args.soft_mask_epochs is None else int(args.soft_mask_epochs),
                stabilize_epochs=int(args.stabilize_epochs),
                ste_epochs=int(args.ste_epochs),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                weight_decay_dend_soma=None if args.weight_decay_dend_soma is None else float(args.weight_decay_dend_soma),
                seed=int(args.seed),
                S_min=float(args.S_min),
                S_max=float(args.S_max),
                th_len=int(args.th_len),
                v_th=float(args.v_th),
                v_pre=float(args.v_pre),
                num_workers=int(args.num_workers),
                download=bool(args.download),
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
                lambda_ortho=float(args.lambda_ortho),
                lambda_s=float(args.lambda_s),
                same_label_n_per_label=int(args.same_label_n_per_label),
                balanced_global_n_per_label=int(args.balanced_global_n_per_label),
                probe_plot=False,
                plot_epochs=None if args.plot_epochs is None else [int(v) for v in args.plot_epochs],
                psd_window=int(args.psd_window),
                psd_overlap=int(args.psd_overlap),
                window_fn=str(args.window_fn),
                userbin_edges=None if args.userbin_edges is None or len(args.userbin_edges) == 0 else [float(v) for v in args.userbin_edges],
                rf_reset_mode=str(args.rf_reset_mode),
                w_clip_edges=None if args.w_clip_edges is None or len(args.w_clip_edges) == 0 else [float(v) for v in args.w_clip_edges],
                alpha_clip_edges=None if args.alpha_clip_edges is None or len(args.alpha_clip_edges) == 0 else [float(v) for v in args.alpha_clip_edges],
                band_neuron_ends=None if args.band_neuron_ends is None or len(args.band_neuron_ends) == 0 else [str(v) for v in args.band_neuron_ends],
                tear=int(args.tear),
                readout_mode=str(mode),
                exp_name=exp_name,
                timestamp=None if args.timestamp is None else str(args.timestamp),
                device=str(args.device) if args.device is not None else f'cuda:{int(args.gpu)}',
            )
            print(run_root)


if __name__ == '__main__':
    main()
