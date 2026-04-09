"""psd_analysis entry script for model PSD experiment."""

from __future__ import annotations

import argparse

from src.util.psd_analysis_driver import PSDAnalysisArgs, run_psd_analysis


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for psd_analysis experiment."""

    p = argparse.ArgumentParser(description="psd_analysis")
    p.add_argument("--dataset", default="s-mnist")
    p.add_argument("--model", nargs="+", default=["lif"])
    p.add_argument("--readout_mode", nargs="+", default=["final_membrane"])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--out_root", default="outputs/psd_analysis")
    p.add_argument("--psd_window", type=int, default=64)
    p.add_argument("--psd_overlap", type=int, default=32)
    p.add_argument("--window_fn", default="legacy_ignored")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--same_label_n_per_label", type=int, default=8)
    p.add_argument("--balanced_global_n_per_label", type=int, default=8)
    p.add_argument("--use_torchvision", action="store_true")
    p.add_argument("--userbin_edges", nargs="*", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    p.add_argument("--plot_epoch", "--plot_epochs", nargs="*", type=int, default=None)
    return p.parse_args()


def main() -> None:
    """Run cartesian model/readout serial experiments."""

    args = parse_args()
    for model in args.model:
        for readout in args.readout_mode:
            run_psd_analysis(
                PSDAnalysisArgs(
                    dataset=args.dataset,
                    model=model,
                    readout_mode=readout,
                    out_root=args.out_root,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    hidden_size=args.hidden_size,
                    psd_window=args.psd_window,
                    psd_overlap=args.psd_overlap,
                    userbin_edges=args.userbin_edges,
                    plot_epochs=args.plot_epoch,
                    seed=args.seed,
                    same_label_n_per_label=args.same_label_n_per_label,
                    balanced_global_n_per_label=args.balanced_global_n_per_label,
                    use_torchvision=args.use_torchvision,
                )
            )


if __name__ == "__main__":
    main()
