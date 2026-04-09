"""dataset_psd entry script for dataset/probe-set PSD reference generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.registry import build_dataset_bundle, extract_labels
from src.plot.plotting import flush_plot_tasks, shutdown_plot_worker
from src.signal.psd_artifacts import combined_exact_psd_payload_from_maps_torch, save_psd_bundle
from src.stat.probe_set import build_canonical_label_order, select_probe_sets
from src.util.seed import make_worker_init_fn, set_global_seed


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dataset_psd experiment."""

    p = argparse.ArgumentParser(description="dataset_psd")
    p.add_argument("--dataset", default="s-mnist")
    p.add_argument("--plot_target", choices=["dataset", "probe_set", "both"], default="both")
    p.add_argument("--out_root", default="outputs/dataset_psd")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--psd_window", type=int, default=64)
    p.add_argument("--psd_overlap", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--same_label_n_per_label", type=int, default=8)
    p.add_argument("--balanced_global_n_per_label", type=int, default=8)
    p.add_argument("--use_torchvision", action="store_true")
    p.add_argument("--userbin_edges", nargs="*", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    return p.parse_args()


def _loader_to_maps(loader: DataLoader) -> np.ndarray:
    """Collect input maps in (S,R,T) format from dataloader."""

    maps = []
    for x, _ in loader:
        maps.append(x.permute(0, 2, 1).numpy())
    return np.concatenate(maps, axis=0)


def _save_input_bundle(loader: DataLoader, out_dir: Path, userbin_edges, psd_window: int, psd_overlap: int) -> None:
    payload = combined_exact_psd_payload_from_maps_torch(
        _loader_to_maps(loader),
        userbin_edges,
        psd_window,
        psd_overlap,
    )
    save_psd_bundle(payload, out_dir, save_db_plots=True, save_summary_json=True)


def _make_loader(ds, batch_size: int, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=make_worker_init_fn(seed),
        generator=generator,
    )


def main() -> None:
    """Run dataset input PSD reference saving workflow."""

    args = parse_args()
    set_global_seed(args.seed, deterministic=True)

    bundle = build_dataset_bundle(args.dataset, seed=args.seed, use_torchvision=args.use_torchvision)
    root = Path(args.out_root) / bundle.name
    root.mkdir(parents=True, exist_ok=True)

    split_datasets = {"train": bundle.train, "test": bundle.test}

    if args.plot_target in ("dataset", "both"):
        for split, ds in split_datasets.items():
            loader = _make_loader(ds, batch_size=args.batch_size, seed=args.seed)
            _save_input_bundle(
                loader,
                root / "dataset" / split / "input",
                args.userbin_edges,
                args.psd_window,
                args.psd_overlap,
            )

    if args.plot_target in ("probe_set", "both"):
        for split, ds in split_datasets.items():
            labels = extract_labels(ds)
            canonical = build_canonical_label_order(range(len(ds)), labels.tolist(), args.seed, split=split)
            probe = select_probe_sets(
                canonical,
                same_label_n=args.same_label_n_per_label,
                balanced_n=args.balanced_global_n_per_label,
            )

            # same_label probes
            for label, indices in probe.same_label.items():
                subset = Subset(ds, indices)
                loader = _make_loader(subset, batch_size=args.batch_size, seed=args.seed)
                _save_input_bundle(
                    loader,
                    root / "probe_set_reference" / split / "same_label" / f"label_{label}" / "input",
                    args.userbin_edges,
                    args.psd_window,
                    args.psd_overlap,
                )

            # balanced_global probes
            subset = Subset(ds, probe.balanced_global)
            loader = _make_loader(subset, batch_size=args.batch_size, seed=args.seed)
            _save_input_bundle(
                loader,
                root / "probe_set_reference" / split / "balanced_global" / "input",
                args.userbin_edges,
                args.psd_window,
                args.psd_overlap,
            )

    flush_plot_tasks()
    shutdown_plot_worker(wait=True)


if __name__ == "__main__":
    main()
