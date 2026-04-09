"""Main orchestration logic for psd_analysis experiment."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.registry import build_dataset_bundle, extract_labels
from src.model.first_spike_loss import FirstSpikeCriterion
from src.model.psd_training import train_for_psd
from src.model.snn_builder import build_snn_classifier
from src.plot.deferred_plot_tasks import add_deferred_plot_task, render_deferred_plot_tasks
from src.plot.plotting import flush_plot_tasks, save_lineplot, shutdown_plot_worker
from src.readout.readout import apply_readout
from src.signal.psd_artifacts import combined_exact_psd_payload_from_maps_torch, save_psd_bundle
from src.stat.probe_set import build_canonical_label_order, select_probe_sets
from src.util.seed import make_worker_init_fn, set_global_seed
from .io_paths import build_run_root


@dataclass
class PSDAnalysisArgs:
    """Argument bundle consumed by run_psd_analysis."""

    dataset: str
    model: str
    readout_mode: str
    out_root: str
    epochs: int = 3
    batch_size: int = 32
    hidden_size: int = 64
    psd_window: int = 64
    psd_overlap: int = 32
    userbin_edges: List[float] | None = None
    plot_epochs: List[int] | None = None
    seed: int = 42
    same_label_n_per_label: int = 8
    balanced_global_n_per_label: int = 8
    use_torchvision: bool = False


def normalize_plot_epochs(plot_epochs: Iterable[int] | None, epochs: int) -> List[int]:
    """Normalize selected plot epochs with dedupe and bounds check."""

    if epochs <= 0:
        return []
    if plot_epochs is None:
        return list(range(1, epochs + 1))
    uniq = sorted(set(int(e) for e in plot_epochs))
    for e in uniq:
        if e < 1 or e > epochs:
            raise ValueError(f"plot epoch out of range: {e}")
    return uniq


def _make_loader(ds: Dataset, batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        worker_init_fn=make_worker_init_fn(seed),
        generator=g,
    )


def _collect_maps(model, loader, readout_mode: str, device: torch.device) -> np.ndarray:
    """Collect output membrane maps in (S,R,T) for PSD payload computation."""

    maps = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            _, mem = model(x, final_membrane_disable_spike=(readout_mode == "final_membrane"))
            maps.append(mem.permute(0, 2, 1).cpu().numpy())
    return np.concatenate(maps, axis=0)


def _probe_accuracy_counts(model, loader: DataLoader, readout_mode: str, criterion, device: torch.device) -> Tuple[int, int]:
    """Compute probe-set accuracy with criterion-consistent prediction rule."""

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            spikes, mem = model(x, final_membrane_disable_spike=(readout_mode == "final_membrane"))
            if getattr(criterion, "requires_output_record", False):
                analysis = criterion.analyze_output_record(spikes, mem)
                pred = criterion.predictions_from_analysis(analysis)
            else:
                logits = apply_readout(readout_mode, spikes, mem)
                pred = torch.argmax(logits, dim=-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return correct, total


def _probe_subsets(ds: Dataset, split: str, seed: int, same_label_n: int, balanced_n: int) -> List[Tuple[str, str | None, Dataset]]:
    """Build deterministic same_label/balanced_global probe subsets."""

    labels = extract_labels(ds)
    canonical = build_canonical_label_order(range(len(ds)), labels.tolist(), seed=seed, split=split)
    selected = select_probe_sets(canonical, same_label_n=same_label_n, balanced_n=balanced_n)

    outputs: List[Tuple[str, str | None, Dataset]] = []
    for label, indices in selected.same_label.items():
        outputs.append(("same_label", str(label), Subset(ds, indices)))
    outputs.append(("balanced_global", None, Subset(ds, selected.balanced_global)))
    return outputs


def _save_probe_accuracy(path: Path, epoch: int, split: str, probe_type: str, label: str | None, correct: int, total: int) -> None:
    """Write probe_set_accuracy.txt with required human-readable fields."""

    acc = float(correct) / float(max(total, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"epoch: {epoch}",
                f"split: {split}",
                f"probe_type: {probe_type}",
                f"label: {label if label is not None else 'none'}",
                f"correct: {correct}",
                f"total: {total}",
                f"accuracy: {acc:.6f}",
            ]
        ),
        encoding="utf-8",
    )


def run_psd_analysis(args: PSDAnalysisArgs) -> Path:
    """Run full psd_analysis orchestration and return run root path."""

    set_global_seed(args.seed, deterministic=True)

    bundle = build_dataset_bundle(args.dataset, seed=args.seed, use_torchvision=args.use_torchvision)
    run_root = build_run_root(args.out_root, bundle.name, args.model, args.readout_mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = _make_loader(bundle.train, batch_size=args.batch_size, seed=args.seed, shuffle=True)
    test_loader = _make_loader(bundle.test, batch_size=args.batch_size, seed=args.seed + 1, shuffle=False)

    model_mode = "lif" if "lif" in args.model else "rf"
    model = build_snn_classifier(bundle.input_channels, args.hidden_size, bundle.num_classes, mode=model_mode)
    criterion = FirstSpikeCriterion() if args.readout_mode == "first_spike" else torch.nn.CrossEntropyLoss()

    history = []
    selected_epochs = normalize_plot_epochs(args.plot_epochs, args.epochs)
    if args.epochs > 0:
        history = train_for_psd(model, train_loader, test_loader, args.readout_mode, args.epochs, criterion, device)

    userbin = args.userbin_edges or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    for epoch in selected_epochs:
        for split_name, ds in (("train", bundle.train), ("test", bundle.test)):
            for probe_type, label, subset in _probe_subsets(
                ds,
                split=split_name,
                seed=args.seed,
                same_label_n=args.same_label_n_per_label,
                balanced_n=args.balanced_global_n_per_label,
            ):
                loader = _make_loader(subset, batch_size=args.batch_size, seed=args.seed, shuffle=False)
                maps = _collect_maps(model, loader, args.readout_mode, device)
                payload = combined_exact_psd_payload_from_maps_torch(maps, userbin, args.psd_window, args.psd_overlap)

                scope_dir = run_root / f"epoch_{epoch:04d}" / split_name / probe_type
                if label is not None:
                    scope_dir = scope_dir / f"label_{label}"
                out_dir = scope_dir / "output" / "membrane"
                add_deferred_plot_task(save_psd_bundle, payload, out_dir, True, True)

                correct, total = _probe_accuracy_counts(model, loader, args.readout_mode, criterion, device)
                _save_probe_accuracy(
                    scope_dir / "probe_set_accuracy.txt",
                    epoch=epoch,
                    split=split_name,
                    probe_type=probe_type,
                    label=label,
                    correct=correct,
                    total=total,
                )

    rendered = render_deferred_plot_tasks()
    (run_root / "rendered_plot_tasks.txt").write_text(str(rendered), encoding="utf-8")

    acc_csv = run_root / "train_test_accuracy.csv"
    with acc_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_accuracy", "test_accuracy"])
        for row in history:
            writer.writerow([row.epoch, row.train_accuracy, row.test_accuracy])

    if history:
        save_lineplot(
            np.asarray([h.epoch for h in history], dtype=np.float64),
            np.asarray([h.test_accuracy for h in history], dtype=np.float64),
            run_root / "train_test_accuracy.png",
            title="test accuracy",
        )

    training_complete = run_root / "training_complete_stats"
    training_complete.mkdir(parents=True, exist_ok=True)
    (training_complete / "train_test_accuracy.csv").write_text(acc_csv.read_text(encoding="utf-8"), encoding="utf-8")
    if (run_root / "train_test_accuracy.png").exists():
        (training_complete / "train_test_accuracy.png").write_bytes((run_root / "train_test_accuracy.png").read_bytes())

    for split_name, ds in (("train", bundle.train), ("test", bundle.test)):
        for probe_type, label, subset in _probe_subsets(
            ds,
            split=split_name,
            seed=args.seed,
            same_label_n=args.same_label_n_per_label,
            balanced_n=args.balanced_global_n_per_label,
        ):
            loader = _make_loader(subset, batch_size=args.batch_size, seed=args.seed, shuffle=False)
            correct, total = _probe_accuracy_counts(model, loader, args.readout_mode, criterion, device)
            out = training_complete / "probe_set_accuracy" / split_name / probe_type
            if label is not None:
                out = out / f"label_{label}"
            _save_probe_accuracy(
                out / "probe_set_accuracy.txt",
                epoch=(history[-1].epoch if history else 0),
                split=split_name,
                probe_type=probe_type,
                label=label,
                correct=correct,
                total=total,
            )

    flush_plot_tasks()
    shutdown_plot_worker(wait=True)
    return run_root
