#!/usr/bin/env python3
"""
Checkpoint-only train/test accuracy evaluator and plotter.

이 스크립트는 이미 만들어진 `.pt` 체크포인트 폴더를 대상으로 한다.
학습은 전혀 수행하지 않고, 각 체크포인트를 로드한 뒤 train/test split에 대해
추론 평가만 수행해서 정확도 CSV와 정확도 플롯을 저장한다.

권장 위치:
    <project_root>/src/checkpoint_accuracy_eval_plot.py

권장 실행:
    cd <project_root>
    python -m src.checkpoint_accuracy_eval_plot \
      --checkpoint /abs/path/to/checkpoints/<case_id> \
      --prep_root /abs/path/to/prepared_data_root \
      --output_root /abs/path/to/output_accuracy_plot \
      --batch_size 256 \
      --gpu_index 0 \
      --overwrite

직접 실행도 가능:
    python src/checkpoint_accuracy_eval_plot.py ...
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

# `python src/checkpoint_accuracy_eval_plot.py`로 직접 실행해도
# `from src...` import가 깨지지 않도록 project root를 sys.path에 추가한다.
# 파일을 프로젝트 밖에 둔 상태로 실행할 때는 현재 작업 디렉토리가 project root이면 동작한다.
_PROJECT_ROOT_CANDIDATES = [Path(__file__).resolve().parents[1], Path.cwd().resolve()]
for _candidate in _PROJECT_ROOT_CANDIDATES:
    if (_candidate / "src").is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
        break

from src.data.registry import make_loader, resolve_dataset_bundle  # noqa: E402
from src.model.model_registry import ModelSpec, canonicalize_model_token  # noqa: E402
from src.model.snn_builder import build_snn_classifier  # noqa: E402
from src.model.training import EpochMetrics, evaluate_one_epoch  # noqa: E402
from src.readout.readout import build_readout, canonicalize_readout_mode  # noqa: E402
from src.util.config import compact_yaml, load_manifest, resolve_manifest_path, save_yaml  # noqa: E402
from src.util.csv_schema import common_row, write_common_csv  # noqa: E402
from src.util.paths import timestamped_output_root  # noqa: E402
from src.util.checkpoints import checkpoint_state_dict, load_state_dict_compatible, load_torch_checkpoint  # noqa: E402

SOURCE_PROGRAM = "checkpoint_accuracy_eval_plot"

FIGSIZE = (14, 4)          # 3.5:1
LINEWIDTH = 3.5
TITLE_SIZE = 26
LABEL_SIZE = 21
TICK_SIZE = 19
DPI = 300


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate existing checkpoints by inference only and render train/test accuracy plot."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Single .pt checkpoint file or a checkpoint directory containing .pt files for one run.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset token. If omitted, dataset_token stored in the first checkpoint is used.",
    )
    parser.add_argument(
        "--prep_root",
        default=None,
        help=(
            "Prepared data root containing <dataset>/manifest.yaml. "
            "If omitted, prep_root stored in the checkpoint is used."
        ),
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Directory where checkpoint_accuracy_metrics.csv and checkpoint_accuracy.png will be written.",
    )
    parser.add_argument("--batch_size", required=True, type=int, help="Evaluation batch size.")
    parser.add_argument(
        "--gpu_index",
        type=int,
        default=0,
        help="CUDA GPU index. Use -1 to force CPU. If CUDA is unavailable, CPU is used.",
    )
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count. Default: 0.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for DataLoader generator and basic RNGs.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "test"),
        default=("train", "test"),
        help="Which splits to evaluate. Default: train test.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="If --checkpoint is a directory, recursively search for .pt files. Use only for one run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV/PNG/manifest files in output_root.",
    )
    parser.add_argument(
        "--run_timestamp",
        default=None,
        help="Execution timestamp suffix for the output run directory. Defaults to Asia/Seoul current time.",
    )
    parser.add_argument(
        "--timestamped_output",
        default="true",
        help="true이면 output_root 아래 실행시각 폴더를 자동 생성한다. false이면 기존 경로에 직접 저장한다.",
    )
    parser.add_argument(
        "--plot_name",
        default="checkpoint_accuracy.png",
        help="Output plot file name. Default: checkpoint_accuracy.png.",
    )
    parser.add_argument(
        "--csv_name",
        default="checkpoint_accuracy_metrics.csv",
        help="Output CSV file name. Default: checkpoint_accuracy_metrics.csv.",
    )
    parser.add_argument(
        "--ylim_0_1",
        action="store_true",
        default=True,
        help="Keep y-axis fixed to [0, 1]. Default: enabled.",
    )
    parser.add_argument(
        "--no_ylim_0_1",
        action="store_false",
        dest="ylim_0_1",
        help="Disable fixed [0, 1] y-axis.",
    )
    return parser


def _torch_load_checkpoint(path: Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return load_torch_checkpoint(path, map_location=map_location)


def _checkpoint_epoch(payload: Mapping[str, Any], path: Path) -> tuple[int | None, str]:
    try:
        return int(payload["epoch"]), ""
    except Exception:
        return None, f"checkpoint {path.name} is missing integer epoch metadata; lexical order was used"


def _resolve_checkpoint_files(path: Path, *, recursive: bool) -> tuple[list[Path], list[str]]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"--checkpoint path does not exist: {path}")
    if path.is_file():
        if path.suffix != ".pt":
            raise ValueError(f"File checkpoint input must be a .pt file: {path}")
        return [path], []
    if not path.is_dir():
        raise ValueError(f"--checkpoint must be a .pt file or directory: {path}")

    pattern = "**/*.pt" if recursive else "*.pt"
    pt_files = sorted(path.glob(pattern), key=lambda item: str(item))
    if not pt_files:
        hint = " Try --recursive if the .pt files are under nested run directories." if not recursive else ""
        raise ValueError(f"Checkpoint directory contains no .pt files: {path}.{hint}")

    warnings: list[str] = []
    sortable: list[tuple[int, str, Path]] = []
    missing_epoch = False
    for pt_file in pt_files:
        payload = _torch_load_checkpoint(pt_file, map_location="cpu")
        epoch, warning = _checkpoint_epoch(payload, pt_file)
        if warning:
            warnings.append(warning)
            missing_epoch = True
        sortable.append((10**18 if epoch is None else int(epoch), pt_file.name, pt_file))
    if missing_epoch:
        return [item[2] for item in sorted(sortable, key=lambda item: item[1])], warnings
    return [item[2] for item in sorted(sortable, key=lambda item: (item[0], item[1]))], warnings


def _resolve_device(gpu_index: int) -> torch.device:
    index = int(gpu_index)
    if index < 0:
        return torch.device("cpu")
    if torch.cuda.is_available():
        if index >= torch.cuda.device_count():
            raise ValueError(f"--gpu_index {index} is invalid for {torch.cuda.device_count()} CUDA device(s).")
        torch.cuda.set_device(index)
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


def _seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _model_family(spec: ModelSpec) -> str:
    if spec.family in {"cnn_lif", "cnn_rf"}:
        return "cnn"
    if spec.family in {"lif", "rf"}:
        return "dense_snn"
    return str(spec.family)


def _json_metadata(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value
    return compact_yaml(value)


def _axis_metadata_columns(manifest: Mapping[str, Any], *, psd_axis_kind: str) -> dict[str, str]:
    logical_shape = manifest.get("psd_logical_shape")
    static_repeat_t = ""
    if isinstance(logical_shape, Mapping):
        static_repeat_t = logical_shape.get("T", logical_shape.get("time", ""))
    return {
        "prep_profile": str(manifest.get("prep_profile", psd_axis_kind)),
        "psd_axis_kind": str(manifest.get("psd_axis_kind", psd_axis_kind)),
        "psd_time_axis": str(manifest.get("psd_time_axis", "")),
        "psd_row_axes": _json_metadata(manifest.get("psd_row_axes")),
        "psd_flatten_rule": str(manifest.get("psd_flatten_rule", "")),
        "psd_logical_shape": _json_metadata(logical_shape),
        "static_repeat_T": "" if static_repeat_t is None else str(static_repeat_t),
    }


def _read_manifest(manifest_path: Path) -> dict[str, Any]:
    payload = load_manifest(manifest_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Prepared manifest must be a mapping: {manifest_path}")
    return payload


def _payload_dataset(payload: Mapping[str, Any]) -> str:
    return str(payload.get("dataset_token") or payload.get("training_args", {}).get("dataset") or "").strip()


def _payload_prep_root(payload: Mapping[str, Any]) -> str:
    return str(payload.get("prep_root") or payload.get("training_args", {}).get("prep_root") or "").strip()


def _payload_seed(payload: Mapping[str, Any]) -> int:
    value = payload.get("seed", payload.get("training_args", {}).get("seed", 0))
    return int(value)


def _payload_model_config(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    model_config = payload.get("model_config")
    if isinstance(model_config, Mapping):
        return model_config
    training_args = payload.get("training_args") if isinstance(payload.get("training_args"), Mapping) else {}
    legacy = {
        "input_dim": payload.get("input_dim") or training_args.get("input_dim"),
        "sequence_length": payload.get("sequence_length") or training_args.get("sequence_length"),
        "num_classes": payload.get("num_classes") or training_args.get("num_classes"),
        "input_shape": payload.get("input_shape") or training_args.get("input_shape"),
        "hidden_spec": payload.get("hidden_spec") or training_args.get("hidden_spec") or payload.get("arch_spec") or training_args.get("arch_spec"),
        "arch_spec": payload.get("arch_spec") or training_args.get("arch_spec") or payload.get("hidden_spec") or training_args.get("hidden_spec"),
        "v_th": payload.get("v_th") or training_args.get("v_th") or 1.0,
    }
    if all(legacy.get(k) not in (None, "") for k in ("input_dim", "sequence_length", "num_classes")):
        return legacy
    raise ValueError("Checkpoint model_config is missing and legacy dimension metadata is incomplete.")


def _payload_readout_mode(payload: Mapping[str, Any]) -> str:
    readout_config = payload.get("readout_config")
    if isinstance(readout_config, Mapping):
        mode = str(readout_config.get("mode") or readout_config.get("readout_mode") or "").strip()
    else:
        training_args = payload.get("training_args") if isinstance(payload.get("training_args"), Mapping) else {}
        mode = str(payload.get("readout_mode") or training_args.get("readout_mode") or "").strip()
    if not mode:
        raise ValueError("Checkpoint readout_config/readout_mode is missing mode.")
    return canonicalize_readout_mode(mode)


def _build_model_from_checkpoint(payload: Mapping[str, Any], *, device: torch.device):
    training_args = payload.get("training_args") if isinstance(payload.get("training_args"), Mapping) else {}
    model_token = str(payload.get("model_token") or payload.get("model") or training_args.get("model") or "").strip()
    if not model_token:
        raise ValueError("Checkpoint is missing model_token.")
    spec = canonicalize_model_token(model_token)
    model_config = _payload_model_config(payload)

    mode = _payload_readout_mode(payload)
    input_dim = int(model_config["input_dim"])
    sequence_length = int(model_config["sequence_length"])
    num_classes = int(model_config["num_classes"])
    input_shape = model_config.get("input_shape")
    if input_shape is not None:
        input_shape = [int(v) for v in input_shape]

    if spec.family in {"cnn_lif", "cnn_rf"}:
        hidden_spec = "-"
    else:
        hidden_spec = str(model_config.get("hidden_spec") or model_config.get("arch_spec") or "")

    v_th = float(model_config.get("v_th", 1.0))
    readout = build_readout(mode, num_classes=num_classes, sequence_length=sequence_length, device=device)
    model = build_snn_classifier(
        model_token=spec,
        input_dim=input_dim,
        sequence_length=sequence_length,
        num_classes=num_classes,
        input_shape=input_shape,
        hidden_sizes=None,
        arch_spec=hidden_spec,
        output_layer_overrides=readout.output_layer_overrides(),
        v_th=v_th,
    ).to(device)

    load_state_dict_compatible(model, checkpoint_state_dict(payload), context='checkpoint_accuracy_eval state_dict', strict=True)
    model.eval()
    readout.to(device)
    readout.eval()
    return model, readout, spec, mode


def _run_identity(payload: Mapping[str, Any]) -> tuple[str, str, str, int]:
    dataset = _payload_dataset(payload)
    training_args = payload.get("training_args") if isinstance(payload.get("training_args"), Mapping) else {}
    model_token = str(payload.get("model_token") or payload.get("model") or training_args.get("model") or "").strip()
    readout_mode = _payload_readout_mode(payload)
    seed = _payload_seed(payload)
    return dataset, model_token, readout_mode, seed


def _validate_same_run(payloads: Sequence[Mapping[str, Any]], checkpoint_files: Sequence[Path]) -> tuple[str, str, str, int]:
    if not payloads:
        raise ValueError("No checkpoint payloads were provided.")
    first = _run_identity(payloads[0])
    for payload, path in zip(payloads[1:], checkpoint_files[1:]):
        current = _run_identity(payload)
        if current != first:
            raise ValueError(
                "All checkpoints must belong to one run. "
                f"First identity={first}, but {path} has identity={current}."
            )
    return first


def _assert_unique_epochs(payloads: Sequence[Mapping[str, Any]], checkpoint_files: Sequence[Path]) -> None:
    seen: dict[int, Path] = {}
    for payload, path in zip(payloads, checkpoint_files):
        epoch, _warning = _checkpoint_epoch(payload, path)
        if epoch is None:
            continue
        if epoch in seen:
            raise ValueError(f"Duplicate checkpoint epoch {epoch}: {seen[epoch]} and {path}")
        seen[int(epoch)] = path


def _make_metric_rows(
    *,
    dataset: str,
    run_id: str,
    axis_base: Mapping[str, Any],
    model_spec: ModelSpec,
    readout_mode: str,
    seed: int,
    checkpoint_epoch: int,
    scope: str,
    metrics: EpochMetrics,
) -> list[dict[str, str]]:
    base = {
        "category": "training_metric",
        "source_program": SOURCE_PROGRAM,
        "dataset": dataset,
        "run_id": run_id,
        **dict(axis_base),
        "model_token": model_spec.canonical_token,
        "model_family": _model_family(model_spec),
        "readout_mode": readout_mode,
        "seed": int(seed),
        "epoch": int(checkpoint_epoch),
        "scope": scope,
    }
    return [
        common_row(**base, metric="accuracy", value=float(metrics.accuracy), value_unit="fraction"),
        common_row(**base, metric="loss", value=float(metrics.loss), value_unit="loss"),
        common_row(**base, metric="correct", value=int(metrics.correct), value_unit="count"),
        common_row(**base, metric="total", value=int(metrics.total), value_unit="count"),
    ]


def _accuracy_points(rows: Sequence[Mapping[str, str]], *, split: str) -> tuple[list[int], list[float]]:
    selected: list[tuple[int, float]] = []
    for row in rows:
        if str(row.get("metric", "")).strip() != "accuracy":
            continue
        if str(row.get("scope", "")).strip() != split:
            continue
        selected.append((int(float(str(row.get("epoch", "0")))), float(str(row.get("value", "nan")))))
    selected.sort(key=lambda item: item[0])
    return [item[0] for item in selected], [item[1] for item in selected]


def _safe_output_path(output_root: Path, file_name: str) -> Path:
    name = Path(file_name).name
    if not name:
        raise ValueError(f"Invalid output file name: {file_name!r}")
    return output_root / name


def _assert_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Pass --overwrite to replace it.")


def _render_accuracy_plot(
    rows: Sequence[Mapping[str, str]],
    *,
    output_path: Path,
    overwrite: bool,
    ylim_0_1: bool,
) -> None:
    _assert_can_write(output_path, overwrite=overwrite)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    plotted = 0
    for split, label in (("train", "Train Acc."), ("test", "Test Acc.")):
        xs, ys = _accuracy_points(rows, split=split)
        if not xs:
            continue
        ax.plot(xs, ys, linewidth=LINEWIDTH, marker="o", label=label)
        plotted += 1

    if plotted == 0:
        raise ValueError("No accuracy rows were available for plotting.")

    ax.set_title("Train/Test Accuracy", fontsize=TITLE_SIZE, fontweight="bold", pad=10)
    ax.set_xlabel("Epoch", fontsize=LABEL_SIZE, fontweight="bold", labelpad=8)
    ax.set_ylabel("Acc.", fontsize=LABEL_SIZE, fontweight="bold", labelpad=8)
    ax.grid(False)

    all_epochs = sorted({int(float(str(row.get("epoch", "0")))) for row in rows if str(row.get("metric", "")) == "accuracy"})
    if all_epochs:
        ax.set_xticks(all_epochs)

    if ylim_0_1:
        ax.set_ylim(0.0, 1.0)
    else:
        values = [float(str(row.get("value", "nan"))) for row in rows if str(row.get("metric", "")) == "accuracy"]
        finite_values = [v for v in values if np.isfinite(v)]
        if finite_values:
            lo = min(finite_values)
            hi = max(finite_values)
            pad = max(0.01, (hi - lo) * 0.08)
            ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))

    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    legend = ax.legend(frameon=False, loc="best")
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontweight("bold")
            text.set_fontsize(TICK_SIZE)

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _write_manifest_yaml(
    *,
    output_path: Path,
    overwrite: bool,
    payload: Mapping[str, Any],
) -> None:
    _assert_can_write(output_path, overwrite=overwrite)
    save_yaml(output_path, dict(payload))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if int(args.batch_size) < 1:
        parser.error("--batch_size must be >= 1.")
    if int(args.num_workers) < 0:
        parser.error("--num_workers must be >= 0.")

    checkpoint_input = Path(args.checkpoint).expanduser().resolve()
    output_root = timestamped_output_root(args.output_root, run_timestamp=getattr(args, 'run_timestamp', None), prefix=SOURCE_PROGRAM, enabled=getattr(args, 'timestamped_output', True))
    output_root.mkdir(parents=True, exist_ok=True)

    csv_path = _safe_output_path(output_root, args.csv_name)
    plot_path = _safe_output_path(output_root, args.plot_name)
    manifest_path = output_root / "checkpoint_accuracy_eval_manifest.yaml"
    _assert_can_write(csv_path, overwrite=bool(args.overwrite))
    _assert_can_write(plot_path, overwrite=bool(args.overwrite))
    _assert_can_write(manifest_path, overwrite=bool(args.overwrite))

    checkpoint_files, ordering_warnings = _resolve_checkpoint_files(checkpoint_input, recursive=bool(args.recursive))
    payloads = [_torch_load_checkpoint(path, map_location="cpu") for path in checkpoint_files]
    _assert_unique_epochs(payloads, checkpoint_files)
    checkpoint_dataset, _model_token, _readout_mode, checkpoint_seed = _validate_same_run(payloads, checkpoint_files)

    if args.dataset is None:
        dataset_token = checkpoint_dataset
    else:
        dataset_token = str(args.dataset).strip()
        if checkpoint_dataset and dataset_token != checkpoint_dataset:
            parser.error(f"--dataset {dataset_token!r} does not match checkpoint dataset_token {checkpoint_dataset!r}.")
    if not dataset_token:
        parser.error("Dataset token is missing. Pass --dataset or use checkpoints with dataset_token metadata.")

    if args.prep_root is None:
        checkpoint_prep_root = _payload_prep_root(payloads[0])
        if not checkpoint_prep_root:
            parser.error("prep_root is missing. Pass --prep_root or use checkpoints with prep_root metadata.")
        prep_root = Path(checkpoint_prep_root).expanduser().resolve()
    else:
        prep_root = Path(args.prep_root).expanduser().resolve()

    prepared_manifest_path = resolve_manifest_path(prep_root / dataset_token)
    if not prepared_manifest_path.exists():
        parser.error(f"Prepared manifest is missing: {prepared_manifest_path}")

    seed = checkpoint_seed if args.seed is None else int(args.seed)
    _seed_everything(seed)
    device = _resolve_device(int(args.gpu_index))

    bundle = resolve_dataset_bundle(dataset_token, prep_root=prep_root)
    manifest = _read_manifest(bundle.manifest_path)
    axis_base = _axis_metadata_columns(manifest, psd_axis_kind=bundle.psd_axis_kind)

    split_to_dataset = {
        "train": bundle.train_dataset,
        "test": bundle.test_dataset,
    }

    all_rows: list[dict[str, str]] = []
    summary: list[dict[str, Any]] = []
    for checkpoint_path, payload in zip(checkpoint_files, payloads):
        epoch, warning = _checkpoint_epoch(payload, checkpoint_path)
        if epoch is None:
            epoch = len(summary) + 1
        model, readout, model_spec, readout_mode = _build_model_from_checkpoint(payload, device=device)
        run_id = f"{dataset_token}_{model_spec.canonical_token}_{readout_mode}_seed{_payload_seed(payload)}"

        checkpoint_summary: dict[str, Any] = {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_epoch": int(epoch),
            "warning": warning,
        }

        for split in args.splits:
            loader = make_loader(
                split_to_dataset[str(split)],
                batch_size=int(args.batch_size),
                shuffle=False,
                num_workers=int(args.num_workers),
                pin_memory=device.type == "cuda",
                seed=seed,
            )
            metrics = evaluate_one_epoch(
                model,
                loader,
                readout=readout,
                device=device,
                progress_desc=f"eval {split} epoch {int(epoch)}",
            )
            all_rows.extend(
                _make_metric_rows(
                    dataset=dataset_token,
                    run_id=run_id,
                    axis_base=axis_base,
                    model_spec=model_spec,
                    readout_mode=readout_mode,
                    seed=_payload_seed(payload),
                    checkpoint_epoch=int(epoch),
                    scope=str(split),
                    metrics=metrics,
                )
            )
            checkpoint_summary[f"{split}_accuracy"] = float(metrics.accuracy)
            checkpoint_summary[f"{split}_loss"] = float(metrics.loss)
            checkpoint_summary[f"{split}_correct"] = int(metrics.correct)
            checkpoint_summary[f"{split}_total"] = int(metrics.total)

        summary.append(checkpoint_summary)
        del model, readout
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_common_csv(csv_path, all_rows)
    _render_accuracy_plot(all_rows, output_path=plot_path, overwrite=True, ylim_0_1=bool(args.ylim_0_1))
    _write_manifest_yaml(
        output_path=manifest_path,
        overwrite=True,
        payload={
            "status": "ok",
            "source_program": SOURCE_PROGRAM,
            "checkpoint_input": str(checkpoint_input),
            "checkpoint_files": [str(path) for path in checkpoint_files],
            "ordering_warnings": ordering_warnings,
            "dataset": dataset_token,
            "prep_root": str(prep_root),
            "device": str(device),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "splits": list(args.splits),
            "csv_path": str(csv_path),
            "plot_path": str(plot_path),
            "summary": summary,
        },
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "source_program": SOURCE_PROGRAM,
                "checkpoint_count": len(checkpoint_files),
                "csv_path": str(csv_path),
                "plot_path": str(plot_path),
                "manifest_path": str(manifest_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
