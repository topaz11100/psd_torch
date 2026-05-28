#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


DATASET_SEQUENCE_LENGTH = {
    "s-mnist": 784,
    "s_mnist": 784,
    "smnist": 784,
    "shd": 1200,
}


def torch_load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    if not isinstance(payload, dict):
        raise ValueError(f"not a dict payload: {path}")

    return payload


def infer_dataset_token(payload: Mapping[str, Any], path: Path) -> str:
    raw = str(
        payload.get("dataset_token")
        or payload.get("training_args", {}).get("dataset")
        or ""
    ).strip().lower()

    path_lower = str(path).lower()

    for item in (raw, path_lower):
        if "s-mnist" in item or "s_mnist" in item or "smnist" in item:
            return "s-mnist"
        if re.search(r"(^|[^a-z0-9])shd([^a-z0-9]|$)", item):
            return "shd"

    raise ValueError("cannot infer dataset token")


def sequence_length_for_dataset(dataset_token: str) -> int:
    key = dataset_token.strip().lower()
    if key not in DATASET_SEQUENCE_LENGTH:
        raise ValueError(f"unsupported dataset token: {dataset_token!r}")
    return int(DATASET_SEQUENCE_LENGTH[key])


def strip_state_prefix(key: str) -> str:
    for prefix in ("module._orig_mod.", "_orig_mod.", "module."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def infer_arch_from_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    hidden_by_index: dict[int, tuple[int, int]] = {}
    output_shape: tuple[int, int] | None = None

    for raw_key, value in state_dict.items():
        key = strip_state_prefix(str(raw_key))

        if not isinstance(value, torch.Tensor):
            continue
        if value.ndim != 2:
            continue

        m = re.fullmatch(r"hidden_layers\.(\d+)\.input_weight", key)
        if m:
            idx = int(m.group(1))
            hidden_by_index[idx] = (int(value.shape[0]), int(value.shape[1]))
            continue

        if key == "output_layer.input_weight":
            output_shape = (int(value.shape[0]), int(value.shape[1]))

    if not hidden_by_index:
        raise ValueError("cannot infer hidden layers from state_dict")

    if output_shape is None:
        raise ValueError("cannot infer num_classes from output_layer.input_weight")

    hidden_sizes = [hidden_by_index[i][0] for i in sorted(hidden_by_index)]
    first_idx = min(hidden_by_index)
    input_dim = int(hidden_by_index[first_idx][1])
    num_classes = int(output_shape[0])

    # 중요: 이 레포는 comma-delimited hidden_spec를 요구함.
    hidden_spec = ",".join(str(v) for v in hidden_sizes)

    return {
        "input_dim": input_dim,
        "num_classes": num_classes,
        "hidden_spec": hidden_spec,
    }


def infer_readout_mode(payload: Mapping[str, Any]) -> str:
    mode = str(
        payload.get("readout_mode")
        or payload.get("training_args", {}).get("readout_mode")
        or ""
    ).strip()

    if not mode:
        raise ValueError("cannot infer readout_mode")

    return mode


def infer_v_th(payload: Mapping[str, Any]) -> float:
    training_args = payload.get("training_args")
    if isinstance(training_args, Mapping) and "v_th" in training_args:
        return float(training_args["v_th"])

    return 1.0


def is_hyphen_integer_spec(value: Any) -> bool:
    text = str(value).strip()
    return bool(re.fullmatch(r"\d+(?:-\d+)+", text))


def hyphen_to_comma_spec(value: Any) -> str:
    return str(value).strip().replace("-", ",")


def is_probably_training_checkpoint(payload: Mapping[str, Any]) -> bool:
    if "state_dict" not in payload:
        return False
    if "model_token" not in payload and "training_args" not in payload:
        return False
    if "dataset_token" not in payload and "training_args" not in payload:
        return False
    return True


def build_reference_metadata(payload: Mapping[str, Any], path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("state_dict is missing or not a mapping")

    dataset_token = infer_dataset_token(payload, path)
    arch = infer_arch_from_state_dict(state_dict)
    readout_mode = infer_readout_mode(payload)

    model_config = {
        "input_dim": int(arch["input_dim"]),
        "sequence_length": int(sequence_length_for_dataset(dataset_token)),
        "num_classes": int(arch["num_classes"]),
        "input_shape": None,
        "hidden_spec": str(arch["hidden_spec"]),
        "arch_spec": str(arch["hidden_spec"]),
        "v_th": float(infer_v_th(payload)),
    }

    readout_config = {
        "mode": str(readout_mode),
        "readout_mode": str(readout_mode),
    }

    return model_config, readout_config


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path = path.resolve()
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)

    tmp_path = Path(tmp_name)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def patch_checkpoint(path: Path, *, write: bool, backup: bool) -> tuple[str, str]:
    payload = torch_load_checkpoint(path)

    if not is_probably_training_checkpoint(payload):
        return "skip", "not a training checkpoint-like payload"

    ref_model_config, ref_readout_config = build_reference_metadata(payload, path)

    changed: list[str] = []

    model_config = payload.get("model_config")
    if not isinstance(model_config, Mapping):
        payload["model_config"] = dict(ref_model_config)
        changed.append("add model_config")
    else:
        new_model_config = dict(model_config)

        # 누락된 key만 채움. 이미 있는 정상 key는 덮어쓰지 않음.
        for key, value in ref_model_config.items():
            if key not in new_model_config or new_model_config[key] in (None, ""):
                new_model_config[key] = value
                changed.append(f"fill model_config.{key}")

        # 이전에 잘못 들어간 32-32-32 형식만 교정.
        for key in ("hidden_spec", "arch_spec"):
            if key in new_model_config and is_hyphen_integer_spec(new_model_config[key]):
                old = str(new_model_config[key])
                new_model_config[key] = hyphen_to_comma_spec(new_model_config[key])
                changed.append(f"fix model_config.{key}: {old} -> {new_model_config[key]}")

        if new_model_config != dict(model_config):
            payload["model_config"] = new_model_config

    readout_config = payload.get("readout_config")
    if not isinstance(readout_config, Mapping):
        payload["readout_config"] = dict(ref_readout_config)
        changed.append("add readout_config")
    else:
        new_readout_config = dict(readout_config)

        # 누락된 key만 채움. 기존 정상 값은 덮어쓰지 않음.
        for key, value in ref_readout_config.items():
            if key not in new_readout_config or new_readout_config[key] in (None, ""):
                new_readout_config[key] = value
                changed.append(f"fill readout_config.{key}")

        if new_readout_config != dict(readout_config):
            payload["readout_config"] = new_readout_config

    if not changed:
        return "skip", "nothing to change"

    final_mc = payload["model_config"]
    final_rc = payload["readout_config"]
    detail = (
        "; ".join(changed)
        + f" | hidden_spec={final_mc.get('hidden_spec')}"
        + f" | arch_spec={final_mc.get('arch_spec')}"
        + f" | T={final_mc.get('sequence_length')}"
        + f" | readout={final_rc.get('mode')}"
    )

    if not write:
        return "dry-run", detail

    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)

    atomic_torch_save(payload, path)
    return "patched", detail


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Safely fill missing checkpoint metadata and fix only invalid hyphen hidden_spec."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    pt_files = sorted(root.rglob("*.pt"), key=lambda p: str(p))
    counts = {"patched": 0, "dry-run": 0, "skip": 0, "error": 0}

    for path in pt_files:
        try:
            status, detail = patch_checkpoint(
                path,
                write=bool(args.write),
                backup=not bool(args.no_backup),
            )
            counts[status] += 1
            print(f"[{status}] {path.relative_to(root)} :: {detail}")
        except Exception as exc:
            counts["error"] += 1
            print(f"[error] {path.relative_to(root)} :: {type(exc).__name__}: {exc}")

    print("[summary] " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())