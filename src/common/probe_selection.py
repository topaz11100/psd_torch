from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Mapping


ProbeScopeMap = Dict[str, Dict[int, List[int]]]


def group_indices_by_label(dataset, num_classes: int) -> Dict[int, List[int]]:
    grouped: Dict[int, List[int]] = {int(c): [] for c in range(int(num_classes))}
    for idx in range(len(dataset)):
        _, label = dataset[int(idx)]
        grouped.setdefault(int(label), []).append(int(idx))
    return grouped



def _stable_probe_rank(split_name: str, base_seed: int, label: int, dataset_index: int) -> bytes:
    """Return a stable hash rank for one dataset index.

    The canonical rank intentionally depends only on the split, the user seed,
    the label id, and the dataset index itself. Model scenario details must not
    perturb the order. Scope-specific counts are applied later as prefix lengths.
    """
    payload = f"probe|{str(split_name)}|{int(base_seed)}|{int(label)}|{int(dataset_index)}"
    return hashlib.sha1(payload.encode("utf-8")).digest()



def canonical_label_probe_order(
    dataset,
    num_classes: int,
    *,
    split_name: str,
    base_seed: int,
) -> Dict[int, List[int]]:
    """Build one canonical deterministic order per label.

    The returned order is shared by all probe scopes. `same_label` and
    `balanced_global` are prefix slices of the same label-wise canonical order.
    """
    grouped = group_indices_by_label(dataset, int(num_classes))
    ordered: Dict[int, List[int]] = {}
    for label in range(int(num_classes)):
        idxs = sorted(int(v) for v in grouped.get(int(label), []))
        if len(idxs) == 0:
            continue
        ranked = sorted(
            idxs,
            key=lambda idx: (
                _stable_probe_rank(str(split_name), int(base_seed), int(label), int(idx)),
                int(idx),
            ),
        )
        ordered[int(label)] = [int(v) for v in ranked]
    return ordered



def select_fixed_probe_scopes(
    dataset,
    num_classes: int,
    *,
    split_name: str,
    base_seed: int,
    same_label_n: int,
    balanced_n: int,
) -> ProbeScopeMap:
    """Select deterministic probe scopes from shared canonical label orders.

    Guarantees:
    - only split, seed, label, and dataset index determine the canonical order
    - model scenario, readout mode, timestamp, output root, etc. do not matter
    - `same_label` / `balanced_global` differ only by how many prefix elements
      they take from the same per-label canonical order
    - when `same_label_n == balanced_n`, the per-label selections are identical
    """
    label_orders = canonical_label_probe_order(
        dataset,
        int(num_classes),
        split_name=str(split_name),
        base_seed=int(base_seed),
    )
    same_label: Dict[int, List[int]] = {}
    balanced_global: Dict[int, List[int]] = {}
    for label in sorted(label_orders.keys()):
        ordered = [int(v) for v in label_orders[int(label)]]
        same_label[int(label)] = ordered[: min(int(same_label_n), len(ordered))]
        balanced_global[int(label)] = ordered[: min(int(balanced_n), len(ordered))]
    return {
        "same_label": same_label,
        "balanced_global": balanced_global,
    }



def _normalized_scope_map(scope_map: Mapping[Any, Any]) -> Dict[str, List[int]]:
    normalized: Dict[str, List[int]] = {}
    for raw_label, raw_indices in sorted(((int(k), v) for k, v in scope_map.items()), key=lambda kv: kv[0]):
        if raw_indices is None:
            normalized[str(int(raw_label))] = []
            continue
        if not isinstance(raw_indices, (list, tuple)):
            raise TypeError(f"scope indices for label={raw_label} must be list/tuple, got {type(raw_indices)!r}")
        normalized[str(int(raw_label))] = [int(v) for v in raw_indices]
    return normalized



def flatten_scope_indices(split_scopes: Mapping[str, Any], scope_name: str) -> List[int]:
    if str(scope_name) not in split_scopes:
        raise KeyError(f"missing scope: {scope_name}")
    normalized = _normalized_scope_map(split_scopes[str(scope_name)])
    flat: List[int] = []
    for label in sorted(int(k) for k in normalized.keys()):
        flat.extend(int(v) for v in normalized[str(int(label))])
    return flat



def probe_union_indices(split_scopes: Mapping[str, Any]) -> List[int]:
    seen = set()
    for scope_name in ("same_label", "balanced_global"):
        if scope_name not in split_scopes:
            continue
        seen.update(int(v) for v in flatten_scope_indices(split_scopes, scope_name))
    return sorted(int(v) for v in seen)



def probe_scope_signature(split_scopes: Mapping[str, Any]) -> str:
    payload = {
        "same_label": _normalized_scope_map(split_scopes.get("same_label", {})),
        "balanced_global": _normalized_scope_map(split_scopes.get("balanced_global", {})),
        "same_label_flat": flatten_scope_indices(split_scopes, "same_label") if "same_label" in split_scopes else [],
        "balanced_global_flat": flatten_scope_indices(split_scopes, "balanced_global") if "balanced_global" in split_scopes else [],
        "probe_union": probe_union_indices(split_scopes),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
