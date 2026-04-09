"""Deterministic probe set selection utilities."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


@dataclass
class ProbeSetSelection:
    """Indices selected for same_label and balanced_global scopes."""

    same_label: Dict[int, List[int]]
    balanced_global: List[int]


def _stable_int_key(split: str, seed: int, label: int, index: int) -> int:
    text = f"{split}|{seed}|{label}|{index}".encode("utf-8")
    return int(hashlib.sha256(text).hexdigest()[:16], 16)


def build_canonical_label_order(indices: Iterable[int], labels: Sequence[int], seed: int, split: str) -> Dict[int, List[int]]:
    """Build deterministic per-label order based only on split/seed/label/index."""

    grouped: Dict[int, List[int]] = defaultdict(list)
    for idx in indices:
        grouped[int(labels[idx])].append(int(idx))

    for label, rows in grouped.items():
        rows.sort(key=lambda x: (_stable_int_key(split, seed, label, x), x))
    return dict(grouped)


def select_probe_sets(canonical: Dict[int, List[int]], same_label_n: int, balanced_n: int) -> ProbeSetSelection:
    """Select same_label and balanced_global prefixes from canonical orders."""

    same = {label: rows[:same_label_n] for label, rows in canonical.items()}
    balanced: List[int] = []
    for label in sorted(canonical):
        balanced.extend(canonical[label][:balanced_n])
    return ProbeSetSelection(same_label=same, balanced_global=balanced)
