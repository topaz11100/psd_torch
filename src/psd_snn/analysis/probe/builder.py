from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence, Optional
import random

@dataclass
class ProbeManifest:
    family: str
    seed: int
    selected_indices: List[int]
    class_counts: Dict[int, int]
    quotas: Dict[int, int]


def _group(labels):
    by = {}
    for i, y in enumerate(labels):
        by.setdefault(int(y), []).append(i)
    return by


def build_probe_indices(labels: Sequence[int], family: str, sample_count: int, seed: int, target_labels: Optional[Sequence[int]] = None, exclusion_indices: Optional[set[int]] = None) -> ProbeManifest:
    if family not in {'balanced_global', 'distributed_set', 'label_set', 'label_single'}:
        raise ValueError('unsupported probe family')
    rnd = random.Random(seed)
    by = _group(labels)
    selected, quotas = [], {}
    exclusion_indices = exclusion_indices or set()

    if family == 'balanced_global':
        classes = sorted(by)
        per = max(1, sample_count // len(classes))
        for c in classes:
            pool = [i for i in by[c] if i not in exclusion_indices]; rnd.shuffle(pool)
            take = min(per, len(pool)); quotas[c] = take; selected.extend(pool[:take])
    elif family == 'distributed_set':
        total = len(labels)
        classes = sorted(by)
        raw = {c: (len(by[c]) / total) * sample_count for c in classes}
        base = {c: int(raw[c]) for c in classes}
        rem = sample_count - sum(base.values())
        fracs = sorted(classes, key=lambda c: (raw[c] - base[c], -c), reverse=True)
        for c in fracs[:rem]:
            base[c] += 1
        quotas = dict(base)
        for c in classes:
            pool = [i for i in by[c] if i not in exclusion_indices]; rnd.shuffle(pool)
            selected.extend(pool[:quotas[c]])
    elif family == 'label_set':
        if not target_labels:
            raise ValueError('label_set requires target_labels')
        k = max(1, sample_count // len(target_labels))
        for c in sorted(set(int(x) for x in target_labels)):
            pool = [i for i in by.get(c, []) if i not in exclusion_indices]; rnd.shuffle(pool)
            take = min(k, len(pool)); quotas[c] = take; selected.extend(pool[:take])
    else:  # label_single
        classes = sorted(by if target_labels is None else set(int(x) for x in target_labels))
        for c in classes:
            pool = [i for i in by.get(c, []) if i not in exclusion_indices]
            rnd.shuffle(pool)
            if pool:
                selected.append(pool[0]); quotas[c] = 1
            else:
                quotas[c] = 0

    counts = {}
    for i in selected:
        y = int(labels[i]); counts[y] = counts.get(y, 0) + 1
    return ProbeManifest(family=family, seed=seed, selected_indices=sorted(selected), class_counts=counts, quotas=quotas)
