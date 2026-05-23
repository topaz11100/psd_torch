from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence, Optional
import hashlib
import json
import random

CANONICAL_PROBE_FAMILIES = {'balanced_global', 'distributed_set', 'label_set', 'label_single'}


@dataclass
class ProbeManifest:
    probe_family: str
    split: str
    scope: str
    seed: int
    sample_count: int
    selected_indices: List[int]
    selected_labels: List[int] | None
    class_counts: Dict[int, int] | None
    quotas: Dict[int, int] | None
    target_labels: List[int] | None = None
    exclusion_family: str | None = None
    exclusion_scope: str | None = None
    excluded_indices: List[int] | None = None
    excluded_sample_count: int | None = None
    selection_rule: str = 'deterministic'
    probe_manifest_id: str | None = None


def _group(labels):
    by = {}
    for i, y in enumerate(labels):
        by.setdefault(int(y), []).append(i)
    return by


def _scope(split: str, family: str, labels: Optional[Sequence[int]] = None, exclusion_family: str | None = None) -> str:
    if family == 'label_set' and labels:
        uniq = sorted(set(int(x) for x in labels))
        if len(uniq) == 1:
            return f'{split}_label_set_label={uniq[0]}'
        return f'{split}_label_set'
    if family == 'label_single' and exclusion_family:
        return f'{split}_label_single_excluding_{exclusion_family}'
    return f'{split}_{family}'


def _manifest_id(payload: dict) -> str:
    s = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def build_probe_indices(labels: Sequence[int], family: str, sample_count: int, seed: int, target_labels: Optional[Sequence[int]] = None, exclusion_indices: Optional[set[int]] = None, split: str = 'test', exclusion_family: str | None = None) -> ProbeManifest:
    if family not in CANONICAL_PROBE_FAMILIES:
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
            if len(pool) < quotas[c]:
                raise ValueError('insufficient class samples for distributed_set quota')
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
            if not pool:
                raise ValueError(f'label_single empty after exclusion for label={c}')
            selected.append(pool[0]); quotas[c] = 1

    counts = {}
    for i in selected:
        y = int(labels[i]); counts[y] = counts.get(y, 0) + 1
    selected = sorted(selected)
    sel_labels = [int(labels[i]) for i in selected]
    scope = _scope(split, family, target_labels, exclusion_family)
    meta = {
        'probe_family': family, 'split': split, 'scope': scope, 'seed': seed, 'sample_count': sample_count,
        'selected_indices': selected, 'target_labels': sorted(set(int(x) for x in target_labels)) if target_labels else None,
        'exclusion_family': exclusion_family,
    }
    pid = _manifest_id(meta)
    return ProbeManifest(probe_family=family, split=split, scope=scope, seed=seed, sample_count=sample_count, selected_indices=selected, selected_labels=sel_labels, class_counts=counts, quotas=quotas, target_labels=(sorted(set(int(x) for x in target_labels)) if target_labels else None), exclusion_family=exclusion_family, exclusion_scope=(_scope(split, exclusion_family) if exclusion_family else None), excluded_indices=sorted(exclusion_indices) if exclusion_indices else None, excluded_sample_count=(len(exclusion_indices) if exclusion_indices else 0), selection_rule='deterministic', probe_manifest_id=pid)
