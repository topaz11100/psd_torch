from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class PSDModelVariant:
    token: str
    base_model: str
    structure_mask: bool
    clip_params: bool


_VARIANTS = {
    "rf": PSDModelVariant(token="rf", base_model="RF", structure_mask=False, clip_params=False),
    "rf_struct": PSDModelVariant(token="rf_struct", base_model="RF", structure_mask=True, clip_params=False),
    "rf_clip": PSDModelVariant(token="rf_clip", base_model="RF", structure_mask=False, clip_params=True),
    "rf_structclip": PSDModelVariant(token="rf_structclip", base_model="RF", structure_mask=True, clip_params=True),
    "lif": PSDModelVariant(token="lif", base_model="LIF", structure_mask=False, clip_params=False),
    "lif_struct": PSDModelVariant(token="lif_struct", base_model="LIF", structure_mask=True, clip_params=False),
    "lif_clip": PSDModelVariant(token="lif_clip", base_model="LIF", structure_mask=False, clip_params=True),
    "lif_structclip": PSDModelVariant(token="lif_structclip", base_model="LIF", structure_mask=True, clip_params=True),
}


def parse_psd_model_variant(name: str) -> Optional[PSDModelVariant]:
    key = str(name).strip().lower()
    return _VARIANTS.get(key)


def _validate_edges(edges: Sequence[float], *, lower: float, upper: float, name: str) -> List[float]:
    vals = [float(v) for v in edges]
    if len(vals) < 2:
        raise ValueError(f"{name} must contain at least two values")
    if vals[0] < lower - 1e-12 or vals[-1] > upper + 1e-12:
        raise ValueError(f"{name} must lie in [{lower}, {upper}], got {vals}")
    for i in range(len(vals) - 1):
        if not vals[i] < vals[i + 1]:
            raise ValueError(f"{name} must be strictly increasing, got {vals}")
    return vals


def validate_rf_clip_edges(edges: Sequence[float]) -> List[float]:
    return _validate_edges(edges, lower=0.0, upper=0.5, name="w_clip_edges")


def validate_lif_clip_edges(edges: Sequence[float]) -> List[float]:
    return _validate_edges(edges, lower=0.0, upper=1.0, name="alpha_clip_edges")




def infer_num_groups_from_band_neuron_ends(band_neuron_ends: Sequence[str]) -> int:
    texts = [str(x) for x in band_neuron_ends]
    if len(texts) == 0:
        raise ValueError("band_neuron_ends must contain at least one layer entry")
    counts = []
    for text in texts:
        parts = [p.strip() for p in str(text).split(",") if p.strip()]
        counts.append(len(parts) + 1)
    first = int(counts[0])
    for c in counts[1:]:
        if int(c) != int(first):
            raise ValueError(f"all band_neuron_ends entries must imply the same number of groups, got {counts}")
    return int(first)

def default_band_neuron_ends(hidden: Sequence[int], num_groups: int) -> List[str]:
    if int(num_groups) < 1:
        raise ValueError(f"num_groups must be >= 1, got {num_groups}")
    out: List[str] = []
    for width in [int(h) for h in hidden]:
        if int(num_groups) == 1:
            out.append("")
            continue
        edges: List[str] = []
        for g in range(1, int(num_groups)):
            end = int(round(float(width) * g / float(num_groups)))
            end = min(max(end, 1), int(width) - 1)
            edges.append(str(end))
        uniq: List[str] = []
        for item in edges:
            if not uniq or uniq[-1] != item:
                uniq.append(item)
        if len(uniq) != int(num_groups) - 1:
            raise ValueError(
                f"cannot build a valid default band_neuron_ends for hidden width {width} and num_groups {num_groups}"
            )
        out.append(",".join(uniq))
    return out


def parse_band_neuron_ends(hidden: Sequence[int], band_neuron_ends: Sequence[str], num_groups: int) -> List[List[int]]:
    hidden = [int(h) for h in hidden]
    if len(band_neuron_ends) != len(hidden):
        raise ValueError(
            f"band_neuron_ends must have one entry per hidden layer (expected {len(hidden)}, got {len(band_neuron_ends)})"
        )
    out: List[List[int]] = []
    for li, (width, text) in enumerate(zip(hidden, band_neuron_ends), start=1):
        parts = [p.strip() for p in str(text).split(",") if p.strip()]
        if len(parts) != max(0, int(num_groups) - 1):
            raise ValueError(
                f"hidden layer {li}: expected {int(num_groups)-1} cumulative end indices, got {len(parts)}"
            )
        ends = [int(p) for p in parts]
        prev = 0
        for e in ends:
            if e <= prev:
                raise ValueError(f"hidden layer {li}: band ends must be strictly increasing (got {ends})")
            if e >= int(width):
                raise ValueError(f"hidden layer {li}: end index {e} must be < width {width}")
            prev = e
        out.append(ends)
    return out


def validate_tear(tear: int, num_hidden_layers: int) -> int:
    t = int(tear)
    if t < 1 or t > int(num_hidden_layers):
        raise ValueError(f"tear must satisfy 1 <= tear <= {int(num_hidden_layers)}, got {tear}")
    return t


def group_ids_from_ends(width: int, cumulative_ends: Sequence[int], num_groups: int) -> np.ndarray:
    width = int(width)
    ends = list(int(x) for x in cumulative_ends) + [width]
    gids = np.zeros(width, dtype=np.int64)
    start = 0
    for g, end in enumerate(ends):
        gids[start:end] = int(g)
        start = end
    if start != width or gids.shape[0] != width:
        raise RuntimeError("failed to build group ids")
    if int(gids.max()) + 1 != int(num_groups):
        raise RuntimeError("group count mismatch while building group ids")
    return gids


def groups_from_cli(hidden: Sequence[int], *, band_neuron_ends: Optional[Sequence[str]], num_groups: int) -> tuple[list[list[int]], list[np.ndarray]]:
    if band_neuron_ends is None:
        band_neuron_ends = default_band_neuron_ends(hidden, num_groups=num_groups)
    ends = parse_band_neuron_ends(hidden, band_neuron_ends, num_groups=num_groups)
    gids = [group_ids_from_ends(int(width), end_list, num_groups=num_groups) for width, end_list in zip(hidden, ends)]
    return ends, gids
