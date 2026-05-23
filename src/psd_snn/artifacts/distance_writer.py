from __future__ import annotations
import csv
from pathlib import Path
import math

from psd_snn.artifacts.identity import SpectralArtifactIdentity, enforce_compatibility


def _centered_l2(a, b):
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    return math.sqrt(sum((((x - ma) - (y - mb)) ** 2) for x, y in zip(a, b)))


def _diff_l2(a, b):
    da = [a[i + 1] - a[i] for i in range(len(a) - 1)]
    db = [b[i + 1] - b[i] for i in range(len(b) - 1)]
    return math.sqrt(sum(((x - y) ** 2) for x, y in zip(da, db)))


def _flatten_matrix(m):
    return [v for row in m for v in row]


def distance_value(left, right, metric: str, diff_axis: str = "time_frequency"):
    lt = left["type"]
    if lt == "spectral_curve":
        a, b = left["power"], right["power"]
        return _centered_l2(a, b) if metric == "centered_l2" else _diff_l2(a, b)
    if lt == "spectral_matrix_1d":
        if metric == "centered_l2":
            return _centered_l2(_flatten_matrix(left["matrix"]), _flatten_matrix(right["matrix"]))
        a = [_diff_l2(r1, r2) for r1, r2 in zip(left["matrix"], right["matrix"])]
        return sum(a)
    if lt == "spectral_matrix_2d":
        la, rb = left["matrix"], right["matrix"]
        if metric == "centered_l2":
            return _centered_l2(_flatten_matrix(la), _flatten_matrix(rb))
        if diff_axis == "row_frequency":
            if left.get("row_axis_semantics") == "unordered":
                raise ValueError("unordered row_axis_semantics cannot use row_frequency diff")
            da = [ [la[i+1][j]-la[i][j] for j in range(len(la[0]))] for i in range(len(la)-1) ]
            db = [ [rb[i+1][j]-rb[i][j] for j in range(len(rb[0]))] for i in range(len(rb)-1) ]
            return _centered_l2(_flatten_matrix(da), _flatten_matrix(db))
        if diff_axis == "both_frequency_axes":
            dt = [ [la[i][j+1]-la[i][j] for j in range(len(la[0])-1)] for i in range(len(la)) ]
            dr = [ [la[i+1][j]-la[i][j] for j in range(len(la[0]))] for i in range(len(la)-1) ]
            et = [ [rb[i][j+1]-rb[i][j] for j in range(len(rb[0])-1)] for i in range(len(rb)) ]
            er = [ [rb[i+1][j]-rb[i][j] for j in range(len(rb[0]))] for i in range(len(rb)-1) ]
            return _centered_l2(_flatten_matrix(dt)+_flatten_matrix(dr), _flatten_matrix(et)+_flatten_matrix(er))
        da = [ [la[i][j+1]-la[i][j] for j in range(len(la[0])-1)] for i in range(len(la)) ]
        db = [ [rb[i][j+1]-rb[i][j] for j in range(len(rb[0])-1)] for i in range(len(rb)) ]
        return _centered_l2(_flatten_matrix(da), _flatten_matrix(db))
    raise ValueError("unsupported artifact type for distance")


def write_spectral_distance_csv(out_dir: str, rows: list[dict]):
    p = Path(out_dir) / "spectral_distance.csv"
    if not rows:
        return p
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return p


def build_distance_row(left: dict, right: dict, metric: str, diff_axis: str = "time_frequency") -> dict:
    li = SpectralArtifactIdentity(**left["identity"])
    ri = SpectralArtifactIdentity(**right["identity"])
    enforce_compatibility(li, ri, metric)
    v = distance_value(left, right, metric, diff_axis=diff_axis)
    meta = left.get("meta", {})
    return {
        "artifact_type": "spectral_distance",
        "run_id": meta.get("run_id"),
        "left_checkpoint_epoch": meta.get("checkpoint_epoch"),
        "right_checkpoint_epoch": right.get("meta", {}).get("checkpoint_epoch"),
        "split": meta.get("split"),
        "scope": meta.get("scope"),
        "probe_family": meta.get("probe_family"),
        "layer_name": meta.get("layer_name"),
        "series": meta.get("series"),
        "spectral_axis": li.spectral_axis,
        "distance_metric": metric,
        "diff_axis": diff_axis,
        "pca_basis_id": li.pca_basis_id,
        "userbin_axes": li.userbin_axes,
        "row_axis_semantics": li.row_axis_semantics,
        "fftshift_row": li.fftshift_row,
        "compatibility_key": li.key(),
        "comparison_type": "pair",
        "value": v,
        "value_unit": "l2",
        "status": "ok",
        "reason": None,
    }
