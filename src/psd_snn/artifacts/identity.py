from __future__ import annotations
from dataclasses import dataclass, asdict
import json, hashlib


@dataclass
class SpectralArtifactIdentity:
    artifact_type: str
    spectral_axis: str
    scale: str
    centering: str
    window: str | None = None
    representative: str | None = None
    analysis_method: str | None = None
    pca_basis_id: str | None = None
    userbin_axes: str | None = None
    userbin_reducer: str | None = None
    row_axis_semantics: str | None = None
    fftshift_row: bool | None = None
    row_bin_edges: tuple[float, ...] | None = None
    column_bin_edges: tuple[float, ...] | None = None

    def key(self) -> str:
        d = asdict(self)
        s = json.dumps(d, sort_keys=True, separators=(",",":"))
        return hashlib.sha256(s.encode()).hexdigest()[:16]


def enforce_compatibility(left: SpectralArtifactIdentity, right: SpectralArtifactIdentity, metric: str):
    if metric not in {"centered_l2", "diff_l2"}:
        raise ValueError("only centered_l2/diff_l2 are supported")
    if left.artifact_type != right.artifact_type:
        raise ValueError("artifact_type mismatch")
    if left.spectral_axis != right.spectral_axis:
        raise ValueError("exact/userbin mismatch")
    if left.scale != right.scale:
        raise ValueError("scale mismatch")
    if left.centering != right.centering:
        raise ValueError("centering mismatch")
    if left.window != right.window:
        raise ValueError("window mismatch")
    if left.representative != right.representative:
        raise ValueError("representative mismatch")
    if left.analysis_method != right.analysis_method:
        raise ValueError("analysis_method mismatch")
    if left.representative == "pca" or right.representative == "pca":
        if not left.pca_basis_id or not right.pca_basis_id:
            raise ValueError("pca_basis_id required for PCA distance")
        if left.pca_basis_id != right.pca_basis_id:
            raise ValueError("different pca_basis_id")
    if left.userbin_axes != right.userbin_axes:
        raise ValueError("userbin_axes mismatch")
    if left.userbin_reducer != right.userbin_reducer:
        raise ValueError("userbin_reducer mismatch")
    if left.row_axis_semantics != right.row_axis_semantics:
        raise ValueError("row_axis_semantics mismatch")
    if left.fftshift_row != right.fftshift_row:
        raise ValueError("fftshift_row mismatch")
    if left.row_bin_edges != right.row_bin_edges:
        raise ValueError("row_bin_edges mismatch")
    if left.column_bin_edges != right.column_bin_edges:
        raise ValueError("column_bin_edges mismatch")
