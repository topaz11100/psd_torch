from __future__ import annotations
from dataclasses import dataclass
import hashlib, json
from pathlib import Path
import torch

@dataclass(frozen=True)
class PCABasisKey:
    reference_checkpoint_epoch: int | None
    reference_checkpoint_id: str | None
    reference_split: str
    reference_scope: str
    reference_probe_family: str
    layer_index: int
    layer_name: str
    signal_kind: str
    series: str
    n_components: int
    row_count: int
    centering: bool
    sign_convention: str = 'largest_abs_loading_positive'

@dataclass
class PCABasisRecord:
    basis_id: str
    key: PCABasisKey
    mean: torch.Tensor
    components: torch.Tensor
    explained_variance: torch.Tensor
    explained_variance_ratio: torch.Tensor
    row_count: int
    n_components: int
    created_from: dict
    artifact_path: str | None = None

@dataclass
class PCAFitRequest:
    maps: torch.Tensor
    key: PCABasisKey
    created_from: dict

@dataclass
class PCAApplyRequest:
    maps: torch.Tensor
    key: PCABasisKey

class PCABasisStore:
    def __init__(self, out_dir: str | None = None):
        self._by_key: dict[PCABasisKey, PCABasisRecord] = {}
        self.out_dir = Path(out_dir) if out_dir else None

    def _basis_id(self, key: PCABasisKey, mean: torch.Tensor, components: torch.Tensor) -> str:
        h = hashlib.sha256()
        h.update(json.dumps(key.__dict__, sort_keys=True).encode())
        h.update(mean.detach().cpu().numpy().tobytes())
        h.update(components.detach().cpu().numpy().tobytes())
        return h.hexdigest()[:16]

    def fit(self, req: PCAFitRequest) -> PCABasisRecord:
        s, r, t = req.maps.shape
        if req.key.n_components > req.key.row_count:
            raise ValueError('n_components > row_count')
        if r != req.key.row_count:
            raise ValueError('row_count mismatch')
        x = req.maps.permute(0,2,1).reshape(-1, r)
        mu = x.mean(0, keepdim=True)
        xc = x - mu if req.key.centering else x
        _, sval, v = torch.linalg.svd(xc, full_matrices=False)
        w = v[:req.key.n_components].T
        for ci in range(w.shape[1]):
            idx = torch.argmax(torch.abs(w[:, ci]))
            if w[idx, ci] < 0:
                w[:, ci] *= -1
        ev = (sval[:req.key.n_components] ** 2) / max(1, (x.shape[0] - 1))
        evr = ev / torch.clamp(ev.sum(), min=1e-12)
        bid = self._basis_id(req.key, mu, w)
        rec = PCABasisRecord(basis_id=bid, key=req.key, mean=mu, components=w, explained_variance=ev, explained_variance_ratio=evr, row_count=r, n_components=req.key.n_components, created_from=req.created_from)
        self._by_key[req.key] = rec
        return rec

    def find(self, key: PCABasisKey) -> PCABasisRecord | None:
        return self._by_key.get(key)

    def apply(self, req: PCAApplyRequest) -> tuple[torch.Tensor, PCABasisRecord]:
        rec = self.find(req.key)
        if rec is None:
            raise ValueError('pca_basis_missing')
        s, r, t = req.maps.shape
        if r != rec.row_count:
            raise ValueError('row_count mismatch')
        if req.key.layer_index != rec.key.layer_index or req.key.layer_name != rec.key.layer_name:
            raise ValueError('layer mismatch')
        if req.key.signal_kind != rec.key.signal_kind:
            raise ValueError('signal_kind mismatch')
        if req.key.series != rec.key.series:
            raise ValueError('series mismatch')
        y = ((req.maps.permute(0,2,1) - rec.mean) @ rec.components).permute(0,2,1)
        return y, rec

    def save_tensor_artifact(self, rec: PCABasisRecord) -> str:
        if self.out_dir is None:
            raise ValueError('out_dir not configured')
        p = self.out_dir / 'pca_basis'
        p.mkdir(parents=True, exist_ok=True)
        fp = p / f'basis_{rec.basis_id}.pt'
        torch.save({'basis_id':rec.basis_id,'key':rec.key.__dict__,'mean':rec.mean,'components':rec.components,'explained_variance':rec.explained_variance,'explained_variance_ratio':rec.explained_variance_ratio}, fp)
        rec.artifact_path = str(fp)
        return str(fp)
