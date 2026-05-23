from __future__ import annotations
import csv
from pathlib import Path
import torch


class ArtifactReader:
    def __init__(self, out_dir: str):
        self.out = Path(out_dir)

    def _read_csv(self, name: str):
        p = self.out / name
        if not p.exists():
            raise FileNotFoundError(str(p))
        with p.open() as f:
            return list(csv.DictReader(f))

    def read_analysis_manifest(self): return self._read_csv('analysis_manifest.csv')
    def read_spectral_curve(self): return self._read_csv('spectral_curve.csv')
    def read_spectral_matrix_1d(self): return self._read_csv('spectral_matrix_1d.csv')
    def read_spectral_matrix_2d(self): return self._read_csv('spectral_matrix_2d.csv')
    def read_spectral_matrix_2d_axes(self):
        return self._read_csv('spectral_matrix_2d_row_axis.csv'), self._read_csv('spectral_matrix_2d_column_axis.csv')
    def read_pca_basis_metadata(self): return self._read_csv('pca_basis.csv')
    def read_trace_manifest(self): return self._read_csv('trace_manifest.csv')

    def load_pca_basis_tensor(self, path: str):
        p = Path(path)
        if not p.is_absolute(): p = self.out / p
        if not p.exists(): raise FileNotFoundError(str(p))
        return torch.load(p, map_location='cpu')

    def load_trace_chunk(self, path: str):
        p = Path(path)
        if not p.is_absolute(): p = self.out / p
        if not p.exists(): raise FileNotFoundError(str(p))
        return torch.load(p, map_location='cpu')
