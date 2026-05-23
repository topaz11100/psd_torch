from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from psd_snn.artifacts.reader import ArtifactReader


def plot_spectral_curve(reader: ArtifactReader, out_path: str):
    rows = reader.read_spectral_curve(); xs=[float(r['frequency']) for r in rows]; ys=[float(r['value']) for r in rows]
    plt.figure(); plt.plot(xs, ys); plt.tight_layout(); plt.savefig(out_path); plt.close()


def plot_spectral_matrix_1d(reader: ArtifactReader, out_path: str):
    rows = reader.read_spectral_matrix_1d(); rmax=max(int(r['row_index']) for r in rows)+1; fmax=max(int(r['frequency_index']) for r in rows)+1
    m=np.zeros((rmax,fmax));
    for r in rows: m[int(r['row_index']), int(r['frequency_index'])]=float(r['value'])
    plt.figure(); plt.imshow(m, aspect='auto'); plt.colorbar(); plt.tight_layout(); plt.savefig(out_path); plt.close()


def plot_spectral_matrix_2d(reader: ArtifactReader, out_path: str):
    rows=reader.read_spectral_matrix_2d(); cols=[k for k in rows[0].keys() if k.startswith('time_freq_')]
    m=np.array([[float(r[c]) for c in cols] for r in rows])
    plt.figure(); plt.imshow(m, aspect='auto'); plt.colorbar(); plt.tight_layout(); plt.savefig(out_path); plt.close()


def plot_pca_explained_variance(reader: ArtifactReader, out_path: str):
    rows=reader.read_pca_basis_metadata(); rows=sorted(rows, key=lambda r:int(r['component_id']))
    xs=[int(r['component_id']) for r in rows]; ys=[float(r['explained_variance_ratio']) for r in rows]
    plt.figure(); plt.bar(xs, ys); plt.tight_layout(); plt.savefig(out_path); plt.close()


def plot_trace_chunk(reader: ArtifactReader, out_path: str):
    rows=reader.read_trace_manifest(); valid=[r for r in rows if r.get('path')]
    if not valid: raise ValueError('no trace chunk path')
    t=reader.load_trace_chunk(valid[0]['path'])
    m=t[0].detach().cpu().numpy()
    plt.figure(); plt.imshow(m, aspect='auto'); plt.tight_layout(); plt.savefig(out_path); plt.close()
