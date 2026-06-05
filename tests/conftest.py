"""Pytest-wide runtime guards for CPU-only CI."""

from __future__ import annotations

import os

# Must run before individual test modules import torch/numpy.  The project is
# GPU-first, but CI smoke tests execute small 784-step SNNs on CPU; uncapped BLAS
# threads can oversubscribe and get subprocess tests killed on constrained hosts.
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
os.environ.setdefault('PSD_TORCH_CPU_THREADS', '1')
