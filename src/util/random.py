"""Global random-seed helpers.

The project relies on deterministic probe selection and reproducible training
loops. These helpers centralize the seed / worker policy so the entrypoints do
not silently diverge.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


_DEFAULT_CUBLAS_WORKSPACE_CONFIG = ':4096:8'


def build_torch_generator(seed: int | None) -> torch.Generator | None:
    """Return one deterministic ``torch.Generator`` when a seed is provided.

    Using an explicit generator makes DataLoader shuffling and worker base-seed
    assignment independent from unrelated global RNG consumption.
    """

    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def seed_dataloader_worker(_worker_id: int) -> None:
    """Seed Python / NumPy / Torch inside one DataLoader worker.

    The worker also clamps its intra-op thread count to one. Without this, a
    multi-worker DataLoader can oversubscribe CPU cores because each worker may
    try to spawn its own OpenMP thread pool.
    """

    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def seed_everything(seed: int) -> int:
    """Seed Python, NumPy, and Torch deterministically.

    This function is called once near each official entrypoint / driver start so
    that Python hashing, NumPy RNG, Torch CPU RNG, Torch CUDA RNG, and cuDNN
    algorithm selection stay aligned.
    """

    seed = int(seed)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', _DEFAULT_CUBLAS_WORKSPACE_CONFIG)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    return seed


__all__ = ['build_torch_generator', 'seed_dataloader_worker', 'seed_everything']
