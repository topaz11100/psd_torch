"""전역 시드 헬퍼 모음."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


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
    """Python/NumPy/Torch 시드를 고정하되 deterministic 모드는 사용하지 않는다."""

    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Avoid calling torch.use_deterministic_algorithms(False) here.  That no-op
    # can trigger extra compiler/runtime imports on recent PyTorch builds during
    # compile=false cold starts.  Deterministic mode is already off by default;
    # callers that need strict determinism should opt in explicitly.
    return seed


__all__ = ['build_torch_generator', 'seed_dataloader_worker', 'seed_everything']
