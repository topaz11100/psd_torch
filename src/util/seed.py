"""Global seed management for deterministic reproducibility."""

from __future__ import annotations

import os
import random

import numpy as np


try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Set Python/NumPy/(optional)PyTorch seeds for full reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if hasattr(torch, "use_deterministic_algorithms"):
                torch.use_deterministic_algorithms(True, warn_only=True)


def make_worker_init_fn(base_seed: int):
    """Create dataloader worker init function with deterministic per-worker seed."""

    def _init(worker_id: int) -> None:
        worker_seed = int(base_seed) + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        if torch is not None:
            torch.manual_seed(worker_seed)

    return _init
