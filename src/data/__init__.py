"""Dataset-facing compatibility wrappers for the reorganized src/data tree."""

from .SHD import EventH5Dataset, get_shd_loaders
from .s_mnist import SequentialMNIST

__all__ = ['EventH5Dataset', 'get_shd_loaders', 'SequentialMNIST']
