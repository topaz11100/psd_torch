"""Filesystem-level compatibility wrapper.

This file keeps the requested tree shape. Importable code should prefer
``src.data.s_mnist`` because Python module names cannot contain ``-``.
"""

from src.data.s_mnist import SequentialMNIST

__all__ = ['SequentialMNIST']
