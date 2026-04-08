"""SHD dataset compatibility exports for the reorganized src/data tree."""

from src.data.bundle import EventH5Dataset, get_shd_loaders

__all__ = ['EventH5Dataset', 'get_shd_loaders']
