"""TS-LIF origin wrapper placeholder.

The bundled origin file is a commented release artifact in this repository.
This module keeps an explicit wrapper entry-point and raises a clear error.
"""

from __future__ import annotations


class TSLIFNode:
    """Placeholder for TS-LIF node when source artifact is non-importable."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "TS-LIF origin file in this repository is commented out and cannot be imported directly."
        )
