"""Helpers for stable importing of Origin modules with spaced paths."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]
ORIGIN_ROOT = ROOT / "Origin"


def load_module_from_path(module_name: str, rel_path: str) -> ModuleType:
    """Load a Python module from Origin/ relative path."""

    target = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, target)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module: {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
