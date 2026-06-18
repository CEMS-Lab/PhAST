"""Compatibility helpers for legacy module import paths."""

from __future__ import annotations

from importlib import import_module
import sys
from types import ModuleType


def alias_module(alias: str, target: str) -> ModuleType:
    """Expose ``target`` under the already-imported module name ``alias``.

    Several public imports from early PhAST releases used flat module paths
    such as ``phast.mesh`` and ``phast.damage_solver``. The implementation now
    lives in organized subpackages, but these imports remain supported by
    installing the target module object under the legacy name.
    """

    module = import_module(target)
    sys.modules[alias] = module
    return module
