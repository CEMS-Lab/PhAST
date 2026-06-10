"""Backward-compatible shim for :mod:`phast.physics.boundary_conditions`."""
from importlib import import_module as _import_module
import sys as _sys

_module = _import_module("phast.physics.boundary_conditions")
globals().update({
    _name: _value for _name, _value in vars(_module).items()
    if _name not in {"__builtins__", "__loader__", "__spec__"}
})
_sys.modules[__name__] = _module
__all__ = [name for name in vars(_module) if not name.startswith("__")]
