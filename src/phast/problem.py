"""Backward-compatible shim for :mod:`phast.core.problem`."""
from importlib import import_module as _import_module
import sys as _sys
import types as _types

_module = _import_module("phast.core.problem")
globals().update({
    _name: _value for _name, _value in vars(_module).items()
    if _name not in {
        "__builtins__", "__cached__", "__file__", "__loader__", "__name__",
        "__package__", "__spec__",
    }
})


class _ForwardingModule(_types.ModuleType):
    def __setattr__(self, name, value):
        setattr(_module, name, value)
        super().__setattr__(name, value)


_sys.modules[__name__].__class__ = _ForwardingModule
__all__ = [name for name in vars(_module) if not name.startswith("__")]
