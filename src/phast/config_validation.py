"""Backward-compatible shim for :mod:`phast.config.config_validation`."""
from importlib import import_module as _import_module
import sys as _sys

_module = _import_module("phast.config.config_validation")
globals().update({
    _name: _value for _name, _value in vars(_module).items()
    if _name not in {
        "__builtins__", "__cached__", "__file__", "__loader__", "__name__",
        "__package__", "__spec__",
    }
})

_canonical_node_sets_from_mesh_file = _module._node_sets_from_mesh_file


def _sync_monkeypatched_helpers() -> None:
    """Forward legacy-module monkeypatches to the canonical implementation."""
    helper = globals().get("_node_sets_from_mesh_file")
    if helper is not None and helper is not _module._node_sets_from_mesh_file:
        _module._node_sets_from_mesh_file = helper


def validate_config_file(*args, **kwargs):
    _sync_monkeypatched_helpers()
    return _module.validate_config_file(*args, **kwargs)


def validate_config_file_with_warnings(*args, **kwargs):
    _sync_monkeypatched_helpers()
    return _module.validate_config_file_with_warnings(*args, **kwargs)


def validate_config(*args, **kwargs):
    _sync_monkeypatched_helpers()
    return _module.validate_config(*args, **kwargs)


def assert_valid(*args, **kwargs):
    _sync_monkeypatched_helpers()
    return _module.assert_valid(*args, **kwargs)

__all__ = [name for name in vars(_module) if not name.startswith("__")]
