"""Backward-compatible alias for :mod:`phast.config.config_validation`."""

from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("phast.config.config_validation")
