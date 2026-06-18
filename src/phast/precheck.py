"""Compatibility wrapper for :mod:`phast.config.precheck`."""

from .config.precheck import *  # noqa: F401,F403
from .config.precheck import main as _main


if __name__ == "__main__":
    _main()
