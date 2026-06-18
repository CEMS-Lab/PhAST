"""Compatibility wrapper for :mod:`phast.config.run_config`."""

from .config.run_config import *  # noqa: F401,F403
from .config.run_config import main as _main


if __name__ == "__main__":
    _main()
