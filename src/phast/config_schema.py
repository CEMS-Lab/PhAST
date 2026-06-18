"""Compatibility wrapper for :mod:`phast.config.config_schema`."""

from .config.config_schema import *  # noqa: F401,F403
from .config.config_schema import main as _main


if __name__ == "__main__":
    raise SystemExit(_main())
