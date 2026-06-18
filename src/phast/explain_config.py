"""Compatibility wrapper for :mod:`phast.config.explain_config`."""

from .config.explain_config import *  # noqa: F401,F403
from .config.explain_config import main as _main


if __name__ == "__main__":
    raise SystemExit(_main())
