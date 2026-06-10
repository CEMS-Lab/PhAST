"""Generate or check the checked-in JSON Schema for YAML configs."""

from __future__ import annotations

import os
import sys

from phast.config_schema import main

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(_REPO_ROOT, "configs", "phast.schema.json")


if __name__ == "__main__":
    argv = list(sys.argv[1:])
    if "--output" not in argv and "-o" not in argv:
        argv = ["--output", OUTPUT_PATH] + argv
    raise SystemExit(main(argv))
