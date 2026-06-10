"""Pytest config for the src/-layout PhAST repo.

The importable package lives at ``src/phast/``. After
``pip install -e .`` the package is on ``sys.path`` via the editable
install, so tests can simply ``from phast.X import ...``.

We do two things here:

1. Point pytest at ``src/`` for the worktree-under-test case where the
   editable install has not been refreshed (e.g. a worker that pulled a
   new branch but did not re-install). Combined with the stale-module
   check below, this guarantees that the package imported by the test
   suite is the one in *this* checkout, not whatever lives in
   ``site-packages``.

2. Tell pytest to ignore non-test directories that contain ``__init__.py``
   for packaging reasons but should never be collected as test targets.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_SRC = _REPO_ROOT / "src"

# Prepend src/ so `import phast` resolves to this checkout even
# if a stale editable install points elsewhere. Idempotent.
_src_str = str(_SRC)
if _src_str not in sys.path:
    sys.path.insert(0, _src_str)

# Drop a stale phast if it was imported from a non-checkout path.
import importlib  # noqa: E402

if "phast" in sys.modules:
    _existing = sys.modules["phast"]
    _existing_path = getattr(_existing, "__file__", "") or ""
    if not _existing_path.startswith(str(_SRC)):
        for _name in [
            n for n in sys.modules
            if n == "phast" or n.startswith("phast.")
        ]:
            del sys.modules[_name]
        importlib.import_module("phast")

collect_ignore = [
    "src",
    "examples",
    "scripts",
    "papers/paper",
    "docs",
    "configs",
    "notebooks",
    "reports",
    "reference_solutions",
    "handoff_akantu",
]
