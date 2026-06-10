"""Smoke tests for the --time_integrator opt-in flag (#102/#570/#573).

Verifies the CLI surface added to ``run_config.py`` for the dynamic
benchmarks:

* ``--time_integrator central_difference`` - default Velocity-Verlet /
  explicit Newmark central-difference path.
* ``--time_integrator verlet`` / ``newmark`` - accepted aliases for
  the same path (``newmark`` is legacy naming).
* ``--time_integrator generalized_alpha`` / ``gen_alpha`` - accepted
  aliases for the opt-in implicit dynamic path.

We intentionally do **not** spawn a full B5/B1 dynamic run here; that
would need an HPC mesh. Instead we use ``--validate-only`` for the
central-difference path (cheap schema-only run) and exercise aliases via
schema-only validation.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1].parent
CONFIG = REPO_ROOT / "phast" / "configs" / "B5_pmma_branching.yaml"


def _run_module(*extra_args: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "phast", "run", str(CONFIG), *extra_args]
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=120, env=env, cwd=str(REPO_ROOT),
    )


@pytest.mark.slow
def test_central_difference_default_validate_only():
    """Default central-difference path must accept the YAML cleanly."""
    if not CONFIG.exists():
        pytest.skip(f"config not found: {CONFIG}")
    res = _run_module("--validate-only")
    assert res.returncode == 0, f"stderr={res.stderr}\nstdout={res.stdout}"


@pytest.mark.slow
@pytest.mark.parametrize("alias", ["central_difference", "verlet", "newmark"])
def test_central_difference_aliases_validate_only(alias):
    """Canonical, explicit-Verlet, and legacy Newmark aliases all validate."""
    if not CONFIG.exists():
        pytest.skip(f"config not found: {CONFIG}")
    res = _run_module("--validate-only", "--time_integrator", alias)
    assert res.returncode == 0, f"stderr={res.stderr}\nstdout={res.stdout}"


@pytest.mark.slow
@pytest.mark.parametrize("alias", ["generalized_alpha", "gen_alpha"])
def test_genalpha_aliases_validate_only(alias):
    """Generalized-alpha aliases are accepted by the CLI."""
    if not CONFIG.exists():
        pytest.skip(f"config not found: {CONFIG}")
    res = _run_module("--validate-only", "--time_integrator", alias)
    assert res.returncode == 0, f"stderr={res.stderr}\nstdout={res.stdout}"
