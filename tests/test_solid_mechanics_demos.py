"""Subprocess smoke tests for packaged solid-mechanics demos (#106/#105).

Each demo is launched as a real subprocess; we assert exit-0 and a stable
anchor substring in stdout. Marked ``slow`` so the default ``pytest`` run
(``addopts = "-m 'not slow'"``) skips them; opt in with ``pytest -m slow``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
DEMO_DIR = REPO_ROOT / "examples" / "solid_mechanics"


def _run_demo(filename: str, anchor: str, timeout: int = 60) -> None:
    """Run a demo script and check it exits 0 with the expected anchor."""
    env = os.environ.copy()
    # Symlink avoids cloud-synced paths-with-spaces issues in PYTHONPATH.
    link_dir = Path("/tmp/tps_link_smoke")
    link_dir.mkdir(parents=True, exist_ok=True)
    link = link_dir / "phast_src"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(SRC_ROOT)
    env["PYTHONPATH"] = str(link) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, str(DEMO_DIR / filename)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"{filename} exited {result.returncode}\n"
        f"--- stderr (tail) ---\n{result.stderr[-500:]}"
    )
    assert anchor in result.stdout, (
        f"{filename} stdout missing anchor '{anchor}'\n"
        f"--- stdout (tail) ---\n{result.stdout[-500:]}"
    )


@pytest.mark.slow
def test_linear_plate_runs():
    _run_demo("linear_plate.py", "tip displacement")


@pytest.mark.slow
def test_neohookean_plate_runs():
    _run_demo("neohookean_plate.py", "u_tip", timeout=180)


@pytest.mark.slow
def test_mixed_precision_cg_demo_runs():
    _run_demo("mixed_precision_cg_demo.py", "Mixed-precision CG demo")


@pytest.mark.slow
def test_dynamic_oscillator_genalpha_runs():
    _run_demo("dynamic_oscillator_genalpha.py", "rho_inf=")


@pytest.mark.slow
def test_j2_plasticity_bar_runs():
    _run_demo("j2_plasticity_bar.py", "final vm-yield residual")
