"""Smoke test for the FD-vs-autograd wall-clock figure script.

Runs the generator on a tiny config and asserts:
  * the script exits 0
  * a non-empty PNG and JSON are produced
  * the JSON timing rows are well-formed and the autograd time at
    P=64 is strictly less than the projected FD total at P=64
    (i.e. the qualitative O(P) vs O(1) story holds)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "papers/paper2" / "scripts" / "make_fd_vs_autograd_walltime.py"


def test_script_runs_and_produces_artifacts(tmp_path: Path) -> None:
    if not SCRIPT.exists():
        pytest.skip(f"{SCRIPT} not present")
    out_dir = tmp_path / "figs"
    out_dir.mkdir()
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--Ps", "1", "4", "16", "64",
        "--n_steps", "10",
        "--n_repeats", "1",
        "--out_dir", str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, (
        f"script failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )

    png = out_dir / "fd_vs_autograd_walltime.png"
    js = out_dir / "fd_vs_autograd_walltime.json"
    assert png.exists() and png.stat().st_size > 1024, (
        f"png missing or empty: {png}"
    )
    assert js.exists() and js.stat().st_size > 50

    with open(js) as fp:
        payload = json.load(fp)
    rows = {r["P"]: r for r in payload["rows"]}
    # Qualitative claim: at P=64 the projected FD total exceeds the
    # autograd wall-clock. (Holds with massive margin; loose factor
    # of 5 just so the test isn't fragile to noise.)
    assert rows[64]["t_fd_total_projected"] > 5.0 * rows[64]["t_autograd"]
