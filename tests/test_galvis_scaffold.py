"""Import-sanity check for the Galvis 2026 cross-validation scaffold (#109)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMPARE = REPO / "examples" / "quasistatic" / "galvis_validation" / "compare.py"


def test_compare_help_runs():
    result = subprocess.run(
        [sys.executable, str(COMPARE), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Galvis" in result.stdout or "galvis" in result.stdout


def test_compare_placeholder_short_circuit():
    result = subprocess.run(
        [sys.executable, str(COMPARE), "--benchmark", "sent"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "PLACEHOLDER" in result.stdout
