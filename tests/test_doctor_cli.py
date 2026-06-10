import subprocess
import sys

from phast.doctor import build_report


def test_doctor_report_mentions_workflow_defaults():
    report = build_report()
    assert "backend='auto'" in report
    assert "quasi-static fracture" in report
    assert "explicit dynamics" in report
    assert "PETSc/MUMPS" in report
    assert "Zarr" in report


def test_doctor_cli_runs():
    proc = subprocess.run(
        [sys.executable, "-m", "phast", "doctor"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "PhAST environment doctor" in proc.stdout
    assert "Recommended problem-class defaults" in proc.stdout
