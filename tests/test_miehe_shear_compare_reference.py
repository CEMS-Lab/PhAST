import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARE_PATH = REPO_ROOT / "examples/quasistatic/miehe_shear/compare.py"
SPEC = importlib.util.spec_from_file_location("miehe_shear_compare", COMPARE_PATH)
compare = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = compare
SPEC.loader.exec_module(compare)


PHASEFIELDX_TOP = (
    REPO_ROOT
    / "reference_codes/phasefieldx-main/examples/PhaseFieldFracture"
    / "1712_Single_Edge_Notched_Shear_Test/top.dof"
)
PHASEFIELDX_REACTION = PHASEFIELDX_TOP.with_name("bottom.reaction")


def test_shipped_sens_reference_peak_is_current_miehe_style_gate():
    u, reaction, label = compare.load_reference("shipped")
    peak_i = reaction.argmax()

    assert label == "miehe_sens_load_displacement.csv"
    assert reaction[peak_i] == 0.53118
    assert u[peak_i] == 0.0094


def test_phasefieldx_output_reference_peak_matches_bundled_1712_artifacts():
    if not PHASEFIELDX_TOP.exists() or not PHASEFIELDX_REACTION.exists():
        pytest.skip("PhaseFieldX 1712 executable-output artifacts are not bundled")
    u, reaction, label = compare.load_reference("phasefieldx-output")
    peak_i = reaction.argmax()

    assert label == "phasefieldx_1712_executable_output"
    assert abs(float(reaction[peak_i]) - 0.4946828184283757) < 1e-12
    assert abs(float(u[peak_i]) - 0.0087) < 1e-12


def test_reference_tier_maps_to_expected_sources():
    assert compare.REFERENCE_TIERS["miehe-paper"] == "shipped"
    assert compare.REFERENCE_TIERS["phasefieldx-parity"] == "phasefieldx-output"
    assert compare.tier_from_source("shipped") == "miehe-paper"
    assert compare.tier_from_source("phasefieldx-output") == "phasefieldx-parity"


def test_compare_can_write_nondefault_artifact_names(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ref_csv = tmp_path / "reference.csv"
    ref_csv.write_text(
        "0.0000 0.00000\n"
        "0.0020 0.12000\n"
        "0.0040 0.25000\n"
        "0.0060 0.36000\n"
        "0.0087 0.4946828184283757\n"
        "0.0100 0.45000\n"
    )
    (run_dir / "results.csv").write_text(
        "displacement,reaction_kN\n"
        "0.0000,0.00000\n"
        "0.0020,0.12000\n"
        "0.0040,0.25000\n"
        "0.0060,0.36000\n"
        "0.0087,0.4946828184283757\n"
        "0.0100,0.45000\n"
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(COMPARE_PATH),
            "--run-dir",
            str(run_dir),
            "--reference-csv",
            str(ref_csv),
            "--reference-tier",
            "phasefieldx-parity",
            "--report-name",
            "compare_phasefieldx_output_report.txt",
            "--figure-name",
            "compare_phasefieldx_output.png",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not (run_dir / "compare_report.txt").exists()
    assert not (run_dir / "compare.png").exists()
    assert (run_dir / "compare_phasefieldx_output_report.txt").exists()
    assert (run_dir / "compare_phasefieldx_output.png").exists()
    report = (run_dir / "compare_phasefieldx_output_report.txt").read_text()
    assert "Reference tier  : custom-csv" in report


def test_prepeak_only_gate_allows_bounded_sens_artifact_run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ref_csv = tmp_path / "reference.csv"
    ref_csv.write_text(
        "0.0000 0.00000\n"
        "0.0020 0.12000\n"
        "0.0040 0.25000\n"
        "0.0060 0.36000\n"
        "0.0087 0.49468\n"
        "0.0094 0.53118\n"
    )
    (run_dir / "history.csv").write_text(
        "step,applied_disp,reaction_force\n"
        "0,0.0000,0.0\n"
        "1,0.0020,-120.0\n"
        "2,0.0040,-250.0\n"
        "3,0.0060,-360.0\n"
        "4,0.0087,-494.68\n"
    )

    strict = subprocess.run(
        [
            sys.executable,
            str(COMPARE_PATH),
            "--run-dir",
            str(run_dir),
            "--reference-csv",
            str(ref_csv),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    bounded = subprocess.run(
        [
            sys.executable,
            str(COMPARE_PATH),
            "--run-dir",
            str(run_dir),
            "--reference-csv",
            str(ref_csv),
            "--prepeak-only",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert strict.returncode == 1
    assert bounded.returncode == 0, bounded.stdout + bounded.stderr
    assert "Gate mode               : pre-peak only" in (
        run_dir / "compare_report.txt").read_text()


def test_load_run_falls_back_to_history_when_results_has_no_rows(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "results.csv").write_text("displacement,reaction_kN\n")
    (run_dir / "history.csv").write_text(
        "step,applied_disp,reaction_force\n"
        "0,0.0000,0.0\n"
        "1,0.0087,-494.6828184283757\n"
    )

    u, reaction, source = compare.load_run(run_dir)

    assert source == "history.csv"
    assert u.tolist() == [0.0, 0.0087]
    assert abs(float(reaction[-1]) - 0.4946828184283757) < 1e-12
