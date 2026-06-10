import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_empty_results(run_dir: Path):
    (run_dir / "results.csv").write_text("displacement,reaction_kN\n")


def _write_history(run_dir: Path):
    (run_dir / "history.csv").write_text(
        "step,applied_disp,reaction_force\n"
        "0,0.0000,0.0\n"
        "1,0.0125,-250.0\n"
    )


def test_miehe_tension_loads_history_when_results_has_no_rows(tmp_path):
    compare = _load_module(
        "miehe_tension_compare",
        "examples/quasistatic/miehe_tension/compare.py",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_empty_results(run_dir)
    _write_history(run_dir)

    u, reaction = compare.load_run_results(run_dir)

    assert u.tolist() == [0.0, 0.0125]
    assert reaction.tolist() == [0.0, 0.25]


def test_three_point_bending_loads_history_when_results_has_no_rows(tmp_path):
    compare = _load_module(
        "tpb_compare",
        "examples/quasistatic/three_point_bending/compare.py",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_empty_results(run_dir)
    _write_history(run_dir)

    u, reaction, source = compare.load_run(run_dir)

    assert source == "history.csv"
    assert u.tolist() == [0.0, 0.0125]
    assert reaction.tolist() == [0.0, 0.25]


def test_three_point_bending_prepeak_only_gate_is_explicit(tmp_path):
    compare_path = REPO_ROOT / "examples/quasistatic/three_point_bending/compare.py"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ref_path = (
        REPO_ROOT
        / "examples/quasistatic/three_point_bending/reference_solutions/"
        / "miehe_tpb_load_displacement.csv"
    )
    first = ref_path.read_text().splitlines()[0]
    ref = np.loadtxt(ref_path, delimiter="," if "," in first else None)
    subset = ref[ref[:, 0] <= 0.0386]
    lines = ["step,applied_disp,reaction_force"]
    for i, (u, reaction_kN) in enumerate(subset):
        lines.append(f"{i},{u:.8f},{-1000.0 * reaction_kN:.8f}")
    (run_dir / "history.csv").write_text("\n".join(lines) + "\n")

    strict = subprocess.run(
        [sys.executable, str(compare_path), "--run-dir", str(run_dir)],
        check=False,
        text=True,
        capture_output=True,
    )
    bounded = subprocess.run(
        [
            sys.executable,
            str(compare_path),
            "--run-dir",
            str(run_dir),
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


def test_notched_holed_loads_history_when_results_has_no_rows(tmp_path):
    compare = _load_module(
        "notched_holed_compare",
        "examples/quasistatic/notched_holed_plate/compare.py",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_empty_results(run_dir)
    _write_history(run_dir)

    u, reaction = compare.load_run_results(run_dir)

    assert u.tolist() == [0.0, 0.0125]
    assert reaction.tolist() == [0.0, 0.25]


def test_l_shaped_loads_history_when_results_has_no_rows(tmp_path):
    compare = _load_module(
        "l_shaped_compare",
        "examples/quasistatic/l_shaped_panel/compare.py",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_empty_results(run_dir)
    _write_history(run_dir)

    u, reaction, source = compare.load_run(run_dir)

    assert source == "history.csv"
    assert u.tolist() == [0.0, 0.0125]
    assert reaction.tolist() == [0.0, 0.25]
