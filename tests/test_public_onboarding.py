from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import phast
import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_command(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args], cwd=ROOT, check=True,
        text=True, capture_output=True,
    )


def test_documented_config_validation_and_explanation():
    config = "examples/dynamic/B2_kalthoff_winkler/config.yaml"
    validated = run_command("-m", "phast", "run", config, "--validate-only")
    assert "passes schema validation" in validated.stdout
    explained = run_command("-m", "phast", "explain-config", config)
    assert "Kalthoff-Winkler" in explained.stdout
    assert "h/l0" in explained.stdout


def test_linear_plate_newcomer_run_and_result_loading(tmp_path: Path):
    output = tmp_path / "linear_plate"
    run_command(
        "-m", "phast", "run",
        "examples/solid_mechanics_beta/linear_plate/config.yaml",
        "--output_dir", str(output),
    )
    for filename in (
        "response.csv", "deformed_shape.png", "run_manifest.json",
        "run_metadata.json", "visual_manifest.json",
    ):
        assert (output / filename).is_file()
    result = phast.load_result(output)
    assert result.metadata()["example"] == "solid_mechanics.linear_plate"
    assert result.manifest()["metrics"]["relative_error_percent"] == pytest.approx(-14.983836063497574)
    assert result.visuals()


@pytest.mark.parametrize(
    ("config_text", "expected_error"),
    [
        (
            """schema_version: 1
example: solid_mechanics.linear_plate
material: {E: 2.1e11, nu: 0.3}
loading: {tip_force_y: -1000.0}
""",
            "mesh must be a mapping",
        ),
        (
            """schema_version: 1
example: solid_mechanics.linear_plate
mesh: {nx: -3, ny: 10, length: 1.0, height: 0.2}
material: {E: not-a-number, nu: 0.3}
loading: {tip_force_y: -1000.0}
""",
            "material.E must be a number",
        ),
    ],
)
def test_solid_validate_only_rejects_invalid_configs(
    tmp_path: Path, config_text: str, expected_error: str,
):
    config = tmp_path / "invalid_linear_plate.yaml"
    config.write_text(config_text, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "phast", "run", str(config), "--validate-only"],
        cwd=ROOT, check=False, text=True, capture_output=True,
    )
    assert completed.returncode == 2
    assert expected_error in completed.stderr


def test_heterogeneous_field_teaching_example(tmp_path: Path):
    output = tmp_path / "heterogeneous_fields"
    run_command(
        "examples/heterogeneous_fields/run.py",
        "--config", "examples/heterogeneous_fields/parameters.yaml",
        "--output-dir", str(output),
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["damage_converged"]
    assert 0.0 <= summary["damage_min"] <= summary["damage_max"] <= 1.0
    assert summary["E_min"] < summary["E_max"]
    assert summary["Gc_min"] < summary["Gc_max"]
    for filename in (
        "material_fields.csv", "damage.csv", "material_fields.png",
        "damage_final.png", "run_manifest.json", "run_metadata.json",
        "run_lockfile.json", "visual_manifest.json",
    ):
        assert (output / filename).is_file()
