"""The common explanation command inspects inputs without starting a simulation."""
from pathlib import Path
import os
import subprocess
import sys

import pytest

from phast.config.explain_config import build_explanation


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("relative_path", [
    "standard_workflow/quasistatic_sent/config.yaml",
    "standard_workflow/dynamic_sent/config.yaml",
    "two_material_dcb_beta/config.yaml",
    "two_material_dcb_beta/comparisons/tough_region.yaml",
    "two_material_dcb_beta/comparisons/uniform_layer.yaml",
])
def test_schema_v2_explanation_cli(relative_path, tmp_path):
    config = ROOT / "examples" / relative_path
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT / "src"), str(ROOT))))
    result = subprocess.run(
        [sys.executable, "-m", "phast", "explain-config", str(config)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for heading in ("Schema version: 2", "Materials and assignments", "Boundary conditions",
                    "Analysis steps", "Solver (requested)", "Outputs", "--validate-only"):
        assert heading in result.stdout
    assert "no mesh, solver, or output directory was created" in result.stdout
    assert not (tmp_path / "runs").exists()


def test_explanation_does_not_resolve_multimaterial_mesh(monkeypatch):
    def must_not_build(*args, **kwargs):
        pytest.fail("explanation attempted to build the geometric mesh")

    monkeypatch.setattr("phast.workflow.multimaterial_fracture._build_mesh", must_not_build)
    report, code = build_explanation(str(ROOT / "examples/two_material_dcb_beta/config.yaml"))
    assert code == 0, report
    assert "Declared settings only" in report


def test_invalid_schema_v2_reports_an_input_error(tmp_path):
    config = tmp_path / "invalid.yaml"
    config.write_text("schema_version: 2\nname: incomplete\n", encoding="utf-8")
    report, code = build_explanation(str(config))
    assert code == 2
    assert "Error in" in report
