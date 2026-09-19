"""The Python ProblemSpec and YAML CLI execute the same linear plate."""

import csv
import json
import os
from runpy import run_path
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest

from phast import Problem
from phast.workflow import run_problem_spec
from phast.workflow.execution import WorkflowExecutionError, execution_plan_from_spec
from phast.workflow.specs import BoundaryConditionSpec, InitialConditionSpec, MeshSpec


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples/solid_mechanics_beta/linear_plate/config.yaml"
EXAMPLE_SETUP = run_path(
    str(ROOT / "examples/solid_mechanics_beta/linear_plate/fluent_setup.py"),
    run_name="phast_linear_plate_setup_test",
)


def _problem() -> Problem:
    return EXAMPLE_SETUP["build_problem"]()


def _response(path: Path) -> dict[str, float]:
    with path.open(newline="") as stream:
        return {row["quantity"]: float(row["value"]) for row in csv.DictReader(stream)}


def test_python_spec_and_yaml_cli_match_for_linear_plate(tmp_path: Path) -> None:
    yaml_out = tmp_path / "yaml"
    python_out = tmp_path / "python"
    subprocess.run(
        [sys.executable, "-m", "phast", "run", str(CONFIG), "--output_dir", str(yaml_out)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    spec = _problem().to_spec()
    assert spec.source_path is None
    assert spec.geometry is not None
    assert spec.geometry.units == "m"
    assert "units" not in spec.geometry.parameters
    assert execution_plan_from_spec(spec).direct_execution_supported
    validate_out = tmp_path / "validate_only"
    assert run_problem_spec(spec, output_dir=validate_out, validate_only=True) == 0
    assert not validate_out.exists()
    assert run_problem_spec(spec, output_dir=python_out) == 0

    yaml_manifest = json.loads((yaml_out / "run_manifest.json").read_text())
    python_manifest = json.loads((python_out / "run_manifest.json").read_text())
    for section, keys in (
        ("mesh", ("nx", "ny", "length", "height")),
        ("material", ("E", "nu")),
        ("loading", ("tip_force_y",)),
    ):
        for key in keys:
            assert float(python_manifest["config"][section][key]) == pytest.approx(
                float(yaml_manifest["config"][section][key])
            )

    assert python_manifest["metrics"] == pytest.approx(yaml_manifest["metrics"])
    assert _response(python_out / "response.csv") == pytest.approx(
        _response(yaml_out / "response.csv")
    )
    with (python_out / "response.csv").open(newline="") as stream:
        response_rows = {row["quantity"]: row for row in csv.DictReader(stream)}
    assert response_rows["tip_displacement_fe"]["unit"] == spec.geometry.units
    for field_image in (
        "displacement_magnitude.png",
        "von_mises.png",
        "strain_energy.png",
    ):
        np.testing.assert_array_equal(
            mpimg.imread(python_out / field_image),
            mpimg.imread(yaml_out / field_image),
        )


def test_python_spec_rejects_unknown_source_before_execution(tmp_path: Path) -> None:
    spec = replace(_problem().to_spec(), source="python:other")
    output_dir = tmp_path / "unsupported"
    with pytest.raises(WorkflowExecutionError, match="original YAML source_path"):
        run_problem_spec(spec, output_dir=output_dir)
    assert not output_dir.exists()


@pytest.mark.parametrize("directory", [None, "relative/results"])
def test_python_spec_preserves_default_and_relative_results(
    directory: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _problem().to_spec()
    spec = replace(
        base,
        geometry=replace(base.geometry, parameters={"nx": 2, "ny": 2, "length": 1.0, "height": 0.2}),
        outputs=replace(base.outputs, directory=directory),
    )
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(ROOT / "src"), str(ROOT)]))
    monkeypatch.chdir(tmp_path)
    destination = tmp_path / (directory or "outputs")
    assert run_problem_spec(spec, validate_only=True) == 0
    assert not destination.exists()
    assert run_problem_spec(spec) == 0
    assert (destination / "run_manifest.json").is_file()
    assert (destination / "response.csv").is_file()
    assert (destination / "displacement_magnitude.png").is_file()


def test_problem_geometry_keeps_default_units_out_of_generator_parameters() -> None:
    geometry = Problem("Default geometry units").geometry("structured_grid", nx=2, ny=1)
    assert geometry.config.geometry.units == "mm"
    assert "units" not in geometry.config.geometry.parameters


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("materials", "exactly one material"),
        ("steps", "exactly one analysis step"),
        ("boundary_conditions", "explicit boundary conditions"),
        ("material_model", "material model"),
        ("step_kind", "analysis step kind"),
        ("mesh_path", "mesh file"),
        ("initial_conditions", "initial conditions"),
        ("active_boundary_conditions", "active boundary conditions"),
        ("region_selector", "domain region"),
        ("geometry_kind", "structured_grid geometry"),
        ("geometry_option", "unsupported geometry parameters"),
        ("material_option", "unsupported material parameters"),
        ("load_option", "unsupported load controls"),
        ("solver_option", "nondefault solver settings"),
        ("output_option", "unsupported output settings"),
        ("plots_disabled", "requires plots=True"),
        ("strain_field", "unsupported field output"),
        ("stress_field", "unsupported field output"),
        ("other_example", "only solid_mechanics.linear_plate"),
    ],
)
def test_python_solid_bridge_rejects_unlowered_inputs(
    case: str, message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _problem().to_spec()
    invalid = {
        "materials": replace(
            base,
            materials=base.materials + [
                replace(base.materials[0], name="other", parameters={"E": 1.0, "nu": 0.2})
            ],
        ),
        "steps": replace(
            base,
            analysis_steps=base.analysis_steps + [
                replace(base.analysis_steps[0], name="second", controls={"tip_force_y": -2000.0})
            ],
        ),
        "boundary_conditions": replace(
            base, boundary_conditions=[BoundaryConditionSpec(kind="fix", region="body")]
        ),
        "material_model": replace(
            base, materials=[replace(base.materials[0], model="phase_field")]
        ),
        "step_kind": replace(
            base, analysis_steps=[replace(base.analysis_steps[0], kind="quasi_static")]
        ),
        "mesh_path": replace(base, mesh=MeshSpec(path="plate.msh")),
        "initial_conditions": replace(
            base, initial_conditions=[InitialConditionSpec(field="displacement", value=0.0)]
        ),
        "active_boundary_conditions": replace(
            base,
            analysis_steps=[
                replace(base.analysis_steps[0], active_boundary_conditions=("custom",))
            ],
        ),
        "region_selector": replace(
            base, regions=[replace(base.regions[0], selector={"from_mesh": "Other"})]
        ),
        "geometry_kind": replace(base, geometry=replace(base.geometry, kind="rectangular_sent")),
        "geometry_option": replace(
            base, geometry=replace(base.geometry, parameters={**base.geometry.parameters, "thickness": 1.0})
        ),
        "material_option": replace(
            base,
            materials=[replace(base.materials[0], parameters={**base.materials[0].parameters, "rho": 1.0})],
        ),
        "load_option": replace(
            base,
            analysis_steps=[
                replace(base.analysis_steps[0], controls={"tip_force_y": -1000.0, "ramp": 2.0})
            ],
        ),
        "solver_option": replace(
            base, solver=replace(base.solver, parameters={**base.solver.parameters, "backend": "petsc"})
        ),
        "output_option": replace(
            base, outputs=replace(base.outputs, parameters={**base.outputs.parameters, "trajectory": True})
        ),
        "plots_disabled": replace(
            base, outputs=replace(base.outputs, parameters={**base.outputs.parameters, "plots": False})
        ),
        "strain_field": replace(
            base, outputs=replace(base.outputs, fields=[replace(base.outputs.fields[0], name="strain")])
        ),
        "stress_field": replace(
            base, outputs=replace(base.outputs, fields=[replace(base.outputs.fields[0], name="stress")])
        ),
        "other_example": replace(
            base,
            solver=replace(
                base.solver,
                parameters={**base.solver.parameters, "example": "solid_mechanics.neohookean_plate"},
            ),
        ),
    }[case]

    plan = execution_plan_from_spec(invalid)
    assert not plan.direct_execution_supported
    assert message in plan.execution_note
    monkeypatch.setattr(
        "phast.workflow.execution.subprocess.run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not run for an invalid spec"),
    )
    output_dir = tmp_path / "rejected"
    with pytest.raises(WorkflowExecutionError, match=message):
        run_problem_spec(invalid, output_dir=output_dir)
    assert not output_dir.exists()
