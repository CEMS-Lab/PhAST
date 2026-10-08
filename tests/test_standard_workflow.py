"""Bounded schema-v2 fracture execution through the existing public CLI."""
from __future__ import annotations

import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import h5py
import meshio
import numpy as np
import pytest
import yaml

from phast import Problem
from phast.config.config import load_config
from phast.workflow import (
    execution_plan_from_spec, problem_spec_from_yaml, problem_spec_to_schema_v2_dict,
    run_problem_spec, validate_problem_spec,
)
from phast.workflow.execution import WorkflowExecutionError, _schema_v2_fracture_legacy_yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "standard_workflow"
TOP_LEVEL = [
    "schema_version", "name", "reference", "geometry", "regions", "materials",
    "assignments", "initial_conditions", "boundary_conditions", "analysis_steps",
    "solver", "outputs",
]


def _example(kind: str = "dynamic_sent") -> dict[str, Any]:
    return yaml.safe_load((EXAMPLES / kind / "config.yaml").read_text())


def _write(tmp_path: Path, raw: Any, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def _cli(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT / "src"), str(ROOT))),
               MPLBACKEND="Agg", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    return subprocess.run(
        [sys.executable, "-m", "phast", "run", str(path), *args], cwd=path.parent,
        env=env, text=True, capture_output=True, timeout=60, check=False,
    )


@pytest.mark.parametrize("relative_path", [
    "config.yaml", "comparisons/tough_region.yaml", "comparisons/uniform_layer.yaml",
])
def test_dcb_cli_validation_does_not_solve(relative_path: str, tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "examples" / "two_material_dcb_beta" / relative_path).read_text())
    target = tmp_path / "must_not_exist"
    raw["outputs"]["directory"] = str(target)
    result = _cli(_write(tmp_path, raw), "--validate-only")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not target.exists()


@pytest.mark.parametrize("case", ["negative_Gc", "empty_seed", "interior_boundary"])
def test_dcb_invalid_edits_fail_during_cli_validation(case: str, tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "examples" / "two_material_dcb_beta" / "config.yaml").read_text())
    target = tmp_path / "must_not_exist"
    raw["outputs"]["directory"] = str(target)
    if case == "negative_Gc":
        raw["materials"]["material_2"]["parameters"]["Gc"] = -0.12
    elif case == "empty_seed":
        raw["regions"]["initial_crack"].update({"from": [10.0, 0.0], "to": [11.0, 0.0]})
    else:
        raw["geometry"]["parameters"]["length"] = 12.0
        raw["regions"]["right_clamp"] = {
            "kind": "node_region", "type": "boundary", "axis": "x", "value": 6.0,
        }
    path = _write(tmp_path, raw)
    assert validate_problem_spec(problem_spec_from_yaml(path)), case
    result = _cli(path, "--validate-only")
    assert result.returncode == 2, result.stdout + result.stderr
    assert not target.exists()


@pytest.mark.parametrize("kind", ["dynamic_sent", "quasistatic_sent"])
def test_canonical_load_and_round_trip(kind: str, tmp_path: Path) -> None:
    raw = _example(kind)
    assert list(raw) == TOP_LEVEL
    spec = problem_spec_from_yaml(_write(tmp_path, raw))
    assert not validate_problem_spec(spec)
    assert execution_plan_from_spec(spec).direct_execution_supported
    assert spec.outputs.fields[0].parameters["format"] == "h5"
    restored = problem_spec_from_yaml(_write(tmp_path, problem_spec_to_schema_v2_dict(spec), "roundtrip.yaml"))
    assert _schema_v2_fracture_legacy_yaml(restored) == _schema_v2_fracture_legacy_yaml(spec)
    lowered = _schema_v2_fracture_legacy_yaml(spec)
    assert lowered["output"]["output_dir"] == raw["outputs"]["directory"]
    assert "directory" not in lowered["output"]
    assert lowered["boundary_conditions"][2]["nodes"] == "top"
    assert lowered["initial_conditions"]["preseed_damage"][0]["nodes"] == "crack"


@pytest.mark.parametrize("kind", ["dynamic_sent", "quasistatic_sent"])
def test_cli_validate_run_output_override_and_v1_parity(kind: str, tmp_path: Path) -> None:
    raw = _example(kind)
    ignored = tmp_path / "configured_but_overridden"
    raw["outputs"]["directory"] = str(ignored)
    path = _write(tmp_path, raw)
    target = tmp_path / "v2"
    validation = _cli(path, "--validate-only", "--output_dir", str(target))
    assert validation.returncode == 0, validation.stdout + validation.stderr
    assert "passes schema-v2 workflow contract validation" in validation.stdout
    assert not target.exists()
    run = _cli(path, "--output_dir", str(target))
    assert run.returncode == 0, run.stdout + run.stderr
    assert not ignored.exists()
    assert not (target / "training_data.zarr").exists()
    for name in ("training_data.h5", "config.yaml", "run_lockfile.json", "run_metadata.json",
                 "energy.csv", "results.csv", "solver_telemetry.csv", "timing_per_step.csv"):
        assert (target / name).is_file(), name
    metadata = json.loads((target / "run_metadata.json").read_text())
    assert "num_steps" in json.dumps(metadata)
    with (target / "energy.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 4
    with h5py.File(target / "training_data.h5") as store:
        assert store.attrs["num_steps"] == 4

    # The same normalised inputs still use the v1 loader and original loop.
    spec = problem_spec_from_yaml(path)
    legacy = _schema_v2_fracture_legacy_yaml(spec)
    legacy["geometry"]["parameters"]["output_path"] = str(tmp_path / "v1" / "mesh.msh")
    v1_path = _write(tmp_path, legacy, "legacy.yaml")
    v1_spec = problem_spec_from_yaml(v1_path)
    assert v1_spec.source == "yaml:v1"
    assert execution_plan_from_spec(v1_spec).direct_execution_supported
    v1_run = _cli(v1_path, "--output_dir", str(tmp_path / "v1"))
    assert v1_run.returncode == 0, v1_run.stdout + v1_run.stderr
    np.testing.assert_allclose(
        np.genfromtxt(target / "energy.csv", delimiter=",", skip_header=1),
        np.genfromtxt(tmp_path / "v1" / "energy.csv", delimiter=",", skip_header=1),
        rtol=1.0e-12, atol=1.0e-14,
    )


def test_automatic_cfl_loading_and_hdf5_default(tmp_path: Path) -> None:
    raw = _example()
    raw["analysis_steps"][0]["controls"] = {"t_total": 1.0e-7}
    raw["solver"]["dt_safety"] = 0.5
    raw["outputs"]["fields"][0].pop("format")
    spec = problem_spec_from_yaml(_write(tmp_path, raw))
    assert not validate_problem_spec(spec)
    path = _write(tmp_path, _schema_v2_fracture_legacy_yaml(spec), "legacy.yaml")
    config = load_config(str(path))
    assert config.loading.num_steps == 0
    assert config.solver.dt_safety == 0.5
    assert config.output.trajectory_format == "h5"


@pytest.mark.parametrize("case", [
    "multiple_steps", "step_mismatch", "multimaterial_dynamic", "partial_material",
    "geometric_region", "initial_velocity", "initial_damage_range", "unknown_solver",
    "learned", "implicit_integrator", "adaptive_dt", "unused_backend", "unknown_control",
    "missing_dt", "both_time_limits", "fixed_dt_and_cfl", "unknown_geometry", "3d",
    "higher_order", "material_typo", "negative_density", "nonfinite_material",
    "unknown_output", "vtu", "field_without_trajectory", "field_cadence",
    "history_cadence", "two_reactions", "unknown_visual", "unknown_bc_parameter",
    "nonzero_fix", "bad_trajectory_format",
])
def test_unsupported_choices_fail_before_execution(case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _example()
    if case == "multiple_steps":
        raw["analysis_steps"].append({**copy.deepcopy(raw["analysis_steps"][0]), "name": "second"})
    elif case == "step_mismatch":
        raw["analysis_steps"][0]["type"] = "quasi_static"
    elif case == "multimaterial_dynamic":
        raw["materials"]["second"] = copy.deepcopy(raw["materials"]["glass"])
        raw["assignments"].append({"material": "second", "region": "body"})
    elif case == "partial_material":
        raw["assignments"][0]["region"] = "left_edge"
    elif case == "geometric_region":
        raw["regions"]["left_edge"] = {"type": "boundary", "axis": "x", "value": 0}
    elif case == "initial_velocity":
        raw["initial_conditions"][0]["field"] = "velocity"
    elif case == "initial_damage_range":
        raw["initial_conditions"][0]["value"] = 2.0
    elif case in {"unknown_solver", "learned", "implicit_integrator", "adaptive_dt", "unused_backend"}:
        key, value = {
            "unknown_solver": ("damage_toll", 1.0e-6), "learned": ("damage_update", "learned_proposal"),
            "implicit_integrator": ("time_integrator", "generalized_alpha"),
            "adaptive_dt": ("adaptive_dt", True), "unused_backend": ("backend", "scipy"),
        }[case]
        raw["solver"][key] = value
    elif case == "unknown_control":
        raw["analysis_steps"][0]["controls"]["number_of_stepz"] = 2
    elif case == "missing_dt":
        raw["analysis_steps"][0]["controls"].pop("dt")
    elif case == "both_time_limits":
        raw["analysis_steps"][0]["controls"]["t_total"] = 1.0e-6
    elif case == "fixed_dt_and_cfl":
        raw["solver"]["dt_safety"] = 0.5
    elif case in {"unknown_geometry", "3d", "higher_order"}:
        raw["geometry"]["parameters"][{"unknown_geometry": "width_typo", "3d": "nz", "higher_order": "order"}[case]] = 2
    elif case in {"material_typo", "negative_density", "nonfinite_material"}:
        key, value = {"material_typo": ("Gc_typo", 1.0), "negative_density": ("rho", -1.0), "nonfinite_material": ("E", float("nan"))}[case]
        raw["materials"]["glass"]["parameters"][key] = value
    elif case == "unknown_output":
        raw["outputs"]["trajectory_formatt"] = "h5"
    elif case == "vtu":
        raw["outputs"]["fields"].append({"name": "vtu"})
    elif case == "field_without_trajectory":
        raw["outputs"]["fields"].pop(0)
    elif case == "field_cadence":
        raw["outputs"]["fields"][1]["every"] = 2
    elif case == "history_cadence":
        raw["outputs"]["history"][0]["every"] = 2
    elif case == "two_reactions":
        raw["outputs"]["history"].append({"name": "reaction", "region": "lower_edge", "dof": "y"})
    elif case == "unknown_visual":
        raw["outputs"]["visuals"]["thumbnail"] = True
    elif case == "unknown_bc_parameter":
        raw["boundary_conditions"][2]["ramp_type"] = "linear"
    elif case == "nonzero_fix":
        raw["boundary_conditions"][0]["value"] = 1.0
    elif case == "bad_trajectory_format":
        raw["outputs"]["fields"][0]["format"] = "hdf_typo"
    spec = problem_spec_from_yaml(_write(tmp_path, raw))
    assert validate_problem_spec(spec), case
    monkeypatch.setattr("phast.workflow.execution.subprocess.run", lambda *a, **kw: pytest.fail("invalid spec started a subprocess"))
    with pytest.raises(WorkflowExecutionError):
        run_problem_spec(spec, output_dir=tmp_path / "must_not_exist")
    assert not (tmp_path / "must_not_exist").exists()


@pytest.mark.parametrize("case", ["unknown_top", "materials_scalar", "steps_scalar", "controls_scalar", "active_scalar", "fractional_cadence", "boolean_cadence", "material_sibling", "dof_aliases"])
def test_malformed_v2_input(case: str, tmp_path: Path) -> None:
    raw = _example()
    if case == "unknown_top":
        raw["output"] = {}
    elif case == "materials_scalar":
        raw["materials"] = "glass"
    elif case == "steps_scalar":
        raw["analysis_steps"] = "explicit"
    elif case == "controls_scalar":
        raw["analysis_steps"][0]["controls"] = 1
    elif case == "active_scalar":
        raw["analysis_steps"][0]["active_boundary_conditions"] = "open_upper"
    elif case in {"fractional_cadence", "boolean_cadence"}:
        raw["outputs"]["fields"][0]["every"] = 1.5 if case == "fractional_cadence" else True
    elif case == "material_sibling":
        raw["materials"]["glass"]["Gc"] = 3.0
    elif case == "dof_aliases":
        raw["boundary_conditions"][0]["component"] = 1
    with pytest.raises((ValueError, TypeError)):
        problem_spec_from_yaml(_write(tmp_path, raw))


def test_cli_rejects_unsupported_override_and_malformed_input(tmp_path: Path) -> None:
    path = _write(tmp_path, _example())
    result = _cli(path, "--validate-only", "--num_steps", "2")
    assert result.returncode == 2
    assert "only --output_dir" in result.stderr
    malformed = _write(tmp_path, {**_example(), "ignored_setting": 1}, "malformed.yaml")
    result = _cli(malformed, "--validate-only")
    assert result.returncode == 2
    assert "Unknown schema-v2 top-level keys" in result.stderr
    assert "Traceback" not in result.stderr


def test_primitive_geometry_preserves_mesh_refinement(tmp_path: Path) -> None:
    raw = _example()
    raw["geometry"] = {
        "units": "mm", "primitives": {"plate": {"type": "rectangle", "origin": [0, 0], "size": [4, 4]}},
        "domain": {"base": "plate"}, "mesh": {"element_size": {"default": 0.5}},
    }
    spec = problem_spec_from_yaml(_write(tmp_path, raw))
    assert not validate_problem_spec(spec)
    geometry = _schema_v2_fracture_legacy_yaml(spec)["geometry"]
    assert "type" not in geometry
    assert "parameters" not in geometry
    assert geometry["mesh"] == raw["geometry"]["mesh"]


def test_original_v1_public_dynamic_input_is_preserved() -> None:
    path = ROOT / "examples/dynamic/B3_dynamic_sent/config.yaml"
    spec = problem_spec_from_yaml(path)
    config = load_config(str(path))
    assert spec.source == "yaml:v1"
    assert config.loading.num_steps == 0
    assert config.loading.t_total == pytest.approx(1.0e-4)
    assert config.solver.dt_safety == 0.8
    assert config.material.energy_split == "spectral"
    assert config.output.trajectory_format == "h5"
    assert execution_plan_from_spec(spec).direct_execution_supported


def _fluent_external_problem(tmp_path: Path, *, partial: bool = False) -> Problem:
    mesh_path = tmp_path / "plate.msh"
    mesh = meshio.Mesh(
        points=np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]]),
        cells=[("line", np.array([[0, 1], [1, 2], [2, 3], [3, 0]])),
               ("triangle", np.array([[0, 1, 2], [0, 2, 3]]))],
        field_data={"bottom": [1, 1], "right": [2, 1], "top": [3, 1], "left": [4, 1], "body": [5, 2], "other": [6, 2]},
        cell_data={"gmsh:physical": [np.array([1, 2, 3, 4]), np.array([5, 6 if partial else 5])],
                   "gmsh:geometrical": [np.array([1, 2, 3, 4]), np.array([5, 6 if partial else 5])]},
    )
    meshio.write(mesh_path, mesh, file_format="gmsh22", binary=False)
    return (Problem("Saved two-step quasi-static setup")
        .mesh(mesh_path)
        .region("body", kind="domain", from_mesh="body")
        .region("lower", from_mesh="bottom").region("upper", from_mesh="top")
        .material("miehe_tension", region="body", E=1000.0, l0=0.5, energy_split="isotropic")
        .boundary_condition("fix", region="lower", dof="xy", name="clamp")
        .boundary_condition("prescribe", region="upper", dof="y", value=1.0e-6, name="opening")
        .analysis_step("opening", kind="quasi_static", controls={"num_steps": 2, "dt": 0.5},
                       active_boundary_conditions=["clamp", "opening"])
        .solver("quasi_static", backend="scipy", preconditioner="jacobi")
        .outputs(fields=[{"name": "trajectory", "every": 1, "format": "h5"}],
                 histories=[{"name": "reaction_force", "region": "upper", "dof": "y"}],
                 h5_every=1, plots=False, profile=True, gif=False,
                 output_dir=str(tmp_path / "configured")))


def test_actual_problem_save_validate_and_run(tmp_path: Path) -> None:
    problem = _fluent_external_problem(tmp_path)
    path = tmp_path / "saved.yaml"
    problem.save(path)
    spec = problem_spec_from_yaml(path)
    assert spec.source == "yaml:v2"
    assert spec.mesh is not None and not spec.mesh.parameters
    assert not validate_problem_spec(spec)
    assert spec.outputs.parameters["profile"] is True
    assert "h5" not in spec.outputs.parameters
    validation = _cli(path, "--validate-only")
    assert validation.returncode == 0, validation.stdout + validation.stderr
    run = _cli(path, "--output_dir", str(tmp_path / "saved_run"))
    assert run.returncode == 0, run.stdout + run.stderr
    with h5py.File(tmp_path / "saved_run" / "training_data.h5") as store:
        assert store.attrs["num_steps"] == 2
    assert (tmp_path / "saved_run" / "results.csv").is_file()


def test_saved_material_subdomain_is_not_silently_made_global(tmp_path: Path) -> None:
    problem = _fluent_external_problem(tmp_path, partial=True)
    path = tmp_path / "partial.yaml"
    problem.save(path)
    issues = validate_problem_spec(problem_spec_from_yaml(path))
    assert any("does not contain all 2 elements" in issue.message for issue in issues)
    result = _cli(path, "--validate-only")
    assert result.returncode == 2
    assert "does not contain all 2 elements" in result.stderr


@pytest.mark.parametrize("case", [
    "damage_bc_range", "damage_bc_nan", "displacement_units", "symmetry_axis", "connector_master",
    "connector_dofs", "connector_prescribe", "null_steps", "zero_cfl", "damage_component",
    "symmetry_value", "connector_component", "step_aliases",
])
def test_reviewed_input_failures_are_rejected(case: str, tmp_path: Path) -> None:
    raw = _example()
    if case == "damage_bc_range":
        raw["boundary_conditions"][-1]["value"] = 2.0
    elif case == "damage_bc_nan":
        raw["boundary_conditions"][-1]["value"] = float("nan")
    elif case == "displacement_units":
        raw["boundary_conditions"][2]["value"] = "2 bananas"
    elif case in {"symmetry_axis", "symmetry_value"}:
        raw["boundary_conditions"][0] = {"name": "restrain_left", "type": "symmetry", "region": "left_edge"}
        if case == "symmetry_value":
            raw["boundary_conditions"][0].update(axis="x", value=2.0)
    elif case.startswith("connector"):
        bc = {"name": "restrain_left", "type": "rigid_connector", "region": "left_edge", "master": "right_edge"}
        if case == "connector_master":
            bc.pop("master")
        elif case == "connector_dofs":
            bc["dofs"] = ["z"]
        elif case == "connector_prescribe":
            bc["prescribe"] = {"z": 1.0}
        else:
            bc["component"] = 0
        raw["boundary_conditions"][0] = bc
    elif case in {"null_steps", "zero_cfl"}:
        raw["analysis_steps"][0]["controls"] = {"t_total": 1.0e-7}
        if case == "null_steps":
            raw["analysis_steps"][0]["controls"]["num_steps"] = None
        else:
            raw["solver"]["dt_safety"] = 0.0
    elif case == "damage_component":
        raw["boundary_conditions"][-1]["component"] = 0
    elif case == "step_aliases":
        raw["analysis_steps"][0]["kind"] = "quasi_static"
    path = _write(tmp_path, raw)
    if case == "step_aliases":
        with pytest.raises(ValueError, match="type or kind"):
            problem_spec_from_yaml(path)
    else:
        assert validate_problem_spec(problem_spec_from_yaml(path)), case
