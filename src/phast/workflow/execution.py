"""Internal execution-route planning for workflow specs.

This module does not run solvers. It records the compatibility route a
``ProblemSpec`` would need to use today, leaving solver invocation in the
existing runners.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import inspect
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import yaml

from ..solid_mechanics_runner import solid_example_id, _validate_solid_mechanics_config
from ..config.config import OutputConfig as LegacyOutputConfig
from ..config.config import SolverSettings as LegacySolverSettings
from ..config.config import MaterialConfig, get_geometry_registry
from ..config.config_validation import validate_config
from ..utils.units import MATERIAL_OVERRIDE_KINDS, BOUNDARY_VALUE_QUANTITY_KINDS, parse_quantity
from .capabilities import CapabilityIssue, validate_problem_spec_capabilities
from .specs import ProblemSpec


class WorkflowExecutionError(ValueError):
    """Raised when a ProblemSpec cannot be routed to a supported runner."""


@dataclass(frozen=True)
class WorkflowExecutionPlan:
    route: str
    source: str
    solver_kind: str
    analysis_step_kinds: tuple[str, ...]
    capability_issues: tuple[CapabilityIssue, ...] = ()
    direct_execution_supported: bool = False
    execution_boundary: str = "deferred"
    execution_note: str = ""


_RUN_CONFIG_SOLVERS = {"explicit", "quasi_static", "quasi_static_legacy"}


def _python_solid_mechanics_bridge_issues(spec: ProblemSpec) -> tuple[str, ...]:
    """Reject Python inputs the published linear-plate runner would discard."""
    issues: list[str] = []
    example = _solid_mechanics_example_id_from_spec(spec)
    if example != "solid_mechanics.linear_plate":
        issues.append("only solid_mechanics.linear_plate is supported by this Python bridge")

    if len(spec.materials) != 1:
        issues.append("exactly one material is required")
    if len(spec.analysis_steps) != 1:
        issues.append("exactly one analysis step is required")
    if spec.boundary_conditions:
        issues.append("explicit boundary conditions cannot be lowered")
    if spec.initial_conditions:
        issues.append("initial conditions cannot be lowered")
    if spec.mesh is not None:
        issues.append("mesh file or explicit mesh inputs cannot be lowered")
    if spec.geometry is None or spec.geometry.kind != "structured_grid":
        issues.append("structured_grid geometry is required")
    elif (
        spec.geometry.primitives
        or spec.geometry.domain
        or spec.geometry.named_groups
        or spec.geometry.units != "m"
    ):
        issues.append("custom geometry metadata cannot be lowered")

    if len(spec.regions) > 1 or any(
        region.kind != "domain" or region.selector for region in spec.regions
    ):
        issues.append("only one unselected domain region can be lowered")

    if len(spec.materials) == 1:
        material = spec.materials[0]
        if material.model != "solid_mechanics":
            issues.append("material model must be solid_mechanics")
        expected_region = spec.regions[0].name if len(spec.regions) == 1 else None
        if material.region != expected_region:
            issues.append("material region must cover the full domain")
        if set(material.parameters) - {"E", "nu"}:
            issues.append("unsupported material parameters would be ignored")

    if len(spec.analysis_steps) == 1:
        step = spec.analysis_steps[0]
        if step.kind != "solid_mechanics":
            issues.append("analysis step kind must be solid_mechanics")
        if step.active_boundary_conditions:
            issues.append("active boundary conditions cannot be lowered")
        if set(step.controls) - {"tip_force_y"}:
            issues.append("unsupported load controls would be ignored")

    if spec.geometry is not None and set(spec.geometry.parameters) - {
        "nx", "ny", "length", "height"
    }:
        issues.append("unsupported geometry parameters would be ignored")
    if _legacy_solid_solver(spec):
        issues.append("nondefault solver settings cannot be lowered for linear_plate")
    if set(_legacy_solid_output(spec)) - {"directory", "plots"}:
        issues.append("unsupported output settings would be ignored")
    if spec.outputs.parameters.get("plots") is not True:
        issues.append("linear_plate requires plots=True for its fixed visual bundle")
    if any(
        field.name not in {"displacement", "von_mises", "strain_energy"}
        or field.every != 1 or field.parameters
        for field in spec.outputs.fields
    ):
        issues.append("unsupported field output request would be ignored")
    if any(
        history.name != "response" or history.every != 1 or history.region is not None
        or history.component is not None or history.parameters
        for history in spec.outputs.history
    ):
        issues.append("unsupported history output request would be ignored")
    if any(
        item.kind != "plots" or item.parameters for item in spec.outputs.postprocess
    ):
        issues.append("unsupported postprocess request would be ignored")

    if not issues and example is not None:
        payload = _schema_v2_solid_mechanics_legacy_yaml(spec)
        issues.extend(_validate_solid_mechanics_config(payload, example))
    return tuple(issues)


def execution_plan_from_spec(spec: ProblemSpec) -> WorkflowExecutionPlan:
    """Return the internal compatibility route for a ``ProblemSpec``."""
    issues = tuple(validate_problem_spec_capabilities(spec))
    if issues:
        detail = "; ".join(issue.message for issue in issues)
        raise WorkflowExecutionError(f"ProblemSpec has unsupported capabilities: {detail}")

    if spec.source == "yaml:v2" and spec.solver.kind in _RUN_CONFIG_SOLVERS:
        bridge_issues = _fracture_bridge_issues(spec)
        if bridge_issues:
            raise WorkflowExecutionError("; ".join(bridge_issues))
        if len(spec.materials) > 1 and spec.solver.kind == "quasi_static":
            from .multimaterial_fracture import (
                MultimaterialWorkflowError,
                validate_multimaterial_fracture_spec,
            )

            try:
                validate_multimaterial_fracture_spec(spec)
            except MultimaterialWorkflowError as exc:
                raise WorkflowExecutionError(str(exc)) from exc

    step_kinds = tuple(step.kind for step in spec.analysis_steps)
    python_solid_issues = (
        _python_solid_mechanics_bridge_issues(spec)
        if spec.source == "python:Problem"
        and spec.source_path is None
        and spec.solver.kind == "solid_mechanics"
        else ()
    )
    direct_execution_supported = (
        spec.source != "yaml:v2"
        or (
            spec.solver.kind == "solid_mechanics"
            and _solid_mechanics_example_id_from_spec(spec) is not None
        )
        or _schema_v2_fracture_supported(spec)
    )
    if spec.source == "python:Problem" and spec.source_path is None:
        direct_execution_supported = (
            spec.solver.kind == "solid_mechanics" and not python_solid_issues
        )
    execution_boundary = "existing_runner" if direct_execution_supported else "validate_only"
    execution_note = (
        "Existing compatibility runner can execute this compiled workflow."
        if direct_execution_supported
        else (
            "schema-v2 workflow contracts are validation/migration artifacts "
            "until a safe ProblemSpec-to-runner execution adapter is designed."
        )
    )
    if python_solid_issues:
        execution_boundary = "unsupported"
        execution_note = (
            "Python solid-mechanics ProblemSpec cannot be lowered: "
            + "; ".join(python_solid_issues)
        )
    if spec.solver.kind == "solid_mechanics":
        return WorkflowExecutionPlan(
            route="solid_mechanics_runner",
            source=spec.source,
            solver_kind=spec.solver.kind,
            analysis_step_kinds=step_kinds,
            direct_execution_supported=direct_execution_supported,
            execution_boundary=execution_boundary,
            execution_note=execution_note,
        )
    if spec.solver.kind == "validation_script":
        return WorkflowExecutionPlan(
            route="validation_contract",
            source=spec.source,
            solver_kind=spec.solver.kind,
            analysis_step_kinds=step_kinds,
            direct_execution_supported=False,
            execution_boundary="curated_validation",
            execution_note=(
                "Validation contracts execute only through allowlisted CLI "
                "validation IDs, not through generic ProblemSpec execution."
            ),
        )
    if spec.solver.kind in _RUN_CONFIG_SOLVERS:
        return WorkflowExecutionPlan(
            route="run_config",
            source=spec.source,
            solver_kind=spec.solver.kind,
            analysis_step_kinds=step_kinds,
            direct_execution_supported=direct_execution_supported,
            execution_boundary=execution_boundary,
            execution_note=execution_note,
        )
    raise WorkflowExecutionError(
        f"No execution route registered for solver kind {spec.solver.kind!r}"
    )


def run_problem_spec(
    spec: ProblemSpec,
    *,
    output_dir: str | os.PathLike | None = None,
    validate_only: bool = False,
) -> int:
    """Run a YAML-backed or promoted Python ``ProblemSpec`` through the CLI.

    This is a compatibility bridge, not a new solver adapter. Python-built
    solid-mechanics specs are lowered to the existing YAML runner.
    """
    if spec.source == "yaml:v2":
        from .validation import validate_problem_spec

        issues = validate_problem_spec(spec)
        if issues:
            raise WorkflowExecutionError("; ".join(issue.message for issue in issues))
    plan = execution_plan_from_spec(spec)
    if spec.source_path is None:
        if (
            spec.source == "python:Problem"
            and plan.route == "solid_mechanics_runner"
        ):
            if not plan.direct_execution_supported:
                raise WorkflowExecutionError(plan.execution_note)
            # The lowered YAML is temporary, but completed results must persist.
            configured_directory = spec.outputs.directory
            if configured_directory is None:
                configured_directory = spec.outputs.parameters.get("directory")
            destination = Path(
                output_dir if output_dir is not None else configured_directory or "outputs"
            )
            if not destination.is_absolute():
                destination = Path.cwd() / destination
            return _run_schema_v2_solid_mechanics_spec(
                spec, output_dir=destination, validate_only=validate_only
            )
        raise WorkflowExecutionError(
            "ProblemSpec.run() requires an original YAML source_path. "
            "Use phast.Problem.run() for Python-built problems."
        )
    if spec.source == "yaml:v2" and not validate_only:
        if plan.route == "solid_mechanics_runner" and plan.direct_execution_supported:
            return _run_schema_v2_solid_mechanics_spec(spec, output_dir=output_dir)
        if plan.route == "run_config" and _schema_v2_fracture_supported(spec):
            return _run_schema_v2_fracture_spec(spec, output_dir=output_dir)
        else:
            raise WorkflowExecutionError(
                "schema-v2 ProblemSpec execution is not supported yet; use "
                "ProblemSpec.run(validate_only=True) or python -m phast run "
                "<config> --validate-only."
            )
    if plan.route == "validation_contract":
        raise WorkflowExecutionError(
            "Validation contracts execute only through allowlisted CLI "
            "validation IDs, not through generic ProblemSpec.run()."
        )
    if not plan.direct_execution_supported and not validate_only:
        raise WorkflowExecutionError(plan.execution_note)

    cmd = [
        sys.executable,
        "-m",
        "phast",
        "run",
        str(Path(spec.source_path)),
    ]
    if validate_only:
        cmd.append("--validate-only")
    if output_dir is not None:
        cmd.extend(["--output_dir", os.fspath(output_dir)])
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


def _solid_mechanics_example_id_from_spec(spec: ProblemSpec) -> str | None:
    example = spec.solver.parameters.get("example")
    if not isinstance(example, str):
        return None
    return solid_example_id({"example": example})


def _legacy_solid_mesh(spec: ProblemSpec) -> dict:
    if spec.mesh is not None:
        return _coerce_legacy_values(dict(spec.mesh.parameters))
    if spec.geometry is not None:
        return _coerce_legacy_values(dict(spec.geometry.parameters))
    return {}


def _legacy_solid_material(spec: ProblemSpec) -> dict:
    material = spec.materials[0]
    return _coerce_legacy_values(dict(material.parameters))


def _legacy_solid_loading(spec: ProblemSpec) -> dict:
    step = spec.analysis_steps[0]
    return _coerce_legacy_values(dict(step.controls))


def _legacy_solid_solver(spec: ProblemSpec) -> dict:
    defaults = LegacySolverSettings(solver_type=spec.solver.kind)
    return _coerce_legacy_values(
        {
            key: value
            for key, value in spec.solver.parameters.items()
            if key not in {"example", "type", "kind"}
            and getattr(defaults, key, object()) != value
        }
    )


def _legacy_solid_output(spec: ProblemSpec) -> dict:
    defaults = LegacyOutputConfig()
    output = dict(spec.outputs.parameters)
    if spec.outputs.directory is not None:
        output["directory"] = spec.outputs.directory
    for key in ("fields", "history", "visuals"):
        output.pop(key, None)
    output = {
        key: value
        for key, value in output.items()
        if getattr(defaults, key, object()) != value
    }
    return _coerce_legacy_values(output)


def _coerce_legacy_values(value):
    if isinstance(value, dict):
        return {key: _coerce_legacy_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_coerce_legacy_values(item) for item in value]
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    return value


def _schema_v2_solid_mechanics_legacy_yaml(spec: ProblemSpec) -> dict:
    example = _solid_mechanics_example_id_from_spec(spec)
    if example is None:
        raise WorkflowExecutionError(
            "schema-v2 solid_mechanics execution requires solver.example to "
            "name a promoted solid-mechanics example."
        )
    return {
        "schema_version": 1,
        "example": example,
        "mesh": _legacy_solid_mesh(spec),
        "material": _legacy_solid_material(spec),
        "loading": _legacy_solid_loading(spec),
        "solver": _legacy_solid_solver(spec),
        "output": _legacy_solid_output(spec),
    }


def _run_schema_v2_solid_mechanics_spec(
    spec: ProblemSpec,
    *,
    output_dir: str | os.PathLike | None = None,
    validate_only: bool = False,
) -> int:
    payload = _schema_v2_solid_mechanics_legacy_yaml(spec)
    with tempfile.TemporaryDirectory(prefix="phast-schema-v2-solid-") as tmp:
        lowered = Path(tmp) / "config.yaml"
        lowered.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        cmd = [sys.executable, "-m", "phast", "run", str(lowered)]
        if validate_only:
            cmd.append("--validate-only")
        if output_dir is not None:
            cmd.extend(["--output_dir", os.fspath(output_dir)])
        completed = subprocess.run(cmd, check=False)
        return int(completed.returncode)


def _schema_v2_quasistatic_fracture_supported(spec: ProblemSpec) -> bool:
    return (
        spec.source == "yaml:v2"
        and _quasistatic_fracture_supported(spec)
    )


def _schema_v2_fracture_supported(spec: ProblemSpec) -> bool:
    return _schema_v2_quasistatic_fracture_supported(spec) or (
        spec.source == "yaml:v2"
        and spec.solver.kind == "explicit"
        and len(spec.materials) == 1
        and spec.materials[0].model == "phase_field"
        and len(spec.analysis_steps) == 1
        and spec.analysis_steps[0].kind == "explicit"
        and bool(spec.boundary_conditions)
    )


def _finite_quantity(value: Any, quantity: str | None = None) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError("a finite numeric value is required")
    number = parse_quantity(value, quantity) if quantity else float(value)
    if not math.isfinite(number):
        raise ValueError("a finite numeric value is required")
    return number


def _material_domain_issue(spec: ProblemSpec) -> str | None:
    """Accept a selected external domain only after checking full coverage."""
    material = spec.materials[0]
    if not material.region:
        return None
    region = next((r for r in spec.regions if r.name == material.region), None)
    message = "a single material assignment must cover the full domain"
    if region is None or region.kind != "domain":
        return message
    if not region.selector:
        return None
    aliases = {"from_mesh", "mesh_group", "physical_group", "element_set"}
    if spec.mesh is None or not spec.mesh.path or len(region.selector) != 1 or not set(region.selector) <= aliases:
        return message + "; selected domains require an external mesh element group"
    from ..mesh_inspection import inspect_mesh

    path = Path(spec.mesh.path).expanduser()
    if not path.is_absolute():
        base = Path(spec.source_path).resolve().parent if spec.source_path else Path.cwd()
        path = base / path
    try:
        summary = inspect_mesh(path)
        total = sum(int(cell["count"]) for cell in summary["cells"] if cell["type"] == "triangle")
        if not total or any(cell["type"] not in {"triangle", "line", "vertex"} for cell in summary["cells"]):
            return message + "; the bounded adapter requires a 2D T3 mesh"
        external = next(iter(region.selector.values()))
        group = summary["named_groups"].get(external)
        if group is not None:
            counts = group["cell_counts"] if group["dimension"] == 2 else {}
        else:
            counts = summary["cell_sets"].get(external, {})
        if counts.get("triangle", 0) != total:
            return message + f"; mesh group {external!r} does not contain all {total} elements"
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return message + f"; cannot verify mesh group: {exc}"
    return None


def _fracture_bridge_issues(spec: ProblemSpec) -> tuple[str, ...]:
    """Reject choices that the single-material compatibility runner discards.

    Multimaterial quasi-static execution retains its separate existing contract.
    Validation here does not create a mesh, solver, predictor, or checkpoint.
    """
    if len(spec.materials) > 1 and spec.solver.kind == "quasi_static":
        return ()
    issues: list[str] = []
    if spec.solver.kind not in {"explicit", "quasi_static"}:
        issues.append("schema-v2 fracture requires explicit or quasi_static")
    if len(spec.materials) != 1:
        issues.append("schema-v2 explicit fracture requires exactly one material")
    if len(spec.analysis_steps) != 1:
        issues.append("single-material fracture requires exactly one analysis step")
    if issues:
        return tuple(issues)
    step = spec.analysis_steps[0]
    if step.kind != spec.solver.kind:
        issues.append("analysis step type must match solver.type")
    if not spec.boundary_conditions:
        issues.append("fracture requires boundary_conditions")
    material = spec.materials[0]
    if material.model != "phase_field":
        issues.append("fracture requires material model phase_field")
    regions = {region.name: region for region in spec.regions}
    domain_issue = _material_domain_issue(spec)
    if domain_issue:
        issues.append(domain_issue)
    alias_keys = {"from_mesh", "mesh_group", "physical_group", "node_set", "element_set"}
    for region in spec.regions:
        if region.selector and (
            len(region.selector) != 1
            or not set(region.selector) <= alias_keys
            or not all(isinstance(v, str) and v for v in region.selector.values())
        ):
            issues.append(f"region {region.name!r}: single-material execution requires a mesh-group alias")
    if spec.geometry is not None and spec.mesh is not None:
        issues.append("specify geometry or mesh, not both")
    if spec.geometry is not None:
        geometry = spec.geometry
        if geometry.primitives:
            if geometry.kind != "primitive_dsl" or set(geometry.parameters) - {"mesh"}:
                issues.append("primitive geometry accepts only its mesh refinement parameters")
        else:
            if geometry.units != "mm":
                issues.append("built-in geometry requires units: mm")
            if geometry.domain or geometry.named_groups:
                issues.append("domain and named_groups require primitive geometry")
            registry = get_geometry_registry()
            generator = registry.get(geometry.kind)
            if generator is None or geometry.kind == "rectangular_sent_q4_structured":
                issues.append("single-material schema-v2 geometry requires a supported 2D T3 generator")
            elif set(geometry.parameters) - set(inspect.signature(generator).parameters):
                issues.append("unsupported geometry parameters for the selected generator")
            if geometry.parameters.get("order", 1) != 1:
                issues.append("only first-order 2D fracture elements are supported")
            if geometry.kind == "rectangular_sent_liu_structured" and "h_coarse" in geometry.parameters:
                issues.append("the structured SENT generator uses h_crack, not h_coarse")
    if spec.mesh is not None and (
        not spec.mesh.path or spec.mesh.parameters or spec.mesh.kind not in {"file", "external"}
    ):
        issues.append("external mesh execution requires a path without extra mesh parameters")

    known_material = {item.name for item in fields(MaterialConfig)} - {"preset", "overrides"}
    if set(material.parameters) - known_material:
        issues.append("unsupported material parameters")
    for key, value in material.parameters.items():
        if key in MATERIAL_OVERRIDE_KINDS and value is not None:
            try:
                number = parse_quantity(value, MATERIAL_OVERRIDE_KINDS[key])
                if not math.isfinite(number) or (
                    key in {"E", "Gc", "l0", "rho"} and number <= 0
                ) or (key == "nu" and not -1 < number < 0.5):
                    issues.append(f"material.{key} must be finite and within its physical range")
            except (TypeError, ValueError):
                issues.append(f"material.{key} has an invalid value or unit")
    if material.parameters.get("kinematics", "small_strain") not in {"small_strain", None}:
        issues.append("only small_strain kinematics is supported by this fracture adapter")
    if material.parameters.get("pf_model", "AT2") not in {"AT1", "AT2"}:
        issues.append("this fracture adapter supports AT1 or AT2, not other material models")
    if material.parameters.get("energy_split", "spectral") not in {"spectral", "amor", "isotropic"}:
        issues.append("this fracture adapter supports spectral, amor, or isotropic energy_split")

    solver = spec.solver.parameters
    defaults = LegacySolverSettings(solver_type=spec.solver.kind)
    if solver.get("damage_update", "classical") != "classical" or any(
        solver.get(key) for key in ("damage_predictor", "damage_checkpoint", "damage_predictor_options")
    ):
        issues.append("learned damage execution is not supported by this schema-v2 adapter")
    if solver.get("adaptive_dt", False):
        issues.append("adaptive_dt is unsupported by the fixed-time CLI loading/output schedule")
    try:
        if not 0 < _finite_quantity(solver.get("dt_safety", defaults.dt_safety)) <= 1:
            issues.append("solver.dt_safety must be in (0, 1]")
    except (TypeError, ValueError):
        issues.append("solver.dt_safety must be positive and finite")
    for key, value in _coerce_legacy_values(dict(solver)).items():
        if isinstance(value, float) and not math.isfinite(value):
            issues.append(f"solver.{key} must be finite")
    if spec.solver.kind == "explicit":
        if solver.get("time_integrator", "central_difference") not in {"central_difference", "verlet", "newmark"}:
            issues.append("explicit execution requires the central_difference (Velocity-Verlet) integrator")
        unused = {"backend", "static_tol", "static_max_iter", "max_stagger", "stagger_tol",
                  "stagger_criterion", "stagger_norm", "anderson_depth", "adaptive_stagger_tol", "rho_inf"}
        if any(key in solver and solver[key] != getattr(defaults, key) for key in unused):
            issues.append("quasi-static/implicit solver settings are not used by explicit dynamics")
    else:
        unused = {"time_integrator", "rho_inf", "dt_safety", "damage_every", "fresh_d_in_corrector",
                  "damping_ratio_max", "adaptive_dt_d_threshold"}
        if any(key in solver and solver[key] != getattr(defaults, key) for key in unused):
            issues.append("dynamic solver settings are not used by quasi_static")
    controls = _coerce_legacy_values(step.controls)
    count = controls.get("num_steps", 0)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        issues.append("controls.num_steps must be a nonnegative integer")
    if controls.get("protocol", "simple") != "simple":
        issues.append("schema-v2 single-material execution currently requires protocol: simple")
    allowed_controls = {"protocol", "num_steps", "dt"}
    if spec.solver.kind == "explicit":
        allowed_controls |= {"t_total", "ramp_type", "t_ramp", "v0"}
        count = controls.get("num_steps", 0)
        if not count and not controls.get("t_total", 0):
            issues.append("explicit controls require num_steps with dt, or a positive t_total")
        if count and "dt" not in controls:
            issues.append("explicit num_steps requires dt; omit num_steps and set t_total for automatic CFL stepping")
        if count and controls.get("t_total", 0):
            issues.append("specify num_steps or t_total, not both")
        if not count and "dt" in controls:
            issues.append("t_total-based automatic CFL stepping does not use controls.dt")
        if "dt" in controls and "dt_safety" in solver:
            issues.append("choose controls.dt or solver.dt_safety for automatic CFL stepping, not both")
        ramp = controls.get("ramp_type", "constant")
        if ramp != "constant" and not controls.get("t_ramp", 0):
            issues.append("nonconstant loading requires a positive t_ramp")
        if ramp == "constant" and controls.get("t_ramp", 0):
            issues.append("constant loading does not use t_ramp")
        if ramp != "velocity_impact" and controls.get("v0", 0):
            issues.append("v0 is only used by velocity_impact loading")
    elif not controls.get("num_steps", 0) or "dt" not in controls:
        issues.append("quasi_static controls require positive num_steps and load-factor increment dt")
    if set(controls) - allowed_controls:
        issues.append("unsupported analysis step controls")
    for key in ("dt", "t_total", "t_ramp"):
        if key in controls:
            try:
                value = parse_quantity(controls[key], "time") if isinstance(controls[key], str) else controls[key]
                if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                    issues.append(f"controls.{key} must be positive and finite")
            except (TypeError, ValueError):
                issues.append(f"controls.{key} must be a valid numeric value")

    allowed_bc_parameters = {
        "fix": set(), "prescribe": set(), "neumann": set(), "pf_dirichlet": set(),
        "traction": {"ramp_type", "t_ramp", "t_hold"}, "symmetry": {"axis"},
        "rigid_connector": {"master", "dofs", "prescribe", "rotation_free"},
    }
    for bc in spec.boundary_conditions:
        if set(_legacy_bc_parameters(bc.parameters)) - allowed_bc_parameters.get(bc.kind, set()):
            issues.append(f"boundary condition {bc.name!r} has parameters unused by {bc.kind}")
        if bc.kind in {"fix", "prescribe", "traction", "neumann"} and bc.component not in {0, 1}:
            issues.append(f"boundary condition {bc.name!r} requires dof x or y")
        if bc.kind == "fix" and bc.value not in {None, 0, 0.0}:
            issues.append("fix uses zero displacement; use prescribe for a nonzero value")
        if bc.kind in {"symmetry", "rigid_connector", "pf_dirichlet"} and bc.component is not None:
            issues.append(f"{bc.kind} does not use component/dof")
        if bc.kind in {"symmetry", "rigid_connector"} and bc.value not in {None, 0, 0.0}:
            issues.append(f"{bc.kind} does not use value")
        if bc.kind in {"fix", "prescribe", "traction", "neumann", "pf_dirichlet"}:
            try:
                value = _finite_quantity(0.0 if bc.value is None else bc.value, BOUNDARY_VALUE_QUANTITY_KINDS.get(bc.kind))
                if bc.kind == "pf_dirichlet" and not 0 <= value <= 1:
                    issues.append("pf_dirichlet damage value must be within [0, 1]")
            except (TypeError, ValueError):
                issues.append(f"boundary condition {bc.name!r} requires a finite value with valid units")
        if bc.kind == "symmetry" and bc.parameters.get("axis") not in {"x", "y"}:
            issues.append("symmetry requires axis: x or y")
        if bc.kind == "rigid_connector":
            master = bc.parameters.get("master")
            if not isinstance(master, str) or master not in regions:
                issues.append("rigid_connector requires master to name a declared mesh region")
            dofs = bc.parameters.get("dofs", ["x", "y"])
            allowed_dofs = {"x", "y", 0, 1}
            if not isinstance(dofs, list) or not dofs or any(
                isinstance(dof, bool) or not isinstance(dof, (str, int)) or dof not in allowed_dofs for dof in dofs
            ):
                issues.append("rigid_connector dofs must be a nonempty list of x/y components")
            prescribed = bc.parameters.get("prescribe", {})
            if not isinstance(prescribed, dict):
                issues.append("rigid_connector prescribe must be a component mapping")
            else:
                for component, value in prescribed.items():
                    try:
                        if isinstance(component, bool) or component not in allowed_dofs:
                            raise ValueError("invalid component")
                        _finite_quantity(value, "length")
                    except (TypeError, ValueError):
                        issues.append("rigid_connector prescribe requires x/y components and finite displacements")
        if bc.kind == "traction":
            ramp = bc.parameters.get("ramp_type", "constant")
            if ramp != "constant":
                try:
                    if _finite_quantity(bc.parameters.get("t_ramp"), "time") <= 0:
                        raise ValueError("nonpositive ramp")
                except (TypeError, ValueError):
                    issues.append("nonconstant traction requires a positive finite t_ramp")
        if spec.solver.kind != "explicit" and bc.parameters.get("ramp_type", "constant") != "constant":
            issues.append("quasi_static traction uses the global load schedule, not a per-BC time ramp")
    for initial in spec.initial_conditions:
        if initial.field != "damage" or initial.parameters or not initial.region:
            issues.append("initial_conditions support only damage on a named mesh region")
        if initial.value is not None and (
            not isinstance(initial.value, (int, float)) or isinstance(initial.value, bool)
            or not 0 <= initial.value <= 1
        ):
            issues.append("initial damage value must be within [0, 1]")

    trajectory = next((f for f in spec.outputs.fields if f.name == "trajectory"), None)
    for field in spec.outputs.fields:
        if field.name == "trajectory":
            if set(field.parameters) - {"format"} or field.parameters.get("format", "h5") not in {"h5", "zarr", "both"}:
                issues.append("trajectory accepts only format: h5, zarr, or both")
        elif field.name == "vtu":
            issues.append("vtu field writing is not implemented by the compatibility CLI loop")
        elif trajectory is None or field.every != trajectory.every or field.parameters:
            issues.append("stored fields require a trajectory with the same cadence and no extra parameters")
    reactions = [h for h in spec.outputs.history if h.name in {"reaction", "reaction_force", "load_displacement"}]
    if len(reactions) > 1:
        issues.append("the compatibility runner supports one reaction history selection")
    for history in spec.outputs.history:
        if history.every != 1 or history.parameters:
            issues.append("history outputs support every: 1 and no extra parameters")
        if history in reactions:
            if not history.region or history.component not in {0, 1}:
                issues.append("reaction history requires a region and dof x or y")
        elif history.region is not None or history.component is not None:
            issues.append("only reaction histories accept a region/component")
    for item in spec.outputs.postprocess:
        if item.kind not in {"plots", "animation", "initial_conditions", "damage_final"}:
            issues.append(f"unsupported postprocess request {item.kind!r}")
        elif item.kind == "animation":
            if trajectory is None:
                issues.append("animation requires trajectory output")
            if set(item.parameters) - {"format", "frames", "fields"}:
                issues.append("animation supports only format, frames, and fields")
        elif item.parameters:
            issues.append(f"{item.kind} does not accept parameters")
    output_defaults = LegacyOutputConfig()
    supported_output_parameters = {
        "directory", "output_dir", "print_every", "profile", "h5", "trajectory",
        "trajectory_format", "h5_every", "reaction_node_set", "reaction_component",
        "plots", "gif", "gif_frames", "gif_fields", "animation_format",
        "animation_renderer", "animation_raster_width",
    }
    if any(
        key not in supported_output_parameters and getattr(output_defaults, key, object()) != value
        for key, value in spec.outputs.parameters.items()
    ):
        issues.append("unsupported output parameters for the single-material CLI runner")

    if not issues:
        payload = _fracture_legacy_yaml(spec)
        issues.extend(f"{error.path}: {error.message}" for error in validate_config(payload))
        # The legacy validator checks these enums/ranges in overrides only.
        issues.extend(f"{error.path}: {error.message}" for error in validate_config(
            {"material": {"overrides": material.parameters}}
        ))
    return tuple(issues)


def _quasistatic_fracture_supported(spec: ProblemSpec) -> bool:
    return (
        spec.solver.kind == "quasi_static"
        and bool(spec.materials)
        and all(material.model == "phase_field" for material in spec.materials)
        and bool(spec.boundary_conditions)
        and bool(spec.analysis_steps)
        and all(step.kind == "quasi_static" for step in spec.analysis_steps)
    )


def _legacy_fracture_geometry(spec: ProblemSpec) -> dict:
    if spec.mesh is not None:
        payload = {"mesh_path": spec.mesh.path}
        payload.update(spec.mesh.parameters)
        payload.pop("mesh_type", None)
        payload.pop("kind", None)
        payload.pop("type", None)
        return _coerce_legacy_values({key: value for key, value in payload.items() if value is not None})
    if spec.geometry is None:
        return {}
    payload = {
        "type": spec.geometry.kind,
        "parameters": dict(spec.geometry.parameters),
        "units": spec.geometry.units,
        "primitives": spec.geometry.primitives,
        "domain": spec.geometry.domain,
        "named_groups": spec.geometry.named_groups,
    }
    if spec.geometry.primitives:
        payload.pop("type")
        payload.pop("parameters")
        payload["mesh"] = spec.geometry.parameters.get("mesh")
    return _coerce_legacy_values(
        {key: value for key, value in payload.items() if value not in (None, {}, [])}
    )


def _legacy_fracture_material(spec: ProblemSpec) -> dict:
    material = spec.materials[0]
    parameters = _coerce_legacy_values(dict(material.parameters))
    if _looks_like_preset_material(material.name, parameters):
        return {"preset": material.name, "overrides": parameters}
    return parameters


def _looks_like_preset_material(name: str, parameters: dict) -> bool:
    known_presets = {
        "default",
        "steel_pf",
        "miehe_tension",
        "miehe_shear",
        "three_point_bending",
        "l_shaped_glass",
        "l_shaped_concrete",
        "alumina_kumar",
        "brittle_ceramic",
        "pmma",
        "glass_borden",
        "pmma_bleyer",
        "maraging_steel_kw",
        "soda_lime_glass",
        "cement_mortar_ambati",
    }
    return name in known_presets


def _legacy_region_aliases(spec: ProblemSpec) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for region in spec.regions:
        for key in ("from_mesh", "mesh_group", "physical_group", "node_set", "element_set"):
            value = region.selector.get(key)
            if value not in (None, ""):
                aliases[region.name] = str(value)
                break
    return aliases


def _legacy_region_name(spec: ProblemSpec, region: str | None) -> str | None:
    if region is None:
        return None
    return _legacy_region_aliases(spec).get(region, region)


def _legacy_bc_parameters(parameters: dict) -> dict:
    filtered = dict(parameters)
    if filtered.get("ramp_type") == "constant":
        filtered.pop("ramp_type")
    if filtered.get("t_ramp") == 0.0:
        filtered.pop("t_ramp")
    if filtered.get("t_hold") is None:
        filtered.pop("t_hold", None)
    if filtered.get("rotation_free") is True:
        filtered.pop("rotation_free")
    return filtered


def _legacy_fracture_boundary_conditions(spec: ProblemSpec) -> list[dict]:
    lowered = []
    active_names: set[str] = set()
    if spec.analysis_steps:
        active_names = set(spec.analysis_steps[0].active_boundary_conditions)
    for bc in spec.boundary_conditions:
        if active_names and bc.name not in active_names:
            continue
        entry = {
            "nodes": _legacy_region_name(spec, bc.region),
            "type": bc.kind,
            "component": bc.component,
            "value": bc.value if bc.value is not None else 0.0,
        }
        entry.update(_legacy_bc_parameters(bc.parameters))
        if "master" in entry:
            entry["master"] = _legacy_region_name(spec, entry["master"])
        lowered.append(
            _coerce_legacy_values(
                {key: value for key, value in entry.items() if value is not None}
            )
        )
    return lowered


def _legacy_fracture_loading(spec: ProblemSpec) -> dict:
    return _coerce_legacy_values(dict(spec.analysis_steps[0].controls))


def _legacy_fracture_solver(spec: ProblemSpec) -> dict:
    defaults = LegacySolverSettings(solver_type=spec.solver.kind)
    payload = {"solver_type": spec.solver.kind}
    payload.update(
        {
            key: value
            for key, value in spec.solver.parameters.items()
            if key not in {"type", "kind", "device"}
            and getattr(defaults, key, object()) != value
        }
    )
    return _coerce_legacy_values(payload)


def _legacy_fracture_output(spec: ProblemSpec) -> dict:
    defaults = LegacyOutputConfig()
    output = {
        key: value
        for key, value in dict(spec.outputs.parameters).items()
        if getattr(defaults, key, object()) != value
    }
    if spec.outputs.directory is not None:
        output["output_dir"] = spec.outputs.directory
    output.pop("directory", None)
    for field in spec.outputs.fields:
        if field.name == "trajectory":
            output["trajectory"] = True
            output["h5"] = True
            output["h5_every"] = int(field.every)
            if "format" in field.parameters:
                output["trajectory_format"] = field.parameters["format"]
        elif field.name == "vtu":
            output["vtu"] = True
            output["vtu_every"] = int(field.every)
            if "format" in field.parameters:
                output["viz_format"] = field.parameters["format"]
    for history in spec.outputs.history:
        if history.name in {"reaction", "reaction_force", "load_displacement"}:
            output["reaction_node_set"] = _legacy_region_name(spec, history.region)
            output["reaction_component"] = history.component
    for postprocess in spec.outputs.postprocess:
        if postprocess.kind == "plots":
            output["plots"] = True
        elif postprocess.kind == "animation":
            output["gif"] = True
            parameters = dict(postprocess.parameters)
            if "format" in parameters:
                output["animation_format"] = parameters.pop("format")
            if "frames" in parameters:
                output["gif_frames"] = parameters.pop("frames")
            if "fields" in parameters:
                output["gif_fields"] = parameters.pop("fields")
            output.update(parameters)
    return _coerce_legacy_values(
        {key: value for key, value in output.items() if value is not None}
    )


def _legacy_fracture_device(spec: ProblemSpec) -> dict:
    device = spec.solver.parameters.get("device")
    return {"device": device} if device else {}


def _legacy_fracture_initial_conditions(spec: ProblemSpec) -> dict:
    preseed_damage = []
    for initial in spec.initial_conditions:
        if initial.field != "damage":
            continue
        entry = dict(initial.parameters)
        if initial.region:
            entry["nodes"] = _legacy_region_name(spec, initial.region)
        if initial.value is not None:
            entry["value"] = initial.value
        preseed_damage.append(_coerce_legacy_values(entry))
    if not preseed_damage:
        return {}
    return {"preseed_damage": preseed_damage}


def _quasistatic_fracture_legacy_yaml(spec: ProblemSpec) -> dict:
    if not _quasistatic_fracture_supported(spec):
        raise WorkflowExecutionError(
            "fracture workflow execution currently supports only quasi_static "
            "phase-field specs that lower cleanly to v1 run_config."
        )
    return _fracture_legacy_yaml(spec)


def _fracture_legacy_yaml(spec: ProblemSpec) -> dict[str, Any]:
    """Normalise a supported fracture spec without duplicating a solver loop."""
    return {
        "schema_version": 1,
        "name": spec.name,
        "reference": spec.reference,
        "geometry": _legacy_fracture_geometry(spec),
        "material": _legacy_fracture_material(spec),
        "boundary_conditions": _legacy_fracture_boundary_conditions(spec),
        "loading": _legacy_fracture_loading(spec),
        "solver": _legacy_fracture_solver(spec),
        "output": _legacy_fracture_output(spec),
        "device": _legacy_fracture_device(spec),
        "initial_conditions": _legacy_fracture_initial_conditions(spec),
    }


def _schema_v2_fracture_legacy_yaml(spec: ProblemSpec) -> dict:
    if spec.source != "yaml:v2":
        raise WorkflowExecutionError(
            "schema-v2 fracture execution requires a schema-v2 YAML ProblemSpec."
        )
    if not _schema_v2_fracture_supported(spec):
        raise WorkflowExecutionError("unsupported schema-v2 fracture execution route")
    return _fracture_legacy_yaml(spec)


def _absolutize_legacy_mesh_path(payload: dict, base_dir: Path) -> None:
    geometry = payload.get("geometry")
    if not isinstance(geometry, dict):
        return
    mesh_path = geometry.get("mesh_path")
    if not isinstance(mesh_path, str) or not mesh_path:
        return
    path = Path(mesh_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    geometry["mesh_path"] = str(path.resolve())


def _run_schema_v2_fracture_spec(
    spec: ProblemSpec,
    *,
    output_dir: str | os.PathLike | None = None,
) -> int:
    if len(spec.materials) > 1:
        from .multimaterial_fracture import run_multimaterial_fracture_spec

        return run_multimaterial_fracture_spec(spec, output_dir=output_dir)
    payload = _schema_v2_fracture_legacy_yaml(spec)
    base_dir = Path(spec.source_path).resolve().parent if spec.source_path else Path.cwd()
    _absolutize_legacy_mesh_path(payload, base_dir)
    destination = Path(output_dir or spec.outputs.directory or "outputs").resolve()
    payload["output"]["output_dir"] = str(destination)
    geometry = payload["geometry"]
    if "type" in geometry and not geometry.get("mesh_path"):
        geometry.setdefault("parameters", {}).setdefault("output_path", str(destination / "mesh.msh"))
    with tempfile.TemporaryDirectory(prefix="phast-schema-v2-fracture-") as tmp:
        lowered = Path(tmp) / "config.yaml"
        lowered.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        cmd = [sys.executable, "-m", "phast", "run", str(lowered)]
        if output_dir is not None:
            cmd.extend(["--output_dir", os.fspath(output_dir)])
        completed = subprocess.run(cmd, check=False)
        return int(completed.returncode)
