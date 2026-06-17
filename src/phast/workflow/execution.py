"""Internal execution-route planning for workflow specs.

This module does not run solvers. It records the compatibility route a
``ProblemSpec`` would need to use today, leaving solver invocation in the
existing runners.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml

from ..solid_mechanics_runner import solid_example_id
from ..config.config import OutputConfig as LegacyOutputConfig
from ..config.config import SolverSettings as LegacySolverSettings
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


def execution_plan_from_spec(spec: ProblemSpec) -> WorkflowExecutionPlan:
    """Return the internal compatibility route for a ``ProblemSpec``."""
    issues = tuple(validate_problem_spec_capabilities(spec))
    if issues:
        detail = "; ".join(issue.message for issue in issues)
        raise WorkflowExecutionError(f"ProblemSpec has unsupported capabilities: {detail}")

    step_kinds = tuple(step.kind for step in spec.analysis_steps)
    direct_execution_supported = (
        spec.source != "yaml:v2"
        or (
            spec.solver.kind == "solid_mechanics"
            and _solid_mechanics_example_id_from_spec(spec) is not None
        )
        or _schema_v2_quasistatic_fracture_supported(spec)
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
    """Run a YAML-backed ``ProblemSpec`` through PhAST's existing CLI.

    This is intentionally a compatibility bridge, not a solver adapter. It only
    invokes the public ``python -m phast run`` path for specs that remember the
    YAML file they came from.
    """
    plan = execution_plan_from_spec(spec)
    if spec.source_path is None:
        raise WorkflowExecutionError(
            "ProblemSpec.run() requires an original YAML source_path. "
            "Use phast.Problem.run() for Python-built problems."
        )
    if spec.source == "yaml:v2" and not validate_only:
        if plan.route == "solid_mechanics_runner" and plan.direct_execution_supported:
            return _run_schema_v2_solid_mechanics_spec(spec, output_dir=output_dir)
        if plan.route == "run_config" and _schema_v2_quasistatic_fracture_supported(spec):
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
) -> int:
    payload = _schema_v2_solid_mechanics_legacy_yaml(spec)
    with tempfile.TemporaryDirectory(prefix="phast-schema-v2-solid-") as tmp:
        lowered = Path(tmp) / "config.yaml"
        lowered.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        cmd = [sys.executable, "-m", "phast", "run", str(lowered)]
        if output_dir is not None:
            cmd.extend(["--output_dir", os.fspath(output_dir)])
        completed = subprocess.run(cmd, check=False)
        return int(completed.returncode)


def _schema_v2_quasistatic_fracture_supported(spec: ProblemSpec) -> bool:
    return (
        spec.source == "yaml:v2"
        and _quasistatic_fracture_supported(spec)
    )


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
        if history.name in {"reaction", "reaction_force"}:
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
    return _quasistatic_fracture_legacy_yaml(spec)


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
    payload = _schema_v2_fracture_legacy_yaml(spec)
    base_dir = Path(spec.source_path).resolve().parent if spec.source_path else Path.cwd()
    _absolutize_legacy_mesh_path(payload, base_dir)
    with tempfile.TemporaryDirectory(prefix="phast-schema-v2-fracture-") as tmp:
        lowered = Path(tmp) / "config.yaml"
        lowered.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        cmd = [sys.executable, "-m", "phast", "run", str(lowered)]
        if output_dir is not None:
            cmd.extend(["--output_dir", os.fspath(output_dir)])
        completed = subprocess.run(cmd, check=False)
        return int(completed.returncode)
