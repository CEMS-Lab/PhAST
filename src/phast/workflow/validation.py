"""Unified validation helpers for workflow specs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .capabilities import validate_problem_spec_capabilities
from .execution import WorkflowExecutionError, execution_plan_from_spec
from .regions import validate_problem_spec_regions
from .specs import ProblemSpec


@dataclass(frozen=True)
class WorkflowValidationIssue:
    category: str
    name: str
    message: str


_VALIDATION_ONLY_MATERIAL_MODELS = {
    "cohesive_interface",
    "diffuse_interface",
    "ductile_phase_field",
    "j2_plasticity",
    "validation_artifact",
}

_FRACTURE_SOLVERS = {"explicit", "quasi_static", "quasi_static_legacy"}
_FRACTURE_ANALYSIS_STEPS = {"explicit", "quasi_static", "quasi_static_legacy"}
_FRACTURE_MATERIAL_MODELS = {"phase_field"}
_FRACTURE_BOUNDARY_CONDITIONS = {
    "fix",
    "neumann",
    "pf_dirichlet",
    "prescribe",
    "rigid_connector",
    "symmetry",
    "traction",
}
_COMPONENT_BOUNDARY_CONDITIONS = {"fix", "prescribe", "neumann", "traction"}
_DISPLACEMENT_DIRICHLET_BOUNDARY_CONDITIONS = {"fix", "prescribe"}
_SOLID_BOUNDARY_CONDITIONS = {
    "fix",
    "neumann",
    "prescribe",
    "symmetry",
    "traction",
}
_DYNAMIC_ONLY_FIELD_OUTPUTS = {"velocity", "acceleration"}
_FRACTURE_FIELD_OUTPUTS = {
    "damage",
    "displacement",
    "history_field",
    "history_field_nodal",
    "psi_plus",
    "strain",
    "stress",
    "trajectory",
    "vtu",
}
_SOLID_FIELD_OUTPUTS = {
    "displacement",
    "strain",
    "stress",
    "von_mises",
}
_SOLID_EXAMPLE_EXTRA_FIELDS = {
    "solid_mechanics.linear_plate": {"strain_energy"},
    "solid_mechanics.neohookean_plate": {"jacobian", "strain_energy"},
    "solid_mechanics.j2_bar": {"equivalent_plastic_strain"},
}
_FRACTURE_HISTORY_OUTPUTS = {
    "energy",
    "load_displacement",
    "max_damage",
    "reaction",
    "reaction_force",
    "solver_telemetry",
    "timing_per_step",
}
_SOLID_HISTORY_OUTPUTS = {"response"}
_MESH_REGION_SELECTOR_KEYS = {
    "from_mesh",
    "mesh_group",
    "physical_group",
    "node_set",
    "element_set",
}


def validate_problem_spec(spec: ProblemSpec) -> list[WorkflowValidationIssue]:
    """Validate a ProblemSpec without invoking solver construction."""
    issues: list[WorkflowValidationIssue] = []
    issues.extend(
        _duplicate_name_issues(
            "region",
            "region",
            (region.name for region in spec.regions),
        )
    )
    issues.extend(
        _duplicate_name_issues(
            "analysis_step",
            "analysis step",
            (step.name for step in spec.analysis_steps),
        )
    )
    for issue in validate_problem_spec_capabilities(spec):
        issues.append(
            WorkflowValidationIssue(
                category="capability",
                name=issue.name,
                message=issue.message,
            )
        )
    if spec.solver.kind != "validation_script":
        for material in spec.materials:
            if material.model in _VALIDATION_ONLY_MATERIAL_MODELS:
                issues.append(
                    WorkflowValidationIssue(
                        category="capability",
                        name=material.model,
                        message=(
                            f"Material model {material.model!r} is currently "
                            "restricted to curated validation_script workflow "
                            "contracts."
                        ),
                    )
                )
    issues.extend(_solver_family_compatibility_issues(spec))
    for issue in validate_problem_spec_regions(spec):
        issues.append(
            WorkflowValidationIssue(
                category="region",
                name=issue.region,
                message=issue.message,
            )
        )
    material_name_counts: dict[str, int] = {}
    for material in spec.materials:
        material_name_counts[material.name] = material_name_counts.get(material.name, 0) + 1
    material_regions: dict[str, list[str]] = {}
    for material in spec.materials:
        if material.region:
            material_regions.setdefault(material.region, []).append(material.name)
        elif len(spec.materials) > 1 and material_name_counts.get(material.name, 0) == 1:
            issues.append(
                WorkflowValidationIssue(
                    category="material",
                    name=material.name,
                    message=(
                        f"Material {material.name!r} has no region assignment; "
                        "multiple materials require explicit region assignments."
                    ),
                )
            )
    for region, material_names in material_regions.items():
        if len(material_names) > 1:
            issues.append(
                WorkflowValidationIssue(
                    category="material",
                    name=region,
                    message=(
                        f"Region {region!r} has multiple material assignments: "
                        f"{', '.join(material_names)}"
                    ),
                )
            )
    issues.extend(_mesh_region_mapping_issues(spec))
    boundary_condition_names = {
        boundary_condition.name
        for boundary_condition in spec.boundary_conditions
        if boundary_condition.name
    }
    issues.extend(_boundary_condition_duplicate_name_issues(spec))
    issues.extend(_boundary_condition_value_issues(spec))
    output_groups = (
        ("field", (field.name for field in spec.outputs.fields)),
        ("history", (history.name for history in spec.outputs.history)),
        ("postprocess", (postprocess.kind for postprocess in spec.outputs.postprocess)),
    )
    for output_kind, names in output_groups:
        output_names: dict[str, int] = {}
        for name in names:
            if name:
                output_names[name] = output_names.get(name, 0) + 1
        for name, count in output_names.items():
            if count > 1:
                issues.append(
                    WorkflowValidationIssue(
                        category="output",
                        name=name,
                        message=f"duplicate {output_kind} output name {name!r}",
                    )
                )
    declared_bcs = set(boundary_condition_names)
    for step in spec.analysis_steps:
        for bc_name in step.active_boundary_conditions:
            if bc_name not in declared_bcs:
                issues.append(
                    WorkflowValidationIssue(
                        category="analysis_step",
                        name=bc_name,
                        message=(
                            f"Analysis step {step.name!r} active boundary "
                            f"condition {bc_name!r} is not declared"
                        ),
                    )
                )
    try:
        execution_plan_from_spec(spec)
    except WorkflowExecutionError as exc:
        issues.append(
            WorkflowValidationIssue(
                category="execution_route",
                name=spec.solver.kind,
                message=str(exc),
            )
        )
    return issues


def _boundary_condition_value_issues(
    spec: ProblemSpec,
) -> list[WorkflowValidationIssue]:
    issues: list[WorkflowValidationIssue] = []
    prescribed: dict[tuple[str, int], list[tuple[str, object]]] = {}
    for bc in spec.boundary_conditions:
        if bc.kind in _COMPONENT_BOUNDARY_CONDITIONS:
            if bc.component is not None and bc.component not in {0, 1}:
                issues.append(
                    WorkflowValidationIssue(
                        category="boundary_condition",
                        name=bc.region,
                        message=(
                            f"Boundary condition on region {bc.region!r} has "
                            f"component {bc.component!r}; component must be 0 or 1."
                        ),
                    )
                )
                continue
        if bc.kind in _DISPLACEMENT_DIRICHLET_BOUNDARY_CONDITIONS:
            if bc.component is None:
                continue
            value = 0.0 if bc.kind == "fix" else bc.value
            prescribed.setdefault((bc.region, int(bc.component)), []).append(
                (bc.name or bc.kind, value)
            )
    for (region, component), entries in prescribed.items():
        values = {repr(value) for _name, value in entries}
        if len(values) > 1:
            names = ", ".join(name for name, _value in entries)
            issues.append(
                WorkflowValidationIssue(
                    category="boundary_condition",
                    name=f"{region}:{component}",
                    message=(
                        "conflicting displacement boundary conditions on "
                        f"region {region!r} component {component}: {names}"
                    ),
                )
            )
    return issues


def _mesh_region_mapping_issues(spec: ProblemSpec) -> list[WorkflowValidationIssue]:
    issues: list[WorkflowValidationIssue] = []
    mesh_mappings: dict[tuple[str, str], list[str]] = {}
    for region in spec.regions:
        for key in _MESH_REGION_SELECTOR_KEYS:
            value = region.selector.get(key)
            if value in (None, ""):
                continue
            mesh_mappings.setdefault((key, str(value)), []).append(region.name)
    for (_key, external_name), region_names in mesh_mappings.items():
        if len(region_names) > 1:
            issues.append(
                WorkflowValidationIssue(
                    category="region",
                    name=external_name,
                    message=(
                        f"Mesh region {external_name!r} maps to multiple internal "
                        f"regions: {', '.join(region_names)}"
                    ),
                )
            )
    return issues


def _duplicate_name_issues(
    category: str, label: str, names: Iterable[str]
) -> list[WorkflowValidationIssue]:
    counts: dict[str, int] = {}
    for name in names:
        if name:
            counts[name] = counts.get(name, 0) + 1
    return [
        WorkflowValidationIssue(
            category=category,
            name=name,
            message=f"duplicate {label} name {name!r}",
        )
        for name, count in counts.items()
        if count > 1
    ]


def _solver_family_compatibility_issues(
    spec: ProblemSpec,
) -> list[WorkflowValidationIssue]:
    issues: list[WorkflowValidationIssue] = []
    if spec.solver.kind == "validation_script":
        return issues

    if spec.solver.kind == "solid_mechanics":
        for material in spec.materials:
            if material.model != "solid_mechanics":
                issues.append(
                    WorkflowValidationIssue(
                        category="compatibility",
                        name=material.model,
                        message=(
                            "solver 'solid_mechanics' requires material model "
                            "'solid_mechanics'"
                        ),
                    )
                )
        for step in spec.analysis_steps:
            if step.kind != "solid_mechanics":
                issues.append(
                    WorkflowValidationIssue(
                        category="compatibility",
                        name=step.kind,
                        message=(
                            "solver 'solid_mechanics' requires analysis step kind "
                            "'solid_mechanics'"
                        ),
                    )
                )
        issues.extend(
            _boundary_condition_compatibility_issues(
                spec,
                allowed_boundary_conditions=_SOLID_BOUNDARY_CONDITIONS,
            )
        )
        issues.extend(
            _output_compatibility_issues(
                spec,
                allowed_fields=_solid_allowed_fields(spec),
                allowed_history=_SOLID_HISTORY_OUTPUTS,
            )
        )
        return issues

    if spec.solver.kind in _FRACTURE_SOLVERS:
        for material in spec.materials:
            if material.model not in _FRACTURE_MATERIAL_MODELS:
                issues.append(
                    WorkflowValidationIssue(
                        category="compatibility",
                        name=material.model,
                        message=(
                            f"solver {spec.solver.kind!r} does not support "
                            f"material model {material.model!r}"
                        ),
                    )
                )
        for step in spec.analysis_steps:
            if step.kind not in _FRACTURE_ANALYSIS_STEPS:
                issues.append(
                    WorkflowValidationIssue(
                        category="compatibility",
                        name=step.kind,
                        message=(
                            f"solver {spec.solver.kind!r} does not support "
                            f"analysis step kind {step.kind!r}"
                        ),
                    )
                )
        allowed_fields = set(_FRACTURE_FIELD_OUTPUTS)
        if spec.solver.kind == "explicit":
            allowed_fields.update(_DYNAMIC_ONLY_FIELD_OUTPUTS)
        issues.extend(
            _boundary_condition_compatibility_issues(
                spec,
                allowed_boundary_conditions=_FRACTURE_BOUNDARY_CONDITIONS,
            )
        )
        issues.extend(
            _output_compatibility_issues(
                spec,
                allowed_fields=allowed_fields,
                allowed_history=_FRACTURE_HISTORY_OUTPUTS,
            )
        )
    return issues


def _boundary_condition_duplicate_name_issues(
    spec: ProblemSpec,
) -> list[WorkflowValidationIssue]:
    grouped: dict[str, list] = {}
    for bc in spec.boundary_conditions:
        if bc.name:
            grouped.setdefault(bc.name, []).append(bc)

    issues: list[WorkflowValidationIssue] = []
    for name, bcs in grouped.items():
        if len(bcs) <= 1:
            continue
        regions = {bc.region for bc in bcs}
        kinds = {bc.kind for bc in bcs}
        components = [bc.component for bc in bcs]
        unique_components = {
            int(component) for component in components
            if component is not None
        }
        is_component_split = (
            len(regions) == 1
            and len(kinds) == 1
            and all(component is not None for component in components)
            and len(unique_components) == len(components)
        )
        if not is_component_split:
            issues.append(
                WorkflowValidationIssue(
                    category="boundary_condition",
                    name=name,
                    message=f"duplicate boundary condition name {name!r}",
                )
            )
    return issues


def _solid_allowed_fields(spec: ProblemSpec) -> set[str]:
    allowed = set(_SOLID_FIELD_OUTPUTS)
    example = str(spec.solver.parameters.get("example", ""))
    allowed.update(_SOLID_EXAMPLE_EXTRA_FIELDS.get(example, set()))
    return allowed


def _boundary_condition_compatibility_issues(
    spec: ProblemSpec,
    *,
    allowed_boundary_conditions: set[str],
) -> list[WorkflowValidationIssue]:
    issues: list[WorkflowValidationIssue] = []
    for bc in spec.boundary_conditions:
        if bc.kind not in allowed_boundary_conditions:
            issues.append(
                WorkflowValidationIssue(
                    category="compatibility",
                    name=bc.kind,
                    message=(
                        f"solver {spec.solver.kind!r} does not support "
                        f"boundary condition {bc.kind!r}"
                    ),
                )
            )
    return issues


def _output_compatibility_issues(
    spec: ProblemSpec,
    *,
    allowed_fields: set[str],
    allowed_history: set[str],
) -> list[WorkflowValidationIssue]:
    issues: list[WorkflowValidationIssue] = []
    for field in spec.outputs.fields:
        if field.name not in allowed_fields:
            example = spec.solver.parameters.get("example")
            runner_detail = f" for example {example!r}" if example else ""
            issues.append(
                WorkflowValidationIssue(
                    category="compatibility",
                    name=field.name,
                    message=(
                        f"solver {spec.solver.kind!r} does not support "
                        f"field output {field.name!r}{runner_detail}"
                    ),
                )
            )
    for history in spec.outputs.history:
        if history.name not in allowed_history:
            issues.append(
                WorkflowValidationIssue(
                    category="compatibility",
                    name=history.name,
                    message=(
                        f"solver {spec.solver.kind!r} does not support "
                        f"history output {history.name!r}"
                    ),
                )
            )
    return issues
