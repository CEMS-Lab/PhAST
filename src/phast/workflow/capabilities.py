"""Internal workflow capability registry.

The registry records the public workflow-contract names that current adapters
may emit. It is advisory for now: execution still flows through the existing
YAML runners and solver loops.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .specs import ProblemSpec


@dataclass(frozen=True)
class WorkflowCapability:
    category: str
    name: str
    status: str
    public: bool = True
    description: str = ""
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class CapabilityIssue:
    category: str
    name: str
    message: str


_UNAVAILABLE_STATUSES = {"scaffold", "unsupported"}


_CAPABILITIES: tuple[WorkflowCapability, ...] = (
    WorkflowCapability(
        "solver",
        "explicit",
        "production",
        description="Velocity-Verlet explicit dynamics fracture path.",
    ),
    WorkflowCapability(
        "solver",
        "quasi_static",
        "production",
        description="Staggered quasi-static/static brittle-fracture path.",
    ),
    WorkflowCapability(
        "solver",
        "quasi_static_legacy",
        "beta",
        description="Compatibility secant path retained for selected workflows.",
    ),
    WorkflowCapability(
        "solver",
        "solid_mechanics",
        "production",
        description="Promoted solid-mechanics YAML runner path.",
    ),
    WorkflowCapability(
        "solver",
        "validation_script",
        "beta",
        description="Curated plasticity/interface reproducibility-contract route.",
    ),
    WorkflowCapability(
        "solver",
        "coupled_pf_plasticity_cohesive",
        "unsupported",
        description=(
            "Future coupled plasticity, phase-field, and cohesive-interface "
            "solver; not available as a public workflow."
        ),
    ),
    WorkflowCapability(
        "analysis_step",
        "explicit",
        "production",
        description="Dynamic fracture step represented by existing loading config.",
    ),
    WorkflowCapability(
        "analysis_step",
        "quasi_static",
        "production",
        description="Quasi-static fracture step represented by existing loading config.",
    ),
    WorkflowCapability(
        "analysis_step",
        "quasi_static_legacy",
        "beta",
        description="Legacy quasi-static step represented by existing loading config.",
    ),
    WorkflowCapability(
        "analysis_step",
        "solid_mechanics",
        "production",
        description="Solid-mechanics example load step.",
    ),
    WorkflowCapability(
        "analysis_step",
        "validation_script",
        "beta",
        description="Curated validation-script step for beta physics contracts.",
    ),
    WorkflowCapability(
        "material_model",
        "phase_field",
        "production",
        description="Current brittle phase-field material contract model.",
    ),
    WorkflowCapability(
        "material_model",
        "solid_mechanics",
        "production",
        description="Promoted solid-mechanics material contract model.",
    ),
    WorkflowCapability(
        "material_model",
        "cohesive_interface",
        "beta",
        description="Curated cohesive-interface validation contract material.",
    ),
    WorkflowCapability(
        "material_model",
        "diffuse_interface",
        "beta",
        description="Curated diffuse-interface validation contract material.",
    ),
    WorkflowCapability(
        "material_model",
        "ductile_phase_field",
        "beta",
        description="Curated ductile phase-field/plasticity validation material.",
    ),
    WorkflowCapability(
        "material_model",
        "j2_plasticity",
        "beta",
        description="Curated J2 plasticity validation contract material.",
    ),
    WorkflowCapability(
        "material_model",
        "validation_artifact",
        "beta",
        description="Fallback material marker for curated validation artifacts.",
    ),
    WorkflowCapability(
        "boundary_condition",
        "fix",
        "production",
        description="Dirichlet zero-displacement boundary condition.",
    ),
    WorkflowCapability(
        "boundary_condition",
        "prescribe",
        "production",
        description="Dirichlet prescribed displacement boundary condition.",
    ),
    WorkflowCapability(
        "boundary_condition",
        "traction",
        "production",
        description="Boundary traction/load condition.",
    ),
    WorkflowCapability(
        "boundary_condition",
        "neumann",
        "production",
        description="Neumann force/traction boundary condition.",
    ),
    WorkflowCapability(
        "boundary_condition",
        "pf_dirichlet",
        "production",
        description="Phase-field Dirichlet boundary condition.",
    ),
    WorkflowCapability(
        "boundary_condition",
        "rigid_connector",
        "beta",
        description="Compatibility rigid-connector boundary condition.",
    ),
    WorkflowCapability(
        "boundary_condition",
        "symmetry",
        "production",
        description="Symmetry-plane boundary condition used by public examples.",
    ),
    WorkflowCapability(
        "field_output",
        "trajectory",
        "production",
        description="Stored Zarr/H5 trajectory field output.",
    ),
    WorkflowCapability(
        "field_output",
        "vtu",
        "beta",
        description="VTU/PyVista-style visualization field output.",
    ),
    WorkflowCapability(
        "field_output",
        "higher_order_element_fields",
        "scaffold",
        description=(
            "Placeholder for higher-order element field output; helper kernels "
            "exist but workflow output is not coupled."
        ),
    ),
    WorkflowCapability(
        "field_output",
        "damage",
        "production",
        description="Damage field output.",
    ),
    WorkflowCapability(
        "field_output",
        "displacement",
        "production",
        description="Displacement field output.",
    ),
    WorkflowCapability(
        "field_output",
        "history_field",
        "production",
        description="Phase-field history variable output.",
    ),
    WorkflowCapability(
        "field_output",
        "history_field_nodal",
        "production",
        description="Nodal phase-field history variable output.",
    ),
    WorkflowCapability(
        "field_output",
        "psi_plus",
        "production",
        description="Positive strain-energy density output.",
    ),
    WorkflowCapability(
        "field_output",
        "strain",
        "production",
        description="Stored strain field output.",
    ),
    WorkflowCapability(
        "field_output",
        "stress",
        "production",
        description="Stored stress field output.",
    ),
    WorkflowCapability(
        "field_output",
        "velocity",
        "production",
        description="Dynamic velocity field output.",
    ),
    WorkflowCapability(
        "field_output",
        "acceleration",
        "production",
        description="Dynamic acceleration field output.",
    ),
    WorkflowCapability(
        "field_output",
        "von_mises",
        "beta",
        description="Promoted solid-mechanics von Mises output artifact.",
    ),
    WorkflowCapability(
        "field_output",
        "strain_energy",
        "beta",
        description="Promoted solid-mechanics strain-energy visual artifact.",
    ),
    WorkflowCapability(
        "field_output",
        "equivalent_plastic_strain",
        "beta",
        description="Promoted J2 solid-mechanics plastic-strain visual artifact.",
    ),
    WorkflowCapability(
        "field_output",
        "jacobian",
        "beta",
        description="Promoted nonlinear solid-mechanics Jacobian visual artifact.",
    ),
    WorkflowCapability(
        "history_output",
        "reaction",
        "production",
        description="Reaction history for load-displacement workflows.",
    ),
    WorkflowCapability(
        "history_output",
        "reaction_force",
        "production",
        description="Canonical reaction-force history.",
    ),
    WorkflowCapability(
        "history_output",
        "energy",
        "production",
        description="Energy history output.",
    ),
    WorkflowCapability(
        "history_output",
        "response",
        "production",
        description="Promoted solid-mechanics response output.",
    ),
    WorkflowCapability(
        "history_output",
        "load_displacement",
        "production",
        description="Canonical load-displacement response history.",
    ),
    WorkflowCapability(
        "history_output",
        "max_damage",
        "production",
        description="Maximum damage scalar history.",
    ),
    WorkflowCapability(
        "history_output",
        "solver_telemetry",
        "production",
        description="Solver telemetry history.",
    ),
    WorkflowCapability(
        "history_output",
        "timing_per_step",
        "production",
        description="Per-step timing history.",
    ),
    WorkflowCapability(
        "postprocess",
        "plots",
        "beta",
        description="Shared plot generation postprocess.",
    ),
    WorkflowCapability(
        "postprocess",
        "animation",
        "beta",
        description="Shared animation postprocess.",
    ),
    WorkflowCapability(
        "postprocess",
        "thumbnail",
        "beta",
        description="Thumbnail visual artifact.",
    ),
    WorkflowCapability(
        "postprocess",
        "damage_final",
        "beta",
        description="Final damage visual artifact.",
    ),
    WorkflowCapability(
        "postprocess",
        "energy",
        "beta",
        description="Energy plot visual artifact.",
    ),
    WorkflowCapability(
        "postprocess",
        "initial_conditions",
        "beta",
        description="Initial-condition visual artifact.",
    ),
)


def list_capabilities(
    category: str | None = None, *, public_only: bool = True
) -> list[WorkflowCapability]:
    """Return registered workflow capabilities, optionally filtered."""
    capabilities: Iterable[WorkflowCapability] = _CAPABILITIES
    if category is not None:
        capabilities = (item for item in capabilities if item.category == category)
    if public_only:
        capabilities = (item for item in capabilities if item.public)
    return sorted(capabilities, key=lambda item: (item.category, item.name))


def capability_names(category: str, *, public_only: bool = True) -> tuple[str, ...]:
    """Return registered names for a workflow capability category."""
    names: set[str] = set()
    for capability in list_capabilities(category, public_only=public_only):
        names.add(capability.name)
        names.update(capability.aliases)
    return tuple(sorted(names))


def get_capability(
    category: str, name: str, *, public_only: bool = True
) -> WorkflowCapability:
    """Return a registered workflow capability by category and name."""
    for capability in list_capabilities(category, public_only=public_only):
        if name == capability.name or name in capability.aliases:
            return capability
    available = ", ".join(capability_names(category, public_only=public_only))
    raise KeyError(
        f"Unknown workflow capability {category}:{name!r}. Available names: {available}"
    )


def validate_problem_spec_capabilities(
    spec: ProblemSpec, *, public_only: bool = True
) -> list[CapabilityIssue]:
    """Return advisory issues for unregistered names emitted by a ProblemSpec."""
    issues: list[CapabilityIssue] = []
    _append_missing(issues, "solver", spec.solver.kind, public_only=public_only)
    for step in spec.analysis_steps:
        _append_missing(issues, "analysis_step", step.kind, public_only=public_only)
    for material in spec.materials:
        _append_missing(
            issues, "material_model", material.model, public_only=public_only
        )
    for bc in spec.boundary_conditions:
        _append_missing(
            issues, "boundary_condition", bc.kind, public_only=public_only
        )
    for field_output in spec.outputs.fields:
        _append_missing(
            issues, "field_output", field_output.name, public_only=public_only
        )
    for history_output in spec.outputs.history:
        _append_missing(
            issues, "history_output", history_output.name, public_only=public_only
        )
    for postprocess in spec.outputs.postprocess:
        _append_missing(
            issues, "postprocess", postprocess.kind, public_only=public_only
        )
    return issues


def _append_missing(
    issues: list[CapabilityIssue],
    category: str,
    name: str,
    *,
    public_only: bool,
) -> None:
    try:
        capability = get_capability(category, name, public_only=public_only)
    except KeyError:
        issues.append(
            CapabilityIssue(
                category=category,
                name=name,
                message=f"{category} capability {name!r} is not registered",
            )
        )
        return
    if capability.status in _UNAVAILABLE_STATUSES:
        issues.append(
            CapabilityIssue(
                category=category,
                name=name,
                message=(
                    f"{category} capability {name!r} has status "
                    f"{capability.status!r} and is not available for execution"
                ),
            )
        )
