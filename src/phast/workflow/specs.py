"""Internal workflow contract objects for PhAST problem definitions.

These dataclasses are intentionally internal and additive. They provide one
validated surface that legacy YAML configs and the current fluent ``Problem``
API can compile to before later workflow stages decide how to execute them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _require_mapping(name: str, value: dict[str, Any]) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")


@dataclass(frozen=True)
class GeometrySpec:
    kind: str
    parameters: dict[str, Any] = field(default_factory=dict)
    units: str = "mm"
    primitives: dict[str, Any] | None = None
    domain: dict[str, Any] | None = None
    named_groups: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("GeometrySpec.kind is required")
        _require_mapping("GeometrySpec.parameters", self.parameters)


@dataclass(frozen=True)
class MeshSpec:
    kind: str = "external"
    path: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("MeshSpec.kind is required")
        _require_mapping("MeshSpec.parameters", self.parameters)


@dataclass(frozen=True)
class RegionSpec:
    name: str
    kind: str = "node_set"
    selector: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("RegionSpec.name is required")
        _require_mapping("RegionSpec.selector", self.selector)


@dataclass(frozen=True)
class MaterialSpec:
    name: str
    model: str = "phase_field"
    parameters: dict[str, Any] = field(default_factory=dict)
    region: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("MaterialSpec.name is required")
        _require_mapping("MaterialSpec.parameters", self.parameters)


@dataclass(frozen=True)
class InitialConditionSpec:
    field: str
    region: str | None = None
    value: Any = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.field:
            raise ValueError("InitialConditionSpec.field is required")
        _require_mapping("InitialConditionSpec.parameters", self.parameters)


@dataclass(frozen=True)
class BoundaryConditionSpec:
    kind: str
    region: str
    component: int | None = None
    value: Any = None
    parameters: dict[str, Any] = field(default_factory=dict)
    name: str | None = None

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("BoundaryConditionSpec.kind is required")
        if not self.region:
            raise ValueError("BoundaryConditionSpec.region is required")
        _require_mapping("BoundaryConditionSpec.parameters", self.parameters)


@dataclass(frozen=True)
class AnalysisStepSpec:
    name: str
    kind: str
    controls: dict[str, Any] = field(default_factory=dict)
    active_boundary_conditions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("AnalysisStepSpec.name is required")
        if not self.kind:
            raise ValueError("AnalysisStepSpec.kind is required")
        _require_mapping("AnalysisStepSpec.controls", self.controls)


@dataclass(frozen=True)
class SolverSpec:
    kind: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("SolverSpec.kind is required")
        _require_mapping("SolverSpec.parameters", self.parameters)


@dataclass(frozen=True)
class FieldOutputSpec:
    name: str
    every: int = 1
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FieldOutputSpec.name is required")
        if self.every < 1:
            raise ValueError("FieldOutputSpec.every must be >= 1")
        _require_mapping("FieldOutputSpec.parameters", self.parameters)


@dataclass(frozen=True)
class HistoryOutputSpec:
    name: str
    every: int = 1
    region: str | None = None
    component: int | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("HistoryOutputSpec.name is required")
        if self.every < 1:
            raise ValueError("HistoryOutputSpec.every must be >= 1")
        _require_mapping("HistoryOutputSpec.parameters", self.parameters)


@dataclass(frozen=True)
class PostprocessSpec:
    kind: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("PostprocessSpec.kind is required")
        _require_mapping("PostprocessSpec.parameters", self.parameters)


@dataclass(frozen=True)
class OutputSpec:
    directory: str | None = None
    fields: list[FieldOutputSpec] = field(default_factory=list)
    history: list[HistoryOutputSpec] = field(default_factory=list)
    postprocess: list[PostprocessSpec] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_mapping("OutputSpec.parameters", self.parameters)


@dataclass(frozen=True)
class ProblemSpec:
    name: str
    schema_version: int = 1
    reference: str = ""
    geometry: GeometrySpec | None = None
    mesh: MeshSpec | None = None
    regions: list[RegionSpec] = field(default_factory=list)
    materials: list[MaterialSpec] = field(default_factory=list)
    initial_conditions: list[InitialConditionSpec] = field(default_factory=list)
    boundary_conditions: list[BoundaryConditionSpec] = field(default_factory=list)
    analysis_steps: list[AnalysisStepSpec] = field(default_factory=list)
    solver: SolverSpec = field(default_factory=lambda: SolverSpec(kind="explicit"))
    outputs: OutputSpec = field(default_factory=OutputSpec)
    source: str = "internal"
    source_path: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ProblemSpec.name is required")
        if self.geometry is None and self.mesh is None:
            raise ValueError("ProblemSpec requires geometry or mesh")
        if not self.materials:
            raise ValueError("ProblemSpec requires at least one material")
        if not self.analysis_steps:
            raise ValueError("ProblemSpec requires at least one analysis step")

    def run(self, *, output_dir: str | None = None, validate_only: bool = False) -> int:
        """Run a YAML-backed spec through the existing compatibility CLI."""
        from .execution import run_problem_spec

        return run_problem_spec(
            self,
            output_dir=output_dir,
            validate_only=validate_only,
        )
