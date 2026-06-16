"""Domain-named public helpers for the workflow contract.

These classes keep user-facing names clean while converting to the internal
``*Spec`` contract used by adapters and validators.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .specs import (
    AnalysisStepSpec,
    BoundaryConditionSpec,
    FieldOutputSpec,
    GeometrySpec,
    HistoryOutputSpec,
    InitialConditionSpec,
    MeshSpec,
    OutputSpec,
    PostprocessSpec,
    RegionSpec,
    SolverSpec,
)


_DOF_COMPONENTS = {"x": 0, "y": 1, "0": 0, "1": 1}


def _component(dof: str | int | None) -> int | None:
    if dof is None:
        return None
    if isinstance(dof, int):
        return dof
    key = str(dof).lower().strip()
    if key not in _DOF_COMPONENTS:
        raise ValueError(f"Unsupported dof/component {dof!r}")
    return _DOF_COMPONENTS[key]


@dataclass(frozen=True)
class Geometry:
    kind: str
    parameters: dict[str, Any] = field(default_factory=dict)
    units: str = "mm"
    primitives: dict[str, Any] | None = None
    domain: dict[str, Any] | None = None
    named_groups: dict[str, Any] | None = None

    def __init__(
        self,
        kind: str,
        *,
        units: str = "mm",
        primitives: dict[str, Any] | None = None,
        domain: dict[str, Any] | None = None,
        named_groups: dict[str, Any] | None = None,
        **parameters: Any,
    ):
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "parameters", dict(parameters))
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "primitives", primitives)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "named_groups", named_groups)

    @classmethod
    def rectangle(
        cls,
        *,
        width: Any,
        height: Any,
        units: str = "mm",
        **parameters: Any,
    ) -> "Geometry":
        return cls("rectangle", width=width, height=height, units=units, **parameters)

    def to_spec(self) -> GeometrySpec:
        return GeometrySpec(
            kind=self.kind,
            parameters=dict(self.parameters),
            units=self.units,
            primitives=self.primitives,
            domain=self.domain,
            named_groups=self.named_groups,
        )


@dataclass(frozen=True)
class Mesh:
    path: str
    kind: str = "file"
    parameters: dict[str, Any] = field(default_factory=dict)

    def __init__(self, path: str, kind: str = "file", **parameters: Any):
        object.__setattr__(self, "path", str(path))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "parameters", dict(parameters))

    def to_spec(self) -> MeshSpec:
        return MeshSpec(
            kind=self.kind,
            path=self.path,
            parameters=dict(self.parameters),
        )


@dataclass(frozen=True)
class Region:
    name: str
    kind: str = "region"
    selector: dict[str, Any] = field(default_factory=dict)

    def __init__(self, name: str, kind: str = "region", **selector: Any):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "selector", dict(selector))

    def to_spec(self) -> RegionSpec:
        return RegionSpec(name=self.name, kind=self.kind, selector=dict(self.selector))


@dataclass(frozen=True)
class InitialCondition:
    field: str
    region: str | None = None
    value: Any = None
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def damage(
        cls, *, region: str | None = None, value: Any = 1.0, **parameters: Any
    ) -> "InitialCondition":
        return cls("damage", region=region, value=value, parameters=dict(parameters))

    def to_spec(self) -> InitialConditionSpec:
        return InitialConditionSpec(
            field=self.field,
            region=self.region,
            value=self.value,
            parameters=dict(self.parameters),
        )


@dataclass(frozen=True)
class BoundaryCondition:
    kind: str
    region: str
    dof: str | int | None = None
    value: Any = None
    name: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fixed(
        cls, *, region: str, dof: str | int | None = None, name: str | None = None
    ) -> "BoundaryCondition":
        return cls("fix", region=region, dof=dof, name=name)

    @classmethod
    def displacement(
        cls,
        *,
        region: str,
        dof: str | int | None = None,
        value: Any,
        name: str | None = None,
        **parameters: Any,
    ) -> "BoundaryCondition":
        return cls(
            "prescribe",
            region=region,
            dof=dof,
            value=value,
            name=name,
            parameters=dict(parameters),
        )

    @classmethod
    def traction(
        cls,
        *,
        region: str,
        dof: str | int | None = None,
        value: Any,
        name: str | None = None,
        **parameters: Any,
    ) -> "BoundaryCondition":
        return cls(
            "traction",
            region=region,
            dof=dof,
            value=value,
            name=name,
            parameters=dict(parameters),
        )

    def to_spec(self) -> BoundaryConditionSpec:
        return BoundaryConditionSpec(
            kind=self.kind,
            region=self.region,
            component=_component(self.dof),
            value=self.value,
            name=self.name,
            parameters=dict(self.parameters),
        )


@dataclass(frozen=True)
class AnalysisStep:
    name: str
    kind: str
    controls: dict[str, Any] = field(default_factory=dict)
    active_boundary_conditions: tuple[str, ...] = ()

    def __init__(
        self,
        name: str,
        *,
        kind: str,
        controls: dict[str, Any] | None = None,
        active_boundary_conditions: list[str] | tuple[str, ...] = (),
    ):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "controls", dict(controls or {}))
        object.__setattr__(
            self, "active_boundary_conditions", tuple(active_boundary_conditions)
        )

    def to_spec(self) -> AnalysisStepSpec:
        return AnalysisStepSpec(
            name=self.name,
            kind=self.kind,
            controls=dict(self.controls),
            active_boundary_conditions=tuple(self.active_boundary_conditions),
        )


@dataclass(frozen=True)
class SolverSettings:
    kind: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def __init__(self, kind: str, **parameters: Any):
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "parameters", dict(parameters))

    def to_spec(self) -> SolverSpec:
        return SolverSpec(kind=self.kind, parameters=dict(self.parameters))


@dataclass(frozen=True)
class FieldOutput:
    name: str
    every: int = 1
    parameters: dict[str, Any] = field(default_factory=dict)

    def __init__(self, name: str, every: int = 1, **parameters: Any):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "every", int(every))
        object.__setattr__(self, "parameters", dict(parameters))

    def to_spec(self) -> FieldOutputSpec:
        return FieldOutputSpec(
            name=self.name,
            every=self.every,
            parameters=dict(self.parameters),
        )


@dataclass(frozen=True)
class HistoryOutput:
    name: str
    every: int = 1
    region: str | None = None
    component: int | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        name: str,
        every: int = 1,
        *,
        region: str | None = None,
        dof: str | int | None = None,
        component: str | int | None = None,
        **parameters: Any,
    ):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "every", int(every))
        object.__setattr__(self, "region", region)
        selected_component = dof if dof is not None else component
        object.__setattr__(self, "component", _component(selected_component))
        object.__setattr__(self, "parameters", dict(parameters))

    def to_spec(self) -> HistoryOutputSpec:
        return HistoryOutputSpec(
            name=self.name,
            every=self.every,
            region=self.region,
            component=self.component,
            parameters=dict(self.parameters),
        )


@dataclass(frozen=True)
class Postprocess:
    kind: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def __init__(self, kind: str, **parameters: Any):
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "parameters", dict(parameters))

    def to_spec(self) -> PostprocessSpec:
        return PostprocessSpec(kind=self.kind, parameters=dict(self.parameters))


@dataclass(frozen=True)
class Outputs:
    fields: tuple[Any, ...] = ()
    history: tuple[Any, ...] = ()
    visuals: dict[str, Any] = field(default_factory=dict)
    directory: str | None = None

    def __init__(
        self,
        *,
        fields: list[Any] | tuple[Any, ...] = (),
        history: list[Any] | tuple[Any, ...] = (),
        visuals: dict[str, Any] | None = None,
        directory: str | None = None,
    ):
        object.__setattr__(self, "fields", tuple(fields))
        object.__setattr__(
            self,
            "history",
            tuple(
                item if isinstance(item, HistoryOutput) else dict(item)
                for item in history
            ),
        )
        object.__setattr__(self, "visuals", dict(visuals or {}))
        object.__setattr__(self, "directory", directory)

    def to_spec(self) -> OutputSpec:
        field_specs = []
        for field_item in self.fields:
            if isinstance(field_item, FieldOutput):
                field_specs.append(field_item.to_spec())
            elif isinstance(field_item, str):
                field_specs.append(FieldOutputSpec(name=field_item))
            else:
                data = dict(field_item)
                name = data.pop("name")
                every = int(data.pop("every", 1))
                field_specs.append(
                    FieldOutputSpec(name=name, every=every, parameters=data)
                )

        history_specs = []
        for history_item in self.history:
            if isinstance(history_item, HistoryOutput):
                history_specs.append(history_item.to_spec())
                continue
            data = dict(history_item)
            name = data.pop("name")
            every = int(data.pop("every", 1))
            region = data.pop("region", None)
            component = _component(data.pop("dof", data.pop("component", None)))
            history_specs.append(
                HistoryOutputSpec(
                    name=name,
                    every=every,
                    region=region,
                    component=component,
                    parameters=data,
                )
            )

        postprocess = []
        for kind, value in self.visuals.items():
            if not value:
                continue
            if isinstance(value, Postprocess):
                postprocess.append(value.to_spec())
            else:
                postprocess.append(
                    PostprocessSpec(
                        kind=kind,
                        parameters=value if isinstance(value, dict) else {},
                    )
                )
        return OutputSpec(
            directory=self.directory,
            fields=field_specs,
            history=history_specs,
            postprocess=postprocess,
        )
