"""Adapters from existing public inputs to the internal workflow contract."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from ..config.config import (
    BoundaryConditionEntry,
    GeometryConfig,
    InitialConditionsConfig,
    LoadingConfig,
    MaterialConfig,
    OutputConfig,
    ProblemConfig,
    SolverSettings,
    load_config,
)
from ..solid_mechanics_runner import solid_example_id
from .specs import (
    AnalysisStepSpec,
    BoundaryConditionSpec,
    FieldOutputSpec,
    GeometrySpec,
    HistoryOutputSpec,
    InitialConditionSpec,
    MaterialSpec,
    MeshSpec,
    OutputSpec,
    PostprocessSpec,
    ProblemSpec,
    RegionSpec,
    SolverSpec,
)


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return {
            key: _plain(item)
            for key, item in asdict(value).items()
            if item is not None
        }
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_plain(item) for item in value if item is not None]
    return value


def _coerce_numeric_strings(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _coerce_numeric_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_coerce_numeric_strings(item) for item in value]
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    return value


def _drop_empty(mapping: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in mapping.items()
        if value not in (None, {}, [], "")
    }


_DOF_COMPONENTS = {"x": 0, "y": 1, "0": 0, "1": 1}
_BC_KIND_ALIASES = {
    "fixed": "fix",
    "dirichlet": "prescribe",
    "displacement": "prescribe",
    "prescribed_displacement": "prescribe",
    "phase_field_dirichlet": "pf_dirichlet",
}


def _component_from_entry(entry: dict[str, Any]) -> int | None:
    if "component" in entry:
        component = entry["component"]
    else:
        component = entry.get("dof")
    if component is None:
        return None
    if isinstance(component, int):
        return component
    key = str(component).lower().strip()
    if key not in _DOF_COMPONENTS:
        raise ValueError(f"Unsupported dof/component {component!r}")
    return _DOF_COMPONENTS[key]


def _normalized_bc_kind(kind: str | None) -> str:
    if kind is None:
        raise ValueError("Boundary condition requires type or kind")
    key = str(kind).lower().strip()
    return _BC_KIND_ALIASES.get(key, key)


def _geometry_spec(config: GeometryConfig) -> tuple[GeometrySpec | None, MeshSpec | None]:
    parameters = dict(config.parameters or {})
    if config.mesh_path:
        return None, MeshSpec(kind="file", path=str(config.mesh_path))
    if config.primitives:
        return GeometrySpec(
            kind="primitive_dsl",
            parameters=_drop_empty({"mesh": config.mesh}),
            units=config.units,
            primitives=config.primitives,
            domain=config.domain,
            named_groups=config.named_groups,
        ), None
    return GeometrySpec(kind=config.type, parameters=parameters, units=config.units), None


def _material_spec(config: MaterialConfig, *, model: str = "phase_field") -> MaterialSpec:
    parameters = dict(config.overrides or {})
    for key, value in _plain(config).items():
        if key not in {"preset", "overrides"}:
            parameters[key] = value
    name = config.preset or model
    return MaterialSpec(
        name=name,
        model=model,
        parameters=_drop_empty(parameters),
        region=getattr(config, "_workflow_region", None),
    )


def _bc_spec(entry: BoundaryConditionEntry, index: int) -> BoundaryConditionSpec:
    parameters = _plain(entry)
    for key in ("nodes", "type", "component", "value"):
        parameters.pop(key, None)
    return BoundaryConditionSpec(
        kind=entry.type,
        region=entry.nodes,
        component=entry.component,
        value=entry.value,
        parameters=_drop_empty(parameters),
        name=getattr(entry, "_workflow_name", None) or f"bc_{index}_{entry.type}_{entry.nodes}",
    )


def _initial_condition_specs(config: InitialConditionsConfig | None) -> list[InitialConditionSpec]:
    if config is None:
        return []
    specs: list[InitialConditionSpec] = []
    for nodes in config.preseed_notch_nodesets or []:
        specs.append(InitialConditionSpec(field="damage", region=nodes, value=1.0))
    for entry in config.preseed_damage or []:
        parameters = dict(entry)
        region = parameters.pop("nodes", None)
        value = parameters.pop("value", 1.0)
        if region is None and "region" in parameters:
            region = None
        specs.append(
            InitialConditionSpec(
                field="damage",
                region=region,
                value=value,
                parameters=_drop_empty(parameters),
            )
        )
    return specs


def _analysis_step_spec(loading: LoadingConfig, solver: SolverSettings) -> AnalysisStepSpec:
    controls = _drop_empty(
        dict(getattr(loading, "_workflow_controls", None) or _plain(loading))
    )
    kind = solver.solver_type
    name = getattr(loading, "_workflow_step_name", None) or loading.protocol or kind
    return AnalysisStepSpec(
        name=name,
        kind=kind,
        controls=controls,
        active_boundary_conditions=tuple(
            getattr(loading, "_workflow_active_boundary_conditions", ())
        ),
    )


def _solver_spec(config: SolverSettings) -> SolverSpec:
    parameters = _plain(config)
    kind = parameters.pop("solver_type", config.solver_type)
    parameters.update(dict(getattr(config, "_workflow_parameters", {}) or {}))
    return SolverSpec(kind=kind, parameters=_drop_empty(parameters))


def _output_spec(config: OutputConfig) -> OutputSpec:
    params = _plain(config)
    fields: list[FieldOutputSpec] = []
    history: list[HistoryOutputSpec] = []
    postprocess: list[PostprocessSpec] = []
    workflow_fields = list(getattr(config, "_workflow_fields", []) or [])
    workflow_history = list(getattr(config, "_workflow_history", []) or [])

    if config.trajectory and not workflow_fields:
        fields.append(
            FieldOutputSpec(
                name="trajectory",
                every=max(1, int(config.h5_every)),
                parameters={"format": config.trajectory_format},
            )
        )
    if config.vtu:
        fields.append(
            FieldOutputSpec(
                name="vtu",
                every=max(1, int(config.vtu_every)),
                parameters={"format": config.viz_format},
            )
        )
    for field in workflow_fields:
        if isinstance(field, str):
            fields.append(FieldOutputSpec(name=field))
        else:
            item = dict(field or {})
            name = item.pop("name")
            every = int(item.pop("every", 1))
            fields.append(
                FieldOutputSpec(
                    name=name,
                    every=every,
                    parameters=_drop_empty(item),
                )
            )
    if config.reaction_node_set and not workflow_history:
        history.append(
            HistoryOutputSpec(
                name="reaction",
                every=1,
                region=config.reaction_node_set,
                component=config.reaction_component,
            )
        )
    for item in workflow_history:
        if isinstance(item, str):
            history.append(HistoryOutputSpec(name=item))
            continue
        data = dict(item or {})
        name = data.pop("name")
        every = int(data.pop("every", 1))
        region = data.pop("region", None)
        component = _component_from_entry(data)
        data.pop("component", None)
        data.pop("dof", None)
        history.append(
            HistoryOutputSpec(
                name=name,
                every=every,
                region=region,
                component=component,
                parameters=_drop_empty(data),
            )
        )
    if config.plots:
        postprocess.append(PostprocessSpec(kind="plots"))
    if config.gif:
        postprocess.append(
            PostprocessSpec(
                kind="animation",
                parameters={
                    "format": config.animation_format,
                    "fields": config.gif_fields,
                    "frames": config.gif_frames,
                },
            )
        )

    return OutputSpec(
        directory=config.output_dir,
        fields=fields,
        history=history,
        postprocess=postprocess,
        parameters=_drop_empty(params),
    )


def _region_specs(
    geometry: GeometryConfig,
    bcs: list[BoundaryConditionEntry],
    initial_conditions: list[InitialConditionSpec],
    outputs: OutputSpec,
    workflow_regions: list[dict[str, Any]] | None = None,
) -> list[RegionSpec]:
    regions: dict[str, RegionSpec] = {}
    for entry in workflow_regions or []:
        data = dict(entry or {})
        name = data.pop("name")
        kind = data.pop("kind", "region")
        selector = data.pop("selector", data)
        regions[name] = RegionSpec(name=name, kind=kind, selector=dict(selector or {}))
    for name, selector in (geometry.named_groups or {}).items():
        regions[name] = RegionSpec(name=name, kind="geometry_group", selector=selector)
    for entry in bcs:
        regions.setdefault(entry.nodes, RegionSpec(name=entry.nodes))
        if entry.master:
            regions.setdefault(entry.master, RegionSpec(name=entry.master))
    for spec in initial_conditions:
        if spec.region:
            regions.setdefault(spec.region, RegionSpec(name=spec.region))
    for hist in outputs.history:
        if hist.region:
            regions.setdefault(hist.region, RegionSpec(name=hist.region))
    return list(regions.values())


def problem_spec_from_config(
    config: ProblemConfig,
    *,
    source: str = "yaml:v1",
    source_path: str | None = None,
) -> ProblemSpec:
    geometry, mesh = _geometry_spec(config.geometry)
    material_model = (
        "solid_mechanics"
        if config.solver.solver_type == "solid_mechanics"
        else "phase_field"
    )
    material = _material_spec(config.material, model=material_model)
    bcs = [_bc_spec(entry, index) for index, entry in enumerate(config.boundary_conditions)]
    initial_conditions = _initial_condition_specs(config.initial_conditions)
    outputs = _output_spec(config.output)
    regions = _region_specs(
        config.geometry,
        config.boundary_conditions,
        initial_conditions,
        outputs,
        getattr(config, "_workflow_regions", None),
    )
    solver = _solver_spec(config.solver)
    if getattr(config.device, "device", None):
        solver = SolverSpec(
            kind=solver.kind,
            parameters=_drop_empty(
                {**solver.parameters, "device": config.device.device}
            ),
        )

    return ProblemSpec(
        schema_version=config.schema_version,
        name=config.name,
        reference=config.reference,
        geometry=geometry,
        mesh=mesh,
        regions=regions,
        materials=[material],
        initial_conditions=initial_conditions,
        boundary_conditions=bcs,
        analysis_steps=[_analysis_step_spec(config.loading, config.solver)],
        solver=solver,
        outputs=outputs,
        source=source,
        source_path=source_path,
    )


def _solid_mechanics_spec(
    raw: dict[str, Any],
    example_id: str,
    *,
    source_path: str | None = None,
) -> ProblemSpec:
    output = raw.get("output") or {}
    outputs = OutputSpec(
        directory=output.get("directory"),
        fields=[
            FieldOutputSpec(name="displacement"),
            FieldOutputSpec(name="von_mises"),
        ],
        history=[HistoryOutputSpec(name="response")],
        parameters=dict(output),
    )
    return ProblemSpec(
        schema_version=int(raw.get("schema_version", 1)),
        name=example_id,
        mesh=MeshSpec(
            kind="structured_grid",
            parameters=_coerce_numeric_strings(dict(raw.get("mesh") or {})),
        ),
        materials=[
            MaterialSpec(
                name="solid_material",
                model="solid_mechanics",
                parameters=_coerce_numeric_strings(dict(raw.get("material") or {})),
            )
        ],
        analysis_steps=[
            AnalysisStepSpec(
                name="load",
                kind="solid_mechanics",
                controls=_coerce_numeric_strings(dict(raw.get("loading") or {})),
            )
        ],
        solver=SolverSpec(kind="solid_mechanics", parameters={"example": example_id}),
        outputs=outputs,
        source="yaml:v1-solid-mechanics",
        source_path=source_path,
    )


def _schema_v2_geometry(raw: dict[str, Any]) -> tuple[GeometrySpec | None, MeshSpec | None]:
    geometry = dict(raw.get("geometry") or {})
    if geometry.get("mesh_path"):
        parameters = {
            key: value
            for key, value in geometry.items()
            if key not in {"mesh_path", "type", "kind"}
        }
        return None, MeshSpec(
            kind=geometry.get("mesh_type", "file"),
            path=str(geometry["mesh_path"]),
            parameters=_drop_empty(parameters),
        )
    kind = geometry.get("type") or geometry.get("kind")
    if not kind:
        raise ValueError("schema_version 2 geometry requires type, kind, or mesh_path")
    parameters = dict(geometry.get("parameters") or {})
    for key, value in geometry.items():
        if key not in {
            "type",
            "kind",
            "parameters",
            "units",
            "primitives",
            "domain",
            "named_groups",
        }:
            parameters[key] = value
    return GeometrySpec(
        kind=kind,
        parameters=_drop_empty(parameters),
        units=geometry.get("units", "mm"),
        primitives=geometry.get("primitives"),
        domain=geometry.get("domain"),
        named_groups=geometry.get("named_groups"),
    ), None


def _schema_v2_regions(
    raw: dict[str, Any], assignments: list[dict[str, Any]]
) -> list[RegionSpec]:
    regions: dict[str, RegionSpec] = {}
    raw_regions = raw.get("regions") or {}
    if isinstance(raw_regions, dict):
        for name, selector in raw_regions.items():
            selector_map = dict(selector or {}) if isinstance(selector, dict) else {"value": selector}
            kind = selector_map.pop("kind", "region")
            regions[name] = RegionSpec(name=name, kind=kind, selector=selector_map)
    else:
        for entry in raw_regions:
            data = dict(entry or {})
            name = data.pop("name")
            kind = data.pop("kind", "region")
            selector = data.pop("selector", data)
            regions[name] = RegionSpec(name=name, kind=kind, selector=dict(selector or {}))
    for assignment in assignments:
        region = assignment.get("region")
        if region:
            regions.setdefault(
                region,
                RegionSpec(name=region, kind="material_region", selector={}),
            )
    return list(regions.values())


def _schema_v2_materials(
    raw: dict[str, Any], assignments: list[dict[str, Any]]
) -> list[MaterialSpec]:
    raw_materials = raw.get("materials") or {}
    materials_by_name: dict[str, MaterialSpec] = {}
    material_entries: list[MaterialSpec] = []
    if isinstance(raw_materials, dict):
        for name, entry in raw_materials.items():
            data = dict(entry or {})
            model = data.pop("model", "phase_field")
            params = data.pop("parameters", data)
            spec = MaterialSpec(
                name=name,
                model=model,
                parameters=_drop_empty(dict(params or {})),
                region=data.get("region"),
            )
            materials_by_name[name] = spec
            material_entries.append(spec)
    else:
        for entry in raw_materials:
            data = dict(entry or {})
            name = data.pop("name")
            model = data.pop("model", "phase_field")
            region = data.pop("region", None)
            params = data.pop("parameters", data)
            spec = MaterialSpec(
                name=name,
                model=model,
                parameters=_drop_empty(dict(params or {})),
                region=region,
            )
            materials_by_name.setdefault(name, spec)
            material_entries.append(spec)
    if not assignments:
        return material_entries
    assigned: list[MaterialSpec] = []
    for assignment in assignments:
        name = assignment.get("material")
        if name not in materials_by_name:
            raise ValueError(f"Unknown material assignment {name!r}")
        base = materials_by_name[name]
        assigned.append(
            MaterialSpec(
                name=base.name,
                model=base.model,
                parameters=base.parameters,
                region=assignment.get("region", base.region),
            )
        )
    return assigned


def _schema_v2_initial_conditions(raw: dict[str, Any]) -> list[InitialConditionSpec]:
    specs: list[InitialConditionSpec] = []
    for entry in raw.get("initial_conditions") or []:
        data = dict(entry or {})
        field_name = data.pop("field")
        region = data.pop("region", None)
        value = data.pop("value", None)
        specs.append(
            InitialConditionSpec(
                field=field_name,
                region=region,
                value=value,
                parameters=_drop_empty(data),
            )
        )
    return specs


def _schema_v2_boundary_conditions(raw: dict[str, Any]) -> list[BoundaryConditionSpec]:
    specs: list[BoundaryConditionSpec] = []
    for index, entry in enumerate(raw.get("boundary_conditions") or []):
        data = dict(entry or {})
        kind = _normalized_bc_kind(data.pop("type", data.pop("kind", None)))
        region = data.pop("region", data.pop("nodes", None))
        value = data.pop("value", None)
        name = data.pop("name", None) or f"bc_{index}_{kind}_{region}"
        component = _component_from_entry(data)
        data.pop("component", None)
        data.pop("dof", None)
        specs.append(
            BoundaryConditionSpec(
                kind=kind,
                region=region,
                component=component,
                value=value,
                parameters=_drop_empty(data),
                name=name,
            )
        )
    return specs


def _schema_v2_analysis_steps(raw: dict[str, Any]) -> list[AnalysisStepSpec]:
    steps: list[AnalysisStepSpec] = []
    for entry in raw.get("analysis_steps") or []:
        data = dict(entry or {})
        name = data.pop("name")
        kind = data.pop("type", data.pop("kind", None))
        controls = dict(data.pop("controls", {}) or {})
        active_bcs = tuple(data.pop("active_boundary_conditions", ()) or ())
        controls.update(_drop_empty(data))
        steps.append(
            AnalysisStepSpec(
                name=name,
                kind=kind,
                controls=_drop_empty(controls),
                active_boundary_conditions=active_bcs,
            )
        )
    return steps


def _schema_v2_solver(raw: dict[str, Any], steps: list[AnalysisStepSpec]) -> SolverSpec:
    data = dict(raw.get("solver") or {})
    kind = data.pop("type", data.pop("kind", None))
    if kind is None and steps:
        kind = steps[0].kind
    return SolverSpec(kind=kind or "explicit", parameters=_drop_empty(data))


def _schema_v2_outputs(raw: dict[str, Any]) -> OutputSpec:
    data = dict(raw.get("outputs") or {})
    fields = [
        FieldOutputSpec(
            name=entry["name"],
            every=int(entry.get("every", 1)),
            parameters=_drop_empty(
                {key: value for key, value in entry.items() if key not in {"name", "every"}}
            ),
        )
        for entry in data.get("fields", []) or []
    ]
    history = []
    for entry in data.get("history", []) or []:
        item = dict(entry or {})
        name = item.pop("name")
        every = int(item.pop("every", 1))
        region = item.pop("region", None)
        component = _component_from_entry(item)
        item.pop("component", None)
        item.pop("dof", None)
        history.append(
            HistoryOutputSpec(
                name=name,
                every=every,
                region=region,
                component=component,
                parameters=_drop_empty(item),
            )
        )
    postprocess: list[PostprocessSpec] = []
    visuals = data.get("visuals", {}) or {}
    if isinstance(visuals, dict):
        for kind, value in visuals.items():
            if value:
                params = value if isinstance(value, dict) else {}
                postprocess.append(PostprocessSpec(kind=kind, parameters=dict(params)))
    else:
        for entry in visuals:
            item = dict(entry or {})
            kind = item.pop("kind", None)
            if kind is None:
                kind = item.pop("name")
            postprocess.append(PostprocessSpec(kind=kind, parameters=_drop_empty(item)))
    params = {
        key: value
        for key, value in data.items()
        if key not in {"fields", "history", "visuals"}
    }
    return OutputSpec(
        directory=data.get("directory"),
        fields=fields,
        history=history,
        postprocess=postprocess,
        parameters=_drop_empty(params),
    )


def _schema_v2_spec(
    raw: dict[str, Any],
    *,
    source_path: str | None = None,
) -> ProblemSpec:
    assignments = [dict(entry or {}) for entry in raw.get("assignments") or []]
    geometry, mesh = _schema_v2_geometry(raw)
    regions = _schema_v2_regions(raw, assignments)
    initial_conditions = _schema_v2_initial_conditions(raw)
    bcs = _schema_v2_boundary_conditions(raw)
    steps = _schema_v2_analysis_steps(raw)
    outputs = _schema_v2_outputs(raw)
    return ProblemSpec(
        schema_version=2,
        name=raw.get("name") or "PhAST schema-v2 problem",
        reference=raw.get("reference", ""),
        geometry=geometry,
        mesh=mesh,
        regions=regions,
        materials=_schema_v2_materials(raw, assignments),
        initial_conditions=initial_conditions,
        boundary_conditions=bcs,
        analysis_steps=steps,
        solver=_schema_v2_solver(raw, steps),
        outputs=outputs,
        source="yaml:v2",
        source_path=source_path,
    )


_VALIDATION_FAMILY_MATERIALS = {
    "plasticity": "j2_plasticity",
    "ductile_phase_field": "ductile_phase_field",
    "diffuse_interface": "diffuse_interface",
    "cohesive_elements": "cohesive_interface",
}


def _validation_contract_run_spec(
    raw: dict[str, Any],
    run: dict[str, Any],
    *,
    source_path: str | None = None,
) -> ProblemSpec:
    run_id = str(run["id"])
    family = str(run.get("family", "validation"))
    script = str(run.get("script", ""))
    required_artifacts = list(run.get("required_artifacts") or [])
    required_per_case = list(run.get("required_artifacts_per_case") or [])
    one_of_per_case = list(run.get("one_of_artifacts_per_case") or [])
    output_contract = dict((raw.get("defaults") or {}).get("output_contract") or {})
    outputs_base = dict((raw.get("base_paths") or {}))
    return ProblemSpec(
        schema_version=int(raw.get("schema_version", 1)),
        name=run_id,
        reference=str(run.get("claim_boundary", "")),
        mesh=MeshSpec(
            kind="validation_artifacts",
            parameters=_drop_empty(
                {
                    "script": script,
                    "launcher": list(run.get("launcher") or []),
                    "working_dir": (raw.get("defaults") or {}).get("working_dir"),
                }
            ),
        ),
        materials=[
            MaterialSpec(
                name=family,
                model=_VALIDATION_FAMILY_MATERIALS.get(family, "validation_artifact"),
                parameters=_drop_empty(
                    {
                        "family": family,
                        "claim_boundary": run.get("claim_boundary"),
                    }
                ),
            )
        ],
        analysis_steps=[
            AnalysisStepSpec(
                name="validate",
                kind="validation_script",
                controls=_drop_empty(
                    {
                        "mode": (raw.get("defaults") or {}).get("mode"),
                        "script": script,
                        "launcher": list(run.get("launcher") or []),
                        "expected_cases": list(run.get("expected_cases") or []),
                    }
                ),
            )
        ],
        solver=SolverSpec(
            kind="validation_script",
            parameters=_drop_empty(
                {
                    "manifest_type": raw.get("manifest_type"),
                    "run_id": run_id,
                    "family": family,
                    "script": script,
                }
            ),
        ),
        outputs=OutputSpec(
            parameters=_drop_empty(
                {
                    "base_paths": outputs_base,
                    "required_artifacts": required_artifacts,
                    "required_artifacts_per_case": required_per_case,
                    "one_of_artifacts_per_case": one_of_per_case,
                    "output_contract": output_contract,
                }
            )
        ),
        source="yaml:validation-contract",
        source_path=source_path,
    )


def _validation_contract_specs(
    raw: dict[str, Any],
    *,
    source_path: str | None = None,
) -> list[ProblemSpec]:
    return [
        _validation_contract_run_spec(
            raw,
            dict(run or {}),
            source_path=source_path,
        )
        for run in raw.get("runs") or []
    ]


def problem_spec_to_schema_v2_dict(spec: ProblemSpec) -> dict[str, Any]:
    """Serialize a ProblemSpec to the additive schema-v2 YAML dictionary shape."""
    payload: dict[str, Any] = {
        "schema_version": 2,
        "name": spec.name,
    }
    if spec.reference:
        payload["reference"] = spec.reference
    if spec.mesh is not None:
        geometry: dict[str, Any]
        if spec.mesh.path is None:
            geometry = {"type": spec.mesh.kind}
        else:
            geometry = {"mesh_path": spec.mesh.path, "mesh_type": spec.mesh.kind}
        geometry.update(spec.mesh.parameters)
        payload["geometry"] = _drop_empty(geometry)
    elif spec.geometry is not None:
        geometry = {
            "type": spec.geometry.kind,
            "units": spec.geometry.units,
            "parameters": spec.geometry.parameters,
            "primitives": spec.geometry.primitives,
            "domain": spec.geometry.domain,
            "named_groups": spec.geometry.named_groups,
        }
        payload["geometry"] = _drop_empty(geometry)

    if spec.regions:
        payload["regions"] = [
            _drop_empty(
                {
                    "name": region.name,
                    "kind": region.kind,
                    "selector": region.selector,
                }
            )
            for region in spec.regions
        ]

    payload["materials"] = {
        material.name: _drop_empty(
            {
                "model": material.model,
                "parameters": material.parameters,
            }
        )
        for material in spec.materials
    }
    assignments = [
        {"material": material.name, "region": material.region}
        for material in spec.materials
        if material.region
    ]
    if assignments:
        payload["assignments"] = assignments

    if spec.initial_conditions:
        payload["initial_conditions"] = [
            _drop_empty(
                {
                    "field": item.field,
                    "region": item.region,
                    "value": item.value,
                    **item.parameters,
                }
            )
            for item in spec.initial_conditions
        ]

    if spec.boundary_conditions:
        payload["boundary_conditions"] = [
            _drop_empty(
                {
                    "name": bc.name,
                    "type": bc.kind,
                    "region": bc.region,
                    "component": bc.component,
                    "value": bc.value,
                    **bc.parameters,
                }
            )
            for bc in spec.boundary_conditions
        ]

    payload["analysis_steps"] = [
        _drop_empty(
            {
                "name": step.name,
                "type": step.kind,
                "controls": step.controls,
                "active_boundary_conditions": list(step.active_boundary_conditions),
            }
        )
        for step in spec.analysis_steps
    ]
    payload["solver"] = _drop_empty(
        {
            "type": spec.solver.kind,
            **spec.solver.parameters,
        }
    )

    outputs: dict[str, Any] = {}
    if spec.outputs.directory:
        outputs["directory"] = spec.outputs.directory
    if spec.outputs.fields:
        outputs["fields"] = [
            _drop_empty(
                {
                    "name": field.name,
                    "every": field.every,
                    **field.parameters,
                }
            )
            for field in spec.outputs.fields
        ]
    if spec.outputs.history:
        outputs["history"] = [
            _drop_empty(
                {
                    "name": history.name,
                    "every": history.every,
                    "region": history.region,
                    "component": history.component,
                    **history.parameters,
                }
            )
            for history in spec.outputs.history
        ]
    if spec.outputs.postprocess:
        outputs["visuals"] = [
            _drop_empty({"kind": item.kind, **item.parameters})
            for item in spec.outputs.postprocess
        ]
    outputs.update(spec.outputs.parameters)
    if outputs:
        payload["outputs"] = outputs
    return payload


def problem_specs_from_yaml(path: str | Path) -> list[ProblemSpec]:
    yaml_path = Path(path)
    raw = yaml.safe_load(yaml_path.read_text()) or {}
    source_path = str(yaml_path)
    if raw.get("manifest_type") == "reproducibility_contract":
        return _validation_contract_specs(raw, source_path=source_path)
    schema_version = raw.get("schema_version", 1)
    if schema_version in (2, "2"):
        return [_schema_v2_spec(raw, source_path=source_path)]
    if schema_version not in (1, "1"):
        raise ValueError(
            f"schema_version must be integer 1 or 2, got {schema_version!r}"
        )
    example_id = solid_example_id(raw)
    if example_id is not None:
        return [_solid_mechanics_spec(raw, example_id, source_path=source_path)]
    return [
        problem_spec_from_config(
            load_config(str(yaml_path)),
            source="yaml:v1",
            source_path=source_path,
        )
    ]


def problem_spec_from_yaml(path: str | Path) -> ProblemSpec:
    specs = problem_specs_from_yaml(path)
    if not specs:
        raise ValueError(f"No workflow problem specs found in {path}")
    return specs[0]


def problem_spec_from_problem(problem: Any) -> ProblemSpec:
    config = getattr(problem, "config", None)
    if config is None:
        raise TypeError("problem_spec_from_problem expects a phast.Problem-like object")
    return problem_spec_from_config(config, source="python:Problem")
