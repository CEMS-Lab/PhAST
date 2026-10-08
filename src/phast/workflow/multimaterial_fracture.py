"""Reusable schema-v2 runner for heterogeneous quasi-static fracture."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import h5py
import meshio
import numpy as np
from PIL import Image
import torch

from ..damage_solver import PhaseFieldDamageSolver
from ..fem_operators import FEMOperators
from ..material import Material
from ..mechanics_solver import QuasiStaticSolver
from ..mesh import FEMMesh
from ..io_utils import init_h5, write_h5_snapshot
from .specs import ProblemSpec


class MultimaterialWorkflowError(RuntimeError):
    """Raised when the heterogeneous workflow cannot continue safely."""


def _finite_scalar(value: Any, name: str, *, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MultimaterialWorkflowError(f"{name} must be a finite scalar") from exc
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(number):
        raise MultimaterialWorkflowError(f"{name} must be a finite scalar")
    if positive and number <= 0.0:
        raise MultimaterialWorkflowError(f"{name} must be positive")
    return number


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise MultimaterialWorkflowError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _structured_t3(length: float, height: float, nx: int, ny: int,
                   origin: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    length = _finite_scalar(length, "geometry.length", positive=True)
    height = _finite_scalar(height, "geometry.height", positive=True)
    nx, ny = _integer(nx, "geometry.nx", 2), _integer(ny, "geometry.ny", 2)
    if np.asarray(origin).shape != (2,) or not np.isfinite(origin).all():
        raise MultimaterialWorkflowError("geometry.origin must contain two finite coordinates")
    xs = np.linspace(origin[0], origin[0] + length, nx + 1)
    ys = np.linspace(origin[1], origin[1] + height, ny + 1)
    nodes = np.array([(x, y) for y in ys for x in xs], dtype=np.float64)
    elements = []
    stride = nx + 1
    for j in range(ny):
        for i in range(nx):
            n00 = j * stride + i
            n10, n01, n11 = n00 + 1, n00 + stride, n00 + stride + 1
            elements.extend(((n00, n10, n11), (n00, n11, n01)))
    return nodes, np.asarray(elements, dtype=np.int64)


def _build_mesh(spec: ProblemSpec) -> tuple[np.ndarray, np.ndarray]:
    if spec.mesh is not None and spec.mesh.path:
        loaded = meshio.read(spec.mesh.path)
        blocks = [block.data for block in loaded.cells if block.type == "triangle"]
        if not blocks:
            raise MultimaterialWorkflowError("external mesh contains no T3 cells")
        return np.asarray(loaded.points[:, :2], dtype=np.float64), np.vstack(blocks)
    if spec.geometry is None or spec.geometry.kind not in {
        "structured_rectangle", "structured_grid", "rectangle"
    }:
        raise MultimaterialWorkflowError(
            "use structured_rectangle, structured_grid, rectangle, or an external T3 mesh")
    p = spec.geometry.parameters
    length = float(p.get("length", p.get("width", 1.0)))
    height = float(p.get("height", 1.0))
    origin = p.get("origin", [0.0, -0.5 * height])
    return _structured_t3(length, height, p.get("nx", 40),
                          p.get("ny", 20),
                          (float(origin[0]), float(origin[1])))


def _segment_distance(points: np.ndarray, start: np.ndarray,
                      end: np.ndarray) -> np.ndarray:
    tangent = end - start
    norm_sq = float(np.dot(tangent, tangent))
    if norm_sq <= 1.0e-30:
        return np.linalg.norm(points - start, axis=1)
    t = np.clip(np.einsum("ij,j->i", points - start, tangent) / norm_sq, 0.0, 1.0)
    return np.linalg.norm(points - (start + t[:, None] * tangent), axis=1)


def _selector_mask(points: np.ndarray, selector: dict[str, Any], *,
                   nodal_bounds: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    kind = str(selector.get("type", "all")).lower()
    tol = _finite_scalar(selector.get("tolerance", 1.0e-9), "selector.tolerance")
    if tol < 0.0:
        raise MultimaterialWorkflowError("selector.tolerance must be nonnegative")
    if kind == "all":
        return np.ones(points.shape[0], dtype=bool)
    if kind in {"all_of", "any_of"}:
        children = selector.get("selectors")
        if not isinstance(children, list) or not children:
            raise MultimaterialWorkflowError(
                f"{kind} selector requires a non-empty 'selectors' list")
        masks = [_selector_mask(points, child, nodal_bounds=nodal_bounds) for child in children]
        if kind == "all_of":
            return np.logical_and.reduce(masks)
        return np.logical_or.reduce(masks)
    if kind == "not":
        child = selector.get("selector")
        if not isinstance(child, dict):
            raise MultimaterialWorkflowError("not selector requires a 'selector' mapping")
        return ~_selector_mask(points, child, nodal_bounds=nodal_bounds)
    if kind in {"circle", "outside_circle"}:
        centre = np.asarray(selector["center"], dtype=np.float64)
        inside = np.linalg.norm(points - centre, axis=1) <= float(selector["radius"]) + tol
        return ~inside if kind == "outside_circle" else inside
    if kind == "rectangle":
        lo = np.asarray(selector["origin"], dtype=np.float64)
        hi = lo + np.asarray(selector["size"], dtype=np.float64)
        return np.all(points >= lo - tol, axis=1) & np.all(points <= hi + tol, axis=1)
    if kind == "line_segment":
        return _segment_distance(
            points, np.asarray(selector["from"], dtype=np.float64),
            np.asarray(selector["to"], dtype=np.float64)) <= float(selector.get("thickness", tol))
    if kind in {"boundary", "boundary_segment"}:
        if "side" in selector:
            sides = {"left": (0, 0), "right": (0, 1), "bottom": (1, 0), "top": (1, 1)}
            side = str(selector["side"]).lower()
            if side not in sides or "axis" in selector or "value" in selector:
                raise MultimaterialWorkflowError(
                    "boundary side must be left/right/top/bottom, without axis or value")
            if nodal_bounds is None:
                raise MultimaterialWorkflowError(
                    "boundary side requires full nodal bounds from a structured rectangle")
            axis, end = sides[side]
            mask = np.isclose(points[:, axis], nodal_bounds[end][axis], atol=tol, rtol=0.0)
        else:
            axis_name = str(selector["axis"]).lower()
            if axis_name not in {"x", "y"}:
                raise MultimaterialWorkflowError("boundary selector axis must be x or y")
            axis = 0 if axis_name == "x" else 1
            mask = np.isclose(points[:, axis], float(selector["value"]), atol=tol)
        for name, limits in (selector.get("bounds") or {}).items():
            bound_axis = 0 if str(name).lower() == "x" else 1
            mask &= points[:, bound_axis] >= float(limits[0]) - tol
            mask &= points[:, bound_axis] <= float(limits[1]) + tol
        return mask
    raise MultimaterialWorkflowError(f"unsupported region selector {kind!r}")


def _resolve_regions(spec: ProblemSpec, nodes: np.ndarray,
                     elements: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    centroids = nodes[elements].mean(axis=1)
    bounds = None if spec.mesh is not None and spec.mesh.path else (nodes.min(axis=0), nodes.max(axis=0))
    return (
        {region.name: _selector_mask(nodes, region.selector, nodal_bounds=bounds) for region in spec.regions},
        {region.name: _selector_mask(centroids, region.selector, nodal_bounds=bounds) for region in spec.regions},
    )


def _shared(spec: ProblemSpec, key: str, default: Any = None) -> Any:
    values = [material.parameters.get(key, default) for material in spec.materials]
    if any(value != values[0] for value in values[1:]):
        raise MultimaterialWorkflowError(
            f"all materials must share {key!r}; received {values}")
    return values[0]


def _assign_materials(spec: ProblemSpec, regions: dict[str, np.ndarray],
                      n_elements: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    E = np.full(n_elements, np.nan)
    Gc = np.full(n_elements, np.nan)
    labels = np.full(n_elements, -1, dtype=np.int64)
    names = []
    for index, material in enumerate(spec.materials):
        if material.region not in regions:
            raise MultimaterialWorkflowError(
                f"material {material.name!r} references unknown region {material.region!r}")
        mask = regions[material.region]
        if not bool(mask.any()):
            raise MultimaterialWorkflowError(f"region {material.region!r} selects no elements")
        if bool((mask & (labels >= 0)).any()):
            raise MultimaterialWorkflowError(
                f"material region {material.region!r} overlaps another assignment")
        E[mask] = _finite_scalar(material.parameters.get("E"), f"{material.name}.E", positive=True)
        Gc[mask] = _finite_scalar(material.parameters.get("Gc"), f"{material.name}.Gc", positive=True)
        labels[mask] = index
        names.append(material.name)
    if bool((labels < 0).any()):
        raise MultimaterialWorkflowError(
            f"material assignments leave {int((labels < 0).sum())} elements unassigned")
    return E, Gc, labels, names


def _validate_materials(spec: ProblemSpec) -> None:
    defaults = {"nu": 0.3, "rho": 1.0, "eta_residual": 1.0e-6,
                "energy_split": "amor", "pf_model": "AT2", "plane_stress": True}
    allowed = {"E", "Gc", "l0", *defaults}
    if not spec.materials:
        raise MultimaterialWorkflowError("at least one material is required")
    for material in spec.materials:
        p = material.parameters
        if material.model != "phase_field" or set(p) - allowed:
            raise MultimaterialWorkflowError(
                f"unsupported material model or parameters for {material.name!r}: {sorted(set(p) - allowed)}")
        for key in ("E", "Gc", "l0", "rho"):
            _finite_scalar(p.get(key, defaults.get(key)), f"{material.name}.{key}", positive=True)
        nu = _finite_scalar(p.get("nu", 0.3), f"{material.name}.nu")
        eta = _finite_scalar(p.get("eta_residual", 1.0e-6), f"{material.name}.eta_residual")
        if not -1.0 < nu < 0.5:
            raise MultimaterialWorkflowError("nu must satisfy -1 < nu < 0.5 for the Amor split")
        if not 0.0 <= eta < 1.0:
            raise MultimaterialWorkflowError("eta_residual must satisfy 0 <= eta_residual < 1")
        if not isinstance(p.get("plane_stress", True), bool):
            raise MultimaterialWorkflowError("plane_stress must be a boolean")
    for key, default in {"l0": None, **defaults}.items():
        _shared(spec, key, default)
    if _shared(spec, "energy_split", "amor") != "amor" or _shared(spec, "pf_model", "AT2") != "AT2":
        raise MultimaterialWorkflowError("current route requires energy_split: amor and pf_model: AT2")


def _resolve_reaction(spec: ProblemSpec, regions: dict[str, np.ndarray],
                      resolved_bcs: list) -> tuple[torch.Tensor | None, int]:
    """Resolve the requested reaction; the 2D integral is per unit thickness."""
    requests = [h for h in spec.outputs.history if h.name in {"reaction", "reaction_force", "load_displacement"}]
    if len(requests) > 1:
        raise MultimaterialWorkflowError("only one reaction/load_displacement history is supported")
    default = next((item for item in resolved_bcs if item[0].kind == "prescribe"), None)
    request = requests[0] if requests else None
    indices = default[1] if default else None
    component = int(default[0].component or 0) if default else 0
    if request is not None:
        if request.region is not None:
            mask = regions.get(request.region)
            if mask is None or not mask.any():
                raise MultimaterialWorkflowError(f"reaction region {request.region!r} selects no nodes")
            indices = torch.as_tensor(np.flatnonzero(mask), dtype=torch.long)
        if request.component is not None:
            if isinstance(request.component, bool) or request.component not in (0, 1):
                raise MultimaterialWorkflowError("reaction component must be 0 or 1")
            component = int(request.component)
        if indices is None:
            raise MultimaterialWorkflowError("reaction output requires a region or a prescribed boundary")
    return indices, component


def _validate_outputs(spec: ProblemSpec) -> None:
    if set(spec.outputs.parameters) - {"directory"}:
        raise MultimaterialWorkflowError("unsupported output parameters")
    fields = {"trajectory", "displacement", "strain", "stress", "damage", "vtu"}
    histories = {"reaction", "reaction_force", "load_displacement", "max_damage", "solver_telemetry", "timing_per_step"}
    for items, names, label in ((spec.outputs.fields, fields, "field"), (spec.outputs.history, histories, "history")):
        seen = set()
        for item in items:
            if item.name not in names or item.name in seen:
                raise MultimaterialWorkflowError(f"unsupported or duplicate {label} output {item.name!r}")
            seen.add(item.name)
            _integer(item.every, f"{label} output every")
            allowed = {"format"} if label == "field" and item.name == "trajectory" else set()
            if set(item.parameters) - allowed:
                raise MultimaterialWorkflowError(f"unsupported parameters for {label} output {item.name!r}")
            if allowed:
                if item.parameters.get("format", "h5") != "h5":
                    raise MultimaterialWorkflowError("the heterogeneous runner supports HDF5 trajectories only")
            elif item.every != 1:
                raise MultimaterialWorkflowError(f"{item.name!r} output does not support custom cadence")
            if label == "history" and item.name not in {"reaction", "reaction_force", "load_displacement"}:
                if item.region is not None or item.component is not None:
                    raise MultimaterialWorkflowError(f"{item.name!r} history does not support region/component selection")
    for item in spec.outputs.postprocess:
        if item.kind not in {"plots", "initial_conditions", "animation"}:
            raise MultimaterialWorkflowError(f"unsupported visual request {item.kind!r}")
        if item.kind == "animation":
            if set(item.parameters) - {"field"} or item.parameters.get("field", "damage") != "damage":
                raise MultimaterialWorkflowError("only the default damage animation is supported")
        elif item.parameters:
            raise MultimaterialWorkflowError(f"unsupported parameters for visual {item.kind!r}")


def _contains_boundary(selector: dict[str, Any]) -> bool:
    kind = str(selector.get("type", "all")).lower()
    return kind in {"boundary", "boundary_segment"} or (
        kind in {"all_of", "any_of"}
        and any(_contains_boundary(child) for child in selector.get("selectors", [])))


def validate_multimaterial_fracture_spec(spec: ProblemSpec) -> None:
    """Preflight material, request and mesh-region inputs without solver construction.

    This builds/reads the mesh and resolves selectors, but creates no output
    directory, FEM operators or solvers. Schema-v2 planning/validate-only may
    call it directly. Errors are reported as MultimaterialWorkflowError.
    """
    if (spec.solver.kind != "quasi_static" or len(spec.analysis_steps) != 1
            or spec.analysis_steps[0].kind != "quasi_static"):
        raise MultimaterialWorkflowError("this route requires one quasi_static analysis step")
    _validate_materials(spec)
    _validate_outputs(spec)
    controls = spec.analysis_steps[0].controls
    if set(controls) - {"number_of_steps", "num_steps"}:
        raise MultimaterialWorkflowError("unsupported quasi-static analysis controls")
    _integer(controls.get("number_of_steps", controls.get("num_steps", 41)), "number_of_steps", 2)
    try:
        nodes, elements = _build_mesh(spec)
        if not np.isfinite(nodes).all():
            raise MultimaterialWorkflowError("mesh coordinates must be finite")
        node_regions, element_regions = _resolve_regions(spec, nodes, elements)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise MultimaterialWorkflowError(f"invalid geometry or region selector: {exc}") from exc
    _assign_materials(spec, element_regions, len(elements))
    positive_seed = False
    fixed = np.zeros(len(nodes), dtype=bool)
    values = np.zeros(len(nodes))
    for initial in spec.initial_conditions:
        if initial.field != "damage" or not initial.region or initial.parameters:
            raise MultimaterialWorkflowError("only scalar region-based damage initial conditions are supported")
        mask = node_regions.get(initial.region)
        if mask is None or not mask.any():
            raise MultimaterialWorkflowError(f"initial damage region {initial.region!r} selects no nodes")
        value = _finite_scalar(1.0 if initial.value is None else initial.value, "initial damage")
        if not 0.0 <= value <= 1.0:
            raise MultimaterialWorkflowError("initial damage must lie in [0, 1]")
        if np.any(fixed & mask & (values != value)):
            raise MultimaterialWorkflowError("conflicting initial damage constraints")
        fixed |= mask
        values[mask] = value
        positive_seed |= value > 0.0
    if not positive_seed:
        raise MultimaterialWorkflowError("a nonempty positive seeded damage region is required")
    active = set(spec.analysis_steps[0].active_boundary_conditions)
    if active - {bc.name for bc in spec.boundary_conditions}:
        raise MultimaterialWorkflowError("active boundary condition name is not declared")
    resolved = _resolve_bcs(spec, node_regions)
    selectors = {region.name: region.selector for region in spec.regions}
    structured = not (spec.mesh is not None and spec.mesh.path)
    exterior = np.any(np.isclose(nodes, nodes.min(axis=0), atol=1.0e-9, rtol=0.0)
                      | np.isclose(nodes, nodes.max(axis=0), atol=1.0e-9, rtol=0.0), axis=1)
    occupied = np.zeros((len(nodes), 2), dtype=bool)
    prescribed_values = np.zeros((len(nodes), 2))
    for bc, indices in resolved:
        if bc.kind not in {"fix", "prescribe"} or bc.parameters:
            raise MultimaterialWorkflowError("only parameter-free fix/prescribe boundaries are supported")
        if bc.component is not None and (isinstance(bc.component, bool) or bc.component not in (0, 1)):
            raise MultimaterialWorkflowError("boundary component must be 0 or 1")
        ids = indices.numpy()
        if structured and _contains_boundary(selectors[bc.region]) and not exterior[ids].all():
            raise MultimaterialWorkflowError(f"boundary region {bc.region!r} selects interior nodes; update geometry-dependent selectors")
        value = 0.0 if bc.kind == "fix" else _finite_scalar(bc.value, "prescribed displacement")
        for component in ((0, 1) if bc.component is None else (bc.component,)):
            if np.any(occupied[ids, component] & (prescribed_values[ids, component] != value)):
                raise MultimaterialWorkflowError("conflicting displacement boundary conditions")
            occupied[ids, component] = True
            prescribed_values[ids, component] = value
    _resolve_reaction(spec, node_regions, resolved)


def _field_residual(solver: PhaseFieldDamageSolver, history: torch.Tensor,
                    damage: torch.Tensor, Gc_field: torch.Tensor) -> torch.Tensor:
    attrs = ("_Gc_l0_e", "_Gc_over_l0_e", "_at1_source_e", "_cg_Gc_l0_e_diag_lap")
    original = {name: getattr(solver, name, None) for name in attrs}
    try:
        Gc_e = Gc_field.to(dtype=solver._cg_dtype, device=solver._cg_device)
        solver._Gc_l0_e = Gc_e * solver._l0
        solver._Gc_over_l0_e = Gc_e / solver._l0
        solver._at1_source_e = torch.zeros_like(Gc_e)
        solver._cg_Gc_l0_e_diag_lap = solver._Gc_l0_e.unsqueeze(1) * solver._cg_diag_lap
        return solver.compute_residual(history, damage)
    finally:
        for name, value in original.items():
            setattr(solver, name, value)


def _projected_norm(residual: torch.Tensor, damage: torch.Tensor,
                    lower: torch.Tensor, fixed: torch.Tensor) -> float:
    projected = residual.clone()
    at_lower = damage <= lower + 1.0e-9
    at_upper = damage >= 1.0 - 1.0e-9
    projected[at_lower] = torch.minimum(projected[at_lower], torch.zeros_like(projected[at_lower]))
    projected[at_upper] = torch.maximum(projected[at_upper], torch.zeros_like(projected[at_upper]))
    projected[fixed] = 0.0
    return float(torch.linalg.vector_norm(projected, ord=float("inf")).item())


def _connected_front(nodes: np.ndarray, elements: np.ndarray, damage: np.ndarray,
                     seeds: np.ndarray, threshold: float) -> float:
    active = damage >= threshold
    starts = [int(node) for node in seeds if active[int(node)]]
    if not starts:
        return float(nodes[seeds, 0].max())
    adjacency = [set() for _ in range(nodes.shape[0])]
    for element in elements:
        for i, node in enumerate(element):
            for neighbour in element[i + 1:]:
                adjacency[int(node)].add(int(neighbour))
                adjacency[int(neighbour)].add(int(node))
    visited = set(starts)
    queue = deque(starts)
    while queue:
        node = queue.popleft()
        for neighbour in adjacency[node]:
            if active[neighbour] and neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    return float(nodes[np.asarray(sorted(visited)), 0].max())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _material_edges(nodes: np.ndarray, elements: np.ndarray,
                    material_ids: np.ndarray) -> np.ndarray:
    """Return mesh edges separating different bulk material assignments."""
    owner: dict[tuple[int, int], int] = {}
    edges = []
    for element, material_id in zip(elements, material_ids):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((int(element[i]), int(element[j]))))
            if edge in owner and owner[edge] != int(material_id):
                edges.append(nodes[list(edge)])
            else:
                owner[edge] = int(material_id)
    return np.asarray(edges, dtype=np.float64).reshape(-1, 2, 2)


def _overlay_materials(ax, interfaces: np.ndarray | None) -> None:
    if interfaces is not None and interfaces.size:
        ax.add_collection(LineCollection(interfaces, colors="0.35", linewidths=0.7))


def _save_h5_state(path: Path, step: int, mesh: FEMMesh, fem: FEMOperators,
                   displacement: torch.Tensor, damage: torch.Tensor,
                   history: torch.Tensor, *, load_factor: float,
                   opening: float, reaction: float,
                   diagnostics: dict[str, Any] | None = None) -> None:
    """Store an accepted state using the shared PhAST HDF5 field schema."""
    strain = fem.compute_strain(displacement)
    stress = fem.compute_stress(displacement, damage, strain=strain)
    psi_plus = fem.compute_psi_plus(displacement, strain=strain)
    with h5py.File(path, "a") as store:
        write_h5_snapshot(
            store, step, mesh, displacement, damage, psi_plus, history,
            eps_xx=strain[0], eps_yy=strain[1], gam_xy=strain[2],
            sxx=stress[0], syy=stress[1], sxy=stress[2],
            reaction_force=reaction, applied_disp=opening, precision="float64",
        )
        group = store[f"simulation_data/steps/step_{step:04d}"]
        group.attrs["load_factor"] = float(load_factor)
        if diagnostics is not None:
            for name in ("projected_damage_residual", "mechanics_residual",
                         "damage_increment", "connected_crack_front_x"):
                group.attrs[name] = float(diagnostics[name])
        store.attrs["num_steps"] = int(step)
        store.attrs["num_snapshots"] = len(store["simulation_data/steps"])


def _plot_field(path: Path, nodes: np.ndarray, elements: np.ndarray,
                values: np.ndarray, title: str, label: str, nodal: bool,
                vmin: float | None = None, vmax: float | None = None,
                *, cmap: str = "viridis",
                interfaces: np.ndarray | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 3.5), constrained_layout=True)
    if nodal:
        image = ax.tripcolor(nodes[:, 0], nodes[:, 1], elements, values,
                             shading="gouraud", vmin=vmin, vmax=vmax, cmap=cmap)
    else:
        image = ax.tripcolor(nodes[:, 0], nodes[:, 1], elements,
                             facecolors=values, shading="flat", vmin=vmin, vmax=vmax,
                             cmap=cmap)
    _overlay_materials(ax, interfaces)
    ax.set(aspect="equal", xlabel="x", ylabel="y", title=title)
    fig.colorbar(image, ax=ax, shrink=0.82, label=label)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _frame(nodes: np.ndarray, elements: np.ndarray, damage: np.ndarray,
           factor: float, interfaces: np.ndarray | None = None) -> Image.Image:
    fig, ax = plt.subplots(figsize=(8.0, 3.5), constrained_layout=True)
    image = ax.tripcolor(nodes[:, 0], nodes[:, 1], elements, damage,
                         shading="gouraud", cmap="Reds", vmin=0.0, vmax=1.0)
    _overlay_materials(ax, interfaces)
    ax.set(aspect="equal", xlabel="x", ylabel="y",
           title=f"Phase-field damage, load factor = {factor:.4f}")
    fig.colorbar(image, ax=ax, shrink=0.82, label="damage d")
    fig.canvas.draw()
    result = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert("RGB")
    plt.close(fig)
    return result


def _resolve_bcs(spec: ProblemSpec, regions: dict[str, np.ndarray]):
    active = set(spec.analysis_steps[0].active_boundary_conditions)
    resolved = []
    for bc in spec.boundary_conditions:
        if active and bc.name not in active:
            continue
        if bc.region not in regions:
            raise MultimaterialWorkflowError(f"unknown boundary region {bc.region!r}")
        indices = np.flatnonzero(regions[bc.region])
        if not indices.size:
            raise MultimaterialWorkflowError(f"boundary region {bc.region!r} selects no nodes")
        resolved.append((bc, torch.as_tensor(indices, dtype=torch.long)))
    return resolved


def _bc_state(n_nodes: int, resolved, factor: float):
    mask = torch.zeros((n_nodes, 2), dtype=torch.bool)
    values = torch.zeros((n_nodes, 2), dtype=torch.float64)
    for bc, indices in resolved:
        components = (0, 1) if bc.component is None else (int(bc.component),)
        for component in components:
            if bc.kind == "fix":
                mask[indices, component] = True
            elif bc.kind == "prescribe":
                mask[indices, component] = True
                values[indices, component] = factor * float(bc.value)
            else:
                raise MultimaterialWorkflowError(
                    f"supported boundary conditions are fix and prescribe, got {bc.kind!r}")
    return mask, values


def run_multimaterial_fracture_spec(spec: ProblemSpec, *, output_dir=None) -> int:
    """Execute a reusable heterogeneous schema-v2 quasi-static problem."""
    started = time.perf_counter()
    validate_multimaterial_fracture_spec(spec)
    if spec.solver.kind != "quasi_static" or len(spec.analysis_steps) != 1:
        raise MultimaterialWorkflowError("this route requires one quasi_static analysis step")
    if any(material.model != "phase_field" for material in spec.materials):
        raise MultimaterialWorkflowError("all materials must use model: phase_field")
    output = Path(output_dir or spec.outputs.directory or "runs/multimaterial_fracture")
    output.mkdir(parents=True, exist_ok=True)
    nodes_np, elements_np = _build_mesh(spec)
    node_regions, element_regions = _resolve_regions(spec, nodes_np, elements_np)
    E_np, Gc_np, material_ids, material_names = _assign_materials(
        spec, element_regions, elements_np.shape[0])
    interfaces = _material_edges(nodes_np, elements_np, material_ids)
    edge_lengths = np.linalg.norm(
        nodes_np[elements_np[:, [1, 2, 0]]] - nodes_np[elements_np[:, [0, 1, 2]]],
        axis=2)
    maximum_edge_length = float(edge_lengths.max())
    nu, l0 = float(_shared(spec, "nu", 0.3)), float(_shared(spec, "l0"))
    energy_split = str(_shared(spec, "energy_split", "amor"))
    plane_stress = bool(_shared(spec, "plane_stress", True))
    pf_model = str(_shared(spec, "pf_model", "AT2"))
    if energy_split != "amor" or pf_model != "AT2":
        raise MultimaterialWorkflowError("current route requires energy_split: amor and pf_model: AT2")

    nodes = torch.as_tensor(nodes_np, dtype=torch.float64)
    elements = torch.as_tensor(elements_np, dtype=torch.long)
    mesh = FEMMesh.from_tensors(nodes, elements, device="cpu", dtype=torch.float64)
    reference = spec.materials[0].parameters
    material = Material(
        E=float(reference["E"]), nu=nu, Gc=float(reference["Gc"]), l0=l0,
        rho=float(reference.get("rho", 1.0)),
        eta_residual=float(reference.get("eta_residual", 1.0e-6)),
        energy_split=energy_split, pf_model=pf_model, plane_stress=plane_stress)
    fem = FEMOperators(mesh, material)
    fem.diff_E_field = torch.as_tensor(E_np, dtype=torch.float64)
    settings = spec.solver.parameters
    mechanics = QuasiStaticSolver(
        fem, backend=str(settings.get("mechanics_backend", "auto")),
        tol=float(settings.get("mechanics_tolerance", 1.0e-9)),
        tol_rel=float(settings.get("mechanics_relative_tolerance", 1.0e-8)),
        max_iter=int(settings.get("mechanics_max_iterations", 30)), line_search=True)
    damage_solver = PhaseFieldDamageSolver(
        fem, tol=float(settings.get("damage_tolerance", 1.0e-8)),
        max_iter=int(settings.get("damage_max_iterations", 500)),
        bounds_method=str(settings.get("damage_bounds_method", "projected_cg")),
        use_multigrid=False)
    Gc_field = torch.as_tensor(Gc_np, dtype=torch.float64)
    resolved_bcs = _resolve_bcs(spec, node_regions)
    reaction_indices, reaction_component = _resolve_reaction(spec, node_regions, resolved_bcs)

    fixed = torch.zeros(mesh.n_nodes, dtype=torch.bool)
    fixed_values = torch.zeros(mesh.n_nodes, dtype=torch.float64)
    for initial in spec.initial_conditions:
        if initial.field == "damage" and initial.region:
            mask = torch.as_tensor(node_regions[initial.region], dtype=torch.bool)
            fixed |= mask
            fixed_values[mask] = float(1.0 if initial.value is None else initial.value)
    seeds = np.flatnonzero(fixed.numpy())
    if not seeds.size:
        raise MultimaterialWorkflowError("a seeded damage region is required")
    damage, displacement = fixed_values.clone(), torch.zeros((mesh.n_nodes, 2), dtype=torch.float64)
    history = torch.zeros(mesh.n_elems, dtype=torch.float64)
    external_force = torch.zeros_like(displacement)

    trajectory_request = next(
        (field for field in spec.outputs.fields if field.name == "trajectory"), None)
    trajectory_path = None
    if trajectory_request is not None:
        trajectory_format = str(trajectory_request.parameters.get("format", "h5"))
        if trajectory_format != "h5":
            raise MultimaterialWorkflowError(
                "the heterogeneous runner supports HDF5 trajectories; "
                "set the trajectory field format to h5 or omit format")
        trajectory_path = output / "training_data.h5"
        if trajectory_path.exists():
            raise MultimaterialWorkflowError(
                f"refusing to overwrite {trajectory_path}; use a new output directory")
        with init_h5(str(trajectory_path), mesh, material) as store:
            store.attrs["format"] = "phast.trajectory.h5"
            store.attrs["analysis"] = "quasi_static"
            store.attrs["material_names"] = json.dumps(material_names)
            group = store["simulation_data"].create_group("material_fields")
            for name, values in (("material_id", material_ids), ("E", E_np), ("Gc", Gc_np)):
                group.create_dataset(name, data=values)
            store["simulation_data/metadata"].attrs["material_fields_path"] = (
                "simulation_data/material_fields")
        _save_h5_state(
            trajectory_path, 0, mesh, fem, displacement, damage, history,
            load_factor=0.0, opening=0.0, reaction=0.0)

    controls = spec.analysis_steps[0].controls
    n_steps = int(controls.get("number_of_steps", controls.get("num_steps", 41)))
    max_stagger = int(settings.get("maximum_staggered_iterations", 100))
    increment_tol = float(settings.get("staggered_damage_tolerance", 1.0e-5))
    residual_tol = float(settings.get("projected_residual_tolerance", 1.0e-5))
    relaxation = float(settings.get("damage_relaxation", 0.7))
    nominal = 1.0 / (n_steps - 1)
    min_increment = float(settings.get("minimum_load_increment", nominal / 16.0))
    max_cutbacks = int(settings.get("maximum_cutbacks", 20))
    threshold = float(settings.get("crack_front_threshold", 0.8))
    targets = deque(float(value) for value in np.linspace(0.0, 1.0, n_steps)[1:])
    accepted_factor, cutbacks, attempt = 0.0, 0, 0
    rows, frames = [], [_frame(nodes_np, elements_np, damage.numpy(), 0.0, interfaces)]

    while targets:
        attempt += 1
        target = targets.popleft()
        base_u, base_d, base_H = displacement.clone(), damage.clone(), history.clone()
        bc_mask, bc_values = _bc_state(mesh.n_nodes, resolved_bcs, target)
        trial_u, trial_d, trial_H = base_u.clone(), base_d.clone(), base_H.clone()
        converged, failure = False, ""
        step_started = time.perf_counter()
        projected_residual = damage_increment = float("inf")
        mechanics_iterations = 0
        for stagger_iteration in range(1, max_stagger + 1):
            trial_u, mechanics_ok, mechanics_iterations = mechanics.solve(
                trial_d, external_force, bc_mask, bc_values, u_init=trial_u)
            if not mechanics_ok:
                failure = f"mechanics residual {mechanics.last_residual}"
                break
            strain = fem.compute_strain(trial_u)
            trial_H = torch.maximum(base_H, fem.compute_psi_plus(trial_u, strain=strain))
            candidate = damage_solver.solve(
                trial_H, base_d, Gc_field=Gc_field,
                pf_dirichlet_mask=fixed, pf_dirichlet_values=fixed_values,
                initial_guess=trial_d)
            if getattr(damage_solver, "last_converged", True) is False:
                failure = "damage subproblem did not converge"
                break
            candidate[fixed] = fixed_values[fixed]
            next_damage = torch.clamp(
                torch.maximum(trial_d + relaxation * (candidate - trial_d), base_d), 0.0, 1.0)
            next_damage[fixed] = fixed_values[fixed]
            damage_increment = float(torch.max(torch.abs(next_damage - trial_d)).item())
            trial_d = next_damage
            projected_residual = _projected_norm(
                _field_residual(damage_solver, trial_H, trial_d, Gc_field),
                trial_d, base_d, fixed)
            if damage_increment <= increment_tol and projected_residual <= residual_tol:
                converged = True
                break
        if converged:
            trial_u, mechanics_ok, mechanics_iterations = mechanics.solve(
                trial_d, external_force, bc_mask, bc_values, u_init=trial_u)
            if not mechanics_ok:
                converged, failure = False, "final mechanics solve failed"
        if not converged:
            span = target - accepted_factor
            if bool(settings.get("automatic_cutback", True)) and span > min_increment and cutbacks < max_cutbacks:
                midpoint = accepted_factor + 0.5 * span
                targets.appendleft(target)
                targets.appendleft(midpoint)
                cutbacks += 1
                print(f"cutback {cutbacks}: factor {target:.6f} failed; retrying {midpoint:.6f}", flush=True)
                continue
            raise MultimaterialWorkflowError(
                f"staggered solve failed at factor {target:.6f}: "
                f"{failure or f'increment={damage_increment:.3e}, residual={projected_residual:.3e}'}")

        displacement, damage, history, accepted_factor = trial_u, trial_d, trial_H, target
        internal = fem.internal_force(displacement, damage)
        reaction_bc = next((item for item in resolved_bcs if item[0].kind == "prescribe"), None)
        reaction = (float(internal[reaction_indices, reaction_component].sum().item())
                    if reaction_indices is not None else 0.0)
        opening = 0.0
        if reaction_bc:
            bc, indices = reaction_bc
            component = int(bc.component or 0)
            prescribed = [
                target * float(item.value)
                for item, _ in resolved_bcs
                if item.kind == "prescribe" and int(item.component or 0) == component
            ]
            if prescribed:
                opening = max(prescribed) - min(prescribed)
                if opening == 0.0:
                    opening = max(abs(value) for value in prescribed)
        damage_np = damage.detach().cpu().numpy()
        front = _connected_front(nodes_np, elements_np, damage_np, seeds, threshold)
        row = {
            "accepted_step": len(rows) + 1, "attempt": attempt,
            "load_factor": target, "opening": opening, "reaction_force": reaction,
            "maximum_damage": float(damage.max().item()),
            "connected_crack_front_x": front,
            "mechanics_iterations": int(mechanics_iterations),
            "mechanics_residual": float(mechanics.last_residual),
            "damage_iterations": int(getattr(damage_solver, "last_iter", 0) or 0),
            "staggered_iterations": int(stagger_iteration),
            "damage_increment": damage_increment,
            "projected_damage_residual": projected_residual,
            "cutbacks_so_far": cutbacks,
            "wall_seconds": time.perf_counter() - step_started,
        }
        for index, name in enumerate(material_names):
            mask = material_ids == index
            row[f"mean_damage_{name}"] = float(damage[elements[mask]].mean().item())
        rows.append(row)
        if trajectory_path is not None and (
                len(rows) % trajectory_request.every == 0 or not targets):
            _save_h5_state(
                trajectory_path, len(rows), mesh, fem, displacement, damage, history,
                load_factor=target, opening=opening, reaction=reaction, diagnostics=row)
        frames.append(_frame(nodes_np, elements_np, damage_np, target, interfaces))
        print(f"accepted {len(rows):03d}: factor={target:.6f} opening={opening:.6e} "
              f"reaction={reaction:.6e} front_x={front:.6f} residual={projected_residual:.3e}", flush=True)

    strain = fem.compute_strain(displacement)
    stress = fem.compute_stress(displacement, damage, strain=strain)
    eps_xx, eps_yy, gamma_xy = [value.detach().cpu().numpy() for value in strain]
    sxx, syy, sxy = [value.detach().cpu().numpy() for value in stress]
    eq_strain = np.sqrt(eps_xx**2 + eps_yy**2 + 0.5 * gamma_xy**2)
    von_mises = np.sqrt(np.maximum(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2, 0.0))
    u_np, d_np = displacement.detach().cpu().numpy(), damage.detach().cpu().numpy()
    u_mag = np.linalg.norm(u_np, axis=1)
    for name in ("history.csv", "solver_telemetry.csv", "timing_per_step.csv"):
        _write_csv(output / name, rows)
    with (output / "nodal_fields.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["node", "x", "y", "ux", "uy", "displacement_magnitude", "damage"])
        for i, (xy, u, um, d) in enumerate(zip(nodes_np, u_np, u_mag, d_np)):
            writer.writerow([i, xy[0], xy[1], u[0], u[1], um, d])
    with (output / "element_fields.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["element", "material", "E", "Gc", "eps_xx", "eps_yy",
                         "gamma_xy", "equivalent_strain", "sxx", "syy", "sxy", "von_mises"])
        for i in range(elements_np.shape[0]):
            writer.writerow([i, material_names[int(material_ids[i])], E_np[i], Gc_np[i],
                             eps_xx[i], eps_yy[i], gamma_xy[i], eq_strain[i],
                             sxx[i], syy[i], sxy[i], von_mises[i]])

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), constrained_layout=True)
    for ax, values, title, label in ((axes[0], E_np, "Young's modulus", "E"),
                                      (axes[1], Gc_np, "Fracture toughness", "Gc")):
        image = ax.tripcolor(nodes_np[:, 0], nodes_np[:, 1], elements_np,
                             facecolors=values, shading="flat")
        ax.set(aspect="equal", xlabel="x", ylabel="y", title=title)
        fig.colorbar(image, ax=ax, shrink=0.82, label=label)
    fig.savefig(output / "material_fields.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.0, 3.5), constrained_layout=True)
    ax.tripcolor(nodes_np[:, 0], nodes_np[:, 1], elements_np,
                 facecolors=material_ids, shading="flat", cmap="Pastel1")
    ax.scatter(nodes_np[seeds, 0], nodes_np[seeds, 1], s=9, color="black", label="initial crack")
    for bc, indices in resolved_bcs:
        ax.scatter(nodes_np[indices.numpy(), 0], nodes_np[indices.numpy(), 1],
                   s=11, label=bc.name or bc.kind)
    ax.set(aspect="equal", xlabel="x", ylabel="y",
           title="Geometry, materials, initial crack, and boundary conditions")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(output / "initial_conditions.png", dpi=180)
    plt.close(fig)
    _plot_field(output / "damage_final.png", nodes_np, elements_np, d_np,
                "Final phase-field damage", "damage d", True, 0.0, 1.0,
                cmap="Reds", interfaces=interfaces)
    _plot_field(output / "displacement_final.png", nodes_np, elements_np, u_mag,
                "Final displacement magnitude", "|u|", True, interfaces=interfaces)
    _plot_field(output / "strain_final.png", nodes_np, elements_np, eq_strain,
                "Final strain tensor norm", "strain tensor norm", False,
                interfaces=interfaces)
    _plot_field(output / "stress_final.png", nodes_np, elements_np, von_mises,
                "Final von Mises stress", "von Mises stress", False,
                cmap="plasma", interfaces=interfaces)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    ax.plot([row["opening"] for row in rows], [abs(row["reaction_force"]) for row in rows], "o-")
    ax.set(xlabel="opening displacement", ylabel="reaction force magnitude",
           title="Load-displacement response")
    fig.savefig(output / "load_displacement.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
    axes[0].semilogy([row["load_factor"] for row in rows],
                     [max(row["projected_damage_residual"], 1e-16) for row in rows], "o-")
    axes[0].set(xlabel="load factor", ylabel="projected damage residual",
                title="Damage convergence")
    axes[1].plot([row["opening"] for row in rows],
                 [row["connected_crack_front_x"] for row in rows], "o-")
    axes[1].set(xlabel="opening displacement", ylabel="connected crack-front x",
                title="Connected crack advance")
    fig.savefig(output / "convergence_and_crack_front.png", dpi=180)
    plt.close(fig)
    frames[0].save(output / "damage_evolution.gif", save_all=True,
                   append_images=frames[1:], duration=180, loop=0)

    meshio.write_points_cells(
        output / "final_fields.vtu",
        np.column_stack([nodes_np, np.zeros(nodes_np.shape[0])]),
        [("triangle", elements_np)],
        point_data={"displacement": np.column_stack([u_np, np.zeros(nodes_np.shape[0])]),
                    "displacement_magnitude": u_mag, "damage": d_np},
        cell_data={"material_id": [material_ids], "E": [E_np], "Gc": [Gc_np],
                   "equivalent_strain": [eq_strain], "von_mises": [von_mises],
                   "eps_xx": [eps_xx], "eps_yy": [eps_yy], "gamma_xy": [gamma_xy],
                   "sxx": [sxx], "syy": [syy], "sxy": [sxy]})
    if spec.source_path:
        shutil.copy2(spec.source_path, output / "config.yaml")
        config_text = Path(spec.source_path).read_text(encoding="utf-8")
    else:
        config_text = f"schema_version: 2\nname: {spec.name}\n"
        (output / "config.yaml").write_text(config_text, encoding="utf-8")
    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": 2, "problem": spec.name,
        "solver_path": "schema-v2 heterogeneous staggered quasi-static AT2",
        "materials": material_names, "n_nodes": int(mesh.n_nodes),
        "n_elements": int(mesh.n_elems), "l0": l0,
        "maximum_element_size": float(mesh.elem_h.max().item()),
        "maximum_element_size_over_l0": float(mesh.elem_h.max().item()) / l0,
        "maximum_element_edge_length": maximum_edge_length,
        "maximum_element_edge_length_over_l0": maximum_edge_length / l0,
        "trajectory_format": "h5" if trajectory_path is not None else None,
        "accepted_steps": len(rows), "cutbacks": cutbacks,
        "final_opening": rows[-1]["opening"],
        "final_reaction_force": rows[-1]["reaction_force"],
        "final_connected_crack_front_x": rows[-1]["connected_crack_front_x"],
        "final_maximum_damage": rows[-1]["maximum_damage"],
        "maximum_projected_damage_residual": max(row["projected_damage_residual"] for row in rows),
        "all_steps_converged": True, "runtime_seconds": elapsed,
        "scope": "heterogeneous E/Gc AT2 teaching calculation; not ASTM D5528 or a calibrated bimaterial benchmark",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "run_metadata.json").write_text(json.dumps({
        "schema_version": 1, "python": platform.python_version(),
        "pytorch": torch.__version__, "device": "cpu", "dtype": "float64",
        "runtime_seconds": elapsed}, indent=2) + "\n", encoding="utf-8")
    (output / "run_lockfile.json").write_text(json.dumps({
        "schema_version": 2,
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "source_config": spec.source_path,
        "solver_parameters": dict(spec.solver.parameters)}, indent=2) + "\n", encoding="utf-8")
    visuals = ["initial_conditions.png", "material_fields.png", "damage_final.png",
               "displacement_final.png", "strain_final.png", "stress_final.png",
               "load_displacement.png", "convergence_and_crack_front.png",
               "damage_evolution.gif"]
    visual_rows = []
    for name in visuals:
        path = output / name
        entry = {"path": name, "size_bytes": path.stat().st_size,
                 "artifact_type": "animation" if path.suffix == ".gif" else "image"}
        if entry["artifact_type"] == "image":
            with Image.open(path) as image:
                entry["width_px"], entry["height_px"] = image.size
        visual_rows.append(entry)
    (output / "visual_manifest.json").write_text(
        json.dumps(visual_rows, indent=2) + "\n", encoding="utf-8")
    files = ["config.yaml", "history.csv", "solver_telemetry.csv", "timing_per_step.csv",
             "nodal_fields.csv", "element_fields.csv", "final_fields.vtu", *visuals,
             "summary.json", "run_metadata.json", "run_lockfile.json",
             "visual_manifest.json", "run_manifest.json"]
    if trajectory_path is not None:
        files.append(trajectory_path.name)
    (output / "run_manifest.json").write_text(json.dumps({
        "schema_version": 1, "problem": spec.name,
        "command": f"python -m phast run {spec.source_path or '<config.yaml>'} --output_dir {output}",
        "files": files, "summary": summary}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0
