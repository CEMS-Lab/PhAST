"""Plain data for the standard figures, taken from a :class:`phast.Result` through its public API.

Everything the drawing code needs is normalised here, once, so the sheets
never look at a layout: the geometry (2D mesh, voxel grid or 3D point cloud),
the drawable fields with their location, components and units, materials and
phases, node sets and boundary conditions, the history tables and the stored
steps. Choices depend only on what the result reports (dimension, grid or
mesh, cell types, field presence, history columns). Information a layout does
not carry is recorded as a footnote, never raised.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

import numpy as np

from ....result import Result, ResultLoadError
from ..surface import Surface, boundary_surface, cells_by_type, near_octant_removed

VECTOR = ("x", "y", "z")
TENSOR_2D = ("xx", "yy", "xy")
TENSOR_3D = ("xx", "yy", "zz", "yz", "xz", "xy")
TENSOR_NAMES = set(TENSOR_3D) | {"zz"}
# VTK cell type -> number of corner nodes (quadratic cells list their corners first)
VTK_CORNERS = {5: 3, 22: 3, 9: 4, 23: 4, 28: 4, 10: 4, 24: 4, 12: 8, 25: 8, 29: 8}
VTK_NAMES = {5: "T3", 22: "T6", 9: "Q4", 23: "Q8", 28: "Q9", 10: "TET4", 24: "TET10", 12: "H8", 25: "H20", 29: "H27"}
MATERIAL_ARRAYS = {"phase_id", "orientation"}
DAMAGE_THRESHOLD = 0.95
MAX_MEASURE_STEPS = 200

# display names: (symbol, words); symbols use STIX mathtext
_NAMES = {
    "displacement": (r"$|\mathbf{u}|$", "displacement magnitude"),
    "velocity": (r"$|\mathbf{v}|$", "velocity magnitude"),
    "acceleration": (r"$|\mathbf{a}|$", "acceleration magnitude"),
    "damage": (r"$d$", "damage"),
    "history_field": (r"$H$", "history variable"),
    "history_field_nodal": (r"$H$", "history variable at nodes"),
    "H_damage_input": (r"$H$", "history used by the damage solve"),
    "psi_plus": (r"$\psi^+$", "tensile energy density"),
    "stress": (r"$\sigma$", "stress"),
    "strain": (r"$\varepsilon$", "strain"),
    "plastic_strain": (r"$\varepsilon^p$", "plastic strain"),
    "equivalent_plastic_strain": (r"$\bar{\varepsilon}^p$", "equivalent plastic strain"),
}
_ORDER = ["displacement", "damage", "history_field", "H_damage_input", "history_field_nodal", "velocity", "acceleration"]


def _plain_units(units: Any) -> str | None:
    if units is None:
        return None
    text = str(units).strip()
    return text or None


def unit_suffix(units: str | None) -> str:
    """`` (mm)`` for a stated unit, nothing for dimensionless ("1") or absent units."""
    if units is None or units in {"1", ""}:
        return ""
    return f" ({units})"


@dataclass
class FieldInfo:
    """One drawable field."""

    name: str                       # name passed to Result.field
    location: str                   # "node" or "cell"
    components: tuple[str, ...]     # () for a scalar
    units: str | None
    kind: str | None = None
    per_point: int = 0              # points per cell when kept per quadrature point
    derived_method: str | None = None
    assumed_components: bool = False
    shear: str | None = None

    @property
    def role(self) -> str:
        if (self.name == "damage" or self.name.startswith("damage_")) and not self.components and not self.per_point:
            return "damage"      # every damage-like field uses the one map on [0, 1]
        if self.per_point:
            return "perpoint"
        if not self.components or len(self.components) == 1:
            return "scalar"
        if tuple(self.components) in (VECTOR[:2], VECTOR):
            return "vector"
        return "components"

    @property
    def symbol(self) -> str | None:
        """Mathematical symbol of a known field, or None."""
        return _NAMES.get(self.name, (None, None))[0]

    @property
    def words(self) -> str:
        return _NAMES.get(self.name, (None, None))[1] or self.name.replace("_", " ")

    @property
    def short(self) -> str:
        """Colour-bar name: the symbol when there is one, else the words."""
        if self.symbol:
            return self.symbol
        return f"|{self.words}|" if self.role == "vector" else self.words

    def title(self) -> str:
        """Panel title of the scalar shown for this field (each name appears once)."""
        known = self.symbol is not None
        if self.role == "vector" and not known:
            text = f"{self.words}, magnitude"
        else:
            text = f"{self.words} {self.symbol}" if known else self.words
        if self.role == "perpoint":
            text += f", mean of the {self.per_point} points of each cell"
        return text

    def component_symbol(self, comp: str) -> str:
        base = _NAMES.get(self.name, (None, None))[0]
        if self.shear == "engineering" and comp in ("xy", "yz", "xz") and base and base.startswith(r"$\varepsilon"):
            base = r"$\gamma" + base[len(r"$\varepsilon"):]
        if base and base.startswith("$") and comp in TENSOR_NAMES:
            return base[:-1] + "_{" + comp + "}$"
        return f"{self.words} {comp}"


@dataclass
class Column:
    values: np.ndarray
    units: str | None = None
    method: str | None = None
    codes: dict[int, str] | None = None     # categorical columns


@dataclass
class Table:
    name: str
    columns: dict[str, Column]
    per_step: bool = False      # one row per stored step (not per increment)

    @property
    def n(self) -> int:
        return max((len(c.values) for c in self.columns.values()), default=0)

    def numeric(self, name: str) -> np.ndarray | None:
        col = self.columns.get(name)
        if col is None or col.codes is not None:
            return None
        return np.asarray(col.values, float)


@dataclass
class Response:
    table: Table
    ys: list[str]
    primary: str
    x: str | None


@dataclass
class InterfaceBlock:
    """One ``/interfaces/<name>`` group: the drawn nodes of every interface cell and its interface-point arrays."""

    name: str
    cell_type: str
    nodes: np.ndarray                   # (n_cells, 2) in 2D, (n_cells, 4) in 3D
    fields: dict[str, dict[str, Any]]


@dataclass
class Mesh2D:
    points: np.ndarray                  # (N, 2)
    groups: list[tuple[np.ndarray, np.ndarray]]   # (cell indices, corner node indices (n, k))
    tri: np.ndarray                     # (T, 3)
    tri_cell: np.ndarray                # (T,)
    areas: np.ndarray                   # (E,)
    n_cells: int
    cell_types: list[str]


def _polygon_areas(xy: np.ndarray) -> np.ndarray:
    """Areas of polygons (n, k, 2) by the shoelace formula."""
    x, y = xy[..., 0], xy[..., 1]
    return 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1))


def build_mesh2d(points: np.ndarray, corner_groups: list[tuple[np.ndarray, np.ndarray]], n_cells: int,
                 cell_types: list[str]) -> Mesh2D:
    tris, tri_cell = [], []
    areas = np.zeros(n_cells)
    for cells, corners in corner_groups:
        if corners.shape[1] == 3:
            tris.append(corners); tri_cell.append(cells)
        else:
            tris += [corners[:, [0, 1, 2]], corners[:, [0, 2, 3]]]
            tri_cell += [cells, cells]
        areas[cells] = _polygon_areas(points[corners])
    tri = np.concatenate(tris).astype(np.int64) if tris else np.zeros((0, 3), np.int64)
    tri_cell_a = np.concatenate(tri_cell).astype(np.int64) if tri_cell else np.zeros(0, np.int64)
    return Mesh2D(points=points, groups=corner_groups, tri=tri, tri_cell=tri_cell_a, areas=areas,
                  n_cells=n_cells, cell_types=cell_types)


def corner_groups_from_vtk(conn: np.ndarray, offsets: np.ndarray, types: np.ndarray) -> tuple[list, list]:
    lut = np.zeros(256, np.int64)
    for t, k in VTK_CORNERS.items():
        lut[t] = k
    types = np.asarray(types, np.int64)
    nc = lut[types]
    groups = []
    for k in (3, 4):
        cells = np.nonzero(nc == k)[0]
        if cells.size:
            groups.append((cells, conn[offsets[cells][:, None] + np.arange(k)[None, :]]))
    names = sorted({VTK_NAMES.get(int(t), f"VTK {int(t)}") for t in np.unique(types)})
    return groups, names


def corner_groups_from_cells(cells: np.ndarray) -> tuple[list, list]:
    cells = np.asarray(cells)
    k = cells.shape[1]
    corners = 3 if k in (3, 6) else 4 if k in (4, 8, 9) else 0
    if corners == 0:
        return [], [f"{k}-node cells"]
    name = {3: "T3", 6: "T6", 4: "Q4", 8: "Q8", 9: "Q9"}[k]
    return [(np.arange(len(cells)), cells[:, :corners].astype(np.int64))], [name]


def node_to_cell(a: np.ndarray) -> np.ndarray:
    """Mean of the eight corner values of every voxel: (nz+1, ny+1, nx+1) to (nz, ny, nx)."""
    return (a[:-1, :-1, :-1] + a[1:, :-1, :-1] + a[:-1, 1:, :-1] + a[:-1, :-1, 1:]
            + a[1:, 1:, :-1] + a[1:, :-1, 1:] + a[:-1, 1:, 1:] + a[1:, 1:, 1:]) / 8.0


def voxel_slice(a: np.ndarray, plane: str, index: int) -> np.ndarray:
    """Section of a grid array ``(nz, ny, nx[, c])`` (cells or points, x fastest).

    Returns rows along the second named axis and columns along the first:
    ``xy`` -> (y, x) at z index ``index``; ``xz`` -> (z, x) at y index;
    ``yz`` -> (z, y) at x index. Draw with ``origin="lower"``.
    """
    if plane == "xy":
        return a[index]
    if plane == "xz":
        return a[:, index, :]
    if plane == "yz":
        return a[:, :, index]
    raise ValueError(f"unknown plane {plane!r}")


def _codes(attr: Any) -> dict[int, str] | None:
    if attr is None:
        return None
    try:
        raw = json.loads(attr) if isinstance(attr, str) else dict(attr)
        return {int(k): str(v) for k, v in raw.items()}
    except (TypeError, ValueError):
        return None


class RunData:
    """Normalised view of one result for drawing."""

    def __init__(self, result: Result):
        self.result = result
        self.notes: dict[str, list[str]] = {}   # footnotes by topic
        self.layouts = result.layouts()
        if not self.layouts or not all(x.startswith("2.") for x in self.layouts):          # 2.0, 2.1, ...
            raise ResultLoadError("the standard figures are drawn from a layout-2.x result file (result.vtkhdf)")
        self.metadata = {**(self._safe(result.metadata, {}) or {}), **(self._safe(result.file_metadata, {}) or {})}
        self.steps: list[int] = list(result.steps())
        times = result.times()
        self.times = None if times is None else np.asarray(times, float)
        self.n_steps = len(self.steps)
        self._mesh_summary = self._safe(result.mesh, {}) or {}
        self._read_geometry()
        self._read_materials()
        self._read_loading()
        self._read_interfaces()
        self._read_fields()
        self._read_tables()
        self._field_cache: dict[tuple[str, int], np.ndarray] = {}

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _safe(fn, default=None):
        try:
            return fn()
        except (ResultLoadError, KeyError, OSError, ValueError, TypeError):
            return default

    def note(self, text: str, topic: str = "geometry") -> None:
        """Record a statement shown as a footnote of the figures that concern ``topic``
        (geometry, materials, loading, fields, history or steps)."""
        items = self.notes.setdefault(topic, [])
        if text not in items:
            items.append(text)

    def notes_for(self, *topics: str) -> list[str]:
        out: list[str] = []
        for topic in topics:
            out += [t for t in self.notes.get(topic, []) if t not in out]
        return out

    @property
    def layout_label(self) -> str:
        return "layout " + ", ".join(self.layouts)

    # -- geometry -------------------------------------------------------------
    def _read_geometry(self) -> None:
        r = self.result
        step_mesh = self._safe(r.step_mesh)
        if step_mesh is not None and len(set(step_mesh.tolist())) > 1:
            raise ValueError("the figures draw one mesh; this file's mesh changes during the run "
                             "(mesh_varies = true, layout 2.1, section 8.6). Read a step with "
                             "result.mesh_arrays(step) and result.field(name, step)")
        self.grid = self._safe(r.grid)
        arrays = self._safe(r.mesh_arrays, {}) or {}
        self.mesh_arrays = arrays
        self.mesh2d: Mesh2D | None = None
        self.points3d: np.ndarray | None = None
        self.groups3d: dict = {}
        self._surfaces: dict[bool, Surface] = {}
        unit_system = self.metadata.get("unit_system") if isinstance(self.metadata, dict) else None
        self.length_units: str | None = None
        if unit_system:
            self.length_units = str(unit_system).split("-")[0]
        if self.grid is not None and "shape" in self.grid:
            self.kind = "grid"
            nx, ny, nz = (int(v) for v in self.grid["shape"])
            self.shape = (nx, ny, nz)
            self.dim = 3
            spacing = self.grid.get("spacing")
            self.spacing = np.asarray(spacing if spacing is not None else [1.0, 1.0, 1.0], float)
            if spacing is None:
                self.note("Voxel spacing is not stored; one voxel is drawn as one length unit.")
            self.origin = np.asarray(self.grid.get("origin", [0.0, 0.0, 0.0]), float)
            self.length_units = self.length_units or self.grid.get("units")
            self.point_shape = (nz + 1, ny + 1, nx + 1)
            self.cell_shape = (nz, ny, nx)
            self.n_points, self.n_cells = int(np.prod(self.point_shape)), int(np.prod(self.cell_shape))
        else:
            points = arrays.get("points")
            if points is None:
                self.kind, self.dim, self.n_points, self.n_cells = "none", int(self.metadata.get("dim", 2) or 2), 0, 0
                self.note("This result stores no mesh coordinates; maps cannot be drawn.")
            else:
                points = np.asarray(points, float)
                if points.shape[1] == 3 and np.allclose(points[:, 2], 0.0):
                    points = points[:, :2]
                self.dim = points.shape[1]
                self.n_points = len(points)
                if "connectivity" in arrays and "offsets" in arrays and "types" in arrays:
                    offsets = np.asarray(arrays["offsets"], np.int64)
                    groups, names = corner_groups_from_vtk(np.asarray(arrays["connectivity"], np.int64), offsets,
                                                           np.asarray(arrays["types"]))
                    self.n_cells = len(offsets) - 1
                elif "cells" in arrays and not isinstance(arrays["cells"], dict):
                    cells = np.asarray(arrays["cells"])
                    groups, names = corner_groups_from_cells(cells)
                    self.n_cells = len(cells)
                else:
                    groups, names = [], []
                    self.n_cells = int(self._mesh_summary.get("n_elements", 0) or 0)
                self.cell_types = names
                if self.dim == 2:
                    self.kind = "mesh2d"
                    self.mesh2d = build_mesh2d(points, groups, self.n_cells, names)
                    if not groups:
                        self.note("Cell connectivity is not stored in a form that can be drawn; "
                                  "nodes are drawn as points.")
                else:
                    self.kind = "mesh3d"
                    self.points3d = points
                    if "connectivity" in arrays and "offsets" in arrays and "types" in arrays:
                        self.groups3d = cells_by_type(arrays["connectivity"], arrays["offsets"], arrays["types"])
                    if self.groups3d:
                        self.note("Unstructured 3D mesh: boundary faces in an isometric projection, whole body and "
                                  "cut-away with the cells of the near octant removed; values on faces are the mean of "
                                  "their nodes (nodal fields) or the value of the cell they belong to.")
                    else:
                        self.note("Unstructured 3D mesh whose cell types are not drawn as surfaces: values are drawn "
                                  "at the nodes as points in an isometric projection.")
        if self.length_units is None and any(str(k).endswith("_mm") for k in self._mesh_summary):
            self.length_units = "mm"
        if self.length_units is None and self.kind != "none":
            self.note("Length units are not stored in this result; coordinates are shown as stored.")
        # display scale: very small cells in mm are shown in micrometres
        self.length_scale, self.length_label = 1.0, self.length_units
        extent = self.domain_lengths()
        if self.length_units == "mm" and extent is not None and 0 < max(extent) < 0.5:
            self.length_scale, self.length_label = 1000.0, "µm"

    def domain_lengths(self) -> np.ndarray | None:
        if self.kind == "grid":
            return np.asarray(self.shape, float) * self.spacing
        if self.kind == "mesh2d":
            p = self.mesh2d.points
            return p.max(axis=0) - p.min(axis=0)
        if self.kind == "mesh3d":
            return self.points3d.max(axis=0) - self.points3d.min(axis=0)
        return None

    def axis_label(self, axis: str) -> str:
        return f"{axis}{unit_suffix(self.length_label)}"

    def surface3d(self, cut: bool) -> Surface:
        """Boundary faces of an unstructured 3D mesh; ``cut`` removes the cells of the octant nearest the viewer."""
        if cut not in self._surfaces:
            keep = near_octant_removed(self.points3d, self.groups3d, self.n_cells) if cut else None
            self._surfaces[cut] = boundary_surface(self.points3d, self.groups3d, keep)
        return self._surfaces[cut]

    # -- cohesive interfaces ----------------------------------------------------------------
    def _read_interfaces(self) -> None:
        """Interface blocks of a layout-2.0 file: nodes of each interface cell and the interface-point arrays."""
        self.interfaces: list[InterfaceBlock] = []
        if self.kind not in ("mesh2d", "mesh3d"):
            return
        files = self._safe(self.result.source_files, []) or []
        if not files:
            return
        import h5py
        try:
            with h5py.File(files[0], "r") as f:
                group = f.get("interfaces")
                for name in (group.keys() if group is not None else []):
                    g = group[name]
                    if "connectivity" not in g:
                        continue
                    conn = np.asarray(g["connectivity"][...], np.int64)
                    ctype = g["connectivity"].attrs.get("cell_type", g.attrs.get("cell_type", ""))
                    ctype = ctype.decode() if isinstance(ctype, bytes) else str(ctype)
                    k = 2 if (ctype == "COH2" or (ctype != "COH4" and self.dim == 2)) else 4
                    fields = {}
                    for key, ds in g.items():
                        if isinstance(ds, h5py.Dataset) and key != "connectivity" and ds.ndim >= 3 and ds.shape[1] == len(conn):
                            a = {kk: (x.decode() if isinstance(x, bytes) else x) for kk, x in ds.attrs.items()}
                            fields[key] = dict(units=_plain_units(a.get("units")), kind=a.get("kind"), n_points=int(ds.shape[2]))
                    if conn.shape[1] >= k and conn.size and 0 <= conn.min() and conn.max() < self.n_points:
                        self.interfaces.append(InterfaceBlock(name, ctype or ("COH2" if k == 2 else "COH4"), conn[:, :k], fields))
        except OSError:
            return
        if self.interfaces:
            self.note("Cohesive interfaces: each interface cell is drawn on its first side (the solver's top side; the first nodes of its "
                      "connectivity); interface-point values are shown as the mean over the points of each cell.",
                      "interfaces")

    def interface_values(self, block: "InterfaceBlock", name: str, step: int) -> np.ndarray:
        """Mean over the interface points of each cell of ``block`` at stored step ``step``: ``(n_cells,)`` (vector
        fields: the magnitude of the mean)."""
        v = np.asarray(self.result.quadrature(name, step, block=block.name), float)
        v = v.mean(axis=1)
        return np.linalg.norm(v, axis=-1) if v.ndim > 1 else v

    # -- materials ---------------------------------------------------------------
    def _read_materials(self) -> None:
        m = self._safe(self.result.materials, {}) or {}
        self.phases: dict[int, dict[str, Any]] = {}
        self.constants: dict[str, Any] = {}
        phase_id = None
        if isinstance(m.get("phases"), dict):
            for pid, attrs in m["phases"].items():
                try:
                    key = int(pid)
                except (TypeError, ValueError):
                    continue
                self.phases[key] = dict(attrs)
            phase_id = m.get("phase_id")
        else:
            names = m.get("phase_names")
            phase_id = m.get("phase_id")
            for key, value in m.items():
                if key in ("phase_id", "phase_names"):
                    continue
                if isinstance(value, (int, float, str)) and not isinstance(value, bool):
                    self.constants[key] = value
            if names is not None and phase_id is not None:
                for pid in np.unique(np.asarray(phase_id)).tolist():
                    pid = int(pid)
                    self.phases[pid] = {"name": str(names[pid]) if 0 <= pid < len(names) else f"phase {pid}"}
        self.phase_cells = None
        if phase_id is not None:
            arr = np.asarray(phase_id)
            try:
                self.phase_cells = self.to_cells(arr)
            except ValueError:
                self.note("The stored phase map does not match the cells; phases are not drawn.", "materials")
        if self.phase_cells is not None:
            for pid in np.unique(self.phase_cells).tolist():
                self.phases.setdefault(int(pid), {"name": f"phase {int(pid)}"})
        if not m:
            self.note("Material data are not stored in this layout.", "materials")

    def to_cells(self, arr: np.ndarray) -> np.ndarray:
        """Cell array in drawing shape: (E,) for meshes, (nz, ny, nx) for grids."""
        if self.kind == "grid":
            return self.on_grid(arr, "cell")
        if arr.shape[0] != self.n_cells:
            raise ValueError("cell count mismatch")
        return arr

    def on_grid(self, arr: np.ndarray, location: str) -> np.ndarray:
        shape = self.point_shape if location == "node" else self.cell_shape
        if tuple(arr.shape[:3]) == shape:
            return arr
        if arr.shape[0] == int(np.prod(shape)):
            return arr.reshape(shape + tuple(arr.shape[1:]))
        raise ValueError(f"array of shape {arr.shape} does not fit the grid {shape}")

    # -- loading ---------------------------------------------------------------
    def _read_loading(self) -> None:
        self.node_sets: dict[str, np.ndarray] = {}
        for name, idx in (self.mesh_arrays.get("node_sets") or {}).items():
            self.node_sets[str(name)] = np.asarray(idx, np.int64).ravel()
        for key, value in self.mesh_arrays.items():
            if key in ("points", "node_sets", "types", "offsets", "connectivity"):
                continue
            arr = np.asarray(value) if not isinstance(value, dict) else None
            if arr is not None and arr.ndim == 1 and arr.shape[0] == self.n_points and self.kind != "grid" \
                    and arr.dtype.kind in "biuf" and self.n_points > 0:
                self.node_sets[key.replace("_", " ")] = np.nonzero(arr)[0]
        bcs = self._safe(self.result.boundary_conditions)
        self.bcs: dict[str, dict[str, Any]] = {}
        self.periodic: dict[str, Any] | None = None
        if bcs is None:
            self.bcs_stored = False
            if self.kind == "grid" and self.grid.get("periodic") is not None:
                flags = self.grid["periodic"]
                axes = [a for a, f in zip("xyz", flags) if f] if isinstance(flags, (list, tuple)) else []
                if axes:
                    self.periodic = {"axes": ",".join(axes)}
            self.note("Boundary conditions are not stored in this layout"
                      + ("; the stored node sets are shown." if self.node_sets else "."), "loading")
        else:
            self.bcs_stored = True
            self.bcs = dict(bcs.get("bc") or {})
            self.periodic = bcs.get("periodic")

    # -- fields ----------------------------------------------------------------------
    def _read_fields(self) -> None:
        self.fields: dict[str, FieldInfo] = {}
        r = self.result
        names = [n for n in r.field_names() if n not in MATERIAL_ARRAYS]
        probe_step = self.steps[-1] if self.steps else -1
        for name in names:
            attrs = self._safe(lambda: r.field_attributes(name), {}) or {}
            try:
                arr = np.asarray(r.field(name, probe_step))
            except (ResultLoadError, KeyError, ValueError, OSError):
                arr = None
                for s in reversed(self.steps):
                    try:
                        arr = np.asarray(r.field(name, s))
                        break
                    except (ResultLoadError, KeyError, ValueError, OSError):
                        continue
                if arr is None:
                    continue
            info = self._classify(name, arr, attrs)
            if info is not None:
                self.fields[name] = info
        if any(f.assumed_components for f in self.fields.values()):
            order = "xx, yy, xy" if self.dim == 2 else "xx, yy, zz, yz, xz, xy"
            self.note(f"Tensor component order is not stored in this layout; {order} is assumed.", "fields")

    def _classify(self, name: str, arr: np.ndarray, attrs: dict[str, Any]) -> FieldInfo | None:
        if arr.ndim == 0 or self.kind == "none":
            return None
        location = attrs.get("location")
        if self.kind == "grid":
            if tuple(arr.shape[:3]) == self.point_shape or arr.shape[0] == self.n_points:
                where = "node"
            elif tuple(arr.shape[:3]) == self.cell_shape or arr.shape[0] == self.n_cells:
                where = "cell"
            else:
                return None
            tail = arr.shape[3:] if tuple(arr.shape[:3]) in (self.point_shape, self.cell_shape) else arr.shape[1:]
        else:
            if location == "node" or (location is None and arr.shape[0] == self.n_points):
                where = "node"
            elif arr.shape[0] == self.n_cells:
                where = "cell"
            else:
                return None
            if (where == "node" and arr.shape[0] != self.n_points) or (where == "cell" and arr.shape[0] != self.n_cells):
                return None
            tail = arr.shape[1:]
        ncomp = int(np.prod(tail)) if tail else 0
        comps_attr = attrs.get("components")
        comps: tuple[str, ...] = ()
        if comps_attr:
            comps = tuple(c.strip() for c in str(comps_attr).split(",") if c.strip()) if isinstance(comps_attr, str) \
                else tuple(str(c) for c in comps_attr)
        per_point = 0
        assumed = False
        if location == "quadrature_point" or (comps and all(re.fullmatch(r"qp\d+", c) for c in comps)):
            per_point = ncomp or 1
            comps = ()
        elif self.dim == 2 and comps == VECTOR and ncomp == 2:
            comps = VECTOR[:2]
        elif not comps and ncomp > 1:
            if where == "node" and ncomp == self.dim:
                comps = VECTOR[: self.dim]
            elif ncomp == 3 and self.dim == 2:
                comps, assumed = TENSOR_2D, True
            elif ncomp == 6 and self.dim == 3:
                comps, assumed = TENSOR_3D, True
            else:
                comps = tuple(f"c{i}" for i in range(ncomp))
        if comps and ncomp and len(comps) != ncomp and not per_point:
            comps = tuple(f"c{i}" for i in range(ncomp))
        return FieldInfo(name=name, location=where, components=comps if len(comps) > 1 else (),
                         units=_plain_units(attrs.get("units")), kind=attrs.get("kind"), per_point=per_point,
                         derived_method=attrs.get("derived_method"), assumed_components=assumed,
                         shear=attrs.get("shear"))

    def ordered_fields(self, roles: set[str], selected: list[str] | None = None) -> list[FieldInfo]:
        chosen = [f for f in self.fields.values() if f.role in roles]
        if selected:
            wanted = set(selected)
            chosen = [f for f in chosen if f.name in wanted or _alias(f.name) & wanted]
        rank = {n: i for i, n in enumerate(_ORDER)}
        return sorted(chosen, key=lambda f: (rank.get(f.name, len(rank)), f.name))

    def values(self, info: FieldInfo, step: int) -> np.ndarray:
        """Field at ``step`` in drawing shape: (N|E[, c]) for meshes, grid-shaped for voxels;
        per-point arrays reduced to the cell mean."""
        key = (info.name, int(step))
        if key not in self._field_cache:
            arr = np.asarray(self.result.field(info.name, step), float)
            if self.kind == "grid":
                arr = self.on_grid(arr, info.location)
            if info.per_point and arr.ndim > (3 if self.kind == "grid" else 1):
                arr = arr.reshape(arr.shape[: (3 if self.kind == "grid" else 1)] + (-1,)).mean(axis=-1)
            if len(self._field_cache) > 64:
                self._field_cache.clear()
            self._field_cache[key] = arr
        return self._field_cache[key]

    def scalar(self, info: FieldInfo, step: int, comp: str | None = None) -> np.ndarray:
        """The scalar drawn for ``info``: magnitude of a vector, one component, or the values."""
        arr = self.values(info, step)
        if comp is not None:
            return arr[..., list(info.components).index(comp)]
        if info.role == "vector":
            return np.linalg.norm(arr, axis=-1)
        if arr.ndim > (3 if self.kind == "grid" else 1):
            return arr[..., 0]
        return arr

    # -- history -----------------------------------------------------------------------
    def _read_tables(self) -> None:
        self.tables: list[Table] = []
        r = self.result
        cols = self._safe(r.history_columns, {}) or {}
        attrs = self._safe(r.history_attributes, {}) or {}
        main: dict[str, Column] = {}
        for key, values in cols.items():
            a = attrs.get(key, {})
            base = key.replace("/", "_")
            values = np.asarray(values)
            if values.ndim == 1:
                main[base] = Column(values.astype(float), _plain_units(a.get("units")), a.get("method"),
                                    _codes(a.get("codes")))
                continue
            comps = [c.strip() for c in str(a.get("components", "")).split(",") if c.strip()]
            flat = values.reshape(values.shape[0], -1)
            if len(comps) != flat.shape[1]:
                comps = [str(i) for i in range(flat.shape[1])]
            for i, comp in enumerate(comps):
                main[f"{base}_{comp}"] = Column(flat[:, i].astype(float), _plain_units(a.get("units")),
                                                a.get("method"))
        if main:
            self.tables.append(Table("history", main))

    def table(self, name: str) -> Table | None:
        return next((t for t in self.tables if t.name == name), None)

    # -- stored steps and history rows ---------------------------------------------------
    def step_rows(self, table: Table) -> dict[int, int]:
        """Row of ``table`` for each stored step that can be matched."""
        rows: dict[int, int] = {}
        if table.per_step and table.n == self.n_steps:
            return {s: i for i, s in enumerate(self.steps)}
        if table.name == "history":
            index = self._safe(self.result.step_index)
            if index is not None:
                for s in self.steps:
                    hits = np.nonzero(np.asarray(index) == s)[0]
                    if hits.size and hits[0] < table.n:
                        rows[s] = int(hits[0])
            return rows
        step_col = table.numeric("step")
        if step_col is not None:
            for s in self.steps:
                hits = np.nonzero(step_col == s)[0]
                if hits.size:
                    rows[s] = int(hits[-1])
        return rows

    def response(self) -> Response | None:
        """Load-like history columns and the column they are plotted against."""
        best = None
        for table in self.tables:
            for group, pattern in enumerate((r"(^|_)reaction", r"^macro_stress_", r"force")):
                cands = []
                for name in table.columns:
                    y = table.numeric(name)
                    if y is None or not re.search(pattern, name, re.I) or not _varies(y):
                        continue
                    cands.append((float(np.nanmax(np.abs(y))), name))
                if cands:
                    cands.sort(reverse=True)
                    top = cands[0][0]
                    ys = [n for v, n in cands if v >= 1e-3 * top][:4]
                    key = (group, -table.n)
                    if best is None or key < best[0]:
                        best = (key, Response(table, ys, ys[0], None))
                    break
        if best is None:
            return None
        resp = best[1]
        resp.x = self.drive(resp.table, resp.primary)
        return resp

    def drive(self, table: Table, y: str | None = None) -> str | None:
        """Column a load-like quantity is plotted against."""
        if y is not None and y.startswith("macro_stress_"):
            partner = "macro_strain_" + y[len("macro_stress_"):]
            if table.numeric(partner) is not None and _varies(table.numeric(partner)):
                return partner
        patterns = [r"^load_parameter$", r"^applied_disp", r"^applied_displacement", r"^load_factor$",
                    r"^load_fraction$", r"displacement", r"^macro_strain_", r"^time$", r"^time_s$",
                    r"^increment$", r"^step$"]
        for pattern in patterns:
            for name in table.columns:
                if name == y or not re.search(pattern, name):
                    continue
                values = table.numeric(name)
                if values is not None and _varies(values):
                    return name
        return None

    def peak_step(self) -> tuple[int | None, str]:
        resp = self.response()
        if resp is None:
            return None, "no load-like history column is stored"
        y = np.abs(resp.table.numeric(resp.primary))
        if not np.isfinite(y).any():
            return None, "the load-like column has no finite values"
        peak_row = int(np.nanargmax(y))
        rows = self.step_rows(resp.table)
        if not rows:
            return None, "stored steps cannot be matched to history rows"
        label = min(rows, key=lambda s: (abs(rows[s] - peak_row), rows[s]))
        return label, f"stored step nearest to the largest |{resp.primary}|"

    def choose_steps(self, spec: str = "first,peak,last") -> list[tuple[int, str]]:
        """Stored steps to draw from ``first``, ``peak``, ``middle``, ``last`` and integers.

        Integers follow ``Result.field``: the step value the result stores
        (``Result.steps()``), negative values counting from the end.
        """
        if not self.steps:
            return []
        out: list[tuple[int, str]] = []
        for token in [t.strip() for t in str(spec).split(",") if t.strip()]:
            if token == "first":
                item = (self.steps[0], "first stored step")
            elif token == "last":
                item = (self.steps[-1], "last stored step")
            elif token == "middle":
                item = (self.steps[len(self.steps) // 2], "middle stored step")
            elif token == "peak":
                label, why = self.peak_step()
                if label is None:
                    self.note(f"Peak step: {why}; the middle stored step is drawn instead.", "steps")
                    item = (self.steps[len(self.steps) // 2], "middle stored step")
                else:
                    item = (label, why)
            else:
                value = int(token)
                if value < 0:
                    if -value > len(self.steps):
                        raise ValueError(f"step {value}: only {len(self.steps)} stored steps")
                    item = (self.steps[value], f"stored step {self.steps[value]}")
                elif value in self.steps:
                    item = (value, f"stored step {value}")
                else:
                    raise ValueError(f"step {value} is not stored; stored steps: {self.steps[:12]}"
                                     + (" ..." if len(self.steps) > 12 else ""))
            labels = [s for s, _ in out]
            if item[0] in labels:      # the same stored step chosen twice: keep both reasons
                i = labels.index(item[0])
                if item[1] not in out[i][1]:
                    out[i] = (item[0], f"{out[i][1]}; {item[1]}")
            else:
                out.append(item)
        return out

    def step_caption(self, step: int) -> str:
        """Short description of a stored step: label, time, and the response value when matched."""
        parts = [f"step {step}"]
        if self.times is not None and step in self.steps:
            t = self.times[self.steps.index(step)]
            if np.isfinite(t):
                parts.append(f"t = {t:.4g}")
        resp = self.response()
        if resp is not None:
            row = self.step_rows(resp.table).get(step)
            if row is not None:
                for name in (resp.x, resp.primary):
                    if name is None or name in ("step", "increment", "time"):
                        continue
                    col = resp.table.columns[name]
                    parts.append(f"{name} = {col.values[row] + 0.0:.4g}{unit_suffix(col.units)}")   # no "-0"
        return ", ".join(parts)

    # -- damage measures computed from the fields ----------------------------------------
    def damage_measures(self) -> dict[str, np.ndarray] | None:
        info = self.fields.get("damage")
        if info is None or info.role != "damage":
            return None
        steps = self.steps
        if len(steps) > MAX_MEASURE_STEPS:
            pick = np.unique(np.linspace(0, len(steps) - 1, MAX_MEASURE_STEPS).astype(int))
            steps = [steps[i] for i in pick]
            self.note(f"Damage measures computed from the fields at {len(steps)} of {len(self.steps)} stored steps.", "history")
        out = {"step": [], "max": [], "fraction": []}
        for s in steps:
            try:
                d = np.asarray(self.result.field("damage", s), float)
            except (ResultLoadError, KeyError, ValueError):
                continue
            cell_d, weights = self._cell_damage(d, info.location)
            if cell_d is None:
                continue
            out["step"].append(s)
            out["max"].append(float(np.nanmax(d)))
            out["fraction"].append(float(np.sum(weights * (cell_d > DAMAGE_THRESHOLD)) / np.sum(weights)))
        if not out["step"]:
            return None
        return {k: np.asarray(v, float) for k, v in out.items()}

    def _cell_damage(self, d: np.ndarray, location: str):
        if self.kind == "grid":
            g = self.on_grid(d, location)
            cell = node_to_cell(g) if location == "node" else g
            return cell.ravel(), np.ones(cell.size)
        if self.kind == "mesh2d" and self.mesh2d.groups:
            if location == "cell":
                return d, self.mesh2d.areas
            cell = np.zeros(self.n_cells)
            for cells, corners in self.mesh2d.groups:
                cell[cells] = d[corners].mean(axis=1)
            return cell, self.mesh2d.areas
        return None, None


def _alias(name: str) -> set[str]:
    return {"history_field": {"H", "H_elem", "history"}, "displacement": {"u"}, "damage": {"d"},
            "velocity": {"v"}, "acceleration": {"a"}}.get(name, set())


def _varies(values: np.ndarray | None) -> bool:
    if values is None:
        return False
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return v.size >= 2 and float(np.ptp(v)) > 0.0
