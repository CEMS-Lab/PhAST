"""Read one layout-2.0 result file (``result.vtkhdf``).

The layout is specified in ``docs/user_guide/result_file_layout.md``. This
module converts on read only: the file is opened with mode ``"r"`` for each
call and nothing is written. It depends on h5py and NumPy only.

Conventions of :class:`ResultFile` (section 6 of the specification):

- ``step`` is an index into the stored steps of ``/VTKHDF/Steps``
  (``0 <= step < n_steps``, negative values count from the end). Rows are
  located through the step offsets, never by assuming contiguous storage.
  The increment a stored step belongs to is given by ``history/step_index``.
- ``NSteps`` is authoritative: rows, offsets or values written for a step
  after the last complete one (an interrupted run) are ignored.
- Vectors are returned in the native dimension: 2D runs store 3 components
  with z = 0, and :meth:`field` returns 2. Tensors are returned as stored, in
  the order of the ``components`` attribute.
- UnstructuredGrid arrays are returned as ``(N[, c])`` or ``(E[, c])``;
  history kept per quadrature point is ``(E, n_qp)``. ImageData arrays are
  returned on the grid as ``(nz + 1, ny + 1, nx + 1[, c])`` for points and
  ``(nz, ny, nx[, c])`` for cells, x varying fastest, so that
  ``array.reshape(-1, ...)`` gives the VTK point or cell order.
- Static arrays (``phase_id``, ``orientation``) are returned at any step,
  including for setup files with no stored step.
- Layout 2.1 files may have a mesh that changes (``mesh_varies``): each stored
  step then has the mesh version given by :meth:`step_mesh`, :meth:`field`
  returns arrays of that mesh, and :meth:`mesh_arrays` takes the step whose
  mesh is wanted (the last stored step by default).
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ...result import ResultLoadError

SCHEMA_NAME = "phast.result"
SUPPORTED_MAJOR = 2
VECTOR_COMPONENTS = ("x", "y", "z")


def _text(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"S", "O", "U"}:
            return [_text(v) for v in value.tolist()]
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _attrs(node) -> dict[str, Any]:
    return {key: _text(value) for key, value in node.attrs.items()}


def _components(attrs: dict[str, Any]) -> list[str]:
    raw = attrs.get("components")
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(v).strip() for v in raw]
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def is_result_file(path: str | Path) -> bool:
    """True when ``path`` is an HDF5 file whose root declares ``phast_schema = "phast.result"``."""
    path = Path(path)
    if not path.is_file():
        return False
    try:
        import h5py
    except ImportError:
        return False
    if not h5py.is_hdf5(str(path)):
        return False
    try:
        with h5py.File(path, "r") as handle:
            return _text(handle.attrs.get("phast_schema")) == SCHEMA_NAME
    except OSError:
        return False


class ResultFile:
    """Read-only access to one layout-2.0 ``result.vtkhdf`` file.

    The structure (array names, locations, offsets, step count) is read once
    when the object is created; the file is not expected to change while it
    is being read (the writer rules forbid opening a running file).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Result file does not exist: {self.path}")
        try:
            import h5py  # noqa: F401
        except ImportError as exc:
            raise ResultLoadError("Reading result.vtkhdf requires the h5py package") from exc
        with self._open() as f:
            self._root_attrs = _attrs(f)
            if self._root_attrs.get("phast_schema") != SCHEMA_NAME:
                raise ResultLoadError(f"{self.path} is not a PhAST result file (phast_schema missing)")
            version = str(self._root_attrs.get("phast_schema_version", ""))
            try:
                major = int(version.split(".")[0])
            except ValueError:
                major = -1
            if major != SUPPORTED_MAJOR:
                raise ResultLoadError(
                    f"{self.path} has phast_schema_version {version!r}; this reader "
                    f"reads major version {SUPPORTED_MAJOR}"
                )
            if "VTKHDF" not in f:
                raise ResultLoadError(f"{self.path} has no /VTKHDF group")
            self._scan(f)

    @contextmanager
    def _open(self) -> Iterator[Any]:
        import h5py

        with h5py.File(self.path, "r") as handle:
            yield handle

    # -- structure ------------------------------------------------------------
    def _scan(self, f) -> None:
        v = f["VTKHDF"]
        vattrs = _attrs(v)
        self.type = str(vattrs.get("Type", "UnstructuredGrid"))
        self.dim = int(self._root_attrs.get("dim", 3))
        steps = v["Steps"] if "Steps" in v else None
        nsteps = int(_text(steps.attrs.get("NSteps", 0))) if steps is not None else 0
        if steps is not None and "Values" in steps:
            nsteps = min(nsteps, int(steps["Values"].shape[0]))
        self._n_steps = max(nsteps, 0)
        self._times = (
            np.asarray(steps["Values"][: self._n_steps], dtype=float)
            if steps is not None and "Values" in steps else np.zeros(0)
        )
        if self.type == "ImageData":
            extent = [int(x) for x in np.asarray(v.attrs["WholeExtent"]).ravel()]
            self._grid_shape = (extent[1] - extent[0], extent[3] - extent[2], extent[5] - extent[4])
            nx, ny, nz = self._grid_shape
            self.n_points = (nx + 1) * (ny + 1) * (nz + 1)
            self.n_cells = nx * ny * nz
            self._part_points = self._part_cells = None
        else:
            self._grid_shape = None
            self._part_points = np.asarray(v["NumberOfPoints"][...], dtype=np.int64).ravel()
            self._part_cells = np.asarray(v["NumberOfCells"][...], dtype=np.int64).ravel()
            self.n_points = int(self._part_points[0]) if self._part_points.size else 0
            self.n_cells = int(self._part_cells[0]) if self._part_cells.size else 0
            self._part_offsets = (
                np.asarray(steps["PartOffsets"][: self._n_steps], dtype=np.int64)
                if steps is not None and "PartOffsets" in steps else np.zeros(self._n_steps, np.int64)
            )

        self._varying = bool(self._root_attrs.get("mesh_varies", False))
        self._n_versions = int(self._part_points.size) if self._varying and self._part_points is not None else 1
        self._arrays: dict[str, dict[str, Any]] = {}
        for group, where in (("PointData", "point"), ("CellData", "cell")):
            if group not in v:
                continue
            offsets_group = f"{group[:-4]}DataOffsets"
            for name, ds in v[group].items():
                attrs = _attrs(ds)
                offsets = None
                if steps is not None and offsets_group in steps and name in steps[offsets_group]:
                    offsets = np.asarray(steps[offsets_group][name][: self._n_steps], dtype=np.int64)
                rows_per_step = 1 if self.type == "ImageData" else (self.n_points if where == "point" else self.n_cells)
                # ImageData setup files store static arrays without the time axis (layout section 2.2)
                untimed = self.type == "ImageData" and tuple(ds.shape[:3]) == self._image_shape(where)
                static = untimed or int(ds.shape[0]) == rows_per_step and (
                    offsets is None or self._n_steps == 0 or not offsets.any()
                ) and attrs.get("kind") != "history" and name not in self._state_names(f)
                self._arrays[name] = dict(
                    group=group, where=where, attrs=attrs, offsets=offsets, static=static, untimed=untimed,
                    shape=tuple(int(s) for s in ds.shape),
                )
        self._derived = sorted(f["derived"].keys()) if "derived" in f else []

    def _image_shape(self, where: str) -> tuple:
        nx, ny, nz = self._grid_shape
        return (nz + 1, ny + 1, nx + 1) if where == "point" else (nz, ny, nx)

    def _state_names(self, f) -> set[str]:
        """Arrays the state group links to are time-varying even when one step is stored."""
        names: set[str] = set()
        if "state" not in f:
            return names
        import h5py

        for key in f["state"].keys():
            link = f["state"].get(key, getlink=True)
            if isinstance(link, h5py.SoftLink):
                names.add(link.path.rsplit("/", 1)[-1])
        return names

    # -- summary ----------------------------------------------------------------
    def attributes(self) -> dict[str, Any]:
        """Root attributes (schema, identity, coupling, status) as plain Python."""
        return dict(self._root_attrs)

    def schema_version(self) -> str:
        """``phast_schema_version`` of the file (``"2.0.0"``, ``"2.1.0"``, ...)."""
        return str(self._root_attrs.get("phast_schema_version", ""))

    def meta(self) -> dict[str, Any]:
        """Attributes of ``/meta`` plus its text datasets (resolved configuration, schema)."""
        with self._open() as f:
            if "meta" not in f:
                return {}
            out = _attrs(f["meta"])
            for key in ("config_resolved_yaml", "schema_yaml"):
                if key in f["meta"]:
                    out[key] = _text(f["meta"][key][()])
            return out

    def n_steps(self) -> int:
        return self._n_steps

    def times(self) -> np.ndarray:
        """``/VTKHDF/Steps/Values`` of the complete stored steps."""
        return self._times.copy()

    def field_names(self) -> list[str]:
        """Arrays of ``/VTKHDF`` (points and cells) and of ``/derived``, as stored."""
        return sorted(set(self._arrays) | set(self._derived))

    def is_static(self, name: str) -> bool:
        """True when the array is stored once, without a time axis (for example ``phase_id``)."""
        info = self._arrays.get(name)
        return bool(info and info["static"])

    def field_attributes(self, name: str) -> dict[str, Any]:
        """``location``, ``components``, ``units``, ``kind`` and the other field attributes."""
        if name in self._arrays:
            return dict(self._arrays[name]["attrs"])
        if name in self._derived:
            with self._open() as f:
                group = f["derived"][name]
                attrs = _attrs(group)
                if "values" in group:
                    attrs.update(_attrs(group["values"]))
                return attrs
        raise ResultLoadError(f"Unknown field {name!r} in {self.path}")

    def resolve_step(self, step: int) -> int:
        """Stored-step index for ``step`` (negative values count from the end)."""
        n = self._n_steps
        if n == 0:
            raise ResultLoadError(f"{self.path} contains no stored steps")
        step = int(step)
        if not -n <= step < n:
            raise ResultLoadError(f"Step {step} not found in {self.path} ({n} stored steps)")
        return step % n

    # -- fields -----------------------------------------------------------------
    def field(self, name: str, step: int = -1) -> np.ndarray:
        """Array ``name`` at stored step ``step`` (see the module docstring for shapes)."""
        if name in self._arrays:
            return self._vtk_field(name, step)
        if name in self._derived:
            return self._derived_field(name, step)
        raise ResultLoadError(f"Unknown field {name!r}. Available fields: {', '.join(self.field_names())}")

    def _vtk_field(self, name: str, step: int) -> np.ndarray:
        info = self._arrays[name]
        image = self.type == "ImageData"
        if info["static"]:
            if self._n_steps:
                self.resolve_step(step)
            offset, s = 0, None
        else:
            s = self.resolve_step(step)
            if info["offsets"] is None or info["offsets"].shape[0] <= s:
                raise ResultLoadError(f"Field {name!r} has no offset for step {step} in {self.path}")
            offset = int(info["offsets"][s])
        with self._open() as f:
            ds = f["VTKHDF"][info["group"]][name]
            if image and info["untimed"]:
                array = np.asarray(ds[...])
            elif image:
                if offset >= ds.shape[0]:
                    raise ResultLoadError(f"Field {name!r} has no rows for step {step} in {self.path}")
                array = np.asarray(ds[offset])
            else:
                rows = self._rows_per_step(info["where"], s)
                if offset + rows > ds.shape[0]:
                    raise ResultLoadError(f"Field {name!r} has no rows for step {step} in {self.path}")
                array = np.asarray(ds[offset: offset + rows])
        return self._native(array, info["attrs"])

    def _rows_per_step(self, where: str, s: int | None) -> int:
        counts = self._part_points if where == "point" else self._part_cells
        if s is None or counts.size <= 1:
            return int(counts[0])
        part = int(self._part_offsets[s]) if s < self._part_offsets.size else 0
        return int(counts[min(part, counts.size - 1)])

    def _native(self, array: np.ndarray, attrs: dict[str, Any]) -> np.ndarray:
        if self.dim == 2 and tuple(_components(attrs)) == VECTOR_COMPONENTS and array.shape[-1] == 3:
            return np.ascontiguousarray(array[..., :2])
        return array

    def _increment_of_step(self, s: int, f) -> int:
        if "history" not in f or "step_index" not in f["history"]:
            raise ResultLoadError(f"{self.path} has no history/step_index")
        index = np.asarray(f["history/step_index"][...], dtype=np.int64)
        matches = np.nonzero(index == s)[0]
        if matches.size == 0:
            raise ResultLoadError(f"Stored step {s} has no increment in history/step_index of {self.path}")
        return int(matches[0])

    def _derived_field(self, name: str, step: int) -> np.ndarray:
        s = self.resolve_step(step)
        with self._open() as f:
            group = f["derived"][name]
            if "values" not in group or "step_index" not in group:
                raise ResultLoadError(f"derived/{name} in {self.path} lacks values or step_index")
            increment = self._increment_of_step(s, f)
            index = np.asarray(group["step_index"][...], dtype=np.int64)
            row = int(index[increment]) if increment < index.size else -1
            if row < 0:
                raise ResultLoadError(
                    f"Field {name!r} is stored in /derived only on some increments; "
                    f"step {step} (increment {increment}) has no value"
                )
            values = group["values"]
            attrs = _attrs(values)
            array = np.asarray(values[row])
        return self._native(array, attrs)

    def quadrature(self, name: str, step: int = -1, *, block: str | None = None) -> np.ndarray:
        """Per-point array at stored step ``step``: ``(E_block, n_qp[, c])``.

        Reads ``/quadrature/<block>/<name>``, or ``/interfaces/<block>/<name>``
        for cohesive interface points (``block`` is then the interface name);
        when the array is kept per point inside ``/VTKHDF`` instead (history
        at Q4 or H8 points), that array is returned. ``block`` is required
        only when several blocks hold ``name``.
        """
        with self._open() as f:
            candidates = {}
            for group in ("quadrature", "interfaces"):
                if group in f:
                    for b in f[group].keys():
                        node = f[group][b].get(name)
                        if node is not None and hasattr(node, "shape") and len(node.shape) >= 2:
                            candidates.setdefault(b, f"{group}/{b}/{name}")
            blocks = sorted(candidates)
            if block is not None:
                if block not in blocks:
                    raise ResultLoadError(f"No quadrature array {name!r} in block {block!r} of {self.path}")
                blocks = [block]
            if len(blocks) > 1:
                raise ResultLoadError(
                    f"Quadrature array {name!r} exists in blocks {blocks}; pass block=..."
                )
            if blocks:
                s = self.resolve_step(step)
                ds = f[candidates[blocks[0]]]
                if s >= ds.shape[0]:
                    raise ResultLoadError(f"Quadrature array {name!r} has no rows for step {step} in {self.path}")
                return np.asarray(ds[s])
        info = self._arrays.get(name)
        if info is not None and info["attrs"].get("location") == "quadrature_point":
            return self._vtk_field(name, step)
        raise ResultLoadError(f"No quadrature-point array {name!r} in {self.path}")

    # -- mesh, materials, loading, history ------------------------------------------
    def grid(self) -> dict[str, Any] | None:
        """ImageData grid: shape (nx, ny, nz), spacing, origin, direction and array shapes."""
        if self.type != "ImageData":
            return None
        with self._open() as f:
            v = f["VTKHDF"]
            nx, ny, nz = self._grid_shape
            out: dict[str, Any] = {
                "shape": [nx, ny, nz],
                "spacing": np.asarray(v.attrs["Spacing"], float).tolist(),
                "origin": np.asarray(v.attrs["Origin"], float).tolist(),
                "direction": np.asarray(v.attrs.get("Direction", np.eye(3).ravel()), float).tolist(),
                "index_order": "x fastest",
                "point_shape": [nz + 1, ny + 1, nx + 1],
                "cell_shape": [nz, ny, nx],
            }
            if "mesh/grid" in f:
                out.update({k: v for k, v in _attrs(f["mesh/grid"]).items() if k not in out})
        return out

    def mesh(self) -> dict[str, Any]:
        """Mesh summary: ``n_nodes``, ``n_elements``, ``source`` plus ``dim``, ``type``,
        ``cell_types``, ``node_sets`` and, for ImageData, ``grid``."""
        n_nodes, n_elements = self.n_points, self.n_cells
        if self._varying and self._n_steps:                   # the mesh of the last stored step
            part = int(self._part_offsets[self._n_steps - 1])
            n_nodes, n_elements = int(self._part_points[part]), int(self._part_cells[part])
        out: dict[str, Any] = {
            "n_nodes": n_nodes,
            "n_elements": n_elements,
            "source": self.path.name,
            "dim": self.dim,
            "type": self.type,
        }
        if self._varying:
            out["mesh_versions"] = self._n_versions
        with self._open() as f:
            if "mesh/cell_blocks" in f:
                out["cell_types"] = {
                    b: _text(f["mesh/cell_blocks"][b].attrs.get("cell_type")) for b in f["mesh/cell_blocks"].keys()
                }
            if "mesh/node_sets" in f:
                out["node_sets"] = sorted(f["mesh/node_sets"].keys())
        if self._grid_shape is not None:
            out["grid"] = list(self._grid_shape)
        return out

    def step_mesh(self) -> np.ndarray:
        """Mesh version of each stored step (all zero while the mesh does not change)."""
        if not self._varying:
            return np.zeros(self._n_steps, np.int64)
        return np.asarray(self._part_offsets[: self._n_steps], dtype=np.int64)

    def _version_of(self, step: int | None) -> int:
        if not self._varying:
            return 0
        if self._n_steps == 0:
            return 0
        return int(self.step_mesh()[self.resolve_step(-1 if step is None else step)])

    def mesh_arrays(self, step: int | None = None) -> dict[str, Any]:
        """Points (native dimension), cells per block, VTK arrays, sets and graph edges.

        Keys: ``points`` (N, dim), ``connectivity``, ``offsets``, ``types``
        (VTK), ``cells`` {block: (n, k)}, ``cell_types`` {block: name},
        ``node_sets``, ``face_sets``, ``edge_index`` (when stored) and
        ``grid`` for ImageData, which stores no points or cells. With a
        changing mesh (layout 2.1) ``step`` selects the stored step whose
        mesh is returned (default: the last); ``mesh_version`` and, from the
        second version on, the maps ``cell_parent``, ``node_origin`` and
        ``node_parents`` to the previous mesh are added.
        """
        out: dict[str, Any] = {}
        version = self._version_of(step)
        with self._open() as f:
            v = f["VTKHDF"]
            mesh = f["mesh"] if version == 0 else f[f"mesh/versions/{version}"]
            if self.type != "ImageData":
                n_pts, n_cells, n_conn = self._counts(version)
                p0, c0, k0 = (self._first(version, i) for i in range(3))
                points = np.asarray(v["Points"][p0: p0 + n_pts])
                out["points"] = np.ascontiguousarray(points[:, : self.dim]) if self.dim == 2 else points
                conn = np.asarray(v["Connectivity"][k0: k0 + n_conn])
                offsets = np.asarray(v["Offsets"][c0 + version: c0 + version + n_cells + 1], dtype=np.int64)
                out["connectivity"], out["offsets"] = conn, offsets
                out["types"] = np.asarray(v["Types"][c0: c0 + n_cells])
                cells: dict[str, np.ndarray] = {}
                cell_types: dict[str, str] = {}
                if "cell_blocks" in mesh:
                    for b in mesh["cell_blocks"].keys():
                        g = mesh["cell_blocks"][b]
                        start, stop = (int(x) for x in np.asarray(g["cells"][...]).ravel()[:2])
                        lo, hi = int(offsets[start]), int(offsets[stop])
                        width = np.diff(offsets[start: stop + 1])
                        block_conn = conn[lo:hi]
                        if width.size and np.all(width == width[0]):
                            block_conn = block_conn.reshape(-1, int(width[0]))
                        cells[b] = block_conn
                        cell_types[b] = _text(g.attrs.get("cell_type"))
                out["cells"], out["cell_types"] = cells, cell_types
            else:
                out["grid"] = self.grid()
            for group, key in (("node_sets", "node_sets"), ("face_sets", "face_sets")):
                if group in mesh:
                    out[key] = {name: np.asarray(ds[...]) for name, ds in mesh[group].items()}
            if "graph/edge_index" in mesh:
                out["edge_index"] = np.asarray(mesh["graph/edge_index"][...])
            if "cell_blocks" in mesh:
                out["quadrature"] = {}
                for b in mesh["cell_blocks"].keys():
                    g = mesh["cell_blocks"][b]
                    if "quadrature" in g:
                        q = g["quadrature"]
                        out["quadrature"][b] = dict(
                            _attrs(q),
                            **{k: np.asarray(q[k][...]) for k in ("points", "weights") if k in q},
                        )
            if self._varying:
                out["mesh_version"] = version
                out.update(self._version_maps(mesh, version))
        return out

    def _counts(self, version: int) -> tuple[int, int, int]:
        with self._open() as f:
            v = f["VTKHDF"]
            return tuple(int(v[k][version]) for k in ("NumberOfPoints", "NumberOfCells", "NumberOfConnectivityIds"))

    def _first(self, version: int, which: int) -> int:
        """First point (0), cell (1) or connectivity id (2) of a mesh version."""
        if version == 0:
            return 0
        with self._open() as f:
            key = ("NumberOfPoints", "NumberOfCells", "NumberOfConnectivityIds")[which]
            return int(np.sum(np.asarray(f["VTKHDF"][key][:version], dtype=np.int64)))

    @staticmethod
    def _version_maps(mesh, version: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if version:
            for key in ("cell_parent", "node_origin", "node_parents"):
                if key in mesh:
                    out[key] = np.asarray(mesh[key][...])
        if "hanging_nodes" in mesh:
            out["hanging_nodes"] = {k: np.asarray(mesh["hanging_nodes"][k][...]) for k in mesh["hanging_nodes"]}
        return out

    def mesh_versions(self) -> list[dict[str, Any]]:
        """One entry per mesh version: ``n_points``, ``n_cells``, ``first_increment`` and, from version 1 on,
        the maps to the previous mesh and the hanging-node constraints. A fixed mesh has one entry."""
        out = []
        with self._open() as f:
            for version in range(self._n_versions):
                n_pts, n_cells, _ = self._counts(version) if self.type != "ImageData" else (self.n_points, self.n_cells, 0)
                mesh = f["mesh"] if version == 0 else f[f"mesh/versions/{version}"]
                entry = dict(version=version, n_points=n_pts, n_cells=n_cells,
                             first_increment=int(mesh.attrs.get("first_increment", 0)))
                entry.update(self._version_maps(mesh, version))
                out.append(entry)
        return out

    def materials(self) -> dict[str, Any]:
        """``phases`` {id: attributes}, ``phase_id`` and ``orientation`` when stored."""
        out: dict[str, Any] = {"phases": {}}
        with self._open() as f:
            if "materials" not in f:
                return out
            m = f["materials"]
            if "phases" in m:
                for pid in sorted(m["phases"].keys(), key=lambda k: (not k.isdigit(), int(k) if k.isdigit() else 0, k)):
                    attrs = _attrs(m["phases"][pid])
                    if "units" in attrs and isinstance(attrs["units"], str):
                        try:
                            attrs["units"] = json.loads(attrs["units"])
                        except ValueError:
                            pass
                    if "stiffness" in m["phases"][pid]:
                        attrs["stiffness"] = np.asarray(m["phases"][pid]["stiffness"][...])
                    out["phases"][int(pid) if pid.isdigit() else pid] = attrs
        for key in ("phase_id", "orientation"):
            if key in self._arrays:
                out[key] = self.field(key, -1) if self._arrays[key]["static"] else None
        return out

    def boundary_conditions(self) -> dict[str, Any]:
        """``bc`` {name: attributes} and ``periodic`` (attributes or None) from ``/loading``."""
        out: dict[str, Any] = {"bc": {}, "periodic": None}
        with self._open() as f:
            if "loading" not in f:
                return out
            if "bc" in f["loading"]:
                out["bc"] = {k: _attrs(g) for k, g in f["loading/bc"].items()}
            if "periodic" in f["loading"]:
                out["periodic"] = _attrs(f["loading/periodic"])
        return out

    def step_index(self) -> np.ndarray:
        """``history/step_index``: stored step of each converged increment, -1 when none.

        Entries beyond the last complete step of an interrupted file are set to -1.
        """
        with self._open() as f:
            if "history" not in f or "step_index" not in f["history"]:
                return np.zeros(0, np.int64)
            index = np.asarray(f["history/step_index"][...], dtype=np.int64)
        index = index.copy()
        index[index >= self._n_steps] = -1
        return index

    def _history(self) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        columns: dict[str, np.ndarray] = {}
        attrs: dict[str, dict[str, Any]] = {}
        with self._open() as f:
            if "history" not in f:
                return columns, attrs

            def visit(key, obj):
                if hasattr(obj, "shape") and hasattr(obj, "dtype"):
                    columns[key] = np.asarray(obj[...])
                    attrs[key] = _attrs(obj)

            f["history"].visititems(visit)
        return columns, attrs

    def history_columns(self) -> dict[str, np.ndarray]:
        """Every dataset under ``/history`` by its path, e.g. ``reaction/top`` (n_inc, 3)."""
        return self._history()[0]

    def history_attributes(self) -> dict[str, dict[str, Any]]:
        """Attributes of every ``/history`` dataset by its path (``units``, ``method``, ``codes``, ...)."""
        return self._history()[1]

    def history_tables(self) -> dict[str, list[dict[str, Any]]]:
        """Rows of ``/history`` as tables (the form ``Result.history`` returns).

        - ``history``: one row per converged increment with every column of
          ``/history`` outside the long tables. Grouped columns are joined with
          an underscore (``energy/elastic`` -> ``energy_elastic``) and vector
          columns are split by their ``components`` attribute
          (``reaction/top`` -> ``reaction_top_x``, ``reaction_top_y``, ...).
        - ``energy``: ``increment``, ``time`` and each ``energy/<term>`` as ``<term>``.
        - ``crack_tips`` (layout 2.1): one row per tip and increment, the
          position split as ``position_x``, ``position_y``, ``position_z``.
        - ``substeps`` (layout 2.1): one row per damage solve; categorical
          columns are returned as their names.
        """
        columns, attrs = self._history()
        long_tables = ("crack_tips/", "substeps/")
        main = {k: v for k, v in columns.items() if not k.startswith(long_tables)}
        tables: dict[str, list[dict[str, Any]]] = {"history": _rows(_flatten(main, attrs))}
        energy = {k[len("energy/"):]: v for k, v in main.items() if k.startswith("energy/")}
        if energy:
            base = {k: main[k] for k in ("increment", "time") if k in main}
            tables["energy"] = _rows({**base, **energy})
        for name in ("crack_tips", "substeps"):
            prefix = name + "/"
            table = {k[len(prefix):]: v for k, v in columns.items() if k.startswith(prefix)}
            if not table:
                continue
            tattrs = {k[len(prefix):]: v for k, v in attrs.items() if k.startswith(prefix)}
            if name == "substeps":
                table = {k: _decode(v, tattrs.get(k, {})) for k, v in table.items()}
            tables[name] = _rows(_flatten(table, tattrs) if name == "crack_tips" else table)
        return tables

    def crack_tips(self) -> dict[str, Any]:
        """The crack-tip table of a layout-2.1 file: ``increment``, ``tip_id``, ``position`` (n, dim), ``threshold``,
        ``time`` of each row and ``speed``. The speed is read from the file when it has been added at close;
        otherwise it is computed with the same fit and ``speed_source`` is ``"computed"`` (else ``"file"``).
        Raises ``ResultLoadError`` when the file has no crack tips."""
        with self._open() as f:
            if "history/crack_tips" not in f:
                raise ResultLoadError(f"{self.path} has no history/crack_tips table")
            g = f["history/crack_tips"]
            inc = np.asarray(g["increment"][...], np.int64)
            tip = np.asarray(g["tip_id"][...], np.int64)
            position = np.asarray(g["position"][...])
            threshold = np.asarray(g["threshold"][...])
            hinc = np.asarray(f["history/increment"][...], np.int64)
            time_of = dict(zip(hinc.tolist(), np.asarray(f["history/time"][...], float).tolist()))
            time = np.array([time_of.get(int(i), np.nan) for i in inc])
            out = {"increment": inc, "tip_id": tip, "position": position[:, : self.dim], "threshold": threshold,
                   "time": time, "position_attributes": _attrs(g["position"])}
            if "speed" in g:
                out["speed"] = np.asarray(g["speed"][...])[: inc.size]
                out["speed_source"] = "file"
                out["speed_fit"] = _attrs(g["speed"])
        if "speed" not in out:
            from .crack_tips import FIT_METHOD, FIT_POLYORDER, FIT_WINDOW, fit_speed
            out["speed"] = fit_speed(tip, position, time)
            out["speed_source"] = "computed"
            out["speed_fit"] = {"fit_method": FIT_METHOD, "fit_window": FIT_WINDOW, "fit_polyorder": FIT_POLYORDER}
        return out

    def validation(self) -> list[dict[str, Any]]:
        """Rows of ``meta/validation`` (layout 2.1): ``name``, ``value``, ``tolerance``, ``passed`` and, when stored,
        ``units`` and ``description``; an empty list when the file has none."""
        with self._open() as f:
            if "meta/validation" not in f:
                return []
            g = f["meta/validation"]
            cols = {k: np.asarray(g[k][...]) for k in g}
            n = len(cols["name"]) if "name" in cols else 0
            return [{k: (_text(v[i]) if v.dtype.kind in "OSU" else v[i].item()) for k, v in cols.items()} for i in range(n)]

    def initial_conditions(self) -> dict[str, np.ndarray]:
        """Arrays of ``initial_conditions/`` (layout 2.1) by name, in native dimension for 2D vectors; empty when none."""
        out: dict[str, np.ndarray] = {}
        with self._open() as f:
            if "initial_conditions" not in f:
                return out
            for name, ds in f["initial_conditions"].items():
                out[name] = self._native(np.asarray(ds[...]), _attrs(ds))
        return out

    def __repr__(self) -> str:
        return f"ResultFile(path={str(self.path)!r})"


def _flatten(columns: dict[str, np.ndarray], attrs: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    flat: dict[str, np.ndarray] = {}
    for key, values in columns.items():
        base = key.replace("/", "_")
        if values.ndim == 1:
            flat[base] = values
            continue
        comps = _components(attrs.get(key, {}))
        values2 = values.reshape(values.shape[0], -1)
        if len(comps) != values2.shape[1]:
            comps = [str(i) for i in range(values2.shape[1])]
        for i, comp in enumerate(comps):
            flat[f"{base}_{comp}"] = values2[:, i]
    return flat


def _decode(values: np.ndarray, attrs: dict[str, Any]) -> np.ndarray:
    """Names of a categorical column (``codes`` attribute, a JSON object code -> name); other columns unchanged."""
    raw = attrs.get("codes")
    if raw is None:
        return values
    try:
        names = {int(k): v for k, v in json.loads(raw).items()}
    except (TypeError, ValueError):
        return values
    return np.array([names.get(int(c), "") for c in values], dtype=object)


def _rows(columns: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    if not columns:
        return []
    n = max(len(v) for v in columns.values())
    rows = []
    for i in range(n):
        rows.append({k: ((v[i].item() if hasattr(v[i], "item") else v[i]) if i < len(v) else None)
                     for k, v in columns.items()})
    return rows
