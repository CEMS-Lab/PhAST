"""Writer for the PhAST result file, layout 2.0.

One file per run, appended one converged increment at a time in the write order
of the specification (``docs/user_guide/result_file_layout.md``, section 4):
field rows, ``Steps`` offsets and values, ``NSteps``, ``history/`` rows, flush.
A run killed at any point opens with its previous complete steps.

Typical use::

    mesh = UnstructuredMesh(points, {"T3": cells}, node_sets={"top": top})
    fields = [Field("displacement", "node", "x,y", "mm", "state"),
              Field("damage", "node", "d", "1", "state"),
              Field("H", "cell", "H", "MPa", "history")]
    with ResultWriter(run_dir / "result.vtkhdf", mesh, fields, dim=2,
                      analysis="plane_strain", run_id="sent-001") as w:
        for inc in range(n):
            ...
            w.append(dict(increment=inc, time=t, dt=dt, load_parameter=u_top,
                          damage_solved=1), fields=dict(displacement=u, damage=d, H=H))

The file is written as ``<name>.partial`` and renamed after it closes and
validates; ``<name>.sha256`` holds its checksum.

The writer writes version 2.0.0. The first use of an item added in layout 2.1
(crack-tip and substep tables, velocity conditions, initial conditions, wave
speeds in the material table, validation results, a changing mesh) moves the
file to 2.1.0 before that item is written: the root version attribute and
``meta/schema_yaml`` change together, so a file killed afterwards is still
checked by the rules of its own version.
"""
from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import os
import platform
import signal
import sys
import warnings
from dataclasses import dataclass, field as _field
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import numpy as np

from .cells import cell_info, edge_index
from .schema import SCHEMA_NAME, SCHEMA_VERSION, SCHEMA_VERSION_21, load_schema, schema_text

LIBVER = ("earliest", "v114")
TIME_LIMIT_EXIT_CODE = 11
_SYNCED_MARKERS = ("CloudStorage", "OneDrive", "Dropbox", "Mobile Documents", "Google Drive")
_STATIC_OFFSETS = ("PartOffsets", "NumberOfParts", "PointOffsets", "CellOffsets", "ConnectivityIdOffsets")

# Units of history columns whose meaning fixes them (unit system mm-MPa-N-s).
_DEFAULT_UNITS = {
    "increment": "1", "step_index": "1", "damage_solved": "1", "time": "s", "dt": "s", "load_factor": "1",
    "cutbacks": "1", "stagger_iterations": "1", "mechanics_iterations": "1", "residual": "1",
}
_GROUP_UNITS = {"energy": "mJ", "reaction": "N", "macro_stress": "MPa", "macro_strain": "1", "mean_strain": "1"}
_INT8_COLUMNS = {"damage_solved"}
_DEFAULT_UNITS["mesh_index"] = "1"
# long tables of layout 2.1 (history/<table>/<column>)
_TABLE_UNITS = {
    "crack_tips": {"increment": "1", "tip_id": "1", "position": "mm", "threshold": "1", "speed": "mm/s"},
    "substeps": {"increment": "1", "call": "1", "damage_increment": "1", "cg_iterations": "1", "cg_residual": "1",
                 "cg_rhs_norm": "1", "cg_initial_residual": "1", "cg_residual_reference": "1",
                 "cg_relative_residual": "1", "kkt_max": "1", "relative_kkt_max": "1",
                 "projected_relative_residual": "1", "projected_relative_residual_max": "1", "damage_min": "1",
                 "damage_max": "1", "min_increment": "1", "walltime_s": "s", "cooldown_remaining": "1",
                 "retry_counter": "1"},
}
for _name in ("cg_converged", "bounds_satisfied", "irreversibility_satisfied", "amg_eligible", "rebuild_attempted",
              "rebuild_succeeded", "requested_preconditioner", "effective_preconditioner", "amg_skip_reason",
              "fallback_target", "fallback_reason"):
    _TABLE_UNITS["substeps"][_name] = "1"
_FLAG_COLUMNS = {"cg_converged", "bounds_satisfied", "irreversibility_satisfied", "amg_eligible", "rebuild_attempted",
                 "rebuild_succeeded"}
_ONE_D_MESSAGE = "one-dimensional runs (line cells) are outside layout 2.x: dim must be 2 or 3"
_RESUMABLE_STATUS = ("running", "time_limited", "failed", "recovered")   # a file closed "complete" is never reopened
_TABLE_GROUPS = ("crack_tips", "substeps")                                  # long tables below history/


class ResumeRefused(RuntimeError):
    """A result file cannot be resumed as asked; the file and its checksum are left unchanged."""


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def _array(x: Any) -> np.ndarray:
    """NumPy view of a NumPy array, torch tensor or sequence (detached, on the CPU)."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _is_synced(path: Path) -> bool:
    return any(part == m or part.startswith(m + "-") for part in path.resolve().parts for m in _SYNCED_MARKERS)


@dataclass
class Field:
    """Declaration of one stored array.

    ``location``: ``node``, ``cell``, ``quadrature_point`` (per-point values of
    a cell array), ``interface_point``. ``components``: comma-separated names
    in stored order (2D vectors are declared ``x,y`` and stored with ``z = 0``).
    ``kind``: ``state``, ``history``, ``derived``.
    ``placement``: ``view`` (inside ``/VTKHDF``, every stored step), ``derived``
    (``/derived/<name>``, any increments), ``quadrature:<block>`` or
    ``interface:<name>`` (raw per-point arrays at every stored step).
    ``static``: written once with the mesh (for example ``phase_id``); give ``data``.
    """

    name: str
    location: str
    components: str
    units: str
    kind: str
    placement: str = "view"
    dtype: Any = None
    static: bool = False
    data: Any = None
    attrs: dict = _field(default_factory=dict)

    def __post_init__(self):
        if "/" in self.name or "." in self.name:
            raise ValueError(f"field name {self.name!r} contains '/' or '.'")
        if self.kind not in ("state", "history", "derived"):
            raise ValueError(f"field {self.name!r}: unknown kind {self.kind!r}")
        if self.dtype is None:
            self.dtype = np.float64
        self.dtype = np.dtype(self.dtype)
        if self.kind in ("state", "history") and self.name != "phase_id" and self.dtype != np.float64:
            raise ValueError(f"field {self.name!r}: {self.kind} fields are float64 (got {self.dtype})")
        if self.name == "phase_id":
            self.dtype = np.dtype(np.uint8)

    @property
    def key(self) -> str:
        """Name used in ``append(fields=...)``: ``<block>/<name>`` for per-point and interface arrays."""
        if self.placement.startswith(("quadrature:", "interface:")):
            return f"{self.placement.split(':', 1)[1]}/{self.name}"
        return self.name

    @property
    def n_components(self) -> int:
        return len(self.components.split(","))

    @property
    def is_vector(self) -> bool:
        return self.components in ("x,y", "x,y,z")


@dataclass
class UnstructuredMesh:
    """Mesh of an UnstructuredGrid run. ``blocks`` maps a cell type to its cells (VTK node order).

    ``quadrature`` maps a block to ``(points, weights, rule, point_order)``.
    ``permutation`` maps a block to a node permutation applied to bring the
    solver's node order to VTK order (``cells[:, permutation]``).
    ``hanging`` holds the constraints of a non-conforming mesh (layout 2.1): the
    value at each constrained node is the weighted sum of its master nodes.
    """

    points: Any
    blocks: dict
    node_sets: dict = _field(default_factory=dict)
    face_sets: dict = _field(default_factory=dict)
    quadrature: dict = _field(default_factory=dict)
    permutation: dict = _field(default_factory=dict)
    graph: bool = True
    hanging: dict | None = None      # {"constrained_nodes": (n,), "master_nodes": (n, m), "weights": (n, m)}

    kind = "UnstructuredGrid"

    def __post_init__(self):
        pts = _array(self.points).astype(np.float64)
        if pts.ndim == 1 or (pts.ndim == 2 and pts.shape[1] == 1):
            raise ValueError(f"points of shape {pts.shape}: {_ONE_D_MESSAGE}")
        if pts.ndim != 2 or pts.shape[1] not in (2, 3):
            raise ValueError(f"points must be (N, 2) or (N, 3), got {pts.shape}")
        if pts.shape[1] == 2:
            pts = np.concatenate([pts, np.zeros((len(pts), 1))], axis=1)
        self.points = pts
        blocks = {}
        for ctype, cells in self.blocks.items():
            info = cell_info(ctype)
            cells = _array(cells).astype(np.int64)
            if ctype in self.permutation:
                cells = cells[:, np.asarray(self.permutation[ctype])]
            if cells.ndim != 2 or cells.shape[1] != info["nodes"]:
                raise ValueError(f"block {ctype}: cells must be (E, {info['nodes']}), got {cells.shape}")
            blocks[ctype] = cells
        self.blocks = blocks

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def n_cells(self) -> int:
        return sum(len(c) for c in self.blocks.values())


@dataclass
class ImageGrid:
    """Regular voxel grid of an ImageData run: ``shape = (nx, ny, nz)`` cells, x fastest.

    Arrays may be passed flat (x fastest, VTK order) or shaped ``(nz, ny, nx[, c])``
    for cells and ``(nz+1, ny+1, nx+1[, c])`` for points.
    """

    shape: tuple
    spacing: tuple
    origin: tuple = (0.0, 0.0, 0.0)
    cell_type: str = "H8"
    quadrature: dict = _field(default_factory=dict)
    node_sets: dict = _field(default_factory=dict)
    face_sets: dict = _field(default_factory=dict)

    kind = "ImageData"

    def __post_init__(self):
        self.shape = tuple(int(n) for n in self.shape)
        if len(self.shape) != 3 or min(self.shape) < 1:
            raise ValueError(f"grid shape must be (nx, ny, nz) with every entry >= 1, got {self.shape}")

    @property
    def n_points(self) -> int:
        nx, ny, nz = self.shape
        return (nx + 1) * (ny + 1) * (nz + 1)

    @property
    def n_cells(self) -> int:
        nx, ny, nz = self.shape
        return nx * ny * nz

    def grid_shape(self, where: str) -> tuple:
        nx, ny, nz = self.shape
        return (nz + 1, ny + 1, nx + 1) if where == "point" else (nz, ny, nx)


def estimate_size(mesh: UnstructuredMesh | ImageGrid, fields: Iterable[Field], n_steps: int) -> dict:
    """Estimated bytes of the time-indexed arrays over ``n_steps`` stored steps, split by kind."""
    out = {"primary": 0, "derived": 0}
    for f in fields:
        if f.static or f.placement == "derived":
            continue
        n = mesh.n_points if f.location == "node" else mesh.n_cells
        ncomp = 3 if f.is_vector else f.n_components
        out["derived" if f.kind == "derived" else "primary"] += n * ncomp * f.dtype.itemsize * n_steps
    out["total"] = out["primary"] + out["derived"]
    return out


class SoftStop:
    """Signal handler that only sets a flag; the run loop finishes the increment and stops."""

    def __init__(self, signals: tuple = (signal.SIGTERM,) + ((signal.SIGUSR1,) if hasattr(signal, "SIGUSR1") else ())) -> None:
        self.requested = False
        self.signum = None
        self._previous = {}
        for s in signals:
            try:
                self._previous[s] = signal.signal(s, self._handler)
            except (ValueError, OSError):          # not the main thread, or unsupported signal
                pass

    def _handler(self, signum, frame):
        self.requested = True
        self.signum = signum

    def restore(self) -> None:
        for s, h in self._previous.items():
            signal.signal(s, h)
        self._previous = {}


class ResultWriter:
    """Append-only writer of one layout-2.0 result file. See the module docstring."""

    def __init__(
        self,
        path: str | os.PathLike,
        mesh: UnstructuredMesh | ImageGrid,
        fields: Iterable[Field] = (),
        *,
        dim: int,
        analysis: str,
        run_id: str,
        profile: str = "run",
        history_coupling: str = "fresh",
        history_update: str = "hard_max",
        history_update_beta: float | None = None,
        kinematics: str = "small",
        parent_run_id: str | None = None,
        materials: Mapping[Any, Mapping[str, Any]] | None = None,
        bcs: Mapping[str, Mapping[str, Any]] | None = None,
        periodic: Mapping[str, Any] | None = None,
        history_units: Mapping[str, str] | None = None,
        history_attrs: Mapping[str, Mapping[str, Any]] | None = None,
        config: Any = None,
        meta: Mapping[str, Any] | None = None,
        expected_steps: int | None = None,
        size_limit_gb: float = 4.0,
        sidecars: bool = True,
        validate_on_close: bool = True,
        mesh_varies: bool = False,
    ):
        self.final_path = Path(path)
        if self.final_path.suffix != ".vtkhdf":
            raise ValueError(f"result files end in .vtkhdf got {self.final_path.name}")
        self.path = self.final_path.with_name(self.final_path.name + ".partial")
        if _is_synced(self.final_path):
            warnings.warn(f"{self.final_path.parent} looks like a synchronised folder; write results on local "
                          "disk and copy the closed file (writer rule 1 of the layout specification)", stacklevel=2)
        if int(dim) not in (2, 3):
            raise ValueError(f"dim = {dim!r}: {_ONE_D_MESSAGE}" if int(dim) == 1 else f"dim must be 2 or 3, got {dim!r}")
        self.mesh = mesh
        self.image = mesh.kind == "ImageData"
        self.mesh_varies = bool(mesh_varies)
        if self.mesh_varies and self.image:
            raise ValueError("mesh_varies needs an UnstructuredMesh; an ImageData grid has one fixed shape")
        # items added in layout 2.1 that the constructor arguments already use
        self.features_21: list[str] = []
        if self.mesh_varies:
            self.features_21.append("mesh_varies")
        if any(str(a.get("kind")) == "velocity" for a in (bcs or {}).values()):
            self.features_21.append("velocity_bc")
        if any(k in a for a in (materials or {}).values() for k in ("c_p", "c_s", "c_R")):
            self.features_21.append("wave_speeds")
        if getattr(mesh, "hanging", None):
            self.features_21.append("hanging_nodes")
        self.version = SCHEMA_VERSION_21 if self.features_21 else SCHEMA_VERSION
        schema = load_schema(self.version)
        self.schema = schema
        self.dim = int(dim)
        self.profile = profile
        self.history_units = dict(history_units or {})
        self.history_attrs = {k: dict(v) for k, v in (history_attrs or {}).items()}
        self.sidecars = sidecars
        self.validate_on_close = validate_on_close
        self.n_steps = 0
        self.n_rows = 0
        self.closed = False
        self.report = None
        self._columns: dict[str, h5py.Dataset] | None = None
        self._csv = None
        self._tables: dict[str, dict] = {}            # 2.1 long tables: name -> columns, row count, category codes
        self._validation_written = False
        self.n_versions = 1
        self.cur_version = 0

        fields = list(fields)
        names = [f.key for f in fields]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate fields: {sorted(n for n in names if names.count(n) > 1)}")
        self.size_estimate = None
        moved = []
        if expected_steps:
            self.size_estimate = estimate_size(mesh, fields, int(expected_steps))
            if self.size_estimate["total"] > size_limit_gb * 1e9:
                for f in fields:
                    if f.kind == "derived" and f.placement == "view" and not f.static:
                        f.placement = "derived"
                        moved.append(f.name)
        self.moved_out_of_view = moved
        self._check_history_variables(fields, history_coupling)
        if self.mesh_varies:
            odd = [f.key for f in fields if not f.static and f.placement != "view"]
            if odd:
                raise ValueError(f"fields {odd} are stored outside /VTKHDF with a fixed shape; with mesh_varies every "
                                 "field is stored inside /VTKHDF (layout 2.1, section 8.6)")
        self.fields = {f.key: f for f in fields}

        root = dict(phast_schema=SCHEMA_NAME, phast_schema_version=self.version, profile=profile, dim=self.dim,
                    analysis=analysis, unit_system="mm-MPa-N-s", status="running", created=_now(), run_id=str(run_id),
                    history_coupling=history_coupling, history_update=history_update, kinematics=kinematics,
                    mesh_varies=self.mesh_varies, index_basis="increment")
        for key, val in (("parent_run_id", parent_run_id), ("history_update_beta", history_update_beta)):
            if val is not None:
                root[key] = val
        for key, rule in schema["root_attributes"]["required"].items():
            if "values" in rule and root[key] not in rule["values"]:
                raise ValueError(f"{key} = {root[key]!r}; allowed: {rule['values']}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(self.path, "w", libver=LIBVER, track_order=True, locking="best-effort")
        try:
            for k, v in root.items():
                self.f.attrs[k] = v
            self._create_vtkhdf()
            self._create_groups(materials or {}, bcs or {}, periodic, config, meta or {})
            self.f.flush()
        except Exception:
            self.f.close()
            raise

    # --- resume -----------------------------------------------------------------

    @classmethod
    def resume(cls, path: str | os.PathLike, mesh: UnstructuredMesh | ImageGrid, fields: Iterable[Field] = (), *,
               upto_increment: int, run_id: str | None = None,
               history_units: Mapping[str, str] | None = None, history_attrs: Mapping[str, Mapping[str, Any]] | None = None,
               sidecars: bool = True, validate_on_close: bool = True, backup: bool = True) -> "ResultWriter":
        """Reopen an interrupted file and continue appending after the increment ``upto_increment``.

        ``path`` is the final name (``<name>.vtkhdf``). The source is ``<name>.vtkhdf.partial`` (killed run) or the
        closed ``<name>.vtkhdf`` with status ``time_limited``, ``failed`` or ``recovered``; a file closed ``complete``,
        a changing mesh (``mesh_varies``), a different ``run_id`` or mesh, a field list that differs from the file, a
        checkpoint increment that is not in the file, or stored steps missing below it are refused with
        :class:`ResumeRefused` before anything is changed. ``mesh`` and ``fields`` are those of the first run (the
        caller declares them the same way); fields that the first run moved out of the view by the size rule are moved
        again from the file's ``derived_outside_view`` record.

        Every history row after ``upto_increment`` and every stored step that only those rows refer to is cut
        (the run recomputes them from its checkpoint); the long tables follow. The file is renamed to ``.partial``,
        its status becomes ``running``, ``finished`` (and ``status_reason``) are removed and the resume is recorded:
        root attribute ``resume_count`` and the JSON list ``resumes`` in the attributes of ``meta`` (time,
        increment, rows and steps kept and cut, previous status and SHA-256, job number). With ``backup`` the bytes
        before the cut are kept as ``<file>.pre-resume-<k>.bak``. The result of the finished run equals an
        uninterrupted one in every stored array and history column (the wall-clock columns are measurements).
        """
        final = Path(path)
        if final.suffix != ".vtkhdf":
            raise ValueError(f"result files end in .vtkhdf, got {final.name}")
        partial = final.with_name(final.name + ".partial")
        if partial.exists() and final.exists():
            raise ResumeRefused(f"{partial.name} and {final.name} both exist; which one is the run is for a person to decide")
        src = partial if partial.exists() else final
        if not src.exists():
            raise ResumeRefused(f"no result file to resume: neither {final.name} nor {partial.name} exists in {final.parent}")
        fields = list(fields)
        upto = int(upto_increment)

        # ---- read-only checks: nothing is changed unless every one passes
        try:
            fr = h5py.File(src, "r", libver=LIBVER, locking="best-effort")
        except OSError as exc:
            raise ResumeRefused(f"cannot open {src.name} for reading: {exc}") from exc
        try:
            with fr:
                checked = cls._check_resumable(fr, src, mesh, fields, upto, run_id)
        except (KeyError, ValueError, OSError, IndexError) as exc:       # a file cut in the middle of a write
            raise ResumeRefused(f"{src.name} is damaged or incomplete and cannot be resumed: {type(exc).__name__}: {exc}") from exc
        keep, n, moved, prior_rows, prior_steps, prior_status, version = checked
        n_backup = 1
        while backup and src.with_name(f"{src.name}.pre-resume-{n_backup}.bak").exists():
            n_backup += 1
        return cls._resume_apply(final, partial, src, mesh, fields, upto, keep, n, moved, prior_rows, prior_steps,
                                 prior_status, version, n_backup, backup, history_units, history_attrs, sidecars,
                                 validate_on_close)

    @classmethod
    def _check_resumable(cls, fr, src, mesh, fields, upto, run_id):
        """The read-only part of :meth:`resume`; returns what the change needs or raises :class:`ResumeRefused`."""
        from .recover import _consistent_steps
        a = fr.attrs
        text = lambda v: v.decode() if isinstance(v, bytes) else str(v)
        if text(a.get("phast_schema", "")) != SCHEMA_NAME or "VTKHDF" not in fr or "history" not in fr:
            raise ResumeRefused(f"{src.name} is not a PhAST result file")
        if text(a.get("profile", "")) != "run":
            raise ResumeRefused(f"profile {text(a.get('profile'))!r}: only a run file has increments to resume")
        status = text(a.get("status", ""))
        if status == "complete":
            raise ResumeRefused(f"{src.name} is closed complete: there is nothing to resume")
        if status not in _RESUMABLE_STATUS:
            raise ResumeRefused(f"status {status!r}: resumable states are {list(_RESUMABLE_STATUS)}")
        if bool(a.get("mesh_varies", False)):
            raise ResumeRefused("a file with a changing mesh (mesh_varies) cannot be resumed: the mesh versions of "
                                "the cut steps are not restored")
        if run_id is not None and text(a.get("run_id", "")) != str(run_id):
            raise ResumeRefused(f"run identity differs: the file is run {text(a.get('run_id'))!r}, not {run_id!r}")
        v = fr["VTKHDF"]
        image = mesh.kind == "ImageData"
        if text(v.attrs.get("Type", b"")) != mesh.kind:
            raise ResumeRefused(f"mesh differs: the file holds {text(v.attrs.get('Type'))}, the run {mesh.kind}")
        if image:
            nx, ny, nz = mesh.shape
            same = (np.array_equal(v.attrs["WholeExtent"], [0, nx, 0, ny, 0, nz])
                    and np.array_equal(v.attrs["Spacing"], np.asarray(mesh.spacing, float))
                    and np.array_equal(v.attrs["Origin"], np.asarray(mesh.origin, float)))
        else:
            same = (v["Points"].shape == mesh.points.shape and np.array_equal(v["Points"][...], mesh.points)
                    and int(v["NumberOfCells"][0]) == mesh.n_cells)
        if not same:
            raise ResumeRefused("mesh differs from the mesh of the file")
        moved = [x for x in text(fr["meta"].attrs.get("derived_outside_view", "")).split(",") if x]
        for f_ in fields:
            if f_.name in moved and f_.kind == "derived" and f_.placement == "view" and not f_.static:
                f_.placement = "derived"
        expect = set()
        for f_ in fields:
            where = "PointData" if f_.location == "node" else "CellData"
            if f_.static or f_.placement == "view":
                expect.add(f"VTKHDF/{where}/{f_.name}")
            elif f_.placement == "derived":
                expect.add(f"derived/{f_.name}/values")
            elif f_.placement.startswith("quadrature:"):
                expect.add(f"quadrature/{f_.placement.split(':', 1)[1]}/{f_.name}")
            elif f_.placement.startswith("interface:"):
                expect.add(f"interfaces/{f_.placement.split(':', 1)[1]}/{f_.name}")
        have = set()
        for grp in ("VTKHDF/PointData", "VTKHDF/CellData", "quadrature", "interfaces", "derived"):
            if grp in fr:
                fr[grp].visititems(lambda n, o, grp=grp: have.add(f"{grp}/{n}") if isinstance(o, h5py.Dataset) else None)
        have = {x for x in have if not x.endswith("/step_index")}
        if grp_missing := sorted(expect - have):
            raise ResumeRefused(f"the file lacks the declared arrays {grp_missing[:4]}: the fields differ from the first run")
        if extra := sorted(have - expect):
            raise ResumeRefused(f"the file holds arrays that are not declared: {extra[:4]}")
        inc = np.asarray(fr["history/increment"][...], np.int64)
        where_ = np.flatnonzero(inc == upto)
        if where_.size != 1:
            raise ResumeRefused(f"increment {upto} is not in the file (it holds increments "
                                f"{int(inc.min()) if inc.size else None} to {int(inc.max()) if inc.size else None}, "
                                f"{inc.size} rows): the checkpoint is ahead of the stored rows")
        keep = int(where_[0]) + 1
        si = np.asarray(fr["history/step_index"][:keep], np.int64)
        n = int(si.max()) + 1 if (si >= 0).any() else 0
        nsteps_file, consistent = _consistent_steps(fr)
        if consistent < n:
            raise ResumeRefused(f"only {consistent} complete stored steps are in the file, the rows up to increment "
                                f"{upto} refer to {n}")
        prior_rows, prior_steps, prior_status = int(inc.size), int(nsteps_file), status
        version = text(a["phast_schema_version"])
        return keep, n, moved, prior_rows, prior_steps, prior_status, version

    @classmethod
    def _resume_apply(cls, final, partial, src, mesh, fields, upto, keep, n, moved, prior_rows, prior_steps,
                      prior_status, version, n_backup, backup, history_units, history_attrs, sidecars, validate_on_close):
        """The changing part of :meth:`resume`: backup, rename to ``.partial``, cut, restore the writer's state."""
        from .recover import _clear_flag, _trim_history, trim_steps
        import shutil
        bak = src.with_name(f"{src.name}.pre-resume-{n_backup}.bak") if backup else None
        previous_sha = sha256_file(src)
        sidecar = final.with_name(final.name + ".sha256")
        sidecar_text = sidecar.read_text(encoding="utf-8") if sidecar.exists() else None
        if bak is not None:
            shutil.copyfile(src, bak)
        w = cls.__new__(cls)
        w.f = None
        try:
            if src == final:
                os.replace(final, partial)
                if sidecar_text is not None:
                    sidecar.unlink()
            _clear_flag(partial, None)
            w.final_path, w.path = final, partial
            w.f = h5py.File(partial, "r+", libver=LIBVER, locking="best-effort")
            w._restore(mesh, fields, keep, n, version, moved, history_units, history_attrs, sidecars, validate_on_close,
                       _trim_history, trim_steps)
            root = w.f.attrs
            resume_no = int(root.get("resume_count", 0)) + 1
            for k in ("finished", "status_reason", "recovered_from_steps"):
                if k in root:
                    del root[k]
            root["status"] = "running"
            root["resume_count"] = np.int64(resume_no)
            meta = w.f["meta"]
            log = json.loads(meta.attrs["resumes"]) if "resumes" in meta.attrs else []
            log.append(dict(resumed_at=_now(), from_increment=upto, kept_rows=keep, kept_steps=n,
                            cut_rows=prior_rows - keep, cut_steps=prior_steps - n, previous_status=prior_status,
                            previous_sha256=previous_sha, source=src.name, backup=bak.name if bak else None,
                            slurm_job_id=os.environ.get("SLURM_JOB_ID", ""), machine=platform.node()))
            meta.attrs["resumes"] = json.dumps(log)
            w.f.flush()
        except BaseException:
            try:
                if w.f is not None:
                    w.f.close()
            except Exception:
                pass
            if bak is not None and bak.exists():                      # put the original bytes back under the original name
                for p_ in (partial, final):
                    if p_.exists() and p_ != src:
                        p_.unlink()
                shutil.copyfile(bak, src)
                if sidecar_text is not None:
                    sidecar.write_text(sidecar_text, encoding="utf-8")
            raise
        return w

    def _restore(self, mesh, fields, keep, n, version, moved, history_units, history_attrs, sidecars, validate_on_close,
                 _trim_history, trim_steps) -> None:
        """Cut the open file to ``keep`` history rows and ``n`` stored steps and rebuild the writer's handles on it."""
        f = self.f
        trimmed: list[str] = []
        trim_steps(f, n, trimmed)
        _trim_history(f, n, trimmed, rows=keep)
        if "history/crack_tips/speed" in f:
            del f["history/crack_tips/speed"]                         # derived at close from the rows that remain
        self.mesh, self.image, self.mesh_varies = mesh, mesh.kind == "ImageData", False
        self.version = version
        self.schema = load_schema(version)
        meta = f["meta"]
        self.features_21 = [x for x in (meta.attrs.get("layout_2_1_features", "") or "").split(",") if x]
        self.dim, self.profile = int(f.attrs["dim"]), "run"
        self.history_units = dict(history_units or {})
        self.history_attrs = {k: dict(x) for k, x in (history_attrs or {}).items()}
        self.sidecars, self.validate_on_close = sidecars, validate_on_close
        self.n_steps, self.n_rows, self.closed, self.report = n, keep, False, None
        self.size_estimate, self.moved_out_of_view = None, list(moved)
        self.n_versions, self.cur_version = 1, 0
        self._ver_off = [(0, 0, 0)]
        if not self.image:
            self._geom_total = [mesh.n_points, mesh.n_cells, len(self._geometry_arrays(mesh)[0])]
        self.fields = {fl.key: fl for fl in fields}
        v = f["VTKHDF"]
        self.pd, self.cd, self.steps = v["PointData"], v["CellData"], v["Steps"]
        self.values = self.steps["Values"]
        self.static_offsets = {k: self.steps[k] for k in _STATIC_OFFSETS}
        pdo, cdo = self.steps["PointDataOffsets"], self.steps["CellDataOffsets"]
        self._rows_done = {"point": n if self.image else n * mesh.n_points, "cell": n if self.image else n * mesh.n_cells}
        self.series, self.statics, self.outside, self.derived = [], [], [], {}
        for fl in self.fields.values():
            where = "point" if fl.location == "node" else "cell"
            grp, ogrp = (self.pd, pdo) if where == "point" else (self.cd, cdo)
            if fl.static:
                self.statics.append((fl, grp[fl.name], ogrp[fl.name], where))
            elif fl.placement == "view":
                self.series.append((fl, grp[fl.name], ogrp[fl.name]))
        self._static_offsets = {fl.name: [0] for fl, *_ in self.statics}
        self.block_ranges = {mesh.cell_type: (0, mesh.n_cells)} if self.image else self._block_ranges(mesh)
        for fl in self.fields.values():
            if fl.static or fl.placement == "view":
                continue
            if fl.placement == "derived":
                g = f[f"derived/{fl.name}"]
                self.derived[fl.name] = (fl, g["values"], g["step_index"])
            elif fl.placement.startswith("quadrature:"):
                self.outside.append((fl, f[f"quadrature/{fl.placement.split(':', 1)[1]}/{fl.name}"]))
            elif fl.placement.startswith("interface:"):
                self.outside.append((fl, f[f"interfaces/{fl.placement.split(':', 1)[1]}/{fl.name}"]))
        h = f["history"]
        cols: dict[str, h5py.Dataset] = {}
        h.visititems(lambda name, o: cols.__setitem__(name, o)
                     if isinstance(o, h5py.Dataset) and name.split("/", 1)[0] not in _TABLE_GROUPS else None)
        self._columns = cols or None
        self._tables = {}
        for table in _TABLE_GROUPS:
            if table not in h:
                continue
            tcols: dict[str, h5py.Dataset] = {}
            h[table].visititems(lambda name, o: tcols.__setitem__(name, o) if isinstance(o, h5py.Dataset) else None)
            count = min((d.shape[0] for d in tcols.values()), default=0)
            codes = {}
            for cname, ds in tcols.items():
                if "codes" in ds.attrs:
                    codes[cname] = {t: int(c) for c, t in json.loads(ds.attrs["codes"]).items()}
            self._tables[table] = dict(columns=tcols, n=count, codes=codes)
        self._validation_written = "validation" in meta
        self._csv = None
        if sidecars:
            hist = self.final_path.parent / "history.csv"
            if hist.exists():
                with hist.open(newline="", encoding="utf-8") as fh:
                    rows = list(csv.reader(fh))
                if rows:
                    self._csv = rows[0]
                    with hist.open("w", newline="", encoding="utf-8") as fh:
                        csv.writer(fh).writerows(rows[: 1 + keep])

    # --- creation ---------------------------------------------------------------

    def _check_history_variables(self, fields, coupling):
        names = {f.name for f in fields}
        if coupling == "lagged" and "H" in names and "H_damage_input" not in names:
            raise ValueError("history_coupling = lagged requires the field H_damage_input (layout section 2.4)")
        for f in fields:
            if f.placement.startswith(("quadrature:", "interface:")) and f.kind == "history" and f.location != "interface_point":
                raise ValueError(f"history field {f.name!r} per point belongs inside /VTKHDF as a multi-component "
                                 "cell array (layout section 2.3)")

    def _resizable(self, group, name, tail, dtype, chunk_rows, attrs=None):
        ds = group.create_dataset(name, shape=(0,) + tuple(tail), maxshape=(None,) + tuple(tail), dtype=dtype,
                                  chunks=(max(1, chunk_rows),) + tuple(tail), track_times=False)
        for k, v in (attrs or {}).items():
            ds.attrs[k] = v
        return ds

    def _field_attrs(self, f: Field) -> dict:
        a = dict(location=f.location, components=("x,y,z" if f.is_vector else f.components), units=f.units, kind=f.kind)
        a.update(f.attrs)
        return a

    def _shape_tail(self, f: Field, where: str) -> tuple:
        ncomp = 3 if f.is_vector else f.n_components
        return () if ncomp == 1 else (ncomp,)

    def _create_vtkhdf(self):
        v = self.f.create_group("VTKHDF", track_order=True)
        v.attrs["Version"] = np.array([2, 2])
        m = self.mesh
        self._ver_off = [(0, 0, 0)]                # first point, cell and connectivity id of each mesh version
        if self.image:
            v.attrs.create("Type", np.bytes_("ImageData"))
            nx, ny, nz = m.shape
            v.attrs["WholeExtent"] = np.array([0, nx, 0, ny, 0, nz])
            v.attrs["Origin"] = np.asarray(m.origin, float)
            v.attrs["Spacing"] = np.asarray(m.spacing, float)
            v.attrs["Direction"] = np.eye(3).ravel()
        else:
            v.attrs.create("Type", np.bytes_("UnstructuredGrid"))
            conn, sizes, types = self._geometry_arrays(m)
            self._geom_total = [m.n_points, m.n_cells, len(conn)]
            for k, d in (("NumberOfPoints", [m.n_points]), ("NumberOfCells", [m.n_cells]),
                         ("NumberOfConnectivityIds", [len(conn)]), ("Points", m.points),
                         ("Connectivity", conn.astype(np.int64)),
                         ("Offsets", np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)),
                         ("Types", np.asarray(types, np.uint8))):
                arr = np.asarray(d)
                if self.mesh_varies:       # every mesh version appends its rows (layout section 8.6)
                    v.create_dataset(k, data=arr, maxshape=(None,) + arr.shape[1:], track_times=False,
                                     chunks=(max(1, min(len(arr), 1 << 16)),) + arr.shape[1:])
                else:
                    v.create_dataset(k, data=arr, track_times=False)
        self.pd = v.create_group("PointData", track_order=True)
        self.cd = v.create_group("CellData", track_order=True)
        st = v.create_group("Steps")
        st.attrs["NSteps"] = 0
        self.steps = st
        self.values = self._resizable(st, "Values", (), np.float64, 256)
        self.static_offsets = {k: self._resizable(st, k, (1,) if k in ("CellOffsets", "ConnectivityIdOffsets") else (),
                                                  np.int64, 256) for k in _STATIC_OFFSETS}
        pdo, cdo = st.create_group("PointDataOffsets"), st.create_group("CellDataOffsets")
        self._rows_done = {"point": 0, "cell": 0}   # rows (ImageData: time entries) written per location

        self.series = []       # (field, dataset, offsets dataset)
        self.statics = []      # (field, dataset, offsets dataset, location) of static arrays
        for f in self.fields.values():
            if f.location not in ("node", "cell", "quadrature_point", "interface_point"):
                raise ValueError(f"field {f.name!r}: unknown location {f.location!r}")
            where = "point" if f.location == "node" else "cell"
            grp, ogrp = (self.pd, pdo) if where == "point" else (self.cd, cdo)
            n = m.n_points if where == "point" else m.n_cells
            tail = self._shape_tail(f, where)
            if f.static:
                arr = self._prepare(f, f.data, where, n)
                # ImageData: a time axis of length 1 in run files; none in setup files, which ParaView
                # reads as data without steps (layout section 2.2)
                timed = self.image and self.profile != "setup"
                data = arr[None] if timed else arr
                if self.mesh_varies:       # one block of rows per mesh version
                    ds = grp.create_dataset(f.name, data=data, dtype=f.dtype, track_times=False,
                                            maxshape=(None,) + data.shape[1:],
                                            chunks=(max(1, min(len(data), 1 << 16)),) + data.shape[1:])
                else:
                    ds = grp.create_dataset(f.name, data=data, dtype=f.dtype, track_times=False)
                for k, val in self._field_attrs(f).items():
                    ds.attrs[k] = val
                self.statics.append((f, ds, self._resizable(ogrp, f.name, (), np.int64, 256), where))
            elif f.placement == "view":
                if self.image:
                    ds = self._resizable(grp, f.name, m.grid_shape(where) + tail, f.dtype, 1, self._field_attrs(f))
                else:
                    ds = self._resizable(grp, f.name, tail, f.dtype, n, self._field_attrs(f))
                self.series.append((f, ds, self._resizable(ogrp, f.name, (), np.int64, 256)))
        self._static_offsets = {f.name: [0] for f, *_ in self.statics}     # first row of each mesh version

    @staticmethod
    def _geometry_arrays(m):
        cells = list(m.blocks.values())
        conn = np.concatenate([c.ravel() for c in cells]) if cells else np.zeros(0, np.int64)
        sizes = np.concatenate([np.full(len(c), c.shape[1]) for c in cells]) if cells else np.zeros(0, np.int64)
        types = np.concatenate([np.full(len(c), cell_info(t)["vtk"]) for t, c in m.blocks.items()]) if cells else []
        return conn, sizes, types

    def _create_groups(self, materials, bcs, periodic, config, meta):
        f, m = self.f, self.mesh
        mesh = f.create_group("mesh")
        if self.image:
            g = mesh.create_group("grid")
            g.create_dataset("shape", data=np.asarray(m.shape, np.int64))
            g.create_dataset("spacing", data=np.asarray(m.spacing, float))
            g.create_dataset("origin", data=np.asarray(m.origin, float))
            g.attrs["index_order"] = "x fastest"
            blocks = {m.cell_type: (0, m.n_cells)}
        else:
            blocks = self._block_ranges(m)
        self.block_ranges = blocks
        self._write_mesh_contents(mesh, m, blocks)

        mat = f.create_group("materials")
        for pid, attrs in materials.items():
            g = mat.create_group(f"phases/{pid}")
            for k, val in attrs.items():
                if k == "stiffness":
                    g.create_dataset("stiffness", data=np.asarray(val, np.float64))
                elif isinstance(val, Mapping):
                    g.attrs[k] = json.dumps(val)
                else:
                    g.attrs[k] = val
        for name, link in (("phase_id", "phase_id"), ("orientation", "orientation")):
            if link in self.fields:
                mat[name] = h5py.SoftLink(f"/VTKHDF/CellData/{link}")

        load = f.create_group("loading")
        for k, attrs in bcs.items():
            g = load.create_group(f"bc/{k}")
            for a, val in attrs.items():
                g.attrs[a] = val
        if periodic:
            g = load.create_group("periodic")
            for a, val in periodic.items():
                g.attrs[a] = val

        state = f.create_group("state")
        for name in self.schema["state_fields"] + ["H_damage_input"]:
            fl = self.fields.get(name)
            if fl is not None and fl.placement == "view" and name != "H_damage_input":
                group = "PointData" if fl.location == "node" else "CellData"
                state[name] = h5py.SoftLink(f"/VTKHDF/{group}/{name}")

        # time-indexed arrays outside /VTKHDF: per-point raw arrays and derived arrays
        self.outside = []      # (field, dataset) at every stored step
        self.derived = {}      # name -> (field, values dataset, step_index dataset)
        for fl in self.fields.values():
            if fl.static or fl.placement == "view":
                continue
            if fl.placement == "derived":
                g = f.require_group(f"derived/{fl.name}")
                n = m.n_points if fl.location == "node" else m.n_cells
                tail = (n,) + self._shape_tail(fl, "")
                vals = self._resizable(g, "values", tail, fl.dtype, 1, self._field_attrs(fl))
                idx = self._resizable(g, "step_index", (), np.int64, 1024, {"units": "1"})
                self.derived[fl.name] = (fl, vals, idx)
            elif fl.placement.startswith("quadrature:"):
                block = fl.placement.split(":", 1)[1]
                a, b = self.block_ranges[block]
                nq = self._n_qp(block)
                tail = (b - a, nq) + (() if fl.n_components == 1 else (fl.n_components,))
                g = f.require_group(f"quadrature/{block}")
                self.outside.append((fl, self._resizable(g, fl.name, tail, fl.dtype, 1, self._field_attrs(fl))))
            elif fl.placement.startswith("interface:"):
                if "shape" not in fl.attrs:
                    raise ValueError(f"interface field {fl.name!r} needs attrs['shape'] = (n_interface_cells, n_points)")
                tail = tuple(fl.attrs["shape"]) + (() if fl.n_components == 1 else (fl.n_components,))
                g = f.require_group(f"interfaces/{fl.placement.split(':', 1)[1]}")
                attrs = {k: v for k, v in self._field_attrs(fl).items() if k != "shape"}
                self.outside.append((fl, self._resizable(g, fl.name, tail, fl.dtype, 1, attrs)))
            else:
                raise ValueError(f"field {fl.name!r}: unknown placement {fl.placement!r}")

        f.create_group("history")
        meta_g = f.create_group("meta")
        cfg = config if isinstance(config, str) or config is None else _dump_yaml(config)
        cfg = cfg if cfg is not None else ""
        ds = meta_g.create_dataset("config_resolved_yaml", data=cfg, dtype=h5py.string_dtype("utf-8"))
        ds.attrs["sha256"] = hashlib.sha256(cfg.encode("utf-8")).hexdigest()
        meta_g.create_dataset("schema_yaml", data=schema_text(self.version), dtype=h5py.string_dtype("utf-8"))
        if self.features_21:
            meta_g.attrs["layout_2_1_features"] = ",".join(self.features_21)
        for k, val in {**_provenance(), **dict(meta)}.items():
            meta_g.attrs[k] = json.dumps(val) if isinstance(val, (Mapping, list, tuple)) else val
        if self.size_estimate is not None:
            meta_g.attrs["size_estimate_bytes"] = int(self.size_estimate["total"])
        if self.moved_out_of_view:
            meta_g.attrs["derived_outside_view"] = ",".join(self.moved_out_of_view)

    @staticmethod
    def _block_ranges(m) -> dict:
        blocks, start = {}, 0
        for ctype, cells in m.blocks.items():
            blocks[ctype] = (start, start + len(cells))
            start += len(cells)
        return blocks

    def _write_mesh_contents(self, mesh, m, blocks):
        """Cell blocks, sets, graph and hanging-node constraints of one mesh below the group ``mesh``."""
        for ctype, (a, b) in blocks.items():
            g = mesh.create_group(f"cell_blocks/{ctype}")
            g.attrs["cell_type"] = ctype
            g.attrs["vtk_type"] = cell_info(ctype)["vtk"]
            g.attrs["node_order"] = "vtk"
            g.create_dataset("cells", data=np.array([a, b], np.int64))
            if ctype in m.quadrature:
                pts, wts, rule, order = m.quadrature[ctype]
                q = g.create_group("quadrature")
                q.create_dataset("points", data=np.asarray(pts, float))
                q.create_dataset("weights", data=np.asarray(wts, float))
                q.attrs["rule"] = rule
                q.attrs["point_order"] = order
        for name, idx in m.node_sets.items():
            mesh.require_group("node_sets").create_dataset(name, data=_array(idx).astype(np.int64))
        for name, pairs in m.face_sets.items():
            mesh.require_group("face_sets").create_dataset(name, data=_array(pairs).astype(np.int64).reshape(-1, 2))
        if not self.image and m.graph:
            mesh.create_group("graph").create_dataset("edge_index", data=edge_index(m.blocks))
        hang = getattr(m, "hanging", None)
        if hang:
            g = mesh.create_group("hanging_nodes")
            g.create_dataset("constrained_nodes", data=_array(hang["constrained_nodes"]).astype(np.int64))
            g.create_dataset("master_nodes", data=_array(hang["master_nodes"]).astype(np.int64))
            g.create_dataset("weights", data=_array(hang["weights"]).astype(np.float64))
            g.attrs["rule"] = "value of a constrained node = sum of weight times value of its master nodes"

    def _n_qp(self, block: str) -> int:
        q = self.mesh.quadrature.get(block)
        if q is None:
            raise ValueError(f"block {block!r} has no quadrature rule; pass mesh quadrature={{'{block}': ...}}")
        return len(q[1])

    def write_array(self, path: str, data: Any, attrs: Mapping[str, Any] | None = None) -> None:
        """Write a static array outside ``/VTKHDF`` before the first increment (for example interface connectivity)."""
        self._check_static_path(path)
        parent, leaf = path.strip("/").rsplit("/", 1)
        ds = self.f.require_group(parent).create_dataset(leaf, data=_array(data), track_times=False)
        for k, v in (attrs or {}).items():
            ds.attrs[k] = v

    def write_attrs(self, path: str, attrs: Mapping[str, Any]) -> None:
        """Set attributes on a group outside ``/VTKHDF`` before the first increment (created when absent)."""
        self._check_static_path(path)
        g = self.f.require_group(path.strip("/"))
        for k, v in attrs.items():
            g.attrs[k] = json.dumps(v) if isinstance(v, Mapping) else v

    def _check_static_path(self, path: str):
        top = path.strip("/").split("/", 1)[0]
        if top in ("VTKHDF", "history", "state", "derived"):
            raise ValueError(f"{path}: written by the writer itself")
        if self.n_rows:
            raise RuntimeError("static groups are created before the first increment (writer rule 2)")

    # --- arrays -----------------------------------------------------------------

    def _prepare(self, f: Field, value, where: str, n: int) -> np.ndarray:
        a = _array(value)
        ncomp = f.n_components
        if self.image:
            gshape = self.mesh.grid_shape(where)
            if a.shape[:3] == gshape:
                a = a.reshape((n,) + a.shape[3:])
        if ncomp == 1 and a.ndim == 2 and a.shape[1] == 1:
            a = a[:, 0]
        expect = (n,) if ncomp == 1 else (n, ncomp)
        if a.shape != expect:
            raise ValueError(f"field {f.name!r}: expected shape {expect}, got {a.shape}")
        if f.dtype.kind in "iu" and a.dtype.kind in "iu" and a.size:
            info = np.iinfo(f.dtype)
            if a.min() < info.min or a.max() > info.max:
                raise ValueError(f"field {f.name!r}: values outside the range of {f.dtype}")
        elif not np.can_cast(a.dtype, f.dtype, casting="same_kind"):
            raise ValueError(f"field {f.name!r}: cannot store {a.dtype} as {f.dtype}")
        if f.is_vector and ncomp == 2:
            a = np.concatenate([a, np.zeros((n, 1), a.dtype)], axis=1)
        a = a.astype(f.dtype, copy=False)
        if self.image:
            a = a.reshape(self.mesh.grid_shape(where) + a.shape[1:])
        return a

    # --- appending ----------------------------------------------------------------

    def append(self, row: Mapping[str, Any], fields: Mapping[str, Any] | None = None,
               derived: Mapping[str, Any] | None = None, *, mesh: UnstructuredMesh | None = None,
               mesh_maps: Mapping[str, Any] | None = None, static: Mapping[str, Any] | None = None) -> int:
        """Record one converged increment; store fields when ``fields`` is given. Returns its step index or -1.

        ``row`` holds the ``history/`` columns (``step_index`` is set here).
        ``fields`` must contain every time-indexed field in the view and every
        per-point or interface array; ``derived`` holds ``/derived`` arrays
        stored at this increment.

        With ``mesh_varies`` a new ``mesh`` (and its ``mesh_maps`` to the
        previous mesh, see :meth:`register_mesh`) takes effect from this
        increment: the fields passed are on the new mesh.
        """
        if self.closed:
            raise RuntimeError("writer is closed")
        if self.profile == "setup":
            raise RuntimeError("a setup-profile file has no increments")
        if mesh is not None or mesh_maps is not None or static is not None:
            if mesh is None:
                raise ValueError("mesh_maps and static arrays belong to a new mesh: pass mesh=")
            self.register_mesh(mesh, mesh_maps, static)
        step = -1
        if fields is not None:
            step = self._write_step(fields, float(_array(row.get("time", self.n_steps))))
        row = dict(row)
        if "step_index" in row and int(row["step_index"]) != step:
            raise ValueError("step_index is assigned by the writer")
        row["step_index"] = step
        if self.mesh_varies:
            row["mesh_index"] = self.cur_version
        self._write_derived(derived or {})
        self._write_row(row)
        self.f.flush()                                                     # 5. flush
        if self.sidecars:
            self._write_sidecars(row)
        return step

    def _n(self, where: str) -> int:
        """Points or cells of the mesh now in force."""
        return self.mesh.n_points if where == "point" else self.mesh.n_cells

    def _write_step(self, fields: Mapping[str, Any], t: float) -> int:
        expected = {f.key for f, *_ in self.series} | {f.key for f, _ in self.outside}
        missing, extra = expected - set(fields), set(fields) - expected
        if missing:
            raise ValueError(f"stored step lacks fields {sorted(missing)}; every array in the view exists "
                             "at every stored step (store others with placement='derived')")
        if extra:
            raise ValueError(f"undeclared fields {sorted(extra)}")
        s = self.n_steps
        prepared = []
        for f, ds, ods in self.series:
            where = "point" if f.location == "node" else "cell"
            prepared.append((ds, self._prepare(f, fields[f.key], where, self._n(where)), ods, where))
        outside = []
        for f, ds in self.outside:
            a = _array(fields[f.key]).astype(f.dtype)
            if a.shape != ds.shape[1:]:
                raise ValueError(f"field {f.name!r}: expected shape {ds.shape[1:]}, got {a.shape}")
            outside.append((ds, a))
        for ds, a, ods, where in prepared:                                 # 1. field rows
            rows = 1 if self.image else self._n(where)
            ds.resize(ds.shape[0] + rows, axis=0)
            ds[-rows:] = a[None] if self.image else a
        for ds, a in outside:
            ds.resize(s + 1, axis=0)
            ds[s] = a
        for k, ds in self.static_offsets.items():                          # 2. offsets and values
            ds.resize(s + 1, axis=0)
            ds[s] = self._geometry_offset(k)
        for f, ds, ods in self.series:
            where = "point" if f.location == "node" else "cell"
            ods.resize(s + 1, axis=0)
            ods[s] = self._rows_done[where]
        for f, ds, ods, where in self.statics:
            ods.resize(s + 1, axis=0)
            ods[s] = self._static_offsets[f.name][self.cur_version]
        self.values.resize(s + 1, axis=0)
        self.values[s] = t
        self.steps.attrs["NSteps"] = s + 1                                 # 3. step count
        for where in ("point", "cell"):
            self._rows_done[where] += 1 if self.image else self._n(where)
        self.n_steps = s + 1
        return s

    def _geometry_offset(self, name: str) -> int:
        """Value of a ``Steps`` geometry dataset for the mesh in force (all zero while the mesh is fixed)."""
        if name == "NumberOfParts":
            return 1
        if not self.mesh_varies:
            return 0
        if name == "PartOffsets":
            return self.cur_version
        point, cell, conn = self._ver_off[self.cur_version]
        return {"PointOffsets": point, "CellOffsets": cell, "ConnectivityIdOffsets": conn}[name]

    def _write_derived(self, derived: Mapping[str, Any]):
        unknown = set(derived) - set(self.derived)
        if unknown:
            raise ValueError(f"undeclared derived arrays {sorted(unknown)}")
        for name, (f, vals, idx) in self.derived.items():
            k = -1
            if name in derived:
                a = _array(derived[name])
                if f.is_vector and f.n_components == 2:
                    a = np.concatenate([a, np.zeros((len(a), 1), a.dtype)], axis=1)
                a = a.astype(f.dtype)
                if a.shape != vals.shape[1:]:
                    raise ValueError(f"derived {name!r}: expected shape {vals.shape[1:]}, got {a.shape}")
                k = vals.shape[0]
                vals.resize(k + 1, axis=0)
                vals[k] = a
            idx.resize(self.n_rows + 1, axis=0)
            idx[self.n_rows] = k

    def _column_attrs(self, name: str) -> dict:
        attrs = dict(self.history_attrs.get(name, {}))
        units = self.history_units.get(name, attrs.pop("units", None))
        if units is None:
            leaf = name.rsplit("/", 1)[-1]
            group = name.split("/", 1)[0] if "/" in name else None
            if group in _TABLE_UNITS:
                units = _TABLE_UNITS[group].get(leaf)
            else:
                units = _DEFAULT_UNITS.get(name) or _GROUP_UNITS.get(group)
        if units is None:
            raise ValueError(f"history column {name!r} needs units (history_units={{'{name}': ...}})")
        attrs["units"] = units
        if name.startswith("energy/") and "method" not in attrs:
            raise ValueError(f"energy column {name!r} needs a method attribute (history_attrs)")
        if name.startswith("reaction/"):
            attrs.setdefault("components", "x,y,z")
            attrs.setdefault("sign_convention", "force exerted by the support on the body")
        return attrs

    def _make_columns(self, group, row: Mapping[str, Any], prefix: str = ""):
        table = prefix.split("/", 1)[0] if prefix else ""
        cols = {}
        for name, value in row.items():
            full = prefix + name
            leaf = name.rsplit("/", 1)[-1]
            text = table == "substeps" and isinstance(value, str)
            value = _array(0 if text else value)
            if table == "substeps" and (leaf in _FLAG_COLUMNS or value.dtype == bool or text):
                dtype = np.int8                           # flags, and text as codes (see _append_columns)
            elif leaf in _INT8_COLUMNS:
                dtype = np.int8
            elif np.issubdtype(value.dtype, np.integer) or value.dtype == bool:
                dtype = np.int64
            elif np.issubdtype(value.dtype, np.floating):
                dtype = np.float64
            else:
                raise ValueError(f"history column {full!r}: unsupported value {value!r}")
            if (name.startswith("reaction/") or full == "crack_tips/position") and value.shape == (2,):
                value = np.append(value, 0.0)
            attrs = self._column_attrs(full)
            if text:
                attrs["codes"] = json.dumps({})
            cols[name] = (value.shape, dtype, attrs)
        # every column is checked before any is created, so a refused row leaves no partial table
        return {name: self._resizable(group, name, shape, dtype, 1024, attrs) for name, (shape, dtype, attrs) in cols.items()}

    def _append_columns(self, cols, row, n, prefix="", codes=None):
        missing = set(cols) - set(row)
        extra = set(row) - set(cols)
        if extra:
            raise ValueError(f"history columns {sorted(extra)} were not in the first row; "
                             "every column is created before the first flush")
        for name, ds in cols.items():
            if name in missing:
                if not np.issubdtype(ds.dtype, np.floating):
                    raise ValueError(f"history row lacks integer column {prefix + name!r}")
                value = np.full(ds.shape[1:], np.nan)
            elif codes is not None and "codes" in ds.attrs:
                value = self._code_of(ds, codes.setdefault(name, {}), row[name])
            else:
                value = _array(row[name])
                if (name.startswith("reaction/") or prefix + name == "crack_tips/position") and value.shape == (2,):
                    value = np.append(value, 0.0)
            ds.resize(n + 1, axis=0)
            ds[n] = value

    @staticmethod
    def _code_of(ds, known: dict, value) -> int:
        """Code of a text value of a categorical column; a new value extends the ``codes`` attribute first."""
        text = "" if value is None else str(value)
        if text not in known:
            if len(known) >= 127:
                raise ValueError(f"column {ds.name}: more than 127 distinct values")
            known[text] = len(known)
            ds.attrs["codes"] = json.dumps({str(c): t for t, c in known.items()})
        return known[text]

    def _write_row(self, row: Mapping[str, Any]):                           # 4. history row
        if self._columns is None:
            required = self.schema["history_columns"]["required"]
            missing = [c for c in required if c not in row]
            if missing:
                raise ValueError(f"history row lacks required columns {missing}")
            self._columns = self._make_columns(self.f["history"], row)
        self._append_columns(self._columns, row, self.n_rows)
        self.n_rows += 1

    # --- layout 2.1 --------------------------------------------------------------

    def _enable_21(self, feature: str) -> None:
        """Move the file to layout 2.1.0 (root version, embedded schema) before the first item that needs it."""
        if feature not in self.features_21:
            self.features_21.append(feature)
        self.f["meta"].attrs["layout_2_1_features"] = ",".join(self.features_21)
        if self.version == SCHEMA_VERSION_21:
            return
        self.version = SCHEMA_VERSION_21
        self.schema = load_schema(self.version)
        self.f.attrs["phast_schema_version"] = self.version
        meta = self.f["meta"]
        del meta["schema_yaml"]
        meta.create_dataset("schema_yaml", data=schema_text(self.version), dtype=h5py.string_dtype("utf-8"))
        self.f.flush()

    def register_mesh(self, mesh: UnstructuredMesh, maps: Mapping[str, Any] | None,
                      static: Mapping[str, Any] | None = None) -> int:
        """Add a mesh version (layout 2.1, ``mesh_varies = True``) and make it the mesh in force.

        ``maps`` relate the new mesh to the previous version: ``cell_parent``
        ``(E_new,)`` (cell of the previous mesh each cell came from, -1 for
        none), ``node_origin`` ``(N_new,)`` (index of the node in the previous
        mesh, -1 for a new node) and ``node_parents`` ``(N_new, 2)`` (the two
        previous nodes whose mean is a new node, -1 for existing nodes). See
        :func:`refinement_maps` for the solver's refinement dictionaries.
        ``static`` gives the data of every static field (for example
        ``phase_id``) on the new mesh. Returns the version number.
        """
        if not self.mesh_varies:
            raise ValueError("the file was created with mesh_varies=False; pass mesh_varies=True to the writer")
        if not isinstance(mesh, UnstructuredMesh):
            raise TypeError("a new mesh is an UnstructuredMesh")
        maps = dict(maps or {})
        missing = [k for k in ("cell_parent", "node_origin", "node_parents") if k not in maps]
        if missing:
            raise ValueError(f"mesh version lacks the maps {missing} to the previous mesh")
        shapes = {"cell_parent": (mesh.n_cells,), "node_origin": (mesh.n_points,), "node_parents": (mesh.n_points, 2)}
        arrays = {}
        for key, want in shapes.items():
            arrays[key] = _array(maps[key]).astype(np.int64)
            if arrays[key].shape != want:
                raise ValueError(f"map {key}: expected shape {want}, got {arrays[key].shape}")
        static = dict(static or {})
        need = [f.name for f, *_ in self.statics]
        if sorted(static) != sorted(need):
            raise ValueError(f"static fields {need} need data on every mesh version, got {sorted(static)}")
        k = self.n_versions
        v = self.f["VTKHDF"]
        conn, sizes, types = self._geometry_arrays(mesh)
        offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        old_mesh = self.mesh
        self.mesh = mesh                                                        # rows below use the new sizes
        try:
            statics = [(f, ds, ods, self._prepare(f, static[f.name], where, self._n(where)))
                       for f, ds, ods, where in self.statics]
        except Exception:
            self.mesh = old_mesh
            raise
        totals = list(self._geom_total)
        for name, rows in (("Points", mesh.points), ("Connectivity", conn), ("Offsets", offsets),
                           ("Types", np.asarray(types, np.uint8))):            # 1. geometry rows
            ds = v[name]
            rows = np.asarray(rows, ds.dtype)
            ds.resize(ds.shape[0] + len(rows), axis=0)
            ds[-len(rows):] = rows
        for name, value in (("NumberOfPoints", mesh.n_points), ("NumberOfCells", mesh.n_cells),
                            ("NumberOfConnectivityIds", len(conn))):
            ds = v[name]
            ds.resize(ds.shape[0] + 1, axis=0)
            ds[-1] = value
        self._geom_total = [totals[0] + mesh.n_points, totals[1] + mesh.n_cells, totals[2] + len(conn)]
        self._ver_off.append(tuple(totals))
        for f, ds, ods, data in statics:                                        # 2. static fields on the new mesh
            self._static_offsets[f.name].append(ds.shape[0])
            ds.resize(ds.shape[0] + len(data), axis=0)
            ds[-len(data):] = data
        g = self.f.require_group("mesh").require_group(f"versions/{k}")          # 3. the version group
        self.block_ranges = self._block_ranges(mesh)
        self._write_mesh_contents(g, mesh, self.block_ranges)
        g.attrs["n_points"], g.attrs["n_cells"] = mesh.n_points, mesh.n_cells
        g.attrs["first_increment"] = self.n_rows
        g.attrs["node_parents_rule"] = "a new node takes the mean of its two parent nodes"
        for key, a in arrays.items():
            g.create_dataset(key, data=a)
        self.n_versions += 1
        self.cur_version = k
        return k

    def write_initial(self, field: str | Field, data: Any, specification: Mapping[str, Any] | None = None) -> None:
        """Store an initial condition below ``initial_conditions/`` before the first increment (layout 2.1).

        ``field`` names a declared field (the array then carries its location,
        components, units and kind) or is a :class:`Field`. The data are the
        array at the start of the first increment, on the initial mesh.
        ``specification`` records how it was set (for example the pre-seeded
        crack regions) and is stored as JSON in the ``specification`` attribute.
        """
        if self.n_rows:
            raise RuntimeError("initial conditions are written before the first increment (writer rule 2)")
        f = self.fields[field] if isinstance(field, str) else field
        if f is None:
            raise ValueError(f"{field!r} is not a declared field; pass a Field")
        self._enable_21("initial_conditions")
        where = "point" if f.location == "node" else "cell"
        arr = self._prepare(f, data, where, self._n(where))
        grp = self.f.require_group("initial_conditions")
        ds = grp.create_dataset(f.name, data=arr, dtype=f.dtype, track_times=False)
        for key, val in self._field_attrs(f).items():
            ds.attrs[key] = val
        if specification is not None:
            ds.attrs["specification"] = json.dumps(specification, default=str)

    def write_validation(self, rows: Iterable[Mapping[str, Any]], source: str | None = None) -> None:
        """Store validation results in ``meta/validation`` (layout 2.1), once, any time before the file closes.

        Each row has ``name``, ``value`` (NaN for a condition that is met or not),
        ``tolerance`` (NaN when there is none) and ``passed``; ``units`` and
        ``description`` are optional. ``source`` names the summary the rows
        come from, as a relative file name.
        """
        if self._validation_written:
            raise RuntimeError("meta/validation is written once")
        rows = [dict(r) for r in rows]
        for r in rows:
            missing = [c for c in ("name", "value", "tolerance", "passed") if c not in r]
            if missing:
                raise ValueError(f"validation row {r.get('name')!r} lacks {missing}")
        self._enable_21("validation")
        self._validation_written = True
        g = self.f["meta"].create_group("validation")
        text = h5py.string_dtype("utf-8")
        g.create_dataset("name", data=[str(r["name"]) for r in rows], dtype=text)
        g.create_dataset("value", data=np.array([float(r["value"]) for r in rows], np.float64))
        g.create_dataset("tolerance", data=np.array([float(r["tolerance"]) for r in rows], np.float64))
        g.create_dataset("passed", data=np.array([int(bool(r["passed"])) for r in rows], np.int8))
        if any("units" in r for r in rows):
            g.create_dataset("units", data=[str(r.get("units", "")) for r in rows], dtype=text)
        if any("description" in r for r in rows):
            g.create_dataset("description", data=[str(r.get("description", "")) for r in rows], dtype=text)
        g.attrs["overall_passed"] = int(all(bool(r["passed"]) for r in rows))
        if source:
            g.attrs["source"] = source

    def append_crack_tip(self, row: Mapping[str, Any], method: str | None = None) -> None:
        """Record one crack tip of an increment in ``history/crack_tips/`` (layout 2.1).

        ``row``: ``increment``, ``tip_id`` (0, 1, ... within the increment),
        ``position`` (x, y[, z] in mm) and ``threshold`` (the damage value that
        defines the tip). ``method`` says how the tip was found and is stored
        on ``position``. Rows are in increment order; the crack speed is added
        at close (:mod:`phast.io.result.crack_tips`).
        """
        self._append_table("crack_tips", row, {"position": {"components": "x,y,z", **({"method": method} if method else {})}})

    def append_substep(self, row: Mapping[str, Any]) -> None:
        """Record one damage solve in ``history/substeps/`` (layout 2.1).

        ``row``: ``increment``, ``call`` (index of the solve within the
        increment) and the solve's telemetry. Booleans are stored as 0/1,
        text values (preconditioner names, reasons) as codes listed in the
        column's ``codes`` attribute.
        """
        self._append_table("substeps", row, {})

    def _append_table(self, table: str, row: Mapping[str, Any], attrs: Mapping[str, Mapping[str, Any]]) -> None:
        if self.closed:
            raise RuntimeError("writer is closed")
        need = self.schema_for_table(table)["required"]
        missing = [c for c in need if c not in row]
        if missing:
            raise ValueError(f"{table} row lacks {missing}")
        self._enable_21(table)
        state = self._tables.get(table)
        if state is None:
            group = self.f["history"].create_group(table)
            cols = self._make_columns(group, row, f"{table}/")
            for name, a in attrs.items():
                if name in cols:
                    for k, v in a.items():
                        cols[name].attrs[k] = v
            state = self._tables[table] = dict(columns=cols, n=0, codes={})
        self._append_columns(state["columns"], row, state["n"], f"{table}/", state["codes"])
        state["n"] += 1

    def schema_for_table(self, table: str) -> dict:
        return load_schema(SCHEMA_VERSION_21)["history_tables"][table]

    # --- sidecars -----------------------------------------------------------------

    def _write_sidecars(self, row):
        flat = {}
        for k, v in row.items():
            a = _array(v)
            if a.ndim == 0:
                flat[k.replace("/", "_")] = a.item()
            else:
                for c, x in zip("xyz", a.ravel()):
                    flat[f"{k.replace('/', '_')}_{c}"] = float(x)
        hist = self.final_path.parent / "history.csv"
        new = self._csv is None
        if new:
            self._csv = list(flat)
            with hist.open("w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(self._csv)
        with hist.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([flat.get(k, "") for k in self._csv])
        prog = dict(run_id=str(self.f.attrs["run_id"]), increment=flat.get("increment"), time=flat.get("time"),
                    stored_steps=self.n_steps, rows=self.n_rows, status="running", updated=_now())
        tmp = self.final_path.parent / "progress.json.tmp"
        tmp.write_text(json.dumps(prog, indent=1), encoding="utf-8")
        os.replace(tmp, self.final_path.parent / "progress.json")

    # --- closing ------------------------------------------------------------------

    def close(self, status: str = "complete") -> Path:
        """Set the final status, close, validate, rename ``.partial`` to the final name and write its SHA-256."""
        if self.closed:
            return self.final_path
        allowed = self.schema["root_attributes"]["required"]["status"]["values"]
        if status not in allowed or status == "running":
            raise ValueError(f"final status {status!r}; allowed: {[s for s in allowed if s != 'running']}")
        if self._columns is None:                  # no increment recorded: required columns, empty
            names = ["step_index"] if self.profile == "setup" else self.schema["history_columns"]["required"]
            for name in names:
                dtype = np.int8 if name in _INT8_COLUMNS else np.int64 if name in ("increment", "step_index") else np.float64
                units = self.history_units.get(name, _DEFAULT_UNITS.get(name, "1"))
                self._resizable(self.f["history"], name, (), dtype, 1024, {"units": units})
        self._finalize_crack_tips()
        self.f.attrs["status"] = status
        self.f.attrs["finished"] = _now()
        self.f.close()
        self.closed = True
        target = self.path
        if self.validate_on_close:
            self.report = _validate(self.path)
        errors = _errors(self.report)
        if errors:
            warnings.warn(f"{self.path.name} failed validation and keeps its .partial name: "
                          + "; ".join(errors[:5]), stacklevel=2)
        else:
            os.replace(self.path, self.final_path)
            target = self.final_path
            digest = sha256_file(target)
            target.with_name(target.name + ".sha256").write_text(f"{digest}  {target.name}\n", encoding="utf-8")
        if self.sidecars:
            prog_path = self.final_path.parent / "progress.json"
            if prog_path.exists():
                prog = json.loads(prog_path.read_text(encoding="utf-8"))
                prog.update(status=status, updated=_now())
                prog_path.write_text(json.dumps(prog, indent=1), encoding="utf-8")
        return target

    def flush(self) -> None:
        """Flush the file (``append`` flushes; call this after rows added with ``append_crack_tip`` or ``append_substep``)."""
        if not self.closed:
            self.f.flush()

    def _finalize_crack_tips(self) -> None:
        """Add the derived crack speed to ``history/crack_tips`` (a killed run keeps the table without it)."""
        state = self._tables.get("crack_tips")
        if not state or not state["n"]:
            return
        from .crack_tips import FIT_METHOD, FIT_POLYORDER, FIT_WINDOW, fit_speed
        g = self.f["history/crack_tips"]
        inc = np.asarray(self.f["history/increment"][...], np.int64)
        time_of = dict(zip(inc.tolist(), np.asarray(self.f["history/time"][...], float).tolist()))
        rows = np.asarray(g["increment"][...], np.int64)
        speed = fit_speed(g["tip_id"][...], g["position"][...], [time_of[int(i)] for i in rows])
        ds = g.create_dataset("speed", data=speed, track_times=False, dtype=np.float64)
        per = self.f["history/time"].attrs.get("units", "s")        # quasi-static runs count a pseudo-time of unit "1"
        per = per.decode() if isinstance(per, bytes) else str(per)
        ds.attrs.update(units="mm/s" if per == "s" else "mm" if per == "1" else f"mm/({per})", fit_method=FIT_METHOD,
                        fit_window=FIT_WINDOW, fit_polyorder=FIT_POLYORDER, derived_from="position and history/time")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.closed:
            self.close("failed" if exc_type is not None else "complete")
        return False

    def write_setup(self) -> Path:
        """Close a ``setup``-profile file (mesh, materials and loading, no increments)."""
        if self.profile != "setup":
            raise RuntimeError("write_setup is for profile='setup'")
        return self.close("complete")


def refinement_maps(n_old_nodes: int, n_new_nodes: int, parent_map: Mapping[int, tuple[int, int]],
                    child_map: Mapping[int, int]) -> dict:
    """Maps of a refinement for :meth:`ResultWriter.register_mesh`, from the solver's dictionaries.

    ``parent_map`` is ``{new node: (node a, node b)}`` for the nodes added by the refinement and ``child_map`` is
    ``{new cell: cell of the previous mesh}`` as returned by ``phast.solvers.adaptive``; nodes of the previous
    mesh keep their numbers, new nodes follow them.
    """
    cell_parent = np.full(len(child_map), -1, np.int64)
    for new, old in child_map.items():
        cell_parent[int(new)] = int(old)
    node_origin = np.arange(n_new_nodes, dtype=np.int64)
    node_origin[n_old_nodes:] = -1
    node_parents = np.full((n_new_nodes, 2), -1, np.int64)
    for new, (a, b) in parent_map.items():
        if not n_old_nodes <= int(new) < n_new_nodes:
            raise ValueError(f"new node {new} is not among the {n_new_nodes - n_old_nodes} nodes added after the "
                             f"{n_old_nodes} of the previous mesh")
        node_parents[int(new)] = (min(int(a), int(b)), max(int(a), int(b)))
    if (node_parents[n_old_nodes:, 0] < 0).any():
        raise ValueError("a node added by the refinement has no parent nodes")
    return {"cell_parent": cell_parent, "node_origin": node_origin, "node_parents": node_parents}


def isotropic_wave_speeds(E: float, nu: float, rho: float, analysis: str = "plane_strain") -> dict:
    """``c_p``, ``c_s`` and ``c_R`` (units of the unit system: mm/s) of an isotropic phase, as the validator checks them.

    ``c_p`` uses the Lame parameter of the analysis (plane stress: ``E nu / (1 - nu^2)``), ``c_s = sqrt(mu / rho)`` and
    ``c_R`` is the Viktorov approximation ``c_s (0.862 + 1.14 nu) / (1 + nu)``, as in ``physics.material``.
    """
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / (1.0 - nu * nu) if analysis == "plane_stress" else E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c_s = (mu / rho) ** 0.5
    return {"c_p": ((lam + 2.0 * mu) / rho) ** 0.5, "c_s": c_s, "c_R": c_s * (0.862 + 1.14 * nu) / (1.0 + nu)}


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _dump_yaml(obj) -> str:
    import yaml
    return yaml.safe_dump(json.loads(json.dumps(obj, default=str)), sort_keys=False)


def _provenance() -> dict:
    from phast.utils.provenance import dependency_versions, git_state
    git = git_state()
    deps = {k: v for k, v in dependency_versions().items() if v is not None}
    deps["HDF5"] = h5py.version.hdf5_version
    sha = git.get("commit") or "unknown"
    return dict(software="phast", software_version=f"{sha}{'-dirty' if git.get('dirty') else ''}",
                dependencies=deps, machine=platform.node(), platform=platform.platform(),
                python=sys.version.split()[0], slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
                command_line=_portable_command(sys.argv), solver_dtype="float64")


def _portable_command(argv) -> str:
    """Command line with the home directory written as ``~`` (no user paths in the file)."""
    home = os.path.expanduser("~")
    return " ".join(a.replace(home, "~") if home and home != "~" else a for a in argv)


def _validate(path):
    try:
        from .validate import validate
    except ImportError:          # validator not installed in this build
        return None
    return validate(path, allow_running=False)


def _errors(report) -> list[str]:
    if report is None:
        return []
    return [f"{x.code}: {x.message}" for x in report.findings if x.level == "ERROR"]
