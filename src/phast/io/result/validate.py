"""Validator for PhAST result files (layout 2.x).

Implements the rules of ``docs/user_guide/result_file_layout.md`` section 5.
The validator reads only the file and the schema YAML; it never relies on how
a file was produced. Arrays are not loaded: only shapes, dtypes, creation
property lists, the step offsets and ``history/step_index``.
"""
from __future__ import annotations

import json
import posixpath
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import yaml

from .schema import LATEST_VERSION, SCHEMA_VERSION, load_schema, schema_version_for

LEVELS = ("ERROR", "WARN", "INFO")
# HDF5 filter ids of the filter names allowed by the schema ("none" means no filter)
_FILTER_IDS = {"gzip": 1, "deflate": 1, "shuffle": 2, "fletcher32": 3}
_WINDOWS_ABS = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")


@dataclass(frozen=True)
class Finding:
    level: str          # ERROR, WARN or INFO
    code: str           # short stable slug, for example "dangling-link"
    path: str           # HDF5 path, or "" for the file as a whole
    message: str


@dataclass
class Report:
    file: str
    findings: list[Finding] = field(default_factory=list)

    def add(self, level: str, code: str, path: str, message: str) -> None:
        self.findings.append(Finding(level, code, path, message))

    def by_level(self, level: str) -> list[Finding]:
        return [f for f in self.findings if f.level == level]

    @property
    def errors(self) -> list[Finding]:
        return self.by_level("ERROR")

    @property
    def warnings(self) -> list[Finding]:
        return self.by_level("WARN")

    @property
    def infos(self) -> list[Finding]:
        return self.by_level("INFO")

    @property
    def ok(self) -> bool:
        return not self.errors

    def codes(self, level: str | None = None) -> set[str]:
        return {f.code for f in self.findings if level is None or f.level == level}

    def to_dict(self) -> dict[str, Any]:
        counts = {lvl: len(self.by_level(lvl)) for lvl in LEVELS}
        return {"file": self.file, "ok": self.ok, "counts": counts, "findings": [asdict(f) for f in self.findings]}

    def format_text(self) -> str:
        lines = [f"{self.file}: {'OK' if self.ok else 'INVALID'} "
                 f"({len(self.errors)} error, {len(self.warnings)} warning, {len(self.infos)} info)"]
        for lvl in LEVELS:
            items = self.by_level(lvl)
            if items:
                lines.append(f"{lvl}:")
                lines += [f"  [{f.code}] {f.path or '/'}: {f.message}" for f in items]
        return "\n".join(lines)


# --- small helpers -----------------------------------------------------------

def _text(value: Any) -> Any:
    """Decode bytes and 0-d arrays of bytes or str to str; leave other values as they are."""
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value[()]
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.str_):
        return str(value)
    return value


def _strings(value: Any) -> list[str]:
    """All string elements of an attribute or dataset value."""
    value = _text(value)
    if isinstance(value, str):
        return [value]
    if isinstance(value, np.ndarray) and value.dtype.kind in "OSU":
        return [s for s in (_text(v) for v in value.ravel()) if isinstance(s, str)]
    return []


def _same(value: Any, expected: Any) -> bool:
    value = _text(value)
    if isinstance(expected, bool):
        return isinstance(value, (bool, np.bool_, np.integer)) and bool(value) == expected
    if isinstance(value, np.generic):
        value = value.item()
    return value == expected


def _iso_with_zone(value: Any) -> bool:
    value = _text(value)
    if not isinstance(value, str):
        return False
    try:
        stamp = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError:
        return False
    return stamp.tzinfo is not None


def _is_absolute_path(text: str) -> bool:
    return text.startswith(("/", "~")) or bool(_WINDOWS_ABS.match(text)) or text.lower().startswith("file:")


def _ncomp(ds: h5py.Dataset, image: bool) -> int:
    """Number of components of a field in /VTKHDF (rows, or time + three grid axes, come first)."""
    lead = 4 if image else 1
    return int(np.prod(ds.shape[lead:])) if ds.ndim > lead else 1


def _filters(ds: h5py.Dataset) -> list[tuple[int, str]]:
    plist = ds.id.get_create_plist()
    out = []
    for i in range(plist.get_nfilters()):
        code, _flags, _values, name = plist.get_filter(i)
        out.append((int(code), _text(name) or ""))
    return out


def _schema_version_of(f: h5py.File) -> str:
    """Schema a file is checked with: the one of its own ``phast_schema_version``; 2.0.0 when that is unusable."""
    try:
        return schema_version_for(_text(f.attrs.get("phast_schema_version")))
    except ValueError:
        return SCHEMA_VERSION


# --- the validator -----------------------------------------------------------

class _Validator:
    def __init__(self, f: h5py.File, report: Report, allow_running: bool) -> None:
        self.f, self.r, self.allow_running = f, report, allow_running
        # a file is checked against the schema of its own version: 2.0 files keep the 2.0 rules
        self.schema_version = _schema_version_of(f)
        self.s = load_schema(self.schema_version)
        self.v = self.s["validator"]
        self.nsteps = 0
        self.image = False
        self.n_points = self.n_cells = None
        self.grid: tuple[int, int, int] | None = None   # (nx, ny, nz)
        self.varying = False                            # mesh_varies = true in a file that allows it
        self.versions: list[tuple[int, int, int]] = []  # (points, cells, connectivity ids) of each mesh version

    def err(self, code: str, path: str, msg: str) -> None:
        self.r.add("ERROR", code, path, msg)

    def warn(self, code: str, path: str, msg: str) -> None:
        self.r.add("WARN", code, path, msg)

    def info(self, code: str, path: str, msg: str) -> None:
        self.r.add("INFO", code, path, msg)

    def run(self) -> None:
        checks: list[Callable[[], None]] = [
            self.root_attributes, self.groups, self.vtkhdf, self.steps, self.fields,
            self.coupling, self.materials, self.history, self.objects,
            self.long_tables, self.loading, self.initial_conditions, self.wave_speeds, self.validation,
            self.varying_mesh,
        ]
        for check in checks:
            try:
                check()
            except Exception as exc:  # a malformed file must give a finding, not a traceback
                self.err("read-error", "", f"{check.__name__}: {type(exc).__name__}: {exc}")

    # root attributes ---------------------------------------------------------
    def root_attributes(self) -> None:
        rules = self.s["root_attributes"]
        attrs = self.f.attrs
        own = {"phast_schema_version", "mesh_varies", "status"}   # checked with their own codes below
        for key, rule in rules["required"].items():
            if key not in attrs:
                self.err("missing-attribute", "/", f"required root attribute {key!r} is missing")
            elif key == "dim" and "dimensions" in self.s and _same(attrs[key], 1):
                self.err("dimension-unsupported", "/", self.s["dimensions"]["excluded"][1])
            elif key not in own:
                self._attr_value(key, attrs[key], rule)
        for key, rule in rules["optional"].items():
            if key in attrs:
                self._attr_value(key, attrs[key], rule)

        if "phast_schema_version" in attrs:
            version = _text(attrs["phast_schema_version"])
            major = SCHEMA_VERSION.split(".")[0]
            m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version) if isinstance(version, str) else None
            if m is None:
                self.err("schema-version", "/", f"phast_schema_version {version!r} is not MAJOR.MINOR.PATCH")
            elif m.group(1) != major:
                self.err("schema-version-major", "/",
                         f"unknown phast_schema_version major {version!r}; this validator reads major {major}")
            elif (int(m.group(2)), int(m.group(3))) > tuple(int(x) for x in LATEST_VERSION.split(".")[1:]):
                self.info("schema-version-newer", "/",
                          f"file written against {version}; checked against {LATEST_VERSION} (items added later are not checked)")
        if "mesh_varies" in attrs:
            allowed = rules["required"]["mesh_varies"].get("values", [rules["required"]["mesh_varies"].get("value")])
            if _same(attrs["mesh_varies"], True) and True in allowed:
                self.varying = True
            elif not _same(attrs["mesh_varies"], False):
                self.err("mesh-varies", "/", "mesh_varies = true is refused by layout 2.0 (planned for 2.1)"
                         if True not in allowed else f"mesh_varies = {_text(attrs['mesh_varies'])!r} must be true or false")
        if "status" in attrs:
            status = _text(attrs["status"])
            if status not in rules["required"]["status"]["values"]:
                self.err("invalid-attribute", "/", f"status = {status!r} is not one of {rules['required']['status']['values']}")
            elif status == "running":
                if self.allow_running:
                    self.info("status-running", "/", "status = running, allowed because running files were permitted (--running-ok)")
                else:
                    self.err("status-running", "/",
                             "status = running: the run did not finish or is still writing. Allowed only on scratch while "
                             "the run is active (pass --running-ok); otherwise use 'phast result recover'")

    def _attr_value(self, key: str, value: Any, rule: dict[str, Any]) -> None:
        value = _text(value)
        bad = None
        if "value" in rule and not _same(value, rule["value"]):
            bad = f"must be {rule['value']!r}"
        elif "values" in rule and not any(_same(value, v) for v in rule["values"]):
            bad = f"must be one of {rule['values']}"
        elif "pattern" in rule and not (isinstance(value, str) and re.fullmatch(rule["pattern"], value)):
            bad = f"must match {rule['pattern']}"
        elif rule.get("format") == "iso8601_with_zone" and not _iso_with_zone(value):
            bad = "must be an ISO 8601 time with a time zone"
        elif rule.get("type") == "str" and not isinstance(value, str):
            bad = "must be a string"
        elif rule.get("type") == "int" and not isinstance(value, (int, np.integer)):
            bad = "must be an integer"
        elif rule.get("type") == "float" and not isinstance(value, (float, int, np.floating, np.integer)):
            bad = "must be a number"
        if bad:
            self.err("invalid-attribute", "/", f"root attribute {key} = {value!r} {bad}")

    # groups ------------------------------------------------------------------
    def groups(self) -> None:
        g = self.s["groups"]
        for name in g["required"]:
            if not isinstance(self.f.get(name), h5py.Group):
                self.err("missing-group", f"/{name}", f"required group /{name} is missing")
        for name in g["optional"]:
            if name not in self.f:
                self.info("optional-absent", f"/{name}", f"optional group /{name} is absent")
        meta = self.f.get("meta")
        if isinstance(meta, h5py.Group):
            for name in g["meta_required"]:
                if name not in meta:
                    self.err("missing-meta", f"/meta/{name}", f"required item meta/{name} is missing")
            for name in g["meta_optional"]:
                if name not in meta:
                    self.info("optional-absent", f"/meta/{name}", f"optional item meta/{name} is absent")

    # /VTKHDF -----------------------------------------------------------------
    def vtkhdf(self) -> None:
        v = self.f.get("VTKHDF")
        if not isinstance(v, h5py.Group):
            return
        spec = self.s["vtkhdf"]
        version = v.attrs.get("Version")
        if version is None:
            self.err("missing-attribute", "/VTKHDF", "attribute Version is missing")
        elif int(np.ravel(version)[0]) != spec["version"][0]:
            self.err("invalid-attribute", "/VTKHDF", f"VTKHDF Version {list(np.ravel(version))} is not major {spec['version'][0]}")
        vtype = _text(v.attrs.get("Type"))
        if vtype not in spec["types"]:
            self.err("invalid-attribute", "/VTKHDF", f"Type = {vtype!r} is not one of {spec['types']}")
            return
        self.image = vtype == "ImageData"
        if self.image:
            for a in spec["imagedata_attributes"]:
                if a not in v.attrs:
                    self.err("missing-attribute", "/VTKHDF", f"ImageData attribute {a} is missing")
            ext = np.ravel(v.attrs.get("WholeExtent", []))
            if ext.size == 6:
                nx, ny, nz = (int(ext[1] - ext[0]), int(ext[3] - ext[2]), int(ext[5] - ext[4]))
                self.grid = (nx, ny, nz)
                self.n_points, self.n_cells = (nx + 1) * (ny + 1) * (nz + 1), max(nx, 1) * max(ny, 1) * max(nz, 1)
            for a in ("Origin", "Spacing"):
                if a in v.attrs and np.asarray(v.attrs[a]).dtype != np.float64:
                    self.err("dtype", "/VTKHDF", f"ImageData {a} is {np.asarray(v.attrs[a]).dtype}, coordinates must be float64")
            return
        for name in spec["unstructured_required"]:
            if name not in v:
                self.err("missing-dataset", f"/VTKHDF/{name}", f"UnstructuredGrid dataset {name} is missing")
        if "NumberOfPoints" in v and "NumberOfCells" in v:
            parts = 1
            if "Steps" in v and "NumberOfParts" in v["Steps"] and v["Steps/NumberOfParts"].shape[0]:
                parts = max(1, int(v["Steps/NumberOfParts"][0]))
            self.n_points = int(np.sum(v["NumberOfPoints"][:parts]))
            self.n_cells = int(np.sum(v["NumberOfCells"][:parts]))
        pts = v.get("Points")
        if isinstance(pts, h5py.Dataset):
            if pts.dtype != np.float64:
                self.err("dtype", pts.name, f"Points are {pts.dtype}, coordinates must be float64")
            if pts.ndim != 2 or pts.shape[1] != spec["points_components"]:
                self.err("points-components", pts.name, f"Points have shape {pts.shape}, expected (N, {spec['points_components']})")
            if self.n_points is not None and pts.shape[0] < self.n_points:
                self.err("array-short", pts.name, f"Points has {pts.shape[0]} rows, NumberOfPoints is {self.n_points}")

    # /VTKHDF/Steps -----------------------------------------------------------
    def steps(self) -> None:
        v = self.f.get("VTKHDF")
        if not isinstance(v, h5py.Group):
            return
        st = v.get("Steps")
        if not isinstance(st, h5py.Group):
            self.err("missing-group", "/VTKHDF/Steps", "group /VTKHDF/Steps is missing (required also for NSteps = 0)")
            return
        if "NSteps" not in st.attrs:
            self.err("missing-attribute", st.name, "attribute NSteps is missing")
            return
        n = self.nsteps = int(st.attrs["NSteps"])
        if _text(self.f.attrs.get("profile")) == "setup" and n != self.s["profile_rules"]["setup"]["nsteps"]:
            self.err("setup-nsteps", st.name, f"profile = setup requires NSteps = 0, found {n}")
        if self.image and n == 1:
            self.warn("imagedata-single-step", st.name, "ImageData with exactly one stored step: ParaView 5.13 "
                      "shows no arrays (it reads fewer than two steps as data without steps); phast.load_result reads it")
        for name in self.s["vtkhdf"]["steps_required"]:
            ds = st.get(name)
            if not isinstance(ds, h5py.Dataset):
                self.err("missing-dataset", f"{st.name}/{name}", f"Steps dataset {name} is missing")
            elif ds.shape[0] < n:
                self.err("steps-short", ds.name, f"{name} has {ds.shape[0]} entries, NSteps is {n}")
        for group, ogroup, rows in (("PointData", "PointDataOffsets", self.n_points), ("CellData", "CellDataOffsets", self.n_cells)):
            data = v.get(group)
            if not isinstance(data, h5py.Group):
                continue
            offs = st.get(ogroup)
            for name in data:
                ds = data.get(name)
                if not isinstance(ds, h5py.Dataset):
                    continue
                if self._grid_shape(ds, group, untimed_ok=n == 0) == "untimed":
                    continue
                ods = offs.get(name) if isinstance(offs, h5py.Group) else None
                if not isinstance(ods, h5py.Dataset):
                    self.warn("offsets-missing", ds.name, f"no {ogroup}/{name}; readers treat the array as static")
                    continue
                if ods.shape[0] < n:
                    self.err("steps-short", ods.name, f"{ogroup}/{name} has {ods.shape[0]} entries, NSteps is {n}")
                    continue
                if n == 0:
                    continue
                per = 1 if self.image else rows
                if per is None:
                    continue
                if self.varying:
                    need = int(np.max(ods[:n] + self._rows_of_steps(st, group, n)))
                else:
                    need = int(np.max(ods[:n])) + per
                if ds.shape[0] < need:
                    unit = "time entries" if self.image else "rows"
                    self.err("array-short", ds.name,
                             f"{ds.shape[0]} {unit}, but the offsets of the {n} stored steps need {need}")

    def _rows_of_steps(self, st: h5py.Group, group: str, n: int) -> np.ndarray:
        """Rows of every stored step of a varying mesh: the points or cells of the mesh version it uses."""
        v = self.f["VTKHDF"]
        counts = np.asarray(v["NumberOfPoints" if group == "PointData" else "NumberOfCells"][...], np.int64)
        parts = np.asarray(st["PartOffsets"][:n], np.int64)
        ok = (parts >= 0) & (parts < counts.size)
        return np.where(ok, counts[np.clip(parts, 0, max(counts.size - 1, 0))], 0)

    def _grid_shape(self, ds: h5py.Dataset, group: str, untimed_ok: bool = False) -> str | None:
        if not (self.image and self.grid):
            return None
        nx, ny, nz = self.grid
        expect = (nz + 1, ny + 1, nx + 1) if group == "PointData" else (max(nz, 1), max(ny, 1), max(nx, 1))
        if untimed_ok and tuple(ds.shape[:3]) == expect:
            return "untimed"                  # static array of a file without steps (layout section 2.2)
        if ds.ndim < 4 or tuple(ds.shape[1:4]) != expect:
            self.err("array-shape", ds.name, f"shape {ds.shape} does not match (time, {', '.join(map(str, expect))}[, c])")

    # field datasets ----------------------------------------------------------
    def _field_datasets(self) -> list[h5py.Dataset]:
        """Datasets in /VTKHDF/PointData and CellData, plus any dataset elsewhere carrying a ``kind`` attribute."""
        out: list[h5py.Dataset] = []
        for group in ("VTKHDF/PointData", "VTKHDF/CellData"):
            g = self.f.get(group)
            if isinstance(g, h5py.Group):
                out += [g[k] for k in g if isinstance(g.get(k, getlink=True), h5py.HardLink) and isinstance(g[k], h5py.Dataset)]
        for name, ds in self._hard_datasets():
            if not name.startswith("VTKHDF/") and "kind" in ds.attrs:
                out.append(ds)
        return out

    def _hard_datasets(self) -> list[tuple[str, h5py.Dataset]]:
        found: list[tuple[str, h5py.Dataset]] = []

        def visit(name: str, link: Any) -> None:
            if isinstance(link, h5py.HardLink):
                obj = self.f.get(name)
                if isinstance(obj, h5py.Dataset):
                    found.append((name, obj))
        self.f.visititems_links(visit)
        return found

    def fields(self) -> None:
        fa = self.s["field_attributes"]
        dt = self.s["dtypes"]
        phase_ok = set(self.v["phase_id_dtypes"])
        for ds in self._field_datasets():
            path, attrs = ds.name, ds.attrs
            missing = [a for a in fa["required"] if a not in attrs]
            if missing:
                self.err("field-attribute-missing", path, f"field lacks attribute(s) {', '.join(missing)}")
            loc, kind = _text(attrs.get("location")), _text(attrs.get("kind"))
            if loc is not None and loc not in fa["location"]:
                self.err("field-attribute-value", path, f"location = {loc!r} is not one of {fa['location']}")
            if kind is not None and kind not in fa["kind"]:
                self.err("field-attribute-value", path, f"kind = {kind!r} is not one of {fa['kind']}")
            for key in ("derived_method", "measure", "shear"):
                val = _text(attrs.get(key))
                if val is not None and val not in fa[key]:
                    self.err("field-attribute-value", path, f"{key} = {val!r} is not one of {fa[key]}")
            # precision
            if kind in dt["float64_required"]:
                if path.rsplit("/", 1)[-1] == "phase_id" and ds.dtype.name in phase_ok:
                    pass
                elif ds.dtype != np.float64:
                    self.err("dtype", path, f"kind = {kind} field is {ds.dtype}, must be float64")
            elif kind in dt["float32_allowed"] and ds.dtype.kind == "f" and ds.dtype.itemsize < 4:
                self.err("dtype", path, f"kind = {kind} field is {ds.dtype}; at least float32 is required")
            in_view = path.startswith("/VTKHDF/")
            comps = _text(attrs.get("components"))
            if in_view and isinstance(comps, str):
                names = [c.strip() for c in comps.split(",")]
                nc = _ncomp(ds, self.image)
                if len(names) != nc:
                    self.err("components-count", path, f"components {comps!r} name {len(names)} components, the array stores {nc}")
                if names == ["x", "y"]:
                    self.warn("vector-2-components", path,
                              "2-component vector in /VTKHDF; ParaView warps it wrongly. Store x,y,z with z = 0")
            if path.startswith("/quadrature/") and ds.shape and ds.shape[0] < self.nsteps:
                self.err("array-short", path, f"{ds.shape[0]} time entries, NSteps is {self.nsteps}")
        self._history_per_point()

    def _history_per_point(self) -> None:
        cd = self.f.get("VTKHDF/CellData")
        blocks = self.f.get("mesh/cell_blocks")
        if not isinstance(cd, h5py.Group) or not isinstance(blocks, h5py.Group):
            return
        multi = []
        for bname, b in blocks.items():
            if not isinstance(b, h5py.Group):
                continue
            ctype = _text(b.attrs.get("cell_type"))
            pts = b.get("quadrature/points")
            if isinstance(pts, h5py.Dataset) and pts.ndim >= 1:
                nqp = int(pts.shape[0])
            else:
                nqp = min(self.s["cell_types"].get(ctype, {}).get("quadrature_points", [1]))
            if nqp > 1:
                multi.append(f"{bname} ({ctype}, {nqp} points)")
        if not multi:
            return
        for name in cd:
            ds = cd.get(name)
            if (isinstance(ds, h5py.Dataset) and _text(ds.attrs.get("kind")) == "history"
                    and _text(ds.attrs.get("location")) == "cell"):
                self.warn("history-cell-mean", ds.name,
                          f"history stored per cell while cell block(s) {', '.join(multi)} keep it per quadrature point; "
                          "averaging makes recomputation inexact")

    def coupling(self) -> None:
        if _text(self.f.attrs.get("history_coupling")) != "lagged":
            return
        name = self.s["history_variables"]["damage_input"]["name"]
        if not any(f"VTKHDF/{g}/{name}" in self.f for g in ("CellData", "PointData")):
            self.err("damage-input-missing", "/VTKHDF",
                     f"history_coupling = lagged requires {name} in /VTKHDF/CellData or /VTKHDF/PointData")

    # materials ---------------------------------------------------------------
    def materials(self) -> None:
        phases = self.f.get("materials/phases")
        if not isinstance(phases, h5py.Group):
            return

        def check(obj: Any, path: str) -> None:
            for key, val in obj.attrs.items():
                arr = np.asarray(val)
                if arr.dtype.kind in "iuf" and arr.dtype != np.float64:
                    self.err("dtype", path, f"material value {key} is {arr.dtype}, must be float64")
            if isinstance(obj, h5py.Dataset) and obj.dtype.kind in "iuf" and obj.dtype != np.float64:
                self.err("dtype", path, f"material dataset is {obj.dtype}, must be float64")

        check(phases, phases.name)
        phases.visititems(lambda name, obj: check(obj, obj.name))

    # history -----------------------------------------------------------------
    def history(self) -> None:
        h = self.f.get("history")
        if not isinstance(h, h5py.Group):
            return
        cols = self.s["history_columns"]
        if _text(self.f.attrs.get("profile")) != "setup":
            for name in cols["required"]:
                if name not in h:
                    self.err("history-column-missing", f"/history/{name}", f"required history column {name} is missing")
        integer_ok = set(self.v["integer_history_columns"])
        long_tables = self.s.get("history_tables", {})            # 2.1: crack_tips, substeps
        tables: dict[str, list[h5py.Dataset]] = {}

        def visit(name: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            leaf = name.rsplit("/", 1)[-1]
            first = name.split("/", 1)[0] if "/" in name else ""
            table = first if first in long_tables else ""
            tables.setdefault(table, []).append(obj)
            table_ints = set(long_tables[table]["integer"]) if table in long_tables else integer_ok
            if "units" not in obj.attrs:
                self.err("history-units", obj.name, "history dataset has no units attribute")
            kind = obj.dtype.kind
            if kind == "f" and obj.dtype != np.float64:
                self.err("dtype", obj.name, f"history column is {obj.dtype}, must be float64")
            elif kind in "iub":
                if "codes" in obj.attrs:
                    if obj.dtype != np.dtype(self.s["categorical_encoding"]["dtype"]):
                        self.err("dtype", obj.name, f"categorical column is {obj.dtype}, must be int8")
                    try:
                        json.loads(_text(obj.attrs["codes"]))
                    except (TypeError, ValueError):
                        self.warn("categorical-codes", obj.name, "'codes' is not a JSON object")
                elif leaf not in table_ints:
                    self.err("dtype", obj.name, f"history column is {obj.dtype}; only counters, indices and flags "
                                                f"({', '.join(sorted(table_ints))}) may be integer, others float64")
            elif kind not in "f":
                self.err("dtype", obj.name, f"history column has dtype {obj.dtype}, expected float64 or an integer counter")
        h.visititems(visit)
        for table, dsets in tables.items():
            lengths = {d.shape[0] if d.shape else 0 for d in dsets}
            if len(lengths) > 1:
                where = f"/history/{table}" if table else "/history"
                self.err("history-length", where, f"history columns have different row counts {sorted(lengths)}")
        self._step_index(h)

    def _step_index(self, h: h5py.Group) -> None:
        ds = h.get("step_index")
        if not isinstance(ds, h5py.Dataset) or ds.ndim != 1:
            return
        si = np.asarray(ds[...], np.int64)
        stored_rows = np.flatnonzero(si >= 0)
        stored = si[stored_rows]
        if stored.size and not np.array_equal(stored, np.arange(stored.size)):
            self.err("step-index-order", ds.name, "stored step indices are not 0, 1, 2, ... in increment order")
        if stored.size and int(stored.max()) >= self.nsteps:
            self.err("step-index-range", ds.name, f"step_index refers to step {int(stored.max())}, NSteps is {self.nsteps}")
        if (si < -1).any():
            self.err("step-index-range", ds.name, "step_index values below -1")
        gaps = np.diff(stored_rows)
        if gaps.size >= 2:
            every = self._configured_every()
            k = every or int(gaps[0])
            body, last = gaps[:-1], int(gaps[-1])
            if (body != k).any() or last > k:
                self.warn("step-index-gaps", ds.name,
                          f"stored increments are irregular (gaps {gaps.tolist()}); "
                          + (f"output.result.every = {every} does not explain them" if every else
                             "state the storage cadence as output.result.every in the configuration"))

    def _configured_every(self) -> int | None:
        cfg = self.f.get("meta/config_resolved_yaml")
        if not isinstance(cfg, h5py.Dataset):
            return None
        try:
            data = yaml.safe_load(_text(cfg[()]))
            every = data["output"]["result"]["every"]
            return int(every) if every else None
        except Exception:
            return None


    # layout 2.1 items (each is a no-op for a file checked against the 2.0 schema) ------------------
    def _history_increments(self) -> np.ndarray | None:
        ds = self.f.get("history/increment")
        return np.asarray(ds[...], np.int64) if isinstance(ds, h5py.Dataset) and ds.ndim == 1 else None

    def long_tables(self) -> None:
        """``history/crack_tips`` and ``history/substeps``: required columns, shapes and links to the increments."""
        cfg = self.s.get("history_tables", {})
        h = self.f.get("history")
        if not isinstance(h, h5py.Group):
            return
        increments = self._history_increments()
        for name, rule in cfg.items():
            g = h.get(name)
            if not isinstance(g, h5py.Group):
                continue
            path = f"/history/{name}"
            missing = [c for c in rule["required"] if c not in g]
            if missing:
                self.err("table-column-missing", path, f"table {name} lacks required column(s) {', '.join(missing)}")
                continue
            n = g[rule["required"][0]].shape[0]
            inc = np.asarray(g["increment"][...], np.int64)
            if increments is not None and inc.size and not np.isin(inc, increments).all():
                self.err("table-increment", f"{path}/increment",
                         "rows refer to increments that are not rows of /history/increment")
            if inc.size > 1 and (np.diff(inc) < 0).any():
                self.err("table-order", f"{path}/increment", "rows are not in increment order")
            if name == "crack_tips":
                self._crack_tips(g, rule, n, path)
            else:
                call = np.asarray(g["call"][...], np.int64)
                if (call < 0).any():
                    self.err("table-value", f"{path}/call", "call numbers are negative")
                if n and any(g[c].dtype.kind == "i" and "codes" in g[c].attrs for c in g):
                    for c in g:
                        if "codes" in g[c].attrs:
                            try:
                                codes = json.loads(_text(g[c].attrs["codes"]))
                                used = set(np.unique(g[c][...]).tolist())
                                if not used <= {int(k) for k in codes}:
                                    self.err("categorical-codes", g[c].name, "values without an entry in 'codes'")
                            except (TypeError, ValueError):
                                self.warn("categorical-codes", g[c].name, "'codes' is not a JSON object")

    def _crack_tips(self, g: h5py.Group, rule: dict, n: int, path: str) -> None:
        pos = g["position"]
        if pos.ndim != 2 or pos.shape[1] != 3:
            self.err("table-shape", pos.name, f"position has shape {pos.shape}, expected (n, 3)")
        else:
            comps = _text(pos.attrs.get("components"))
            if comps is not None and [c.strip() for c in str(comps).split(",")] != rule["position_components"]:
                self.err("components-count", pos.name, f"components {comps!r}; expected x,y,z")
            if "components" not in pos.attrs:
                self.err("field-attribute-missing", pos.name, "position lacks the components attribute")
        if (np.asarray(g["tip_id"][...]) < 0).any():
            self.err("table-value", f"{path}/tip_id", "tip ids are negative")
        thr = np.asarray(g["threshold"][...], float)
        if thr.size and not ((thr > 0) & (thr <= 1)).all():
            self.err("table-value", f"{path}/threshold", "damage thresholds must lie in (0, 1]")
        sp = g.get("speed")
        if isinstance(sp, h5py.Dataset):
            if sp.shape[0] > n:
                self.err("history-length", sp.name, f"speed has {sp.shape[0]} rows, the table has {n}")
            for a in ("fit_method", "fit_window", "fit_polyorder"):
                if a not in sp.attrs:
                    self.err("derived-fit", sp.name, f"speed is derived and lacks the attribute {a}")

    def loading(self) -> None:
        kinds = self.s.get("loading", {}).get("bc_kinds")
        bc = self.f.get("loading/bc")
        if not kinds or not isinstance(bc, h5py.Group):
            return
        for name, g in bc.items():
            if not isinstance(g, h5py.Group):
                continue
            kind = _text(g.attrs.get("kind"))
            if kind is None:
                self.err("bc-kind", g.name, "boundary condition has no kind attribute")
            elif kind not in kinds:
                self.err("bc-kind", g.name, f"kind = {kind!r} is not one of {kinds}")
            elif kind == "velocity" and "components" not in g.attrs:
                self.err("bc-kind", g.name, "a velocity condition needs its components")

    def initial_conditions(self) -> None:
        g = self.f.get("initial_conditions")
        spec = self.s.get("initial_conditions")
        if spec is None or not isinstance(g, h5py.Group):
            return
        for name, ds in g.items():
            if not isinstance(ds, h5py.Dataset):
                continue
            if name not in spec["fields"]:
                self.warn("initial-field", ds.name, f"initial field {name!r} is not one of {spec['fields']}")
            loc = _text(ds.attrs.get("location"))
            n = self.n_points if loc == "node" else self.n_cells
            if self.varying or n is None or loc not in ("node", "cell", "quadrature_point"):
                continue
            if self.image:
                nx, ny, nz = self.grid or (0, 0, 0)
                expect = (nz + 1, ny + 1, nx + 1) if loc == "node" else (max(nz, 1), max(ny, 1), max(nx, 1))
                if tuple(ds.shape[:3]) != expect:
                    self.err("initial-shape", ds.name, f"shape {ds.shape} does not start with the grid shape {expect}")
            elif ds.shape[0] != n:
                self.err("initial-shape", ds.name, f"{ds.shape[0]} rows, the mesh has {n} {'points' if loc == 'node' else 'cells'}")

    def wave_speeds(self) -> None:
        spec = self.s.get("material_wave_speeds")
        phases = self.f.get("materials/phases")
        if spec is None or not isinstance(phases, h5py.Group):
            return
        analysis = _text(self.f.attrs.get("analysis"))
        for pid, g in phases.items():
            if not isinstance(g, h5py.Group):
                continue
            present = [a for a in ("c_p", "c_s", "c_R") if a in g.attrs]
            if not present:
                continue
            if "rho" not in g.attrs:
                self.err("wave-speed", g.name, "wave speeds are given without the density rho")
                continue
            units = _text(g.attrs.get("units"))
            try:
                units = json.loads(units) if isinstance(units, str) else {}
            except ValueError:
                units = {}
            for a in ("rho", *present):
                if units.get(a) != spec["units"][a]:
                    self.err("wave-speed-units", g.name, f"units of {a} must be {spec['units'][a]!r}, found {units.get(a)!r}")
            if _text(g.attrs.get("elastic_model")) != "isotropic" or not {"E", "nu"} <= set(g.attrs):
                continue
            e, nu, rho = float(g.attrs["E"]), float(g.attrs["nu"]), float(g.attrs["rho"])
            mu = e / (2.0 * (1.0 + nu))
            lam = (e * nu / (1.0 - nu * nu)) if analysis == "plane_stress" else e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            c_s = (mu / rho) ** 0.5
            ref = {"c_s": c_s, "c_p": ((lam + 2.0 * mu) / rho) ** 0.5, "c_R": c_s * (0.862 + 1.14 * nu) / (1.0 + nu)}
            for a in present:
                if abs(float(g.attrs[a]) - ref[a]) > spec["relative_tolerance"] * ref[a]:
                    self.err("wave-speed", g.name, f"{a} = {float(g.attrs[a])!r} differs from {ref[a]!r}, the value of "
                                                   f"E, nu, rho under {analysis}")

    def validation(self) -> None:
        spec = self.s.get("validation")
        g = self.f.get("meta/validation")
        if spec is None or not isinstance(g, h5py.Group):
            return
        missing = [k for k in spec["datasets"] if k not in g]
        if missing:
            self.err("validation-column-missing", g.name, f"meta/validation lacks {', '.join(missing)}")
            return
        lengths = {k: g[k].shape[0] for k in spec["datasets"]}
        if len(set(lengths.values())) > 1:
            self.err("validation-length", g.name, f"columns have different row counts {lengths}")
        for key in ("value", "tolerance"):
            if g[key].dtype != np.float64:
                self.err("dtype", g[key].name, f"{key} is {g[key].dtype}, must be float64")
        passed = np.asarray(g["passed"][...])
        if g["passed"].dtype.kind not in "iub" or (passed.size and not np.isin(passed, [0, 1]).all()):
            self.err("validation-value", g["passed"].name, "passed holds 0 or 1")
        if "overall_passed" in g.attrs and passed.size and int(g.attrs["overall_passed"]) != int(passed.all()):
            self.err("validation-value", g.name, "overall_passed disagrees with the rows")
        for k in spec["datasets"]:
            if "name" == k and g[k].dtype.kind not in "OSU":
                self.err("dtype", g[k].name, "name is a text column")

    def varying_mesh(self) -> None:
        if not self.varying:
            return
        spec = self.s["varying_mesh"]
        v = self.f.get("VTKHDF")
        if not isinstance(v, h5py.Group):
            return
        if self.image:
            self.err("varying-mesh", "/", "mesh_varies = true needs an UnstructuredGrid; an ImageData grid is fixed")
            return
        for grp in ("quadrature", "derived", "interfaces"):
            g = self.f.get(grp)
            if isinstance(g, h5py.Group) and len(g):
                self.err("varying-mesh", f"/{grp}", f"/{grp} holds arrays of a fixed shape; they are not supported "
                                                    "with a changing mesh (store the field inside /VTKHDF)")
        counts = {k: np.asarray(v[k][...], np.int64) for k in ("NumberOfPoints", "NumberOfCells", "NumberOfConnectivityIds")
                  if k in v}
        if len(counts) != 3 or len({c.size for c in counts.values()}) != 1:
            self.err("varying-mesh", "/VTKHDF", "NumberOfPoints, NumberOfCells and NumberOfConnectivityIds need one entry per mesh version")
            return
        n_ver = counts["NumberOfPoints"].size
        self.versions = list(zip(*(counts[k].tolist() for k in ("NumberOfPoints", "NumberOfCells", "NumberOfConnectivityIds"))))
        need = {"Points": int(counts["NumberOfPoints"].sum()), "Types": int(counts["NumberOfCells"].sum()),
                "Offsets": int((counts["NumberOfCells"] + 1).sum()), "Connectivity": int(counts["NumberOfConnectivityIds"].sum())}
        for name, rows in need.items():
            if name in v and v[name].shape[0] < rows:
                self.err("array-short", v[name].name, f"{v[name].shape[0]} rows, the {n_ver} mesh versions need {rows}")
        before = {k: np.concatenate([[0], np.cumsum(c)[:-1]]) for k, c in counts.items()}
        st = v.get("Steps")
        n = self.nsteps
        if isinstance(st, h5py.Group) and n and all(k in st for k in ("PartOffsets", "NumberOfParts", "PointOffsets",
                                                                        "CellOffsets", "ConnectivityIdOffsets")):
            parts = np.asarray(st["PartOffsets"][:n], np.int64)
            if not (np.asarray(st["NumberOfParts"][:n]) == 1).all():
                self.err("varying-mesh", st.name, "NumberOfParts must be 1 at every step: one mesh version per step")
            if ((parts < 0) | (parts >= n_ver)).any():
                self.err("varying-mesh", f"{st.name}/PartOffsets", f"mesh version outside 0..{n_ver - 1}")
            else:
                for key, ds in (("NumberOfPoints", "PointOffsets"), ("NumberOfCells", "CellOffsets"),
                                ("NumberOfConnectivityIds", "ConnectivityIdOffsets")):
                    got = np.asarray(st[ds][:n], np.int64).reshape(n, -1)[:, 0]
                    if not np.array_equal(got, before[key][parts]):
                        self.err("varying-mesh", f"{st.name}/{ds}",
                                 f"offsets do not point at the first {key[8:].lower()} of the mesh version of each step")
        hist = self.f.get("history")
        mi = hist.get(spec["history_column"]) if isinstance(hist, h5py.Group) else None
        if not isinstance(mi, h5py.Dataset):
            self.err("history-column-missing", f"/history/{spec['history_column']}",
                     "mesh_varies = true requires the history column mesh_index (mesh version of each increment)")
        elif mi.shape[0]:
            idx = np.asarray(mi[...], np.int64)
            if ((idx < 0) | (idx >= n_ver)).any() or (np.diff(idx) < 0).any():
                self.err("varying-mesh", mi.name, "mesh_index must be non-decreasing and a valid mesh version")
            elif isinstance(st, h5py.Group) and n and "PartOffsets" in st and "step_index" in hist:
                si = np.asarray(hist["step_index"][: idx.size], np.int64)
                rows = np.flatnonzero((si >= 0) & (si < n))
                if not np.array_equal(idx[rows], np.asarray(st["PartOffsets"][:n], np.int64)[si[rows]]):
                    self.err("varying-mesh", mi.name, "mesh_index disagrees with the mesh version of the stored step")
        mesh = self.f.get("mesh")
        vers = mesh.get("versions") if isinstance(mesh, h5py.Group) else None
        found = sorted(int(k) for k in vers) if isinstance(vers, h5py.Group) else []
        if found != list(range(1, n_ver)):
            self.err("varying-mesh", "/mesh/versions", f"groups {found}; the {n_ver} mesh versions need 1..{n_ver - 1} "
                                                       "(version 0 is the mesh/ group itself)")
            return
        for k in range(1, n_ver):
            self._mesh_version(vers[str(k)], k, spec)
        for k in range(n_ver):
            self._version_sets(mesh if k == 0 else vers[str(k)], k, self.versions[k][0], self.versions[k][1])

    def _mesh_version(self, g: h5py.Group, k: int, spec: dict) -> None:
        n_pts, n_cells, _ = self.versions[k]
        old_pts, old_cells, _ = self.versions[k - 1]
        missing = [d for d in spec["version_datasets"] if d not in g]
        if missing:
            self.err("mesh-version", g.name, f"mesh version {k} lacks {', '.join(missing)}")
            return
        shapes = {"cell_parent": (n_cells,), "node_origin": (n_pts,), "node_parents": (n_pts, 2)}
        for d, want in shapes.items():
            if tuple(g[d].shape) != want:
                self.err("mesh-version", g[d].name, f"shape {g[d].shape}, expected {want}")
                return
        parent, origin, par = (np.asarray(g[d][...], np.int64) for d in ("cell_parent", "node_origin", "node_parents"))
        if ((parent < -1) | (parent >= old_cells)).any():
            self.err("mesh-version", g["cell_parent"].name, f"cell parents must be -1 or a cell of version {k - 1}")
        if ((origin < -1) | (origin >= old_pts)).any():
            self.err("mesh-version", g["node_origin"].name, f"node origins must be -1 or a node of version {k - 1}")
        new = origin < 0
        if ((par[~new] != -1).any() or (par[new] < 0).any() or (par >= old_pts).any()):
            self.err("mesh-version", g["node_parents"].name,
                     "new nodes need two parent nodes of the previous version, existing nodes none (-1)")
        kept = origin[~new]
        if kept.size != np.unique(kept).size:
            self.err("mesh-version", g["node_origin"].name, "two nodes claim the same origin node")
        hang = g.get("hanging_nodes")
        if isinstance(hang, h5py.Group):
            self._hanging(hang, n_pts)

    def _hanging(self, hang: h5py.Group, n_pts: int) -> None:
        spec = self.s["varying_mesh"]["hanging_node_datasets"]
        missing = [d for d in spec if d not in hang]
        if missing:
            self.err("hanging-nodes", hang.name, f"lacks {', '.join(missing)}")
            return
        con, mas, wts = (np.asarray(hang[d][...]) for d in spec)
        if mas.shape != wts.shape or mas.ndim != 2 or mas.shape[0] != con.shape[0]:
            self.err("hanging-nodes", hang.name, f"shapes {con.shape}, {mas.shape}, {wts.shape} do not agree")
            return
        if wts.dtype != np.float64:
            self.err("dtype", hang["weights"].name, f"weights are {wts.dtype}, must be float64")
        if ((con < 0) | (con >= n_pts)).any() or ((mas < 0) | (mas >= n_pts)).any():
            self.err("hanging-nodes", hang.name, "node indices outside the mesh")
        elif np.isin(mas, con).any():
            self.err("hanging-nodes", hang.name, "a master node is itself constrained")
        if wts.size and not np.allclose(wts.sum(axis=1), 1.0, atol=1e-12):
            self.err("hanging-nodes", hang["weights"].name, "weights of a constrained node must sum to 1")

    def _version_sets(self, g: h5py.Group, k: int, n_pts: int, n_cells: int) -> None:
        blocks = g.get("cell_blocks")
        if isinstance(blocks, h5py.Group) and len(blocks):
            total = 0
            for b in blocks.values():
                if "cells" in b:
                    a, z = (int(x) for x in np.asarray(b["cells"][...]).ravel()[:2])
                    total += z - a
            if total != n_cells:
                self.err("mesh-version", blocks.name, f"cell blocks of version {k} cover {total} cells, the mesh has {n_cells}")
        sets = g.get("node_sets")
        if isinstance(sets, h5py.Group):
            for name, ds in sets.items():
                idx = np.asarray(ds[...])
                if idx.size and (idx.min() < 0 or idx.max() >= n_pts):
                    self.err("mesh-version", ds.name, f"node set {name!r} refers to nodes outside the {n_pts} of version {k}")

    # every object: links, virtual datasets, filters, absolute paths -------------
    def objects(self) -> None:
        allowed = {_FILTER_IDS[name] for name in self.s["file"]["allowed_filters"]}
        suffixes = tuple(self.v["reference_attribute_suffixes"])

        def check_attrs(obj: Any, path: str) -> None:
            for key in obj.attrs:
                if not key.endswith(suffixes):
                    continue
                for text in _strings(obj.attrs[key]):
                    self._absolute(text, path, f"attribute {key}")

        check_attrs(self.f, "/")

        def visit(name: str, link: Any) -> None:
            path = "/" + name
            if isinstance(link, h5py.SoftLink):
                target = link.path if link.path.startswith("/") else posixpath.normpath(
                    posixpath.join(posixpath.dirname(path), link.path))
                if target not in self.f:
                    self.err("dangling-link", path, f"soft link to {link.path} does not resolve")
                elif not (target + "/").startswith("/VTKHDF/"):
                    self.err("link-outside-vtkhdf", path, f"soft link to {link.path} points outside /VTKHDF")
                return
            if isinstance(link, h5py.ExternalLink):
                self.err("external-link", path, f"external link to {link.filename}:{link.path}; links point into /VTKHDF only")
                return
            obj = self.f.get(name)
            if obj is None:
                return
            check_attrs(obj, path)
            if not isinstance(obj, h5py.Dataset):
                return
            if obj.is_virtual:
                self.err("virtual-dataset", path, "virtual datasets are not allowed in stored files")
                return
            for code, fname in _filters(obj):
                if code not in allowed:
                    self.err("filter-not-allowed", path,
                             f"filter {fname or '?'} (id {code}) is not one of the built-in filters "
                             f"none, {', '.join(self.s['file']['allowed_filters'])}")
            if name.rsplit("/", 1)[-1].endswith(suffixes) and obj.dtype.kind in "OSU" and obj.size <= 100_000:
                for text in _strings(obj[()]):
                    self._absolute(text, path, "dataset")
        self.f.visititems_links(visit)

    def _absolute(self, text: str, path: str, what: str) -> None:
        if not text or not _is_absolute_path(text):
            return
        if text.startswith("/") and text in self.f:    # an HDF5 path inside this file, not a file reference
            return
        self.err("absolute-path", path, f"{what} holds the absolute path {text!r}; references must be relative paths or SHA-256 values")


def validate(path: str | Path, allow_running: bool = False) -> Report:
    """Check a result file against layout 2.x and return the findings.

    ``allow_running`` accepts ``status = running`` (a file on scratch whose run
    is active); the validator cannot see where a file lives, so the caller says.
    """
    report = Report(str(path))
    try:
        f = h5py.File(path, "r")
    except OSError as exc:
        report.add("ERROR", "file-unreadable", "",
                   f"cannot open as HDF5 ({exc}); an interrupted file may need 'phast result recover'")
        return report
    with f:
        _Validator(f, report, allow_running).run()
    return report
