"""Recovery of interrupted PhAST result files (layout 2.x).

``recover`` works on a copy and never modifies the source: it clears the
open-for-write flag, trims every time-indexed dataset to the longest
consistent step count, sets ``status = recovered`` and validates the copy.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from .validate import Report, _text, validate

LIBVER = ("earliest", "v114")
_STEP_DATASETS = ("Values", "PartOffsets", "NumberOfParts", "PointOffsets", "CellOffsets", "ConnectivityIdOffsets")


@dataclass
class Recovery:
    source: str
    output: str
    flag_cleared_with: str          # "h5clear" or "h5py" (h5py opens superblock v0 to v2 files with no flag check)
    original_nsteps: int
    nsteps: int
    history_rows: tuple[int, int]   # rows before and after trimming
    trimmed: list[str] = field(default_factory=list)
    report: Report | None = None

    def summary(self) -> str:
        lines = [f"recovered {self.source} -> {self.output}",
                 f"  open-for-write flag cleared with {self.flag_cleared_with}",
                 f"  NSteps {self.original_nsteps} -> {self.nsteps}; history rows {self.history_rows[0]} -> {self.history_rows[1]}",
                 f"  trimmed {len(self.trimmed)} dataset(s)"]
        lines += [f"    {p}" for p in self.trimmed]
        return "\n".join(lines)


def default_output(src: Path) -> Path:
    name = src.name[:-len(".partial")] if src.name.endswith(".partial") else src.name
    stem = name[:-len(".vtkhdf")] if name.endswith(".vtkhdf") else Path(name).stem
    return src.with_name(f"{stem}.recovered.vtkhdf")


def _clear_flag(path: Path, use_h5clear: bool | None) -> str:
    """Clear the superblock status flag; return the means used."""
    exe = shutil.which("h5clear") if use_h5clear is not False else None
    if use_h5clear and exe is None:
        raise FileNotFoundError("h5clear was requested but is not on PATH")
    if exe:
        done = subprocess.run([exe, "-s", str(path)], capture_output=True, text=True)
        if done.returncode == 0:
            return "h5clear"
    # Files written with libver ("earliest", "v114") have a superblock without the status flag,
    # so opening them read-write with h5py succeeds; files with a set flag fail here.
    try:
        with h5py.File(path, "r+", libver=LIBVER):
            pass
    except OSError as exc:
        raise OSError(f"cannot open {path} read-write and h5clear is unavailable or failed: {exc}") from exc
    return "h5py"


def _trim(group: h5py.Group, name: str, rows: int | np.ndarray, trimmed: list[str]) -> None:
    """Keep the first ``rows`` entries (or the rows selected by a boolean mask) of ``group[name]``."""
    ds = group[name]
    if isinstance(rows, np.ndarray):
        mask = np.zeros(ds.shape[0], bool)
        mask[: rows.size] = rows[: ds.shape[0]]
        if mask.all():
            return
        data = ds[...][mask]
    else:
        if ds.shape[0] <= rows:
            return
        data = None
    new_len = rows if data is None else data.shape[0]
    if ds.chunks is not None and (ds.maxshape[0] is None or ds.maxshape[0] >= new_len):
        if data is not None:
            ds[:new_len] = data
        ds.resize(new_len, axis=0)
    else:                                   # contiguous: rewrite with the same dtype, attributes and filters
        data = ds[:new_len] if data is None else data
        attrs = dict(ds.attrs)
        kw = dict(dtype=ds.dtype, chunks=ds.chunks, compression=ds.compression,
                  compression_opts=ds.compression_opts, shuffle=ds.shuffle, fletcher32=ds.fletcher32)
        del group[name]
        new = group.create_dataset(name, data=data, track_times=False, **{k: v for k, v in kw.items() if v})
        for k, v in attrs.items():
            new.attrs[k] = v
    trimmed.append(f"{group.name.rstrip('/')}/{name}")


def _consistent_steps(f: h5py.File) -> tuple[int, int]:
    """(NSteps attribute, longest step count consistent with every Steps dataset and array)."""
    v, st = f["VTKHDF"], f["VTKHDF/Steps"]
    nsteps = int(st.attrs["NSteps"])
    n = nsteps
    for name in _STEP_DATASETS:
        if name in st:
            n = min(n, st[name].shape[0])
    image = _is_image(v)
    varying = _varying(f)
    n_points = n_cells = None
    if not image and "NumberOfPoints" in v:
        parts = int(st["NumberOfParts"][0]) if "NumberOfParts" in st and st["NumberOfParts"].shape[0] else 1
        n_points, n_cells = int(np.sum(v["NumberOfPoints"][:max(parts, 1)])), int(np.sum(v["NumberOfCells"][:max(parts, 1)]))
    for group, ogroup, rows in (("PointData", "PointDataOffsets", n_points), ("CellData", "CellDataOffsets", n_cells)):
        if group not in v or ogroup not in st:
            continue
        for name, ods in st[ogroup].items():
            n = min(n, ods.shape[0])
            if name not in v[group]:
                continue
            per = 1 if image else rows
            if per is None:
                continue
            offs = ods[:n]
            if varying and "PartOffsets" in st:
                per = _rows_of_steps(v, st, group, n)       # the mesh version of each step sets its rows
            complete = offs + per <= v[group][name].shape[0]
            # the longest prefix of steps whose rows are all present
            n = min(n, int(np.argmin(complete)) if not complete.all() else n)
    return nsteps, max(n, 0)


def trim_steps(f: h5py.File, n: int, trimmed: list[str]) -> None:
    """Keep the first ``n`` stored steps of an open file: ``Steps``, the field rows, ``quadrature`` and their offsets."""
    v, st = f["VTKHDF"], f["VTKHDF/Steps"]
    image = _is_image(v)
    for name in _STEP_DATASETS:
        if name in st:
            _trim(st, name, n, trimmed)
    for group, ogroup in (("PointData", "PointDataOffsets"), ("CellData", "CellDataOffsets")):
        if ogroup not in st:
            continue
        for name in list(st[ogroup]):
            ods = st[ogroup][name]
            if group in v and name in v[group] and n > 0:
                if _varying(f):
                    keep = int(np.max(ods[:n] + _rows_of_steps(v, st, group, n)))
                else:
                    keep = int(np.max(ods[:n])) + _rows_per_step(v, group, image)
                _trim(v[group], name, keep, trimmed)
            _trim(st[ogroup], name, n, trimmed)
    st.attrs["NSteps"] = np.int64(n)
    if "quadrature" in f:
        for block in f["quadrature"].values():
            for name in list(block):
                if isinstance(block[name], h5py.Dataset):
                    _trim(block, name, n, trimmed)


def recover(src: str | Path, dst: str | Path | None = None, *, use_h5clear: bool | None = None,
            overwrite: bool = False) -> Recovery:
    """Recover an interrupted result file into a copy and validate the copy.

    ``use_h5clear``: None uses ``h5clear -s`` when it is on PATH and falls back
    to h5py; True requires it; False uses h5py only.
    """
    src = Path(src)
    dst = default_output(src) if dst is None else Path(dst)
    if dst.resolve() == src.resolve():
        raise ValueError("recover writes a copy; the output must differ from the source")
    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst} exists; pass overwrite=True (CLI: --force) to replace it")
    shutil.copyfile(src, dst)
    cleared = _clear_flag(dst, use_h5clear)

    trimmed: list[str] = []
    with h5py.File(dst, "r+", libver=LIBVER) as f:
        original, n = _consistent_steps(f)
        trim_steps(f, n, trimmed)
        rows = _trim_history(f, n, trimmed)
        f.attrs["status"] = "recovered"
        f.attrs["recovered_from_steps"] = np.int64(original)
        f.attrs["finished"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    result = Recovery(str(src), str(dst), cleared, original, n, rows, trimmed)
    result.report = validate(dst)
    return result


def _varying(f: h5py.File) -> bool:
    return bool(f.attrs.get("mesh_varies", False))


def _rows_of_steps(v: h5py.Group, st: h5py.Group, group: str, n: int) -> np.ndarray:
    """Rows of each of the first ``n`` steps of a changing mesh: the points or cells of its mesh version."""
    counts = np.asarray(v["NumberOfPoints" if group == "PointData" else "NumberOfCells"][...], np.int64)
    parts = np.asarray(st["PartOffsets"][:n], np.int64)
    return counts[np.clip(parts, 0, max(counts.size - 1, 0))]


def _is_image(v: h5py.Group) -> bool:
    return _text(v.attrs.get("Type")) == "ImageData"


def _rows_per_step(v: h5py.Group, group: str, image: bool) -> int:
    if image:
        return 1
    key = "NumberOfPoints" if group == "PointData" else "NumberOfCells"
    return int(v[key][0])


def _trim_history(f: h5py.File, n: int, trimmed: list[str], rows: int | None = None) -> tuple[int, int]:
    """Keep history rows up to the last row that refers to a kept step; trim the long tables and derived to match.

    ``rows`` keeps exactly that many history rows instead (resume at a given increment); ``n`` is then the
    number of stored steps the kept rows refer to.
    """
    h = f.get("history")
    if not isinstance(h, h5py.Group) or "step_index" not in h:
        return (0, 0)
    main: list[tuple[h5py.Group, str]] = []
    long_tables: dict[str, list[tuple[h5py.Group, str]]] = {}     # crack_tips, substeps: rows per event

    def visit(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset):
            parent, leaf = (name.rsplit("/", 1) if "/" in name else ("", name))
            entry = (h[parent] if parent else h, leaf)
            table = name.split("/", 1)[0] if parent in ("crack_tips", "substeps") else None
            (long_tables.setdefault(table, []) if table else main).append(entry)
    h.visititems(visit)
    before = min(g[k].shape[0] for g, k in main)
    si = h["step_index"][:before]
    kept_rows = np.flatnonzero((si >= 0) & (si < n))
    keep = int(kept_rows[-1]) + 1 if kept_rows.size else 0
    if rows is not None:
        keep = min(int(rows), before)
    for g, k in main:
        _trim(g, k, keep, trimmed)
    last_increment = int(h["increment"][keep - 1]) if keep and "increment" in h else -1
    for table, members in long_tables.items():
        if f"{table}/increment" not in h:
            continue
        count = min(g[k].shape[0] for g, k in members)
        mask = h[f"{table}/increment"][:count] <= last_increment
        cut = len(trimmed)
        for g, k in members:
            _trim(g, k, mask, trimmed)
        if table == "crack_tips" and len(trimmed) > cut and "speed" in h["crack_tips"]:
            del h["crack_tips/speed"]             # the fit of the remaining rows differs: readers recompute it
    derived = f.get("derived")
    if isinstance(derived, h5py.Group):
        for g in derived.values():
            if isinstance(g, h5py.Group) and "step_index" in g:
                _trim(g, "step_index", keep, trimmed)
                idx = g["step_index"][...]
                if "values" in g:
                    _trim(g, "values", int(idx.max()) + 1 if idx.size and idx.max() >= 0 else 0, trimmed)
    return (before, keep)
