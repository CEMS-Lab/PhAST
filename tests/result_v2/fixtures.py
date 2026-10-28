"""Reference builder for the layout-2.0 conformance fixtures.

Deliberately simple and independent of ``src/phast/io/result``: it writes each
file directly with h5py from a plain description, step by step in the write
order of the specification, so the writer can be checked against
files it did not produce. Every fixture comes with an expected-values entry
(step count, sizes, probe values) used by the h5py tests and by the ParaView
check; probe values encode their indices so that transposed or misordered
storage cannot pass.

Usage: ``python -m result_v2.fixtures <out_dir>`` writes all fixtures and
``expected.json``. Data are synthetic; the fixtures test the format, not physics.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

SCHEMA_VERSION = "2.0.0"
LIBVER = ("earliest", "v114")
CREATED = "2026-10-26T12:00:00+00:00"
VTK = {"T3": 5, "Q4": 9, "H8": 12}
TENSOR_2D = "xx,yy,xy"
TENSOR_3D = "xx,yy,zz,yz,xz,xy"


class RefusedLayout(ValueError):
    """Raised for descriptions layout 2.0 does not allow (for example a changing mesh)."""


# --- meshes ------------------------------------------------------------------

def tri_grid(nx, ny, lx=10.0, ly=8.0):
    xs, ys = np.linspace(0, lx, nx + 1), np.linspace(0, ly, ny + 1)
    pts = np.array([[x, y, 0.0] for y in ys for x in xs])
    cells = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i; b, c, d = a + 1, a + nx + 2, a + nx + 1
            cells += [[a, b, c], [a, c, d]]
    return pts, np.array(cells, np.int64)


def quad_grid(nx, ny, lx=9.0, ly=9.0):
    xs, ys = np.linspace(0, lx, nx + 1), np.linspace(0, ly, ny + 1)
    pts = np.array([[x, y, 0.0] for y in ys for x in xs])
    cells = [[j * (nx + 1) + i, j * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i]
             for j in range(ny) for i in range(nx)]
    return pts, np.array(cells, np.int64)


def edges_of(cells):
    e = set()
    for c in cells:
        for a, b in zip(c, np.roll(c, -1)):
            e.add((int(a), int(b))); e.add((int(b), int(a)))
    return np.array(sorted(e), np.int64).T


# --- values that encode their own position ------------------------------------

def coded(n_steps, n, ncomp=None, base=0.0):
    """Value = base + 1000*step + index + 0.01*component (component axis optional)."""
    s = np.arange(n_steps)[:, None]; i = np.arange(n)[None, :]
    v = base + 1000.0 * s + i
    if ncomp is None:
        return v.astype(float)
    return (v[..., None] + 0.01 * np.arange(ncomp)[None, None, :]).astype(float)


def per_point(n_steps, n_cells, n_qp, base=0.0):
    """Value = base + 1000*step + 10*cell + quadrature point (point index readable)."""
    s = np.arange(n_steps)[:, None, None]; c = np.arange(n_cells)[None, :, None]; q = np.arange(n_qp)[None, None, :]
    return (base + 1000.0 * s + 10.0 * c + q).astype(float)


# --- writer ------------------------------------------------------------------

def _set_attrs(obj, attrs):
    for k, v in (attrs or {}).items():
        obj.attrs[k] = v


def _ds(group, name, data, attrs=None, **kw):
    d = group.create_dataset(name, data=data, track_times=False, **kw)
    _set_attrs(d, attrs)
    return d


def _resizable(group, name, shape_tail, dtype, chunk_rows, attrs=None):
    d = group.create_dataset(name, shape=(0,) + tuple(shape_tail), maxshape=(None,) + tuple(shape_tail),
                             chunks=(max(1, chunk_rows),) + tuple(shape_tail), dtype=dtype, track_times=False)
    _set_attrs(d, attrs)
    return d


def write_fixture(path, desc, kill=None):
    """Write one file from ``desc``. ``kill=(step, stage)`` exits without closing the file.

    stage: ``after_rows`` | ``after_offsets`` | ``after_nsteps``.
    """
    if desc["root"].get("mesh_varies"):
        raise RefusedLayout("layout 2.0 refuses changing meshes (mesh_varies = true)")
    path = Path(path)
    f = h5py.File(path, "w", libver=LIBVER, track_order=True)
    root = dict(phast_schema="phast.result", phast_schema_version=SCHEMA_VERSION, unit_system="mm-MPa-N-s",
                created=CREATED, mesh_varies=False, index_basis="increment", kinematics="small",
                history_update="hard_max", history_coupling="fresh", status="running")
    root.update({k: v for k, v in desc["root"].items() if k != "final_status"}); root["mesh_varies"] = False
    _set_attrs(f, root)

    v = f.create_group("VTKHDF", track_order=True)
    v.attrs["Version"] = np.array([2, 2])
    image = desc["kind"] == "image"
    if image:
        v.attrs.create("Type", np.bytes_("ImageData"))
        nx, ny, nz = desc["grid"]
        v.attrs["WholeExtent"] = np.array([0, nx, 0, ny, 0, nz])
        v.attrs["Origin"] = np.zeros(3); v.attrs["Spacing"] = np.array(desc["spacing"], float)
        v.attrs["Direction"] = np.eye(3).ravel()
        n_pts, n_cells = (nx + 1) * (ny + 1) * (nz + 1), nx * ny * nz
    else:
        v.attrs.create("Type", np.bytes_("UnstructuredGrid"))
        pts, cells, types = desc["points"], desc["cells"], desc["types"]
        conn = np.concatenate([np.asarray(c) for c in cells]); off = np.concatenate([[0], np.cumsum([len(c) for c in cells])])
        n_pts, n_cells = len(pts), len(cells)
        for k, d in [("NumberOfPoints", [n_pts]), ("NumberOfCells", [n_cells]), ("NumberOfConnectivityIds", [len(conn)]),
                     ("Points", np.asarray(pts, float)), ("Connectivity", conn.astype(np.int64)),
                     ("Offsets", off.astype(np.int64)), ("Types", np.asarray(types, np.uint8))]:
            _ds(v, k, d)
    pd, cd = v.create_group("PointData", track_order=True), v.create_group("CellData", track_order=True)
    st = v.create_group("Steps"); st.attrs["NSteps"] = 0
    vals = _resizable(st, "Values", (), float, 256)
    offs = {k: _resizable(st, k, sh, np.int64, 256) for k, sh in
            [("PartOffsets", ()), ("NumberOfParts", ()), ("PointOffsets", ()), ("CellOffsets", (1,)), ("ConnectivityIdOffsets", (1,))]}
    pdo, cdo = st.create_group("PointDataOffsets"), st.create_group("CellDataOffsets")

    # static arrays: written once, offset 0 at every step (ImageData needs a time axis of length 1)
    statics = []
    for name, (arr, attrs) in desc.get("static_cell", {}).items():
        a = np.asarray(arr)
        _ds(cd, name, a[None] if image else a, attrs)
        statics.append((_resizable(cdo, name, (), np.int64, 256), 0))

    # time-varying arrays
    series = []
    for where, grp, ogrp, n in (("point", pd, pdo, n_pts), ("cell", cd, cdo, n_cells)):
        for name, (arr, attrs, dtype) in desc.get(f"{where}_fields", {}).items():
            arr = np.asarray(arr, dtype)
            tail = arr.shape[1:]
            if image:
                ds = _resizable(grp, name, tail, dtype, 1, attrs)
            else:
                ds = _resizable(grp, name, tail[1:], dtype, n, attrs)
            series.append((ds, arr, _resizable(ogrp, name, (), np.int64, 256), n))

    _write_groups(f, desc, n_pts, n_cells)
    hist = desc.get("history", {})
    hgroup = f.require_group("history")
    hcols = {}
    for name, (col, attrs) in hist.items():
        col = np.asarray(col)
        hcols[name] = (_resizable(hgroup, name, col.shape[1:], col.dtype, 1024, attrs), col)

    step_of_increment = np.asarray(hist["step_index"][0])
    times = desc["step_times"]
    stored = 0
    for inc in range(len(step_of_increment)):
        s = int(step_of_increment[inc])
        if s >= 0:
            for ds, arr, ods, n in series:                       # 1. rows of the new step
                ds.resize(ds.shape[0] + (1 if image else n), axis=0)
                if image:
                    ds[-1] = arr[s]
                else:
                    ds[-n:] = arr[s]
            if kill == (s, "after_rows"):
                f.flush(); os._exit(0)
            for k, ds in offs.items():                           # 2. offsets and values
                ds.resize(s + 1, axis=0); ds[s] = 1 if k == "NumberOfParts" else 0
            for ds, arr, ods, n in series:
                ods.resize(s + 1, axis=0); ods[s] = s * (1 if image else n)
            for ods, o in statics:
                ods.resize(s + 1, axis=0); ods[s] = o
            vals.resize(s + 1, axis=0); vals[s] = times[s]
            if kill == (s, "after_offsets"):
                f.flush(); os._exit(0)
            st.attrs["NSteps"] = s + 1                           # 3. step count
            stored = s + 1
            if kill == (s, "after_nsteps"):
                f.flush(); os._exit(0)
        for name, (ds, col) in hcols.items():                    # 4. history row
            ds.resize(inc + 1, axis=0); ds[inc] = col[inc]
        f.flush()                                                # 5. flush
    f.attrs["status"] = desc["root"].get("final_status", "complete")
    f.attrs["finished"] = CREATED
    f.close()
    return stored


def _write_groups(f, desc, n_pts, n_cells):
    mesh = f.require_group("mesh")
    for name, b in desc.get("cell_blocks", {}).items():
        g = mesh.require_group(f"cell_blocks/{name}")
        _set_attrs(g, dict(cell_type=b["cell_type"], vtk_type=VTK[b["cell_type"]], node_order="vtk"))
        _ds(g, "cells", np.asarray(b["range"], np.int64))
        q = g.require_group("quadrature")
        _ds(q, "points", np.asarray(b["qp_points"], float)); _ds(q, "weights", np.asarray(b["qp_weights"], float))
        q.attrs["rule"] = b["rule"]; q.attrs["point_order"] = b["point_order"]
    if "grid" in desc:
        g = mesh.require_group("grid")
        _ds(g, "shape", np.asarray(desc["grid"], np.int64)); _ds(g, "spacing", np.asarray(desc["spacing"], float))
        _ds(g, "origin", np.zeros(3)); g.attrs["index_order"] = "x fastest"
    for name, idx in desc.get("node_sets", {}).items():
        _ds(mesh.require_group("node_sets"), name, np.asarray(idx, np.int64))
    if "edge_index" in desc:
        _ds(mesh.require_group("graph"), "edge_index", desc["edge_index"])
    mat = f.require_group("materials")
    for pid, attrs in desc["phases"].items():
        _set_attrs(mat.require_group(f"phases/{pid}"), attrs)
    if "phase_id" in desc.get("static_cell", {}):
        mat["phase_id"] = h5py.SoftLink("/VTKHDF/CellData/phase_id")
    load = f.require_group("loading")
    for k, attrs in desc.get("bc", {}).items():
        _set_attrs(load.require_group(f"bc/{k}"), attrs)
    if "periodic" in desc:
        _set_attrs(load.require_group("periodic"), desc["periodic"])
    state = f.require_group("state")
    for name, target in desc.get("state_links", {}).items():
        state[name] = h5py.SoftLink(target)
    for path, (arr, attrs) in desc.get("extra", {}).items():
        parent, leaf = path.rsplit("/", 1)
        _ds(f.require_group(parent), leaf, arr, attrs)
    for path, attrs in desc.get("extra_attrs", {}).items():
        _set_attrs(f.require_group(path), attrs)
    meta = f.require_group("meta")
    _ds(meta, "config_resolved_yaml", desc.get("config", "fixture: true\n"))
    _ds(meta, "schema_yaml", f"schema: phast.result\nversion: {SCHEMA_VERSION}\n")
    _set_attrs(meta, dict(software="phast", software_version="fixture", machine="fixture", solver_dtype="float64"))
    _set_attrs(meta, desc.get("meta_attrs", {}))


# --- common pieces -----------------------------------------------------------

def U(n_steps, n, dim=2):
    """Displacement stored with 3 components; z is zero in 2D."""
    a = coded(n_steps, n, 3, base=0.5)
    if dim == 2:
        a[..., 2] = 0.0
    return a


def F(location, components, units, kind, **more):
    return dict(location=location, components=components, units=units, kind=kind, **more)


def history(n_inc, step_index, extra=None, dt=None, solved=None):
    dt = np.full(n_inc, 0.01) if dt is None else np.asarray(dt, float)
    solved = np.ones(n_inc, np.int8) if solved is None else np.asarray(solved, np.int8)
    h = {
        "increment": (np.arange(n_inc, dtype=np.int64), {"units": "1"}),
        "dt": (dt, {"units": "1"}),
        "time": (np.cumsum(dt), {"units": "1"}),
        "load_parameter": (0.001 * np.arange(1, n_inc + 1), {"units": "mm"}),
        "step_index": (np.asarray(step_index, np.int64), {"units": "1"}),
        "damage_solved": (solved, {"units": "1"}),
        "reaction/top": (np.stack([np.zeros(n_inc), -10.0 * np.arange(n_inc), np.zeros(n_inc)], 1),
                         {"units": "N", "components": "x,y,z", "sign_convention": "force exerted by the support on the body"}),
        "energy/elastic": (0.1 * np.arange(n_inc), {"units": "mJ", "method": "integral of psi over cells"}),
        "energy/fracture": (0.01 * np.arange(n_inc), {"units": "mJ", "method": "AT2 crack surface functional"}),
        "energy/external_work": (0.11 * np.arange(n_inc), {"units": "mJ", "method": "trapezoidal reaction times displacement"}),
        "energy/balance_residual": (np.zeros(n_inc), {"units": "mJ", "method": "external_work - elastic - fracture"}),
    }
    h.update(extra or {})
    return h


def qp_block(cell_type, n_cells):
    if cell_type == "T3":
        return dict(cell_type="T3", range=[0, n_cells], qp_points=[[1 / 3, 1 / 3]], qp_weights=[0.5], rule="centroid", point_order="single point")
    if cell_type == "Q4":
        a = 1 / np.sqrt(3)
        return dict(cell_type="Q4", range=[0, n_cells], qp_points=[[-a, -a], [a, -a], [-a, a], [a, a]], qp_weights=[1, 1, 1, 1],
                    rule="gauss 2x2", point_order="eta outer, xi inner, minus then plus")
    a = 1 / np.sqrt(3)
    pts = [[x, y, z] for z in (-a, a) for y in (-a, a) for x in (-a, a)]
    return dict(cell_type="H8", range=[0, n_cells], qp_points=pts, qp_weights=[1] * 8, rule="gauss 2x2x2",
                point_order="zeta outer, eta, xi inner, minus then plus")


ISO = {"name": "matrix", "elastic_model": "isotropic", "E": 210000.0, "nu": 0.3, "fracture_law": "AT2", "Gc": 2.7, "ell": 0.1,
       "units": '{"E": "MPa", "Gc": "N/mm", "ell": "mm"}'}


# --- the fixtures -------------------------------------------------------------

def t3_notched_plate():
    pts, cells = tri_grid(5, 4); n, e = len(pts), len(cells)
    step_index = [0, -1, 1, -1, 2]                     # fields stored every second increment
    ns = 3
    d = dict(kind="ug", points=pts, cells=cells, types=[VTK["T3"]] * e,
             root=dict(profile="run", dim=2, analysis="plane_strain", run_id="fixture-t3"),
             point_fields={"displacement": (U(ns, n), F("node", "x,y,z", "mm", "state"), float),
                           "damage": (coded(ns, n, base=0.25) / 1e4, F("node", "d", "1", "state"), float)},
             cell_fields={"H": (coded(ns, e, base=0.75), F("cell", "H", "MPa", "history"), float),
                          "stress": (coded(ns, e, 3, base=0.125), F("cell", TENSOR_2D, "MPa", "derived", derived_method="exact"), np.float32),
                          "stress_zz": (coded(ns, e, base=0.375), F("cell", "zz", "MPa", "derived", derived_method="exact"), np.float32),
                          "strain": (coded(ns, e, 3) * 1e-6, F("cell", TENSOR_2D, "1", "derived", derived_method="exact", shear="engineering"), np.float32)},
             cell_blocks={"T3": qp_block("T3", e)}, node_sets={"top": np.arange(n - 6, n), "bottom": np.arange(6)},
             edge_index=edges_of(cells), phases={0: ISO},
             bc={"top": dict(set="top", kind="displacement", components="y", value="load_parameter"),
                 "bottom": dict(set="bottom", kind="displacement", components="x,y", value=0.0)},
             state_links={"displacement": "/VTKHDF/PointData/displacement", "damage": "/VTKHDF/PointData/damage", "H": "/VTKHDF/CellData/H"},
             history=history(5, step_index), step_times=[0.01, 0.03, 0.05])
    probes = [dict(array="displacement", where="point", step=s, index=i, component=c, value=float(d["point_fields"]["displacement"][0][s, i, c]))
              for s in range(ns) for i, c in [(3, 0), (n - 1, 1)]]
    probes += [dict(array="H", where="cell", step=s, index=i, component=0, value=float(d["cell_fields"]["H"][0][s, i])) for s in range(ns) for i in (0, e - 1)]
    probes += [dict(array="stress", where="cell", step=2, index=7, component=2, value=float(np.float32(d["cell_fields"]["stress"][0][2, 7, 2])))]
    return d, dict(type="UnstructuredGrid", nsteps=ns, n_points=n, n_cells=e, probes=probes)


def q4_plate():
    pts, cells = quad_grid(3, 3); n, e = len(pts), len(cells); ns = 3
    H = per_point(ns, e, 4, base=0.5)
    sig_qp = (per_point(ns, e, 4)[..., None] + 0.01 * np.arange(3)).astype(np.float32)
    d = dict(kind="ug", points=pts, cells=cells, types=[VTK["Q4"]] * e,
             root=dict(profile="run", dim=2, analysis="plane_strain", run_id="fixture-q4"),
             point_fields={"displacement": (U(ns, n), F("node", "x,y,z", "mm", "state"), float),
                           "damage": (coded(ns, n) / 1e4, F("node", "d", "1", "state"), float)},
             cell_fields={"H": (H, F("quadrature_point", "qp0,qp1,qp2,qp3", "MPa", "history"), float),
                          "stress": (sig_qp.mean(axis=2), F("cell", TENSOR_2D, "MPa", "derived", derived_method="qp_mean_volume_weighted"), np.float32)},
             cell_blocks={"Q4": qp_block("Q4", e)}, phases={0: ISO}, edge_index=edges_of(cells),
             extra={"quadrature/Q4/stress": (sig_qp, F("quadrature_point", TENSOR_2D, "MPa", "derived"))},
             state_links={"displacement": "/VTKHDF/PointData/displacement", "damage": "/VTKHDF/PointData/damage", "H": "/VTKHDF/CellData/H"},
             history=history(3, [0, 1, 2]), step_times=[0.01, 0.02, 0.03])
    probes = [dict(array="H", where="cell", step=s, index=c, component=q, value=float(H[s, c, q])) for s in range(ns) for c, q in [(0, 3), (e - 1, 1), (4, 2)]]
    return d, dict(type="UnstructuredGrid", nsteps=ns, n_points=n, n_cells=e, probes=probes)


def two_phase_2d():
    pts, cells = tri_grid(4, 4); n, e = len(pts), len(cells); ns = 3
    phase = (np.arange(e) >= e // 2).astype(np.uint8)
    d = dict(kind="ug", points=pts, cells=cells, types=[VTK["T3"]] * e,
             root=dict(profile="run", dim=2, analysis="plane_stress", run_id="fixture-two-phase"),
             static_cell={"phase_id": (phase, F("cell", "id", "1", "state"))},
             point_fields={"displacement": (U(ns, n), F("node", "x,y,z", "mm", "state"), float),
                           "damage": (coded(ns, n) / 1e4, F("node", "d", "1", "state"), float)},
             cell_fields={"H": (coded(ns, e), F("cell", "H", "MPa", "history"), float)},
             cell_blocks={"T3": qp_block("T3", e)}, edge_index=edges_of(cells),
             phases={0: ISO, 1: dict(ISO, name="inclusion", E=400000.0, Gc=10.0)},
             state_links={"displacement": "/VTKHDF/PointData/displacement", "damage": "/VTKHDF/PointData/damage", "H": "/VTKHDF/CellData/H"},
             history=history(3, [0, 1, 2]), step_times=[0.01, 0.02, 0.03])
    probes = [dict(array="phase_id", where="cell", step=s, index=i, component=0, value=float(phase[i])) for s in range(ns) for i in (0, e - 1)]
    return d, dict(type="UnstructuredGrid", nsteps=ns, n_points=n, n_cells=e, probes=probes)


def voxel_h8():
    nx, ny, nz = 6, 5, 4; ns = 3                       # non-cubic so a transpose shows
    def node_code(s):
        k, j, i = np.meshgrid(np.arange(nz + 1), np.arange(ny + 1), np.arange(nx + 1), indexing="ij")
        return 1e6 * s + 1e4 * i + 1e2 * j + k
    damage = np.stack([node_code(s) for s in range(ns)]).astype(float)                       # [t, z, y, x]
    k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    cell = 1e4 * i + 1e2 * j + k
    H = np.stack([1e6 * s + 10 * cell[..., None] + np.arange(8) for s in range(ns)]).astype(float)  # [t, z, y, x, 8]
    phase = ((i + j + k) % 3).astype(np.uint8)
    disp = np.repeat(damage[..., None], 3, axis=-1) + 0.01 * np.arange(3)
    d = dict(kind="image", grid=(nx, ny, nz), spacing=(0.5, 0.5, 0.5),
             root=dict(profile="run", dim=3, analysis="3d", run_id="fixture-voxel"),
             static_cell={"phase_id": (phase, F("cell", "id", "1", "state"))},
             point_fields={"displacement": (disp, F("node", "x,y,z", "mm", "state"), float),
                           "damage": (damage, F("node", "d", "1", "state"), float)},
             cell_fields={"H": (H, F("quadrature_point", ",".join(f"qp{q}" for q in range(8)), "MPa", "history"), float)},
             cell_blocks={"H8": qp_block("H8", nx * ny * nz)},
             phases={0: dict(ISO, name="fibre"), 1: dict(ISO, name="interface"), 2: dict(ISO, name="matrix")},
             periodic=dict(axes="x,y,z", controlled="xx", free="yy,zz,xy", fixed_zero="yz,xz"),
             state_links={"displacement": "/VTKHDF/PointData/displacement", "damage": "/VTKHDF/PointData/damage", "H": "/VTKHDF/CellData/H"},
             history=history(3, [0, 1, 2], extra={"macro_stress/xx": (np.array([0.0, 50.0, 80.0]), {"units": "MPa"}),
                                                  "macro_strain/xx": (np.array([0.0, 2e-4, 4e-4]), {"units": "1"})}),
             step_times=[0.01, 0.02, 0.03])
    pid = lambda i, j, k: i + (nx + 1) * (j + (ny + 1) * k)
    cid = lambda i, j, k: i + nx * (j + ny * k)
    probes = []
    for s in range(ns):
        for (i, j, k) in [(5, 0, 0), (0, 4, 0), (0, 0, 3), (6, 5, 4)]:
            probes.append(dict(array="damage", where="point", step=s, index=pid(i, j, k), component=0, value=float(damage[s, k, j, i])))
        for (i, j, k), q in [((5, 1, 3), 7), ((0, 4, 2), 1)]:
            probes.append(dict(array="H", where="cell", step=s, index=cid(i, j, k), component=q, value=float(H[s, k, j, i, q])))
        probes.append(dict(array="phase_id", where="cell", step=s, index=cid(4, 3, 2), component=0, value=float(phase[2, 3, 4])))
    return d, dict(type="ImageData", nsteps=ns, n_points=(nx + 1) * (ny + 1) * (nz + 1), n_cells=nx * ny * nz, probes=probes)


def cohesive_dcb():
    pts, cells = tri_grid(6, 2, lx=12.0, ly=4.0); n, e = len(pts), len(cells); ns = 3
    mid = np.arange(2 * 7, 3 * 7)                      # node row along the interface
    coh = np.array([[mid[a], mid[a + 1], mid[a + 1], mid[a]] for a in range(6)], np.int64)
    coh_d = (per_point(ns, len(coh), 2) / 1e4).astype(float)
    d = dict(kind="ug", points=pts, cells=cells, types=[VTK["T3"]] * e,
             root=dict(profile="run", dim=2, analysis="plane_strain", run_id="fixture-dcb"),
             point_fields={"displacement": (U(ns, n), F("node", "x,y,z", "mm", "state"), float),
                           "damage": (coded(ns, n) / 1e4, F("node", "d", "1", "state"), float)},
             cell_fields={"H": (coded(ns, e), F("cell", "H", "MPa", "history"), float)},
             cell_blocks={"T3": qp_block("T3", e)}, phases={0: ISO},
             extra={"interfaces/crack/connectivity": (coh, {"cell_type": "COH2"}),
                    "interfaces/crack/quadrature/points": (np.array([[-1 / np.sqrt(3)], [1 / np.sqrt(3)]]), {}),
                    "interfaces/crack/cohesive_damage": (coh_d, F("interface_point", "d", "1", "history"))},
             extra_attrs={"interfaces/crack": dict(law="exponential", strength=30.0, Gc=0.5, units='{"strength": "MPa", "Gc": "N/mm"}')},
             state_links={"displacement": "/VTKHDF/PointData/displacement", "damage": "/VTKHDF/PointData/damage", "H": "/VTKHDF/CellData/H"},
             history=history(3, [0, 1, 2], extra={"energy/cohesive_dissipation": (np.array([0.0, 0.02, 0.05]), {"units": "mJ", "method": "sum over interface points"})}),
             step_times=[0.01, 0.02, 0.03])
    probes = [dict(array="damage", where="point", step=s, index=mid[3], component=0, value=float(d["point_fields"]["damage"][0][s, mid[3]])) for s in range(ns)]
    return d, dict(type="UnstructuredGrid", nsteps=ns, n_points=n, n_cells=e, probes=probes)


def sent_lagged():
    """Explicit run with lagged history coupling: H and the history the damage solve used (H_damage_input)."""
    pts, cells = tri_grid(6, 4); n, e = len(pts), len(cells); ns = 6
    H = coded(ns, e, base=0.5)
    solved = np.array([1, 1, 0, 0, 1, 0], np.int8)
    stored_incs = [2, 3, 5]                      # stress is stored on some increments only, in /derived
    stress = (coded(len(stored_incs), e, 3) / 1e3).astype(np.float32)
    stress_idx = np.full(ns, -1, np.int64); stress_idx[stored_incs] = np.arange(len(stored_incs))
    extra_hist = {"mechanics_iterations": (np.array([1, 1, 1, 1, 1, 1]), {"units": "1"})}
    d = dict(kind="ug", points=pts, cells=cells, types=[VTK["T3"]] * e,
             root=dict(profile="run", dim=2, analysis="plane_strain", run_id="fixture-sent-lagged", history_coupling="lagged"),
             point_fields={"displacement": (U(ns, n), F("node", "x,y,z", "mm", "state"), float),
                           "damage": (coded(ns, n) / 1e4, F("node", "d", "1", "state"), float)},
             cell_fields={"H": (H, F("cell", "H", "MPa", "history"), float),
                          "H_damage_input": (H - 0.5, F("cell", "H", "MPa", "history"), float)},
             cell_blocks={"T3": qp_block("T3", e)}, phases={0: ISO}, edge_index=edges_of(cells),
             extra={"derived/stress/values": (stress, F("cell", TENSOR_2D, "MPa", "derived", derived_method="exact")),
                    "derived/stress/step_index": (stress_idx, {"units": "1"})},
             state_links={"displacement": "/VTKHDF/PointData/displacement", "damage": "/VTKHDF/PointData/damage", "H": "/VTKHDF/CellData/H"},
             history=history(ns, list(range(ns)), extra=extra_hist, solved=solved), step_times=list(np.cumsum(np.full(ns, 0.01))))
    probes = [dict(array=a, where="cell", step=s, index=4, component=0, value=float(arr[s, 4]))
              for s in (0, 3, 5) for a, arr in (("H", H), ("H_damage_input", H - 0.5))]
    return d, dict(type="UnstructuredGrid", nsteps=ns, n_points=n, n_cells=e, probes=probes)


def explicit_dynamic():
    pts, cells = tri_grid(5, 3); n, e = len(pts), len(cells); ns = 4
    dt = np.array([1e-8, 1e-8, 5e-9, 2.5e-9])
    d = dict(kind="ug", points=pts, cells=cells, types=[VTK["T3"]] * e,
             root=dict(profile="run", dim=2, analysis="plane_strain", run_id="fixture-dynamic"),
             point_fields={"displacement": (U(ns, n), F("node", "x,y,z", "mm", "state"), float),
                           "velocity": (U(ns, n) * 10, F("node", "x,y,z", "mm/s", "state"), float),
                           "acceleration": (U(ns, n) * 100, F("node", "x,y,z", "mm/s2", "state"), float),
                           "damage": (coded(ns, n) / 1e4, F("node", "d", "1", "state"), float)},
             cell_fields={"H": (coded(ns, e), F("cell", "H", "MPa", "history"), float)},
             cell_blocks={"T3": qp_block("T3", e)}, phases={0: dict(ISO, rho=2.45e-9)},
             state_links={"displacement": "/VTKHDF/PointData/displacement", "velocity": "/VTKHDF/PointData/velocity",
                          "acceleration": "/VTKHDF/PointData/acceleration", "damage": "/VTKHDF/PointData/damage", "H": "/VTKHDF/CellData/H"},
             history=history(ns, list(range(ns)), dt=dt, extra={"energy/kinetic": (np.array([0, 0.01, 0.02, 0.025]), {"units": "mJ", "method": "lumped mass"})}),
             step_times=list(np.cumsum(dt)))
    probes = [dict(array="velocity", where="point", step=s, index=5, component=1, value=float(d["point_fields"]["velocity"][0][s, 5, 1])) for s in (0, 2, 3)]
    return d, dict(type="UnstructuredGrid", nsteps=ns, n_points=n, n_cells=e, probes=probes, times=[float(x) for x in np.cumsum(dt)])


def j2_plasticity():
    pts, cells = tri_grid(4, 3); n, e = len(pts), len(cells); ns = 3
    epeq = coded(ns, e) / 1e5
    d = dict(kind="ug", points=pts, cells=cells, types=[VTK["T3"]] * e,
             root=dict(profile="run", dim=2, analysis="plane_strain", run_id="fixture-j2"),
             point_fields={"displacement": (U(ns, n), F("node", "x,y,z", "mm", "state"), float),
                           "damage": (np.zeros((ns, n)), F("node", "d", "1", "state"), float)},
             cell_fields={"equivalent_plastic_strain": (epeq, F("cell", "eps_p_eq", "1", "history"), float),
                          "plastic_strain": (coded(ns, e, 3) / 1e5, F("cell", TENSOR_2D, "1", "history", shear="engineering"), float),
                          "plastic_strain_zz": (coded(ns, e) / 1e5, F("cell", "zz", "1", "history"), float),
                          "stress": (coded(ns, e, 3), F("cell", TENSOR_2D, "MPa", "derived", derived_method="exact"), np.float32)},
             cell_blocks={"T3": qp_block("T3", e)}, phases={0: dict(ISO, plasticity="J2", yield_stress=250.0)},
             state_links={"displacement": "/VTKHDF/PointData/displacement", "damage": "/VTKHDF/PointData/damage"},
             history=history(ns, [0, 1, 2], extra={"energy/plastic_work": (np.array([0, 0.3, 0.9]), {"units": "mJ", "method": "sum of stress times plastic strain increment"})}),
             step_times=[0.01, 0.02, 0.03])
    probes = [dict(array="equivalent_plastic_strain", where="cell", step=s, index=e - 1, component=0, value=float(epeq[s, e - 1])) for s in range(ns)]
    return d, dict(type="UnstructuredGrid", nsteps=ns, n_points=n, n_cells=e, probes=probes)


def setup_profile():
    pts, cells = tri_grid(4, 3); e = len(cells)
    phase = (np.arange(e) % 2).astype(np.uint8)
    d = dict(kind="ug", points=pts, cells=cells, types=[VTK["T3"]] * e,
             root=dict(profile="setup", dim=2, analysis="plane_strain", run_id="fixture-setup", final_status="complete"),
             static_cell={"phase_id": (phase, F("cell", "id", "1", "state"))},
             cell_blocks={"T3": qp_block("T3", e)}, phases={0: ISO, 1: dict(ISO, name="inclusion")},
             bc={"top": dict(set="top", kind="displacement", components="y", value="load_parameter")},
             node_sets={"top": np.arange(len(pts) - 5, len(pts))},
             history={"step_index": (np.zeros(0, np.int64), {"units": "1"})}, step_times=[])
    probes = [dict(array="phase_id", where="cell", step=None, index=i, component=0, value=float(phase[i])) for i in (0, 1, e - 1)]
    return d, dict(type="UnstructuredGrid", nsteps=0, n_points=len(pts), n_cells=e, probes=probes)


FIXTURES = {"t3_notched_plate": t3_notched_plate, "q4_plate": q4_plate, "two_phase_2d": two_phase_2d, "voxel_h8": voxel_h8,
            "cohesive_dcb": cohesive_dcb, "sent_lagged": sent_lagged, "explicit_dynamic": explicit_dynamic,
            "j2_plasticity": j2_plasticity, "setup_profile": setup_profile}
KILL_STAGES = ("after_rows", "after_offsets", "after_nsteps")


def build_all(out_dir):
    """Write every fixture, the interrupted variants and ``expected.json``; return the expected dict."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    expected = {}
    for name, fn in FIXTURES.items():
        desc, exp = fn()
        write_fixture(out / f"{name}.vtkhdf", desc)
        expected[name] = dict(exp, file=f"{name}.vtkhdf", status="complete")
    # interrupted at step 2 of the T3 plate at each point of the write order
    base_desc, base_exp = t3_notched_plate()
    for stage in KILL_STAGES:
        fname = f"interrupted_{stage}.vtkhdf"
        subprocess.run([sys.executable, "-c",
                        "import sys; sys.path[:0] = sys.argv[3:]; from result_v2.fixtures import t3_notched_plate, write_fixture;"
                        " write_fixture(sys.argv[1], t3_notched_plate()[0], kill=(2, sys.argv[2]))",
                        str(out / fname), stage, *sys.path], check=True)
        # NSteps is the commit point of a step: killed before it is written the file keeps steps 0 and 1,
        # killed after it the file counts step 2 as well, and only its history row is missing
        nsteps = 3 if stage == "after_nsteps" else 2
        probes = [p for p in base_exp["probes"] if p["step"] is not None and p["step"] < nsteps]
        expected[f"interrupted_{stage}"] = dict(base_exp, file=fname, nsteps=nsteps, probes=probes, status="running")
    try:
        desc, _ = t3_notched_plate(); desc["root"]["mesh_varies"] = True
        write_fixture(out / "refused.vtkhdf", desc)
        expected["refused_mesh_varies"] = dict(refused=False)
    except RefusedLayout:
        expected["refused_mesh_varies"] = dict(refused=True)
    expected = json.loads(json.dumps(expected, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    (out / "expected.json").write_text(json.dumps(expected, indent=1))
    return expected


if __name__ == "__main__":
    print(json.dumps({k: v.get("nsteps") for k, v in build_all(sys.argv[1]).items()}))
