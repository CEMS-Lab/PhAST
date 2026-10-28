"""Reference builder for the layout-2.1 fixtures of meshes that change during a run.

Like ``fixtures.py`` it writes each file directly with h5py from a plain description and does not
import the writer of ``src/phast/io/result``: the writer is checked against files it did not produce. Two fixtures:

- ``refined_t3``: a unit square of two triangles, bisected twice (newest-vertex bisection, conforming):
  2 -> 4 -> 8 cells, 4 -> 5 -> 9 points; five increments, fields stored at increments 0, 2 and 4 (three steps,
  each on a different mesh), the refinements at increments that store none; a ``phase_id`` that is static
  on each mesh and carried through the cell parents.
- ``refined_q4_hanging``: two quadrilaterals, the left one split into four children; the node on the shared
  edge is hanging for the right cell and constrained to the mean of the edge's end nodes.

Geometry of a changing mesh follows the VTKHDF time-dependent layout: ``NumberOfPoints``, ``NumberOfCells``
and ``NumberOfConnectivityIds`` have one entry per mesh version (a "part"), ``Points``, ``Connectivity``,
``Offsets`` (cells + 1 per part) and ``Types`` are concatenated, and ``Steps/PartOffsets`` names the version
of each stored step. Usage: ``python -m result_v2.fixtures_21 <out_dir>`` (run from the tests directory) writes both files and
``expected_21.json`` (per-step sizes and probe values for ``paraview_check_21.py``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

from result_v2.fixtures import CREATED, ISO, LIBVER, VTK, F, _ds, _set_attrs, coded, per_point, qp_block

SCHEMA_VERSION = "2.1.0"


def _mesh_t3():
    pts0 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    c0 = np.array([[0, 1, 2], [0, 2, 3]])
    pts1 = np.vstack([pts0, [[0.5, 0.5, 0]]])
    c1 = np.array([[1, 0, 4], [1, 4, 2], [3, 0, 4], [3, 4, 2]])
    pts2 = np.vstack([pts1, [[0.5, 0, 0], [1, 0.5, 0], [0, 0.5, 0], [0.5, 1, 0]]])
    c2 = np.array([[4, 1, 5], [4, 5, 0], [4, 1, 6], [4, 6, 2], [4, 3, 7], [4, 7, 0], [4, 3, 8], [4, 8, 2]])
    return [
        dict(points=pts0, cells=c0, node_sets=dict(bottom=[0, 1], top=[2, 3])),
        dict(points=pts1, cells=c1, node_sets=dict(bottom=[0, 1], top=[2, 3]),
             cell_parent=[0, 0, 1, 1], node_origin=[0, 1, 2, 3, -1], node_parents=[[-1, -1]] * 4 + [[0, 2]]),
        dict(points=pts2, cells=c2, node_sets=dict(bottom=[0, 5, 1], top=[2, 8, 3]),
             cell_parent=[0, 0, 1, 1, 2, 2, 3, 3], node_origin=list(range(5)) + [-1] * 4,
             node_parents=[[-1, -1]] * 5 + [[0, 1], [1, 2], [0, 3], [2, 3]]),
    ]


def _mesh_q4():
    pts0 = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0]], float)
    c0 = np.array([[0, 1, 4, 3], [1, 2, 5, 4]])
    pts1 = np.vstack([pts0, [[0.5, 0, 0], [1, 0.5, 0], [0.5, 1, 0], [0, 0.5, 0], [0.5, 0.5, 0]]])
    c1 = np.array([[0, 6, 10, 9], [6, 1, 7, 10], [10, 7, 4, 8], [9, 10, 8, 3], [1, 2, 5, 4]])
    return [
        dict(points=pts0, cells=c0, node_sets=dict(left=[0, 3], right=[2, 5])),
        dict(points=pts1, cells=c1, node_sets=dict(left=[0, 9, 3], right=[2, 5]),
             cell_parent=[0, 0, 0, 0, 1], node_origin=list(range(6)) + [-1] * 5,
             node_parents=[[-1, -1]] * 6 + [[0, 1], [1, 4], [3, 4], [0, 3], [0, 4]],
             hanging=dict(constrained_nodes=[7], master_nodes=[[1, 4]], weights=[[0.5, 0.5]])),
    ]


def _write(path, desc):
    """Write one file. ``desc``: meshes, vtk cell type, per-increment mesh index and stored step, fields."""
    meshes = desc["meshes"]
    vtk = desc["vtk"]
    increments = desc["increments"]                       # list of dict(mesh=version, step=stored step or -1)
    f = h5py.File(path, "w", libver=LIBVER, track_order=True)
    root = dict(phast_schema="phast.result", phast_schema_version=SCHEMA_VERSION, unit_system="mm-MPa-N-s", created=CREATED,
                mesh_varies=True, index_basis="increment", kinematics="small", history_update="hard_max",
                history_coupling="fresh", status="running", profile="run", dim=2, analysis="plane_strain",
                run_id=desc["run_id"])
    _set_attrs(f, root)
    v = f.create_group("VTKHDF", track_order=True)
    v.attrs["Version"] = np.array([2, 2])
    v.attrs.create("Type", np.bytes_("UnstructuredGrid"))
    npts = np.array([len(m["points"]) for m in meshes])
    ncell = np.array([len(m["cells"]) for m in meshes])
    nconn = np.array([m["cells"].size for m in meshes])
    for k, d in (("NumberOfPoints", npts), ("NumberOfCells", ncell), ("NumberOfConnectivityIds", nconn),
                 ("Points", np.vstack([m["points"] for m in meshes])),
                 ("Connectivity", np.concatenate([m["cells"].ravel() for m in meshes]).astype(np.int64)),
                 ("Offsets", np.concatenate([np.arange(0, m["cells"].shape[1] * (len(m["cells"]) + 1), m["cells"].shape[1])
                                             for m in meshes]).astype(np.int64)),
                 ("Types", np.full(int(ncell.sum()), vtk, np.uint8))):
        _ds(v, k, d)
    first = lambda a: np.concatenate([[0], np.cumsum(a)[:-1]])
    pt0, cell0, conn0 = first(npts), first(ncell), first(nconn)
    pd, cd = v.create_group("PointData", track_order=True), v.create_group("CellData", track_order=True)
    st = v.create_group("Steps")
    stored = [(i, inc) for i, inc in enumerate(increments) if inc["step"] >= 0]
    nsteps = len(stored)
    st.attrs["NSteps"] = nsteps
    version = np.array([inc["mesh"] for _, inc in stored])
    _ds(st, "Values", np.array(desc["times"], float))
    _ds(st, "NumberOfParts", np.ones(nsteps, np.int64))
    _ds(st, "PartOffsets", version.astype(np.int64))
    _ds(st, "PointOffsets", pt0[version].astype(np.int64))
    _ds(st, "CellOffsets", cell0[version].astype(np.int64).reshape(-1, 1))
    _ds(st, "ConnectivityIdOffsets", conn0[version].astype(np.int64).reshape(-1, 1))
    pdo, cdo = st.create_group("PointDataOffsets"), st.create_group("CellDataOffsets")

    def put(group, ogroup, name, rows_per_step, per_version, attrs, dtype=float):
        """Rows of a time-varying array (stored per step) or of a version-static one (``per_version`` rows)."""
        if per_version is None:
            data = np.concatenate([np.asarray(r, dtype) for r in rows_per_step])
            off = np.concatenate([[0], np.cumsum([len(r) for r in rows_per_step])[:-1]])
        else:
            data = np.concatenate([np.asarray(r, dtype) for r in per_version])
            starts = first([len(r) for r in per_version])
            off = starts[version]
        _ds(group, name, data, attrs)
        _ds(ogroup, name, off.astype(np.int64))

    for name, (rows, attrs) in desc["point_fields"].items():
        put(pd, pdo, name, rows, None, attrs)
    for name, (rows, attrs) in desc["cell_fields"].items():
        put(cd, cdo, name, rows, None, attrs)
    put(cd, cdo, "phase_id", None, desc["phase_id"], F("cell", "id", "1", "state"), np.uint8)

    mesh = f.create_group("mesh")
    for k, m in enumerate(meshes):
        g = mesh if k == 0 else mesh.create_group(f"versions/{k}")
        b = g.create_group("cell_blocks/" + desc["block"])
        _set_attrs(b, dict(cell_type=desc["block"], vtk_type=vtk, node_order="vtk"))
        _ds(b, "cells", np.array([0, len(m["cells"])], np.int64))
        q = b.create_group("quadrature")
        _ds(q, "points", np.asarray(desc["quadrature"]["qp_points"], float))
        _ds(q, "weights", np.asarray(desc["quadrature"]["qp_weights"], float))
        q.attrs["rule"], q.attrs["point_order"] = desc["quadrature"]["rule"], desc["quadrature"]["point_order"]
        for name, idx in m["node_sets"].items():
            _ds(g.require_group("node_sets"), name, np.asarray(idx, np.int64))
        if "hanging" in m:
            h = g.create_group("hanging_nodes")
            _ds(h, "constrained_nodes", np.asarray(m["hanging"]["constrained_nodes"], np.int64))
            _ds(h, "master_nodes", np.asarray(m["hanging"]["master_nodes"], np.int64))
            _ds(h, "weights", np.asarray(m["hanging"]["weights"], float))
            h.attrs["rule"] = "value of a constrained node = sum of weight times value of its master nodes"
        if k:
            first_inc = min(i for i, inc in enumerate(increments) if inc["mesh"] == k)
            _set_attrs(g, dict(n_points=len(m["points"]), n_cells=len(m["cells"]), first_increment=first_inc,
                               node_parents_rule="a new node takes the mean of its two parent nodes"))
            _ds(g, "cell_parent", np.asarray(m["cell_parent"], np.int64))
            _ds(g, "node_origin", np.asarray(m["node_origin"], np.int64))
            _ds(g, "node_parents", np.asarray(m["node_parents"], np.int64))
    mat = f.create_group("materials")
    _set_attrs(mat.create_group("phases/0"), ISO)
    _set_attrs(mat.create_group("phases/1"), dict(ISO, name="inclusion", E=400000.0))
    mat["phase_id"] = h5py.SoftLink("/VTKHDF/CellData/phase_id")
    bc = f.create_group("loading/bc/top")
    _set_attrs(bc, dict(set=desc["load_set"], kind="displacement", components="y", value="load_parameter"))
    state = f.create_group("state")
    for name, target in (("displacement", "PointData"), ("damage", "PointData"), ("H", "CellData")):
        state[name] = h5py.SoftLink(f"/VTKHDF/{target}/{name}")

    hist = f.create_group("history")
    n_inc = len(increments)
    dt = 0.1
    cols = {
        "increment": (np.arange(n_inc, dtype=np.int64), {"units": "1"}),
        "dt": (np.full(n_inc, dt), {"units": "1"}),
        "time": (dt * np.arange(1, n_inc + 1), {"units": "1"}),
        "load_parameter": (0.001 * np.arange(1, n_inc + 1), {"units": "mm"}),
        "step_index": (np.array([inc["step"] for inc in increments], np.int64), {"units": "1"}),
        "damage_solved": (np.ones(n_inc, np.int8), {"units": "1"}),
        "mesh_index": (np.array([inc["mesh"] for inc in increments], np.int64), {"units": "1"}),
    }
    for name, (col, attrs) in cols.items():
        _ds(hist, name, col, attrs)
    meta = f.create_group("meta")
    _ds(meta, "config_resolved_yaml", "fixture: true\n")
    _ds(meta, "schema_yaml", f"schema: phast.result\nversion: {SCHEMA_VERSION}\n")
    _set_attrs(meta, dict(software="phast", software_version="fixture", machine="fixture", solver_dtype="float64"))
    f.attrs["status"] = "complete"
    f.attrs["finished"] = CREATED
    f.close()


def _fields(meshes, version_of_step, qp):
    """Values that encode step, entity and component."""
    pt = {"displacement": [], "damage": []}
    ce = {"H": []}
    for s, k in enumerate(version_of_step):
        n, e = len(meshes[k]["points"]), len(meshes[k]["cells"])
        pt["displacement"].append(coded(1, n, 3, base=0.5)[0] + 1000.0 * s)
        pt["damage"].append(coded(1, n, base=0.25)[0] / 1e4 + 1e-1 * s)
        ce["H"].append(per_point(1, e, qp, base=0.75)[0] + 1000.0 * s if qp > 1 else coded(1, e, base=0.75)[0] + 1000.0 * s)
    return pt, ce


def _describe(name, meshes, block, increments, qp, load_set):
    version_of_step = [inc["mesh"] for inc in increments if inc["step"] >= 0]
    pt, ce = _fields(meshes, version_of_step, qp)
    phase = [np.array([i % 2 for i in range(len(meshes[0]["cells"]))], np.uint8)]
    for m in meshes[1:]:
        phase.append(phase[-1][np.asarray(m["cell_parent"])])
    h_components = ",".join(f"qp{i}" for i in range(qp)) if qp > 1 else "H"
    h_location = "quadrature_point" if qp > 1 else "cell"
    desc = dict(
        run_id=f"fixture-{name}", meshes=meshes, block=block, vtk=VTK[block], increments=increments, load_set=load_set,
        times=[0.1 * (i + 1) for i, inc in enumerate(increments) if inc["step"] >= 0],
        point_fields={"displacement": (pt["displacement"], F("node", "x,y,z", "mm", "state")),
                      "damage": (pt["damage"], F("node", "d", "1", "state"))},
        cell_fields={"H": (ce["H"], F(h_location, h_components, "MPa", "history"))},
        phase_id=phase, quadrature=qp_block(block, 1))
    desc["quadrature"] = {k: v for k, v in qp_block(block, 1).items() if k in ("qp_points", "qp_weights", "rule", "point_order")}
    return desc, pt, ce, phase


def _expected(file, meshes, desc, pt, ce, phase, extra=None):
    version_of_step = [inc["mesh"] for inc in desc["increments"] if inc["step"] >= 0]
    steps = [dict(n_points=len(meshes[k]["points"]), n_cells=len(meshes[k]["cells"])) for k in version_of_step]
    probes = []
    for s, k in enumerate(version_of_step):
        n, e = steps[s]["n_points"], steps[s]["n_cells"]
        for i in sorted({0, n - 1, min(4, n - 1)}):
            probes.append(dict(array="displacement", where="point", step=s, index=int(i), component=1,
                               value=float(pt["displacement"][s][i, 1])))
            probes.append(dict(array="damage", where="point", step=s, index=int(i), component=0,
                               value=float(pt["damage"][s][i])))
        for i in sorted({0, e - 1}):
            h = np.atleast_2d(ce["H"][s].reshape(e, -1))
            probes.append(dict(array="H", where="cell", step=s, index=int(i), component=int(h.shape[1] - 1),
                               value=float(h[i, -1])))
            probes.append(dict(array="phase_id", where="cell", step=s, index=int(i), component=0,
                               value=float(phase[k][i])))
    cells = [dict(step=s, cell=e - 1, points=[int(x) for x in meshes[k]["cells"][-1]])
             for s, (k, st) in enumerate(zip(version_of_step, steps)) for e in [st["n_cells"]]]
    points = [dict(step=s, point=n - 1, xyz=[float(x) for x in meshes[k]["points"][-1]])
              for s, (k, st) in enumerate(zip(version_of_step, steps)) for n in [st["n_points"]]]
    return dict(file=file, nsteps=len(steps), steps=steps, times=list(desc["times"]), probes=probes, cells=cells,
                points=points, **(extra or {}))


def refined_t3():
    meshes = _mesh_t3()
    increments = [dict(mesh=0, step=0), dict(mesh=1, step=-1), dict(mesh=1, step=1), dict(mesh=2, step=-1),
                  dict(mesh=2, step=2)]          # fields every second increment; each refinement comes at an increment without fields
    desc, pt, ce, phase = _describe("refined-t3", meshes, "T3", increments, 1, "top")
    return desc, _expected("refined_t3.vtkhdf", meshes, desc, pt, ce, phase)


def refined_q4_hanging():
    meshes = _mesh_q4()
    increments = [dict(mesh=0, step=0), dict(mesh=1, step=1), dict(mesh=1, step=2)]
    desc, pt, ce, phase = _describe("refined-q4-hanging", meshes, "Q4", increments, 4, "right")
    return desc, _expected("refined_q4_hanging.vtkhdf", meshes, desc, pt, ce, phase)


FIXTURES_21 = {"refined_t3": refined_t3, "refined_q4_hanging": refined_q4_hanging}


def build_all(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    expected = {}
    for name, fn in FIXTURES_21.items():
        desc, exp = fn()
        _write(out / exp["file"], desc)
        expected[name] = exp
    (out / "expected_21.json").write_text(json.dumps(expected, indent=1))
    return expected


if __name__ == "__main__":
    print(json.dumps({k: v["nsteps"] for k, v in build_all(sys.argv[1]).items()}))
