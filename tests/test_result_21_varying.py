"""Layout 2.1: meshes that change during a run (refinement).

The reference files come from the independent builder ``tests/result_v2/fixtures_21.py`` (ParaView 5.13 reads them:
``tests/result_v2/paraview_check_21.py``). ``ResultWriter`` must regenerate them dataset by dataset; the
validator, recovery and the reader are checked on them; the maps are also checked against the solver's own
refinement routines.
"""
from __future__ import annotations

import shutil

import h5py
import numpy as np
import pytest
import torch

from phast import load_result
from phast.io.result import Field, ImageGrid, ResultWriter, UnstructuredMesh, recover, validate
from phast.io.result.writer import refinement_maps
from result_v2.fixtures_21 import FIXTURES_21, build_all
from result_v2.helpers import compare


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("fixtures_21")
    return out, build_all(out)


def errors(path):
    return [f"{x.code}: {x.message}" for x in validate(path).findings if x.level == "ERROR"]


def copy_of(built, name, tmp_path, tag="x"):
    out, expected = built
    dst = tmp_path / f"{tag}_{name}.vtkhdf"
    shutil.copyfile(out / expected[name]["file"], dst)
    return dst


# --- reference files -----------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(FIXTURES_21))
def test_reference_fixture_validates(built, name):
    out, expected = built
    report = validate(out / expected[name]["file"])
    assert report.ok and not report.warnings, report.format_text()


@pytest.mark.parametrize("name", sorted(FIXTURES_21))
def test_the_reader_returns_each_step_on_its_own_mesh(built, name):
    out, expected = built
    exp = expected[name]
    r = load_result(out / exp["file"])
    assert r.layouts() == ["2.1"] and r.n_steps() == exp["nsteps"]
    for s, size in enumerate(exp["steps"]):
        assert r.field("displacement", s).shape == (size["n_points"], 2)
        assert r.field("damage", s).shape == (size["n_points"],)
        assert r.field("phase_id", s).shape == (size["n_cells"],)
        arrays = r.mesh_arrays(s)
        assert arrays["points"].shape == (size["n_points"], 2) and arrays["mesh_version"] == r.step_mesh()[s]
    for p in exp["probes"]:
        if p["where"] == "cell" and p["array"] == "H":
            h = np.asarray(r.field("H", p["step"])).reshape(exp["steps"][p["step"]]["n_cells"], -1)
            assert h[p["index"], p["component"]] == pytest.approx(p["value"], rel=1e-12)
        elif p["array"] == "damage":
            assert r.field("damage", p["step"])[p["index"]] == pytest.approx(p["value"], rel=1e-12)
        elif p["array"] == "displacement":
            assert r.field("displacement", p["step"])[p["index"], 1] == pytest.approx(p["value"], rel=1e-12)
        elif p["array"] == "phase_id":
            assert r.field("phase_id", p["step"])[p["index"]] == p["value"]
    assert r.mesh()["n_nodes"] == exp["steps"][-1]["n_points"] and r.mesh()["mesh_versions"] >= 2
    assert r.history("history")[-1]["mesh_index"] == r.step_mesh()[-1]


def test_t3_maps_follow_the_geometry(built):
    out, expected = built
    r = load_result(out / expected["refined_t3"]["file"])
    versions = r.mesh_versions()
    assert [(v["n_points"], v["n_cells"]) for v in versions] == [(4, 2), (5, 4), (9, 8)]
    assert [v["first_increment"] for v in versions] == [0, 1, 3]
    assert r.step_mesh().tolist() == [0, 1, 2]
    assert r.history("history")[2]["mesh_index"] == 1 and [row["mesh_index"] for row in r.history("history")] == [0, 1, 1, 2, 2]
    for k in (1, 2):
        v, prev = versions[k], r.mesh_arrays(k - 1)
        cur = r.mesh_arrays(k)
        parents, origin, par = v["cell_parent"], v["node_origin"], v["node_parents"]
        # existing nodes keep their coordinates; a new node is the mean of its two parents
        old = origin >= 0
        assert np.array_equal(cur["points"][old], prev["points"][origin[old]])
        new = ~old
        assert np.allclose(cur["points"][new], 0.5 * (prev["points"][par[new, 0]] + prev["points"][par[new, 1]]))
        # a child lies inside its parent: its centroid is in the parent triangle (areas of the children add up)
        def area(p, c):
            d1, d2 = p[c[:, 1]] - p[c[:, 0]], p[c[:, 2]] - p[c[:, 0]]
            return 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
        a_prev, a_cur = area(prev["points"], prev["cells"]["T3"]), area(cur["points"], cur["cells"]["T3"])
        assert np.allclose(np.bincount(parents, weights=a_cur, minlength=len(a_prev)), a_prev)
    assert r.field("phase_id", 0).tolist() == [0, 1] and r.field("phase_id", 2).tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert np.array_equal(r.field("phase_id", 2), r.field("phase_id", 1)[versions[2]["cell_parent"]])
    assert r.mesh_arrays(2)["node_sets"]["bottom"].tolist() == [0, 5, 1]


def test_q4_hanging_node_constraint(built):
    out, expected = built
    r = load_result(out / expected["refined_q4_hanging"]["file"])
    v = r.mesh_versions()[1]
    hang = v["hanging_nodes"]
    assert hang["constrained_nodes"].tolist() == [7] and hang["master_nodes"].tolist() == [[1, 4]]
    pts = r.mesh_arrays(1)["points"]
    assert np.allclose(pts[7], (hang["weights"][0] * pts[hang["master_nodes"][0]].T).sum(axis=1))
    assert r.mesh_arrays(1)["mesh_version"] == 1 and "hanging_nodes" not in r.mesh_versions()[0]
    assert np.array_equal(r.mesh_arrays(1)["cells"]["Q4"][-1], [1, 2, 5, 4])


# --- ResultWriter regenerates them --------------------------------------------------------

def replay(desc, path):
    """Write ``desc`` of fixtures_21 through ``ResultWriter``."""
    meshes, qp = desc["meshes"], desc["quadrature"]
    block = desc["block"]
    quad = {block: (qp["qp_points"], qp["qp_weights"], qp["rule"], qp["point_order"])}

    def umesh(m):
        return UnstructuredMesh(m["points"], {block: m["cells"]}, node_sets=m["node_sets"], quadrature=quad,
                                hanging=m.get("hanging"))
    fields = [Field("displacement", "node", "x,y,z", "mm", "state"), Field("damage", "node", "d", "1", "state")]
    h_attrs = desc["cell_fields"]["H"][1]
    fields.append(Field("H", h_attrs["location"], h_attrs["components"], "MPa", "history"))
    fields.append(Field("phase_id", "cell", "id", "1", "state", static=True, data=desc["phase_id"][0]))
    w = ResultWriter(path, umesh(meshes[0]), fields, dim=2, analysis="plane_strain", run_id=desc["run_id"],
                     mesh_varies=True, materials={0: _iso(), 1: dict(_iso(), name="inclusion", E=400000.0)},
                     bcs={"top": dict(set=desc["load_set"], kind="displacement", components="y", value="load_parameter")},
                     history_units={"time": "1", "dt": "1", "load_parameter": "mm"}, sidecars=False)
    current = 0
    for i, inc in enumerate(desc["increments"]):
        row = dict(increment=i, time=0.1 * (i + 1), dt=0.1, load_parameter=0.001 * (i + 1), damage_solved=1)
        kw = {}
        if inc["mesh"] != current:
            m = meshes[inc["mesh"]]
            kw = dict(mesh=umesh(m), mesh_maps={k: m[k] for k in ("cell_parent", "node_origin", "node_parents")},
                      static={"phase_id": desc["phase_id"][inc["mesh"]]})
            current = inc["mesh"]
        fl = None
        if inc["step"] >= 0:
            s = inc["step"]
            fl = dict(displacement=desc["point_fields"]["displacement"][0][s], damage=desc["point_fields"]["damage"][0][s],
                      H=desc["cell_fields"]["H"][0][s])
        assert w.append(row, fields=fl, **kw) == inc["step"]
    return w.close()


def _iso():
    from result_v2.fixtures import ISO
    return dict(ISO)


@pytest.mark.parametrize("name", sorted(FIXTURES_21))
def test_writer_regenerates_the_fixture(name, tmp_path):
    from result_v2.fixtures_21 import _write
    desc, _ = FIXTURES_21[name]()
    _write(tmp_path / "ref.vtkhdf", desc)
    out = replay(desc, tmp_path / "run" / "result.vtkhdf")
    assert out.name == "result.vtkhdf" and not (tmp_path / "run" / "result.vtkhdf.partial").exists()
    assert compare(tmp_path / "ref.vtkhdf", out) == []
    assert validate(out).ok
    with h5py.File(out, "r") as f:
        assert f.attrs["phast_schema_version"] == "2.1.0" and f.attrs["mesh_varies"]
        assert f["meta"].attrs["layout_2_1_features"].startswith("mesh_varies")


# --- validator: broken copies -------------------------------------------------------------------------

def shorten(f, path, rows):
    """Replace a dataset by its first ``rows`` rows (the fixtures are not chunked, so they cannot be resized)."""
    ds = f[path]
    data, attrs = ds[:rows], dict(ds.attrs)
    del f[path]
    new = f.create_dataset(path, data=data)
    for k, v in attrs.items():
        new.attrs[k] = v


def _mutated(built, tmp_path, name, fn, tag):
    path = copy_of(built, name, tmp_path, tag)
    with h5py.File(path, "r+") as f:
        fn(f)
    return errors(path)


BREAKS = {
    "part-offsets": ("refined_t3", lambda f: f["VTKHDF/Steps/PartOffsets"].__setitem__(2, 7), "varying-mesh"),
    "point-offsets": ("refined_t3", lambda f: f["VTKHDF/Steps/PointOffsets"].__setitem__(2, 5), "varying-mesh"),
    "cell-offsets": ("refined_t3", lambda f: f["VTKHDF/Steps/CellOffsets"].__setitem__((2, 0), 3), "varying-mesh"),
    "mesh-index": ("refined_t3", lambda f: f["history/mesh_index"].__setitem__(4, 1), "varying-mesh"),
    "mesh-index-missing": ("refined_t3", lambda f: f["history"].__delitem__("mesh_index"), "history-column-missing"),
    "counts-length": ("refined_t3", lambda f: shorten(f, "VTKHDF/NumberOfCells", 2), "varying-mesh"),
    "points-short": ("refined_t3", lambda f: shorten(f, "VTKHDF/Points", 10), "array-short"),
    "version-group": ("refined_t3", lambda f: f["mesh/versions"].__delitem__("2"), "varying-mesh"),
    "cell-parent-range": ("refined_t3", lambda f: f["mesh/versions/1/cell_parent"].__setitem__(0, 5), "mesh-version"),
    "node-origin-range": ("refined_t3", lambda f: f["mesh/versions/2/node_origin"].__setitem__(1, 9), "mesh-version"),
    "node-parents-new": ("refined_t3", lambda f: f["mesh/versions/1/node_parents"].__setitem__(4, [-1, -1]), "mesh-version"),
    "node-parents-old": ("refined_t3", lambda f: f["mesh/versions/1/node_parents"].__setitem__(0, [1, 2]), "mesh-version"),
    "origin-duplicate": ("refined_t3", lambda f: f["mesh/versions/1/node_origin"].__setitem__(1, 0), "mesh-version"),
    "maps-missing": ("refined_t3", lambda f: f["mesh/versions/1"].__delitem__("cell_parent"), "mesh-version"),
    "maps-shape": ("refined_t3", lambda f: (f["mesh/versions/1"].__delitem__("node_origin"),
                                            f["mesh/versions/1"].create_dataset("node_origin", data=np.zeros(3, np.int64))), "mesh-version"),
    "node-set-range": ("refined_t3", lambda f: f["mesh/versions/2/node_sets/top"].__setitem__(1, 40), "mesh-version"),
    "hanging-sum": ("refined_q4_hanging", lambda f: f["mesh/versions/1/hanging_nodes/weights"].__setitem__((0, 0), 0.7), "hanging-nodes"),
    "hanging-master": ("refined_q4_hanging", lambda f: f["mesh/versions/1/hanging_nodes/master_nodes"].__setitem__((0, 0), 7), "hanging-nodes"),
    "hanging-range": ("refined_q4_hanging", lambda f: f["mesh/versions/1/hanging_nodes/constrained_nodes"].__setitem__(0, 99), "hanging-nodes"),
    "array-short": ("refined_t3", lambda f: shorten(f, "VTKHDF/PointData/damage", 12), "array-short"),
    "fixed-shape-group": ("refined_t3", lambda f: f.create_dataset("quadrature/T3/stress", data=np.zeros((3, 2, 3))), "varying-mesh"),
}


@pytest.mark.parametrize("key", sorted(BREAKS))
def test_broken_copy_gives_its_code(built, tmp_path, key):
    name, fn, code = BREAKS[key]
    found = _mutated(built, tmp_path, name, fn, key)
    assert any(e.startswith(code + ":") for e in found), found


def test_a_2_0_file_with_a_changing_mesh_is_still_refused(built, tmp_path):
    found = _mutated(built, tmp_path, "refined_t3", lambda f: f.attrs.__setitem__("phast_schema_version", "2.0.0"), "v20")
    assert any(e.startswith("mesh-varies:") for e in found)


def test_mesh_varies_in_an_image_file_is_refused(tmp_path):
    grid = ImageGrid((2, 2, 2), (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="UnstructuredMesh"):
        ResultWriter(tmp_path / "a.vtkhdf", grid, [Field("damage", "node", "d", "1", "state")], dim=3, analysis="3d",
                     run_id="x", mesh_varies=True)


# --- writer refusals ----------------------------------------------------------------------------------------

def _w(tmp_path, **kw):
    mesh = UnstructuredMesh([[0, 0], [1, 0], [0, 1.0]], {"T3": [[0, 1, 2]]})
    return ResultWriter(tmp_path / "result.vtkhdf", mesh, kw.pop("fields", [Field("damage", "node", "d", "1", "state")]),
                        dim=2, analysis="plane_strain", run_id="x", sidecars=False,
                        history_units={"load_parameter": "mm"}, **kw)


ROW = dict(increment=0, time=1.0, dt=1.0, load_parameter=0.0, damage_solved=1)


def test_the_writer_refuses_what_a_changing_mesh_cannot_hold(tmp_path):
    with pytest.raises(ValueError, match="stored outside /VTKHDF"):
        _w(tmp_path, mesh_varies=True, fields=[Field("damage", "node", "d", "1", "state"),
                                               Field("stress", "cell", "xx,yy,xy", "MPa", "derived", placement="derived")])
    w = _w(tmp_path, mesh_varies=True)
    fine = UnstructuredMesh([[0, 0], [1, 0], [0, 1.0], [1, 1]], {"T3": [[0, 1, 2], [1, 3, 2]]})
    maps = dict(cell_parent=[0, 0], node_origin=[0, 1, 2, -1], node_parents=[[-1, -1]] * 3 + [[1, 2]])
    with pytest.raises(ValueError, match="mesh_maps and static arrays belong to a new mesh"):
        w.append(ROW, mesh_maps=maps)
    with pytest.raises(ValueError, match="lacks the maps"):
        w.append(ROW, mesh=fine, fields=dict(damage=np.zeros(4)))
    with pytest.raises(ValueError, match="map node_parents: expected shape"):
        w.append(ROW, mesh=fine, mesh_maps=dict(maps, node_parents=[[0, 1]] * 3), fields=dict(damage=np.zeros(4)))
    with pytest.raises(TypeError, match="UnstructuredMesh"):
        w.append(ROW, mesh="not a mesh", mesh_maps=maps)
    w.append(ROW, mesh=fine, mesh_maps=maps, fields=dict(damage=np.arange(4.0)))        # the valid one still works
    w.close()
    assert validate(tmp_path / "result.vtkhdf").ok
    w2 = _w(tmp_path / "fixed")
    with pytest.raises(ValueError, match="mesh_varies=False"):
        w2.append(ROW, mesh=fine, mesh_maps=maps, fields=dict(damage=np.arange(4.0)))
    w2.close()


def test_static_data_is_needed_on_every_mesh(tmp_path):
    mesh = UnstructuredMesh([[0, 0], [1, 0], [0, 1.0]], {"T3": [[0, 1, 2]]})
    fields = [Field("damage", "node", "d", "1", "state"), Field("phase_id", "cell", "id", "1", "state", static=True, data=[0])]
    w = ResultWriter(tmp_path / "result.vtkhdf", mesh, fields, dim=2, analysis="plane_strain", run_id="x", sidecars=False,
                     mesh_varies=True, history_units={"load_parameter": "mm"})
    fine = UnstructuredMesh([[0, 0], [1, 0], [0, 1.0], [1, 1]], {"T3": [[0, 1, 2], [1, 3, 2]]})
    maps = dict(cell_parent=[0, 0], node_origin=[0, 1, 2, -1], node_parents=[[-1, -1]] * 3 + [[1, 2]])
    with pytest.raises(ValueError, match="static fields"):
        w.append(ROW, mesh=fine, mesh_maps=maps, fields=dict(damage=np.zeros(4)))
    w.append(ROW, mesh=fine, mesh_maps=maps, static=dict(phase_id=[0, 0]), fields=dict(damage=np.zeros(4)))
    w.close()
    assert validate(tmp_path / "result.vtkhdf").ok


# --- the maps of the solver's own refinement -----------------------------------------------------------------------

def _fem_square():
    from phast.core.mesh import FEMMesh
    nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]], dtype=torch.float64)
    elems = torch.tensor([[1, 0, 4], [1, 4, 2], [3, 0, 4], [3, 4, 2]])
    return FEMMesh.from_tensors(nodes, elems, node_sets={"bottom": torch.tensor([0, 1]), "top": torch.tensor([2, 3])},
                                device="cpu", dtype=torch.float64)


def _umesh(fem, block):
    return UnstructuredMesh(fem.nodes.numpy(), {block: fem.elements.numpy()},
                            node_sets={k: v.numpy() for k, v in fem.node_sets.items()})


def test_maps_of_the_solvers_t3_refinement(tmp_path):
    from phast.solvers.adaptive import refine_mesh
    m0 = _fem_square()
    m1, parent_map, child_map = refine_mesh(m0, torch.tensor([True, False, False, False]))
    maps = refinement_maps(m0.n_nodes, m1.n_nodes, parent_map, child_map)
    fields = [Field("damage", "node", "d", "1", "state")]
    w = ResultWriter(tmp_path / "result.vtkhdf", _umesh(m0, "T3"), fields, dim=2, analysis="plane_strain", run_id="amr",
                     mesh_varies=True, history_units={"load_parameter": "mm"}, sidecars=False)
    w.append(ROW, fields=dict(damage=np.linspace(0, 1, m0.n_nodes)))
    w.append(dict(ROW, increment=1, time=2.0), fields=dict(damage=np.linspace(0, 1, m1.n_nodes)), mesh=_umesh(m1, "T3"),
             mesh_maps=maps)
    w.close()
    path = tmp_path / "result.vtkhdf"
    assert validate(path).ok
    r = load_result(path)
    v = r.mesh_versions()[1]
    pts0, pts1 = r.mesh_arrays(0)["points"], r.mesh_arrays(1)["points"]
    new = v["node_origin"] < 0
    assert new.sum() == m1.n_nodes - m0.n_nodes >= 1
    assert np.allclose(pts1[new], 0.5 * (pts0[v["node_parents"][new, 0]] + pts0[v["node_parents"][new, 1]]))
    assert np.array_equal(v["cell_parent"], [child_map[i] for i in range(m1.n_elems)])
    assert r.field("damage", 0).shape == (5,) and r.field("damage", 1).shape == (m1.n_nodes,)
    assert set(r.mesh_arrays(1)["node_sets"]) == {"bottom", "top"}


def test_refinement_maps_reject_inconsistent_input():
    with pytest.raises(ValueError, match="not among the"):
        refinement_maps(4, 6, {2: (0, 1)}, {0: 0})
    with pytest.raises(ValueError, match="no parent nodes"):
        refinement_maps(4, 6, {4: (0, 1)}, {0: 0})


# --- recovery ------------------------------------------------------------------------------------------------------

def test_recover_trims_a_changing_mesh_file_to_its_complete_steps(built, tmp_path):
    src = copy_of(built, "refined_t3", tmp_path, "killed")
    with h5py.File(src, "r+") as f:                       # as if killed after the offsets of step 2, before NSteps
        f["VTKHDF/Steps"].attrs["NSteps"] = 2
        f.attrs["status"] = "running"
    result = recover(src, tmp_path / "recovered.vtkhdf", use_h5clear=False)
    assert result.report.ok, result.report.format_text()
    r = load_result(result.output)
    assert r.n_steps() == 2 and r.step_mesh().tolist() == [0, 1]
    assert r.field("damage", 1).shape == (5,) and r.mesh_arrays(1)["mesh_version"] == 1
    with h5py.File(result.output, "r") as f:
        assert f["history/mesh_index"][...].tolist() == [0, 1, 1]
        assert f["VTKHDF/PointData/damage"].shape[0] == 4 + 5
