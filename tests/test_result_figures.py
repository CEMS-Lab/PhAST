"""``phast figures``: standard figures of a stored run (layout 2.x).

Every conformance fixture is drawn (the interrupted files too, with fewer
stored steps); voxel sections are checked against the coded fixture values so
that a transposed section cannot pass; the reader methods the figures rely on
are tested here as well.
"""
from __future__ import annotations

import hashlib
import json

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import phast
from phast.io.result.figures import draw_all, main
from phast.io.result.figures.data import (FieldInfo, RunData, build_mesh2d, corner_groups_from_cells, corner_groups_from_vtk,
                                          node_to_cell, voxel_slice)
from phast.io.result.figures.maps import _surface_polygons, draw_mesh_values, limits, section_extent
from phast.io.result.figures.sheets import (_short_attribute, compute_interface_norms, field_pages, field_sheet,
                                            setup_sheet)
from phast.io.result.surface import boundary_surface, cells_by_type, face_normals, grid_surface, near_octant_removed
from result_v2 import fixtures3d as f3
from result_v2 import fixtures as fx
from phast.io.result.figures.style import figure, png_size
from result_v2.fixtures import build_all

DPI = 70


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("result_v2_figures")
    return out, build_all(out)


def _fixture_files(built):
    out, expected = built
    return {name: (out / exp["file"], exp) for name, exp in expected.items()
            if "file" in exp and (out / exp["file"]).is_file()}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_outputs(out, rows):
    manifest = json.loads((out / "manifest.json").read_text())
    assert [r["file"] for r in manifest] == [r["file"] for r in rows]
    for row in manifest:
        png = out / row["file"]
        assert png.is_file()
        w, h = png_size(png)
        assert (w, h) == (row["width_px"], row["height_px"])
        assert w <= 2000 and h < 2000 and row["review_dimension_passed"]
        assert row["artifact_type"] == "image" and row["size_bytes"] == png.stat().st_size
        assert row["kind"] in {"setup", "fields", "components", "damage_steps", "evolution", "interfaces"}
        assert row["source_files"]
    return manifest


def test_every_fixture_draws(built, tmp_path):
    files = _fixture_files(built)
    assert {"t3_notched_plate", "voxel_h8", "sent_lagged", "setup_profile", "interrupted_after_rows"} <= set(files)
    for name, (path, exp) in files.items():
        before = _sha(path)
        out = tmp_path / name
        rows = draw_all(path, out, dpi=DPI)
        manifest = _check_outputs(out, rows)
        assert _sha(path) == before                            # read only
        assert manifest[0]["source_files"][0] == {"file": path.name, "sha256": before}
        kinds = {r["kind"] for r in manifest}
        if exp["nsteps"] == 0:
            assert kinds == {"setup"}, name
            continue
        assert {"setup", "fields", "evolution"} <= kinds, name
        steps = {s for r in manifest for s in r["steps"]}
        assert steps <= set(range(exp["nsteps"])), name       # interrupted files: complete steps only
        assert "damage_steps" in kinds


def test_fixture_specific_sheets(built, tmp_path):
    files = _fixture_files(built)
    rows = draw_all(files["t3_notched_plate"][0], tmp_path / "t3", dpi=DPI)
    comps = [r for r in rows if r["kind"] == "components"]
    assert comps and set(comps[0]["fields"]) == {"stress", "strain"}
    assert any("engineering shear" in n for n in comps[0]["notes"])
    fields = [r for r in rows if r["kind"] == "fields"][0]
    assert {"displacement", "damage", "history_field", "stress_zz"} <= set(fields["fields"])

    q4 = draw_all(files["q4_plate"][0], tmp_path / "q4", dpi=DPI)
    q4_fields = [r for r in q4 if r["kind"] == "fields"][0]
    assert any("mean over the points" in n for n in q4_fields["notes"])

    lagged = draw_all(files["sent_lagged"][0], tmp_path / "lagged", dpi=DPI)
    evolution = [r for r in lagged if r["kind"] == "evolution"][0]
    assert {"damage_solved", "mechanics_iterations", "energy_elastic"} <= set(evolution["history_columns"])

    voxel = draw_all(files["voxel_h8"][0], tmp_path / "voxel", dpi=DPI)
    assert any(r["kind"] == "fields" and "history_field" in r["fields"] for r in voxel)

    run = RunData(phast.load_result(files["sent_lagged"][0]))
    assert run.fields["damage"].role == "damage"              # every damage-like field: one map on [0, 1]
    cmap, norm = limits([np.full(3, 1e-4)], run.fields["damage"].role)
    assert (norm.vmin, norm.vmax) == (0.0, 1.0)


def test_quadratic_cells_use_their_corner_nodes():
    # one T6 and one Q8 in VTK order (corners first, then mid-side nodes)
    points = np.array([[0, 0], [2, 0], [0, 2], [1, 0], [1, 1], [0, 1],
                       [2, 0.0], [4, 0], [4, 2], [2, 2], [3, 0], [4, 1], [3, 2], [2, 1]], float)
    conn = np.concatenate([np.arange(6), np.arange(6, 14)])
    offsets = np.array([0, 6, 14])
    groups, names = corner_groups_from_vtk(conn, offsets, np.array([22, 23]))
    assert names == ["Q8", "T6"]
    corners = {len(c[0]): c for _, c in groups}
    assert corners[3].tolist() == [[0, 1, 2]] and corners[4].tolist() == [[6, 7, 8, 9]]
    mesh = build_mesh2d(points, groups, 2, names)
    assert mesh.areas.tolist() == pytest.approx([2.0, 4.0])
    assert len(mesh.tri) == 3 and sorted(mesh.tri_cell.tolist()) == [0, 1, 1]
    for cells, expected in ((np.arange(12).reshape(2, 6), [[0, 1, 2], [6, 7, 8]]),
                            (np.arange(18).reshape(2, 9), [[0, 1, 2, 3], [9, 10, 11, 12]])):
        _, legacy_corners = corner_groups_from_cells(cells)[0][0]
        assert legacy_corners.tolist() == expected

    class _Run:                                               # the attributes draw_mesh_values reads
        mesh2d, length_scale = mesh, 1.0

        @staticmethod
        def axis_label(axis):
            return axis

    with figure(3, 2) as fig:
        ax = fig.add_subplot(1, 1, 1)
        art = draw_mesh_values(ax, _Run, np.arange(len(points), dtype=float), "node", "viridis",
                               matplotlib.colors.Normalize(0, 13))
        fig.canvas.draw()                                     # mid-side nodes are unused points of the triangulation
        assert art is not None


def test_steps_first_peak_last_and_integers(built):
    files = _fixture_files(built)
    run = RunData(phast.load_result(files["t3_notched_plate"][0]))
    chosen = run.choose_steps("first,peak,last")
    # reaction_top_y = -10 * increment: largest at the last increment, stored as step 2
    assert [s for s, _ in chosen] == [0, 2]
    assert "largest" in chosen[1][1] and "last" in chosen[1][1]
    assert [s for s, _ in run.choose_steps("0,-2,-1")] == [0, 1, 2]
    with pytest.raises(ValueError, match="not stored"):
        run.choose_steps("7")
    interrupted = RunData(phast.load_result(files["interrupted_after_offsets"][0]))
    assert [s for s, _ in interrupted.choose_steps("first,peak,last")] == [0, 1]


def test_voxel_sections_follow_the_coded_values(built):
    files = _fixture_files(built)
    run = RunData(phast.load_result(files["voxel_h8"][0]))
    nx, ny, nz = run.shape
    assert (nx, ny, nz) == (6, 5, 4)
    s = 2
    d = run.values(run.fields["damage"], s)                 # value = 1e6 s + 1e4 i + 1e2 j + k
    assert d.shape == (nz + 1, ny + 1, nx + 1)
    code = lambda i, j, k: 1e6 * s + 1e4 * i + 1e2 * j + k  # noqa: E731
    xy, xz, yz = voxel_slice(d, "xy", 3), voxel_slice(d, "xz", 4), voxel_slice(d, "yz", 5)
    assert xy.shape == (ny + 1, nx + 1) and xy[2, 5] == code(5, 2, 3)      # rows y, columns x
    assert xz.shape == (nz + 1, nx + 1) and xz[1, 6] == code(6, 4, 1)      # rows z, columns x
    assert yz.shape == (nz + 1, ny + 1) and yz[3, 2] == code(5, 2, 3)      # rows z, columns y
    H = run.values(run.fields["history_field"], s)          # per point -> cell mean (points 0..7 -> +3.5)
    assert H.shape == (nz, ny, nx)
    assert H[2, 3, 4] == pytest.approx(1e6 * s + 10 * (1e4 * 4 + 1e2 * 3 + 2) + 3.5)
    assert node_to_cell(d)[0, 0, 0] == pytest.approx(code(0.5, 0.5, 0.5))
    ext = section_extent(run, "xz", "cell")
    assert ext == pytest.approx([0.0, nx * 0.5, 0.0, nz * 0.5])
    phase = run.phase_cells                                  # (i + j + k) % 3
    assert phase.shape == (nz, ny, nx) and phase[2, 3, 4] == (4 + 3 + 2) % 3


def test_older_stores_are_refused(tmp_path):
    run = tmp_path / "older"
    run.mkdir()
    (run / "run_metadata.json").write_text(json.dumps({"problem": "older store"}), encoding="utf-8")
    with pytest.raises(phast.ResultLoadError, match="layout-2.x"):
        draw_all(run, tmp_path / "figs", dpi=DPI)
    assert main([str(run), "-o", str(tmp_path / "figs")]) == 2


def test_units_come_from_the_file_when_the_run_directory_has_json_metadata(built, tmp_path):
    import json as _json
    import shutil
    path, _ = _fixture_files(built)["t3_notched_plate"]
    run = tmp_path / "run"
    run.mkdir()
    shutil.copyfile(path, run / "result.vtkhdf")
    (run / "run_metadata.json").write_text(_json.dumps({"problem": "json metadata"}), encoding="utf-8")
    result = phast.load_result(run)
    assert result.metadata() == {"problem": "json metadata"}
    assert result.file_metadata()["unit_system"] == "mm-MPa-N-s" and result.file_metadata()["meta"]["software"]
    assert RunData(result).length_units == "mm"


def test_reader_attribute_additions(built):
    files = _fixture_files(built)
    result = phast.load_result(files["t3_notched_plate"][0])
    attrs = result.field_attributes("H")                      # alias of history_field, stored as H
    assert attrs["units"] == "MPa" and attrs["location"] == "cell" and attrs["kind"] == "history"
    assert result.field_attributes("history_field") == attrs
    hist = result.history_attributes()
    assert hist["energy/elastic"]["method"] == "integral of psi over cells"
    assert hist["reaction/top"]["components"] == "x,y,z"
    assert result.source_files() == [files["t3_notched_plate"][0]]
    with pytest.raises(phast.ResultLoadError):
        result.field_attributes("nonexistent")


def test_cli(built, tmp_path, capsys):
    files = _fixture_files(built)
    path = files["two_phase_2d"][0]
    assert main([str(path), "-o", str(tmp_path / "cli"), "--steps", "0,-1", "--dpi", str(DPI),
                 "--fields", "damage,u", "--deform-scale", "auto"]) == 0
    manifest = json.loads((tmp_path / "cli" / "manifest.json").read_text())
    fields = [r for r in manifest if r["kind"] == "fields"]
    assert [r["step"] for r in fields] == [0, 2]
    assert set(fields[0]["fields"]) == {"damage", "displacement"}
    assert any("deformed shape" in n for n in fields[0]["notes"])
    assert "manifest" in capsys.readouterr().out
    assert main([str(path), "-o", str(tmp_path / "bad"), "--steps", "9"]) == 2


def test_many_fields_split_into_state_and_derived_sheets(built):
    run = RunData(phast.load_result(_fixture_files(built)["voxel_h8"][0]))   # 6 x 5 x 4: three sections + iso
    state = [FieldInfo(n, "node", (), "1", kind="state") for n in ("damage", "a", "b")]
    derived = [FieldInfo(n, "cell", (), "MPa", kind="derived") for n in ("p", "q", "r", "s")]
    pages = field_pages(run, state + derived)
    assert [g for g, _ in pages] == ["state", "derived", "derived"]           # at most three rows per sheet
    assert [len(m) for _, m in pages] == [3, 3, 1]
    assert [g for g, _ in field_pages(run, state[:2])] == ["fields"]


def test_labels_name_each_field_once():
    fluct = FieldInfo("displacement_fluctuation", "node", ("x", "y", "z"), "mm", kind="state")
    assert fluct.title() == "displacement fluctuation, magnitude"
    assert FieldInfo("displacement", "node", ("x", "y"), "mm").title().count("displacement") == 1
    assert FieldInfo("psi_driving", "cell", (), "MPa").title() == "psi driving"
    H = FieldInfo("history_field", "cell", (), "MPa", per_point=8)
    assert H.title() == "history variable $H$, mean of the 8 points of each cell"


def test_material_text_keeps_numbers_and_short_strings():
    assert _short_attribute("E", 210000.0, {"E": "MPa"}) == "E = 210000 MPa"
    assert _short_attribute("fracture_law", "PFCZM", {}) == "fracture_law = PFCZM"
    assert _short_attribute("residual_stiffness", 1e-7, {"residual_stiffness": "1"}) == "residual_stiffness = 1e-07"
    assert _short_attribute("sources", '{"E": "micro_phase_field.phases.fibre.E"}', {}) is None
    assert _short_attribute("description", "x" * 41, {}) is None
    assert _short_attribute("stiffness", np.eye(6), {}) is None


def test_partial_redraw_keeps_figures_of_other_kinds(tmp_path):
    import json as _json
    from phast.io.result.figures import draw_all
    from result_v2.fixtures import FIXTURES, write_fixture
    write_fixture(tmp_path / "t3.vtkhdf", FIXTURES["t3_notched_plate"]()[0])
    out = tmp_path / "fig"
    draw_all(tmp_path / "t3.vtkhdf", out)
    before = sorted(p.name for p in out.glob("*.png"))
    draw_all(tmp_path / "t3.vtkhdf", out, kinds=("evolution",))
    assert sorted(p.name for p in out.glob("*.png")) == before
    listed = sorted(r["file"] for r in _json.loads((out / "manifest.json").read_text()))
    assert listed == before


# --- cohesive interfaces and unstructured 3D meshes (surfaces, cut-aways) ----------------------------------------


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    out = tmp_path_factory.mktemp("figure_surfaces")
    built = f3.build_all(out)
    fx.write_fixture(out / "cohesive_dcb.vtkhdf", fx.cohesive_dcb()[0])
    return out, {k: p for k, (p, _) in built.items()}


def _kinds(rows):
    return [r["kind"] for r in rows]


def test_unstructured_3d_meshes_are_drawn_as_surfaces(files, tmp_path):
    _, built = files
    for name in ("h8_block", "tet4_block", "h8_cohesive"):
        run = RunData(phast.load_result(built[name]))
        assert run.kind == "mesh3d" and run.groups3d
        assert any("boundary faces" in n for n in run.notes_for("geometry"))
        rows = draw_all(built[name], tmp_path / name, dpi=DPI)
        kinds = _kinds(rows)
        assert {"setup", "fields", "components", "damage_steps", "evolution"} <= set(kinds)
        for row in rows:
            assert max(row["width_px"], row["height_px"]) < 2000
            assert (tmp_path / name / row["file"]).is_file()
        damage = next(r for r in rows if r["kind"] == "damage_steps")
        assert damage["fields"] == ["damage"]
        fields = next(r for r in rows if r["kind"] == "fields")
        assert {"displacement", "damage", "history_field"} <= set(fields["fields"])       # cell-valued H is drawn too


def test_front_faces_of_the_hexahedral_block(files):
    _, built = files
    run = RunData(phast.load_result(built["h8_block"]))
    n_faces = sum(len(g[1]) for g in _surface_polygons(run, False))
    assert n_faces == 4 * 3 + 3 * 2 + 4 * 2                       # top, +x side, +y side of a 4 x 3 x 2 block
    cut = sum(len(g[1]) for g in _surface_polygons(run, True))
    assert cut == 4 * 3 - 2 + 3 * 2 - 1 + 4 * 2 - 1 + 2 + 1 + 1   # two cells removed at the near corner; floor and walls of the notch appear


def test_the_fallback_for_cell_types_without_surfaces_still_draws(files, tmp_path):
    _, built = files
    run = RunData(phast.load_result(built["h8_block"]))
    run.groups3d = {}
    assert setup_sheet(run, tmp_path / "setup.png", DPI)["kind"] == "setup"
    info = run.fields["damage"]
    norms = {("damage", None): (matplotlib.colormaps["magma"], matplotlib.colors.Normalize(0, 1))}
    row = field_sheet(run, 0, "first", [info], norms, tmp_path / "f.png", DPI)
    assert row is not None and max(png_size(tmp_path / "f.png")) < 2000


def test_interface_sheets_draw_the_interface_arrays(files, tmp_path):
    out, built = files
    for path, kind, n_cells in ((out / "cohesive_dcb.vtkhdf", "mesh2d", 6), (built["h8_cohesive"], "mesh3d", 12)):
        run = RunData(phast.load_result(path))
        assert run.kind == kind and len(run.interfaces) == 1
        block = run.interfaces[0]
        assert block.name == "crack" and len(block.nodes) == n_cells
        assert list(block.fields) == ["cohesive_damage"]
        raw = np.asarray(run.result.quadrature("cohesive_damage", 1, block="crack"))
        np.testing.assert_allclose(run.interface_values(block, "cohesive_damage", 1), raw.mean(axis=1))
        norms = compute_interface_norms(run, [0, 1, 2])
        assert norms["cohesive_damage"][1].vmin == 0.0 and norms["cohesive_damage"][1].vmax == 1.0   # damage-like: [0, 1]
        rows = draw_all(path, tmp_path / kind, dpi=DPI)
        sheets = [r for r in rows if r["kind"] == "interfaces"]
        assert [r["step"] for r in sheets] == [r["steps"][0] for r in sheets] and len(sheets) >= 2
        assert all(r["fields"] == ["cohesive_damage"] for r in sheets)
        manifest = json.loads((tmp_path / kind / "manifest.json").read_text())
        assert {r["file"] for r in manifest if r["kind"] == "interfaces"} == {r["file"] for r in sheets}
        setup = next(r for r in rows if r["kind"] == "setup")
        assert any("Cohesive interface" in n for n in setup["notes"])


def test_runs_without_interfaces_have_no_interface_sheet(files, tmp_path):
    _, built = files
    rows = draw_all(built["h8_block"], tmp_path / "plain", dpi=DPI, kinds=("fields",))
    assert "interfaces" not in _kinds(rows)
    assert RunData(phast.load_result(built["h8_block"])).interfaces == []


# --- surface helpers ----------------------------------------------------------------------------------------

def test_hexahedral_and_tetrahedral_boundaries_are_closed_and_outward():
    pts, hexes = f3.hex_mesh(4, 3, 2)
    tets = f3.split_tets(pts, hexes)
    centre = pts.mean(axis=0)
    for conn, vtk, n_expected in ((hexes, 12, 52), (tets, 10, 104)):
        groups = cells_by_type(conn.ravel(), np.arange(0, conn.size + 1, conn.shape[1]), np.full(len(conn), vtk))
        surf = boundary_surface(pts, groups)
        assert surf.n_faces == n_expected
        for nodes, owner in surf.polygons():
            n = face_normals(pts, nodes)
            assert (np.einsum("ij,ij->i", n, pts[nodes].mean(axis=1) - centre) > 0).all()
            assert ((owner >= 0) & (owner < len(conn))).all()
        t, tc = surf.triangles()
        assert len(t) == len(surf.tris) + 2 * len(surf.quads) and len(tc) == len(t)


def test_cut_away_removes_the_near_octant_cells():
    pts, hexes = f3.hex_mesh(4, 3, 2)
    groups = cells_by_type(hexes.ravel(), np.arange(0, hexes.size + 1, 8), np.full(len(hexes), 12))
    keep = near_octant_removed(pts, groups, len(hexes))
    assert (~keep).sum() == 2
    removed = np.nonzero(~keep)[0]
    centroids = pts[hexes[removed]].mean(axis=1)
    assert (centroids > pts.mean(axis=0) * 0 + np.array([2.0, 1.5, 1.0])).all()


def test_voxel_surface_faces_and_normals():
    mask = np.ones((2, 2, 3), bool)
    nodes, vox, axis = grid_surface(mask)
    assert len(nodes) == 2 * (3 * 2 + 3 * 2 + 2 * 2)
    nx, ny = 3, 2
    i = nodes % (nx + 1)
    j = (nodes // (nx + 1)) % (ny + 1)
    k = nodes // ((nx + 1) * (ny + 1))
    p = np.stack([i, j, k], axis=-1).astype(float)
    n = np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])
    assert (np.einsum("ij,ij->i", n, p.mean(axis=1) - np.array([1.5, 1.0, 1.0])) > 0).all()
    assert set(axis.tolist()) == {0, 1, 2, 3, 4, 5}
    mask[1, 1, 1:] = False                                          # remove two voxels at the corner
    nodes2, vox2, _ = grid_surface(mask)
    assert len(nodes2) == 32 and mask[tuple(vox2.T)].all()          # every face belongs to a present voxel
    assert len(grid_surface(np.zeros((2, 2, 2), bool))[0]) == 0
    single = grid_surface(np.ones((1, 1, 1), bool))[0]
    assert len(single) == 6
