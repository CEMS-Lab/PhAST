"""phast.load_result on layout-2.x files.

The tests use the conformance fixtures of ``tests/result_v2``, which an
independent builder writes with h5py, so the reader is checked against files it
did not produce.
"""
from __future__ import annotations

import json
import shutil

import h5py
import numpy as np
import pytest

import phast
from phast.io.result import ResultFile
from result_v2.fixtures import KILL_STAGES, build_all


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("result_v2_reader")
    return out, build_all(out)


def _files(built):
    out, expected = built
    return [(name, out / exp["file"], exp) for name, exp in expected.items() if "file" in exp]


def _probe(result, exp, probe):
    step = -1 if probe["step"] is None else probe["step"]
    array = result.field(probe["array"], step)
    if exp["type"] == "ImageData":
        grid_dims = 3
        array = array.reshape((-1,) + array.shape[grid_dims:])
    value = np.atleast_1d(np.asarray(array[probe["index"]], float))
    return float(value[probe["component"]] if value.size > 1 else value[0])


# --- layout 2.0 ------------------------------------------------------------------

def test_every_fixture_loads_with_its_step_count_and_mesh(built):
    for name, path, exp in _files(built):
        result = phast.load_result(path)
        assert result.layouts() == ["2.0"], name
        assert result.file == path.resolve()
        assert result.n_steps() == exp["nsteps"], name
        assert result.steps() == list(range(exp["nsteps"]))
        mesh = result.mesh()
        assert (mesh["n_nodes"], mesh["n_elements"]) == (exp["n_points"], exp["n_cells"]), name
        assert mesh["source"] == path.name and mesh["type"] == exp["type"]


def test_probe_values_match_through_load_result(built):
    for name, path, exp in _files(built):
        result = phast.load_result(path)
        for probe in exp["probes"]:
            got = _probe(result, exp, probe)
            assert got == pytest.approx(probe["value"], rel=1e-6, abs=1e-9), f"{name}: {probe}"


def test_two_dimensional_vectors_have_two_components(built):
    out, expected = built
    result = phast.load_result(out / expected["explicit_dynamic"]["file"])
    n = expected["explicit_dynamic"]["n_points"]
    for name in ("displacement", "u", "velocity", "v", "acceleration", "a"):
        assert result.field(name, 1).shape == (n, 2), name
    assert result.mesh_arrays()["points"].shape == (n, 2)
    with h5py.File(out / expected["explicit_dynamic"]["file"], "r") as f:
        assert f["VTKHDF/PointData/velocity"].shape[1] == 3
    np.testing.assert_allclose(result.times(), expected["explicit_dynamic"]["times"])


def test_tensors_keep_their_stored_components(built):
    out, expected = built
    result = phast.load_result(out / expected["j2_plasticity"]["file"])
    e = expected["j2_plasticity"]["n_cells"]
    assert result.field("plastic_strain", 2).shape == (e, 3)
    assert result.field("stress").dtype == np.float32
    assert result.field("equivalent_plastic_strain").dtype == np.float64


def test_image_data_arrays_are_returned_on_the_grid(built):
    out, expected = built
    result = phast.load_result(out / expected["voxel_h8"]["file"])
    nx, ny, nz = 6, 5, 4
    assert result.field("damage", 0).shape == (nz + 1, ny + 1, nx + 1)
    assert result.field("u", 0).shape == (nz + 1, ny + 1, nx + 1, 3)
    assert result.field("H", 2).shape == (nz, ny, nx, 8)
    assert result.field("phase_id", 1).shape == (nz, ny, nx)
    damage = result.field("damage", 1)
    # value encodes 1e6*step + 1e4*i + 1e2*j + k for node (i, j, k)
    assert damage[3, 2, 5] == pytest.approx(1e6 + 1e4 * 5 + 1e2 * 2 + 3)
    grid = result.grid()
    assert grid["shape"] == [nx, ny, nz]
    assert grid["point_shape"] == [nz + 1, ny + 1, nx + 1] and grid["cell_shape"] == [nz, ny, nx]
    assert grid["spacing"] == [0.5, 0.5, 0.5] and grid["index_order"] == "x fastest"
    assert result.boundary_conditions()["periodic"]["controlled"] == "xx"
    names = {pid: p["name"] for pid, p in result.materials()["phases"].items()}
    assert names == {0: "fibre", 1: "interface", 2: "matrix"}


def test_history_per_quadrature_point_is_cells_by_points(built):
    out, expected = built
    result = phast.load_result(out / expected["q4_plate"]["file"])
    e = expected["q4_plate"]["n_cells"]
    h = result.field("H", 1)
    assert h.shape == (e, 4)
    assert h[2, 3] == pytest.approx(0.5 + 1000.0 + 10.0 * 2 + 3)
    np.testing.assert_array_equal(result.quadrature("H", 1), h)
    sig = result.quadrature("stress", 1)
    assert sig.shape == (e, 4, 3)
    assert sig[2, 3, 1] == pytest.approx(1000.0 + 20.0 + 3.0 + 0.01, rel=1e-6)
    assert result.mesh_arrays()["quadrature"]["Q4"]["point_order"].startswith("eta outer")
    with pytest.raises(phast.ResultLoadError, match="No quadrature-point array"):
        result.quadrature("damage", 0)


def test_cohesive_interface_points_are_read_as_quadrature(built):
    out, expected = built
    result = phast.load_result(out / expected["cohesive_dcb"]["file"])
    coh = result.quadrature("cohesive_damage", 2)
    assert coh.shape == (6, 2)
    assert coh[4, 1] == pytest.approx((2000.0 + 40.0 + 1.0) / 1e4)


def test_lagged_history_differs_from_damage_input(built):
    out, expected = built
    result = phast.load_result(out / expected["sent_lagged"]["file"])
    assert result.metadata()["history_coupling"] == "lagged"
    for step in range(result.n_steps()):
        h, h_in = result.field("H", step), result.field("H_damage_input", step)
        assert not np.allclose(h, h_in)
        np.testing.assert_allclose(h - h_in, 0.5)
    assert result.has_field("H_damage_input")


def test_derived_fields_map_through_step_index(built):
    out, expected = built
    result = phast.load_result(out / expected["sent_lagged"]["file"])
    assert "stress" in result.field_names()
    e = expected["sent_lagged"]["n_cells"]
    for step, row in ((2, 0), (3, 1), (5, 2), (-1, 2)):
        values = result.field("stress", step)
        assert values.shape == (e, 3)
        assert values[7, 0] == pytest.approx((1000.0 * row + 7) / 1e3, rel=1e-6)
    with pytest.raises(phast.ResultLoadError, match="only on some increments"):
        result.field("stress", 0)


def test_history_tables_and_columns(built):
    out, expected = built
    result = phast.load_result(out / expected["t3_notched_plate"]["file"])
    np.testing.assert_array_equal(result.step_index(), [0, -1, 1, -1, 2])
    columns = result.history_columns()
    assert columns["reaction/top"].shape == (5, 3)
    rows = result.history("history")
    assert len(rows) == 5
    assert rows[3]["reaction_top_y"] == pytest.approx(-30.0)
    assert rows[2]["energy_elastic"] == pytest.approx(0.2)
    assert rows[4]["step_index"] == 2
    assert result.history("energy")[1]["fracture"] == pytest.approx(0.01)
    lagged = phast.load_result(out / expected["sent_lagged"]["file"])
    assert [row["damage_solved"] for row in lagged.history("history")] == [1, 1, 0, 0, 1, 0]
    assert lagged.history_attributes()["energy/elastic"]["method"] == "integral of psi over cells"


def test_steps_are_indices_with_negative_values_from_the_end(built):
    out, expected = built
    result = phast.load_result(out / expected["t3_notched_plate"]["file"])
    np.testing.assert_array_equal(result.field("damage", -3), result.field("damage", 0))
    np.testing.assert_array_equal(result.field("damage"), result.field("damage", 2))
    with pytest.raises(phast.ResultLoadError, match="Step 3 not found"):
        result.field("damage", 3)
    with pytest.raises(phast.ResultLoadError, match="not stored directly"):
        result.field("von_mises")
    with pytest.raises(phast.ResultLoadError, match="Unknown field"):
        result.field("missing")


def test_setup_profile_has_mesh_and_materials_but_no_steps(built):
    out, expected = built
    exp = expected["setup_profile"]
    result = phast.load_result(out / exp["file"])
    assert result.n_steps() == 0 and result.steps() == [] and result.times().size == 0
    assert result.field_names() == ["phase_id"]
    assert result.field("phase_id").shape == (exp["n_cells"],)
    arrays = result.mesh_arrays()
    assert arrays["points"].shape == (exp["n_points"], 2)
    assert arrays["cells"]["T3"].shape == (exp["n_cells"], 3)
    assert arrays["cell_types"] == {"T3": "T3"}
    assert set(arrays["node_sets"]) == {"top"}
    materials = result.materials()
    assert materials["phases"][1]["name"] == "inclusion"
    assert materials["phases"][0]["units"]["Gc"] == "N/mm"
    assert materials["phase_id"].shape == (exp["n_cells"],)
    assert result.boundary_conditions()["bc"]["top"]["value"] == "load_parameter"
    assert result.step_index().size == 0
    assert result.metadata()["profile"] == "setup"
    with pytest.raises(phast.ResultLoadError):
        result.field("damage")


def test_interrupted_files_load_their_complete_steps(built):
    out, expected = built
    for stage in KILL_STAGES:
        exp = expected[f"interrupted_{stage}"]
        result = phast.load_result(out / exp["file"])
        n = exp["nsteps"]
        assert result.n_steps() == n, stage
        assert result.metadata()["status"] == "running"
        for probe in exp["probes"]:
            assert _probe(result, exp, probe) == pytest.approx(probe["value"], rel=1e-6, abs=1e-9)
        assert result.field("displacement", -1).shape == (exp["n_points"], 2)
        with pytest.raises(phast.ResultLoadError, match=f"Step {n} not found"):
            result.field("damage", n)
        assert result.step_index().max() < n


def test_mesh_arrays_of_an_unstructured_run(built):
    out, expected = built
    exp = expected["t3_notched_plate"]
    arrays = phast.load_result(out / exp["file"]).mesh_arrays()
    assert arrays["points"].shape == (exp["n_points"], 2)
    assert arrays["cells"]["T3"].shape == (exp["n_cells"], 3)
    assert arrays["edge_index"].shape[0] == 2
    assert arrays["types"].tolist() == [5] * exp["n_cells"]
    assert set(arrays["node_sets"]) == {"top", "bottom"}


def test_run_directory_with_result_file_needs_no_json_and_wins(built, tmp_path):
    out, expected = built
    run = tmp_path / "run"
    run.mkdir()
    shutil.copyfile(out / expected["t3_notched_plate"]["file"], run / "result.vtkhdf")
    result = phast.load_result(run)
    assert result.path == run.resolve() and result.file == (run / "result.vtkhdf").resolve()
    assert result.metadata()["run_id"] == "fixture-t3"
    assert result.manifest()["meta"]["software"] == "phast"
    assert result.mesh()["source"] == "result.vtkhdf"

    # an older store and sidecars in the same directory: the file still supplies the fields
    (run / "run_metadata.json").write_text(json.dumps({"problem": "both"}), encoding="utf-8")
    (run / "history.csv").write_text("step,max_d\n0,0.5\n", encoding="utf-8")
    with h5py.File(run / "training_data.h5", "w") as h5:
        h5.create_group("simulation_data/steps/step_0000").create_dataset("velocity", data=np.zeros((2, 2)))
    result = phast.load_result(run)
    assert result.layouts() == ["2.0"]
    assert "velocity" not in result.field_names()
    assert result.metadata() == {"problem": "both"}
    assert result.history("history") == [{"step": 0, "max_d": 0.5}]
    assert result.history("max_damage")[0]["max_damage"] == 0.5
    assert len(result.history("energy")) == 5


def test_unsupported_major_version_is_refused(built, tmp_path):
    out, expected = built
    copy = tmp_path / "future.vtkhdf"
    shutil.copyfile(out / expected["q4_plate"]["file"], copy)
    with h5py.File(copy, "r+") as f:
        f.attrs["phast_schema_version"] = "3.0.0"
    with pytest.raises(phast.ResultLoadError, match="major version 2"):
        ResultFile(copy)


def test_methods_of_layout_2_report_an_older_store(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_metadata.json").write_text(json.dumps({"problem": "older store"}), encoding="utf-8")
    result = phast.load_result(run)
    assert result.layouts() == [] and result.history_columns() == {} and result.history_attributes() == {}
    for call in (result.step_index, result.boundary_conditions, result.steps, result.mesh_arrays,
                 lambda: result.quadrature("history")):
        with pytest.raises(phast.ResultLoadError, match="needs a layout-2.x result file"):
            call()


def test_unrecognised_file_is_refused(tmp_path):
    path = tmp_path / "notes.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("x", data=np.zeros(3))
    with pytest.raises(phast.ResultLoadError, match="not a layout-2.x PhAST result file"):
        phast.load_result(path)
