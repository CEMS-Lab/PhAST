"""Layout-2.0 conformance fixtures: structure and values, read back with h5py only.

The fixtures come from the independent reference builder in
``tests/result_v2/fixtures.py``. The ParaView side of the check is
``tests/result_v2/paraview_check.py`` (run with pvpython).
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from phast.io.result import load_schema
from result_v2.fixtures import FIXTURES, KILL_STAGES, build_all

SCHEMA = load_schema()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("result_v2")
    return out, build_all(out)


def complete_files(built):
    out, expected = built
    return [(name, out / exp["file"], exp) for name, exp in expected.items() if "file" in exp]


def read_probe(f, exp, probe):
    """Value at a probe, located through the VTKHDF step offsets (no other knowledge of the layout)."""
    v = f["VTKHDF"]
    group = "PointData" if probe["where"] == "point" else "CellData"
    ds = v[group][probe["array"]]
    step = probe["step"]
    off = 0 if step is None else int(v["Steps"][f"{group[:-4]}DataOffsets"][probe["array"]][step])
    if exp["type"] == "ImageData":
        block = ds[off]
        flat = block.reshape((-1,) + block.shape[3:])          # [z, y, x] flattened with x fastest = VTK order
        val = flat[probe["index"]]
    else:
        n = exp["n_points"] if probe["where"] == "point" else exp["n_cells"]
        val = ds[off: off + n][probe["index"]]
    val = np.atleast_1d(np.asarray(val, float))
    return float(val[probe["component"]] if val.size > 1 else val[0])


def test_all_fixtures_and_variants_are_built(built):
    _, expected = built
    assert set(FIXTURES) <= set(expected)
    assert all(f"interrupted_{s}" in expected for s in KILL_STAGES)
    assert expected["refused_mesh_varies"]["refused"] is True


def test_root_attributes_follow_the_schema(built):
    rules = SCHEMA["root_attributes"]["required"]
    for name, path, exp in complete_files(built):
        with h5py.File(path, "r") as f:
            for key, rule in rules.items():
                assert key in f.attrs, f"{name}: missing root attribute {key}"
                val = f.attrs[key]
                val = val.decode() if isinstance(val, bytes) else val
                if "values" in rule:
                    assert val in rule["values"], f"{name}: {key}={val!r}"
                if "value" in rule:
                    assert val == rule["value"] or (isinstance(rule["value"], bool) and bool(val) == rule["value"]), f"{name}: {key}={val!r}"
            assert f.attrs["status"] == exp["status"], name


def test_required_groups_and_meta(built):
    for name, path, exp in complete_files(built):
        with h5py.File(path, "r") as f:
            for g in SCHEMA["groups"]["required"]:
                assert g in f, f"{name}: missing group {g}"
            for m in SCHEMA["groups"]["meta_required"]:
                assert m in f["meta"], f"{name}: missing meta/{m}"


def test_step_counts_and_offsets_are_consistent(built):
    for name, path, exp in complete_files(built):
        with h5py.File(path, "r") as f:
            st = f["VTKHDF/Steps"]
            n = int(st.attrs["NSteps"])
            assert n == exp["nsteps"], name
            if exp["status"] == "complete":
                assert len(st["Values"]) == n, name
            for g, rows in (("PointData", exp["n_points"]), ("CellData", exp["n_cells"])):
                for arr, ods in st[f"{g[:-4]}DataOffsets"].items():
                    assert len(ods) >= n, f"{name}: {arr} offsets"
                    if n:
                        per = 1 if exp["type"] == "ImageData" else rows
                        assert int(ods[n - 1]) + per <= f["VTKHDF"][g][arr].shape[0], f"{name}: {arr} rows"


def test_links_resolve_and_no_virtual_datasets(built):
    for name, path, exp in complete_files(built):
        with h5py.File(path, "r") as f:
            def visit(key, link):
                if isinstance(link, h5py.SoftLink):
                    assert link.path.startswith("/VTKHDF/"), f"{name}: link {key} points outside /VTKHDF"
                    assert link.path in f, f"{name}: dangling link {key}"
            f.visititems_links(visit)
            def no_vds(key, obj):
                if isinstance(obj, h5py.Dataset):
                    assert not obj.is_virtual, f"{name}: virtual dataset {key}"
            f.visititems(no_vds)


def test_field_attributes_and_precision(built):
    req = SCHEMA["field_attributes"]["required"]
    f64 = {"state", "history"}
    for name, path, exp in complete_files(built):
        with h5py.File(path, "r") as f:
            for g in ("PointData", "CellData"):
                for arr, ds in f["VTKHDF"][g].items():
                    for a in req:
                        assert a in ds.attrs, f"{name}: {g}/{arr} lacks {a}"
                    kind = ds.attrs["kind"]
                    if kind in f64 and arr != "phase_id":
                        assert ds.dtype == np.float64, f"{name}: {arr} is {ds.dtype}"
                    assert kind in SCHEMA["field_attributes"]["kind"]
                    assert ds.attrs["location"] in SCHEMA["field_attributes"]["location"]


def test_history_columns_have_units_and_required_names(built):
    for name, path, exp in complete_files(built):
        with h5py.File(path, "r") as f:
            h = f["history"]
            if exp["nsteps"] == 0:
                continue
            for col in SCHEMA["history_columns"]["required"]:
                assert col in h, f"{name}: history/{col}"
            def units(key, obj):
                if isinstance(obj, h5py.Dataset):
                    assert "units" in obj.attrs, f"{name}: history/{key} lacks units"
            h.visititems(units)
            idx = h["step_index"][...]
            assert idx.max() + 1 == exp["nsteps"] or exp["status"] == "running"


def test_probe_values(built):
    for name, path, exp in complete_files(built):
        with h5py.File(path, "r") as f:
            for p in exp["probes"]:
                got = read_probe(f, exp, p)
                assert got == pytest.approx(p["value"], rel=1e-6, abs=1e-9), f"{name}: {p}"


def test_lagged_history_fields_differ(built):
    out, expected = built
    with h5py.File(out / expected["sent_lagged"]["file"], "r") as f:
        assert f.attrs["history_coupling"] == "lagged"
        cd = f["VTKHDF/CellData"]
        assert not np.allclose(cd["H"][...], cd["H_damage_input"][...])
        assert f["history/damage_solved"].dtype == np.int8
        assert set(f["history/damage_solved"][...].tolist()) == {0, 1}


def test_setup_profile_has_no_steps(built):
    out, expected = built
    with h5py.File(out / expected["setup_profile"]["file"], "r") as f:
        assert f.attrs["profile"] == "setup" and int(f["VTKHDF/Steps"].attrs["NSteps"]) == 0


def test_interrupted_files_keep_their_complete_steps(built):
    out, expected = built
    for stage in KILL_STAGES:
        exp = expected[f"interrupted_{stage}"]
        with h5py.File(out / exp["file"], "r") as f:
            assert f.attrs["status"] == "running"
            assert int(f["VTKHDF/Steps"].attrs["NSteps"]) == 2
            for p in exp["probes"]:
                assert read_probe(f, exp, p) == pytest.approx(p["value"], rel=1e-6, abs=1e-9)
