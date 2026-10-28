"""Writer core (layout 2.0): the writer regenerates every conformance fixture.

Each fixture description from ``tests/result_v2/fixtures.py`` (the independent
reference builder) is replayed through ``ResultWriter``; every dataset and
field attribute of the reference file must appear in the writer's file with the
same values. Also: kill after a flushed step, dtype rules, ``.partial`` and
rename, the size rule for derived fields, refusals.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from phast.io.result import Field, ImageGrid, ResultWriter, UnstructuredMesh, estimate_size
from result_v2.fixtures import FIXTURES, write_fixture
from result_v2.helpers import compare

# Only the directory holding the phast package: other sys.path entries (some tests add src/phast
# itself) would let phast/io shadow the standard library io module in the child process.
_SRC = str(Path(__import__("phast").__file__).resolve().parents[1])


def _field(name, arr_attrs, dtype=None, placement="view", static=False, data=None):
    a = dict(arr_attrs)
    extra = {k: a.pop(k) for k in list(a) if k not in ("location", "components", "units", "kind")}
    return Field(name, a["location"], a["components"], a["units"], a["kind"], placement=placement,
                 dtype=dtype, static=static, data=data, attrs=extra)


def replay(desc, path):
    """Write the fixture description ``desc`` with the writer."""
    blocks = desc.get("cell_blocks", {})
    quad = {b["cell_type"]: (b["qp_points"], b["qp_weights"], b["rule"], b["point_order"]) for b in blocks.values()}
    if desc["kind"] == "image":
        mesh = ImageGrid(desc["grid"], desc["spacing"], cell_type="H8", quadrature=quad)
    else:
        ctype = next(iter(blocks.values()))["cell_type"]
        mesh = UnstructuredMesh(desc["points"], {ctype: desc["cells"]}, node_sets=desc.get("node_sets", {}),
                                quadrature=quad, graph="edge_index" in desc)
    fields = [_field(n, a, static=True, data=arr) for n, (arr, a) in desc.get("static_cell", {}).items()]
    series = {}
    for where in ("point_fields", "cell_fields"):
        for n, (arr, a, dt) in desc.get(where, {}).items():
            fields.append(_field(n, a, dtype=dt))
            series[n] = np.asarray(arr, dt)
    statics, derived_vals = {}, {}
    for p, (arr, a) in desc.get("extra", {}).items():
        if p.startswith("quadrature/"):
            _, block, name = p.split("/")
            fields.append(_field(name, a, dtype=np.asarray(arr).dtype, placement=f"quadrature:{block}"))
            series[f"{block}/{name}"] = np.asarray(arr)
        elif p.startswith("derived/") and p.endswith("/values"):
            name = p.split("/")[1]
            fields.append(_field(name, a, dtype=np.asarray(arr).dtype, placement="derived"))
            derived_vals[name] = np.asarray(arr)
        elif p.startswith("interfaces/") and a.get("kind") == "history":
            name = p.rsplit("/", 1)[1]
            arr = np.asarray(arr)
            fields.append(_field(name, dict(a, shape=arr.shape[1:]), placement="interface:" + p.split("/")[1]))
            series[f"{p.split('/')[1]}/{name}"] = arr
        elif not p.startswith("derived/"):
            statics[p] = (arr, a)
    hist = desc["history"]
    units = {k: a["units"] for k, (c, a) in hist.items()}
    hattrs = {k: {x: y for x, y in a.items() if x not in ("units", "codes")} for k, (c, a) in hist.items()}
    root = desc["root"]
    w = ResultWriter(path, mesh, fields, dim=root["dim"], analysis=root["analysis"], run_id=root["run_id"],
                     profile=root.get("profile", "run"), history_coupling=root.get("history_coupling", "fresh"),
                     materials=desc["phases"], bcs=desc.get("bc"), periodic=desc.get("periodic"),
                     history_units=units, history_attrs=hattrs, config=desc.get("config", "fixture: true\n"),
                     meta=desc.get("meta_attrs", {}), sidecars=False)
    for p, (arr, a) in statics.items():
        w.write_array(p, arr, a)
    for p, a in desc.get("extra_attrs", {}).items():
        w.write_attrs(p, a)
    if w.profile == "setup":
        return w.write_setup()
    step_of = np.asarray(hist["step_index"][0])
    dstep = {n: np.asarray(desc["extra"][f"derived/{n}/step_index"][0]) for n in derived_vals}
    for inc in range(len(step_of)):
        row = {k: c[inc] for k, (c, a) in hist.items() if k != "step_index"}
        s = int(step_of[inc])
        fl = {n: a[s] for n, a in series.items()} if s >= 0 else None
        dv = {n: derived_vals[n][dstep[n][inc]] for n in derived_vals if dstep[n][inc] >= 0}
        assert w.append(row, fields=fl, derived=dv) == s
    if desc.get("validation"):
        w.write_validation(desc["validation"], source=desc.get("validation_source"))
    return w.close(root.get("final_status", "complete"))


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_writer_regenerates_fixture(name, tmp_path):
    desc, _ = FIXTURES[name]()
    write_fixture(tmp_path / "ref.vtkhdf", desc)
    desc, _ = FIXTURES[name]()
    out = replay(desc, tmp_path / "run" / "result.vtkhdf")
    assert out.name == "result.vtkhdf" and not (tmp_path / "run" / "result.vtkhdf.partial").exists()
    assert (tmp_path / "run" / "result.vtkhdf.sha256").read_text().split()[0]
    assert compare(tmp_path / "ref.vtkhdf", out) == []


def test_writer_regenerated_fixtures_validate(tmp_path):
    validate = pytest.importorskip("phast.io.result.validate").validate
    for name, fn in FIXTURES.items():
        out = replay(fn()[0], tmp_path / name / "result.vtkhdf")
        report = validate(out)
        assert [x for x in report.findings if x.level == "ERROR"] == [], name


def test_round_trip_float64_is_exact(tmp_path):
    rng = np.random.default_rng(0)
    pts = rng.random((7, 2)); cells = np.array([[0, 1, 2], [2, 3, 4], [4, 5, 6]])
    fields = [Field("displacement", "node", "x,y", "mm", "state"), Field("H", "cell", "H", "MPa", "history")]
    u, H = rng.random((3, 7, 2)), rng.random((3, 3)) * 1e-17 + 1.0
    with ResultWriter(tmp_path / "result.vtkhdf", UnstructuredMesh(pts, {"T3": cells}), fields, dim=2,
                      analysis="plane_strain", run_id="rt", history_units={"load_parameter": "mm"}) as w:
        for s in range(3):
            w.append(dict(increment=s, time=0.1 * (s + 1), dt=0.1, load_parameter=0.01 * s, damage_solved=1),
                     fields=dict(displacement=u[s], H=H[s]))
    with h5py.File(tmp_path / "result.vtkhdf", "r") as f:
        disp = f["VTKHDF/PointData/displacement"][...].reshape(3, 7, 3)
        assert np.array_equal(disp[..., :2], u) and not disp[..., 2].any()
        assert np.array_equal(f["VTKHDF/CellData/H"][...].reshape(3, 3), H)
        assert f["VTKHDF/CellData/H"].dtype == np.float64
        assert np.array_equal(f["VTKHDF/Steps/Values"][...], [0.1, 0.2, 0.30000000000000004])
        assert f["history/time"].attrs["units"] == "s" and f["history/step_index"][...].tolist() == [0, 1, 2]
    assert (tmp_path / "history.csv").read_text().count("\n") == 4
    assert json.loads((tmp_path / "progress.json").read_text())["status"] == "complete"


def test_dtype_rules(tmp_path):
    with pytest.raises(ValueError, match="float64"):
        Field("damage", "node", "d", "1", "state", dtype=np.float32)
    assert Field("stress", "cell", "xx,yy,xy", "MPa", "derived", dtype=np.float32).dtype == np.float32
    assert Field("phase_id", "cell", "id", "1", "state").dtype == np.uint8
    mesh = UnstructuredMesh(np.eye(3)[:, :2], {"T3": [[0, 1, 2]]})
    w = ResultWriter(tmp_path / "result.vtkhdf", mesh, [Field("damage", "node", "d", "1", "state")], dim=2,
                     analysis="plane_strain", run_id="dt", history_units={"load_parameter": "mm"}, sidecars=False)
    with pytest.raises(ValueError, match="expected shape"):
        w.append(dict(increment=0, time=1.0, dt=1.0, load_parameter=0.0, damage_solved=1), fields=dict(damage=np.zeros(4)))
    w.append(dict(increment=0, time=1.0, dt=1.0, load_parameter=0.0, damage_solved=1),
             fields=dict(damage=np.zeros(3, np.float32)))
    w.close()
    with h5py.File(tmp_path / "result.vtkhdf", "r") as f:
        assert f["VTKHDF/PointData/damage"].dtype == np.float64


def test_refusals(tmp_path):
    mesh = UnstructuredMesh(np.eye(3)[:, :2], {"T3": [[0, 1, 2]]})
    common = dict(dim=2, analysis="plane_strain", run_id="x", sidecars=False)
    with pytest.raises(ValueError, match="H_damage_input"):
        ResultWriter(tmp_path / "a.vtkhdf", mesh, [Field("H", "cell", "H", "MPa", "history")],
                     history_coupling="lagged", **common)
    with pytest.raises(ValueError, match=".vtkhdf"):
        ResultWriter(tmp_path / "a.h5", mesh, [], **common)
    w = ResultWriter(tmp_path / "b.vtkhdf", mesh, [Field("damage", "node", "d", "1", "state")],
                     history_units={"load_parameter": "mm"}, **common)
    with pytest.raises(ValueError, match="lacks fields"):
        w.append(dict(increment=0, time=1.0, dt=1.0, load_parameter=0.0, damage_solved=1), fields={})
    with pytest.raises(ValueError, match="needs units"):
        w.append(dict(increment=0, time=1.0, dt=1.0, load_parameter=0.0, damage_solved=1, mystery=2.0))
    w.close()


def test_size_rule_moves_derived_fields_out_of_the_view(tmp_path):
    grid = ImageGrid((48, 48, 48), (1.0, 1.0, 1.0))
    fields = [Field("displacement", "node", "x,y,z", "mm", "state"),
              Field("stress", "cell", "xx,yy,zz,yz,xz,xy", "MPa", "derived")]
    est = estimate_size(grid, fields, 150)
    assert est["primary"] == 49 ** 3 * 3 * 8 * 150 and est["derived"] == 48 ** 3 * 6 * 8 * 150
    w = ResultWriter(tmp_path / "result.vtkhdf", grid, fields, dim=3, analysis="3d", run_id="big",
                     expected_steps=150, size_limit_gb=0.1, sidecars=False)
    assert w.moved_out_of_view == ["stress"]
    w.close()
    with h5py.File(tmp_path / "result.vtkhdf", "r") as f:
        assert "stress" not in f["VTKHDF/CellData"] and "derived/stress/values" in f
        assert f["meta"].attrs["derived_outside_view"] == "stress"


_KILL = """
import os, sys, numpy as np
from phast.io.result import Field, ResultWriter, UnstructuredMesh
mesh = UnstructuredMesh(np.array([[0, 0], [1, 0], [0, 1.0]]), {"T3": [[0, 1, 2]]})
w = ResultWriter(sys.argv[1], mesh, [Field("damage", "node", "d", "1", "state")], dim=2, analysis="plane_strain",
                 run_id="kill", history_units={"load_parameter": "mm"}, sidecars=False)
for s in range(3):
    w.append(dict(increment=s, time=s + 1.0, dt=1.0, load_parameter=0.0, damage_solved=1), fields=dict(damage=np.full(3, s + 0.5)))
os._exit(0)
"""


def test_kill_after_flush_keeps_complete_steps(tmp_path):
    subprocess.run([sys.executable, "-c", _KILL, str(tmp_path / "result.vtkhdf")], check=True,
                   env={**os.environ, "PYTHONPATH": _SRC})
    partial = tmp_path / "result.vtkhdf.partial"
    assert partial.exists() and not (tmp_path / "result.vtkhdf").exists()
    with h5py.File(partial, "r") as f:
        assert f.attrs["status"] == "running" and int(f["VTKHDF/Steps"].attrs["NSteps"]) == 3
        assert f["VTKHDF/PointData/damage"][-3:].tolist() == [2.5, 2.5, 2.5]
        assert f["history/step_index"][...].tolist() == [0, 1, 2]


def test_setup_profile_grows_nothing(tmp_path):
    mesh = UnstructuredMesh(np.eye(3)[:, :2], {"T3": [[0, 1, 2]]})
    w = ResultWriter(tmp_path / "setup.vtkhdf", mesh, [Field("phase_id", "cell", "id", "1", "state", static=True, data=[1])],
                     dim=2, analysis="plane_strain", run_id="s", profile="setup", sidecars=False)
    with pytest.raises(RuntimeError):
        w.append(dict(increment=0, time=1.0, dt=1.0, load_parameter=0.0, damage_solved=1))
    w.write_setup()
    with h5py.File(tmp_path / "setup.vtkhdf", "r") as f:
        assert int(f["VTKHDF/Steps"].attrs["NSteps"]) == 0 and f["VTKHDF/CellData/phase_id"].dtype == np.uint8


def test_finalize_archive_keeps_values_and_adds_filters(tmp_path):
    from phast.io.result import finalize
    out = replay(FIXTURES["voxel_h8"]()[0], tmp_path / "result.vtkhdf")
    write_fixture(tmp_path / "ref.vtkhdf", FIXTURES["voxel_h8"]()[0])
    finalize(out)
    assert compare(tmp_path / "ref.vtkhdf", out) == []
    with h5py.File(out, "r") as f:
        h = f["VTKHDF/CellData/H"]
        assert h.compression == "gzip" and h.shuffle and h.fletcher32
        assert isinstance(f.get("state/H", getlink=True), h5py.SoftLink)
    assert (tmp_path / "result.vtkhdf.sha256").read_text().split()[0] == __import__("phast.io.result", fromlist=["x"]).sha256_file(out)


@pytest.mark.skipif(not hasattr(__import__("signal"), "SIGUSR1"), reason="needs SIGUSR1")
def test_soft_stop_only_sets_a_flag():
    import os
    import signal
    from phast.io.result import SoftStop
    stop = SoftStop(signals=(signal.SIGUSR1,))
    try:
        os.kill(os.getpid(), signal.SIGUSR1)
        assert stop.requested and stop.signum == signal.SIGUSR1
    finally:
        stop.restore()


def test_imagedata_setup_has_untimed_static_arrays_and_one_step_warns(tmp_path):
    from phast import load_result
    from phast.io.result import validate
    grid = ImageGrid((3, 2, 2), (1.0, 1.0, 1.0))
    phase = np.arange(12) % 2
    w = ResultWriter(tmp_path / "setup" / "result.vtkhdf", grid,
                     [Field("phase_id", "cell", "id", "1", "state", static=True, data=phase)],
                     dim=3, analysis="3d", run_id="vs", profile="setup", sidecars=False)
    out = w.write_setup()
    with h5py.File(out, "r") as f:
        assert f["VTKHDF/CellData/phase_id"].shape == (2, 2, 3)
    assert [x.code for x in validate(out).findings if x.level != "INFO"] == []
    assert load_result(out).field("phase_id").ravel().tolist() == phase.tolist()

    fields = [Field("phase_id", "cell", "id", "1", "state", static=True, data=phase), Field("damage", "node", "d", "1", "state")]
    w = ResultWriter(tmp_path / "one" / "result.vtkhdf", grid, fields, dim=3, analysis="3d", run_id="v1",
                     history_units={"load_parameter": "1"}, sidecars=False)
    w.append(dict(increment=0, time=1.0, dt=1.0, load_parameter=1e-4, damage_solved=1), fields=dict(damage=np.zeros(36)))
    out = w.close()
    assert [x.code for x in validate(out).findings if x.level == "WARN"] == ["imagedata-single-step"]
    assert load_result(out).field("damage").shape == (3, 3, 4)


def test_synchronised_folder_detection():
    from phast.io.result.writer import _is_synced
    from pathlib import Path
    assert _is_synced(Path("/home/a/CloudStorage/Sync-Org/run/result.vtkhdf"))
    assert _is_synced(Path("/home/a/OneDrive/run/result.vtkhdf"))
    assert _is_synced(Path("/home/a/Dropbox/run/result.vtkhdf"))
    assert not _is_synced(Path("/tmp/home-a-OneDrive-Org-x/scratch/result.vtkhdf"))


def test_cli_finalize_archives_and_validates(tmp_path):
    from phast.io.result.cli import main
    out = replay(FIXTURES["t3_notched_plate"]()[0], tmp_path / "result.vtkhdf")
    assert main(["finalize", str(out)]) == 1
    assert main(["finalize", "--archive", str(out)]) == 0
    with h5py.File(out, "r") as f:
        assert f["VTKHDF/PointData/displacement"].compression == "gzip"
    assert main(["validate", str(out)]) == 0
