"""Layout 2.1 core: version gating, crack-tip and substep tables, velocity conditions, initial
conditions, wave speeds, validation results and the one-dimensional exclusion.

Rules every test leans on: a file is 2.0.0 until the first item added in 2.1 is used, the validator
checks a file with the schema of its own version, and a 2.0.0 file never gets a new finding.
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
import yaml

from phast import load_result
from phast.io.result import Field, ResultWriter, UnstructuredMesh, recover, validate
from phast.io.result.crack_tips import FIT_POLYORDER, FIT_WINDOW, fit_speed
from phast.io.result.writer import isotropic_wave_speeds

def warnings_ignored():
    """A file that fails validation on close warns and keeps its .partial name: the tests expect that."""
    import warnings
    import contextlib

    @contextlib.contextmanager
    def ctx():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            yield
    return ctx()


ROOT = Path(__file__).resolve().parents[1]
_SRC = str(ROOT / "src")
PTS = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
CELLS = [[0, 1, 2], [0, 2, 3]]


def mesh():
    return UnstructuredMesh(PTS, {"T3": CELLS}, node_sets={"top": [2, 3], "bottom": [0, 1]})


def writer(path, **kw):
    kw.setdefault("sidecars", False)
    kw.setdefault("history_units", {"load_parameter": "mm"})
    return ResultWriter(path, kw.pop("mesh", mesh()), kw.pop("fields", [Field("damage", "node", "d", "1", "state")]),
                        dim=kw.pop("dim", 2), analysis="plane_strain", run_id="t21", **kw)


def row(i, dt=0.1):
    return dict(increment=i, time=dt * (i + 1), dt=dt, load_parameter=0.01 * i, damage_solved=1)


def errors(path, **kw):
    return [f"{x.code}: {x.message}" for x in validate(path, **kw).findings if x.level == "ERROR"]


def run_with_tips(path, n=8, **kw):
    w = writer(path, **kw)
    for i in range(n):
        w.append(row(i), fields=dict(damage=np.full(4, 0.1 * i)))
        w.append_crack_tip(dict(increment=i, tip_id=0, position=(10.0 + 0.5 * i + 0.1 * (-1) ** i, 2.0), threshold=0.5),
                           method="d = 0.5 isoline, farthest in +x")
    return w


# --- version gating ------------------------------------------------------------------------

def test_a_plain_file_is_2_0_0_and_embeds_the_2_0_schema(tmp_path):
    with writer(tmp_path / "result.vtkhdf") as w:
        w.append(row(0), fields=dict(damage=np.zeros(4)))
    with h5py.File(tmp_path / "result.vtkhdf", "r") as f:
        assert f.attrs["phast_schema_version"] == "2.0.0"
        assert yaml.safe_load(f["meta/schema_yaml"][()].decode())["version"] == "2.0.0"
        assert "layout_2_1_features" not in f["meta"].attrs
    assert errors(tmp_path / "result.vtkhdf") == []


def test_first_2_1_item_moves_the_file_to_2_1_0_before_it_is_written(tmp_path):
    w = writer(tmp_path / "result.vtkhdf")
    w.append(row(0), fields=dict(damage=np.zeros(4)))
    assert w.f.attrs["phast_schema_version"] == "2.0.0"
    w.append_crack_tip(dict(increment=0, tip_id=0, position=(1.0, 2.0), threshold=0.5))
    assert w.f.attrs["phast_schema_version"] == "2.1.0"
    assert yaml.safe_load(w.f["meta/schema_yaml"][()].decode())["version"] == "2.1.0"
    assert w.f["meta"].attrs["layout_2_1_features"] == "crack_tips"
    w.close()
    assert errors(tmp_path / "result.vtkhdf") == []
    assert load_result(tmp_path / "result.vtkhdf").layouts() == ["2.1"]


_KILL_AFTER_TABLE = """
import os, sys, numpy as np
from phast.io.result import Field, ResultWriter, UnstructuredMesh
mesh = UnstructuredMesh(np.array([[0, 0], [1, 0], [1, 1.0], [0, 1]]), {"T3": [[0, 1, 2], [0, 2, 3]]})
w = ResultWriter(sys.argv[1], mesh, [Field("damage", "node", "d", "1", "state")], dim=2, analysis="plane_strain",
                 run_id="kill", history_units={"load_parameter": "mm"}, sidecars=False)
for s in range(4):
    w.append(dict(increment=s, time=0.1 * (s + 1), dt=0.1, load_parameter=0.0, damage_solved=1), fields=dict(damage=np.full(4, s + 0.5)))
    w.append_crack_tip(dict(increment=s, tip_id=0, position=(1.0 + s, 0.5), threshold=0.5))
    w.append_substep(dict(increment=s, call=0, cg_iterations=7 + s, cg_converged=True, effective_preconditioner="amg"))
w.f.flush()
os._exit(0)
"""


def test_a_killed_2_1_file_is_checked_by_its_own_rules_and_recovers(tmp_path):
    subprocess.run([sys.executable, "-c", _KILL_AFTER_TABLE, str(tmp_path / "result.vtkhdf")], check=True,
                   env={**os.environ, "PYTHONPATH": _SRC})
    partial = tmp_path / "result.vtkhdf.partial"
    assert errors(partial, allow_running=True) == []              # tables and version agree: valid for 2.1
    with h5py.File(partial, "r") as f:
        assert f.attrs["phast_schema_version"] == "2.1.0" and "speed" not in f["history/crack_tips"]
    # the same content judged by the 2.0 rules is what the early version bump prevents
    legacy = tmp_path / "legacy.vtkhdf"
    legacy.write_bytes(partial.read_bytes())
    with h5py.File(legacy, "r+") as f:
        f.attrs["phast_schema_version"] = "2.0.0"
    assert any(e.startswith("dtype:") for e in errors(legacy, allow_running=True))
    # recovery trims the long tables with the history rows
    with h5py.File(partial, "r+") as f:
        n = f["history/increment"].shape[0]
        for name in list(f["history"]):
            if isinstance(f["history"][name], h5py.Dataset):
                f["history"][name].resize(n - 1, axis=0)         # as if the last history row never arrived
    result = recover(partial, tmp_path / "recovered.vtkhdf", use_h5clear=False)
    assert result.report.ok, result.report.format_text()
    with h5py.File(result.output, "r") as f:
        assert f["history/crack_tips/increment"][...].tolist() == [0, 1, 2]
        assert f["history/substeps/increment"][...].tolist() == [0, 1, 2]


# --- crack tips --------------------------------------------------------------------------------

def test_crack_tip_table_and_speed(tmp_path):
    w = run_with_tips(tmp_path / "result.vtkhdf")
    w.close()
    path = tmp_path / "result.vtkhdf"
    assert errors(path) == []
    with h5py.File(path, "r") as f:
        g = f["history/crack_tips"]
        assert g["increment"][...].tolist() == list(range(8)) and g["tip_id"].dtype == np.int64
        pos = g["position"]
        assert pos.shape == (8, 3) and pos.dtype == np.float64 and pos.attrs["units"] == "mm"
        assert pos.attrs["components"] == "x,y,z" and "isoline" in pos.attrs["method"]
        assert not pos[:, 2].any() and np.allclose(g["threshold"][...], 0.5)
        sp = g["speed"]
        assert sp.attrs["fit_window"] == FIT_WINDOW == 5 and sp.attrs["fit_polyorder"] == FIT_POLYORDER == 2
        assert sp.attrs["units"] == "mm/s" and "position" in sp.attrs["derived_from"]
        time = f["history/time"][...]
        expect = fit_speed(np.zeros(8, int), pos[...], time)
        assert np.isnan(sp[0]) and np.array_equal(sp[1:], expect[1:])
    ct = load_result(path).crack_tips()
    assert ct["speed_source"] == "file" and ct["position"].shape == (8, 2)
    assert np.allclose(ct["time"], 0.1 * np.arange(1, 9))
    # the smoothed speed of a tip moving 5 mm/s with +-0.1 mm jitter stays near the truth (the raw one does not)
    assert np.all(np.abs(ct["speed"][1:] - 5.0) < 1.5)
    raw = np.abs(np.diff(ct["position"][:, 0])) / 0.1
    assert raw.max() - raw.min() > 2.0 * (ct["speed"][1:].max() - ct["speed"][1:].min())
    rows = load_result(path).history("crack_tips")
    assert len(rows) == 8 and {"position_x", "position_y", "position_z", "threshold"} <= set(rows[0])


def test_crack_speed_is_the_backward_difference_of_the_smoothed_position():
    """Reference: scipy's Savitzky-Golay filter on each coordinate, then the length of the backward difference."""
    from scipy.signal import savgol_filter

    rng = np.random.default_rng(0)
    n = 30
    time = 1.0e-6 * np.arange(1, n + 1)
    position = np.stack([50.0 + np.arange(1, n + 1) + rng.uniform(-0.5, 0.5, n), rng.uniform(-0.1, 0.1, n),
                         np.zeros(n)], axis=1)
    smooth = np.stack([savgol_filter(position[:, c], window_length=5, polyorder=2, mode="interp") for c in range(3)],
                      axis=1)
    expected = np.r_[np.nan, np.linalg.norm(np.diff(smooth, axis=0), axis=1) / np.diff(time)]
    mine = fit_speed(np.zeros(n, int), position, time)
    assert np.isnan(mine[0]) and np.allclose(mine[1:], expected[1:], rtol=1e-12)
    two = fit_speed(np.r_[np.zeros(n, int), np.ones(n, int)], np.vstack([position, position + 3.0]), np.r_[time, time])
    assert np.allclose(two[1:n], expected[1:]) and np.allclose(two[n + 1:], expected[1:])   # tips are followed separately


def test_crack_speed_is_computed_on_read_when_the_file_has_none(tmp_path):   # a run killed before close
    w = run_with_tips(tmp_path / "result.vtkhdf")
    w.close()
    with h5py.File(tmp_path / "result.vtkhdf", "r+") as f:
        del f["history/crack_tips/speed"]
    ct = load_result(tmp_path / "result.vtkhdf").crack_tips()
    assert ct["speed_source"] == "computed" and ct["speed_fit"]["fit_window"] == 5 and np.isnan(ct["speed"][0])


def test_two_tips_and_a_branching_increment(tmp_path):
    w = writer(tmp_path / "result.vtkhdf")
    for i in range(6):
        w.append(row(i), fields=dict(damage=np.zeros(4)))
        w.append_crack_tip(dict(increment=i, tip_id=0, position=(1.0 + i, 0.0, 0.0), threshold=0.5))
        if i >= 3:
            w.append_crack_tip(dict(increment=i, tip_id=1, position=(3.0 + 0.5 * (i - 3), 1.0, 0.0), threshold=0.5))
    w.close()
    assert errors(tmp_path / "result.vtkhdf") == []
    ct = load_result(tmp_path / "result.vtkhdf").crack_tips()
    assert ct["tip_id"].tolist() == [0, 0, 0, 0, 1, 0, 1, 0, 1]
    assert np.isnan(ct["speed"][[0, 4]]).all()                          # the first row of each tip has no speed


def test_crack_tip_validator_rules(tmp_path):
    w = run_with_tips(tmp_path / "result.vtkhdf")
    w.close()
    base = tmp_path / "result.vtkhdf"

    def mutate(name, fn):
        out = tmp_path / f"{name}.vtkhdf"
        out.write_bytes(base.read_bytes())
        with h5py.File(out, "r+") as f:
            fn(f)
        return errors(out)

    assert any("table-increment" in e for e in mutate("inc", lambda f: f["history/crack_tips/increment"].__setitem__(3, 99)))
    assert any("table-value" in e for e in mutate("thr", lambda f: f["history/crack_tips/threshold"].__setitem__(1, 1.5)))
    assert any("table-value" in e for e in mutate("tip", lambda f: f["history/crack_tips/tip_id"].__setitem__(1, -1)))
    assert any("derived-fit" in e for e in mutate("fit", lambda f: f["history/crack_tips/speed"].attrs.__delitem__("fit_window")))
    assert any("history-length" in e for e in mutate("len", lambda f: f["history/crack_tips/threshold"].resize(7, axis=0)))
    assert any("table-column-missing" in e for e in mutate("col", lambda f: f["history/crack_tips"].__delitem__("threshold")))
    assert any("table-order" in e for e in mutate("ord", lambda f: f["history/crack_tips/increment"].__setitem__(slice(0, 2), [5, 4])))


# --- substeps -------------------------------------------------------------------------------------

def _solve(i, call, **extra):
    base = dict(increment=i, call=call, requested_preconditioner="amg", effective_preconditioner="amg",
                amg_eligible=True, amg_skip_reason="not evaluated", damage_increment=0.01 * i, rebuild_attempted=False,
                rebuild_succeeded=False, fallback_target="", fallback_reason="", cooldown_remaining=0, retry_counter=0,
                cg_iterations=10 + call, cg_residual=1e-9, cg_converged=True, damage_min=0.0, damage_max=0.2 * i,
                min_increment=0.0, bounds_satisfied=True, irreversibility_satisfied=True, walltime_s=0.01)
    base.update(extra)
    return base


def test_substep_table_encodes_text_and_flags(tmp_path):
    w = writer(tmp_path / "result.vtkhdf")
    for i in range(3):
        w.append(row(i), fields=dict(damage=np.zeros(4)))
        w.append_substep(_solve(i, 0))
        if i == 2:                                   # a second solve with a fallback, new text values
            w.append_substep(_solve(i, 1, effective_preconditioner="jacobi", fallback_target="jacobi",
                                    fallback_reason="amg setup failed", cg_converged=False, amg_eligible=False))
    w.close()
    path = tmp_path / "result.vtkhdf"
    assert errors(path) == []
    with h5py.File(path, "r") as f:
        g = f["history/substeps"]
        assert g["increment"][...].tolist() == [0, 1, 2, 2] and g["call"][...].tolist() == [0, 0, 0, 1]
        assert g["cg_converged"].dtype == np.int8 and g["cg_converged"][...].tolist() == [1, 1, 1, 0]
        assert g["cg_iterations"].dtype == np.int64 and g["walltime_s"].attrs["units"] == "s"
        codes = json.loads(g["effective_preconditioner"].attrs["codes"])
        assert codes == {"0": "amg", "1": "jacobi"} and g["effective_preconditioner"][...].tolist() == [0, 0, 0, 1]
        assert g["effective_preconditioner"].dtype == np.int8
        assert g["cg_residual"].dtype == np.float64
    rows = load_result(path).history("substeps")
    assert [r["effective_preconditioner"] for r in rows] == ["amg", "amg", "amg", "jacobi"]
    assert rows[3]["fallback_reason"] == "amg setup failed" and rows[3]["cg_converged"] == 0
    # the main history table does not absorb the long tables
    assert len(load_result(path).history("history")) == 3
    assert {"crack_tips", "substeps"} & set(load_result(path).history_names()) == {"substeps"}


def test_substep_table_rejects_a_row_without_its_key(tmp_path):
    w = writer(tmp_path / "result.vtkhdf")
    w.append(row(0), fields=dict(damage=np.zeros(4)))
    with pytest.raises(ValueError, match="lacks"):
        w.append_substep(dict(increment=0, cg_iterations=3))
    w.close()


# --- velocity conditions, wave speeds -------------------------------------------------------------

def test_velocity_condition_makes_a_2_1_file_and_is_checked(tmp_path):
    bcs = {"00_left": dict(set="left", kind="velocity", components="x", value=16500.0, value_units="mm/s",
                           ramp_type="velocity_impact", t_ramp=1e-6)}
    with writer(tmp_path / "result.vtkhdf", bcs=bcs) as w:
        w.append(row(0), fields=dict(damage=np.zeros(4)))
    path = tmp_path / "result.vtkhdf"
    with h5py.File(path, "r") as f:
        assert f.attrs["phast_schema_version"] == "2.1.0" and f["meta"].attrs["layout_2_1_features"] == "velocity_bc"
    assert errors(path) == []
    assert load_result(path).boundary_conditions()["bc"]["00_left"]["kind"] == "velocity"
    with h5py.File(path, "r+") as f:
        f["loading/bc/00_left"].attrs["kind"] = "velcoity"
    assert any("bc-kind" in e for e in errors(path))
    with h5py.File(path, "r+") as f:                           # a 2.0 file keeps its 2.0 rules: no bc-kind finding
        f.attrs["phast_schema_version"] = "2.0.0"
    assert not any("bc-kind" in e for e in errors(path))


def test_wave_speeds_follow_from_the_elastic_constants(tmp_path):
    E, nu, rho = 190000.0, 0.3, 8.0e-9
    c = isotropic_wave_speeds(E, nu, rho, "plane_strain")
    mu = E / 2.6
    lam = E * nu / (1.3 * 0.4)
    assert c["c_s"] == pytest.approx((mu / rho) ** 0.5, rel=1e-14)
    assert c["c_p"] == pytest.approx(((lam + 2 * mu) / rho) ** 0.5, rel=1e-14)
    assert c["c_R"] == pytest.approx(c["c_s"] * (0.862 + 1.14 * nu) / 1.3, rel=1e-14)
    ps = isotropic_wave_speeds(E, nu, rho, "plane_stress")
    assert ps["c_p"] == pytest.approx((E / (1 - nu ** 2) / rho) ** 0.5, rel=1e-14) and ps["c_s"] == c["c_s"]

    def make(name, **over):
        phase = dict(name="steel", elastic_model="isotropic", E=E, nu=nu, rho=rho, **c,
                     units={"E": "MPa", "rho": "t/mm^3", "c_p": "mm/s", "c_s": "mm/s", "c_R": "mm/s"})
        phase.update(over)
        with warnings_ignored():
            with writer(tmp_path / f"{name}.vtkhdf", materials={0: phase}) as w:
                w.append(row(0), fields=dict(damage=np.zeros(4)))
        final = tmp_path / f"{name}.vtkhdf"
        return final if final.exists() else final.with_name(final.name + ".partial")   # an invalid file keeps .partial

    ok = make("ok")
    assert errors(ok) == []
    assert load_result(ok).materials()["phases"][0]["c_p"] == c["c_p"]
    assert any("wave-speed:" in e and "c_p" in e for e in errors(make("bad", c_p=c["c_p"] * 1.01)))
    assert any("wave-speed-units" in e for e in errors(make("units", units={"E": "MPa", "rho": "t/mm^3", "c_p": "m/s",
                                                                              "c_s": "mm/s", "c_R": "mm/s"})))
    nr = dict(name="steel", elastic_model="isotropic", E=E, nu=nu, c_p=c["c_p"], units={"c_p": "mm/s", "rho": "t/mm^3"})
    with warnings_ignored(), writer(tmp_path / "norho.vtkhdf", materials={0: nr}) as w:
        w.append(row(0), fields=dict(damage=np.zeros(4)))
    assert any("wave-speed:" in e and "density" in e for e in errors(tmp_path / "norho.vtkhdf.partial"))


# --- initial conditions ------------------------------------------------------------------------

def test_initial_conditions_group(tmp_path):
    fields = [Field("displacement", "node", "x,y", "mm", "state"), Field("damage", "node", "d", "1", "state"),
              Field("H", "cell", "H", "MPa", "history")]
    w = writer(tmp_path / "result.vtkhdf", fields=fields)
    d0 = np.array([1.0, 0.0, 0.0, 1.0])
    w.write_initial("damage", d0, specification=dict(preseed_damage=[dict(nodes="notch", value=1.0)]))
    w.write_initial("displacement", np.zeros((4, 2)))
    w.write_initial("H", np.array([5.0, 0.0]))
    with pytest.raises(ValueError, match="expected shape"):
        w.write_initial("damage", np.zeros(5))
    w.append(row(0), fields=dict(displacement=np.zeros((4, 2)), damage=d0, H=np.array([5.0, 0.0])))
    with pytest.raises(RuntimeError, match="before the first increment"):
        w.write_initial("damage", d0)
    w.close()
    path = tmp_path / "result.vtkhdf"
    assert errors(path) == []
    with h5py.File(path, "r") as f:
        ic = f["initial_conditions"]
        assert sorted(ic) == ["H", "damage", "displacement"] and f.attrs["phast_schema_version"] == "2.1.0"
        assert ic["damage"].dtype == np.float64 and ic["damage"].attrs["kind"] == "state"
        assert json.loads(ic["damage"].attrs["specification"])["preseed_damage"][0]["nodes"] == "notch"
        assert ic["displacement"].shape == (4, 3) and ic["displacement"].attrs["components"] == "x,y,z"
        assert ic["H"].attrs["kind"] == "history" and f["meta"].attrs["layout_2_1_features"] == "initial_conditions"
    ic = load_result(path).initial_conditions()
    assert np.array_equal(ic["damage"], d0) and ic["displacement"].shape == (4, 2)
    with h5py.File(path, "r+") as f:
        data = f["initial_conditions/damage"][...]
        del f["initial_conditions/damage"]
        f["initial_conditions/damage"] = data[:3]
        for k, v in dict(location="node", components="d", units="1", kind="state").items():
            f["initial_conditions/damage"].attrs[k] = v
    assert any("initial-shape" in e for e in errors(path))


# --- validation results -----------------------------------------------------------------------------

ROWS = [dict(name="peak reaction error", value=0.0108, tolerance=0.05, passed=True, units="1",
             description="relative error of the peak reaction against the reference"),
        dict(name="pre-peak L2 error", value=0.017, tolerance=0.10, passed=True, units="1"),
        dict(name="mechanics converged at every increment", value=float("nan"), tolerance=float("nan"), passed=True)]


def test_validation_rows(tmp_path):
    w = writer(tmp_path / "result.vtkhdf")
    w.append(row(0), fields=dict(damage=np.zeros(4)))
    w.write_validation(ROWS, source="compare_report.txt")
    with pytest.raises(RuntimeError, match="once"):
        w.write_validation(ROWS)
    w.close()
    path = tmp_path / "result.vtkhdf"
    assert errors(path) == []
    with h5py.File(path, "r") as f:
        g = f["meta/validation"]
        assert g["passed"].dtype == np.int8 and g["value"].dtype == np.float64
        assert g.attrs["overall_passed"] == 1 and g.attrs["source"] == "compare_report.txt"
        assert f.attrs["phast_schema_version"] == "2.1.0"
    got = load_result(path).validation()
    assert [r["name"] for r in got] == [r["name"] for r in ROWS] and got[0]["tolerance"] == 0.05
    assert got[0]["passed"] == 1 and np.isnan(got[2]["value"]) and got[1]["units"] == "1"
    with h5py.File(path, "r+") as f:
        f["meta/validation/passed"][1] = 0
    assert any("overall_passed" in e for e in errors(path))
    with h5py.File(path, "r+") as f:
        f["meta/validation/passed"][1] = 3
    assert any("validation-value" in e for e in errors(path))


def test_validation_row_needs_its_four_values(tmp_path):
    w = writer(tmp_path / "result.vtkhdf")
    with pytest.raises(ValueError, match="lacks"):
        w.write_validation([dict(name="x", value=1.0, passed=True)])
    w.close()


# --- one-dimensional runs ---------------------------------------------------------------------------

def test_one_dimensional_runs_are_refused_with_a_clear_message(tmp_path):
    with pytest.raises(ValueError, match="one-dimensional runs .* outside layout 2.x"):
        writer(tmp_path / "a.vtkhdf", dim=1)
    with pytest.raises(ValueError, match="one-dimensional runs .* outside layout 2.x"):
        UnstructuredMesh(np.linspace(0, 1, 5)[:, None], {"T3": CELLS})
    with pytest.raises(ValueError, match="one-dimensional runs .* outside layout 2.x"):
        UnstructuredMesh(np.linspace(0, 1, 5), {"T3": CELLS})
    with pytest.raises(ValueError, match="line cells"):
        UnstructuredMesh(PTS, {"LINE2": [[0, 1]]})
    with pytest.raises(ValueError, match="dim must be 2 or 3"):
        writer(tmp_path / "b.vtkhdf", dim=4)
    # a file that says dim = 1 is refused by the validator of 2.1 with the same message, and by 2.0 as before
    with writer(tmp_path / "ok.vtkhdf", bcs={"k": dict(set="top", kind="velocity", components="x", value=1.0)}) as w:
        w.append(row(0), fields=dict(damage=np.zeros(4)))
    path = tmp_path / "ok.vtkhdf"
    with h5py.File(path, "r+") as f:
        f.attrs["dim"] = 1
    assert any(e.startswith("dimension-unsupported:") and "line cells" in e for e in errors(path))
    with h5py.File(path, "r+") as f:
        f.attrs["phast_schema_version"] = "2.0.0"
    assert any(e.startswith("invalid-attribute:") and "dim" in e for e in errors(path))


# --- the 2.0 rules of a 2.0 file do not move ---------------------------------------------------------------

def test_a_2_0_file_is_not_judged_by_the_2_1_schema(tmp_path):
    with writer(tmp_path / "result.vtkhdf") as w:
        w.append(row(0), fields=dict(damage=np.zeros(4)))
    findings = validate(tmp_path / "result.vtkhdf").findings
    assert {f.path for f in findings if f.code == "optional-absent"} == {"/interfaces", "/quadrature", "/derived"}
    # mesh_varies = true stays an error in a 2.0.0 file
    with h5py.File(tmp_path / "result.vtkhdf", "r+") as f:
        f.attrs["mesh_varies"] = True
    assert any(e.startswith("mesh-varies:") for e in errors(tmp_path / "result.vtkhdf"))
    # a 2.1 file lists the optional groups of 2.1
    with writer(tmp_path / "b.vtkhdf") as w:
        w.append(row(0), fields=dict(damage=np.zeros(4)))
        w.write_validation(ROWS[:1])
    absent = {f.path for f in validate(tmp_path / "b.vtkhdf").findings if f.code == "optional-absent"}
    assert "/initial_conditions" in absent and "/meta/validation" not in absent


# --- figures ----------------------------------------------------------------------------------------------

def test_figures_draw_a_2_1_file_as_a_layout_2_file(tmp_path):
    from phast.io.result.figures.cli import draw_all
    w = run_with_tips(tmp_path / "result.vtkhdf")
    w.close()
    rows = draw_all(tmp_path / "result.vtkhdf", tmp_path / "figs", dpi=50)
    kinds = {r["kind"] for r in rows}
    assert {"setup", "fields", "evolution"} <= kinds
    from phast.io.result.figures.data import RunData
    assert RunData(load_result(tmp_path / "result.vtkhdf")).layout_label == "layout 2.1"


def test_figures_refuse_a_changing_mesh_clearly(tmp_path):
    from phast.io.result.figures.cli import draw_all
    from result_v2.fixtures_21 import build_all
    build_all(tmp_path)
    with pytest.raises(ValueError, match="mesh changes during the run"):
        draw_all(tmp_path / "refined_t3.vtkhdf", tmp_path / "figs", dpi=50)


# --- archive copy and schema ----------------------------------------------------------------------------------

def test_archive_copy_of_a_2_1_file_with_text_columns_is_exact(tmp_path):
    from phast.io.result.finalize import finalize
    w = run_with_tips(tmp_path / "result.vtkhdf")
    w.append_substep(_solve(7, 0))
    w.write_validation(ROWS)
    w.close()
    path = tmp_path / "result.vtkhdf"
    finalize(path)
    assert errors(path) == []
    with h5py.File(path, "r") as f:
        assert f["history/crack_tips/position"].compression == "gzip"
        assert [x.decode() for x in f["meta/validation/name"][...]] == [r["name"] for r in ROWS]


def test_schema_21_is_the_2_0_schema_plus_items():
    from phast.io.result import load_schema
    old, new = load_schema("2.0.0"), load_schema("2.1.0")
    assert new["version"] == "2.1.0" and old["version"] == "2.0.0"
    for key, value in old.items():
        if key in ("version", "root_attributes", "groups", "history_columns", "validator"):
            continue
        assert new[key] == value, key
    assert new["root_attributes"]["required"]["mesh_varies"] == {"values": [False, True]}
    for section in ("required", "optional"):
        assert {k: v for k, v in old["root_attributes"][section].items() if k != "mesh_varies"} == \
            {k: v for k, v in new["root_attributes"][section].items() if k != "mesh_varies"}
    for group in ("required", "meta_required", "meta_attributes"):
        assert new["groups"][group] == old["groups"][group]
    assert set(old["groups"]["optional"]) < set(new["groups"]["optional"])
    assert set(old["validator"]["integer_history_columns"]) < set(new["validator"]["integer_history_columns"])
    assert {"crack_tips", "substeps"} == set(new["history_tables"])


def test_schema_version_for_maps_file_versions():
    from phast.io.result.schema import schema_version_for
    assert [schema_version_for(v) for v in ("2.0.0", "2.0.7", "2.1.0", "2.1.3", "2.9.0")] == \
        ["2.0.0", "2.0.0", "2.1.0", "2.1.0", "2.1.0"]
    for bad in ("3.0.0", "2.1", "x", ""):
        with pytest.raises(ValueError):
            schema_version_for(bad)
