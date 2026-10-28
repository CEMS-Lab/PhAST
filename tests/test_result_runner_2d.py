"""The 2D YAML runner writes a layout-2.0 result file.

Each run is a few increments on a coarse mesh (CPU, float64), driven through
``python -m phast run`` exactly as users run it (``result_v2/runner_helpers.py``).
The checks compare the stored arrays bit for bit with the solver's final state,
so they test float64 storage, the Q4 point order and the time axis.
"""
from __future__ import annotations

import csv
import json

import h5py
import numpy as np
import pytest

from phast.config.result_output import resolve_result_settings
from result_v2 import runner_helpers as R

pytestmark = [pytest.mark.timeout(180)]


def _common_checks(f: h5py.File, out, n_inc: int) -> dict:
    h = f["history"]
    assert f.attrs["status"] == "complete" and f.attrs["phast_schema_version"] == "2.0.0"
    assert h["increment"][...].tolist() == list(range(n_inc))
    step_index = h["step_index"][...]
    n_steps = int(f["VTKHDF/Steps"].attrs["NSteps"])
    assert n_steps == int((step_index >= 0).sum()) and step_index[-1] == n_steps - 1
    assert np.array_equal(h["time"][...], np.cumsum(h["dt"][...]))
    assert np.array_equal(f["VTKHDF/Steps/Values"][...], h["time"][...][step_index >= 0])
    for name in ("displacement", "damage", "H"):
        link = f.get(f"state/{name}", getlink=True)
        assert isinstance(link, h5py.SoftLink)
        assert f[f"state/{name}"].dtype == np.float64
    n_pts = int(f["VTKHDF/NumberOfPoints"][0])
    n_cells = int(f["VTKHDF/NumberOfCells"][0])
    final = np.load(out / "final.npz")
    u = f["VTKHDF/PointData/displacement"][-n_pts:]
    assert np.array_equal(u[:, :2], final["displacement"]) and not u[:, 2].any()
    assert np.array_equal(f["VTKHDF/PointData/damage"][-n_pts:], final["damage"])
    assert np.array_equal(f["VTKHDF/CellData/H"][-n_cells:], final["history_element"])
    cfg = f["meta/config_resolved_yaml"][()].decode()
    assert "result:" in cfg and str(out) not in cfg and not any(
        line.split(":", 1)[-1].strip().startswith("/") for line in cfg.splitlines())
    assert (out / "result.vtkhdf.sha256").exists() and not (out / "result.vtkhdf.partial").exists()
    return dict(n_pts=n_pts, n_cells=n_cells, final=final)


def _signed_reactions(f: h5py.File) -> None:
    top, bottom = f["history/reaction/top"][...], f["history/reaction/bottom"][...]
    assert top.shape[1] == 3 and f["history/reaction/top"].attrs["units"] == "N"
    assert (top[:, 1] > 0).all() and (bottom[:, 1] < 0).all()
    assert np.allclose(top[:, 1] + bottom[:, 1], 0.0, atol=1e-4 * np.abs(top[:, 1]).max())  # equilibrium to the solver tolerance


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """One run per configuration, shared by the content checks and the validator checks."""
    cache = {}

    def get(kind):
        if kind not in cache:
            tmp = tmp_path_factory.mktemp(kind)
            cfg = {"t3": lambda: R.t3_qs(tmp, h5=True, h5_every=2),
                   "q4": lambda: R.q4_qs(tmp, result=dict(derived_location="quadrature_point")),
                   "dynamic": lambda: R.dynamic(tmp),
                   "traction": lambda: R.traction(tmp)}[kind]()
            cache[kind] = R.run(tmp, cfg)[1]
        return cache[kind]
    return get


@pytest.mark.parametrize("kind", ["t3", "q4", "dynamic", "traction"])
def test_result_file_validates(runs, kind):
    R.validate_file(runs(kind) / "result.vtkhdf")


def test_t3_quasi_static_run_writes_result(runs):
    out = runs("t3")
    path = out / "result.vtkhdf"
    with h5py.File(path, "r") as f:
        _common_checks(f, out, 4)
        assert int(f["VTKHDF/Steps"].attrs["NSteps"]) == 4
        assert f["history/time"].attrs["units"] == "1" and f["history/dt"][...].tolist() == [1.0] * 4
        assert np.allclose(f["history/load_parameter"][...], 0.001 * np.arange(1, 5))
        assert f["history/load_parameter"].attrs["units"] == "mm"
        assert f["history/damage_solved"][...].tolist() == [1] * 4
        assert f["mesh/cell_blocks/T3/quadrature"].attrs["rule"] == "centroid"
        assert f["VTKHDF/CellData/stress"].attrs["derived_method"] == "exact"
        assert f["VTKHDF/CellData/stress"].dtype == np.float32
        assert "stress_zz" in f["VTKHDF/CellData"] and "energy/external_work" not in f["history"]
        assert f["history/energy/elastic"].attrs["method"]
        assert {"top", "bottom"} <= set(f["mesh/node_sets"])
        assert f["materials/phases/0"].attrs["degradation_type"] == "standard"
        assert f["materials/phases/0"].attrs["stress_degradation"] == "split"
        _signed_reactions(f)
        r_top = f["history/reaction/top"][...][:, 1]
    with (out / "results.csv").open(newline="") as fh:     # the existing column is kept as abs(R)/1000
        rows = list(csv.DictReader(fh))
    assert np.allclose([float(r["reaction_kN"]) for r in rows], np.abs(r_top) / 1000.0, rtol=1e-6)
    assert (out / "training_data.h5").is_file()            # the trajectory store is written beside the result file


def test_q4_quasi_static_stores_history_per_point(runs):
    out = runs("q4")
    path = out / "result.vtkhdf"
    with h5py.File(path, "r") as f:
        c = _common_checks(f, out, 3)
        H = f["VTKHDF/CellData/H"]
        assert H.shape == (3 * c["n_cells"], 4) and H.attrs["components"] == "qp0,qp1,qp2,qp3"
        assert H.attrs["location"] == "quadrature_point" and H.dtype == np.float64
        assert set(f["VTKHDF/Types"][...].tolist()) == {9}
        q = f["mesh/cell_blocks/Q4/quadrature"]
        assert q.attrs["point_order"] == "eta outer, xi inner, minus then plus"
        a = 1 / np.sqrt(3)
        assert np.allclose(q["points"][...], [[-a, -a], [a, -a], [-a, a], [a, a]]) and np.allclose(q["weights"], 1)
        qp = f["quadrature/Q4/stress"]
        assert qp.shape == (3, c["n_cells"], 4, 3)
        mean = f["VTKHDF/CellData/stress"][...].reshape(3, c["n_cells"], 3)
        assert f["VTKHDF/CellData/stress"].attrs["derived_method"] == "qp_mean_volume_weighted"
        assert np.allclose(mean, qp[...].mean(axis=2), rtol=1e-5, atol=1e-3)   # uniform grid: equal weights
        dbc = f["loading/damage_bc"]
        assert [dict(dbc[k].attrs)["set"] for k in dbc] == ["crack"] and "crack" not in str(list(f["loading/bc"]))
        _signed_reactions(f)


def test_explicit_dynamics_accumulates_the_time_steps_taken(runs):
    out = runs("dynamic")
    path = out / "result.vtkhdf"
    with h5py.File(path, "r") as f:
        c = _common_checks(f, out, 30)
        h = f["history"]
        dt = h["dt"][...]
        assert len(set(dt.tolist())) > 3                      # adaptive dt really varied
        assert h["time"].attrs["units"] == "s"
        assert np.array_equal(h["step_index"][...] >= 0, (np.arange(30) % 4 == 0) | (np.arange(30) == 29))
        solved = h["damage_solved"][...]
        count = np.arange(1, 31)                              # the solver solves damage at its first five steps, then every third
        assert solved.tolist() == ((count <= 5) | (count % 3 == 0)).astype(int).tolist()
        for name in ("velocity", "acceleration"):
            arr = f[f"VTKHDF/PointData/{name}"]
            assert arr.shape[1] == 3 and arr.dtype == np.float64
            assert np.array_equal(arr[-c["n_pts"]:, :2], c["final"][name])
        assert "H_damage_input" in f["VTKHDF/CellData"] and "energy/kinetic" in h


@R.requires_sigusr1
def test_time_limit_signal_finishes_the_increment(tmp_path):
    proc, out = R.run(tmp_path, R.t3_qs(tmp_path), inject=("signal", 2), check=False)
    assert proc.returncode == 11, proc.stderr[-2000:]
    with h5py.File(out / "result.vtkhdf", "r") as f:
        assert f.attrs["status"] == "time_limited"
        assert f["history/increment"][...].tolist() == [0, 1]
        assert f["history/step_index"][...].tolist() == [0, 1]
    assert (out / "energy.csv").exists() and (out / "run_metadata.json").exists()


def test_mesh_change_stops_the_file_and_keeps_the_other_outputs(tmp_path):
    proc, out = R.run(tmp_path, R.t3_qs(tmp_path), inject=("mesh", 3))
    assert "mesh changed at increment 2" in proc.stderr
    with h5py.File(out / "result.vtkhdf", "r") as f:
        assert f.attrs["status"] == "failed" and "mesh changed" in f["meta"].attrs["incomplete_reason"]
        assert f["history/increment"][...].tolist() == [0, 1]
    with (out / "energy.csv").open() as fh:
        assert len(fh.read().strip().splitlines()) == 5


def test_failure_closes_with_status_failed_and_result_false_disables(tmp_path):
    proc, out = R.run(tmp_path, R.q4_qs(tmp_path), inject=("raise", 2), check=False)
    assert proc.returncode != 0 and "injected by the test driver" in proc.stderr
    with h5py.File(out / "result.vtkhdf", "r") as f:
        assert f.attrs["status"] == "failed" and "RuntimeError" in f["meta"].attrs["failure"]
        assert f["history/increment"][...].tolist() == [0]
    cfg = R.q4_qs(tmp_path, result=False)
    cfg["loading"]["num_steps"] = 1
    _, out = R.run(tmp_path, cfg, name="off")
    assert not (out / "result.vtkhdf").exists() and (out / "energy.csv").exists()


def test_settings_resolution():
    s = resolve_result_settings(True, explicit=False, h5_every=20)
    assert s["every"] == 1 and s["fields"] == ["displacement", "damage", "H", "stress", "strain"]
    s = resolve_result_settings(True, explicit=True, h5_every=20)
    assert s["every"] == 20 and not s["every_given"] and "velocity" in s["fields"]
    assert resolve_result_settings(False, explicit=False, h5_every=1) is None
    assert resolve_result_settings(None, explicit=False, h5_every=1) is None
    with pytest.raises(ValueError, match="drops state"):
        resolve_result_settings({"fields": ["displacement", "damage"]}, explicit=False, h5_every=1)
    with pytest.raises(ValueError, match="unknown keys"):
        resolve_result_settings({"evry": 2}, explicit=False, h5_every=1)
    with pytest.raises(ValueError, match="derived_every 1"):
        resolve_result_settings({"derived_location": "quadrature_point", "derived_every": 2}, explicit=False,
                                h5_every=1)
    assert json.dumps(resolve_result_settings({"every": 5}, explicit=True, h5_every=20))


def test_traction_run_records_work_and_ramp_factors(runs):
    out = runs("traction")
    with h5py.File(out / "result.vtkhdf", "r") as f:
        h = f["history"]
        n = len(h["increment"])
        assert f.attrs["status"] == "complete" and n >= 2
        assert np.array_equal(h["time"][...], np.cumsum(h["dt"][...]))
        assert h["energy/balance_relative"].attrs["units"] == "1" and h["energy/external_work"].attrs["method"]
        assert h["energy/external_work"][-1] != 0.0
        bc = {k: dict(f["loading/bc"][k].attrs) for k in f["loading/bc"]}
        top = next(a for a in bc.values() if a["set"] == "top")
        bottom = next(a for a in bc.values() if a["set"] == "bottom")
        assert top["kind"] == "traction" and top["value"] in h and bottom["value"] in h
        start = np.concatenate([[0.0], np.cumsum(h["dt"][...])[:-1]])
        assert np.allclose(h[top["value"]][...], np.minimum(start / 1.0e-8, 1.0), rtol=1e-12, atol=1e-15)
        assert np.allclose(h[bottom["value"]][...], 1.0)
        # work of the applied force with the left-point rule, recomputed from the stored columns
        work = h["energy/external_work"][...]
        total = h["energy/elastic"][...] + h["energy/fracture"][...] + h["energy/kinetic"][...]
        assert np.allclose(h["energy/balance_residual"][...], total - total[0] - work, rtol=1e-9, atol=1e-18)
        assert work[0] == 0.0 and h["energy/balance_residual"][0] == 0.0     # the ramp starts at zero


def _short_default_deck(tmp_path, **output):
    """Three increments of the T3 deck with the ``result`` key left out and the trajectory store on."""
    cfg = R.t3_qs(tmp_path, h5=True, h5_every=2, **output)
    cfg["output"].pop("result", None)
    cfg["loading"]["num_steps"] = 3
    return cfg


def test_the_result_file_is_written_by_default_beside_the_legacy_files(tmp_path):
    from phast.config.config import OutputConfig
    import phast

    assert OutputConfig().result is True
    assert resolve_result_settings(OutputConfig().result, explicit=False, h5_every=1) is not None
    cfg = _short_default_deck(tmp_path)
    assert "result" not in cfg["output"]
    _, out = R.run(tmp_path, cfg, name="default")
    R.validate_file(out / "result.vtkhdf")
    for name in ("result.vtkhdf.sha256", "training_data.h5", "energy.csv", "results.csv", "run_metadata.json",
                 "run_lockfile.json"):
        assert (out / name).is_file(), name
    with h5py.File(out / "training_data.h5", "r") as store:
        labels = sorted(store["simulation_data/steps"])
    assert labels == ["step_0000", "step_0002"]                      # legacy labels follow output.h5_every
    r = phast.load_result(out)                                       # the result file wins in the directory
    assert r.steps() == [0, 1, 2] and r.has_field("damage") and r.field("damage", step=-1).ndim == 1
    assert [float(x) for x in r.history_columns()["increment"]] == [0.0, 1.0, 2.0]
    assert r.history("results") and r.history("energy")              # the CSV histories keep their names


def test_an_explicit_false_writes_no_result_file_and_keeps_the_legacy_files(tmp_path):
    import phast

    cfg = _short_default_deck(tmp_path)
    cfg["output"]["result"] = False
    _, out = R.run(tmp_path, cfg, name="off")
    assert not (out / "result.vtkhdf").exists() and not (out / "result.vtkhdf.sha256").exists()
    assert not (out / "result.vtkhdf.partial").exists()
    for name in ("training_data.h5", "energy.csv", "results.csv"):
        assert (out / name).is_file(), name
    r = phast.load_result(out)                                       # the trajectory store keeps its step labels
    assert r.field("damage", step=2).ndim == 1
    with pytest.raises(phast.ResultLoadError):
        r.steps()


def test_a_synchronised_folder_warns_and_still_writes_the_result_file(tmp_path):
    """The writer's rule 1 is advice: the run is not aborted and the file is written and valid."""
    base = tmp_path / "CloudStorage" / "OneDrive-test" / "run"
    cfg = _short_default_deck(base)
    cfg["loading"]["num_steps"] = 2
    proc, out = R.run(base, cfg, name="synced")
    assert proc.returncode == 0 and "looks like a synchronised folder" in proc.stderr
    R.validate_file(out / "result.vtkhdf")
    with h5py.File(out / "result.vtkhdf", "r") as f:
        assert f.attrs["status"] == "complete"


def test_invalid_result_settings_are_rejected_when_the_configuration_is_validated():
    from phast.config.config_validation import validate_config

    base = {"solver": {"solver_type": "quasi_static"}}
    assert [e for e in validate_config(dict(base, output={"result": {"every": 2}})) if e.path == "output"] == []
    errors = validate_config(dict(base, output={"result": {"evry": 2}}))
    assert any(e.path == "output" and "unknown keys" in e.message for e in errors)
    errors = validate_config({"solver": {"solver_type": "explicit"}, "output": {"result": {"fields": ["damage"]}}})
    assert any(e.path == "output" and "drops state" in e.message for e in errors)
