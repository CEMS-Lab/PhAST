"""Explicit runs with ``adaptive_dt: true`` load the model at the accumulated time.

With adaptive time stepping the step size changes during the run, so the time at
the start of step ``n`` is the sum of the steps already taken and not ``n * dt``.
The load factor, the time-dependent tractions and the time columns of the output
files must all follow that accumulated time. ``history/time`` of the result file
is the accumulated time at the end of each increment; the time at its start is
``history/time - history/dt``.

Each run is a few dozen explicit steps on a coarse mesh (CPU, float64), driven
through ``python -m phast run`` (``result_v2/runner_helpers.py``).
"""
from __future__ import annotations

import csv

import h5py
import numpy as np
import pytest

from phast.config.config import LoadingConfig, compute_load_factor
from result_v2 import runner_helpers as R

pytestmark = [pytest.mark.timeout(180)]

T_RAMP_LOAD = 2.0e-8          # global loading ramp of the deck (R.dynamic)
T_RAMP_TRACTION = 4.0e-8      # per-boundary ramp of the traction, deliberately different


def _traction_deck(tmp):
    """The adaptive T3 deck with its prescribed top displacement replaced by a ramped traction."""
    cfg = R.dynamic(tmp)
    top = next(b for b in cfg["boundary_conditions"] if b["nodes"] == "top")
    top.clear()
    top.update(nodes="top", type="traction", component=1, value=3000.0, ramp_type="linear", t_ramp=T_RAMP_TRACTION)
    return cfg


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    cache = {}

    def get(kind):
        if kind not in cache:
            tmp = tmp_path_factory.mktemp(kind)
            cfg = {"displacement": lambda: R.dynamic(tmp), "traction": lambda: _traction_deck(tmp)}[kind]()
            cache[kind] = R.run(tmp, cfg)[1]
        return cache[kind]
    return get


def _history(out):
    with h5py.File(out / "result.vtkhdf", "r") as f:
        h = f["history"]
        dt, time, lf = h["dt"][...], h["time"][...], h["load_factor"][...]
        traction = {k: h["traction_factor"][k][...] for k in h["traction_factor"]} if "traction_factor" in h else {}
    start = time - dt                                       # time at the start of each increment
    assert np.allclose(time, np.cumsum(dt), rtol=1e-12, atol=0)
    assert dt.max() / dt.min() > 2                          # the step size really varied
    assert not np.allclose(start, np.arange(len(dt)) * dt[0], rtol=1e-3, atol=0)   # not step * (initial dt)
    return start, dt, lf, traction


def _ramp(start, t_ramp):
    return np.minimum(start / t_ramp, 1.0)


def _csv_column(path, name):
    with path.open(newline="") as fh:
        return np.array([float(r[name]) for r in csv.DictReader(fh)])


def test_load_factor_follows_the_accumulated_time(runs):
    out = runs("displacement")
    start, dt, lf, _ = _history(out)
    assert np.sum((lf > 0.0) & (lf < 1.0)) > 10             # a ramp that is mostly not yet saturated
    assert np.allclose(lf, _ramp(start, T_RAMP_LOAD), rtol=0, atol=1e-12)


def test_time_columns_of_the_csv_files_follow_the_accumulated_time(runs):
    out = runs("displacement")
    start, dt, _, _ = _history(out)
    for name, column in (("energy.csv", "time"), ("solver_telemetry.csv", "time"), ("results.csv", "time")):
        assert np.allclose(_csv_column(out / name, column), start, rtol=1e-8, atol=1e-18), name
    assert np.allclose(_csv_column(out / "solver_telemetry.csv", "dt"), dt, rtol=1e-8, atol=0)   # the step used, not the next one


def test_time_dependent_traction_follows_the_accumulated_time(runs):
    out = runs("traction")
    start, dt, lf, traction = _history(out)
    assert list(traction) and all(np.sum((v > 0.0) & (v < 1.0)) > 10 for v in traction.values())
    for factor in traction.values():
        assert np.allclose(factor, _ramp(start, T_RAMP_TRACTION), rtol=0, atol=1e-12)
    assert np.allclose(lf, _ramp(start, T_RAMP_LOAD), rtol=0, atol=1e-12)    # the global factor is independent of it


def test_compute_load_factor_uses_the_given_time_and_defaults_to_step_times_dt():
    loading = LoadingConfig(ramp_type="linear", t_ramp=1.0e-8)
    assert compute_load_factor(4, 1.0e-9, loading) == pytest.approx(0.4)
    assert compute_load_factor(4, 1.0e-9, loading, t=None) == compute_load_factor(4, 1.0e-9, loading)
    assert compute_load_factor(4, 1.0e-9, loading, t=2.5e-9) == pytest.approx(0.25)
    assert compute_load_factor(4, 1.0e-9, loading, t=0.0) == 0.0
