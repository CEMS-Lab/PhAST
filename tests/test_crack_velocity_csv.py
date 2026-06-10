"""
Issue #240 — crack-tip velocity CSV post-processing bugs.

Two bugs in ``BenchmarkPostProcessor._generate_crack_tip_csv``:

1. ``crack_vel_frac_cR`` was unnormalised — when ``run_metadata.json``
   did not carry a wave speed, the reader fell back to ``c_R = 1.0``,
   so the column held raw ``vel/1.0`` (i.e. mm/s) instead of a true
   fraction. After the fix, ``c_R_mm_s`` is preferred (matching the
   solver-internal mm/s convention); legacy ``c_R_m_s`` is converted
   to mm/s before the division. With no wave-speed info, the column
   reports NaN rather than a misleadingly small number.

2. Velocity spikes from raw differentiation of the crack-tip
   x-position. Smoothing the *position* with a 5-point Savitzky-Golay
   filter before differentiation flattens snapshot-level jumps while
   preserving the long-time trend.
"""
from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")


def _make_synthetic_run(tmp_path: Path,
                        c_R_mm_s: float = 2125000.0,  # ~2125 m/s, glass
                        with_jitter: bool = True):
    """Build a fake run dir with a 2-D mesh and a crack tip that
    advances ~uniformly along x with optional jitter to test the SG
    smoother. The non-jittered tip motion gives a constant velocity of
    ``v = 1000 mm/s`` (peanut units; arbitrary), so
    ``crack_vel_frac_cR`` should be ``1000 / c_R_mm_s``.
    """
    n_steps = 30
    # Mesh: 200 nodes on a 2-D rectangular strip [0, 100] x [-1, 1].
    n_x, n_y = 100, 2
    xs = np.linspace(0.0, 100.0, n_x)
    ys = np.linspace(-1.0, 1.0, n_y)
    XX, YY = np.meshgrid(xs, ys, indexing='xy')
    coords = np.column_stack([XX.ravel(), YY.ravel()])
    n_nodes = coords.shape[0]

    # True tip position: linear in time so the analytical velocity is
    # constant (1 mm per 1 us = 1000 mm/s). dt = 1 us per snapshot.
    dt_s = 1.0e-6
    tip_true = np.linspace(50.0 + 1.0, 50.0 + n_steps * 1.0, n_steps)
    if with_jitter:
        rng = np.random.default_rng(0)
        # Element-width-scale snapshot jitter ~ +/- 0.5 mm.
        tip_meas = tip_true + rng.uniform(-0.5, 0.5, size=n_steps)
    else:
        tip_meas = tip_true.copy()

    h5_path = tmp_path / "training_data.h5"
    with h5py.File(h5_path, "w") as f:
        mesh_grp = f.create_group("simulation_data/mesh")
        mesh_grp.create_dataset("node_coordinates", data=coords)
        # Trivial triangle connectivity covering the strip; required by
        # BenchmarkPostProcessor.__init__ even though _generate_crack_tip_csv
        # itself only reads node_coordinates + damage_nodal.
        elems = []
        for i in range(n_x - 1):
            a = i           # row 0 col i
            b = i + 1       # row 0 col i+1
            c = i + n_x     # row 1 col i
            d = i + n_x + 1 # row 1 col i+1
            elems.append([a, b, c])
            elems.append([b, d, c])
        mesh_grp.create_dataset("element_connectivity",
                                data=np.asarray(elems, dtype=np.int64))
        mesh_grp.attrs["n_nodes"] = n_nodes
        mesh_grp.attrs["n_elements"] = len(elems)
        steps_grp = f.create_group("simulation_data/steps")
        for i in range(n_steps):
            g = steps_grp.create_group(f"step_{i:04d}")
            d = np.zeros(n_nodes, dtype=float)
            # Mark every node with x <= measured tip as fully damaged.
            mask = coords[:, 0] <= tip_meas[i]
            d[mask] = 1.0
            g.create_dataset("damage_nodal", data=d)
            g.attrs["time_s"] = float((i + 1) * dt_s)
            g.attrs["energy_elastic"] = 0.1

    # Metadata with c_R in mm/s.
    (tmp_path / "run_metadata.json").write_text(json.dumps({
        "solver": {
            "solver_type": "explicit",
            "dt": dt_s,
            "c_R_mm_s": c_R_mm_s,
            "c_R_m_s": c_R_mm_s * 1.0e-3,
        },
        "material": {"l0": 0.25, "Gc": 2.7, "E": 32e3, "nu": 0.2,
                     "rho": 2.45e-9},
        "problem": "synthetic_crack_velocity",
    }))
    return tmp_path, dt_s, tip_true


def _read_crack_csv(run_dir: Path):
    csv = run_dir / "crack_tip.csv"
    assert csv.exists(), "crack_tip.csv not generated"
    text = csv.read_text().splitlines()
    header = text[0].split(',')
    rows = [r.split(',') for r in text[1:] if r.strip()]

    def col(name):
        i = header.index(name)
        return np.array([float(r[i]) for r in rows], dtype=float)

    return {h: col(h) for h in header
            if h not in ("step", "n_crack_tips", "branched")}


def test_crack_vel_frac_cR_is_normalised(tmp_path):
    """vel_frac == vel_mms / c_R_mm_s, NOT raw mm/s."""
    from phast.postprocess_paper import BenchmarkPostProcessor

    c_R = 2.125e6  # mm/s
    run_dir, _, _ = _make_synthetic_run(tmp_path, c_R_mm_s=c_R,
                                        with_jitter=False)
    pp = BenchmarkPostProcessor(str(run_dir))
    pp._generate_crack_tip_csv()

    cols = _read_crack_csv(run_dir)
    vel = cols["crack_vel_mms"]
    vfrac = cols["crack_vel_frac_cR"]
    # Ratio must equal 1/c_R wherever vel is meaningful.
    # CSV writes vel with %.2f and vel_frac with %.4f, so allow
    # printf-rounding tolerance.
    np.testing.assert_allclose(vfrac[1:], vel[1:] / c_R, atol=5e-4)
    # Sanity: synthetic tip moves at v ~ 1 mm/us = 1e6 mm/s against
    # c_R = 2.125e6 mm/s -> vel_frac ~ 0.47. Pre-fix the column would
    # have held the raw m/s number ~ 1e6, blowing past 1.0 by orders
    # of magnitude.
    assert 0.3 < float(np.nanmax(np.abs(vfrac))) < 0.7, (
        f"vel_frac = {np.nanmax(np.abs(vfrac))} -- expected ~0.47 "
        "(v=1 mm/us / c_R=2.125e6 mm/s). Pre-fix this would have been "
        "raw mm/s ~ 1e6 (issue #240)."
    )
    assert float(np.nanmax(np.abs(vfrac))) < 1.0, (
        "vel_frac >= 1: still unnormalised (issue #240)"
    )


def test_crack_vel_smoothed_no_spikes(tmp_path):
    """SG-smoothing of tip_x suppresses snapshot-jitter spikes."""
    from phast.postprocess_paper import BenchmarkPostProcessor

    run_dir, dt_s, tip_true = _make_synthetic_run(tmp_path,
                                                   with_jitter=True)
    pp = BenchmarkPostProcessor(str(run_dir))
    pp._generate_crack_tip_csv()

    cols = _read_crack_csv(run_dir)
    vel = cols["crack_vel_mms"]

    # True velocity = 1 mm / 1 us = 1e6 mm/s. After 5-pt SG smoothing
    # of tip_x(t), the *interior* velocity samples should cluster near
    # the truth. Without any smoothing, snapshot-jitter (~ +/-0.5 mm
    # in tip_x) with dt=1 us produces velocity spikes >= 5e5 mm/s on
    # individual samples and a much higher max/min ratio than the
    # smoothed version.
    interior = vel[5:-5]
    mean_v = float(np.mean(interior))
    assert abs(mean_v - 1.0e6) / 1.0e6 < 0.20, (
        f"smoothed mean velocity {mean_v:.2e} mm/s differs from truth "
        "1e6 mm/s by > 20%; smoothing pipeline broken (issue #240)."
    )

    # Also: also-run the same H5 with smoothing disabled (raw diff)
    # must produce a strictly larger spike amplitude than the smoothed
    # output. Recompute vel from RAW tip_x using the same H5.
    h5_path = run_dir / "training_data.h5"
    raw_x = []
    raw_t = []
    with h5py.File(h5_path, "r") as f:
        steps = f["simulation_data/steps"]
        coords = np.asarray(f["simulation_data/mesh/node_coordinates"])
        notch_x = (coords[:, 0].min() + coords[:, 0].max()) / 2
        for k in sorted(steps.keys()):
            grp = steps[k]
            d = np.asarray(grp["damage_nodal"])
            if d.max() < 0.5:
                continue
            mask = (d > 0.5) & (coords[:, 0] > notch_x + 0.1)
            if not mask.any():
                continue
            raw_x.append(coords[mask, 0].max())
            raw_t.append(float(grp.attrs["time_s"]))
    raw_x = np.asarray(raw_x)
    raw_t = np.asarray(raw_t)
    raw_vel = np.diff(raw_x) / np.diff(raw_t)
    raw_p2p = float(raw_vel[5:-5].max() - raw_vel[5:-5].min())
    smoothed_p2p = float(interior.max() - interior.min())
    assert smoothed_p2p < raw_p2p, (
        f"SG-smoothed peak-to-peak ({smoothed_p2p:.2e}) is not less than "
        f"raw-diff peak-to-peak ({raw_p2p:.2e}); smoothing inactive (issue #240)."
    )


def test_crack_vel_falls_back_to_nan_without_c_R(tmp_path):
    """No c_R info in metadata -> vel_frac is NaN, not silently 0/1."""
    from phast.postprocess_paper import BenchmarkPostProcessor

    run_dir, _, _ = _make_synthetic_run(tmp_path, with_jitter=False)
    # Strip c_R from metadata.
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    meta["solver"].pop("c_R_mm_s", None)
    meta["solver"].pop("c_R_m_s", None)
    (run_dir / "run_metadata.json").write_text(json.dumps(meta))
    # Remove any cached crack_tip.csv.
    csv = run_dir / "crack_tip.csv"
    if csv.exists():
        csv.unlink()

    pp = BenchmarkPostProcessor(str(run_dir))
    pp._generate_crack_tip_csv()

    cols = _read_crack_csv(run_dir)
    vfrac = cols["crack_vel_frac_cR"]
    assert np.all(np.isnan(vfrac)), (
        f"expected NaN vel_frac without c_R, got {vfrac[:5]}"
    )
