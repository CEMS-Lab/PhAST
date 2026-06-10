"""
Test for the B7 (dynamic crack branching, COMSOL 6.4) comparison
script — pins the energy-unit convention and prevents the issue #209
regression from recurring.

Background
----------
The 2D solver works in (mm, MPa, mJ). The energy integrand units are
``MPa * mm^2 = N``, so the integrated elastic energy is in
``mJ/mm`` per unit out-of-plane thickness. Numerically that is
identical to ``J/m`` — the COMSOL reference convention at thickness
1 m. Therefore ``compare.py`` should *not* apply any thickness
multiplier to the raw H5 attribute.

A previous revision applied ``THICKNESS_MM = 1000`` to the value,
inflating the reported peak to ~341 J vs the 0.26-0.28 J reference
band (issue #209). This test fakes a minimal H5 with a known peak
of 0.33 (in solver-internal units) and asserts the loader plus
post-scaling pipeline returns 0.33 in J — i.e. no spurious *1000.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
COMPARE_PY = (
    HERE.parent
    / "examples"
    / "dynamic"
    / "crack_branching_comsol"
    / "compare.py"
)


def _load_compare_module():
    spec = importlib.util.spec_from_file_location("b7_compare", COMPARE_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_fake_run(tmp_path: Path, peak_value: float = 0.33):
    """Write a minimal run dir with a training_data.h5 containing a
    known elastic-energy peak and a 2-row timing CSV / metadata."""
    h5_path = tmp_path / "training_data.h5"
    with h5py.File(h5_path, "w") as f:
        steps_grp = f.create_group("simulation_data/steps")
        # 11 snapshots, peak in the middle.
        e_curve = np.linspace(0.0, peak_value, 6).tolist() + np.linspace(
            peak_value, 0.05, 6
        ).tolist()[1:]
        for i, e in enumerate(e_curve):
            g = steps_grp.create_group(f"step_{i:04d}")
            g.attrs["energy_elastic"] = float(e)
            g.attrs["time_s"] = float(i) * 5.0e-6  # 0..50 us
    # Minimal timing CSV (step, max_d) — required by load_timing.
    (tmp_path / "timing_per_step.csv").write_text(
        "step,max_d\n0,0.0\n1,1.0\n"
    )
    (tmp_path / "run_metadata.json").write_text(
        json.dumps({"solver": {"dt": 1.0e-6}})
    )
    return tmp_path


def test_load_elastic_energy_returns_raw_attribute(tmp_path):
    run_dir = _make_fake_run(tmp_path, peak_value=0.33)
    mod = _load_compare_module()

    out = mod.load_elastic_energy(run_dir)
    assert out is not None, "loader returned None for valid H5"
    t_us, e_arr = out
    assert e_arr.max() == pytest.approx(0.33, rel=1e-12), (
        "loader rescaled energy attribute; should be raw H5 value"
    )


def test_compare_script_no_spurious_thickness_multiplier(tmp_path, capsys):
    """End-to-end: peak should land in COMSOL band, not 1000x off.

    Catches issue #209: an extra ``* THICKNESS_MM`` (=1000) inflated
    a 0.33 J peak to 330 J, blowing the +/-25% acceptance band by
    three orders of magnitude.
    """
    run_dir = _make_fake_run(tmp_path, peak_value=0.33)
    mod = _load_compare_module()

    # Drive main() via argv; main() calls sys.exit().
    import sys

    old_argv = sys.argv[:]
    sys.argv = ["compare.py", "--run-dir", str(run_dir)]
    try:
        with pytest.raises(SystemExit) as ei:
            mod.main()
    finally:
        sys.argv = old_argv

    text = capsys.readouterr().out
    # The script prints "Elastic peak J (1m)  : <X> J ...".
    # Extract the number and assert it's the raw H5 peak (no *1000).
    for line in text.splitlines():
        if "Elastic peak J" in line:
            num = float(line.split(":")[1].strip().split()[0])
            assert num == pytest.approx(0.33, rel=1e-6), (
                f"compare.py rescaled peak to {num} J; expected 0.33 "
                f"(no thickness multiplier — issue #209)"
            )
            assert num <= 1.0, (
                f"peak {num} J is >1 J — likely the *1000 regression."
            )
            break
    else:
        pytest.fail(f"could not find 'Elastic peak J' in compare output:\n{text}")


def test_compare_script_passes_acceptance_band_with_realistic_peak(
    tmp_path, capsys
):
    """A peak of 0.27 J (mid-band) should yield PASS on the energy line."""
    run_dir = _make_fake_run(tmp_path, peak_value=0.27)
    mod = _load_compare_module()
    import sys

    old_argv = sys.argv[:]
    sys.argv = ["compare.py", "--run-dir", str(run_dir)]
    try:
        with pytest.raises(SystemExit):
            mod.main()
    finally:
        sys.argv = old_argv

    text = capsys.readouterr().out
    for line in text.splitlines():
        if "Elastic peak J" in line:
            assert "PASS" in line, f"expected PASS for in-band peak, got: {line}"
            return
    pytest.fail("no 'Elastic peak J' line in output")


# ---------------------------------------------------------------------------
# Issue #213 — preseeded notch nodes must be excluded from the initiation /
# branching / full-Y detectors, because pf_dirichlet locks them at d=1 from
# t=0 and would otherwise make the initiation time fire at 0 us.
# ---------------------------------------------------------------------------

def _make_preseed_run(tmp_path: Path, n_nodes: int = 20,
                      preseed_idx=(0, 1, 2),
                      crack_grow_step: int = 6):
    """Write a fake run dir whose H5 has a 1-D mesh with ``preseed_idx`` nodes
    pinned at d=1 from step 0. Bulk nodes only saturate from
    ``crack_grow_step`` onward, so the correct initiation time is
    ``crack_grow_step * dt`` (not 0 us)."""
    h5_path = tmp_path / "training_data.h5"
    n_steps = 12
    with h5py.File(h5_path, "w") as f:
        mesh_grp = f.create_group("simulation_data/mesh")
        coords = np.zeros((n_nodes, 2), dtype=float)
        coords[:, 0] = np.linspace(0.0, 100.0, n_nodes)
        mesh_grp.create_dataset("node_coordinates", data=coords)
        mesh_grp.attrs["n_nodes"] = n_nodes
        ns_grp = mesh_grp.create_group("node_sets")
        ns_grp.create_dataset("notch_upper",
                              data=np.asarray(preseed_idx, dtype=np.int64))
        steps_grp = f.create_group("simulation_data/steps")
        for i in range(n_steps):
            g = steps_grp.create_group(f"step_{i:04d}")
            d = np.zeros(n_nodes, dtype=float)
            for j in preseed_idx:
                d[j] = 1.0  # locked from t=0
            if i >= crack_grow_step:
                # Bulk crack-tip node saturates here.
                d[10] = 1.0
            g.create_dataset("damage_nodal", data=d)
            g.attrs["time_s"] = float(i) * 1.0e-6
            g.attrs["energy_elastic"] = 0.1
    # Timing CSV: scalar max_d hits 1.0 at step 0 because of the preseed.
    rows = ["step,max_d"]
    for i in range(n_steps):
        rows.append(f"{i},1.0")
    (tmp_path / "timing_per_step.csv").write_text("\n".join(rows) + "\n")
    (tmp_path / "run_metadata.json").write_text(json.dumps({
        "solver": {"dt": 1.0e-6},
        "preseed_notch_nodesets": ["notch_upper"],
    }))
    return tmp_path


def test_initiation_excludes_preseed_via_metadata(tmp_path):
    run_dir = _make_preseed_run(tmp_path, crack_grow_step=6)
    mod = _load_compare_module()

    preseed_idx, src = mod._resolve_preseed_node_indices(run_dir)
    assert preseed_idx is not None
    assert set(preseed_idx.tolist()) == {0, 1, 2}
    assert src == "metadata"

    out = mod.load_max_d_excluding_preseed(run_dir, preseed_idx)
    assert out is not None
    t_us, max_d = out
    above = np.where(max_d > 0.99)[0]
    assert above.size > 0
    initiation_us = float(t_us[above[0]])
    # Without exclusion the answer would be 0.0 us (preseed locked).
    # With exclusion it should be the bulk crack-tip saturation step
    # (crack_grow_step * dt = 6 us).
    assert initiation_us == pytest.approx(6.0, rel=1e-6), (
        f"expected initiation at 6 us (post-exclusion), got {initiation_us} us"
    )


def test_initiation_falls_back_to_h5_heuristic(tmp_path):
    """No metadata -> name-based heuristic on H5 nodesets (notch_*)."""
    run_dir = _make_preseed_run(tmp_path, crack_grow_step=4)
    # Strip the metadata key so the heuristic has to fire.
    (run_dir / "run_metadata.json").write_text(json.dumps({
        "solver": {"dt": 1.0e-6},
    }))
    mod = _load_compare_module()
    preseed_idx, src = mod._resolve_preseed_node_indices(run_dir)
    assert preseed_idx is not None
    assert "h5-heuristic" in src
    assert set(preseed_idx.tolist()) == {0, 1, 2}


def test_initiation_falls_back_when_no_preseed_info(tmp_path):
    """No metadata key, no H5 nodesets -> None + warning (current behaviour)."""
    run_dir = _make_preseed_run(tmp_path, crack_grow_step=4)
    # Drop the H5 nodesets and the metadata key.
    with h5py.File(run_dir / "training_data.h5", "a") as f:
        del f["simulation_data/mesh/node_sets"]
    (run_dir / "run_metadata.json").write_text(json.dumps({
        "solver": {"dt": 1.0e-6},
    }))
    mod = _load_compare_module()
    preseed_idx, src = mod._resolve_preseed_node_indices(run_dir)
    assert preseed_idx is None
    assert src == "no-nodesets-in-h5"
