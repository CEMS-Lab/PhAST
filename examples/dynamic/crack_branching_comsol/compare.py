#!/usr/bin/env python3
"""Compare a B7 dynamic crack branching run against the COMSOL 6.4
Application Library reference (`reference_solutions/`).

Acceptance criteria (issue #135 / this benchmark):

  - Branching onset time within +/-20% of 33 us
  - Full-Y morphology achieved by t = 75 us (TWO right-side arms;
    issue #314)
  - Elastic-energy peak (per-mm * thickness scale) within +/-25% of
    0.13 - 0.14 J at COMSOL's thickness convention (1 m)

Reads max_d(t) from ``timing_per_step.csv`` to estimate initiation and
saturation times. Reads elastic energy from ``training_data.h5``
(when present) to estimate the elastic-energy peaks. Writes
``compare_report.txt`` and ``compare.png`` next to the run.

Morphology check (issue #314). The previous ``pass_full_y`` test only
required ``max(d) > 0.99``, which gives PASS even on a saturated
straight-line crack (no branching). The previous ``branching_us``
detector took the late-window argmax of the elastic-energy curve,
which can fire on a wave-reflection peak. Both metrics are now
backed by a true topology check:

  1. Interpolate the nodal damage field onto a uniform 2D grid
     (resolution ~l0/2) using a triangular linear interpolator.
  2. Threshold at ``d > 0.5`` and apply :func:`scipy.ndimage.label`
     to get connected components.
  3. Count CCs in the **right of the pre-crack** region
     (``x > a + 2*l0``), drop islands smaller than 5 cells.
  4. ``n_arms_final`` = CC count at the final time step.
     ``branching_components_us`` = first time CC count >= 2.
     ``pass_full_y`` requires ``n_arms_final >= 2``.

The energy-argmax onset is retained as
``branching_energy_argmax_us`` for backward compat / diagnostics.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

HERE = Path(__file__).resolve().parent
REF_TIMES = HERE / "reference_solutions" / "comsol_branching_times.txt"
REF_ENERGY = HERE / "reference_solutions" / "comsol_energy_curve.csv"

# Energy unit convention.
#
# Our 2D solver works in mm (geometry) and MPa (stress). The energy
# integrand has units MPa * mm^2 = N, integrated over the 2D domain
# this gives a quantity with units N (i.e. force) which is the energy
# per unit out-of-plane thickness expressed as N*mm/mm = mJ/mm.
#
# COMSOL reports energy in J at thickness 1 m. Since
#   mJ / mm  ==  J / m
# our raw H5 ``energy_elastic`` is *already* numerically equal to the
# COMSOL convention; no thickness conversion is needed. Earlier
# revisions of this script multiplied by 1000 (interpreting the value
# as mJ at 1mm thickness and rescaling to J at 1m thickness), which
# double-counts the conversion and inflates the peak ~1300x against
# the reference band (issue #209).

# Reference values.
#
# REFRAMING 2026-05-08 (reports/B7_literature_parameter_comparison_2026-
# 05-08.md): the 33 us branching-onset target was COMSOL-only. Of all
# 22 PDFs in refs/, only Ren 2019 (Eng Fract Mech 218:106569) quotes a
# branching-onset value: 68.2 us (mesh 2) / 70.1 us (mesh 1). Borden
# 2012 itself ("not possible to say precisely"), PhaFiDyn 2025, and
# Vinut FEniCS do not state a number. The default acceptance gate is
# now Ren 2019; COMSOL Application Library 33 us is retained as a
# secondary reference but not the default.
#
# (COMSOL Fig 4 reports HALF-plate energies; we run the full-plate
# equivalent, so the expected sim energy is 2x the COMSOL-reported
# value.)
REF_INITIATION_US = 10.0
REF_BRANCH_US = 68.2                    # Ren 2019, mesh 2 (default)
REF_BRANCH_US_COMSOL = 33.0             # COMSOL Application Library outlier
REF_FULL_Y_US = 75.0
COMSOL_HALFPLATE_PEAK_J_LO = 0.13
COMSOL_HALFPLATE_PEAK_J_HI = 0.14
HALFPLATE_TO_FULLPLATE = 2.0
REF_PEAK_J_LO = COMSOL_HALFPLATE_PEAK_J_LO * HALFPLATE_TO_FULLPLATE  # 0.26
REF_PEAK_J_HI = COMSOL_HALFPLATE_PEAK_J_HI * HALFPLATE_TO_FULLPLATE  # 0.28

TOL_BRANCH = 0.20   # +/-20% on branching onset
TOL_PEAK = 0.25     # +/-25% on elastic-energy peak

# Morphology defaults (issue #314).
PRECRACK_LENGTH_MM = 50.0      # B7 ``geometry.parameters.a``
L0_DEFAULT_MM = 0.5            # ``material.l0`` for B7
MORPH_DAMAGE_THRESH = 0.5      # ``d > thresh`` defines a crack pixel
MORPH_MIN_ISLAND = 5           # CCs smaller than this drop out as noise
MORPH_GRID_PER_L0 = 2          # cells per l0 -> ~l0/2 grid spacing


def find_latest_run_dir(base: Path) -> Path | None:
    """Most recent run_* dir containing timing_per_step.csv (here or cwd)."""
    candidates = []
    for root in (base, Path.cwd()):
        candidates.extend(p for p in root.glob("run_*")
                          if (p / "timing_per_step.csv").exists())
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_timing(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_us, max_d) from timing_per_step.csv.

    Note: ``max_d`` here is the *raw* per-step maximum across ALL nodes,
    including any pre-seeded notch nodes pinned to d=1 by ``pf_dirichlet``.
    For initiation / branching / full-Y detection use
    :func:`load_max_d_excluding_preseed` instead, which masks out the
    preseed nodeset (issue #213).
    """
    arr = np.loadtxt(run_dir / "timing_per_step.csv", delimiter=",",
                     skiprows=1, usecols=(0, 1))
    # Need dt to convert step -> time; pull from run_metadata.json
    import json
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    dt = float(meta["solver"]["dt"])
    t_us = arr[:, 0] * dt * 1e6
    return t_us, arr[:, 1]


def _resolve_preseed_node_indices(run_dir: Path) -> tuple[np.ndarray | None, str]:
    """Return (preseed_node_indices, source_label).

    Lookup chain (issue #213):
      1. ``run_metadata.json``: ``preseed_notch_nodesets`` lists nodeset names.
         Cross-reference against ``simulation_data/mesh/node_sets`` in the H5.
      2. Fallback heuristic: any nodeset in the H5 whose name starts with
         ``notch_``.
      3. None — caller falls back to current behaviour with a warning.
    """
    import json as _json
    h5_path = run_dir / "training_data.h5"
    if not h5_path.exists():
        return None, "no-h5"
    try:
        import h5py
    except ImportError:
        return None, "no-h5py"

    # Names from metadata (preferred).
    names: list[str] = []
    meta_path = run_dir / "run_metadata.json"
    if meta_path.exists():
        try:
            meta = _json.loads(meta_path.read_text())
            v = meta.get("preseed_notch_nodesets") \
                or meta.get("user_preseed_notch_nodesets")
            if isinstance(v, list):
                names = [str(x) for x in v]
        except Exception:
            pass

    with h5py.File(h5_path, "r") as f:
        ns_grp = None
        if "simulation_data/mesh/node_sets" in f:
            ns_grp = f["simulation_data/mesh/node_sets"]

        if ns_grp is None:
            return None, "no-nodesets-in-h5"

        all_names = list(ns_grp.keys())
        if not names:
            # Fallback heuristic: notch_* names.
            names = [n for n in all_names if n.startswith("notch_")]
            source = "h5-heuristic-notch_*"
        else:
            source = "metadata"

        idx_list = []
        for n in names:
            if n in ns_grp:
                idx_list.append(np.asarray(ns_grp[n], dtype=np.int64))
        if not idx_list:
            return None, f"{source}-no-match"
        return np.unique(np.concatenate(idx_list)), source


def load_max_d_excluding_preseed(
    run_dir: Path,
    preseed_idx: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute per-step ``max(d)`` over non-preseed nodes from H5.

    Returns ``(t_us, max_d)`` or ``None`` if H5 is unavailable.
    When ``preseed_idx`` is None, returns the raw all-node maximum.
    """
    h5_path = run_dir / "training_data.h5"
    if not h5_path.exists():
        return None
    try:
        import h5py
    except ImportError:
        return None
    times = []
    maxd = []
    with h5py.File(h5_path, "r") as f:
        if "simulation_data" in f:
            steps = f["simulation_data/steps"]
        else:
            steps = f["steps"] if "steps" in f else f
        n_nodes = None
        if "simulation_data/mesh" in f:
            n_nodes = int(f["simulation_data/mesh"].attrs.get(
                "n_nodes", 0)) or None
        for key in sorted(steps.keys()):
            if not key.startswith("step_"):
                continue
            grp = steps[key]
            if "damage_nodal" not in grp:
                continue
            d = np.asarray(grp["damage_nodal"], dtype=np.float64).ravel()
            if preseed_idx is not None and preseed_idx.size:
                mask = np.ones(d.shape[0], dtype=bool)
                valid = preseed_idx[(preseed_idx >= 0) & (preseed_idx < d.shape[0])]
                mask[valid] = False
                d = d[mask]
            times.append(float(grp.attrs.get("time_s", 0.0)))
            maxd.append(float(d.max()) if d.size else 0.0)
    if not times:
        return None
    return np.asarray(times) * 1e6, np.asarray(maxd)


def load_elastic_energy(run_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Try to read elastic energy from training_data.h5 attributes; fall back
    to ``energy.csv`` (columns ``step,t_s,elastic,kinetic,fracture,total``)
    when H5 is unavailable.
    """
    h5_path = run_dir / "training_data.h5"
    if h5_path.exists():
        try:
            import h5py
            times = []
            elastic = []
            with h5py.File(h5_path, "r") as f:
                steps = f["simulation_data/steps"] if "simulation_data" in f else f
                for key in sorted(steps.keys()):
                    if not key.startswith("step_"):
                        continue
                    grp = steps[key]
                    energy_attr = (
                        "energy_elastic" if "energy_elastic" in grp.attrs
                        else "elastic_energy" if "elastic_energy" in grp.attrs
                        else None)
                    if energy_attr is None:
                        continue
                    times.append(float(grp.attrs.get("time_s", 0.0)))
                    elastic.append(float(grp.attrs[energy_attr]))
            if times:
                return np.asarray(times) * 1e6, np.asarray(elastic)
        except ImportError:
            pass
    # Fallback: energy.csv (always emitted next to timing_per_step.csv).
    csv_path = run_dir / "energy.csv"
    if csv_path.exists():
        try:
            arr = np.loadtxt(csv_path, delimiter=",", skiprows=1)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                t_us = arr[:, 1] * 1e6
                elastic = arr[:, 2]
                return t_us, elastic
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------
# Morphology-aware Y-detection (issue #314)
# ---------------------------------------------------------------------


def _grid_damage(
    coords: np.ndarray,
    elements: np.ndarray | None,
    damage: np.ndarray,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    dx: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate ``damage`` from triangle-node values onto a uniform 2D grid.

    Returns ``(xs, ys, D)`` with ``D`` shape ``(ny, nx)`` (rows along y,
    columns along x). NaN-filled cells outside the convex hull are
    treated as no-damage (replaced with 0) so the connected-component
    pass does not split a valid arm at a sliver of missing coverage.
    """
    import matplotlib.tri as mtri
    nx = max(2, int(np.ceil((x_hi - x_lo) / dx)) + 1)
    ny = max(2, int(np.ceil((y_hi - y_lo) / dx)) + 1)
    xs = np.linspace(x_lo, x_hi, nx)
    ys = np.linspace(y_lo, y_hi, ny)
    XG, YG = np.meshgrid(xs, ys)
    if elements is not None and elements.size:
        tri = mtri.Triangulation(coords[:, 0], coords[:, 1], elements)
        interp = mtri.LinearTriInterpolator(tri, damage)
        D = np.asarray(interp(XG, YG))
    else:
        # Fallback: nearest-neighbour. Used only when element_connectivity
        # is missing from the H5 (older runs / synthetic test fixtures).
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)
        flat = np.column_stack([XG.ravel(), YG.ravel()])
        _, idx = tree.query(flat)
        D = damage[idx].reshape(XG.shape)
    D = np.where(np.isnan(D), 0.0, D)
    return xs, ys, D


def count_arms_in_region(
    D: np.ndarray,
    xs: np.ndarray,
    x_min_arm: float,
    threshold: float = MORPH_DAMAGE_THRESH,
    min_island: int = MORPH_MIN_ISLAND,
) -> int:
    """Count distinct propagating arms in ``x > x_min_arm`` (mm).

    Naively counting 2D connected components of ``d > threshold`` over
    the right-of-pre-crack strip undercounts a Y-branch: the fork
    point lies inside the strip, so both arms remain a single CC.
    Instead we (1) run :func:`scipy.ndimage.label` on the whole grid
    to filter noise islands smaller than ``min_island`` cells, then
    (2) for each column ``j`` in ``x > x_min_arm`` count maximal
    contiguous y-runs of cleaned ``True`` and return the maximum over
    all such columns. A straight crack hits 1 run/column; a Y has
    columns with 2 runs (one per arm) past the fork; a triple branch
    has columns with 3 runs.

    Components with fewer than ``min_island`` cells are dropped as
    measurement noise (e.g. d>0.5 stragglers from interpolator
    overshoot or stress-wave grazing). See issue #314.

    Returns the integer number of arms.
    """
    from scipy.ndimage import label
    if D.ndim != 2:
        raise ValueError(f"D must be 2D, got shape {D.shape}")
    if xs.ndim != 1 or xs.size != D.shape[1]:
        raise ValueError(
            f"xs shape {xs.shape} must match D columns {D.shape[1]}")

    # Step 1: filter noise islands on the FULL grid (not just the
    # x>x_min strip) so a chain of small specks straddling the cutoff
    # does not split into a "real" island when restricted.
    mask = D > threshold
    if not mask.any():
        return 0
    labels, n_lab = label(mask)
    if n_lab == 0:
        return 0
    sizes = np.bincount(labels.ravel())[1:]  # skip background label 0
    if min_island > 1:
        small = np.where(sizes < min_island)[0] + 1
        if small.size:
            mask[np.isin(labels, small)] = False
    if not mask.any():
        return 0

    # Step 2: per-column y-run count in the right-of-pre-crack strip.
    col_mask = xs > x_min_arm
    if not col_mask.any():
        return 0
    sub = mask[:, col_mask]
    if not sub.any():
        return 0
    # ``runs[j]`` = number of maximal True-runs in column j.
    # Equivalent to: count of False->True transitions, plus 1 if the
    # column starts True.
    cols = sub.astype(np.int8)
    starts = cols[0:1, :]                                  # (1, ncols)
    transitions = (cols[1:, :] - cols[:-1, :]) == 1        # rising edges
    runs_per_col = starts.sum(axis=0) + transitions.sum(axis=0)
    return int(runs_per_col.max())


def _load_mesh_and_steps(run_dir: Path):
    """Open the H5 and return (coords, elements, steps_iter, h5_handle).

    ``steps_iter`` yields (step_key, time_us, damage_nodal). Caller
    must close ``h5_handle`` when done.
    Returns ``None`` if H5 is unavailable.
    """
    h5_path = run_dir / "training_data.h5"
    if not h5_path.exists():
        return None
    try:
        import h5py
    except ImportError:
        return None
    f = h5py.File(h5_path, "r")
    sd = f["simulation_data"] if "simulation_data" in f else f
    if "mesh" not in sd:
        f.close()
        return None
    mesh = sd["mesh"]
    if "node_coordinates" not in mesh:
        f.close()
        return None
    coords = np.asarray(mesh["node_coordinates"], dtype=np.float64)
    elements = None
    if "element_connectivity" in mesh:
        elements = np.asarray(mesh["element_connectivity"], dtype=np.int64)
    if "steps" not in sd:
        f.close()
        return None
    steps_grp = sd["steps"]
    keys = sorted(k for k in steps_grp.keys() if k.startswith("step_"))
    return coords, elements, keys, steps_grp, f


def compute_morphology_timeseries(
    run_dir: Path,
    *,
    precrack_length_mm: float = PRECRACK_LENGTH_MM,
    l0_mm: float = L0_DEFAULT_MM,
    grid_per_l0: int = MORPH_GRID_PER_L0,
    threshold: float = MORPH_DAMAGE_THRESH,
    min_island: int = MORPH_MIN_ISLAND,
) -> dict | None:
    """Walk the H5 timeseries and compute per-step right-side arm counts.

    Returns a dict with keys:
      - ``t_us``         : (T,) array of times in us
      - ``arm_counts``   : (T,) array of CC counts in x > a + 2*l0
      - ``n_arms_final`` : int, last step's count
      - ``t_branch_us``  : float, first time arm_counts >= 2 (NaN if never)
      - ``D_final``      : (ny, nx) damage grid at the last step
      - ``xs``, ``ys``   : grid axes
      - ``x_min_arm``    : the right-side cutoff used (mm)

    Returns ``None`` if H5 / mesh data is unavailable.
    """
    bundle = _load_mesh_and_steps(run_dir)
    if bundle is None:
        return None
    coords, elements, keys, steps_grp, fh = bundle
    try:
        x_min_arm = float(precrack_length_mm + 2.0 * l0_mm)
        x_lo = float(coords[:, 0].min())
        x_hi = float(coords[:, 0].max())
        y_lo = float(coords[:, 1].min())
        y_hi = float(coords[:, 1].max())
        dx = float(l0_mm) / max(1, int(grid_per_l0))
        t_us_list: list[float] = []
        arm_counts: list[int] = []
        D_final = None
        xs_final = None
        ys_final = None
        for key in keys:
            grp = steps_grp[key]
            if "damage_nodal" not in grp:
                continue
            d = np.asarray(grp["damage_nodal"], dtype=np.float64).ravel()
            if d.shape[0] != coords.shape[0]:
                # Mesh / damage size mismatch — bail rather than mislead.
                continue
            t_us = float(grp.attrs.get("time_s", 0.0)) * 1e6
            xs, ys, D = _grid_damage(
                coords, elements, d,
                x_lo, x_hi, y_lo, y_hi, dx,
            )
            n = count_arms_in_region(
                D, xs, x_min_arm,
                threshold=threshold, min_island=min_island,
            )
            t_us_list.append(t_us)
            arm_counts.append(n)
            D_final = D
            xs_final = xs
            ys_final = ys
        if not t_us_list:
            return None
        t_arr = np.asarray(t_us_list)
        n_arr = np.asarray(arm_counts, dtype=np.int64)
        # First time arms >= 2.
        idx = np.where(n_arr >= 2)[0]
        t_branch_us = float(t_arr[idx[0]]) if idx.size else float("nan")
        return {
            "t_us": t_arr,
            "arm_counts": n_arr,
            "n_arms_final": int(n_arr[-1]),
            "t_branch_us": t_branch_us,
            "D_final": D_final,
            "xs": xs_final,
            "ys": ys_final,
            "x_min_arm": x_min_arm,
        }
    finally:
        fh.close()


def _read_precrack_length(run_dir: Path) -> float:
    """Try ``config.yaml`` then fall back to default 50 mm."""
    cfg = run_dir / "config.yaml"
    if cfg.exists():
        try:
            import yaml
            doc = yaml.safe_load(cfg.read_text())
            a = (doc or {}).get("geometry", {}).get("parameters", {}).get("a")
            if a is not None:
                return float(a)
        except Exception:
            pass
    return PRECRACK_LENGTH_MM


def _read_l0(run_dir: Path) -> float:
    """Try ``run_metadata.json`` material.l0 then ``config.yaml`` then default."""
    meta_path = run_dir / "run_metadata.json"
    if meta_path.exists():
        try:
            import json as _json
            doc = _json.loads(meta_path.read_text())
            v = doc.get("material", {}).get("l0")
            if v is not None:
                return float(v)
        except Exception:
            pass
    cfg = run_dir / "config.yaml"
    if cfg.exists():
        try:
            import yaml
            doc = yaml.safe_load(cfg.read_text())
            v = (doc or {}).get("material", {}).get("overrides", {}).get("l0")
            if v is None:
                v = (doc or {}).get("material", {}).get("l0")
            if v is None:
                v = (doc or {}).get("geometry", {}).get(
                    "parameters", {}).get("l0")
            if v is not None:
                return float(v)
        except Exception:
            pass
    return L0_DEFAULT_MM


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--half-plate", action="store_true",
                    help=("Compare against the half-plate (COMSOL native) "
                          "energy band 0.13-0.14 J instead of the full-plate "
                          "equivalent 0.26-0.28 J. Issue #299."))
    args = ap.parse_args()

    if args.half_plate:
        ref_peak_lo = COMSOL_HALFPLATE_PEAK_J_LO
        ref_peak_hi = COMSOL_HALFPLATE_PEAK_J_HI
    else:
        ref_peak_lo = REF_PEAK_J_LO
        ref_peak_hi = REF_PEAK_J_HI

    run_dir = args.run_dir or find_latest_run_dir(HERE)
    if run_dir is None:
        print(f"FAIL: no run_*/timing_per_step.csv under {HERE} or cwd; "
              f"run B7_dynamic_crack_branching_comsol.yaml first.",
              file=sys.stderr)
        sys.exit(2)
    run_dir = run_dir.resolve()

    t_us, max_d_raw = load_timing(run_dir)

    # Preseed-aware max(d) (issue #213): pf_dirichlet locks notch nodes
    # at d=1 from t=0, so the raw all-node maximum hits 1.0 immediately
    # and the initiation detector reports 0 us. Recompute max(d) over
    # non-preseed nodes from the H5 snapshots when available.
    preseed_idx, preseed_src = _resolve_preseed_node_indices(run_dir)
    excl = load_max_d_excluding_preseed(run_dir, preseed_idx)
    if excl is not None:
        t_us, max_d = excl
        if preseed_idx is None:
            print("WARN: preseed nodeset not found in metadata or H5; "
                  "initiation/full-Y use raw all-node max(d) (may include "
                  "pf_dirichlet-locked notch nodes -- issue #213).",
                  file=sys.stderr)
        else:
            print(f"NOTE: excluded {preseed_idx.size} preseed nodes from "
                  f"max(d) (source: {preseed_src}).")
    else:
        max_d = max_d_raw
        print("WARN: H5 unavailable; falling back to timing_per_step.csv "
              "max(d) (includes preseeded notch nodes -- issue #213).",
              file=sys.stderr)

    # Initiation: first time max_d crosses 0.99 (saturated crack tip)
    above = np.where(max_d > 0.99)[0]
    initiation_us = float(t_us[above[0]]) if above.size else float("nan")

    # Energy-argmax onset (kept for backward compat; see #314 for
    # discussion of why this can be a false positive). The COMSOL
    # reference (Fig 4) shows two elastic-energy peaks; the second
    # *can* correspond to branching, but on simulations where no
    # branch ever forms the late argmax just picks the wave-reflection
    # peak. We therefore now report this alongside the morphology-based
    # ``branching_components_us`` below.
    branching_energy_argmax_us = float("nan")
    energy_data = load_elastic_energy(run_dir)
    if energy_data is not None:
        t_e_us, e_arr = energy_data
        if t_e_us.size >= 5:
            window = max(3, t_e_us.size // 50)
            kernel = np.ones(window) / window
            e_smooth = np.convolve(e_arr, kernel, mode="same")
            early_mask = t_e_us <= 20.0
            if early_mask.any():
                i_first = int(np.argmax(e_smooth[early_mask]))
                t_first = float(t_e_us[early_mask][i_first])
            else:
                t_first = REF_INITIATION_US
            late_mask = t_e_us > (t_first + 5.0)
            if late_mask.any():
                i_late = int(np.argmax(e_smooth[late_mask]))
                branching_energy_argmax_us = float(t_e_us[late_mask][i_late])

    # Morphology-aware Y-detection (issue #314).
    a_mm = _read_precrack_length(run_dir)
    l0_mm = _read_l0(run_dir)
    morph = compute_morphology_timeseries(
        run_dir,
        precrack_length_mm=a_mm,
        l0_mm=l0_mm,
    )
    if morph is None:
        n_arms_final = -1     # unknown
        branching_components_us = float("nan")
        morph_status = "skipped: H5 / mesh unavailable"
    else:
        n_arms_final = morph["n_arms_final"]
        branching_components_us = morph["t_branch_us"]
        morph_status = (
            f"x_min_arm={morph['x_min_arm']:.2f} mm, "
            f"grid={morph['D_final'].shape}, "
            f"n_steps={morph['t_us'].size}")

    # ``branching_us``: prefer the morphology-based detector. If the
    # run lacks H5 we fall back to the energy argmax for backward
    # compatibility (issue #314).
    if not np.isnan(branching_components_us):
        branching_us = branching_components_us
        branch_source = "components"
    else:
        branching_us = branching_energy_argmax_us
        branch_source = "energy_argmax"

    # Elastic-energy peak. Internal units mJ/mm == J/m (see header
    # comment), so the raw H5 value is already in COMSOL's J-at-1m
    # convention; no thickness scaling is applied.
    energy_peak_J = float("nan")
    if energy_data is not None:
        _, e_arr = energy_data
        energy_peak_J = float(e_arr.max()) if e_arr.size else float("nan")

    # Acceptance
    err_branch = (abs(branching_us - REF_BRANCH_US) / REF_BRANCH_US
                  if not np.isnan(branching_us) else float("inf"))
    pass_branch = err_branch <= TOL_BRANCH

    # ``pass_full_y`` (issue #314): require BOTH (a) the simulation ran
    # to t >= 75 us, AND (b) the final damage field has 2+ arms in the
    # right-of-precrack region. Saturated max(d) on a straight crack
    # is no longer enough.
    if morph is None:
        # H5 unavailable: fall back to the legacy max(d)>0.99 test, but
        # mark the result so the report is honest about it.
        pass_full_y = bool(t_us.max() >= REF_FULL_Y_US and (max_d > 0.99).any())
        full_y_label = "max(d) saturated (legacy, no H5)"
    else:
        ran_to_75 = bool(t_us.max() >= REF_FULL_Y_US)
        pass_full_y = bool(ran_to_75 and n_arms_final >= 2)
        full_y_label = (
            f"n_arms_final={n_arms_final}, ran_to_75us={ran_to_75}")

    pass_peak = (ref_peak_lo * (1 - TOL_PEAK)
                 <= energy_peak_J
                 <= ref_peak_hi * (1 + TOL_PEAK)) if not np.isnan(energy_peak_J) else False

    overall = pass_branch and pass_full_y and pass_peak

    n_arms_str = (f"{n_arms_final}"
                  if n_arms_final >= 0 else "unknown (no H5)")
    branch_components_str = (
        f"{branching_components_us:6.2f} us"
        if not np.isnan(branching_components_us) else "  n/a "
    )
    branch_argmax_str = (
        f"{branching_energy_argmax_us:6.2f} us"
        if not np.isnan(branching_energy_argmax_us) else "  n/a "
    )

    report = [
        "Dynamic crack branching (COMSOL 6.4) -- comparison report",
        "=" * 60,
        f"Run dir: {run_dir.name}",
        "",
        f"Initiation     (sim) : {initiation_us:6.2f} us  "
        f"(ref ~ {REF_INITIATION_US} us)",
        f"Branching onset(sim) : {branching_us:6.2f} us  "
        f"(ref {REF_BRANCH_US} us [Ren 2019]; "
        f"COMSOL App. Lib. {REF_BRANCH_US_COMSOL} us [outlier]; "
        f"tol {TOL_BRANCH*100:.0f}%, src={branch_source}) -> "
        f"{'PASS' if pass_branch else 'FAIL'}",
        f"  branching_components : {branch_components_str}  "
        f"(first t with >=2 right-side arms; #314)",
        f"  branching_energy_argmax : {branch_argmax_str}  "
        f"(legacy late-window argmax, kept for compat)",
        f"n_arms_final         : {n_arms_str}  ({morph_status})",
        f"Full-Y by 75us       : {full_y_label} -> "
        f"{'PASS' if pass_full_y else 'FAIL'}",
        f"Elastic peak J (1m)  : {energy_peak_J:.4f} J  "
        f"(ref {ref_peak_lo}-{ref_peak_hi} J"
        f"{' [half-plate]' if args.half_plate else ''}"
        f", tol {TOL_PEAK*100:.0f}%) -> "
        f"{'PASS' if pass_peak else 'FAIL'}",
        "",
        f"OVERALL: {'PASS' if overall else 'FAIL'}",
        "",
        "Crack-path morphology is verified by the connected-component "
        "count of d>0.5 in the right-of-precrack region; see also "
        "damage_final.png against the COMSOL PDF Fig 3 panels at "
        "10/33/45/75 us.",
    ]
    text = "\n".join(report)
    print(text)
    (run_dir / "compare_report.txt").write_text(text + "\n")

    # Plot max_d(t)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(t_us, max_d, "C0-", lw=1.4, label="max(d) -- sim")
    ax.axvline(REF_INITIATION_US, color="green", ls="--", lw=0.8,
               label=f"ref initiation ({REF_INITIATION_US} us)")
    ax.axvline(REF_BRANCH_US, color="red", ls="--", lw=0.8,
               label=f"ref branching ({REF_BRANCH_US} us)")
    ax.axvline(REF_FULL_Y_US, color="purple", ls=":", lw=0.8,
               label=f"ref full-Y ({REF_FULL_Y_US} us)")
    if morph is not None and not np.isnan(branching_components_us):
        ax.axvline(branching_components_us, color="orange", ls="-", lw=0.8,
                   label=f"sim branching CC ({branching_components_us:.1f} us)")
    ax.set_xlabel(r"time $t$ [us]")
    ax.set_ylabel(r"$\max(d)$")
    ax.set_title("Dynamic crack branching -- max(d) vs t")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "compare.png", dpi=200)
    plt.close(fig)

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
