#!/usr/bin/env python3
"""Compare a B5 PMMA dynamic-branching run against the Bleyer 2017
reference values (`reference_solutions/`).

Acceptance criteria (issue #256, formal source ``ACCEPTANCE.md``):

  * Final morphology    : exactly 1 arm for ΔU ≤ 0.038 mm; ≥ 2 arms
                          for ΔU ≥ 0.040 mm (Bleyer Fig 5).
  * Γ/Gc envelope peak  : ΔU = 0.035 → 1.0–1.5; ΔU = 0.045 → ≥ 2.0
                          (Bleyer Fig 6, ±20 %).
  * Peak v / cR         : in [0.30, 0.75], plateau ~0.6 (Bleyer Fig 4
                          / Fig 9, ±20 %).

Reads ``crack_tip.csv`` (crack-tip position + velocity), ``energy.csv``
(elastic / kinetic / fracture totals per step), and ``training_data.h5``
(damage field) from a run directory. Writes ``compare_report.txt``
and ``compare.png`` next to the run.

The morphology counter is the same connected-component / per-column
y-run scheme used in
``examples/dynamic/crack_branching_comsol/compare.py``
(issue #314); reusing rather than reinventing keeps the two
benchmarks' detectors algorithmically identical.

Usage:
    python -m phast.examples.dynamic.branching_pmma.compare \\
        --run-dir <path-to-run-dir> --delta-U 0.040
"""
from __future__ import annotations

import argparse
import csv
import json
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
REF_TABLE = HERE / "reference_solutions" / "bleyer_branching_table.csv"

# Tolerances on the energy and velocity gates (Gate 2 + 3); the
# morphology gate (Gate 1) is binary.
TOL_GAMMA_FRAC = 0.20
TOL_VELOCITY_FRAC = 0.20

# Geometry constants from Bleyer 2017 (also pinned in the YAML
# ``configs/benchmarks/dynamic/B5_pmma_branching.yaml``).
PRECRACK_LENGTH_MM = 4.0
L0_DEFAULT_MM = 0.1
GC_DEFAULT = 0.3            # N / mm

# Morphology constants (mirror dynamic_crack_branching_comsol/compare.py).
MORPH_DAMAGE_THRESH = 0.5
MORPH_MIN_ISLAND = 5
MORPH_GRID_PER_L0 = 2


# ----------------------------------------------------------------------
# Reference loading
# ----------------------------------------------------------------------

def load_reference_table(path: Path = REF_TABLE) -> dict[float, dict]:
    """Return ``{delta_U: row_dict}`` from the Bleyer reference CSV.

    ``row_dict`` keys: ``morphology`` (int), ``gamma_over_gc_lo``,
    ``gamma_over_gc_hi``, ``v_over_cR_peak``, ``note``.
    """
    out: dict[float, dict] = {}
    with path.open() as fh:
        reader = csv.DictReader(fh,
                                fieldnames=None,
                                # skip comment lines beginning with '#'
                                restkey=None,
                                restval="")
        # csv.DictReader does not know about '#' comments; pre-filter.
        rows = []
        for raw in path.read_text().splitlines():
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            rows.append(raw)
        from io import StringIO
        reader = csv.DictReader(StringIO("\n".join(rows)))
        for row in reader:
            try:
                key = float(row["delta_U_mm"])
            except (KeyError, ValueError, TypeError):
                continue
            out[key] = {
                "morphology": int(row["morphology"]),
                "gamma_over_gc_lo": float(row["gamma_over_gc_lo"]),
                "gamma_over_gc_hi": float(row["gamma_over_gc_hi"]),
                "v_over_cR_peak": float(row["v_over_cR_peak"]),
                "note": row.get("note", ""),
            }
    return out


def closest_reference(table: dict[float, dict], delta_U: float
                      ) -> tuple[float, dict]:
    """Pick the table row closest to ``delta_U`` (mm)."""
    if not table:
        raise RuntimeError(f"empty reference table at {REF_TABLE}")
    keys = sorted(table.keys())
    closest = min(keys, key=lambda k: abs(k - delta_U))
    return closest, table[closest]


# ----------------------------------------------------------------------
# Run-dir loading
# ----------------------------------------------------------------------

def find_latest_run_dir(base: Path) -> Path | None:
    """Most recent run dir under ``base`` containing the expected files."""
    candidates = []
    for root in (base, Path.cwd()):
        for p in list(root.glob("run_*")) + list(root.glob("bleyer_*")):
            if (p / "crack_tip.csv").exists() and (p / "energy.csv").exists():
                candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_crack_tip(run_dir: Path) -> np.ndarray | None:
    """Return ``crack_tip.csv`` as structured ndarray, or None on failure."""
    p = run_dir / "crack_tip.csv"
    if not p.exists():
        return None
    try:
        # columns: step,t_us,crack_tip_x_mm,n_crack_tips,crack_vel_mms,
        #          crack_vel_frac_cR,branched
        return np.genfromtxt(p, delimiter=",", names=True, dtype=None,
                             encoding=None)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"WARN: failed to read {p}: {exc}", file=sys.stderr)
        return None


def load_energy(run_dir: Path) -> np.ndarray | None:
    """Return ``energy.csv`` as structured ndarray, or None on failure."""
    p = run_dir / "energy.csv"
    if not p.exists():
        return None
    try:
        return np.genfromtxt(p, delimiter=",", names=True, dtype=None,
                             encoding=None)
    except Exception as exc:  # pragma: no cover
        print(f"WARN: failed to read {p}: {exc}", file=sys.stderr)
        return None


def read_meta(run_dir: Path) -> dict:
    """Best-effort read of run_metadata.json; returns {} on failure."""
    p = run_dir / "run_metadata.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def read_l0(run_dir: Path) -> float:
    meta = read_meta(run_dir)
    v = meta.get("material", {}).get("l0")
    if v is not None:
        try:
            return float(v)
        except Exception:
            pass
    cfg = run_dir / "config.yaml"
    if cfg.exists():
        try:
            import yaml
            doc = yaml.safe_load(cfg.read_text()) or {}
            v = (doc.get("material") or {}).get("l0")
            if v is not None:
                return float(v)
        except Exception:
            pass
    return L0_DEFAULT_MM


# ----------------------------------------------------------------------
# Morphology (lifted/adapted from dynamic_crack_branching_comsol)
# ----------------------------------------------------------------------

def _grid_damage(coords, elements, damage, x_lo, x_hi, y_lo, y_hi, dx):
    """Interpolate node damage onto a uniform grid; same algo as B7."""
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
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)
        flat = np.column_stack([XG.ravel(), YG.ravel()])
        _, idx = tree.query(flat)
        D = damage[idx].reshape(XG.shape)
    D = np.where(np.isnan(D), 0.0, D)
    return xs, ys, D


def count_arms_in_region(D, xs, x_min_arm,
                         threshold=MORPH_DAMAGE_THRESH,
                         min_island=MORPH_MIN_ISLAND):
    """Number of distinct propagating arms in ``x > x_min_arm``."""
    from scipy.ndimage import label
    if D.ndim != 2 or xs.ndim != 1 or xs.size != D.shape[1]:
        raise ValueError("shape mismatch")
    mask = D > threshold
    if not mask.any():
        return 0
    labels, n_lab = label(mask)
    if n_lab == 0:
        return 0
    sizes = np.bincount(labels.ravel())[1:]
    if min_island > 1:
        small = np.where(sizes < min_island)[0] + 1
        if small.size:
            mask[np.isin(labels, small)] = False
    if not mask.any():
        return 0
    col_mask = xs > x_min_arm
    if not col_mask.any():
        return 0
    sub = mask[:, col_mask]
    if not sub.any():
        return 0
    cols = sub.astype(np.int8)
    starts = cols[0:1, :]
    transitions = (cols[1:, :] - cols[:-1, :]) == 1
    runs_per_col = starts.sum(axis=0) + transitions.sum(axis=0)
    return int(runs_per_col.max())


def final_arm_count(run_dir: Path, l0_mm: float) -> tuple[int, str]:
    """Return ``(n_arms_final, status)``. Reads the LAST damage_nodal step."""
    h5_path = run_dir / "training_data.h5"
    if not h5_path.exists():
        return -1, "skipped: training_data.h5 not present"
    try:
        import h5py
    except ImportError:
        return -1, "skipped: h5py not available"
    try:
        with h5py.File(h5_path, "r") as f:
            sd = f["simulation_data"] if "simulation_data" in f else f
            if "mesh" not in sd:
                return -1, "skipped: mesh group missing in H5"
            mesh = sd["mesh"]
            if "nodes" not in mesh:
                return -1, "skipped: mesh/nodes missing"
            coords = np.asarray(mesh["nodes"], dtype=np.float64)
            if coords.shape[1] >= 2:
                coords = coords[:, :2]
            elements = None
            if "elements" in mesh:
                elements = np.asarray(mesh["elements"], dtype=np.int64)
                if elements.ndim == 2 and elements.shape[1] > 3:
                    elements = elements[:, :3]
            steps = sd.get("steps", sd)
            step_keys = sorted(
                k for k in steps.keys() if k.startswith("step_"))
            if not step_keys:
                return -1, "skipped: no step_* groups in H5"
            last = steps[step_keys[-1]]
            if "damage_nodal" not in last:
                return -1, "skipped: damage_nodal missing in final step"
            d = np.asarray(last["damage_nodal"], dtype=np.float64).ravel()
        x_min_arm = PRECRACK_LENGTH_MM + 2.0 * l0_mm
        x_lo, x_hi = float(coords[:, 0].min()), float(coords[:, 0].max())
        y_lo, y_hi = float(coords[:, 1].min()), float(coords[:, 1].max())
        dx = max(l0_mm / MORPH_GRID_PER_L0, 1e-3)
        xs, ys, D = _grid_damage(coords, elements, d,
                                 x_lo, x_hi, y_lo, y_hi, dx)
        n = count_arms_in_region(D, xs, x_min_arm)
        return n, f"x_min_arm={x_min_arm:.2f} mm, grid={D.shape}"
    except Exception as exc:
        return -1, f"morphology error: {exc!r}"


# ----------------------------------------------------------------------
# Γ / Gc and v / cR aggregation
# ----------------------------------------------------------------------

def gamma_over_gc_peak(crack_tip: np.ndarray, energy: np.ndarray,
                       gc: float = GC_DEFAULT) -> float:
    """Estimate peak Γ/Gc during the single-tip phase.

    Γ(t) = d(E_frac)/d(l), where l(t) = crack_tip_x − a is the apparent
    crack length. We compute Γ = d(E_frac)/d(t) divided by d(l)/d(t)
    using a smoothed finite difference on shared timestamps, then mask
    to single-tip windows (``branched == 0`` in crack_tip.csv) and take
    the trailing-window max.

    Returns ``nan`` on failure or insufficient data.
    """
    if crack_tip is None or energy is None:
        return float("nan")
    if "t_us" not in crack_tip.dtype.names:
        return float("nan")
    if "fracture" not in energy.dtype.names or "t_s" not in energy.dtype.names:
        return float("nan")
    t_us_ct = crack_tip["t_us"].astype(np.float64)
    x_mm = crack_tip["crack_tip_x_mm"].astype(np.float64)
    branched = (crack_tip["branched"].astype(np.int64)
                if "branched" in crack_tip.dtype.names
                else np.zeros_like(x_mm, dtype=np.int64))
    t_us_e = (energy["t_s"].astype(np.float64) * 1e6)
    e_frac = energy["fracture"].astype(np.float64)
    if t_us_ct.size < 5 or t_us_e.size < 5:
        return float("nan")
    # Interpolate E_frac onto crack_tip timestamps, then dE/dl ~ dE/dt
    # divided by dl/dt computed on the same grid.
    e_on_ct = np.interp(t_us_ct, t_us_e, e_frac)
    l_mm = x_mm - PRECRACK_LENGTH_MM
    # Smooth with a small box kernel.
    w = max(3, t_us_ct.size // 20)
    kern = np.ones(w) / w
    e_s = np.convolve(e_on_ct, kern, mode="same")
    l_s = np.convolve(l_mm,    kern, mode="same")
    dE = np.gradient(e_s, t_us_ct)
    dL = np.gradient(l_s, t_us_ct)
    eps = 1e-9
    gamma = np.where(dL > eps, dE / np.maximum(dL, eps), np.nan)
    gamma_over_gc = gamma / gc
    # Mask: drop branched samples and the leading-edge garbage (l < 2 l0).
    mask = (branched == 0) & (l_s > 0.2)
    if not mask.any():
        return float("nan")
    valid = gamma_over_gc[mask]
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return float("nan")
    return float(np.nanpercentile(valid, 95))   # 95th-percentile peak


def velocity_peak(crack_tip: np.ndarray) -> float:
    """Return smoothed peak ``crack_vel_frac_cR``.

    Raw tip velocity from finite differencing of element-aligned
    crack-tip positions is noisy (single-step jumps of one element
    spike to >1 cR). Bleyer 2017 Figs 4/9 plot smoothed values that
    plateau near 0.6 cR with a hard limit ~0.75. We emulate that by
    boxcar-smoothing with window = max(5, N/40), then taking the 95th
    percentile of the smoothed signal in the single-tip phase.
    """
    if crack_tip is None:
        return float("nan")
    if "crack_vel_frac_cR" not in crack_tip.dtype.names:
        return float("nan")
    v = crack_tip["crack_vel_frac_cR"].astype(np.float64)
    branched = (crack_tip["branched"].astype(np.int64)
                if "branched" in crack_tip.dtype.names
                else np.zeros_like(v, dtype=np.int64))
    mask = (branched == 0) & np.isfinite(v) & (v > 0.0)
    if mask.sum() < 5:
        return float("nan")
    v_clean = v[mask]
    w = max(5, v_clean.size // 40)
    kern = np.ones(w) / w
    v_smooth = np.convolve(v_clean, kern, mode="same")
    return float(np.nanpercentile(v_smooth, 95))


# ----------------------------------------------------------------------
# Gate evaluation
# ----------------------------------------------------------------------

def gate_morphology(n_arms: int, expected: int) -> tuple[bool, str]:
    if n_arms < 0:
        return False, "morphology unavailable"
    if expected == 1:
        ok = n_arms == 1
        return ok, f"n_arms={n_arms} (expected 1)"
    # Branching expected.
    ok = n_arms >= 2
    return ok, f"n_arms={n_arms} (expected ≥ 2)"


def gate_gamma(observed: float, lo: float, hi: float,
               tol: float = TOL_GAMMA_FRAC) -> tuple[bool, str]:
    if not np.isfinite(observed):
        return False, "Γ/Gc unavailable"
    lo_t = lo * (1 - tol)
    hi_t = hi * (1 + tol)
    ok = lo_t <= observed <= hi_t
    return ok, (f"Γ/Gc = {observed:.2f} (ref [{lo:.2f}, {hi:.2f}],"
                f" tol ±{tol*100:.0f}% → [{lo_t:.2f}, {hi_t:.2f}])")


def gate_velocity(observed: float, ref: float,
                  tol: float = TOL_VELOCITY_FRAC) -> tuple[bool, str]:
    if not np.isfinite(observed):
        return False, "v/cR unavailable"
    lo_t = max(0.0, ref - 0.30)        # absolute floor: 0.30
    hi_t = min(0.75, ref + 0.15)       # absolute cap : 0.75
    ok = lo_t <= observed <= hi_t
    return ok, (f"v/cR_peak = {observed:.3f} (ref ~{ref:.2f},"
                f" window [{lo_t:.2f}, {hi_t:.2f}])")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Path to run directory (default: latest under HERE)")
    ap.add_argument("--delta-U", type=float, required=False, default=None,
                    help="Pre-strain ΔU in mm (e.g. 0.035, 0.040, 0.045). "
                         "If omitted, read from run_metadata.json.")
    args = ap.parse_args()

    run_dir = args.run_dir or find_latest_run_dir(HERE)
    if run_dir is None:
        print(f"FAIL: no run dir with crack_tip.csv + energy.csv under "
              f"{HERE} or cwd.", file=sys.stderr)
        sys.exit(2)
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        print(f"FAIL: run dir does not exist: {run_dir}", file=sys.stderr)
        sys.exit(2)

    # Resolve ΔU.
    delta_U = args.delta_U
    if delta_U is None:
        meta = read_meta(run_dir)
        delta_U = (meta.get("loading", {}).get("prestrain_displacement")
                   or meta.get("delta_U"))
        if delta_U is not None:
            delta_U = float(delta_U)
    if delta_U is None:
        # Last-resort: parse run-dir name (e.g. "bleyer_a_dU0.035").
        name = run_dir.name
        if "dU" in name:
            try:
                delta_U = float(name.split("dU")[1].split("_")[0])
            except Exception:
                delta_U = None
    if delta_U is None:
        print("FAIL: could not determine ΔU; pass --delta-U explicitly.",
              file=sys.stderr)
        sys.exit(2)

    table = load_reference_table()
    ref_dU, ref_row = closest_reference(table, delta_U)

    # Load run data.
    crack_tip = load_crack_tip(run_dir)
    energy = load_energy(run_dir)
    l0_mm = read_l0(run_dir)

    # Compute observables.
    n_arms, morph_status = final_arm_count(run_dir, l0_mm)
    gamma_peak = gamma_over_gc_peak(crack_tip, energy)
    v_peak = velocity_peak(crack_tip)

    # Gate evaluation.
    pass_morph, txt_morph = gate_morphology(n_arms, ref_row["morphology"])
    pass_gamma, txt_gamma = gate_gamma(
        gamma_peak,
        ref_row["gamma_over_gc_lo"],
        ref_row["gamma_over_gc_hi"])
    pass_vel, txt_vel = gate_velocity(v_peak, ref_row["v_over_cR_peak"])

    # Overall: when morphology is available it must pass. When the H5 is
    # not present (n_arms < 0) we mark Gate 1 as ``UNKNOWN`` and require
    # Gates 2 + 3 to BOTH pass for an overall PASS. With a present H5,
    # the rule is morphology pass AND at least one of Γ/v passes.
    morph_unknown = n_arms < 0
    if morph_unknown:
        overall = pass_gamma and pass_vel
        morph_label = "UNKNOWN"
    else:
        overall = pass_morph and (pass_gamma or pass_vel)
        morph_label = "PASS" if pass_morph else "FAIL"

    report = [
        "B5 PMMA dynamic branching (Bleyer 2017) — comparison report",
        "=" * 60,
        f"Run dir   : {run_dir.name}",
        f"ΔU (req)  : {delta_U:.4f} mm",
        f"Ref ΔU    : {ref_dU:.3f} mm  ({ref_row['note']})",
        f"l0 (run)  : {l0_mm:.3f} mm",
        "",
        "Gate 1 — morphology  (Bleyer Fig 5):",
        f"  {txt_morph}  ({morph_status}) -> {morph_label}",
        "",
        "Gate 2 — Γ / Gc peak (Bleyer Fig 6):",
        f"  {txt_gamma} -> {'PASS' if pass_gamma else 'FAIL'}",
        "",
        "Gate 3 — v / cR peak (Bleyer Fig 4 / Fig 9):",
        f"  {txt_vel} -> {'PASS' if pass_vel else 'FAIL'}",
        "",
        f"OVERALL: {'PASS' if overall else 'FAIL'}",
        "  Rule: with H5, Gate 1 must PASS plus at least one of Gate 2/3.",
        "  Without H5 (Gate 1 UNKNOWN), Gates 2+3 must BOTH PASS.",
        "",
        "Source figures: refs/Bleyer (2017), pages 88-89, 91.",
    ]
    text = "\n".join(report)
    print(text)
    (run_dir / "compare_report.txt").write_text(text + "\n")

    # Plot v(t) and Γ/Gc evolution.
    try:
        if crack_tip is not None and "t_us" in crack_tip.dtype.names:
            t_us = crack_tip["t_us"].astype(np.float64)
            v = crack_tip["crack_vel_frac_cR"].astype(np.float64)
            fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
            axes[0].plot(t_us, v, "C0-", lw=1.4)
            axes[0].axhline(ref_row["v_over_cR_peak"], color="red", ls="--",
                            lw=0.8,
                            label=f"ref ~{ref_row['v_over_cR_peak']:.2f}")
            axes[0].axhline(0.75, color="purple", ls=":", lw=0.7,
                            label="limit ~0.75 cR")
            axes[0].set_xlabel(r"$t$ [µs]")
            axes[0].set_ylabel(r"$v / c_R$")
            axes[0].set_title(rf"crack-tip velocity (Bleyer Fig 4),"
                              rf" $\Delta U = {delta_U:.3f}$ mm")
            axes[0].grid(alpha=0.3)
            axes[0].legend(fontsize=8)

            # Γ/Gc vs crack position.
            if energy is not None and "fracture" in energy.dtype.names:
                e_frac = energy["fracture"].astype(np.float64)
                t_e = energy["t_s"].astype(np.float64) * 1e6
                e_on_ct = np.interp(t_us, t_e, e_frac)
                x = crack_tip["crack_tip_x_mm"].astype(np.float64)
                l = x - PRECRACK_LENGTH_MM
                w = max(3, t_us.size // 20)
                kern = np.ones(w) / w
                de = np.gradient(np.convolve(e_on_ct, kern, mode="same"),
                                 t_us)
                dl = np.gradient(np.convolve(l, kern, mode="same"), t_us)
                gam_t = np.where(dl > 1e-9, de / np.maximum(dl, 1e-9),
                                 np.nan) / GC_DEFAULT
                axes[1].plot(x, gam_t, "C2-", lw=1.4)
                axes[1].axhline(1.0, color="black", ls=":", lw=0.6,
                                label=r"$\Gamma = G_c$")
                axes[1].axhline(2.0, color="red", ls="--", lw=0.7,
                                label=r"branching threshold $2 G_c$")
                axes[1].set_ylim(0, 4)
                axes[1].set_xlabel(r"$x$ [mm]")
                axes[1].set_ylabel(r"$\Gamma / G_c$")
                axes[1].set_title(r"normalised dissipation rate"
                                  r" (Bleyer Fig 6)")
                axes[1].grid(alpha=0.3)
                axes[1].legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(run_dir / "compare.png", dpi=200)
            plt.close(fig)
    except Exception as exc:  # pragma: no cover
        print(f"WARN: plot generation failed: {exc!r}", file=sys.stderr)

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
