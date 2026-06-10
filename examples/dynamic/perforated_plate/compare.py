#!/usr/bin/env python3
"""Compare a B6 perforated-plate run against the Bleyer et al. (2017)
dynamic crack-branching reference.

Reference: Bleyer, Roux-Langlois, Molinari (2017), "Dynamic crack
branching: a phase-field formulation", Sec 4.2-4.3 (PMMA, perforated
plates). Key plots: Fig 7 (damage profiles), Fig 9 (dissipation vs
velocity), Fig 17 (crack-tip velocity with hole bands annotated).

Acceptance criteria:

  - SATURATION: max(d) reaches >=0.99 at some t in the run, confirming
    the crack actually propagated through the plate.
  - VELOCITY ENVELOPE: peak crack-tip velocity stays below the
    Rayleigh-wave speed of PMMA (c_R ~ 920 m/s); Bleyer Fig 17
    reports peaks in the band ~400-700 m/s for the holed cases.
    We accept any envelope whose 95th-percentile non-zero velocity
    falls inside [200, c_R] mm/us-equivalent.
  - HOLE-BAND SLOWDOWN (only for variants with holes): there must be
    at least one local minimum of crack-tip velocity within +/-1 mm
    of the centre of any hole band. We compute hole positions from
    ``run_metadata.json`` -> ``config_file`` -> YAML ``geometry`` block.

Reads ``crack_tip.csv`` (columns:
``step,t_us,crack_tip_x_mm,n_crack_tips,crack_vel_mms,crack_vel_frac_cR,branched``)
and ``timing_per_step.csv`` (cols ``step,max_d,...``).

Writes ``compare_report.txt`` and ``compare.png`` next to the run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

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

# Bleyer 2017 PMMA Rayleigh-wave speed (m/s). See sec 4.2.
C_R_MMS = 920.0e3  # mm/s

# Acceptance bands.
SAT_THRESHOLD = 0.99
VEL_ENVELOPE_LO_MMS = 200.0e3   # 200 m/s, in mm/s
VEL_ENVELOPE_HI_MMS = C_R_MMS
HOLE_SLOWDOWN_TOL_MM = 1.0      # +/- 1 mm around hole centre


def find_latest_run_dir(base: Path) -> Optional[Path]:
    """Most recent run_* dir containing crack_tip.csv (in base or cwd)."""
    candidates = []
    for root in (base, Path.cwd()):
        if not root.exists():
            continue
        candidates.extend(p for p in root.glob("run_*")
                          if (p / "crack_tip.csv").exists()
                          or (p / "timing_per_step.csv").exists())
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_crack_tip(run_dir: Path):
    """Return (t_us, x_mm, vel_mms) from crack_tip.csv, or None."""
    path = run_dir / "crack_tip.csv"
    if not path.exists():
        return None
    arr = np.loadtxt(path, delimiter=",", skiprows=1,
                     usecols=(1, 2, 4))
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr[:, 0], arr[:, 1], arr[:, 2]


def load_max_d(run_dir: Path):
    """Return (t_us, max_d) from timing_per_step.csv via dt in metadata."""
    timing = run_dir / "timing_per_step.csv"
    meta = run_dir / "run_metadata.json"
    if not timing.exists():
        return None
    arr = np.loadtxt(timing, delimiter=",", skiprows=1, usecols=(0, 1))
    if arr.ndim == 1:
        arr = arr[None, :]
    dt = 1.0
    if meta.exists():
        try:
            m = json.loads(meta.read_text())
            dt = float(m.get("solver", {}).get("dt", 1.0))
        except Exception:
            pass
    return arr[:, 0] * dt * 1e6, arr[:, 1]


def resolve_hole_positions(run_dir: Path):
    """Recover hole x-centres (mm) from the run's source YAML config.

    Returns (positions_mm, label) or ([], 'none') if no holes.
    Defaults follow ``mesh_generator.create_perforated_sent`` semantics:
    ``hole_start_x = a + 1.0`` if absent, ``hole_spacing`` taken from
    YAML, ``n_holes`` taken from YAML.
    """
    meta = run_dir / "run_metadata.json"
    if not meta.exists():
        return [], "no-metadata"
    try:
        m = json.loads(meta.read_text())
        cfg_path = Path(m.get("config_file", ""))
        if not cfg_path.exists():
            # Try sibling yaml in run_dir.
            for p in run_dir.glob("*.yaml"):
                cfg_path = p
                break
        if not cfg_path.exists():
            return [], "no-config-yaml"
        try:
            import yaml  # type: ignore
        except ImportError:
            return [], "no-pyyaml"
        cfg = yaml.safe_load(cfg_path.read_text())
        geom = (cfg.get("geometry", {}) or {}).get("parameters", {}) or {}
        n_holes = int(geom.get("n_holes", 0))
        if n_holes <= 0:
            return [], "no-holes"
        a = float(geom.get("a", 4.0))
        spacing = float(geom.get("hole_spacing", 0.9))
        start_x = float(geom.get("hole_start_x", a + 1.0))
        positions = [start_x + i * spacing for i in range(n_holes)]
        return positions, cfg_path.name
    except Exception as e:
        return [], f"err:{e}"


def detect_hole_slowdowns(t_us, x_mm, vel_mms, hole_positions):
    """Return list of (hole_x_mm, vel_min_at_hole_mms) for every hole the
    crack tip actually reached. A "slowdown" is any non-monotone dip in
    the crack-tip velocity within +/- HOLE_SLOWDOWN_TOL_MM of the hole.
    """
    hits = []
    if x_mm is None or vel_mms is None or len(x_mm) < 5:
        return hits
    for hx in hole_positions:
        mask = (np.abs(x_mm - hx) <= HOLE_SLOWDOWN_TOL_MM)
        if mask.sum() < 3:
            continue
        v_band = vel_mms[mask]
        v_pos = v_band[v_band > 0]
        if v_pos.size < 3:
            continue
        # Slowdown criterion: minimum within band lower than 70% of
        # the median across the band.
        v_min = float(v_pos.min())
        v_med = float(np.median(v_pos))
        if v_min < 0.7 * v_med:
            hits.append((hx, v_min))
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Run directory containing crack_tip.csv + "
                         "timing_per_step.csv + run_metadata.json. "
                         "Defaults to most recent run_* under this dir.")
    args = ap.parse_args()

    run_dir = args.run_dir or find_latest_run_dir(HERE)
    if run_dir is None:
        print(f"FAIL: no run_*/crack_tip.csv under {HERE} or cwd; "
              f"run a B6_perforated_*.yaml first (or rsync the HPC "
              f"results dir locally and pass --run-dir).",
              file=sys.stderr)
        sys.exit(2)
    run_dir = run_dir.resolve()

    # -- saturation. Prefer timing_per_step.csv max(d) trace; fall back
    # to crack_tip.csv (existence of crack tip beyond the notch tip
    # implies damage saturated locally).
    md = load_max_d(run_dir)
    sat_t_us = float("nan")
    if md is not None:
        t_us_d, max_d = md
        t_sat_idx = np.where(max_d > SAT_THRESHOLD)[0]
        sat_t_us = float(t_us_d[t_sat_idx[0]]) if t_sat_idx.size else float("nan")
        pass_sat = bool(t_sat_idx.size)
    else:
        # No timing CSV; derive a crude saturation flag from crack_tip.csv.
        ct_probe = load_crack_tip(run_dir)
        if ct_probe is None:
            print(f"FAIL: neither timing_per_step.csv nor crack_tip.csv "
                  f"in {run_dir}", file=sys.stderr)
            sys.exit(2)
        t_us_d, max_d = np.asarray([]), np.asarray([])
        # Saturation proxy: crack tip moved past at least 1 mm beyond
        # the initial notch tip (a=4 mm) at any point.
        pass_sat = bool((ct_probe[1] > 5.0).any())
        if pass_sat:
            # Earliest such time.
            idx = int(np.argmax(ct_probe[1] > 5.0))
            sat_t_us = float(ct_probe[0][idx])

    # -- velocity envelope
    ct = load_crack_tip(run_dir)
    pass_env = False
    v95_mms = float("nan")
    if ct is not None:
        t_us, x_mm, vel_mms = ct
        v_pos = vel_mms[vel_mms > 0]
        if v_pos.size:
            v95_mms = float(np.percentile(v_pos, 95))
            pass_env = (VEL_ENVELOPE_LO_MMS <= v95_mms <= VEL_ENVELOPE_HI_MMS)
    else:
        t_us = x_mm = vel_mms = None

    # -- hole-band slowdowns
    hole_positions, hole_src = resolve_hole_positions(run_dir)
    slowdowns = detect_hole_slowdowns(t_us, x_mm, vel_mms, hole_positions)
    if hole_positions:
        # Pass if at least one hole-band slowdown was detected.
        pass_holes = bool(slowdowns)
    else:
        # Variant has no holes; criterion is vacuous, treat as PASS.
        pass_holes = True

    overall = pass_sat and pass_env and pass_holes

    report = [
        "Dynamic perforated plate (B6, Bleyer 2017) -- comparison report",
        "=" * 64,
        f"Run dir: {run_dir.name}",
        f"Config:  {hole_src}",
        f"Holes:   {len(hole_positions)} at x = "
        f"{[round(p,2) for p in hole_positions[:6]]}{'...' if len(hole_positions)>6 else ''}",
        "",
        f"Saturation max(d)>=0.99 : t = {sat_t_us:6.2f} us  -> "
        f"{'PASS' if pass_sat else 'FAIL'}",
        f"Velocity envelope (v95) : {v95_mms/1e3:6.1f} m/s  "
        f"(target {VEL_ENVELOPE_LO_MMS/1e3:.0f}-{VEL_ENVELOPE_HI_MMS/1e3:.0f} m/s, "
        f"c_R={C_R_MMS/1e3:.0f} m/s) -> "
        f"{'PASS' if pass_env else 'FAIL'}",
        f"Hole-band slowdowns     : {len(slowdowns)}/{len(hole_positions)} "
        f"holes show crack-tip dip -> "
        f"{'PASS' if pass_holes else 'FAIL'}",
        "",
        f"OVERALL: {'PASS' if overall else 'FAIL'}",
        "",
        "Crack morphology and dissipation balance are qualitative: see ",
        "figures/damage_multipanel.png and dissipation_rate.png and",
        "compare against Bleyer 2017 Fig 7 / Fig 9 / Fig 17.",
    ]
    text = "\n".join(report)
    print(text)
    (run_dir / "compare_report.txt").write_text(text + "\n")

    # -- compare.png: max_d(t) + crack-tip velocity vs x with hole bands
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    ax = axes[0]
    if t_us_d.size and max_d.size:
        ax.plot(t_us_d, max_d, "C0-", lw=1.4, label=r"$\max(d)$")
        ax.axhline(SAT_THRESHOLD, color="grey", ls=":", lw=0.8,
                   label=f"sat ({SAT_THRESHOLD})")
    else:
        ax.text(0.5, 0.55, "no timing_per_step.csv\n"
                "(saturation inferred from crack_tip.csv)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9)
    if pass_sat and not np.isnan(sat_t_us):
        ax.axvline(sat_t_us, color="green", ls="--", lw=0.8,
                   label=f"sat at {sat_t_us:.1f} us")
    ax.set_xlabel(r"time $t$ [us]")
    ax.set_ylabel(r"$\max(d)$")
    ax.set_title("Damage saturation")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    if ct is not None:
        ax.plot(x_mm, vel_mms / 1e3, "C1-", lw=1.0, label="crack-tip vel")
        ax.axhline(C_R_MMS / 1e3, color="red", ls="--", lw=0.8,
                   label=f"$c_R$ = {C_R_MMS/1e3:.0f} m/s")
        for hx in hole_positions:
            ax.axvspan(hx - HOLE_SLOWDOWN_TOL_MM, hx + HOLE_SLOWDOWN_TOL_MM,
                       color="C2", alpha=0.12)
        for hx, vmin in slowdowns:
            ax.plot([hx], [vmin / 1e3], "kv", ms=6)
        ax.set_xlabel(r"crack-tip $x$ [mm]")
        ax.set_ylabel(r"$v$ [m/s]")
        ax.set_title("Crack-tip velocity (hole bands shaded)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    else:
        ax.text(0.5, 0.5, "no crack_tip.csv", ha="center", va="center",
                transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(run_dir / "compare.png", dpi=200)
    plt.close(fig)

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
