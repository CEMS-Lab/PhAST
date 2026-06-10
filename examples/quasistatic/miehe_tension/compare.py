#!/usr/bin/env python3
"""Compare quasi-static SENT (Miehe tension) run against PhaseFieldX 1711.

Loads the most recent ``run_*`` directory's ``results.csv`` from this
benchmark folder (or a path passed via ``--run-dir``) and the reference
load-displacement curve under ``reference_solutions/``. Computes peak
reaction force, the displacement at peak, the relative L2 error of the
load-displacement curve interpolated onto the reference grid, the
pre-peak L2 (overlap segment up to u_at_peak_ref) and the dissipated-
energy ratio (trapezoidal integral of F-vs-u over the simulated range).

Acceptance criteria (issue #256/#132 readiness gates):
  * peak-reaction relative error <= 5%               (gate active)
  * pre-peak load-displacement L2 <= 10%             (gate active)
  * dissipated-energy relative error <= 15%          (gate active)
  * post-peak L2 (u > u_at_peak_ref)                 (reported only;
       opt in to gating with --strict-postpeak; uses L2_PREPEAK_TOL)

The full-curve L2 is reported INFO-only; pure displacement-controlled
Newton cannot trace the unstable branch (dF/du < 0), so a full-curve
L2 gate would penalise a physically-correct solver simply because the
loading protocol cannot reach the negative-slope branch.

Writes ``compare_report.txt`` and ``compare.png`` into the chosen run
directory and prints a one-line PASS/FAIL summary.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# NumPy >=2.0 renamed trapz -> trapezoid; HPC env may still be on 1.x.
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
if _trapezoid is None:  # pragma: no cover - only possible on unusual NumPy builds
    raise RuntimeError("NumPy provides neither trapezoid nor trapz")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# STIX serif + stix mathtext for paper-quality figures.
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
REF = HERE / "reference_solutions" / "miehe_sent_load_displacement.csv"

CANONICAL_REFERENCE_RUNS = [
    HERE / "reference_runs" / "qs_sent_41278_medium" / "run",
    HERE / "reference_runs" / "qs_sent_41278_coarse" / "run",
    HERE / "reference_runs" / "qs_sent_42928_medium" / "run",
    HERE / "reference_runs" / "qs_sent_37992" / "run",
    HERE / "hpc_results" / "job32462_qs_sent" / "run",
]

# Tolerances (issue #256/#132 -- post-peak readiness).
PEAK_TOL = 0.05         # peak-reaction relative error
L2_PREPEAK_TOL = 0.10   # pre-peak segment (u <= u_at_peak_ref)
ENERGY_TOL = 0.15       # dissipated energy = integral F du over overlap


def find_latest_run_dir(base: Path) -> Path | None:
    candidates = sorted(
        [p for p in base.glob("run_*") if (p / "results.csv").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def find_default_reference_run() -> Path | None:
    for run_dir in CANONICAL_REFERENCE_RUNS:
        if (run_dir / "results.csv").exists():
            return run_dir
    return None


def _load_results_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    # Some results.csv files have a trailing energies block prefixed with '#'.
    rows = []
    with csv_path.open() as fh:
        header = None
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if header is None:
                header = parts
                continue
            try:
                rows.append([float(x) for x in parts[: len(header)]])
            except ValueError:
                continue
    if not rows:
        raise RuntimeError(f"No data rows in {csv_path}")
    arr = np.asarray(rows)
    cols = {name: i for i, name in enumerate(header)}
    disp = arr[:, cols["displacement"]]
    react = np.abs(arr[:, cols["reaction_kN"]])
    return disp, react


def _load_history_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    with csv_path.open() as fh:
        header = None
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if header is None:
                header = parts
                continue
            try:
                rows.append([float(x) for x in parts[: len(header)]])
            except ValueError:
                continue
    if not rows:
        raise RuntimeError(f"No data rows in {csv_path}")
    arr = np.asarray(rows)
    cols = {name: i for i, name in enumerate(header)}
    disp = arr[:, cols["applied_disp"]]
    react = np.abs(arr[:, cols["reaction_force"]]) / 1000.0
    return disp, react


def load_run_results(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (displacement_mm, |reaction_kN|) from a run CSV."""
    results = run_dir / "results.csv"
    history = run_dir / "history.csv"
    if results.exists():
        try:
            return _load_results_csv(results)
        except RuntimeError:
            if not history.exists():
                raise
    if history.exists():
        return _load_history_csv(history)
    raise FileNotFoundError(f"No results.csv or history.csv under {run_dir}")


def load_reference() -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(REF)
    return data[:, 0], data[:, 1]


def relative_l2_error(u_ref, R_ref, u_sim, R_sim,
                      u_lo=None, u_hi=None) -> float:
    """L2 error of R_sim interpolated onto u_ref, normalised by ||R_ref||.

    Optionally restrict to ``[u_lo, u_hi]`` (default: full overlap with
    the simulated range).
    """
    lo = max(u_sim.min(), u_lo if u_lo is not None else -np.inf)
    hi = min(u_sim.max(), u_hi if u_hi is not None else np.inf)
    mask = (u_ref >= lo) & (u_ref <= hi)
    if mask.sum() < 4:
        return float("nan")
    R_sim_on_ref = np.interp(u_ref[mask], u_sim, R_sim)
    diff = R_sim_on_ref - R_ref[mask]
    denom = np.linalg.norm(R_ref[mask])
    return float(np.linalg.norm(diff) / denom) if denom > 0 else float("nan")


def dissipated_energy_error(u_ref, R_ref, u_sim, R_sim) -> tuple[float, float, float]:
    """Trapezoidal energy integral over the overlapping displacement range.

    Returns ``(W_sim, W_ref, rel_err)`` where ``W = integral F du`` over
    ``[max(u_min), min(u_max)]``. Both curves are interpolated onto a
    union grid of their displacement points within the overlap, which
    handles the irregular spacing of the digitised reference.
    """
    lo = max(u_ref.min(), u_sim.min())
    hi = min(u_ref.max(), u_sim.max())
    if hi <= lo:
        return float("nan"), float("nan"), float("nan")
    ug = np.unique(np.concatenate([
        u_ref[(u_ref >= lo) & (u_ref <= hi)],
        u_sim[(u_sim >= lo) & (u_sim <= hi)],
    ]))
    if ug.size < 4:
        return float("nan"), float("nan"), float("nan")
    R_ref_g = np.interp(ug, u_ref, R_ref)
    R_sim_g = np.interp(ug, u_sim, R_sim)
    W_ref = float(_trapezoid(R_ref_g, ug))
    W_sim = float(_trapezoid(R_sim_g, ug))
    rel_err = abs(W_sim - W_ref) / W_ref if W_ref > 0 else float("nan")
    return W_sim, W_ref, rel_err


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Specific run_* directory; default = most recent.")
    ap.add_argument("--strict-postpeak", action="store_true",
                    help="Also gate on post-peak L2 with L2_PREPEAK_TOL.")
    args = ap.parse_args()

    run_dir = args.run_dir
    if run_dir is None:
        run_dir = find_default_reference_run()
    if run_dir is None:
        run_dir = find_latest_run_dir(HERE)
    if run_dir is not None:
        run_dir = run_dir.resolve()
    if run_dir is None or not (run_dir / "results.csv").exists():
        print("FAIL: no run directory with results.csv found under "
              f"{HERE}; run run.py first.", file=sys.stderr)
        sys.exit(2)

    u_sim, R_sim = load_run_results(run_dir)
    u_ref, R_ref = load_reference()

    peak_sim = float(R_sim.max())
    peak_ref = float(R_ref.max())
    u_at_peak_sim = float(u_sim[int(np.argmax(R_sim))])
    u_at_peak_ref = float(u_ref[int(np.argmax(R_ref))])
    peak_err = abs(peak_sim - peak_ref) / peak_ref
    l2_full = relative_l2_error(u_ref, R_ref, u_sim, R_sim)
    l2_pre = relative_l2_error(u_ref, R_ref, u_sim, R_sim,
                               u_hi=u_at_peak_ref)
    sim_reached_peak = u_sim.max() >= u_at_peak_ref
    # Post-peak L2 only meaningful if the simulation extended past u_peak.
    if sim_reached_peak and u_sim.max() > u_at_peak_ref:
        l2_post = relative_l2_error(u_ref, R_ref, u_sim, R_sim,
                                    u_lo=u_at_peak_ref)
    else:
        l2_post = float("nan")
    W_sim, W_ref, energy_err = dissipated_energy_error(
        u_ref, R_ref, u_sim, R_sim)

    pass_peak = sim_reached_peak and peak_err <= PEAK_TOL
    pass_l2_pre = (not np.isnan(l2_pre)) and l2_pre <= L2_PREPEAK_TOL
    pass_energy = (not np.isnan(energy_err)) and energy_err <= ENERGY_TOL
    if args.strict_postpeak:
        pass_l2_post = (not np.isnan(l2_post)) and l2_post <= L2_PREPEAK_TOL
    else:
        pass_l2_post = True
    overall_pass = pass_peak and pass_l2_pre and pass_energy and pass_l2_post

    report = [
        "Miehe SENT (PhaseFieldX 1711) -- comparison report",
        "=" * 56,
        f"Run dir         : {run_dir.name}",
        f"Reference       : {REF.name}",
        "",
        f"Peak reaction (sim)     : {peak_sim:8.4f} kN at u = {u_at_peak_sim:.5f} mm",
        f"Peak reaction (ref)     : {peak_ref:8.4f} kN at u = {u_at_peak_ref:.5f} mm",
        f"Sim u_max               : {u_sim.max():.5f} mm "
        f"({'covers' if sim_reached_peak else 'STOPS BEFORE'} reference peak)",
        f"Peak relative error     : {peak_err*100:6.2f} %    "
        f"(tol {PEAK_TOL*100:.0f} %)  -> {'PASS' if pass_peak else 'FAIL'}",
        f"Pre-peak L2 rel error   : {l2_pre*100:6.2f} %    "
        f"(tol {L2_PREPEAK_TOL*100:.0f} %)  -> {'PASS' if pass_l2_pre else 'FAIL'}",
        (f"Post-peak L2 rel error  : {l2_post*100:6.2f} %    "
         f"(tol {L2_PREPEAK_TOL*100:.0f} %)  -> "
         f"{'PASS' if pass_l2_post else 'FAIL'}"
         f"{'' if args.strict_postpeak else '   (reported only)'}")
        if not np.isnan(l2_post)
        else (f"Post-peak L2 rel error  :   N/A    "
              f"(run did not reach post-peak)"
              f"{'   -> FAIL' if (args.strict_postpeak and not pass_l2_post) else ''}"),
        f"Dissipated-energy err   : {energy_err*100:6.2f} %    "
        f"(W_sim={W_sim:.5f}, W_ref={W_ref:.5f} kN*mm)  "
        f"(tol {ENERGY_TOL*100:.0f} %)  -> {'PASS' if pass_energy else 'FAIL'}",
        f"Full-curve L2 (info)    : {l2_full*100:6.2f} %    "
        f"(NOT GATED -- needs arc-length for snap-back)",
        "",
        f"OVERALL: {'PASS' if overall_pass else 'FAIL'}",
    ]
    text = "\n".join(report)
    print(text)
    (run_dir / "compare_report.txt").write_text(text + "\n")

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(u_ref, R_ref, "k-", lw=2.0, label="Miehe / PhaseFieldX 1711")
    ax.plot(u_sim, R_sim, "C0--", lw=1.4,
            label="phast (quasi-static)")
    ax.set_xlabel(r"applied displacement $u_y$ [mm]")
    ax.set_ylabel(r"reaction force $|R_y|$ [kN]")
    ax.set_title("SENT (Miehe 2010) -- load-displacement")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    ax.set_xlim(0, max(u_ref.max(), u_sim.max()) * 1.05)
    fig.tight_layout()
    fig_path = run_dir / "compare.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {fig_path}")

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
