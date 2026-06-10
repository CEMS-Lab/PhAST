#!/usr/bin/env python3
"""Compare quasi-static three-point bending run against PhaseFieldX 1714.

Loads the most recent ``run_*`` directory under this benchmark folder
and the reference load-displacement curve under ``reference_solutions/``.

Acceptance criteria (issue #256/#132 readiness gates):
  * peak-reaction relative error <= 5%               (gate active)
  * pre-peak load-displacement L2 <= 10%             (gate active)
  * dissipated-energy relative error <= 15%          (gate active)
  * post-peak L2 (u > u_at_peak_ref)                 (reported only;
       opt in to gating with --strict-postpeak; uses L2_PREPEAK_TOL)

The full-curve L2 is reported INFO-only; pure displacement-controlled
Newton cannot trace the unstable branch (dF/du < 0), so a full-curve
L2 gate would penalise a physically-correct solver simply because the
loading protocol cannot reach the negative-slope branch. The TPB
reference shows a sharp post-peak collapse from 0.0347 to 0.0014 kN
within 0.0003 mm, which only an arc-length-controlled solver can
trace -- our staggered displacement-controlled run at the same u
sustains the pre-collapse branch instead.

Reads ``results.csv`` (preferred) or falls back to ``history.csv``.
Reaction in ``history.csv`` is in newtons; the loader rescales to kN.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# NumPy >=2.0 renamed trapz -> trapezoid; np.trapz removed in 2.0.
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)

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
REF = HERE / "reference_solutions" / "miehe_tpb_load_displacement.csv"

# Tolerances (issue #256/#132 -- post-peak readiness).
PEAK_TOL = 0.05         # peak-reaction relative error
L2_PREPEAK_TOL = 0.10   # pre-peak segment (u <= u_at_peak_ref)
ENERGY_TOL = 0.15       # dissipated energy = integral F du over overlap


def find_latest_run_dir(base: Path) -> Path | None:
    cands = [p for p in base.glob("run_*")
             if (p / "results.csv").exists() or (p / "history.csv").exists()]
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _load_csv(path, disp_col, react_col, scale_kN):
    rows, header = [], None
    with path.open() as fh:
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
        raise RuntimeError(f"No data rows in {path}")
    arr = np.asarray(rows)
    cols = {n: i for i, n in enumerate(header)}
    return arr[:, cols[disp_col]], np.abs(arr[:, cols[react_col]]) * scale_kN


def load_run(run_dir):
    if (run_dir / "results.csv").exists():
        try:
            u, R = _load_csv(run_dir / "results.csv",
                             "displacement", "reaction_kN", 1.0)
            return u, R, "results.csv"
        except RuntimeError:
            if not (run_dir / "history.csv").exists():
                raise
    if (run_dir / "history.csv").exists():
        u, R = _load_csv(run_dir / "history.csv",
                         "applied_disp", "reaction_force", 1.0 / 1000.0)
        return u, R, "history.csv"
    raise FileNotFoundError(run_dir)


def relative_l2_error(u_ref, R_ref, u_sim, R_sim, u_lo=None, u_hi=None):
    """L2 error of R_sim interpolated onto u_ref, normalised by ||R_ref||.

    Optionally restrict to ``[u_lo, u_hi]``.
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


def dissipated_energy_error(u_ref, R_ref, u_sim, R_sim):
    """Trapezoidal energy integral over the overlapping displacement range."""
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
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--strict-postpeak", action="store_true",
                    help="Also gate on post-peak L2 with L2_PREPEAK_TOL.")
    ap.add_argument("--prepeak-only", action="store_true",
                    help=("Gate only the stable pre-peak branch plus energy "
                          "overlap. This is for bounded displacement-control "
                          "artifact runs; strict validation still requires "
                          "covering the reference peak."))
    args = ap.parse_args()

    run_dir = args.run_dir or find_latest_run_dir(HERE)
    if run_dir is None:
        print("FAIL: no run directory found.", file=sys.stderr)
        sys.exit(2)
    run_dir = run_dir.resolve()

    u_sim, R_sim, src = load_run(run_dir)
    first = REF.read_text().splitlines()[0]
    delim = "," if "," in first else None
    data = np.loadtxt(REF, delimiter=delim)
    u_ref, R_ref = data[:, 0], data[:, 1]

    peak_sim = float(R_sim.max())
    peak_ref = float(R_ref.max())
    u_at_peak_sim = float(u_sim[int(np.argmax(R_sim))])
    u_at_peak_ref = float(u_ref[int(np.argmax(R_ref))])
    peak_err = abs(peak_sim - peak_ref) / peak_ref
    l2_full = relative_l2_error(u_ref, R_ref, u_sim, R_sim)
    l2_pre = relative_l2_error(u_ref, R_ref, u_sim, R_sim,
                               u_hi=u_at_peak_ref)
    sim_reached_peak = u_sim.max() >= u_at_peak_ref
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
    overall_pass = (
        pass_l2_pre and pass_energy if args.prepeak_only
        else pass_peak and pass_l2_pre and pass_energy and pass_l2_post
    )

    lines = [
        "Three-point bending (PhaseFieldX 1714) -- comparison report",
        "=" * 60,
        f"Run dir         : {run_dir.name}",
        f"Source CSV      : {src}",
        f"Reference       : {REF.name}",
        "",
        f"Peak reaction (sim)     : {peak_sim:8.5f} kN at u = {u_at_peak_sim:.5f} mm",
        f"Peak reaction (ref)     : {peak_ref:8.5f} kN at u = {u_at_peak_ref:.5f} mm",
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
        f"(W_sim={W_sim:.6f}, W_ref={W_ref:.6f} kN*mm)  "
        f"(tol {ENERGY_TOL*100:.0f} %)  -> {'PASS' if pass_energy else 'FAIL'}",
        f"Full-curve L2 (info)    : {l2_full*100:6.2f} %    "
        f"(NOT GATED -- needs arc-length for snap-back)",
        f"Gate mode               : {'pre-peak only' if args.prepeak_only else 'strict peak'}",
        "",
        f"OVERALL: {'PASS' if overall_pass else 'FAIL'}",
    ]
    text = "\n".join(lines)
    print(text)
    (run_dir / "compare_report.txt").write_text(text + "\n")

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(u_ref, R_ref, "k-", lw=2.0, label="Miehe / PhaseFieldX 1714")
    ax.plot(u_sim, R_sim, "C2--", lw=1.4,
            label="phast (quasi-static)")
    ax.set_xlabel(r"applied displacement $|u_y|$ [mm]")
    ax.set_ylabel(r"reaction force $|R_y|$ [kN]")
    ax.set_title("Three-point bending (Miehe 2010) -- load-displacement")
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
