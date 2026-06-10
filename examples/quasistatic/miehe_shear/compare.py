#!/usr/bin/env python3
"""Compare quasi-static SENS (Miehe shear) run against PhaseFieldX 1712.

Loads the most recent ``run_*`` directory under this benchmark folder
and the reference load-displacement curve under ``reference_solutions/``.
Computes peak reaction force and the L2 error of the load-displacement
curve.

Acceptance criterion (issue #119, post-peak split #132):
  * peak-reaction relative error <= 5 %              (gate active)
  * pre-peak L2 (u <= u_at_peak_ref) <= 10 %         (gate active)
  * post-peak L2 (u > u_at_peak_ref)                 (reported only;
       opt in to gating with --strict-postpeak; uses L2_TOL)

Reads ``results.csv`` (preferred) or falls back to ``history.csv``;
both schemas have ``displacement`` (mm) and a reaction column. Reaction
is converted from N to kN if needed (heuristic: max |R| > 5).
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
REPO_ROOT = HERE.parents[2]
REF = HERE / "reference_solutions" / "miehe_sens_load_displacement.csv"
PHASEFIELDX_1712 = (
    REPO_ROOT
    / "reference_codes/phasefieldx-main/examples/PhaseFieldFracture"
    / "1712_Single_Edge_Notched_Shear_Test"
)
PEAK_TOL = 0.05
L2_TOL = 0.10
REFERENCE_TIERS = {
    "miehe-paper": "shipped",
    "phasefieldx-parity": "phasefieldx-output",
}


def find_latest_run_dir(base: Path) -> Path | None:
    candidates = []
    for p in base.glob("run_*"):
        if (p / "results.csv").exists() or (p / "history.csv").exists():
            candidates.append(p)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_run(run_dir: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (displacement [mm], |reaction [kN]|, source_filename)."""
    results = run_dir / "results.csv"
    history = run_dir / "history.csv"
    if results.exists():
        try:
            return _load_csv(results, "displacement", "reaction_kN", scale_kN=1.0)
        except RuntimeError:
            if not history.exists():
                raise
    if history.exists():
        # history.csv has reaction_force in N, applied_disp in mm.
        return _load_csv(history, "applied_disp", "reaction_force",
                         scale_kN=1.0 / 1000.0)
    raise FileNotFoundError(f"No results.csv or history.csv under {run_dir}")


def _load_csv(path: Path, disp_col: str, react_col: str,
              scale_kN: float) -> tuple[np.ndarray, np.ndarray, str]:
    rows = []
    header = None
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
    disp = arr[:, cols[disp_col]]
    react = np.abs(arr[:, cols[react_col]]) * scale_kN
    return disp, react, path.name


def load_reference(source: str, csv_path: Path | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    """Return reference displacement [mm], reaction [kN], and label."""
    if csv_path is not None:
        data = np.loadtxt(csv_path)
        return data[:, 0], np.abs(data[:, 1]), csv_path.name
    if source == "shipped":
        data = np.loadtxt(REF)
        return data[:, 0], np.abs(data[:, 1]), REF.name
    if source == "phasefieldx-output":
        top = PHASEFIELDX_1712 / "top.dof"
        reaction = PHASEFIELDX_1712 / "bottom.reaction"
        if not top.exists() or not reaction.exists():
            raise FileNotFoundError(
                "PhaseFieldX 1712 executable-output reference is missing; "
                f"expected {top} and {reaction}")
        disp_by_step = {}
        for raw in top.read_text().splitlines():
            if not raw.strip() or raw.startswith("#"):
                continue
            step, ux, *_ = raw.split()
            disp_by_step[int(step)] = float(ux)
        pairs = []
        for raw in reaction.read_text().splitlines():
            if not raw.strip() or raw.startswith("#"):
                continue
            step, rx, *_ = raw.split()
            step_i = int(step)
            if step_i in disp_by_step:
                pairs.append((disp_by_step[step_i], abs(float(rx))))
        if not pairs:
            raise RuntimeError(f"No matched PhaseFieldX output rows under {PHASEFIELDX_1712}")
        arr = np.asarray(pairs, dtype=float)
        return arr[:, 0], arr[:, 1], "phasefieldx_1712_executable_output"
    raise ValueError(f"Unknown reference source: {source}")


def tier_from_source(source: str, custom_csv: bool = False) -> str:
    if custom_csv:
        return "custom-csv"
    for tier, tier_source in REFERENCE_TIERS.items():
        if tier_source == source:
            return tier
    return source


def relative_l2_error(u_ref, R_ref, u_sim, R_sim,
                      u_lo=None, u_hi=None) -> float:
    """L2 of R_sim interpolated onto u_ref, normalised by ||R_ref||.

    Optionally restrict to ``[u_lo, u_hi]`` (default: full sim overlap).
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--reference-tier",
                    choices=tuple(REFERENCE_TIERS),
                    default=None,
                    help=("Named validation tier. 'phasefieldx-parity' "
                          "checks the driver setup against bundled "
                          "PhaseFieldX 1712 executable outputs; "
                          "'miehe-paper' preserves the shipped 0.53118 kN "
                          "paper-style gate. Overrides --reference-source."))
    ap.add_argument("--reference-source", choices=("shipped", "phasefieldx-output"),
                    default="phasefieldx-output",
                    help=("Reference curve to use. Default is "
                          "'phasefieldx-output' because this driver follows "
                          "the bundled PhaseFieldX 1712 l0=0.06 setup. Use "
                          "'shipped' or --reference-tier miehe-paper for the "
                          "stricter 0.53118 kN paper-style gate."))
    ap.add_argument("--reference-csv", type=Path, default=None,
                    help="Optional two-column displacement/reaction CSV overriding --reference-source.")
    ap.add_argument("--report-name", default="compare_report.txt",
                    help="Report filename written inside --run-dir.")
    ap.add_argument("--figure-name", default="compare.png",
                    help="Figure filename written inside --run-dir.")
    ap.add_argument("--strict-postpeak", action="store_true",
                    help="Also gate on post-peak L2 with L2_TOL.")
    ap.add_argument("--prepeak-only", action="store_true",
                    help=("Gate only the stable pre-peak branch. This is for "
                          "bounded displacement-control artifact runs; strict "
                          "validation still requires covering the reference peak."))
    args = ap.parse_args()

    run_dir = args.run_dir or find_latest_run_dir(HERE)
    if run_dir is None:
        print("FAIL: no run directory found; run run.py first.",
              file=sys.stderr)
        sys.exit(2)
    run_dir = run_dir.resolve()

    reference_source = (
        REFERENCE_TIERS[args.reference_tier]
        if args.reference_tier is not None
        else args.reference_source
    )
    u_sim, R_sim, src = load_run(run_dir)
    u_ref, R_ref, ref_label = load_reference(reference_source, args.reference_csv)
    reference_tier = tier_from_source(
        reference_source, custom_csv=args.reference_csv is not None)

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

    pass_peak = sim_reached_peak and peak_err <= PEAK_TOL
    pass_l2_pre = (not np.isnan(l2_pre)) and l2_pre <= L2_TOL
    if args.strict_postpeak:
        pass_l2_post = (not np.isnan(l2_post)) and l2_post <= L2_TOL
    else:
        pass_l2_post = True
    overall_pass = (
        pass_l2_pre if args.prepeak_only
        else pass_peak and pass_l2_pre and pass_l2_post
    )

    lines = [
        "Miehe SENS (PhaseFieldX 1712) -- comparison report",
        "=" * 56,
        f"Run dir         : {run_dir.name}",
        f"Source CSV      : {src}",
        f"Reference       : {ref_label}",
        f"Reference tier  : {reference_tier}",
        f"Reference source: {reference_source if args.reference_csv is None else 'custom-csv'}",
        "",
        f"Peak reaction (sim)     : {peak_sim:8.4f} kN at u = {u_at_peak_sim:.5f} mm",
        f"Peak reaction (ref)     : {peak_ref:8.4f} kN at u = {u_at_peak_ref:.5f} mm",
        f"Sim u_max               : {u_sim.max():.5f} mm "
        f"({'covers' if sim_reached_peak else 'STOPS BEFORE'} reference peak)",
        f"Peak relative error     : {peak_err*100:6.2f} %    "
        f"(tol {PEAK_TOL*100:.0f} %)  -> {'PASS' if pass_peak else 'FAIL'}",
        f"Pre-peak L2 rel error   : {l2_pre*100:6.2f} %    "
        f"(tol {L2_TOL*100:.0f} %)  -> {'PASS' if pass_l2_pre else 'FAIL'}",
        (f"Post-peak L2 rel error  : {l2_post*100:6.2f} %    "
         f"(tol {L2_TOL*100:.0f} %)  -> "
         f"{'PASS' if pass_l2_post else 'FAIL'}"
         f"{'' if args.strict_postpeak else '   (reported only)'}")
        if not np.isnan(l2_post)
        else (f"Post-peak L2 rel error  :   N/A    "
              f"(run did not reach post-peak)"
              f"{'   -> FAIL' if (args.strict_postpeak and not pass_l2_post) else ''}"),
        f"Full-curve L2 (info)    : {l2_full*100:6.2f} %    (NOT GATED)",
        f"Gate mode               : {'pre-peak only' if args.prepeak_only else 'strict peak'}",
        "",
        f"OVERALL: {'PASS' if overall_pass else 'FAIL'}",
    ]
    text = "\n".join(lines)
    print(text)
    (run_dir / args.report_name).write_text(text + "\n")

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(u_ref, R_ref, "k-", lw=2.0, label="Miehe / PhaseFieldX 1712")
    ax.plot(u_sim, R_sim, "C1--", lw=1.4,
            label="phast (quasi-static)")
    ax.set_xlabel(r"applied shear displacement $u_x$ [mm]")
    ax.set_ylabel(r"reaction force $|R_x|$ [kN]")
    ax.set_title("SENS (Miehe 2010) -- load-displacement")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    ax.set_xlim(0, max(u_ref.max(), u_sim.max()) * 1.05)
    fig.tight_layout()
    fig_path = run_dir / args.figure_name
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {fig_path}")

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
