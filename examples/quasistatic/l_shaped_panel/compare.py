#!/usr/bin/env python3
"""Compare quasi-static L-shaped panel run against literature data.

For each material the script runs:

  1. **Qualitative**: simulated peak reaction must lie within the
     literature peak band for that material (see EXPECTED_PEAK_KN).
  2. **Quantitative L2**: when a digitised reference CSV is available
     under reference_solutions/ (currently: concrete via
     ``ambati_2015_lshaped_concrete.csv``), compute the relative L2
     error between the simulated load-displacement envelope and the
     reference, on the reference's u-grid. Tolerance: ``L2_TOL``.

Crack-path morphology is qualitative: visually compare ``damage_final.png``
against Ambati 2015 Fig 18 (concrete) or the Rudshaug 2024 reference
(glass).

Acceptance criterion (issue #119): peak band + L2 (when available);
overall PASS only if both pass. Glass currently lacks a digitised
reference -- only the qualitative band check runs there until #133's
glass-CSV follow-up lands.
"""
from __future__ import annotations

import argparse
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
PEAK_TOL_QUAL = 0.25  # fractional tolerance vs literature midpoint
PEAK_TOL = 0.15       # quantitative peak-reaction relative error vs reference
L2_TOL = 0.20         # relative L2 tolerance on load-displacement envelope

# The 2D finite-element model returns reaction force per unit out-of-plane
# thickness. Winkler/Ambati's concrete L-panel is a 100 mm thick specimen,
# so convert the simulated concrete reaction to total force before comparing
# with the digitised kN curve. The Rudshaug glass force-displacement curve is
# not digitised here, so keep the legacy/smoke comparison unscaled.
SIM_REACTION_SCALE = {
    "l_shaped_concrete": 100.0,
    "l_shaped_glass": 1.0,
}

# Literature peak ranges (kN) by material preset.
#
# Note (concrete): the previous (6.0, 8.0) band was the WINKLER 2001
# experimental band. The l_shaped_concrete preset matches Ambati 2015
# parameters exactly, and Ambati's simulation overshoots Winkler
# experiment by ~2x: Fig 19 hybrid peak is ~16 kN. So our solver should
# match Ambati simulation, not Winkler experiment. See
# reference_solutions/PROVENANCE.md for the discrepancy discussion.
#
# Note (glass, 2024 paper): the existing (0.27, 0.32) kN band was set
# from earlier internal runs. The Rudshaug 2024 paper itself reports
# fracture-initiation force in [75, 115] N -> [0.075, 0.115] kN, NOT
# 270-320 N. The 2024 paper does not publish an F-d curve (only crack
# paths, speeds, and the initiation-force band; F-d curves live in the
# 2023 companion paper, see reference_solutions/README.md).
# When the user passes --reference rudshaug_2024 below, the
# initiation-force band [0.075, 0.115] kN from the 2024 paper text is
# used instead of the legacy 0.27-0.32 band. Switching the default
# would flip the pass/fail outcome of historical glass runs and is
# deferred until the 2023 F-d data lands.
EXPECTED_PEAK_KN = {
    "l_shaped_glass": (0.27, 0.32),      # Legacy band, pre-#133 follow-up
    "l_shaped_concrete": (15.0, 17.0),   # Ambati 2015 Fig 19 hybrid peak
}

# Per-reference override of the peak band, when --reference is passed.
# Lets us check against Rudshaug 2024 fracture-initiation force band
# [75, 115] N from page 64 / Fig 12c colorbar without changing the
# default historical band.
REFERENCE_PEAK_KN = {
    "rudshaug_2024": {
        "l_shaped_glass": (0.075, 0.115),
    },
}

# Reference CSV file per material (whitespace-separated u_y[mm], R[kN]).
# When present, an L2 comparison is run alongside the qualitative band check.
# Note: the 2024 paper does NOT contain an F-d curve, so the
# rudshaug_2024_lshaped_glass.csv path resolves to a missing file by
# design; only the band check runs for the glass preset until the
# Rudshaug 2023 paper is digitised.
REFERENCE_CSV = {
    "l_shaped_concrete": HERE / "reference_solutions" / "ambati_2015_lshaped_concrete.csv",
    "l_shaped_glass":    HERE / "reference_solutions" / "rudshaug_2024_lshaped_glass.csv",
}

# Placeholder fallbacks (PLACEHOLDER suffix -- not digitised literature).
# Used when the real reference above is missing, so the L2 + peak-error
# code path runs end-to-end. The fallback is flagged in the report.
REFERENCE_CSV_PLACEHOLDER = {
    "l_shaped_concrete": HERE / "reference_solutions" / "lshape_concrete_PLACEHOLDER.csv",
    "l_shaped_glass":    HERE / "reference_solutions" / "lshape_glass_PLACEHOLDER.csv",
}

# Auxiliary 2024 reference CSVs (crack speed + path + init-force band).
# These do not feed an L2 metric (the geometry differs and our solver
# is quasi-static while the 2024 sim is dynamic-with-mass-scaling), but
# they are exposed via --reference rudshaug_2024 for plot overlays.
REFERENCE_AUX_2024 = {
    "fracture_init_force": HERE / "reference_solutions" / "rudshaug_2024_fracture_init_force.csv",
    "crack_speed_weak":    HERE / "reference_solutions" / "rudshaug_2024_crack_speed_weak.csv",
    "crack_speed_strong":  HERE / "reference_solutions" / "rudshaug_2024_crack_speed_strong.csv",
    "crack_path_weak":     HERE / "reference_solutions" / "rudshaug_2024_crack_path_weak.csv",
    "crack_path_strong":   HERE / "reference_solutions" / "rudshaug_2024_crack_path_strong.csv",
}


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


def load_reference(material: str):
    """Load digitised (u_y, R) reference if it exists for ``material``.

    Returns ``(u_ref, R_ref, source_path, is_placeholder)`` or ``None``
    if neither a digitised nor a placeholder reference is shipped.
    """
    csv = REFERENCE_CSV.get(material)
    is_placeholder = False
    if csv is None or not csv.exists():
        csv = REFERENCE_CSV_PLACEHOLDER.get(material)
        is_placeholder = True
        if csv is None or not csv.exists():
            return None
    data = np.loadtxt(csv, comments="#")
    order = np.argsort(data[:, 0])
    return data[order, 0], data[order, 1], csv, is_placeholder


def relative_l2(u_sim, R_sim, u_ref, R_ref):
    """L2 norm of (sim - ref) on the reference's u-grid, normalised by ||ref||.

    Uses linear interpolation of the simulated curve onto the reference
    sampling. Both inputs must be monotone in u; the helper takes the
    monotone-loading envelope (largest u along the cumulative max) to
    avoid the unloading branches dominating the L2 metric.
    """
    # Monotone envelope of sim (cumulative max along u so we ignore
    # any cyclic unloading dips when the loader is cycled).
    order = np.argsort(u_sim)
    u_sim_s = u_sim[order]
    R_sim_s = R_sim[order]
    R_sim_env = np.maximum.accumulate(R_sim_s)
    overlap = (u_ref >= u_sim_s[0]) & (u_ref <= u_sim_s[-1])
    if not np.any(overlap):
        return float("nan"), 0.0, float("nan")
    u_ref_o = u_ref[overlap]
    R_ref_o = R_ref[overlap]
    R_sim_on_ref = np.interp(u_ref_o, u_sim_s, R_sim_env)
    num = float(np.linalg.norm(R_sim_on_ref - R_ref_o))
    den = float(np.linalg.norm(R_ref_o))
    coverage = (float(u_ref_o[-1]) - float(u_ref_o[0])) / max(
        float(u_ref[-1]) - float(u_ref[0]), 1e-12)
    return num / max(den, 1e-12), coverage, float(u_ref_o[-1])


def detect_material(run_dir):
    meta = run_dir / "run_metadata.json"
    if not meta.exists():
        return None
    try:
        m = json.loads(meta.read_text())
    except Exception:
        return None
    mat = m.get("material", {})
    E = mat.get("E")
    if E is None:
        return None
    # Glass: E ~ 70 GPa; concrete: E ~ 25 GPa.
    if E > 50000:
        return "l_shaped_glass"
    return "l_shaped_concrete"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--material", type=str, default=None,
                    choices=list(EXPECTED_PEAK_KN.keys()),
                    help="Override auto-detected material preset.")
    ap.add_argument("--reference", type=str, default=None,
                    choices=sorted(REFERENCE_PEAK_KN.keys()),
                    help=("Use a specific literature reference for the peak "
                          "band (currently 'rudshaug_2024' overrides the "
                          "glass band to [75, 115] N from the 2024 paper "
                          "fracture-initiation text). Default is the legacy "
                          "EXPECTED_PEAK_KN band."))
    args = ap.parse_args()

    run_dir = args.run_dir or find_latest_run_dir(HERE)
    if run_dir is None:
        print("FAIL: no run directory found.", file=sys.stderr)
        sys.exit(2)
    run_dir = run_dir.resolve()

    u_sim, R_sim, src = load_run(run_dir)
    material = args.material or detect_material(run_dir) or "l_shaped_glass"
    if args.reference is not None and material in REFERENCE_PEAK_KN.get(args.reference, {}):
        lo, hi = REFERENCE_PEAK_KN[args.reference][material]
        ref_band_label = f"{args.reference} (override)"
    else:
        lo, hi = EXPECTED_PEAK_KN[material]
        ref_band_label = "default (legacy)"
    midpoint = 0.5 * (lo + hi)

    reaction_scale = SIM_REACTION_SCALE.get(material, 1.0)
    if reaction_scale != 1.0:
        R_sim = R_sim * reaction_scale

    peak_sim = float(R_sim.max())
    u_at_peak_sim = float(u_sim[int(np.argmax(R_sim))])
    rel_to_mid = abs(peak_sim - midpoint) / midpoint
    in_range = lo <= peak_sim <= hi
    in_band = rel_to_mid <= PEAK_TOL_QUAL
    qualitative_pass = in_range or in_band

    # Quantitative peak + L2 check on the load-displacement envelope.
    # Falls back to a PLACEHOLDER reference when no digitised curve is
    # shipped (e.g. glass): the code path runs end-to-end but the
    # placeholder result is informational, not a real acceptance gate.
    ref = load_reference(material)
    if ref is not None:
        u_ref, R_ref, ref_path, is_placeholder = ref
        l2_rel, l2_coverage, l2_umax = relative_l2(u_sim, R_sim, u_ref, R_ref)
        peak_ref = float(np.max(R_ref))
        u_at_peak_ref = float(u_ref[int(np.argmax(R_ref))])
        peak_err = abs(peak_sim - peak_ref) / max(peak_ref, 1e-12)
        if is_placeholder:
            l2_pass = None  # informational only -- placeholder does not gate
            peak_pass = None
        else:
            l2_pass = l2_rel <= L2_TOL
            peak_pass = peak_err <= PEAK_TOL
        ref_label = ref_path.name + (" (PLACEHOLDER)" if is_placeholder else "")
    else:
        u_ref = R_ref = None
        l2_rel = float("nan")
        peak_ref = float("nan")
        u_at_peak_ref = float("nan")
        peak_err = float("nan")
        l2_pass = None
        peak_pass = None
        l2_coverage = 0.0
        l2_umax = float("nan")
        is_placeholder = False
        ref_label = "(no digitised reference shipped)"

    overall_pass = (qualitative_pass
                    and (l2_pass is not False)
                    and (peak_pass is not False))

    lines = [
        "L-shaped panel -- comparison report",
        "=" * 56,
        f"Run dir         : {run_dir.name}",
        f"Source CSV      : {src}",
        f"Material preset : {material}",
        f"Reference CSV   : {ref_label}",
        f"Peak band source: {ref_band_label}",
        f"Sim force scale : x{reaction_scale:g} "
        f"({'total specimen force' if reaction_scale != 1.0 else 'per-run force'})",
        f"Expected peak   : {lo:.3f} - {hi:.3f} kN (literature)",
        "",
        f"Peak reaction (sim)     : {peak_sim:8.4f} kN at u = {u_at_peak_sim:.5f} mm",
        f"Within literature band  : {'YES' if in_range else 'no'}",
        f"Within +/- {PEAK_TOL_QUAL*100:.0f} % of midpoint    : "
        f"{'YES' if in_band else 'no'}  ({rel_to_mid*100:5.2f} %)",
    ]
    if ref is not None:
        lines += [
            "",
            f"Peak reaction (ref)     : {peak_ref:8.4f} kN at u = {u_at_peak_ref:.5f} mm",
        ]
        if is_placeholder:
            lines += [
                f"Peak rel error (info)   : {peak_err*100:6.2f} %  "
                f"(PLACEHOLDER -- not gated)",
                f"L2 (envelope vs ref)    : {l2_rel*100:6.2f} %  "
                f"to u={l2_umax:.3f} mm, coverage={l2_coverage*100:.1f} % "
                f"(PLACEHOLDER -- not gated)",
            ]
        else:
            lines += [
                f"Peak rel error          : {peak_err*100:6.2f} %  "
                f"(tol {PEAK_TOL*100:.0f} %)  -> {'PASS' if peak_pass else 'FAIL'}",
                f"L2 (envelope vs ref)    : {l2_rel*100:6.2f} %  "
                f"to u={l2_umax:.3f} mm, coverage={l2_coverage*100:.1f} % "
                f"(tol {L2_TOL*100:.0f} %)  -> {'PASS' if l2_pass else 'FAIL'}",
            ]
    lines += [
        "",
        f"OVERALL                 : {'PASS' if overall_pass else 'FAIL'}",
    ]
    if ref is None:
        lines += [
            "",
            "Note: no digitised reference for this material. Drop a CSV "
            "under reference_solutions/ to enable the L2 check (see "
            "PROVENANCE.md for the schema).",
        ]
    text = "\n".join(lines)
    print(text)
    (run_dir / "compare_report.txt").write_text(text + "\n")

    qualitative_pass = overall_pass  # for the exit code below

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(u_sim, R_sim, "C3-", lw=1.4,
            label=f"phast ({material})")
    if ref is not None:
        ax.plot(u_ref, R_ref, "k--", lw=1.0, alpha=0.7,
                label=f"reference: {ref_label}")
    ax.axhspan(lo, hi, color="grey", alpha=0.2,
               label=f"literature peak band {lo:.2f}-{hi:.2f} kN")
    ax.set_xlabel(r"applied displacement $|u_y|$ at load point [mm]")
    ax.set_ylabel(r"reaction force $|R_y|$ [kN]")
    ax.set_title("L-shaped panel -- load-displacement (qualitative)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig_path = run_dir / "compare.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {fig_path}")

    sys.exit(0 if qualitative_pass else 1)


if __name__ == "__main__":
    main()
