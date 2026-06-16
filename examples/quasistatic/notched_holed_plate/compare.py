#!/usr/bin/env python3
"""Compare quasi-static notched-holed-plate run against the COMSOL 6.4
Application Library reference (`comsol_load_displacement.csv`).

Displacement convention (issue #223 fix):
  The COMSOL Application Library report plots reaction force against
  **total elongation** of the specimen (top_pin_y - bottom_pin_y =
  2 x per-pin displacement). The YAML-driven simulator's `results.csv`
  `displacement` column reports the **per-pin** value configured in
  `configs/benchmarks/quasistatic/QS_notched_holed_plate.yaml`.
  To compare on a common axis we keep the sim in per-pin units (cleaner
  internal convention, matches `bcs.add` semantics) and reframe the
  reference into per-pin units by dividing by 2 in `load_reference()`.
  The acceptance constants below (`REF_FIRST_DISP_MM`,
  `REF_SECOND_DISP_MM`) are stated in per-pin units accordingly.

Acceptance criteria (issue #119, benchmark #1):
  * first  peak load:        within +/-10% of 0.63 kN
  * first  peak displacement: within +/-15% of 0.165 mm (per-pin;
                              0.33 mm total elongation in the PDF)
  * second peak load:        within +/-20% of 0.15 kN

The script picks the two highest distinct local maxima of the simulated
load--displacement curve (separated by at least the dip in between).
Crack-path morphology is qualitative: `damage_final.png` from the run is
saved alongside `compare.png` for visual side-by-side check against
Figure 4 of the COMSOL PDF.
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
REF = HERE / "comsol_load_displacement.csv"
# The public example root carries the promoted strict-parity reference package.
CANONICAL_REFERENCE_RUNS: list[Path] = [HERE]

# Reference values from the COMSOL PDF text.
REF_FIRST_LOAD_KN = 0.63
# Per-pin units: PDF reports 0.33 mm total elongation -> 0.165 mm per-pin
# (see displacement-convention note in module docstring; issue #223).
REF_FIRST_DISP_MM = 0.165
REF_SECOND_LOAD_KN = 0.15
# Per-pin units: PDF reports 1.7 mm total elongation -> 0.85 mm per-pin.
REF_SECOND_DISP_MM = 0.85

TOL_FIRST_LOAD = 0.10
TOL_FIRST_DISP = 0.15
TOL_SECOND_LOAD = 0.20


def find_latest_run_dir(base: Path) -> Path | None:
    """Find the most recent ``run_*/`` directory containing run output CSVs.

    Searches both this benchmark directory and the repo's working
    directory (the YAML driver creates run dirs under cwd by default).
    """
    search_roots = [base, Path.cwd()]
    candidates = []
    for root in search_roots:
        candidates.extend(p for p in root.glob("run_*")
                          if (p / "results.csv").exists()
                          or (p / "history.csv").exists())
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def find_default_reference_run() -> Path | None:
    for run_dir in CANONICAL_REFERENCE_RUNS:
        if (run_dir / "results.csv").exists() or (run_dir / "history.csv").exists():
            return run_dir
    return None


def _load_csv(
    csv_path: Path,
    disp_col: str,
    react_col: str,
    scale_kN: float,
) -> tuple[np.ndarray, np.ndarray]:
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
    disp = arr[:, cols[disp_col]]
    react = np.abs(arr[:, cols[react_col]]) * scale_kN
    return disp, react


def load_run_results(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (displacement_mm, |reaction_kN|)."""
    results = run_dir / "results.csv"
    history = run_dir / "history.csv"
    if results.exists():
        try:
            return _load_csv(results, "displacement", "reaction_kN", 1.0)
        except RuntimeError:
            if not history.exists():
                raise
    if history.exists():
        return _load_csv(history, "applied_disp", "reaction_force", 1.0 / 1000.0)
    raise FileNotFoundError(f"No results.csv or history.csv under {run_dir}")


def load_reference() -> tuple[np.ndarray, np.ndarray]:
    rows = []
    with REF.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            try:
                rows.append([float(parts[0]), float(parts[1])])
            except (ValueError, IndexError):
                # header line or malformed; skip
                continue
    data = np.asarray(rows)
    # Reference CSV reports total elongation (= top_pin_y - bottom_pin_y
    # = 2 x per-pin displacement); see the displacement convention documented
    # in PROVENANCE.md and the module docstring above.
    # Convert to per-pin to match the sim's `results.csv` displacement
    # column. (Issue #223.)
    data[:, 0] = data[:, 0] / 2.0
    return data[:, 0], data[:, 1]


def find_two_peaks(u: np.ndarray, R: np.ndarray) -> tuple[tuple, tuple]:
    """Find the first and second local maxima.

    First peak: global maximum.
    Second peak: largest value at any sample taken at u >= u_first + 0.2 mm
    (well past the first-peak softening trough).
    """
    i1 = int(np.argmax(R))
    u1, R1 = float(u[i1]), float(R[i1])

    mask2 = u >= (u1 + 0.2)
    if not mask2.any():
        return (u1, R1), (float("nan"), float("nan"))
    R2_arr = R[mask2]
    u2_arr = u[mask2]
    j = int(np.argmax(R2_arr))
    return (u1, R1), (float(u2_arr[j]), float(R2_arr[j]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Specific run_* directory; default = most recent.")
    args = ap.parse_args()

    run_dir = args.run_dir
    if run_dir is None:
        run_dir = find_default_reference_run()
    if run_dir is None:
        run_dir = find_latest_run_dir(HERE)
    if run_dir is None or not (
        (run_dir / "results.csv").exists() or (run_dir / "history.csv").exists()
    ):
        print(f"FAIL: no run output CSV under {HERE}; run the YAML config first.",
              file=sys.stderr)
        sys.exit(2)
    run_dir = run_dir.resolve()

    u_sim, R_sim = load_run_results(run_dir)
    u_ref, R_ref = load_reference()
    (u1_s, R1_s), (u2_s, R2_s) = find_two_peaks(u_sim, R_sim)

    err_R1 = abs(R1_s - REF_FIRST_LOAD_KN) / REF_FIRST_LOAD_KN
    err_u1 = abs(u1_s - REF_FIRST_DISP_MM) / REF_FIRST_DISP_MM
    err_R2 = (abs(R2_s - REF_SECOND_LOAD_KN) / REF_SECOND_LOAD_KN
              if not np.isnan(R2_s) else float("inf"))

    pass_R1 = err_R1 <= TOL_FIRST_LOAD
    pass_u1 = err_u1 <= TOL_FIRST_DISP
    pass_R2 = err_R2 <= TOL_SECOND_LOAD
    overall = pass_R1 and pass_u1 and pass_R2

    report = [
        "Notched holed plate (COMSOL 6.4 brittle_fracture) -- comparison report",
        "=" * 70,
        f"Run dir   : {run_dir.name}",
        f"Reference : {REF.name}",
        "",
        f"First  peak (sim)  : R = {R1_s:7.4f} kN at u = {u1_s:.4f} mm",
        f"First  peak (ref)  : R = {REF_FIRST_LOAD_KN:7.4f} kN at u = "
        f"{REF_FIRST_DISP_MM:.4f} mm",
        f"  load  err: {err_R1*100:6.2f} %  (tol {TOL_FIRST_LOAD*100:.0f} %) "
        f" -> {'PASS' if pass_R1 else 'FAIL'}",
        f"  disp  err: {err_u1*100:6.2f} %  (tol {TOL_FIRST_DISP*100:.0f} %) "
        f" -> {'PASS' if pass_u1 else 'FAIL'}",
        "",
        f"Second peak (sim)  : R = {R2_s:7.4f} kN at u = {u2_s:.4f} mm",
        f"Second peak (ref)  : R = {REF_SECOND_LOAD_KN:7.4f} kN at u = "
        f"{REF_SECOND_DISP_MM:.4f} mm",
        f"  load  err: {err_R2*100:6.2f} %  (tol {TOL_SECOND_LOAD*100:.0f} %) "
        f" -> {'PASS' if pass_R2 else 'FAIL'}",
        "",
        f"OVERALL: {'PASS' if overall else 'FAIL'}",
        "",
        "Crack-path morphology is qualitative: see damage_final.png in the "
        "run directory and compare against Figure 4 of the COMSOL PDF "
        "(curved crack from the notch tip toward the large hole).",
    ]
    text = "\n".join(report)
    print(text)
    (run_dir / "compare_report.txt").write_text(text + "\n")

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(u_ref, R_ref, "k-o", lw=2.0, ms=4,
            label="COMSOL 6.4 (reference)")
    ax.plot(u_sim, R_sim, "C0-", lw=1.4,
            label="phast (quasi-static)")
    ax.plot([u1_s], [R1_s], "C0v", ms=8, label=f"sim 1st peak ({R1_s:.2f} kN)")
    ax.plot([u2_s], [R2_s], "C0s", ms=7, label=f"sim 2nd peak ({R2_s:.2f} kN)")
    ax.axhline(REF_FIRST_LOAD_KN, color="grey", ls=":", lw=0.8, alpha=0.6)
    ax.axhline(REF_SECOND_LOAD_KN, color="grey", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel(r"per-pin displacement $u$ [mm]")
    ax.set_ylabel(r"reaction force $|R_y|$ [kN]")
    ax.set_title("Notched holed plate -- load--displacement vs COMSOL 6.4")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim(0, max(u_ref.max(), u_sim.max()) * 1.05)
    fig.tight_layout()
    fig.savefig(run_dir / "compare.png", dpi=200)
    plt.close(fig)

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
