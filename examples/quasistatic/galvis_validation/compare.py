#!/usr/bin/env python3
"""Compare quasi-static torch_pf run against Galvis et al. 2026.

PLACEHOLDER scaffold (issue #109). Mirrors the L-panel comparator
pattern from PR #370 (issue #119) with a pre-peak / post-peak L2 split,
but ships without digitised reference data.

When real Galvis Figs 4.1-4.4 load-displacement curves are digitised
(see ``provenance.md``), drop them into ``reference_solutions/`` as
``galvis_2026_<benchmark>.csv`` and remove the PLACEHOLDER short-circuit
below.

Reference: Galvis, A.F. et al. (2026), *Phase-field modelling of
quasi-static and dynamic brittle fracture: A FreeFEM++ implementation*,
Eng. Fract. Mech., DOI 10.1016/j.engfracmech.2026.111846.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PLACEHOLDER_CSV = HERE / "reference_solutions" / "galvis_2026_PLACEHOLDER.csv"

BENCHMARKS = ("sent", "shear", "lshape", "notched_holes")

# Acceptance tolerances from issue #109 (NOT enforced while in
# placeholder mode -- they are recorded here so the upgrade path is
# obvious once digitisation lands).
TOL_INIT_DISP = 0.05   # 5% on initiation displacement
TOL_KINK_ANGLE_DEG = 5.0
TOL_PEAK_LOAD = 0.10   # 10% on peak load


def load_reference(benchmark: str) -> tuple[Path, np.ndarray, bool]:
    """Return (path, data, is_placeholder) for the requested benchmark.

    Falls back to the PLACEHOLDER CSV until a digitised
    ``galvis_2026_<benchmark>.csv`` is committed.
    """
    real_csv = HERE / "reference_solutions" / f"galvis_2026_{benchmark}.csv"
    if real_csv.exists():
        return real_csv, np.loadtxt(real_csv, delimiter=",", comments="#"), False
    # Placeholder: do not attempt to parse the stub as a usable curve.
    return PLACEHOLDER_CSV, np.empty((0, 2)), True


def split_l2(u_sim: np.ndarray, R_sim: np.ndarray,
             u_ref: np.ndarray, R_ref: np.ndarray) -> tuple[float, float]:
    """Pre-peak / post-peak relative L2 errors on the reference u-grid."""
    if u_ref.size < 2:
        return float("nan"), float("nan")
    R_interp = np.interp(u_ref, u_sim, R_sim)
    peak_idx = int(np.argmax(R_ref))
    pre, post = slice(0, peak_idx + 1), slice(peak_idx, None)
    def _rl2(s):
        num = np.linalg.norm(R_interp[s] - R_ref[s])
        den = np.linalg.norm(R_ref[s]) or 1.0
        return float(num / den)
    return _rl2(pre), _rl2(post)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", choices=BENCHMARKS, default="sent",
                    help="Galvis 2026 quasi-static benchmark to compare against.")
    ap.add_argument("--sim-csv", type=Path, default=None,
                    help="Optional simulated load-displacement CSV (u_mm, R_kN).")
    args = ap.parse_args()

    ref_path, ref_data, is_placeholder = load_reference(args.benchmark)
    print(f"[galvis-compare] benchmark = {args.benchmark}")
    print(f"[galvis-compare] reference = {ref_path.name}"
          + (" (PLACEHOLDER)" if is_placeholder else ""))

    if is_placeholder:
        print("[galvis-compare] PLACEHOLDER reference -- quantitative "
              "comparison gated on real Galvis digitisation (issue #109).")
        print(f"[galvis-compare] tolerances pending: "
              f"init-disp <= {TOL_INIT_DISP*100:.0f}%, "
              f"kink-angle <= {TOL_KINK_ANGLE_DEG:.0f} deg, "
              f"peak-load <= {TOL_PEAK_LOAD*100:.0f}%.")
        return 0

    if args.sim_csv is None or not args.sim_csv.exists():
        print("[galvis-compare] no --sim-csv provided; reference loaded ok.")
        return 0

    sim = np.loadtxt(args.sim_csv, delimiter=",", comments="#")
    rl2_pre, rl2_post = split_l2(sim[:, 0], sim[:, 1], ref_data[:, 0], ref_data[:, 1])
    print(f"[galvis-compare] pre-peak rel-L2  = {rl2_pre:.4f}")
    print(f"[galvis-compare] post-peak rel-L2 = {rl2_post:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
