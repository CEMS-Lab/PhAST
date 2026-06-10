"""Kalthoff crack-angle extraction (issue #235).

Post-processing utility for the Kalthoff--Winkler benchmark
(B2). The experimentally observed crack initiates at the
notch tip and propagates at ~70 deg from the horizontal
(Kalthoff 2000). This script extracts the angle from a
phast run for the paper-1 mesh-convergence
reconciliation (paper-1 blocker #234).

Pipeline
--------
1. Load ``training_data.h5`` and pull the final timestep's
   ``damage_nodal`` together with the node coordinates.
2. Rasterise the unstructured nodal damage field to a
   uniform 2D grid (linear interpolation, NaN outside the
   convex hull).
3. Threshold ``d > 0.5`` and skeletonise (skimage if
   available, otherwise a manual ring-erosion fallback).
4. Restrict to the upper-right notch-tip quadrant so the
   notch slot itself does not bias the regression.
5. Fit a least-squares line in (x, y) space and report
   the angle from horizontal.
6. Save ``crack_angle.png`` overlay and print a one-line
   summary.

CLI
---
    python extract_crack_angle.py <run_dir> [--mesh-label N]

Example
-------
    python extract_crack_angle.py \
        ../../../benchmarks/timing_comparisons/sent/torch/output \
        --mesh-label 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

try:  # pragma: no cover - import guarded
    from skimage.morphology import skeletonize as _sk_skeletonize

    _HAVE_SKIMAGE = True
except Exception:  # pragma: no cover
    _HAVE_SKIMAGE = False


# ---------------------------------------------------------------------------
# core algorithms (kept pure so tests can call them directly)
# ---------------------------------------------------------------------------

def manual_skeleton(mask: np.ndarray) -> np.ndarray:
    """Cheap 1-ring erosion fallback when skimage is missing.

    Removes one outer pixel from every 4-connected blob; on a
    thin crack this collapses to (almost) a single line.
    """
    m = mask.astype(bool)
    pad = np.pad(m, 1, mode="constant", constant_values=False)
    interior = (
        pad[1:-1, 1:-1]
        & pad[:-2, 1:-1]
        & pad[2:, 1:-1]
        & pad[1:-1, :-2]
        & pad[1:-1, 2:]
    )
    # boundary points are kept iff they have at least one
    # neighbour also on the boundary (so we don't eat the
    # whole crack on thin features)
    boundary = m & ~interior
    skel = m.copy()
    # erode the boundary one ring
    skel[boundary] = False
    # if erosion ate everything (very thin crack), fall back
    # to the original mask
    if not skel.any():
        return m
    return skel


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Skeletonise a 2D binary mask. Uses skimage if available."""
    if _HAVE_SKIMAGE:
        return _sk_skeletonize(mask.astype(bool))
    return manual_skeleton(mask)


def fit_angle_degrees(xs: np.ndarray, ys: np.ndarray) -> float:
    """Least-squares line fit; returns angle from horizontal in degrees.

    Uses ``polyfit`` on whichever axis has the larger spread so
    near-vertical cracks are still well-conditioned.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.size < 2:
        raise ValueError("Need >= 2 skeleton points for a line fit.")
    dx = xs.max() - xs.min()
    dy = ys.max() - ys.min()
    if dx >= dy:
        slope = np.polyfit(xs, ys, 1)[0]
        angle = np.degrees(np.arctan(slope))
    else:
        # fit x = a*y + b, then convert
        inv_slope = np.polyfit(ys, xs, 1)[0]
        # angle from horizontal -> slope = dy/dx = 1/inv_slope
        angle = np.degrees(np.arctan2(1.0, inv_slope))
    # report unsigned acute angle from horizontal
    return float(abs(angle))


def extract_angle_from_mask(mask: np.ndarray) -> Tuple[float, np.ndarray]:
    """Skeletonise ``mask`` and return (angle_deg, skeleton).

    Pixel coordinates: angle is measured in (col, row) image
    space with row pointing *down*. To match the physical
    orientation we flip the row axis.
    """
    skel = skeletonize(mask)
    rows, cols = np.where(skel)
    if rows.size < 2:
        raise ValueError("Skeleton has fewer than 2 pixels; cannot fit.")
    H = mask.shape[0]
    xs = cols.astype(np.float64)
    ys = (H - 1 - rows).astype(np.float64)  # flip y so up is +
    return fit_angle_degrees(xs, ys), skel


# ---------------------------------------------------------------------------
# h5 loading + rasterisation
# ---------------------------------------------------------------------------

def _final_step_key(h5: h5py.File) -> str:
    steps = list(h5["simulation_data/steps"].keys())
    steps.sort()
    return steps[-1]


def load_final_damage(run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(coords, damage)`` from the last timestep."""
    h5_path = run_dir / "training_data.h5"
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)
    with h5py.File(h5_path, "r") as f:
        coords = np.asarray(
            f["simulation_data/mesh/node_coordinates"], dtype=np.float64
        )
        step = _final_step_key(f)
        damage = np.asarray(
            f[f"simulation_data/steps/{step}/damage_nodal"], dtype=np.float64
        )
    return coords, damage


def rasterise(
    coords: np.ndarray,
    damage: np.ndarray,
    nx: int = 400,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear-interpolate scattered nodal damage to a uniform grid.

    Returns (xs_1d, ys_1d, grid) with grid[j, i] = d at (xs[i], ys[j]).
    NaN outside the convex hull is filled with 0.
    """
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    aspect = (ymax - ymin) / (xmax - xmin) if (xmax - xmin) > 0 else 1.0
    ny = max(2, int(round(nx * aspect)))
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys)
    grid = griddata(coords, damage, (XX, YY), method="linear")
    grid = np.nan_to_num(grid, nan=0.0)
    return xs, ys, grid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _restrict_to_notch_quadrant(
    mask: np.ndarray, xs: np.ndarray, ys: np.ndarray
) -> np.ndarray:
    """Keep only the upper-right quadrant relative to the domain centre.

    Kalthoff-Winkler notch is on the left edge ~mid-height; the
    crack runs up and to the right. Discarding the other three
    quadrants protects the line fit from any pre-existing
    horizontal notch slot in the damage field.
    """
    out = mask.copy()
    cx = 0.5 * (xs[0] + xs[-1])
    cy = 0.5 * (ys[0] + ys[-1])
    XX, YY = np.meshgrid(xs, ys)
    keep = (XX >= cx) & (YY >= cy)
    out &= keep
    return out


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        description="Extract Kalthoff crack angle from a phast run.",
    )
    p.add_argument("run_dir", type=Path)
    p.add_argument("--mesh-label", type=int, default=None)
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Damage threshold for the crack mask (default 0.5).",
    )
    p.add_argument("--nx", type=int, default=400)
    p.add_argument(
        "--no-quadrant",
        action="store_true",
        help="Disable the upper-right quadrant restriction.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output overlay PNG (default <run_dir>/crack_angle.png).",
    )
    args = p.parse_args(argv)

    coords, damage = load_final_damage(args.run_dir)
    xs, ys, grid = rasterise(coords, damage, nx=args.nx)
    raw_mask = grid > args.threshold
    mask = (
        raw_mask
        if args.no_quadrant
        else _restrict_to_notch_quadrant(raw_mask, xs, ys)
    )
    if not mask.any():
        print(
            "[extract_crack_angle] WARNING: empty crack mask after "
            "thresholding; falling back to full domain.",
            file=sys.stderr,
        )
        mask = raw_mask
    angle, skel = extract_angle_from_mask(mask)

    label = (
        f"Mesh {args.mesh_label}"
        if args.mesh_label is not None
        else f"Run {args.run_dir.name}"
    )
    print(f"{label}: angle = {angle:.1f} deg (vs ref ~70 deg)")

    out = args.out or (args.run_dir / "crack_angle.png")
    _plot_overlay(grid, mask, skel, xs, ys, angle, label, out)
    print(f"  overlay -> {out}")
    return 0


def _plot_overlay(
    grid: np.ndarray,
    mask: np.ndarray,
    skel: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    angle_deg: float,
    label: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(
        grid,
        extent=(xs[0], xs[-1], ys[0], ys[-1]),
        origin="lower",
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
    )
    rows, cols = np.where(skel)
    H = mask.shape[0]
    sx = xs[cols]
    sy = ys[H - 1 - rows]
    ax.scatter(sx, sy, s=2, c="cyan", label="skeleton")
    if sx.size >= 2:
        # plot the fitted line through the skeleton centroid
        cx, cy = sx.mean(), sy.mean()
        slope = np.tan(np.radians(angle_deg))
        L = 0.5 * (xs[-1] - xs[0])
        line_x = np.array([cx - L, cx + L])
        line_y = cy + slope * (line_x - cx)
        ax.plot(line_x, line_y, "w--", lw=1.5, label=f"fit ({angle_deg:.1f} deg)")
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])
    ax.set_aspect("equal")
    ax.set_title(f"{label}: crack angle = {angle_deg:.1f} deg")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
