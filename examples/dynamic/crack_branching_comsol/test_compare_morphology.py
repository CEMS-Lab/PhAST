"""Tests for the morphology-aware Y-detection in compare.py (issue #314).

The previous ``pass_full_y`` test only required ``max(d) > 0.99`` and
the previous ``branching_us`` was the late-window argmax of the
elastic-energy curve; both are morphology-blind. This module exercises
the new connected-component-based detection by feeding synthetic
damage grids directly into :func:`compare.count_arms_in_region`.

Run with:
    pytest examples/dynamic/crack_branching_comsol/test_compare_morphology.py -v
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COMPARE_PATH = HERE / "compare.py"

# Load compare.py as a module without triggering its __main__ guard.
_spec = importlib.util.spec_from_file_location("compare_b7", COMPARE_PATH)
compare_b7 = importlib.util.module_from_spec(_spec)
sys.modules["compare_b7"] = compare_b7
_spec.loader.exec_module(compare_b7)


# ---------------------------------------------------------------------
# Synthetic damage-grid builders
# ---------------------------------------------------------------------


def _make_grid(nx: int = 401, ny: int = 161,
               x_lo: float = 0.0, x_hi: float = 100.0,
               y_lo: float = 0.0, y_hi: float = 40.0):
    """Return ``(xs, ys, D)`` with D zeros, mirroring B7 geometry."""
    xs = np.linspace(x_lo, x_hi, nx)
    ys = np.linspace(y_lo, y_hi, ny)
    D = np.zeros((ys.size, xs.size), dtype=np.float64)
    return xs, ys, D


def _paint_segment(D: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                   p0: tuple[float, float],
                   p1: tuple[float, float],
                   half_width_mm: float = 0.6,
                   value: float = 1.0) -> None:
    """Paint a thick line segment from p0 to p1 onto ``D`` with d=value."""
    x0, y0 = p0
    x1, y1 = p1
    XG, YG = np.meshgrid(xs, ys)
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    if L2 == 0.0:
        return
    # Project each grid point onto the segment, clamp t in [0,1].
    t = ((XG - x0) * dx + (YG - y0) * dy) / L2
    t = np.clip(t, 0.0, 1.0)
    px = x0 + t * dx
    py = y0 + t * dy
    dist = np.hypot(XG - px, YG - py)
    D[dist <= half_width_mm] = value


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_straight_line_one_arm():
    """A saturated horizontal crack (no branching) must report 1 arm.

    This is the exact false-positive the old ``pass_full_y`` produced
    on B7 run 29126: max(d) hits 1.0 by 75 us yet there is no Y.
    """
    xs, ys, D = _make_grid()
    # Crack along y = 20 from x=0 to x=99 (covers the right-of-precrack
    # region 52 - 100 mm).
    _paint_segment(D, xs, ys, (0.0, 20.0), (99.0, 20.0), half_width_mm=0.6)

    n = compare_b7.count_arms_in_region(D, xs, x_min_arm=52.0)
    assert n == 1, f"expected 1 arm for a straight-line crack, got {n}"


def test_y_shape_two_arms():
    """A Y-shape (left main + 2 right arms) must report 2 right-side arms.

    The pre-fork main lives in ``x < a`` and so is excluded by the
    ``x > a + 2*l0`` cutoff; only the two post-fork arms survive.
    """
    xs, ys, D = _make_grid()
    a = 50.0
    # Pre-crack stem from left edge to fork at (65, 20).
    _paint_segment(D, xs, ys, (0.0, 20.0), (65.0, 20.0))
    # Two arms diverging from the fork.
    _paint_segment(D, xs, ys, (65.0, 20.0), (95.0, 30.0))   # upper arm
    _paint_segment(D, xs, ys, (65.0, 20.0), (95.0, 10.0))   # lower arm

    x_min = a + 2.0 * compare_b7.L0_DEFAULT_MM   # 51.0
    n = compare_b7.count_arms_in_region(D, xs, x_min_arm=x_min)
    assert n == 2, (
        f"expected 2 arms for a Y-shape post-fork, got {n}; "
        f"x_min={x_min}, region cells with d>0.5: "
        f"{int((D[:, xs > x_min] > 0.5).sum())}")


def test_noisy_islands_filtered():
    """Tiny d>0.5 islands must not inflate the arm count."""
    xs, ys, D = _make_grid()
    # Main horizontal crack — 1 arm.
    _paint_segment(D, xs, ys, (0.0, 20.0), (99.0, 20.0), half_width_mm=0.6)
    # Sprinkle 10 single-pixel "islands" in the right region — these
    # are exactly the kind of stress-wave-grazing artefacts the
    # ``min_island=5`` filter is designed to reject.
    rng = np.random.default_rng(seed=314)
    nx_right_lo = int(np.searchsorted(xs, 60.0))
    nx_right_hi = D.shape[1] - 1
    ix = rng.integers(nx_right_lo, nx_right_hi, size=10)
    iy = rng.integers(0, D.shape[0], size=10)
    D[iy, ix] = 1.0
    # Place a single 2x2 cluster (4 cells) to confirm the threshold is
    # strictly < 5, not <= 5: this should also drop out.
    D[5:7, 200:202] = 1.0

    n = compare_b7.count_arms_in_region(
        D, xs, x_min_arm=52.0,
        threshold=0.5, min_island=5,
    )
    assert n == 1, (
        f"expected 1 arm with sub-min_island islands filtered, got {n}")


def test_empty_field_zero_arms():
    """All-zero damage must report 0 arms."""
    xs, ys, D = _make_grid()
    n = compare_b7.count_arms_in_region(D, xs, x_min_arm=52.0)
    assert n == 0, f"expected 0 arms on empty field, got {n}"


def test_three_arms_branched_twice():
    """Three diverging arms must report 3 (sanity for ``n>=2`` test)."""
    xs, ys, D = _make_grid()
    a = 50.0
    _paint_segment(D, xs, ys, (0.0, 20.0), (60.0, 20.0))
    # Upper, middle and lower arms
    _paint_segment(D, xs, ys, (60.0, 20.0), (95.0, 35.0))
    _paint_segment(D, xs, ys, (60.0, 20.0), (95.0, 20.0))
    _paint_segment(D, xs, ys, (60.0, 20.0), (95.0, 5.0))

    x_min = a + 2.0 * compare_b7.L0_DEFAULT_MM
    n = compare_b7.count_arms_in_region(D, xs, x_min_arm=x_min)
    assert n == 3, f"expected 3 arms for triple branch, got {n}"


def test_arm_only_in_left_excluded():
    """Damage entirely in ``x < a`` (not yet propagated past pre-crack)
    reports 0 arms in the right region."""
    xs, ys, D = _make_grid()
    a = 50.0
    # Crack hasn't crossed the precrack tip yet.
    _paint_segment(D, xs, ys, (0.0, 20.0), (40.0, 20.0))

    x_min = a + 2.0 * compare_b7.L0_DEFAULT_MM
    n = compare_b7.count_arms_in_region(D, xs, x_min_arm=x_min)
    assert n == 0, (
        f"expected 0 right-side arms when crack hasn't crossed a, got {n}")


def test_grid_damage_nearest_fallback():
    """When element_connectivity is absent, the nearest-neighbour fallback
    still produces a usable damage grid for the CC pass."""
    # 21x21 nodal grid on [0,20] x [0,20], damage = 1 along y=10.
    nx0 = ny0 = 21
    xn = np.linspace(0.0, 20.0, nx0)
    yn = np.linspace(0.0, 20.0, ny0)
    XN, YN = np.meshgrid(xn, yn)
    coords = np.column_stack([XN.ravel(), YN.ravel()])
    d = (np.abs(YN.ravel() - 10.0) < 0.6).astype(np.float64)

    xs, ys, D = compare_b7._grid_damage(
        coords, None, d,
        x_lo=0.0, x_hi=20.0, y_lo=0.0, y_hi=20.0, dx=0.5,
    )
    assert D.shape == (ys.size, xs.size)
    assert D.max() == 1.0
    assert D.min() == 0.0
    # Damage should sit near the y=10 row; check the row count and that
    # band cells form a single CC.
    n = compare_b7.count_arms_in_region(D, xs, x_min_arm=0.0)
    assert n == 1


def test_compute_morphology_handles_missing_h5(tmp_path):
    """``compute_morphology_timeseries`` returns None when no H5."""
    # No training_data.h5 in tmp_path -> graceful skip.
    out = compare_b7.compute_morphology_timeseries(tmp_path)
    assert out is None


if __name__ == "__main__":
    # Run as plain script: invoke each test_ function in module order.
    import inspect
    failures = []
    for name, obj in inspect.getmembers(sys.modules[__name__],
                                        inspect.isfunction):
        if not name.startswith("test_"):
            continue
        sig = inspect.signature(obj)
        try:
            if "tmp_path" in sig.parameters:
                import tempfile
                with tempfile.TemporaryDirectory() as td:
                    obj(Path(td))
            else:
                obj()
            print(f"[OK]   {name}")
        except AssertionError as exc:
            failures.append((name, str(exc)))
            print(f"[FAIL] {name}: {exc}")
        except Exception as exc:
            failures.append((name, repr(exc)))
            print(f"[ERR]  {name}: {exc!r}")
    if failures:
        print(f"\n{len(failures)} test(s) failed.")
        sys.exit(1)
    print("\nAll morphology tests passed.")
