"""Unit tests for process_zone diagnostics (issue #258, diagnostics-only)."""

from __future__ import annotations

import numpy as np
import pytest

from phast.research.process_zone import fpz_centroid, fpz_to_l0_ratio, fpz_width


def _bell_1d(centre: float, half_width: float, n: int = 401):
    """1-D Gaussian-ish damage profile on x in [0, 1].

    Built so that d > 0.5 over a band of width ≈ 2*half_width around centre.
    """
    x = np.linspace(0.0, 1.0, n)
    # d = exp(-((x-c)/h)^2 * ln(2)) -> d=0.5 at |x-c|=h
    d = np.exp(-((x - centre) / half_width) ** 2 * np.log(2.0))
    return x.reshape(-1, 1), d


def test_fpz_width_synthetic():
    coords, d = _bell_1d(centre=0.5, half_width=0.05)  # band width ≈ 0.1
    w = fpz_width(d, coords, threshold=0.5)
    h = float(coords[1, 0] - coords[0, 0])
    assert abs(w - 0.1) <= 2.0 * h, f"got width {w}, expected ~0.1 +/- {2 * h}"


def test_fpz_centroid_synthetic():
    coords, d = _bell_1d(centre=0.7, half_width=0.05)
    c = fpz_centroid(d, coords, threshold=0.5)
    h = float(coords[1, 0] - coords[0, 0])
    assert abs(c[0] - 0.7) <= 2.0 * h, f"centroid {c[0]} not near 0.7"


def test_fpz_to_l0_ratio_returns_positive():
    coords, d = _bell_1d(centre=0.5, half_width=0.05)
    r = fpz_to_l0_ratio(d, coords, l0=0.05, threshold=0.5)
    assert r > 0.0
    # bandwidth ≈ 0.1, l0 = 0.05 -> ratio ≈ 2 (consistent with AT1-like range)
    assert 1.0 < r < 4.0


def test_fpz_empty_band_returns_zero_and_nan_centroid():
    coords, d = _bell_1d(centre=0.5, half_width=0.05)
    d_low = 0.1 * d  # everywhere < 0.5
    assert fpz_width(d_low, coords, threshold=0.5) == 0.0
    c = fpz_centroid(d_low, coords, threshold=0.5)
    assert all(np.isnan(v) for v in c)


@pytest.mark.slow
def test_fpz_to_l0_ratio_at1_at2_solver_scan():
    """Placeholder for solver-coupled study (skipped by default).

    Expected ranges from literature (Bourdin/Marigo/Pham; Tanné 2018):
        AT2:  fpz_width / l0  ≈ pi   (≈ 3.14)
        AT1:  fpz_width / l0  ≈ 2    (sharper compact-support profile)

    Realising these requires running the staggered solver with a tensile
    1-D bar, extracting the converged d profile, and calling
    ``fpz_to_l0_ratio``. Out of scope for unit-test runtime.
    """
    pytest.skip("solver-coupled scan; run manually for #258 follow-up")
