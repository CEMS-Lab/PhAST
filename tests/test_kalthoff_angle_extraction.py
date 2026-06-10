"""Synthetic tests for the Kalthoff crack-angle extractor (#235).

Both fixtures are 100x100 binary masks with a thin diagonal/
horizontal "crack". The extractor must recover the angle to
within +/- 2 deg.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# Load the extractor module by path: the script lives under
# examples/ which is not a Python package.
_ROOT = Path(__file__).resolve().parents[1]
_MOD_PATH = (
    _ROOT
    / "examples"
    / "dynamic"
    / "kalthoff"
    / "timing_comparison"
    / "extract_crack_angle.py"
)
_spec = importlib.util.spec_from_file_location(
    "extract_crack_angle", _MOD_PATH
)
extract_crack_angle = importlib.util.module_from_spec(_spec)
sys.modules["extract_crack_angle"] = extract_crack_angle
assert _spec.loader is not None
_spec.loader.exec_module(extract_crack_angle)


def _draw_line_mask(
    n: int, angle_deg: float, thickness: int = 2
) -> np.ndarray:
    """100x100-ish binary mask with a thin line at ``angle_deg``.

    The line passes through the centre. We rasterise by
    walking along the line direction and stamping a small
    square of ``thickness`` pixels.
    """
    mask = np.zeros((n, n), dtype=bool)
    cx = cy = (n - 1) / 2.0
    theta = np.radians(angle_deg)
    dx = np.cos(theta)
    dy = np.sin(theta)
    half_len = 0.4 * n
    ts = np.linspace(-half_len, half_len, int(4 * n))
    for t in ts:
        x = cx + t * dx
        # image-space row grows downward; line angle is from
        # horizontal in physical (y-up) coords -> flip.
        y_img = (n - 1 - (cy + t * dy))
        i = int(round(y_img))
        j = int(round(x))
        for di in range(-thickness, thickness + 1):
            for dj in range(-thickness, thickness + 1):
                ii, jj = i + di, j + dj
                if 0 <= ii < n and 0 <= jj < n:
                    mask[ii, jj] = True
    return mask


@pytest.mark.parametrize(
    "true_angle",
    [70.0, 0.0],
    ids=["seventy_deg", "horizontal"],
)
def test_synthetic_crack_angle(true_angle: float) -> None:
    mask = _draw_line_mask(100, true_angle)
    angle, skel = extract_crack_angle.extract_angle_from_mask(mask)
    assert skel.sum() > 0
    # extractor returns the unsigned acute angle from horizontal
    expected = abs(true_angle)
    assert abs(angle - expected) <= 2.0, (
        f"recovered {angle:.2f} deg, expected {expected:.2f} +/- 2"
    )


def test_fit_angle_simple_slope() -> None:
    xs = np.linspace(0.0, 10.0, 50)
    ys = np.tan(np.radians(45.0)) * xs
    angle = extract_crack_angle.fit_angle_degrees(xs, ys)
    assert abs(angle - 45.0) < 1e-6
