"""Crack-tip table of layout 2.1 (``history/crack_tips/``) and the crack speed derived from it.

The speed is computed as ``phast.utils.postprocess_paper`` computes the
crack-tip velocity of the legacy ``crack_tip.csv``: the tip position is
smoothed with a 5-point Savitzky-Golay filter (order 2) before it is
differentiated, so snapshot-level jumps of the position (one element width)
do not appear as velocity spikes. Here the smoothing acts on every
coordinate of each tip separately and the speed is the length of the
backward difference of the smoothed position divided by the time step
between the two rows; the first row of a tip has no speed (NaN).
"""
from __future__ import annotations

from typing import Any

import numpy as np

FIT_METHOD = "savitzky_golay_position_then_backward_difference"
FIT_WINDOW = 5
FIT_POLYORDER = 2


def smooth_positions(x: np.ndarray, window: int = FIT_WINDOW, polyorder: int = FIT_POLYORDER) -> np.ndarray:
    """Savitzky-Golay smoothing of one coordinate; unchanged when it has fewer than ``window`` samples.

    Without scipy a moving average of the same width is used (as the legacy
    post-processor does).
    """
    x = np.asarray(x, float)
    if x.size < window:
        return x.copy()
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(x, window_length=window, polyorder=polyorder, mode="interp")
    except ImportError:
        half = window // 2
        pad = np.concatenate([np.full(half, x[0]), x, np.full(half, x[-1])])
        return np.convolve(pad, np.ones(window) / window, mode="valid")


def fit_speed(tip_id: Any, position: Any, time: Any, window: int = FIT_WINDOW,
              polyorder: int = FIT_POLYORDER) -> np.ndarray:
    """Crack speed (mm/s if the positions are in mm and the times in s) of every row of a tip table.

    ``tip_id`` (n,), ``position`` (n, dim) and ``time`` (n,) are the table
    rows in increment order; each tip is followed separately.
    """
    tip_id = np.asarray(tip_id)
    position = np.asarray(position, float)
    time = np.asarray(time, float)
    speed = np.full(len(tip_id), np.nan)
    for tid in np.unique(tip_id):
        rows = np.flatnonzero(tip_id == tid)
        smooth = np.stack([smooth_positions(position[rows, c], window, polyorder) for c in range(position.shape[1])], axis=1)
        dt = np.diff(time[rows])
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.linalg.norm(np.diff(smooth, axis=0), axis=1) / dt
        speed[rows[1:]] = np.where(dt > 0, step, np.nan)
    return speed


def locate_tip(mesh: Any, d: Any, threshold: float = 0.5, direction: str = "x") -> tuple[float, float] | None:
    """Position ``(x, y)`` of the crack tip of a T3 mesh, or ``None`` when no edge crosses the threshold.

    The tip is the point of the ``d = threshold`` isoline that is farthest
    along ``direction`` (``physics.fracture_mechanics.find_crack_tip``, which
    interpolates the isoline on the cell edges). It handles one tip and T3
    cells only.
    """
    from phast.physics.fracture_mechanics import find_crack_tip

    d_e = d[mesh.elements]
    crossing = False
    for a, b in ((0, 1), (1, 2), (2, 0)):
        crossing = crossing or bool((((d_e[:, a] - threshold) * (d_e[:, b] - threshold)) < 0).any())
    if not crossing:
        return None
    tip, _alpha = find_crack_tip(mesh, d, direction=direction, threshold=threshold)
    return float(tip[0]), float(tip[1])
