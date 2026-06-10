"""Process-zone (FPZ) post-processing diagnostics for AT1/AT2 PF damage (#258).

Diagnostics-only: characterise the diffuse crack-tip band geometry from a
converged damage field. No solver-side changes. Substantive length-scale
extension (anisotropic L, two-field α-d, delayed activation, PF-CZM bridge)
is out of scope here. ``mesh`` may be ``mesh.nodes``, ``mesh.coordinates``,
or an (N, k) ndarray. ``threshold`` selects the band: nodes with d > threshold.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _coords(mesh) -> np.ndarray:
    """Extract (N, 2) nodal coordinates from a mesh-like object or array."""
    if isinstance(mesh, np.ndarray):
        coords = mesh
    elif hasattr(mesh, "nodes"):
        coords = np.asarray(mesh.nodes)
    elif hasattr(mesh, "coordinates"):
        coords = np.asarray(mesh.coordinates)
    else:
        raise TypeError(
            "mesh must expose .nodes / .coordinates, or be an (N, k) ndarray"
        )
    return np.atleast_2d(coords)


def _band_mask(d_field, threshold: float) -> np.ndarray:
    d = np.asarray(d_field).reshape(-1)
    return d > threshold


def fpz_centroid(d_field, mesh, threshold: float = 0.5) -> Tuple[float, ...]:
    """Centroid of the d > threshold band; doubles as a crack-tip locator."""
    coords = _coords(mesh)
    mask = _band_mask(d_field, threshold)
    if not mask.any():
        return tuple(np.full(coords.shape[1], np.nan))
    weights = np.asarray(d_field).reshape(-1)[mask]
    pts = coords[mask]
    c = (weights[:, None] * pts).sum(axis=0) / weights.sum()
    return tuple(float(x) for x in c)


def fpz_width(d_field, mesh, threshold: float = 0.5) -> float:
    """Width of the d > threshold band along the dominant principal axis.

    Computes PCA on the band; the bandwidth is the extent (max-min projected
    coordinate) along the leading principal direction. For 1-D inputs this
    reduces to ``x.max() - x.min()`` over the band, matching the bell-curve
    FWHM for d=exp(-((x-x0)/w)^2) at threshold≈0.5 within one grid spacing.
    """
    coords = _coords(mesh)
    mask = _band_mask(d_field, threshold)
    n = int(mask.sum())
    if n == 0:
        return 0.0
    pts = coords[mask]
    if pts.shape[1] == 1 or n == 1:
        return float(pts[:, 0].max() - pts[:, 0].min())
    centred = pts - pts.mean(axis=0, keepdims=True)
    # leading principal direction via SVD (robust for n >= 2)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axis = vt[0]
    proj = centred @ axis
    return float(proj.max() - proj.min())


def fpz_to_l0_ratio(d_field, mesh, l0: float, threshold: float = 0.5) -> float:
    """Bandwidth / l0. Expected ~O(1) for AT2, ~O(2-4) for AT1 (lit. ranges)."""
    if l0 <= 0:
        raise ValueError("l0 must be positive")
    return fpz_width(d_field, mesh, threshold=threshold) / float(l0)
