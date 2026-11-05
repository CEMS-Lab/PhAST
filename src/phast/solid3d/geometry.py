"""Voxel microstructure builders for periodic cells: spherical pores, spherical particles with an interface shell, and imported phase arrays."""
from __future__ import annotations

import numpy as np

from phast.solid3d.model import PHASE_ID, VoxelGrid

MATRIX, INCLUSION, INTERFACE, PORE = PHASE_ID["mi_matrix"], PHASE_ID["fibre"], PHASE_ID["bn_band"], PHASE_ID["pore"]
ROLES = {"matrix": MATRIX, "inclusion": INCLUSION, "interface": INTERFACE, "pore": PORE}


def voxel_centres(grid: VoxelGrid):
    """Coordinates (mm) of the voxel centres along x, y and z."""
    nx, ny, nz = grid.shape
    hx, hy, hz = grid.h
    return (np.arange(nx) + 0.5) * hx, (np.arange(ny) + 0.5) * hy, (np.arange(nz) + 0.5) * hz


def sphere_pore(grid: VoxelGrid, diameter_fraction: float) -> np.ndarray:
    """Phase array (nz, ny, nx): matrix with one spherical pore at the cell centre; the diameter is ``diameter_fraction`` of the x length."""
    lx, ly, lz = grid.lengths
    radius = 0.5 * float(diameter_fraction) * lx
    x, y, z = voxel_centres(grid)
    d = np.sqrt((x[None, None, :] - lx / 2) ** 2 + (y[None, :, None] - ly / 2) ** 2 + (z[:, None, None] - lz / 2) ** 2)
    phase = np.full(grid.array_shape, MATRIX, dtype=np.uint8)
    phase[d < radius] = PORE
    return phase


def place_particles(lengths, n: int, radius: float, min_gap: float, seed: int, max_tries: int = 200000) -> np.ndarray:
    """Random sequential addition of ``n`` spheres in a periodic box; returns the (n, 3) centres in mm."""
    rng = np.random.default_rng(seed)
    box = np.asarray(lengths, float)
    centres: list[np.ndarray] = []
    d_min = 2.0 * radius + min_gap
    for _ in range(max_tries):
        c = rng.random(3) * box
        if all(np.sqrt((np.minimum(np.abs(c - o), box - np.abs(c - o)) ** 2).sum()) >= d_min for o in centres):
            centres.append(c)
            if len(centres) == n:
                return np.array(centres)
    raise RuntimeError(f"could not place {n} non-overlapping particles (placed {len(centres)}); lower n or the radius")


def distance_to_nearest_centre(grid: VoxelGrid, centres: np.ndarray):
    """Distance (mm) from every voxel centre to the nearest particle centre (periodic minimum image) and the unit vector pointing away from it."""
    lx, ly, lz = grid.lengths
    x, y, z = voxel_centres(grid)
    best = np.full(grid.array_shape, np.inf)
    vec = np.zeros((*grid.array_shape, 3))
    for c in centres:
        dx = x - c[0]; dx -= lx * np.round(dx / lx)
        dy = y - c[1]; dy -= ly * np.round(dy / ly)
        dz = z - c[2]; dz -= lz * np.round(dz / lz)
        comps = (np.broadcast_to(dx[None, None, :], grid.array_shape), np.broadcast_to(dy[None, :, None], grid.array_shape), np.broadcast_to(dz[:, None, None], grid.array_shape))
        d = np.sqrt(comps[0] ** 2 + comps[1] ** 2 + comps[2] ** 2)
        closer = d < best
        if closer.any():
            best = np.where(closer, d, best)
            for k, comp in enumerate(comps):
                vec[..., k] = np.where(closer, comp, vec[..., k])
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = np.where(best[..., None] > 0, vec / best[..., None], 0.0)
    return best, unit


def particles(grid: VoxelGrid, n: int, radius_um: float, interface_um: float, gap_um: float, seed: int):
    """Phase array, interface-normal field and centres (mm) of ``n`` spherical particles with an interface shell of thickness ``interface_um``
    straddling the particle surface (core radius ``radius_um - interface_um / 2``)."""
    shell = interface_um / 1000.0
    radius = radius_um / 1000.0
    centres = place_particles(grid.lengths, n, radius + 0.5 * shell, gap_um / 1000.0, seed)
    dist, unit = distance_to_nearest_centre(grid, centres)
    phase = np.full(grid.array_shape, MATRIX, dtype=np.uint8)
    phase[dist < radius + 0.5 * shell] = INTERFACE
    phase[dist < radius - 0.5 * shell] = INCLUSION
    tangent = np.zeros((*grid.array_shape, 3))
    tangent[..., 2] = 1.0
    on_shell = phase == INTERFACE
    tangent[on_shell] = unit[on_shell]
    return phase, tangent, centres
