"""Boundary surfaces of 3D cells and voxel grids, shared by the figures and the web export.

The surface of an unstructured mesh is the set of cell faces that belong to exactly one cell; the surface of a voxel
grid is the set of voxel faces between a present voxel and an absent one (outside the grid, or removed by a cut-away).
Every face is returned with the cell that owns it and with its nodes ordered so that the normal points out of the body.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# faces of the VTK cell types, listed by their corner nodes
VTK_FACES = {
    12: ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)),     # hexahedron
    10: ((0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)),                                            # tetrahedron
    24: ((0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)),                                            # quadratic tetrahedron, corners
}
VTK_CORNERS_3D = {12: 8, 10: 4, 24: 4}


@dataclass
class Surface:
    """Boundary faces: ``tris`` (F3, 3) and ``quads`` (F4, 4) node indices with their owner cells."""
    tris: np.ndarray
    tri_cell: np.ndarray
    quads: np.ndarray
    quad_cell: np.ndarray

    @property
    def n_faces(self) -> int:
        return int(len(self.tris) + len(self.quads))

    def polygons(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """``[(nodes (F, k), owner cell (F,)), ...]`` for the non-empty groups."""
        return [(n, c) for n, c in ((self.tris, self.tri_cell), (self.quads, self.quad_cell)) if len(n)]

    def triangles(self) -> tuple[np.ndarray, np.ndarray]:
        """All faces as triangles (a quad is split along its first diagonal) and the owner cell of each."""
        q = self.quads
        split = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]) if len(q) else np.zeros((0, 3), np.int64)
        return (np.concatenate([self.tris, split]).astype(np.int64),
                np.concatenate([self.tri_cell, self.quad_cell, self.quad_cell]).astype(np.int64))


def cells_by_type(connectivity: np.ndarray, offsets: np.ndarray, types: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """``{vtk type: (cell ids, corner nodes (n, k))}`` for the 3D cell types this module knows."""
    out = {}
    types = np.asarray(types)
    for t, k in VTK_CORNERS_3D.items():
        ids = np.nonzero(types == t)[0]
        if ids.size:
            out[t] = (ids, np.asarray(connectivity, np.int64)[np.asarray(offsets, np.int64)[ids][:, None] + np.arange(k)])
    return out


def boundary_surface(points: np.ndarray, groups: dict[int, tuple[np.ndarray, np.ndarray]],
                     keep: np.ndarray | None = None) -> Surface:
    """Faces that appear once, oriented outward. ``keep`` (bool per cell id) removes cells before the search."""
    points = np.asarray(points, float)
    by_k: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {3: [], 4: []}
    for t, (ids, nodes) in groups.items():
        if keep is not None:
            sel = keep[ids]
            ids, nodes = ids[sel], nodes[sel]
        if not len(ids):
            continue
        for face in VTK_FACES[t]:
            by_k[len(face)].append((nodes[:, list(face)], ids))
    result = {}
    for k, parts in by_k.items():
        if not parts:
            result[k] = (np.zeros((0, k), np.int64), np.zeros(0, np.int64))
            continue
        faces = np.concatenate([p[0] for p in parts])
        owner = np.concatenate([p[1] for p in parts])
        key = np.sort(faces, axis=1)
        _, first, counts = np.unique(key, axis=0, return_index=True, return_counts=True)
        once = first[counts == 1]
        result[k] = (faces[once], owner[once])
    surf = Surface(result[3][0], result[3][1], result[4][0], result[4][1])
    return orient_outward(points, surf, groups)


def orient_outward(points: np.ndarray, surf: Surface, groups: dict[int, tuple[np.ndarray, np.ndarray]]) -> Surface:
    """Flip faces whose normal points toward the centre of their owner cell."""
    centre = {}
    for ids, nodes in groups.values():
        c = points[nodes].mean(axis=1)
        for i, row in zip(ids.tolist(), c):
            centre[i] = row
    out = []
    for nodes, owner in surf.polygons():
        n = face_normals(points, nodes)
        cell_c = np.array([centre[i] for i in owner.tolist()])
        face_c = points[nodes].mean(axis=1)
        flip = np.einsum("ij,ij->i", n, face_c - cell_c) < 0
        fixed = nodes.copy()
        fixed[flip] = nodes[flip][:, ::-1]
        out.append(fixed)
    it = iter(out)
    tris = next(it) if len(surf.tris) else surf.tris
    quads = next(it) if len(surf.quads) else surf.quads
    return Surface(tris, surf.tri_cell, quads, surf.quad_cell)


def face_normals(points: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Unit normals of polygons (triangles or quads) from the cross product of their diagonals."""
    p = points[nodes]
    if nodes.shape[1] == 3:
        n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    else:
        n = np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.where(length > 0, length, 1.0)


def near_octant_removed(points: np.ndarray, groups: dict[int, tuple[np.ndarray, np.ndarray]], n_cells: int) -> np.ndarray:
    """Bool per cell id: False for the cells whose centroid lies in the octant nearest the viewer (+x, +y, +z of the
    centre of the bounding box), which the cut-away views remove."""
    points = np.asarray(points, float)
    mid = 0.5 * (points.min(axis=0) + points.max(axis=0))
    keep = np.ones(n_cells, bool)
    for ids, nodes in groups.values():
        centroid = points[nodes].mean(axis=1)
        keep[ids] = ~np.all(centroid > mid, axis=1)
    return keep


# --- voxel grids ---------------------------------------------------------------------------------------------

def grid_surface(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Faces of the present voxels (``mask`` is ``(nz, ny, nx)`` bool) that border an absent voxel or the outside.

    Returns ``(nodes (F, 4), voxel (F, 3), axis (F,))``: node indices on the ``(nz + 1, ny + 1, nx + 1)`` point lattice
    (x fastest), ordered counter-clockwise seen from outside; the ``(k, j, i)`` index of the voxel the face belongs to;
    and the face direction: 0, 1, 2 for a normal along +x, +y, +z and 3, 4, 5 for -x, -y, -z.
    """
    mask = np.asarray(mask, bool)
    nz, ny, nx = mask.shape
    pad = np.pad(mask, 1)
    pid = lambda i, j, k: i + (nx + 1) * (j + (ny + 1) * k)
    all_nodes, all_vox, all_axis = [], [], []
    for axis in range(3):
        for high in (True, False):
            shift = [0, 0, 0]
            shift[2 - axis] = 1 if high else -1                  # array axes are (z, y, x)
            nb = pad[1 + shift[0]:1 + shift[0] + nz, 1 + shift[1]:1 + shift[1] + ny, 1 + shift[2]:1 + shift[2] + nx]
            k, j, i = np.nonzero(mask & ~nb)
            if not len(k):
                continue
            if axis == 0:                                         # normal along x
                x = i + (1 if high else 0)
                q = [(x, j, k), (x, j + 1, k), (x, j + 1, k + 1), (x, j, k + 1)]
            elif axis == 1:
                y = j + (1 if high else 0)
                q = [(i, y, k), (i, y, k + 1), (i + 1, y, k + 1), (i + 1, y, k)]
            else:
                z = k + (1 if high else 0)
                q = [(i, j, z), (i + 1, j, z), (i + 1, j + 1, z), (i, j + 1, z)]
            xyz = np.stack([np.stack(c, axis=1) for c in q], axis=1).astype(float)         # (F, 4, 3) lattice coordinates
            n = np.cross(xyz[:, 2] - xyz[:, 0], xyz[:, 3] - xyz[:, 1])
            wanted = (1.0 if high else -1.0) * np.eye(3)[axis]
            flip = (n * wanted).sum(axis=1) < 0
            nodes = np.stack([pid(*c) for c in q], axis=1)
            nodes[flip] = nodes[flip][:, ::-1]
            all_nodes.append(nodes)
            all_vox.append(np.stack([k, j, i], axis=1))
            all_axis.append(np.full(len(k), axis + (0 if high else 3)))
    if not all_nodes:
        return np.zeros((0, 4), np.int64), np.zeros((0, 3), np.int64), np.zeros(0, np.int64)
    return np.concatenate(all_nodes).astype(np.int64), np.concatenate(all_vox).astype(np.int64), np.concatenate(all_axis)
