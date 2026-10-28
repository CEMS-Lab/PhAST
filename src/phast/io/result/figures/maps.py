"""Drawing primitives: maps on 2D meshes, voxel sections and the isometric voxel view."""
from __future__ import annotations

import numpy as np
from matplotlib import colors as mcolors
from matplotlib import colormaps
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.tri import Triangulation

from ..surface import face_normals
from .data import InterfaceBlock, Mesh2D, RunData, node_to_cell, voxel_slice
from .style import DAMAGE_CMAP, DIVERGING_CMAP, SEQUENTIAL_CMAP

C30 = np.cos(np.pi / 6)
SHADE = {"x": 0.88, "y": 0.76, "z": 1.0}     # face brightness: the cell is lit from above
MIN_SECTION_VOXELS = 4


# -- colour limits -----------------------------------------------------------------
def limits(arrays: list[np.ndarray], role: str):
    """(cmap, Normalize) shared by all panels of one quantity."""
    if role == "damage":
        return DAMAGE_CMAP, mcolors.Normalize(0.0, 1.0)
    data = np.concatenate([np.asarray(a, float).ravel() for a in arrays]) if arrays else np.zeros(1)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return SEQUENTIAL_CMAP, mcolors.Normalize(0.0, 1.0)
    lo, hi = np.percentile(data, 0.5), np.percentile(data, 99.5)
    if lo < 0:
        m = max(abs(lo), abs(hi)) or 1e-30
        return DIVERGING_CMAP, mcolors.Normalize(-m, m)
    if hi <= lo:
        hi = lo + (abs(lo) * 1e-6 or 1e-30)
    return SEQUENTIAL_CMAP, mcolors.Normalize(0.0 if lo < 0.2 * hi else lo, hi)


# -- 2D meshes ---------------------------------------------------------------------
def mesh_points(run: RunData, deform: np.ndarray | None = None) -> np.ndarray:
    p = run.mesh2d.points
    if deform is not None:
        p = p + deform
    return p * run.length_scale


def draw_mesh_values(ax, run: RunData, values: np.ndarray, location: str, cmap, norm, deform=None):
    mesh: Mesh2D = run.mesh2d
    p = mesh_points(run, deform)
    if mesh.tri.size == 0:
        art = ax.scatter(p[:, 0], p[:, 1], c=values if location == "node" else None, s=4, cmap=cmap, norm=norm)
    else:
        tri = Triangulation(p[:, 0], p[:, 1], mesh.tri)
        if location == "node":
            art = ax.tripcolor(tri, values, shading="gouraud", cmap=cmap, norm=norm, rasterized=True)
        else:
            art = ax.tripcolor(tri, facecolors=np.asarray(values)[mesh.tri_cell], shading="flat", cmap=cmap,
                               norm=norm, edgecolors="none", rasterized=True)
    _frame_2d(ax, run, p)
    return art


def draw_mesh_cells(ax, run: RunData, facecolours=None, edge="#3A4046", lw=0.25):
    """Cell polygons, optionally filled (one RGBA per cell)."""
    p = mesh_points(run)
    for cells, corners in run.mesh2d.groups:
        fc = "none" if facecolours is None else facecolours[cells]
        ax.add_collection(PolyCollection(p[corners], facecolors=fc, edgecolors=edge, linewidths=lw))
    if not run.mesh2d.groups:
        ax.scatter(p[:, 0], p[:, 1], s=2, color=edge)
    _frame_2d(ax, run, p)


def _frame_2d(ax, run: RunData, p: np.ndarray) -> None:
    lo, hi = p.min(axis=0), p.max(axis=0)
    pad = 0.02 * max(hi - lo)
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal")
    ax.set_xlabel(run.axis_label("x"))
    ax.set_ylabel(run.axis_label("y"))


def auto_deform_scale(run: RunData, u: np.ndarray) -> float:
    size = float(np.max(run.domain_lengths()))
    umax = float(np.nanmax(np.linalg.norm(u, axis=-1))) if u.size else 0.0
    return 0.0 if umax <= 0 else 0.05 * size / umax


# -- voxel sections ------------------------------------------------------------------
PLANES = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}   # (column axis, row axis, normal axis)


def section_planes(run: RunData) -> list[str]:
    """Mid-plane sections whose two in-plane directions have at least four voxels."""
    n = run.shape
    return [p for p, (a, b, _) in PLANES.items() if n[a] >= MIN_SECTION_VOXELS and n[b] >= MIN_SECTION_VOXELS] or ["xy"]


def section_index(run: RunData, plane: str, location: str) -> tuple[int, float]:
    """Mid index along the plane normal and its coordinate (display units)."""
    axis = PLANES[plane][2]
    n = run.shape[axis]
    k = n // 2
    if location == "node":
        coord = run.origin[axis] + k * run.spacing[axis]
    else:
        k = min(k, n - 1)
        coord = run.origin[axis] + (k + 0.5) * run.spacing[axis]
    return k, coord * run.length_scale


def section_extent(run: RunData, plane: str, location: str) -> list[float]:
    a, b, _ = PLANES[plane]
    s = run.length_scale
    ext = []
    for axis in (a, b):
        lo = run.origin[axis]
        hi = lo + run.shape[axis] * run.spacing[axis]
        if location == "node":
            half = 0.5 * run.spacing[axis]
            lo, hi = lo - half, hi + half
        ext += [lo * s, hi * s]
    return ext


def draw_section(ax, run: RunData, grid_values: np.ndarray, plane: str, location: str, cmap, norm):
    k, coord = section_index(run, plane, location)
    img = voxel_slice(grid_values, plane, k)
    art = ax.imshow(img, origin="lower", extent=section_extent(run, plane, location), cmap=cmap, norm=norm,
                    interpolation="nearest", aspect="equal")
    a, b, c = PLANES[plane]
    ax.set_xlabel(run.axis_label("xyz"[a]))
    ax.set_ylabel(run.axis_label("xyz"[b]))
    return art, f"{'xyz'[c]} = {coord:.4g}{(' ' + run.length_label) if run.length_label else ''}"


# -- isometric voxel view (three visible faces, optional cut-away corner) -----------------
def iso(x, y, z):
    """Screen position. Viewer in the +x +y +z octant: x runs to the lower left, y to the lower right, z up."""
    return (y - x) * C30, z - 0.5 * (x + y)


def iso_geometry(run: RunData) -> tuple[bool, float]:
    """(cut-away, z stretch): the near corner is cut away when nz >= 8; thinner cells are
    drawn thicker by zf = max(1, 0.1 max(Lx, Ly) / Lz)."""
    nx, ny, nz = run.shape
    cut = nz >= 8 and nx >= 2 and ny >= 2
    L = run.domain_lengths()
    zf = 1.0 if cut else max(1.0, 0.10 * max(L[0], L[1]) / L[2])
    return cut, zf


def iso_draw(ax, run: RunData, rgba: np.ndarray, cut: bool, zf: float, linec: str = "#16191D") -> dict:
    """Draw the three faces that face the viewer; ``rgba`` is (nz, ny, nx, 4), one colour per voxel."""
    nz, ny, nx = rgba.shape[:3]
    s = run.length_scale
    Lx, Ly, Lz = run.domain_lengths() * s * np.array([1.0, 1.0, zf])
    xe, ye, ze = np.linspace(0, Lx, nx + 1), np.linspace(0, Ly, ny + 1), np.linspace(0, Lz, nz + 1)
    ic, jc, kc = (nx // 2, ny // 2, nz // 2) if cut else (nx, ny, nz)
    pts = []

    def put(A, B, fixed, which, col, hide=None):
        G1, G2 = np.meshgrid(A, B)
        F = np.full_like(G1, fixed)
        X3, Y3, Z3 = {"x": (F, G1, G2), "y": (G1, F, G2), "z": (G1, G2, F)}[which]
        X, Y = iso(X3, Y3, Z3)
        c = np.array(col, float, copy=True)
        c[..., :3] *= SHADE[which]
        if hide is not None:
            c[hide, 3] = 0.0
        pc = ax.pcolormesh(X, Y, np.zeros(c.shape[:2]), shading="flat", antialiased=False, linewidth=0.25,
                           rasterized=True)
        pc.set_array(None)
        pc.set_facecolor(c.reshape(-1, 4))
        pc.set_edgecolor(c.reshape(-1, 4))
        pts.append(np.stack([X.ravel(), Y.ravel()], 1))

    hide_top = np.zeros((ny, nx), bool); hide_top[jc:, ic:] = True
    hide_x = np.zeros((nz, ny), bool); hide_x[kc:, jc:] = True
    hide_y = np.zeros((nz, nx), bool); hide_y[kc:, ic:] = True
    put(xe, ye, Lz, "z", rgba[nz - 1, :, :], hide_top if cut else None)
    put(ye, ze, Lx, "x", rgba[:, :, nx - 1], hide_x if cut else None)
    put(xe, ze, Ly, "y", rgba[:, ny - 1, :], hide_y if cut else None)
    if cut:   # the three faces of the notch
        put(ye[jc:], ze[kc:], xe[ic], "x", rgba[kc:, jc:, ic - 1])
        put(xe[ic:], ze[kc:], ye[jc], "y", rgba[kc:, jc - 1, ic:])
        put(xe[ic:], ye[jc:], ze[kc], "z", rgba[kc - 1, jc:, ic:])
    line = dict(color=linec, lw=0.8, solid_capstyle="round")

    def seg(p, q):
        a, b = iso(*p), iso(*q)
        ax.plot([a[0], b[0]], [a[1], b[1]], **line)

    hexagon = [(Lx, 0, 0), (Lx, Ly, 0), (0, Ly, 0), (0, Ly, Lz), (0, 0, Lz), (Lx, 0, Lz)]
    for a, b in zip(hexagon, hexagon[1:] + hexagon[:1]):
        seg(a, b)
    if cut:
        x0, y0, z0 = xe[ic], ye[jc], ze[kc]
        for p, q in [((x0, y0, z0), (Lx, y0, z0)), ((x0, y0, z0), (x0, Ly, z0)), ((x0, y0, z0), (x0, y0, Lz)),
                     ((Lx, y0, z0), (Lx, Ly, z0)), ((Lx, y0, z0), (Lx, y0, Lz)), ((x0, Ly, z0), (Lx, Ly, z0)),
                     ((x0, Ly, z0), (x0, Ly, Lz)), ((x0, y0, Lz), (Lx, y0, Lz)), ((x0, y0, Lz), (x0, Ly, Lz))]:
            seg(p, q)
    else:
        for q in [(Lx, Ly, 0), (0, Ly, Lz), (Lx, 0, Lz)]:
            seg((Lx, Ly, Lz), q)
    P = np.vstack(pts)
    ax.set_aspect("equal")
    ax.axis("off")
    span = max(Lx, Ly)
    ax.set_xlim(P[:, 0].min() - 0.04 * span, P[:, 0].max() + 0.04 * span)
    ax.set_ylim(P[:, 1].min() - 0.16 * span, P[:, 1].max() + 0.04 * span)
    return dict(L=(Lx, Ly, Lz), P=P)


def iso_axes_labels(ax, g: dict, units: str | None) -> None:
    """Edge lengths, written along the two lower visible edges."""
    Lx, Ly, _ = g["L"]
    d = 0.07 * max(Lx, Ly)
    u = f" {units}" if units else ""
    for p, txt, rot, off in [((Lx, Ly / 2, 0), f"y: {Ly:.4g}{u}", -30, (-0.5, -C30)),
                             ((Lx / 2, Ly, 0), f"x: {Lx:.4g}{u}", 30, (0.5, -C30))]:
        X, Y = iso(*p)
        ax.text(X + off[0] * d, Y + off[1] * d, txt, fontsize=7.5, ha="center", va="center", rotation=rot,
                rotation_mode="anchor", color="#3A4046")


def voxel_rgba(values: np.ndarray, location: str, cmap, norm) -> np.ndarray:
    cell = node_to_cell(values) if location == "node" else values
    return colormaps[cmap](norm(cell)) if isinstance(cmap, str) else cmap(norm(cell))


def draw_points_iso(ax, run: RunData, values, cmap, norm):
    p = run.points3d * run.length_scale
    X, Y = iso(p[:, 0], p[:, 1], p[:, 2])
    art = ax.scatter(X, Y, c=values, s=3, cmap=cmap, norm=norm, linewidths=0)
    ax.set_aspect("equal")
    ax.axis("off")
    return art


# -- unstructured 3D meshes: boundary faces, painter's order, optional cut-away ------------------------
VIEW_DIRECTION = np.ones(3) / np.sqrt(3.0)           # the viewer looks along -(1, 1, 1), as in the voxel view
LIGHT = np.array([0.35, 0.45, 0.82])


def _surface_polygons(run: RunData, cut: bool):
    """Front-facing boundary faces sorted far to near: ``(projected polygons (F, k, 2), nodes, owner cell, brightness)``
    for each face group."""
    P = np.asarray(run.points3d, float) * run.length_scale
    out = []
    for nodes, owner in run.surface3d(cut).polygons():
        n = face_normals(P, nodes)
        front = (n * VIEW_DIRECTION).sum(axis=1) > 1e-9
        nodes, owner, n = nodes[front], owner[front], n[front]
        if not len(nodes):
            continue
        corners = P[nodes]                                                  # (F, k, 3)
        X, Y = iso(corners[..., 0], corners[..., 1], corners[..., 2])
        depth = corners.mean(axis=1).sum(axis=1)
        brightness = 0.62 + 0.38 * np.clip(np.abs((n * (LIGHT / np.linalg.norm(LIGHT))).sum(axis=1)), 0, 1)
        out.append((np.stack([X, Y], axis=-1), nodes, owner, brightness, depth))
    return out


def _paint(ax, groups, colour_of) -> np.ndarray:
    """Draw the groups of faces far to near; ``colour_of(nodes, owner)`` returns one RGBA per face."""
    polys, colours, depth = [], [], []
    for xy, nodes, owner, brightness, d in groups:
        rgba = np.array(colour_of(nodes, owner), float)
        rgba[:, :3] *= brightness[:, None]
        polys.append(xy)
        colours.append(rgba)
        depth.append(d)
    if not polys:
        return np.zeros((0, 2))
    allp = np.concatenate([p if p.shape[1] == 4 else np.concatenate([p, p[:, -1:, :]], axis=1) for p in polys])   # triangles padded to four corners
    allc = np.concatenate(colours)
    order = np.argsort(np.concatenate(depth), kind="stable")
    ax.add_collection(PolyCollection(list(allp[order]), facecolors=allc[order], edgecolors=allc[order], linewidths=0.3))
    return allp.reshape(-1, 2)


def _frame_iso(ax, pts: np.ndarray) -> None:
    ax.set_aspect("equal")
    ax.axis("off")
    if len(pts):
        span = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))) or 1.0
        ax.set_xlim(pts[:, 0].min() - 0.04 * span, pts[:, 0].max() + 0.04 * span)
        ax.set_ylim(pts[:, 1].min() - 0.04 * span, pts[:, 1].max() + 0.04 * span)


def draw_surface_iso(ax, run: RunData, values: np.ndarray, location: str, cmap, norm, cut: bool = False):
    """Boundary faces of an unstructured 3D mesh coloured by a field: a nodal field gives each face the mean of its
    nodes, a cell field the value of the cell the face belongs to. ``cut`` removes the near octant."""
    cm = colormaps[cmap] if isinstance(cmap, str) else cmap
    values = np.asarray(values, float)

    def colour(nodes, owner):
        v = values[nodes].mean(axis=1) if location == "node" else values[owner]
        return cm(norm(v))

    pts = _paint(ax, _surface_polygons(run, cut), colour)
    _frame_iso(ax, pts)
    from matplotlib.cm import ScalarMappable
    return ScalarMappable(norm=norm, cmap=cmap)


def draw_surface_colours(ax, run: RunData, cell_rgba: np.ndarray, cut: bool = False) -> None:
    """Boundary faces coloured by one RGBA per cell (phases)."""
    pts = _paint(ax, _surface_polygons(run, cut), lambda nodes, owner: cell_rgba[owner])
    _frame_iso(ax, pts)


# -- cohesive interfaces ---------------------------------------------------------------------------------
def draw_interfaces_2d(ax, run: RunData, block: InterfaceBlock, values: np.ndarray | None, cmap=None, norm=None,
                       colour: str = "#A3223F", lw: float = 2.6):
    """Interface cells of a 2D mesh as segments on their first side (the solver's ``top`` side), coloured by ``values`` (or one colour)."""
    p = mesh_points(run)
    segs = p[block.nodes[:, :2]]
    lc = LineCollection(segs, linewidths=lw, capstyle="round", zorder=4,
                        **(dict(array=np.asarray(values, float), cmap=cmap, norm=norm) if values is not None
                           else dict(colors=colour)))
    ax.add_collection(lc)
    return lc


def draw_interfaces_iso(ax, run: RunData, block: InterfaceBlock, values: np.ndarray | None, cmap=None, norm=None,
                        colour: str = "#A3223F"):
    """Interface cells of a 3D mesh as quads in the isometric projection, far to near."""
    P = np.asarray(run.points3d, float) * run.length_scale
    corners = P[block.nodes[:, :4]]
    X, Y = iso(corners[..., 0], corners[..., 1], corners[..., 2])
    xy = np.stack([X, Y], axis=-1)
    order = np.argsort(corners.mean(axis=1).sum(axis=1), kind="stable")
    if values is None:
        fc = np.tile(np.array(matplotlib_to_rgba(colour)), (len(xy), 1))
    else:
        fc = (colormaps[cmap] if isinstance(cmap, str) else cmap)(norm(np.asarray(values, float)))
    ax.add_collection(PolyCollection(list(xy[order]), facecolors=fc[order], edgecolors="#16191D", linewidths=0.3, zorder=4))
    from matplotlib.cm import ScalarMappable
    return (ScalarMappable(norm=norm, cmap=cmap) if values is not None else None), xy.reshape(-1, 2)


def matplotlib_to_rgba(colour: str):
    return mcolors.to_rgba(colour)
