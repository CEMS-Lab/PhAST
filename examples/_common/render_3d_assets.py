#!/usr/bin/env python3
"""Render the picture set of a 3D showcase example from a finished run.

Reads the ``result.vtkhdf`` of a run (or the older ``accepted_fields.h5`` with ``history.csv``): nodal damage, element phase labels and the macro history. It writes,
into the example folder:

    initial_conditions.png   the voxel mesh (phases), a mid-plane with every voxel, and the boundary conditions
    damage_evolution.gif / .mp4        the cut-away block coloured by damage next to the stress-strain curve, one frame per stored state
    stress_strain.png                  the macro stress-strain curve of every increment, the stored states marked
    damage_final.png, thumbnail.png    the last stored state
    history.csv                        step, macro strain, macro stress and the energy-balance residual of every increment
    visual_manifest.json               one row per picture (the shared PhAST schema)
    run_manifest.json                  what was drawn, from which file (SHA-256), with which settings

Usage (from the repository root)::

    python examples/_common/render_3d_assets.py pore_in_cube   --run-dir <run folder> --out examples/three_dimensional/pore_in_cube/assets
    python examples/_common/render_3d_assets.py particles_in_cube --run-dir <run folder> --out examples/three_dimensional/particles_in_cube/assets
    python examples/_common/render_3d_assets.py fibre_unit_cell --run-dir <run folder> --out examples/three_dimensional/fibre_unit_cell/assets

The block is drawn with a cut-away corner (x > L/2, y < L/2, z > L/2 removed) so that the interior planes through the cell centre are visible.
Faces are coloured per voxel; no smoothing, no interpolation of the damage field beyond the average of the eight nodal values of an element.
A cell that is uniform along z (the fibre cell, two voxels thick) is drawn repeated along z so that it reads as a block; the manifest says so.
Output PNG files are kept under 2000 px per side.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9, "axes.linewidth": 0.7})

PHASE_FIBRE, PHASE_BAND, PHASE_MATRIX, PHASE_PORE = 0, 1, 3, 4
INK = "#1d2026"
DMG = LinearSegmentedColormap.from_list("damage", ["#dcdfe4", "#f4dc8a", "#ef9a3a", "#c9302c", "#4d0f1f"])
SHADE = {"z": 1.00, "y": 0.88, "x": 0.76}      # lighting factor per face orientation

CASES = {
    "pore_in_cube": dict(
        title="Spherical pore in a cube",
        phases={PHASE_MATRIX: ("matrix", "#dcdfe4"), PHASE_PORE: ("pore", "#3b4048")},
        reps=1,
        bc_note="A pore of diameter 0.4 L at the cell centre; the matrix is isotropic and elastic until damage.",
    ),
    "particles_in_cube": dict(
        title="Spherical particles with a weak interface shell in a cube",
        phases={PHASE_FIBRE: ("particle", "#2c58a0"), PHASE_BAND: ("interface", "#d4a373"), PHASE_MATRIX: ("matrix", "#dcdfe4")},
        reps=1,
        bc_note="Four particles (radius 5 µm) in a periodic cube; the interface shell has a lower strength (63 MPa against 600 MPa) and a much lower fracture energy than the matrix.",
    ),
    "fibre_unit_cell": dict(
        title="Nine-fibre unit cell under transverse tension",
        phases={PHASE_FIBRE: ("fibre", "#2c58a0"), PHASE_BAND: ("interface", "#d4a373"), PHASE_MATRIX: ("matrix", "#dcdfe4")},
        reps=18,
        bc_note="Nine parallel fibres along z. The cell is two voxels thick and uniform along z, so the response is the transverse plane problem with 3D elements.",
    ),
}


# ------------------------------------------------------------------ data
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


class Run:
    """The parts of a finished run that the pictures need.

    Reads ``result.vtkhdf`` (the PhAST result file, through ``phast.load_result``) when the folder has one, otherwise the older
    ``accepted_fields.h5`` with ``history.csv`` that the runs of the shipped pictures wrote."""

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        if (self.dir / "result.vtkhdf").exists():
            self._read_result_file()
        elif (self.dir / "accepted_fields.h5").exists():
            self._read_legacy()
        else:
            raise FileNotFoundError(f"{self.dir} holds neither result.vtkhdf nor accepted_fields.h5")
        self.h_strain = np.array([float(r["macro_strain_xx"]) for r in self.history]) * 100.0
        self.h_stress = np.array([float(r["macro_stress_xx"]) for r in self.history])

    def _read_result_file(self):
        import phast
        self.file = self.dir / "result.vtkhdf"
        r = phast.load_result(str(self.file))
        grid = r.grid()
        self.nx, self.ny, self.nz = (int(v) for v in grid["shape"])
        self.h_um, self.hz_um = float(grid["spacing"][0]) * 1000.0, float(grid["spacing"][2]) * 1000.0
        self.steps = [int(v) for v in r.steps()]
        self.phase = np.asarray(r.field("phase_id", self.steps[0])).reshape(self.nz, self.ny, self.nx)
        self._damage = [np.asarray(r.field("damage", s)).reshape(self.nz + 1, self.ny + 1, self.nx + 1) for s in self.steps]
        self.history = [{k: row[k] for k in ("step", "macro_strain_xx", "macro_stress_xx", "balance_residual_rel")} for row in r.history("history")]
        by_step = {int(row["step"]): row for row in self.history}
        self.strain = np.array([float(by_step[s]["macro_strain_xx"]) for s in self.steps]) * 100.0
        self.stress = np.array([float(by_step[s]["macro_stress_xx"]) for s in self.steps])

    def _read_legacy(self):
        self.file = self.dir / "accepted_fields.h5"
        with h5py.File(self.file, "r") as h5:
            g = h5["simulation_data"]
            self.nx, self.ny, self.nz = (int(v) for v in g["mesh"].attrs["grid_shape"])
            self.h_um = float(np.asarray(g["mesh"].attrs["voxel_size_mm"])[0]) * 1000.0
            self.hz_um = float(np.asarray(g["mesh"].attrs["voxel_size_mm"])[2]) * 1000.0
            self.phase = np.asarray(g["material/phase_id"]).reshape(self.nz, self.ny, self.nx)
            self.steps = [int(v) for v in np.asarray(g["trajectory/step"])]
            self.strain = np.asarray(g["trajectory/macro_strain_xx"]) * 100.0
            self.stress = np.asarray(g["trajectory/macro_stress_xx"])
            flat = np.asarray(g["fields/damage"])
        self._damage = [row.reshape(self.nz + 1, self.ny + 1, self.nx + 1) for row in flat]
        with (self.dir / "history.csv").open() as stream:
            self.history = list(csv.DictReader(stream))

    def element_damage(self, k: int) -> np.ndarray:
        d = self._damage[k]
        nz, ny, nx = self.nz, self.ny, self.nx
        return sum(d[a:nz + a, b:ny + b, c:nx + c] for a in (0, 1) for b in (0, 1) for c in (0, 1)) / 8.0


# ------------------------------------------------------------------ the cut-away block
def _face(face, i, j, fixed, vxy, vz):
    """Quads (N, 4, 3) of the voxel faces (i, j) of one plane; ``fixed`` is the plane position in voxel units of its own normal axis."""
    i = i.astype(float); j = j.astype(float); f = np.full_like(i, float(fixed))
    if face == "z":
        c = [(i * vxy, j * vxy, f * vz), ((i + 1) * vxy, j * vxy, f * vz), ((i + 1) * vxy, (j + 1) * vxy, f * vz), (i * vxy, (j + 1) * vxy, f * vz)]
    elif face == "y":
        c = [(i * vxy, f * vxy, j * vz), ((i + 1) * vxy, f * vxy, j * vz), ((i + 1) * vxy, f * vxy, (j + 1) * vz), (i * vxy, f * vxy, (j + 1) * vz)]
    else:
        c = [(f * vxy, i * vxy, j * vz), (f * vxy, (i + 1) * vxy, j * vz), (f * vxy, (i + 1) * vxy, (j + 1) * vz), (f * vxy, i * vxy, (j + 1) * vz)]
    return np.stack([np.stack(p, axis=-1) for p in c], axis=1)


def block(ax, run: Run, colour, reps: int, elev=22, azim=-60, zoom=1.05):
    """Draw the nx x ny x (nz*reps) block with the corner x > L/2, y < L/2, z > L/2 removed.

    ``colour(layer, rows, cols)`` returns RGB colours for the voxels (layer, row, col) of the stored stack (layer = z index modulo nz)."""
    nx, ny, nz = run.nx, run.ny, run.nz
    NZ = nz * reps
    hx, hy, hz = nx // 2, ny // 2, NZ // 2
    vxy, vz = run.h_um, run.hz_um
    parts = []

    def add(face, a, b, fixed, pick, keep=None):
        ia, ib = np.meshgrid(np.arange(*a), np.arange(*b), indexing="ij"); ia, ib = ia.ravel(), ib.ravel()
        if keep is not None:
            m = keep(ia, ib); ia, ib = ia[m], ib[m]
        if ia.size:
            parts.append((_face(face, ia, ib, fixed, vxy, vz), np.clip(pick(ia, ib) * SHADE[face], 0, 1)))

    add("z", (0, nx), (0, ny), NZ, lambda i, j: colour(nz - 1, j, i), lambda a, b: ~((a >= hx) & (b < hy)))                    # top
    add("y", (0, nx), (0, NZ), 0, lambda i, j: colour(j % nz, np.zeros_like(i), i), lambda a, b: ~((a >= hx) & (b >= hz)))     # front
    add("x", (0, ny), (0, NZ), nx, lambda i, j: colour(j % nz, i, np.full_like(i, nx - 1)), lambda a, b: ~((a < hy) & (b >= hz)))   # right
    add("z", (hx, nx), (0, hy), hz, lambda i, j: colour((hz - 1) % nz, j, i))                                                  # floor of the cut
    add("x", (0, hy), (hz, NZ), hx, lambda i, j: colour(j % nz, i, np.full_like(i, hx - 1)))                                   # wall x = L/2
    add("y", (hx, nx), (hz, NZ), hy, lambda i, j: colour(j % nz, np.full_like(i, hy), i))                                      # wall y = L/2
    quads = np.concatenate([p for p, _ in parts]); cols = np.concatenate([c for _, c in parts])
    ax.add_collection3d(Poly3DCollection(quads, facecolors=cols, edgecolors=cols, linewidths=0.25))
    Lx, Ly, Lz = nx * vxy, ny * vxy, NZ * vz
    ax.set_xlim(0, Lx); ax.set_ylim(0, Ly); ax.set_zlim(0, Lz)
    ax.set_box_aspect((Lx, Ly, Lz), zoom=zoom); ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
    # outline of the block and of the cut
    kw = dict(color="0.25", lw=0.6)
    x2, y2, z2 = Lx / 2, Ly / 2, Lz / 2
    for a, b in [((0, 0, 0), (Lx, 0, 0)), ((Lx, 0, 0), (Lx, Ly, 0)), ((Lx, Ly, 0), (Lx, Ly, z2)), ((0, 0, 0), (0, 0, Lz)), ((0, 0, Lz), (x2, 0, Lz)),
                 ((0, 0, Lz), (0, Ly, Lz)), ((0, Ly, Lz), (x2, Ly, Lz)), ((x2, Ly, Lz), (Lx, Ly, Lz)), ((Lx, Ly, Lz), (Lx, y2, Lz)), ((0, 0, z2), (x2, 0, z2)),
                 ((x2, 0, z2), (x2, y2, z2)), ((x2, y2, z2), (Lx, y2, z2)), ((Lx, y2, z2), (Lx, 0, z2)), ((x2, 0, z2), (x2, 0, Lz)), ((x2, y2, z2), (x2, y2, Lz)),
                 ((Lx, y2, z2), (Lx, y2, Lz)), ((Lx, 0, z2), (Lx, 0, 0)), ((Lx, Ly, z2), (Lx, Ly, Lz)), ((0, Ly, 0), (0, Ly, Lz))]:
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], **kw)
    return Lx, Ly, Lz


def colour_by_damage(run: Run, k: int, spec):
    d = run.element_damage(k); phase = run.phase
    pore = to_rgb(spec["phases"].get(PHASE_PORE, ("", "#000000"))[1]); fib = to_rgb(spec["phases"].get(PHASE_FIBRE, ("", "#000000"))[1])

    def f(layer, rows, cols):
        col = DMG(np.clip(d[layer, rows, cols], 0, 1))[:, :3]
        p = phase[layer, rows, cols]
        col[p == PHASE_PORE] = pore
        col[p == PHASE_FIBRE] = fib
        return col
    return f


def colour_by_phase(run: Run, spec):
    table = {pid: np.array(to_rgb(c)) for pid, (_, c) in spec["phases"].items()}
    phase = run.phase

    def f(layer, rows, cols):
        p = phase[layer, rows, cols]
        out = np.zeros((p.size, 3))
        for pid, c in table.items():
            out[p == pid] = c
        return out
    return f


def phase_legend(spec):
    return [Patch(fc=c, ec="0.5", lw=0.5, label=n) for n, c in spec["phases"].values()]


# ------------------------------------------------------------------ figures
def figure_mesh_bc(run: Run, spec, out: Path, case: str):
    fig = plt.figure(figsize=(18.0, 6.0), dpi=100)
    L = run.nx * run.h_um
    thick = run.nz * spec["reps"] * run.hz_um
    # (a) phases on the block
    a = fig.add_axes([0.00, 0.17, 0.34, 0.76], projection="3d")
    block(a, run, colour_by_phase(run, spec), spec["reps"])
    fig.text(0.015, 0.95, "(a) Phases on the cut-away block", fontsize=11, ha="left", va="center", color=INK)
    a.legend(handles=phase_legend(spec), loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=len(spec["phases"]), frameon=False, fontsize=9)
    # (b) mid-plane with every voxel
    b = fig.add_axes([0.385, 0.24, 0.27, 0.66])
    counts = (run.phase != PHASE_MATRIX).sum(axis=(1, 2))                  # the layer with the most non-matrix voxels (through a particle or the pore)
    layer = int(np.argmax(counts)); plane = run.phase[layer]
    cmap = {pid: np.array(to_rgb(c)) for pid, (_, c) in spec["phases"].items()}
    img = np.zeros((*plane.shape, 3))
    for pid, c in cmap.items():
        img[plane == pid] = c
    b.imshow(img, origin="lower", extent=(0, run.nx * run.h_um, 0, run.ny * run.h_um), interpolation="nearest")
    xs = np.arange(0, run.nx + 1) * run.h_um
    for x in xs:
        b.axvline(x, color="0.25", lw=0.08, alpha=0.5); b.axhline(x, color="0.25", lw=0.08, alpha=0.5)
    b.set_xlabel("x (µm)"); b.set_ylabel("y (µm)"); b.set_aspect("equal")
    for s in ("top", "right"):
        b.spines[s].set_visible(False)
    fig.text(0.385, 0.95, f"(b) Layer z = {(layer + 0.5) * run.hz_um:.1f} µm: every voxel is one element ({run.nx} × {run.ny} voxels of {run.h_um:g} µm)", fontsize=11, ha="left", va="center", color=INK)
    # (c) boundary conditions
    c = fig.add_axes([0.70, 0.17, 0.28, 0.76], projection="3d")
    bc_schematic(c, L, thick)
    c.legend(handles=BC_LEGEND, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=8.8, handlelength=2.2, columnspacing=1.4)
    fig.text(0.70, 0.95, "(c) Boundary conditions and loading", fontsize=11, ha="left", va="center", color=INK)
    n_el = run.nx * run.ny * run.nz
    cap = (f"{spec['title']}. Cell {L:g} × {L:g} × {run.nz * run.hz_um:g} µm, {run.nx} × {run.ny} × {run.nz} = {n_el:,} eight-node hexahedral elements. "
           f"{spec['bc_note']} " + (f"The block is drawn {thick:g} µm thick by repeating the stored layers {spec['reps']} times along z. " if spec["reps"] > 1 else "") +
           f"Boundary conditions: fully periodic in x, y and z; the macro strain E_xx is prescribed in increments and the other five macro stress components are zero; one node is pinned.")
    fig.text(0.015, 0.012, textwrap.fill(cap, 215), fontsize=8.6, ha="left", va="bottom", color="0.25")
    _fit_png(fig, out / "initial_conditions.png")


BC_LEGEND = [Line2D([0], [0], color="#c9302c", lw=2.2, marker=">", ms=6, label=r"x faces: macro strain $E_{xx}$ prescribed"),
             Line2D([0], [0], color="#2c58a0", lw=1.6, ls="--", label="y faces: periodic pair"),
             Line2D([0], [0], color="#3a8f5c", lw=1.6, ls="--", label="z faces: periodic pair"),
             Line2D([0], [0], color="black", marker="o", ls="none", ms=5, label="pinned node (rigid translation)")]


def bc_schematic(ax, L, thick):
    """A box with the three face pairs: the x pair carries the prescribed macro strain, the y and z pairs are identified by periodicity."""
    T = thick
    ax.set_axis_off(); ax.set_box_aspect((L, L, T), zoom=0.86); ax.view_init(elev=22, azim=-60)
    for fixed, axis, colour, alpha in ((0, 0, "#c9302c", 0.22), (L, 0, "#c9302c", 0.22), (0, 1, "#2c58a0", 0.12), (L, 1, "#2c58a0", 0.12), (0, 2, "#3a8f5c", 0.12), (T, 2, "#3a8f5c", 0.12)):
        corners = {0: [(fixed, 0, 0), (fixed, L, 0), (fixed, L, T), (fixed, 0, T)],
                   1: [(0, fixed, 0), (L, fixed, 0), (L, fixed, T), (0, fixed, T)],
                   2: [(0, 0, fixed), (L, 0, fixed), (L, L, fixed), (0, L, fixed)]}[axis]
        ax.add_collection3d(Poly3DCollection([corners], facecolors=colour, alpha=alpha, edgecolors="0.3", linewidths=0.6))
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_zlim(0, T)
    s = 0.24 * L
    for x0, sign in ((0, -1), (L, 1)):
        ax.quiver(x0, L / 2, T / 2, sign * s, 0, 0, color="#c9302c", lw=2.2, arrow_length_ratio=0.3)
    ax.plot([L / 2, L / 2], [0, L], [T, T], color="#2c58a0", lw=1.6, ls="--")
    ax.plot([L / 2, L / 2], [0, L], [0, 0], color="#2c58a0", lw=1.6, ls="--")
    ax.plot([0, 0], [L / 2, L / 2], [0, T], color="#3a8f5c", lw=1.6, ls="--")
    ax.scatter([0], [0], [0], color="black", s=26, depthshade=False)


def figure_curve(run: Run, spec, out: Path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    ax.plot(run.h_strain, run.h_stress, color=INK, lw=1.2, label="every increment")
    ax.plot(run.strain, run.stress, ls="none", marker="o", ms=3.2, color="#c9302c", label=f"{len(run.steps)} stored states")
    ax.set_xlabel(r"macro strain $E_{xx}$ (%)"); ax.set_ylabel(r"macro stress $\Sigma_{xx}$ (MPa)")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0); ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(spec["title"], loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "stress_strain.png"); plt.close(fig)


def frame_figure(run: Run, spec, k: int, size=(9.6, 4.8), dpi=100):
    fig = plt.figure(figsize=size, dpi=dpi)
    a = fig.add_axes([0.0, 0.04, 0.55, 0.84], projection="3d")
    block(a, run, colour_by_damage(run, k, spec), spec["reps"])
    fig.text(0.015, 0.95, spec["title"], fontsize=11, ha="left", va="center", color=INK)
    sub = "damage on the cut-away block (periodic cell, macro strain control)"
    if spec["reps"] > 1:
        sub += f"; stored thickness {run.nz * run.hz_um:g} µm repeated {spec['reps']} times along z"
    fig.text(0.015, 0.905, sub, fontsize=8.5, ha="left", va="center", color="0.35")
    c = fig.add_axes([0.63, 0.40, 0.34, 0.45])
    c.plot(run.h_strain, run.h_stress, color="0.72", lw=1.0)
    upto = run.h_strain <= run.strain[k] + 1e-12
    c.plot(run.h_strain[upto], run.h_stress[upto], color=INK, lw=1.4)
    c.plot([run.strain[k]], [run.stress[k]], marker="s", ms=6, color="#c9302c")
    c.set_xlim(0, run.h_strain.max() * 1.02); c.set_ylim(0, run.h_stress.max() * 1.1)
    c.set_xlabel(r"macro strain $E_{xx}$ (%)", fontsize=9); c.set_ylabel(r"macro stress $\Sigma_{xx}$ (MPa)", fontsize=9)
    for s in ("top", "right"):
        c.spines[s].set_visible(False)
    cb = fig.add_axes([0.63, 0.20, 0.34, 0.025])
    cb.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=DMG, extent=(0, 1, 0, 1)); cb.set_yticks([]); cb.set_xticks([0, 0.5, 1]); cb.tick_params(labelsize=8)
    cb.set_xlabel("damage d", fontsize=8.5, labelpad=2)
    dmax = float(run.element_damage(k)[run.phase != PHASE_PORE].max())
    fig.text(0.63, 0.09, f"increment {run.steps[k]}   E_xx = {run.strain[k]:.3f} %   Σ_xx = {run.stress[k]:.0f} MPa   max d = {dmax:.2f}", fontsize=8.6, ha="left", va="center", color=INK)
    return fig


def figure_final(run: Run, spec, out: Path):
    fig = frame_figure(run, spec, len(run.steps) - 1, size=(12.0, 6.0), dpi=130)
    _fit_png(fig, out / "damage_final.png")
    from PIL import Image
    im = Image.open(out / "damage_final.png"); s = 800 / im.width
    im.resize((800, int(im.height * s)), Image.LANCZOS).save(out / "thumbnail.png", optimize=True)


def animation(run: Run, spec, out: Path, fps: int, video: bool):
    from PIL import Image
    frames = []
    for k in range(len(run.steps)):
        fig = frame_figure(run, spec, k)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)
        print(f"  frame {k + 1}/{len(run.steps)}", flush=True)
    hold = [frames[0]] * 2 + frames + [frames[-1]] * 6                    # a short pause at both ends
    pil = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=192) for f in hold]
    pil[0].save(out / "damage_evolution.gif", save_all=True, append_images=pil[1:], duration=int(1000 / fps), loop=0, optimize=True, disposal=2)
    if video:
        h, w = hold[0].shape[:2]
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", str(out / "damage_evolution.mp4")]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for f in hold:
            proc.stdin.write(np.ascontiguousarray(f).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed; rerun with --no-video or install ffmpeg")
    return len(hold)


def _fit_png(fig, path: Path, limit=1990):
    fig.savefig(path, dpi=fig.dpi)
    plt.close(fig)
    from PIL import Image
    im = Image.open(path)
    if max(im.size) > limit:
        s = limit / max(im.size); im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS).save(path, optimize=True)


# ------------------------------------------------------------------ main
def write_history(run: Run, out: Path):
    cols = ("step", "macro_strain_xx", "macro_stress_xx", "balance_residual_rel")
    with (out / "history.csv").open("w", newline="") as stream:
        w = csv.writer(stream); w.writerow(cols)
        for r in run.history:
            w.writerow([r.get(c, "") for c in cols])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", choices=sorted(CASES)); ap.add_argument("--run-dir", required=True, type=Path); ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fps", type=int, default=8); ap.add_argument("--no-video", action="store_true"); ap.add_argument("--only", choices=("figures", "animation"), default=None)
    a = ap.parse_args(argv)
    spec = CASES[a.case]; run = Run(a.run_dir); a.out.mkdir(parents=True, exist_ok=True)
    made = []
    if a.only in (None, "figures"):
        figure_mesh_bc(run, spec, a.out, a.case); figure_curve(run, spec, a.out); figure_final(run, spec, a.out); write_history(run, a.out)
        made += ["initial_conditions.png", "stress_strain.png", "damage_final.png", "thumbnail.png", "history.csv"]
    n_frames = None
    if a.only in (None, "animation"):
        n_frames = animation(run, spec, a.out, a.fps, not a.no_video)
        made += ["damage_evolution.gif"] + ([] if a.no_video else ["damage_evolution.mp4"])
    from PIL import Image
    rows = []
    for n in made:
        f = a.out / n
        if n.endswith(".png") or n.endswith(".gif"):
            w, h = Image.open(f).size
            rows.append(dict(file=n, artifact_type="image", size_bytes=f.stat().st_size, visual_scope="three_dimensional", width_px=w, height_px=h,
                             review_dimension_passed=bool(max(w, h) <= 2000)))
        elif n.endswith(".mp4"):
            rows.append(dict(file=n, artifact_type="video", size_bytes=f.stat().st_size, visual_scope="three_dimensional"))
    (a.out / "visual_manifest.json").write_text(json.dumps(rows, indent=1) + "\n")
    manifest = dict(
        example=a.case, title=spec["title"], status="beta_periodic_voxel", run_script="run.py",
        source_result=a.run_dir.name, source_file=run.file.name, source_sha256=sha256(run.file),
        grid=dict(nx=run.nx, ny=run.ny, nz=run.nz, voxel_um=run.h_um, elements=run.nx * run.ny * run.nz),
        stored_states=len(run.steps), stored_steps=run.steps, fps=a.fps, animation_frames=n_frames,
        block=dict(cut_away="x > L/2, y < L/2, z > L/2", repeat_along_z=spec["reps"], drawn_thickness_um=run.nz * spec["reps"] * run.hz_um,
                   stored_thickness_um=run.nz * run.hz_um),
        colour="element damage = mean of the eight nodal values; no smoothing", font="STIX serif",
        private_exclusions=["field archives (accepted_fields.h5, result.vtkhdf)", "checkpoints", "raw HPC folders"],
        files=made + ["visual_manifest.json", "run_manifest.json"],
        outputs_sha256={n: sha256(a.out / n) for n in made if (a.out / n).exists()},
    )
    (a.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote", ", ".join(made))


if __name__ == "__main__":
    main()
