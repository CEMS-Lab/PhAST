"""The standard figure sheets: setup, fields, components, damage across steps, evolution.

Each function draws one PNG from a :class:`RunData` and returns its manifest
row (file name, kind, steps, fields). Titles are built from what the result
reports, never from paths or run names.
"""
from __future__ import annotations

import math
from pathlib import Path
import re
import textwrap
from typing import Any

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogFormatterSciNotation, LogLocator, MaxNLocator

from .data import FieldInfo, RunData, Table, unit_suffix
from .maps import (auto_deform_scale, draw_interfaces_2d, draw_interfaces_iso, draw_mesh_cells, draw_mesh_values,
                   draw_points_iso, draw_section, draw_surface_colours, draw_surface_iso, iso, iso_axes_labels,
                   iso_draw, iso_geometry, limits, section_planes, voxel_rgba)
from .style import LINE_COLOURS, SET_COLOURS, clean, content_bottom, figure, phase_colour, save

SCALAR_ROLES = {"damage", "scalar", "vector", "perpoint"}


# -- shared pieces ------------------------------------------------------------------
def describe(run: RunData) -> str:
    """One line describing the result from what it reports."""
    if run.kind == "grid":
        nx, ny, nz = run.shape
        h = run.spacing * run.length_scale
        unit = f" {run.length_label}" if run.length_label else ""
        text = f"Voxel grid {nx} × {ny} × {nz}, voxel {h[0]:.4g}{unit}"
    elif run.kind == "mesh2d":
        types = ", ".join(run.cell_types) if getattr(run, "cell_types", None) else "cells"
        text = f"2D mesh, {run.n_points:,} nodes, {run.n_cells:,} cells ({types})"
        analysis = run.metadata.get("analysis") if isinstance(run.metadata, dict) else None
        if analysis in ("plane_strain", "plane_stress"):
            text += ", " + analysis.replace("_", " ")
    elif run.kind == "mesh3d":
        text = f"3D mesh, {run.n_points:,} nodes, {run.n_cells:,} cells"
    else:
        text = "Result without stored geometry"
    text += f"; {run.n_steps} stored step{'s' if run.n_steps != 1 else ''} ({run.layout_label})"
    status = run.metadata.get("status") if isinstance(run.metadata, dict) else None
    if status and status != "complete":
        text += f"; file status: {status}"
    return text


def footnote(fig, notes: list[str], width_in: float) -> None:
    notes = [n for n in (clean(x) for x in notes) if n]
    if not notes:
        return
    chars = max(60, int(width_in * 15))
    lines = []
    for n in notes:
        lines += textwrap.wrap(n, chars, initial_indent="• ", subsequent_indent="  ")
    y = content_bottom(fig) - 0.08 / fig.get_figheight()
    fig.text(0.0, y, "\n".join(lines), ha="left", va="top", fontsize=7.5, color="#3A4046",
             transform=fig.transFigure)


def heading(fig, text: str, width_in: float | None = None) -> None:
    """Sheet heading, placed above everything drawn (after layout), so it never overlaps a panel title."""
    chars = max(70, int((width_in or fig.get_figwidth()) * 12.5))
    fig.draw_without_rendering()
    top = fig.get_tightbbox(fig.canvas.get_renderer()).y1 / fig.get_figheight()
    fig.text(0.0, top + 0.06 / fig.get_figheight(), "\n".join(textwrap.wrap(text, chars)), ha="left", va="bottom",
             fontsize=10.5, transform=fig.transFigure)


def colorbar(fig, art, ax, label: str, location: str = "bottom"):
    cb = fig.colorbar(art, ax=ax, location=location, fraction=0.05, pad=0.08 if location == "bottom" else 0.03,
                      aspect=22)
    cb.ax.tick_params(labelsize=7.5)
    axis = cb.ax.xaxis if location in ("bottom", "top") else cb.ax.yaxis
    axis.set_major_locator(MaxNLocator(4))
    norm = getattr(art, "norm", None)
    top = max(abs(norm.vmin), abs(norm.vmax)) if norm is not None and norm.vmin is not None else 1.0
    exp = 0 if top == 0 or 1e-2 <= top < 1e4 else int(np.floor(np.log10(top)))
    if exp:
        axis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v / 10 ** exp:.3g}"))
        label += rf"  [$\times10^{{{exp}}}$]"
    else:
        axis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.4g}"))
    cb.set_label(label, fontsize=8.5)
    return cb


def _panel_size(run: RunData, width: float = 3.6) -> tuple[float, float]:
    L = run.domain_lengths()
    aspect = 1.0 if L is None or L[0] <= 0 else float(np.clip(L[1] / L[0], 0.25, 1.4))
    return width, width * aspect + 1.1


def _field_label(info: FieldInfo, comp: str | None = None) -> str:
    """Colour-bar label: symbol and units (the panel title carries the words)."""
    if comp is not None:
        return info.component_symbol(comp) + unit_suffix(info.units)
    if info.role == "perpoint":
        return f"cell mean of {info.short}" + unit_suffix(info.units)
    return info.short + unit_suffix(info.units)


def _field_notes(run: RunData, infos: list[FieldInfo]) -> list[str]:
    notes = []
    vectors = [i.short for i in infos if i.role == "vector"]
    if vectors:
        notes.append(f"{', '.join(vectors)} computed here from the stored components.")
    for info in infos:
        if info.derived_method:
            notes.append(f"{info.words}: stored cell values of a derived quantity ({info.derived_method.replace('_', ' ')}).")
    for info in infos:
        if info.shear == "engineering":
            notes.append(f"{info.words}: shear components are engineering shear, $\\gamma=2\\varepsilon$.")
    if any(i.role == "perpoint" for i in infos):
        notes.append("Values kept at the quadrature points are shown as the unweighted mean over the points of each cell.")
    return notes


# -- setup sheet ----------------------------------------------------------------------
def _phase_table(run: RunData):
    """Sorted phase ids, a colour per id, and the legend label with the share of the domain."""
    ids = sorted(run.phases)
    colour = {pid: phase_colour(i) for i, pid in enumerate(ids)}
    labels = {}
    for pid in ids:
        name = clean(run.phases[pid].get("name", f"phase {pid}")) or f"phase {pid}"
        label = f"{name.replace('_', ' ')} (id {pid})"
        if run.phase_cells is not None:
            mask = run.phase_cells == pid
            if run.kind == "mesh2d":
                share = run.mesh2d.areas[mask].sum() / max(run.mesh2d.areas.sum(), 1e-300)
                label += f", {100 * share:.1f} % of the area"
            else:
                label += f", {100 * mask.mean():.1f} % of the {'volume' if run.kind == 'grid' else 'cells'}"
        labels[pid] = label
    return ids, colour, labels


def _fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


MAX_ATTRIBUTE_TEXT = 40


def _short_attribute(key: str, value: Any, units: dict) -> str | None:
    """``key = value unit`` for numbers and short strings; None for long text, lists and arrays."""
    if isinstance(value, bool) or isinstance(value, (list, tuple, dict, np.ndarray)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        unit = units.get(key)
        return f"{key} = {_fmt_value(float(value) if isinstance(value, float) else value)}" + (
            f" {unit}" if unit and unit != "1" else "")
    text = str(value)
    if len(text) > MAX_ATTRIBUTE_TEXT or "\n" in text or text[:1] in "[{(":
        return None
    return f"{key} = {text}"


def _material_lines(run: RunData) -> tuple[list[str], list[str]]:
    """Material text (numeric constants with units, short strings) and footnotes for what is left out."""
    lines, notes = [], []
    for pid in sorted(run.phases):
        attrs = run.phases[pid]
        units = attrs.get("units") if isinstance(attrs.get("units"), dict) else {}
        parts, left_out = [], 0
        for key, value in attrs.items():
            if key in ("name", "units"):
                continue
            item = _short_attribute(key, value, units)
            if item is None:
                left_out += 1
            else:
                parts.append(item)
        name = clean(str(attrs.get("name", f"phase {pid}"))) or f"phase {pid}"
        lines.append(f"{name} (id {pid})" + (": " + ", ".join(parts) if parts else ""))
        if left_out:
            where = f"materials/phases/{pid}"
            notes.append(f"{name} (id {pid}): {left_out} further attribute{'s' if left_out != 1 else ''} "
                         f"(long text or arrays) in {where}, not shown.")
    if run.constants:
        items = [_short_attribute(k, v, {}) for k, v in run.constants.items()]
        shown = [i for i in items if i]
        if shown:
            lines.append("Constants as stored: " + ", ".join(shown))
        if len(shown) < len(items):
            notes.append(f"{len(items) - len(shown)} further material attributes (long text or arrays) not shown.")
    return lines or ["Not stored in this layout."], notes


def _bc_parts(attrs: dict[str, Any]) -> tuple[str, list[str], Any]:
    kind = str(attrs.get("kind", ""))
    comps = [c.strip() for c in str(attrs.get("components", "")).split(",") if c.strip()]
    return kind, comps, attrs.get("value")


def _bc_text(name: str, attrs: dict[str, Any]) -> str:
    kind, comps, value = _bc_parts(attrs)
    symbol = {"displacement": "u", "traction": "t"}.get(kind)
    target = clean(str(attrs.get("set", name))) or name
    if symbol and comps:
        lhs = " = ".join(f"${symbol}_{{{c}}}$" for c in comps)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            rhs = _fmt_value(float(value))
        else:
            rhs = f"history column {clean(str(value)) or ''}".strip()
        return f"{target}: {kind}, {lhs} = {rhs}"
    text = f"{target}: {kind.replace('_', ' ') or 'condition'}" + (f", components {', '.join(comps)}" if comps else "")
    if isinstance(value, str) and clean(value):
        text += f", following history column {value}"
    return text


def _loading_lines(run: RunData) -> list[str]:
    lines = [_bc_text(k, a) for k, a in run.bcs.items()]
    if run.periodic:
        p = run.periodic
        text = f"periodic along {str(p.get('axes', '')).replace(',', ', ')}"
        for key, word in (("controlled", "controlled"), ("free", "free"), ("fixed_zero", "held at zero")):
            if p.get(key):
                text += f"; {word}: {str(p[key]).replace(',', ', ')}"
        lines.append(text)
    if not lines:
        lines.append("Not stored in this layout." if not run.bcs_stored else "No boundary conditions recorded.")
    return lines


def _text_panel(ax, blocks: list[tuple[str, list[str]]], width_chars: int) -> None:
    """Headed blocks of text, spaced in points so that they never overlap."""
    ax.axis("off")
    height_in = ax.get_position().height * ax.figure.get_figheight()
    line = 8.2 * 1.38 / 72.0 / max(height_in, 0.5)       # one text line in axes fraction
    y = 1.0
    for title, lines in blocks:
        ax.text(0.0, y, title, fontsize=9.5, fontweight="bold", va="top", transform=ax.transAxes)
        y -= 1.5 * line
        for item in lines:
            wrapped = textwrap.wrap(item, width_chars, subsequent_indent="   ") or [""]
            ax.text(0.0, y, "\n".join(wrapped), fontsize=8.2, va="top", transform=ax.transAxes, linespacing=1.25)
            y -= line * (len(wrapped) + 0.3)
        y -= 0.8 * line


def _facts(run: RunData) -> list[str]:
    lines = []
    L = run.domain_lengths()
    if L is not None:
        unit = f" {run.length_label}" if run.length_label else ""
        lines.append("Extent " + " × ".join(f"{v * run.length_scale:.4g}" for v in L) + unit)
    lines.append(describe(run).split(";")[0])
    md = run.metadata if isinstance(run.metadata, dict) else {}
    for key, word in (("history_coupling", "history coupling"), ("history_update", "history update"),
                      ("kinematics", "kinematics")):
        if md.get(key):
            lines.append(f"{word}: {str(md[key]).replace('_', ' ')}")
    if run.n_steps:
        text = f"{run.n_steps} stored steps, labels {run.steps[0]} to {run.steps[-1]}"
        if run.times is not None and np.isfinite(run.times).any():
            text += f", t = {np.nanmin(run.times):.4g} to {np.nanmax(run.times):.4g}"
        lines.append(text)
    else:
        lines.append("No stored step (setup file).")
    return lines


def setup_sheet(run: RunData, path: Path, dpi: int) -> dict[str, Any]:
    ids, colour, labels = _phase_table(run)
    notes = run.notes_for("geometry", "materials", "loading", "interfaces")
    material_text, material_notes = _material_lines(run)
    notes += material_notes
    blocks = [("Geometry and mesh", _facts(run)), ("Materials", material_text),
              ("Loading and boundary conditions", _loading_lines(run))]
    if run.node_sets:
        blocks.append(("Node sets", [f"{clean(n) or 'set'}: {len(v):,} nodes" for n, v in run.node_sets.items()]))
    if run.kind == "grid":
        width, height = 14.0, 7.0
    elif run.kind == "mesh2d":
        pw, ph = _panel_size(run, 7.6)
        width, height = 13.0, max(5.6, ph)
    else:
        width, height = (15.0, 5.6) if run.groups3d else (12.0, 6.0)
    with figure(width, height, layout="constrained" if run.kind == "grid" else None) as fig:
        if run.kind == "grid":
            planes = section_planes(run)
            second = "xz" if "xz" in planes else None
            gs = fig.add_gridspec(2, 3, width_ratios=[1.45, 1.0, 1.0], height_ratios=[1.0, 0.75])
            ax = fig.add_subplot(gs[0, 0])
            _setup_iso(run, ax, ids, colour, notes)
            ax.set_title("(a) Materials, isometric view", loc="left")
            _phase_section(run, fig.add_subplot(gs[0, 1] if second else gs[:, 1]), "xy", ids, colour)
            if second:
                _phase_section(run, fig.add_subplot(gs[1, 1]), second, ids, colour)
            else:
                notes.append(f"Section through the thickness not drawn: only {run.shape[2]} voxels along z.")
            handles = [Patch(facecolor=colour[p], edgecolor="#16191D", lw=0.5, label=labels[p]) for p in ids]
            axl = fig.add_subplot(gs[1, 0])
            axl.axis("off")
            if handles:
                axl.legend(handles=handles, loc="upper left", ncol=2 if len(handles) > 3 else 1, frameon=False,
                           fontsize=8.5, title="Material, share of the volume", title_fontsize=9,
                           alignment="left")
            fig.draw_without_rendering()             # final panel positions before the text is placed
            _text_panel(fig.add_subplot(gs[:, 2]), blocks, 52)
        elif run.kind == "mesh2d":
            gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.12)
            ax = fig.add_subplot(gs[0, 0])
            handles = _setup_mesh2d(run, ax, ids, colour, labels, notes)
            ax.set_title("(a) Mesh, materials, node sets and boundary conditions", loc="left")
            ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False,
                      fontsize=7.8)
            _text_panel(fig.add_subplot(gs[0, 1]), blocks, 50)
        elif run.kind == "mesh3d":
            if run.groups3d:
                gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0])
                rgba = _cell_rgba(run, ids, colour)
                for j, (cut, label) in enumerate(((False, "(a) Materials, isometric surface"),
                                                  (True, "(b) Materials, cut-away"))):
                    ax = fig.add_subplot(gs[0, j])
                    draw_surface_colours(ax, run, rgba, cut)
                    ax.set_title(label, loc="left")
                if run.phase_cells is None:
                    notes.append("No phase map is stored; the body is drawn in one colour.")
                _draw_interfaces_setup(fig.axes[-1], run, notes)
                handles = ([Patch(facecolor=colour[p], edgecolor="#16191D", lw=0.5, label=labels[p]) for p in ids]
                           if run.phase_cells is not None else [])
                if handles:
                    fig.axes[0].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
                                       frameon=False, fontsize=7.8)
                _text_panel(fig.add_subplot(gs[0, 2]), blocks, 44)
            else:
                gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0])
                ax = fig.add_subplot(gs[0, 0])
                draw_points_iso(ax, run, np.zeros(run.n_points), "Greys", None)
                ax.set_title("(a) Nodes, isometric projection", loc="left")
                _text_panel(fig.add_subplot(gs[0, 1]), blocks, 50)
        else:
            _text_panel(fig.add_subplot(1, 1, 1), blocks, 90)
        heading(fig, "Setup. " + describe(run))
        footnote(fig, notes, width)
        w, h = save(fig, path, dpi)
    return dict(file=path.name, kind="setup", step=None, steps=[], fields=["phase_id"] if run.phase_cells is not None else [],
                width_px=w, height_px=h, notes=notes)


def _cell_rgba(run: RunData, ids, colour) -> np.ndarray:
    """One colour per cell: its phase, or a neutral grey when no phase map is stored."""
    if run.phase_cells is not None and ids:
        return _phase_rgba(run, ids, colour)[run.phase_cells]
    return np.tile(np.array([0.85, 0.87, 0.9, 1.0]), (run.n_cells, 1))


def _draw_interfaces_setup(ax, run: RunData, notes: list[str]) -> None:
    """Interface cells of a 3D mesh in one colour on the cut-away panel of the setup sheet."""
    for block in run.interfaces:
        pc, xy = draw_interfaces_iso(ax, run, block, None)
        notes.append(f"Cohesive interface {block.name}: {len(block.nodes):,} cells ({block.cell_type}) drawn in red.")


def _phase_rgba(run: RunData, ids, colour):
    from matplotlib.colors import to_rgba
    lut = np.zeros((max(ids) + 1 if ids else 1, 4))
    for pid in ids:
        lut[pid] = to_rgba(colour[pid])
    return lut


def _setup_iso(run: RunData, ax, ids, colour, notes):
    cut, zf = iso_geometry(run)
    if run.phase_cells is not None and ids:
        rgba = _phase_rgba(run, ids, colour)[run.phase_cells]
    else:
        rgba = np.tile(np.array([0.85, 0.87, 0.9, 1.0]), run.cell_shape + (1,))
    g = iso_draw(ax, run, rgba, cut, zf)
    iso_axes_labels(ax, g, run.length_label)
    notes += iso_note(run)
    controlled = str((run.periodic or {}).get("controlled", ""))
    if controlled in ("xx", "yy", "zz"):
        Gx, Gy, Gz = g["L"]
        c = "xyz".index(controlled[0])
        mid = [(Gx, Gy / 2, Gz / 2), (Gx / 2, Gy, Gz / 2), (Gx / 2, Gy / 2, Gz)][c]
        d = np.eye(3)[c]
        p0 = np.array(iso(*mid))
        n = np.array(iso(*(np.array(mid) + d))) - p0
        n = n / np.hypot(*n) * 0.25 * max(Gx, Gy)
        ax.annotate("", xy=p0 + n, xytext=p0, arrowprops=dict(arrowstyle="-|>", color="#A3223F", lw=2.0))
        ax.text(0.0, 1.0, f"arrow: controlled {controlled} on the {controlled[0]} faces", transform=ax.transAxes,
                color="#A3223F", fontsize=8, ha="left", va="top")


def _phase_section(run: RunData, ax, plane: str, ids, colour):
    from matplotlib.colors import ListedColormap, BoundaryNorm
    if run.phase_cells is None:
        values = np.zeros(run.cell_shape)
        cmap, norm = ListedColormap(["#D9DDE2"]), BoundaryNorm([-0.5, 0.5], 1)
    else:
        index = {pid: i for i, pid in enumerate(ids)}
        values = np.vectorize(index.get)(run.phase_cells) if len(ids) < 64 else run.phase_cells
        cmap = ListedColormap([colour[p] for p in ids])
        norm = BoundaryNorm(np.arange(len(ids) + 1) - 0.5, len(ids))
    _, where = draw_section(ax, run, values, plane, "cell", cmap, norm)
    letter = "b" if plane == "xy" else "c"
    ax.set_title(f"({letter}) Mid-plane section, {where}", loc="left")


def _setup_mesh2d(run: RunData, ax, ids, colour, labels, notes):
    handles = []
    if run.phase_cells is not None and ids:
        rgba = _phase_rgba(run, ids, colour)[run.phase_cells]
        handles += [Patch(facecolor=colour[p], edgecolor="#16191D", lw=0.5, label=labels[p]) for p in ids]
    else:
        rgba = np.tile(np.array([0.86, 0.88, 0.9, 1.0]), (run.n_cells, 1))
        handles.append(Patch(facecolor="#DCE0E5", edgecolor="#16191D", lw=0.5, label="single material (no phase map stored)"))
    many = run.n_cells > 30000
    draw_mesh_cells(ax, run, facecolours=rgba, edge="none" if many else "#3A4046", lw=0.0 if many else 0.2)
    for block in run.interfaces:
        draw_interfaces_2d(ax, run, block, None, colour="#A3223F")
        handles.append(Line2D([], [], color="#A3223F", lw=2.6,
                              label=f"cohesive interface {clean(block.name) or ''}: {len(block.nodes):,} cells ({block.cell_type})"))
    if many:
        notes.append(f"Cell edges are not drawn ({run.n_cells:,} cells).")
    p = run.mesh2d.points * run.length_scale
    size = float(np.max(run.domain_lengths())) * run.length_scale
    bc_by_set: dict[str, list[tuple[str, dict]]] = {}
    for name, attrs in run.bcs.items():
        bc_by_set.setdefault(str(attrs.get("set", name)), []).append((name, attrs))
    shown = 0
    for set_name, idx in sorted(run.node_sets.items(), key=lambda kv: -len(kv[1])):
        idx = idx[(idx >= 0) & (idx < len(p))]
        if idx.size == 0:
            continue
        if shown >= len(SET_COLOURS):
            notes.append(f"Only the first {len(SET_COLOURS)} node sets are drawn.")
            break
        col = SET_COLOURS[shown]
        shown += 1
        bcs = bc_by_set.get(set_name, [])
        label = clean(set_name) or "node set"
        if bcs:
            label = "; ".join(_bc_text(n, a) for n, a in bcs)
        else:
            label = f"node set {label}"
        ax.scatter(p[idx, 0], p[idx, 1], s=9 if idx.size < 400 else 3, marker="^" if bcs else "o", color=col,
                   zorder=4, linewidths=0)
        handles.append(Line2D([], [], marker="^" if bcs else "o", color=col, lw=0, ms=5, label=label))
        centre = p[idx].mean(axis=0)
        for _, attrs in bcs:
            kind, comps, value = _bc_parts(attrs)
            prescribed = not (isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) == 0.0)
            if kind not in ("displacement", "traction") or not prescribed:
                continue
            for c in comps:
                if c not in "xy":
                    continue
                d = np.array([1.0, 0.0]) if c == "x" else np.array([0.0, 1.0])
                inward = d if float(np.dot(p.mean(axis=0) - centre, d)) >= 0 else -d   # keep the arrow inside
                ax.annotate("", xy=centre + 0.16 * size * inward, xytext=centre + 0.02 * size * inward,
                            arrowprops=dict(arrowstyle="<|-|>", color=col, lw=1.6), zorder=5,
                            annotation_clip=False)
    if any(_bc_parts(a)[0] in ("displacement", "traction") for a in run.bcs.values()):
        notes.append("Double arrows mark prescribed non-zero components; the sign follows the stored history column.")
    missing = [s for s in bc_by_set if s not in run.node_sets]
    if missing:
        notes.append("Boundary conditions on sets that are not stored as node sets are listed but not drawn: "
                     + ", ".join(clean(m) or "set" for m in missing) + ".")
    return handles


# -- field sheets ----------------------------------------------------------------------
def iso_note(run: RunData) -> list[str]:
    if run.kind != "grid":
        return []
    cut, zf = iso_geometry(run)
    if cut:
        return ["Isometric views: the near corner (one eighth of the cell) is cut away to show the inside."]
    if zf > 1:
        return [f"Isometric views: the thickness is drawn {zf:.1f} times larger than it is so that it can be seen."]
    return []


def compute_norms(run: RunData, infos: list[FieldInfo], steps: list[int]) -> dict:
    """Colour map and limits per (field, component), shared by every stored step drawn."""
    norms = {}
    for info in infos:
        comps = list(info.components) if info.role == "components" else [None]
        for comp in comps:
            arrays = []
            for s in steps:
                try:
                    arrays.append(run.scalar(info, s, comp))
                except Exception:   # a field absent at one step is skipped there
                    continue
            role = info.role if comp is None else "components"
            norms[(info.name, comp)] = limits(arrays, role)
    return norms


def _deform(run: RunData, step: int, deform_scale) -> tuple[np.ndarray | None, float]:
    if deform_scale in (None, 0, "0", "none") or run.kind != "mesh2d":
        return None, 0.0
    info = run.fields.get("displacement")
    if info is None or info.role != "vector":
        return None, 0.0
    u = run.values(info, step)
    scale = auto_deform_scale(run, u) if deform_scale == "auto" else float(deform_scale)
    return (u * scale if scale > 0 else None), scale


def _draw_scalar(fig, ax, run: RunData, info: FieldInfo, step: int, norms, comp=None, deform=None, plane=None,
                 iso_view=False):
    cmap, norm = norms[(info.name, comp)]
    values = run.scalar(info, step, comp)
    where = ""
    if run.kind == "mesh2d":
        art = draw_mesh_values(ax, run, values, info.location, cmap, norm, deform)
    elif run.kind == "grid" and iso_view:
        cut, zf = iso_geometry(run)
        g = iso_draw(ax, run, voxel_rgba(values, info.location, cmap, norm), cut, zf,
                     "#9AA3AD" if info.role == "damage" else "#16191D")
        iso_axes_labels(ax, g, run.length_label)
        art = ScalarMappable(norm=norm, cmap=cmap)
        where = "isometric view" + (", nodal values averaged to voxels" if info.location == "node" else "")
    elif run.kind == "grid":
        art, where = draw_section(ax, run, values, plane or "xy", info.location, cmap, norm)
    elif run.kind == "mesh3d" and run.groups3d:
        cut = plane in ("cut", "xy")
        art = draw_surface_iso(ax, run, values, info.location, cmap, norm, cut)
        where = "isometric cut-away" if cut else "isometric surface"
    else:
        if info.location != "node":
            ax.axis("off")
            ax.text(0.5, 0.5, "cell values on this unstructured 3D mesh are not drawn", ha="center", fontsize=8)
            return None, ""
        art = draw_points_iso(ax, run, values, cmap, norm)
    return art, where


STATE_NAMES = {"displacement", "damage", "history_field", "history_field_nodal", "H_damage_input", "velocity",
               "acceleration"}
PANELS_PER_ROW = 4          # about 1900 px across at the default resolution
MAX_SHEET_HEIGHT_IN = 12.5


def is_state(info: FieldInfo) -> bool:
    """State and history fields (the solution); everything else is derived or a proposal."""
    if info.kind is not None:
        return info.kind in ("state", "history")
    return info.name in STATE_NAMES


def _block_layout(run: RunData) -> tuple[list[str], float, float]:
    """Panels of one field (sections and isometric view for voxel grids, one map otherwise) and panel size."""
    if run.kind == "grid":
        return section_planes(run) + ["iso"], 3.2, 3.5
    if run.kind == "mesh3d" and run.groups3d:
        return ["surface", "cut"], 3.4, 4.0
    pw, ph = _panel_size(run, 3.6)
    return [None], pw, ph


def field_pages(run: RunData, infos: list[FieldInfo]) -> list[tuple[str, list[FieldInfo]]]:
    """Fields of the scalar sheets of one step: ``[(group, fields), ...]``.

    One sheet when everything fits in two rows of panels; otherwise state
    fields and derived fields on separate sheets, each split further when it
    would be taller than about 12.5 inches.
    """
    infos = [i for i in infos if i.role in SCALAR_ROLES]
    if not infos:
        return []
    panels, _, ph = _block_layout(run)
    per_row = max(1, PANELS_PER_ROW // len(panels))
    max_rows = max(1, int(MAX_SHEET_HEIGHT_IN // ph))
    state = [i for i in infos if is_state(i)]
    derived = [i for i in infos if not is_state(i)]
    if math.ceil(len(infos) / per_row) <= 2 or not state or not derived:
        groups = [("fields", infos)]
    else:
        groups = [("state", state), ("derived", derived)]
    pages = []
    size = per_row * max_rows
    for name, members in groups:
        for start in range(0, len(members), size):
            pages.append((name, members[start:start + size]))
    return pages


def field_sheet(run: RunData, step: int, reason: str, infos: list[FieldInfo], norms, path: Path, dpi: int,
                deform_scale=None, group: str = "fields") -> dict[str, Any] | None:
    infos = [i for i in infos if i.role in SCALAR_ROLES]
    if not infos or run.kind == "none":
        return None
    notes = run.notes_for("geometry", "fields", "steps") + _field_notes(run, infos) + iso_note(run)
    deform, scale = _deform(run, step, deform_scale)
    if deform is not None:
        notes.append(f"Maps drawn on the deformed shape; displacements multiplied by {scale:.3g}.")
    available = []
    for info in infos:
        try:
            run.values(info, step)
            available.append(True)
        except Exception:
            available.append(False)
    infos = [i for i, ok in zip(infos, available) if ok] + [i for i, ok in zip(infos, available) if not ok]
    panels, pw, ph = _block_layout(run)
    if run.kind == "grid" and len(panels) < 4:
        notes.append(f"Sections with fewer than four voxels across are not drawn (grid {run.shape[0]} × "
                     f"{run.shape[1]} × {run.shape[2]}).")
    per_row = max(1, PANELS_PER_ROW // len(panels))
    blocks_row = min(per_row, len(infos))
    nrows = math.ceil(len(infos) / blocks_row)
    ncols = blocks_row * len(panels)
    width, height = pw * ncols, ph * nrows + 0.8
    drawn = []
    with figure(width, height, layout="constrained") as fig:
        axes = fig.subplots(nrows, ncols, squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for k, info in enumerate(infos):
            row, first = divmod(k, blocks_row)
            block = [axes[row, first * len(panels) + j] for j in range(len(panels))]
            try:
                run.values(info, step)
            except Exception:   # stored on some steps only (for example /derived proposals)
                notes.append(f"{info.words} is not stored at step {step}.")
                block[0].text(0.0, 0.5, f"{info.words}: not stored at this step", transform=block[0].transAxes,
                              fontsize=8.5, color="#56606B")
                continue
            for j, (ax, plane) in enumerate(zip(block, panels)):
                ax.axis("on")
                art, where = _draw_scalar(fig, ax, run, info, step, norms, deform=deform,
                                          plane=None if plane == "iso" else plane, iso_view=plane == "iso")
                if run.kind == "grid" or len(panels) > 1:
                    ax.set_title(f"{info.title()}\n{where}" if j == 0 else f"\n{where}", fontsize=8.5, loc="left")
                else:
                    ax.set_title(info.title(), fontsize=9, loc="left")
                if art is not None:
                    colorbar(fig, art, ax, _field_label(info))
            drawn.append(info.name)
        what = {"fields": "Fields", "state": "State fields", "derived": "Derived fields"}[group]
        heading(fig, f"{what} at {run.step_caption(step)} ({reason}). {describe(run)}", width)
        footnote(fig, notes, width)
        w, h = save(fig, path, dpi)
    return dict(file=path.name, kind="fields", group=group, step=step, steps=[step], fields=drawn, width_px=w,
                height_px=h, notes=notes)


def component_sheet(run: RunData, step: int, reason: str, infos: list[FieldInfo], norms, path: Path, dpi: int,
                    deform_scale=None) -> dict[str, Any] | None:
    infos = [i for i in infos if i.role == "components"]
    if not infos or run.kind == "none":
        return None
    notes = run.notes_for("geometry", "fields", "steps") + _field_notes(run, infos)
    deform, scale = _deform(run, step, deform_scale)
    if deform is not None:
        notes.append(f"Maps drawn on the deformed shape; displacements multiplied by {scale:.3g}.")
    ncols = min(6, max(len(i.components) for i in infos))
    rows = []
    for info in infos:
        comps = list(info.components)
        for start in range(0, len(comps), ncols):
            rows.append((info, comps[start:start + ncols]))
    if run.kind == "grid":
        pw, ph = 3.0, 3.0 * float(np.clip(run.shape[1] / run.shape[0], 0.3, 1.4)) + 1.0
    else:
        pw, ph = _panel_size(run, 3.0)
    width, height = pw * ncols, ph * len(rows) + 0.8
    drawn = []
    where = ""
    with figure(width, height, layout="constrained") as fig:
        axes = fig.subplots(len(rows), ncols, squeeze=False)
        for r, (info, comps) in enumerate(rows):
            try:
                run.values(info, step)
            except Exception:
                notes.append(f"{info.words} is not stored at step {step}.")
                for ax in axes[r]:
                    ax.axis("off")
                continue
            for j in range(ncols):
                ax = axes[r, j]
                if j >= len(comps):
                    ax.axis("off")
                    continue
                art, where = _draw_scalar(fig, ax, run, info, step, norms, comp=comps[j], deform=deform, plane="xy")
                ax.set_title(info.component_symbol(comps[j]), fontsize=9.5, loc="left")
                if art is not None:
                    colorbar(fig, art, ax, _field_label(info, comps[j]))
            drawn.append(info.name)
        plane_text = f", mid-plane section {where}" if run.kind == "grid" else ""
        heading(fig, f"Components at {run.step_caption(step)} ({reason}){plane_text}. {describe(run)}", width)
        footnote(fig, notes, width)
        w, h = save(fig, path, dpi)
    return dict(file=path.name, kind="components", step=step, steps=[step], fields=sorted(set(drawn)),
                width_px=w, height_px=h, notes=notes)


def damage_steps_sheet(run: RunData, chosen: list[tuple[int, str]], norms, path: Path, dpi: int,
                       deform_scale=None) -> dict[str, Any] | None:
    info = run.fields.get("damage")
    if info is None or info.role != "damage" or not chosen or run.kind == "none":
        return None
    notes = run.notes_for("geometry", "steps") + iso_note(run)
    rows = ["xy", "iso"] if run.kind == "grid" else (["surface", "cut"] if run.kind == "mesh3d" and run.groups3d else [None])
    if run.kind == "grid" or len(rows) > 1:
        pw, ph = 3.4, 3.4
    else:
        pw, ph = _panel_size(run, 3.7)
    width, height = pw * len(chosen), ph * len(rows) + 0.9
    used = []
    with figure(width, height, layout="constrained") as fig:
        axes = fig.subplots(len(rows), len(chosen), squeeze=False)
        art = None
        for j, (step, reason) in enumerate(chosen):
            try:
                run.values(info, step)
            except Exception:
                for ax in axes[:, j]:
                    ax.axis("off")
                continue
            deform, scale = _deform(run, step, deform_scale)
            for i, plane in enumerate(rows):
                ax = axes[i, j]
                a, where = _draw_scalar(fig, ax, run, info, step, norms, deform=deform,
                                        plane=None if plane == "iso" else plane, iso_view=plane == "iso")
                art = a if a is not None else art
                caption = run.step_caption(step)
                if i == 0:
                    chars = max(30, int(pw * 13))
                    title = "\n".join(textwrap.wrap(reason, chars) + textwrap.wrap(caption, chars)
                                      + ([(f"section {where}" if run.kind == "grid" else where)] if where else []))
                else:
                    title = where
                ax.set_title(title, fontsize=8.3, loc="left")
            used.append(step)
        if art is not None:
            colorbar(fig, art, list(axes.flat), "damage $d$", location="right")
        heading(fig, f"Damage at {len(used)} stored steps. {describe(run)}")
        footnote(fig, notes, width)
        w, h = save(fig, path, dpi)
    return dict(file=path.name, kind="damage_steps", step=None, steps=used, fields=["damage"], width_px=w,
                height_px=h, notes=notes)


# -- cohesive interfaces ------------------------------------------------------------------------
def interface_field_names(run: RunData) -> list[str]:
    return sorted({f for b in run.interfaces for f in b.fields})


def compute_interface_norms(run: RunData, steps: list[int]) -> dict:
    """Colour map and limits per interface field, shared by every stored step drawn (damage-like fields use [0, 1])."""
    norms = {}
    for name in interface_field_names(run):
        arrays = []
        for s in steps:
            for b in run.interfaces:
                if name in b.fields:
                    try:
                        arrays.append(run.interface_values(b, name, s))
                    except Exception:
                        continue
        norms[name] = limits(arrays, "damage" if "damage" in name else "scalar")
    return norms


def interface_sheet(run: RunData, step: int, reason: str, norms: dict, path: Path, dpi: int) -> dict[str, Any] | None:
    """Interface-point arrays of the cohesive interfaces at one stored step, one panel per array, drawn on the
    interface cells over a light outline of the mesh."""
    names = interface_field_names(run)
    if not names or not run.interfaces:
        return None
    notes = run.notes_for("geometry", "interfaces", "steps")
    if run.kind == "mesh3d":
        notes.append("The mesh is drawn as the cut-away of the near octant; interface cells are drawn in full.")
    per_row = 3
    ncols, nrows = min(per_row, len(names)), math.ceil(len(names) / per_row)
    pw, ph = _panel_size(run, 3.8) if run.kind == "mesh2d" else (3.8, 3.8 + 1.1)
    width, height = pw * ncols, ph * nrows + 0.8
    drawn = []
    with figure(width, height, layout="constrained") as fig:
        axes = fig.subplots(nrows, ncols, squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for k, name in enumerate(names):
            ax = axes.flat[k]
            ax.axis("on")
            cmap, norm = norms[name]
            if run.kind == "mesh2d":
                draw_mesh_cells(ax, run, facecolours=None, edge="#B8BEC6", lw=0.2)
            else:
                draw_surface_colours(ax, run, np.tile(np.array([0.9, 0.91, 0.93, 1.0]), (run.n_cells, 1)), cut=True)
            art, units, words, n_qp = None, None, name.replace("_", " "), 0
            pts = []
            for b in run.interfaces:
                if name not in b.fields:
                    continue
                values = run.interface_values(b, name, step)
                units = units or b.fields[name]["units"]
                n_qp = max(n_qp, b.fields[name]["n_points"])
                if run.kind == "mesh2d":
                    art = draw_interfaces_2d(ax, run, b, values, cmap, norm)
                else:
                    art, xy = draw_interfaces_iso(ax, run, b, values, cmap, norm)
                    pts.append(xy)
            if run.kind == "mesh3d" and pts:
                from .maps import _frame_iso
                _frame_iso(ax, np.concatenate(pts + [np.array(ax.dataLim.get_points())]))
            ax.set_title(f"{words}, mean of {n_qp} interface point{'s' if n_qp != 1 else ''}", fontsize=8.8, loc="left")
            if art is not None:
                colorbar(fig, art, ax, f"{words}{unit_suffix(units)}")
            drawn.append(name)
        heading(fig, f"Cohesive interfaces at {run.step_caption(step)} ({reason}). {describe(run)}", width)
        footnote(fig, notes, width)
        w, h = save(fig, path, dpi)
    return dict(file=path.name, kind="interfaces", step=step, steps=[step], fields=drawn, width_px=w, height_px=h,
                notes=notes)


# -- evolution sheet ---------------------------------------------------------------------
def _col_label(table: Table, name: str | None) -> str:
    """History columns are named as stored, with their units when stored."""
    if name is None:
        return "history row"
    col = table.columns[name]
    return name + unit_suffix(col.units)


def _integer_axis(ax, name: str | None) -> None:
    if name in (None, "increment", "step"):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _legend_below(ax, ncol: int = 2) -> None:
    """Legend under the axes, clear of the data."""
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=ncol)


def _x_values(table: Table, name: str | None) -> np.ndarray:
    return np.arange(table.n, dtype=float) if name is None else table.numeric(name)


def _response_panel(run: RunData, ax, chosen) -> list[str]:
    notes = []
    resp = run.response()
    if resp is None:
        return notes
    t = resp.table
    x = _x_values(t, resp.x)
    for k, name in enumerate(resp.ys):
        ax.plot(x, t.numeric(name), color=LINE_COLOURS[k % len(LINE_COLOURS)], lw=1.4,
                label=name + unit_suffix(t.columns[name].units))
    rows = run.step_rows(t)
    y = t.numeric(resp.primary)
    for step, reason in chosen:
        if step in rows:
            r = rows[step]
            ax.plot(x[r], y[r], "s", color="#A3223F", ms=5, zorder=5)
            ax.annotate(f"step {step}", (x[r], y[r]), xytext=(4, 4), textcoords="offset points", fontsize=7.5,
                        color="#A3223F")
    ax.set_xlabel(_col_label(t, resp.x))
    ax.set_ylabel(_col_label(t, resp.primary) if len(resp.ys) == 1 else "load-like columns")
    if len(resp.ys) > 1:
        _legend_below(ax)
    ax.set_title("Load and response", loc="left")
    return notes


def _other_panel(run: RunData, ax) -> bool:
    """Per-step quantities when no load-like column is stored."""
    for table in run.tables:
        drive = run.drive(table)
        ys = [n for n in table.columns if n != drive and re.search(r"displacement|disp|strain|stress|load", n)
              and not re.search(r"^(load_|applied_|time|dt$|increment$|step$)", n)
              and table.numeric(n) is not None and np.ptp(np.nan_to_num(table.numeric(n))) > 0]
        if ys:
            x = _x_values(table, drive)
            for k, n in enumerate(ys[:4]):
                ax.plot(x, table.numeric(n), color=LINE_COLOURS[k], lw=1.4, marker="o", ms=3,
                        label=n + unit_suffix(table.columns[n].units))
            ax.set_xlabel(_col_label(table, drive))
            _legend_below(ax)
            ax.set_title("Stored response quantities", loc="left")
            return True
    return False


def _energy_source(run: RunData) -> tuple[Table, list[str]] | None:
    skip = {"step", "time", "time_s", "increment", "dt"}
    for table in run.tables:
        names = [n for n in table.columns if n.startswith("energy_")]
        if names:
            return table, names
    energy = run.table("energy")
    if energy is not None:
        return energy, [n for n in energy.columns if n not in skip]
    for table in run.tables:
        names = [n for n in table.columns if n in ("elastic", "fracture", "external", "kinetic", "total",
                                                    "balance_residual", "balance_residual_rel")]
        if names:
            return table, names
    return None


def _energy_panel(run: RunData, ax) -> tuple[list[str], bool]:
    src = _energy_source(run)
    if src is None:
        return [], False
    table, names = src
    names = [n for n in names if table.numeric(n) is not None and np.ptp(np.nan_to_num(table.numeric(n))) > 0]
    if not names:
        return [], False
    drive = run.drive(table)
    x = _x_values(table, drive)
    relative = [n for n in names if n.endswith(("_rel", "_relative"))]
    main = [n for n in names if n not in relative]
    units = {table.columns[n].units for n in main} - {None}
    long_methods: list[str] = []
    for k, n in enumerate(main):
        col = table.columns[n]
        label = n[len("energy_"):] if n.startswith("energy_") else n
        if col.method:
            method = clean(col.method)
            if method and len(method) <= MAX_ATTRIBUTE_TEXT:
                label += f" ({method})"
            else:
                long_methods.append(label)
        ls = "--" if "balance" in n else "-"
        ax.plot(x, table.numeric(n), ls, color=LINE_COLOURS[k % len(LINE_COLOURS)], lw=1.3,
                label="\n".join(textwrap.wrap(label, 46)))
    ax.set_xlabel(_col_label(table, drive))
    ax.set_ylabel("energy" + (f" ({units.pop()})" if len(units) == 1 else ""))
    _legend_below(ax, ncol=1)
    if relative:
        ax2 = ax.twinx()
        for n in relative:
            ax2.plot(x, 100 * np.abs(table.numeric(n)), ":", color="#56606B", lw=1.1)
        ax2.set_ylabel("|relative balance residual| (%), dotted", color="#56606B", fontsize=8)
        ax2.tick_params(labelsize=7.5)
    ax.set_title("Energies", loc="left")
    notes = []
    if long_methods:
        where = "history/energy/<term>"
        notes.append(f"Energies: the method attributes of {', '.join(long_methods)} are longer than "
                     f"{MAX_ATTRIBUTE_TEXT} characters and are not shown; they are stored in {where}.")
    return notes, True


def _damage_panel(run: RunData, ax, measures) -> list[str]:
    notes = []
    resp = run.response()
    base = resp.table if resp is not None else (run.table("history")
                                                  or next(iter(run.tables), None))
    stored = []
    for table in ([base] if base is not None else []) + run.tables:
        names = [n for n in table.columns if n in ("max_damage", "max_d", "damaged_volume_fraction", "mean_damage")
                 and table.numeric(n) is not None]
        if names:
            base, stored = table, names
            break
    drive = run.drive(base) if base is not None else None
    if base is not None and stored:
        x = _x_values(base, drive)
        kept: list[str] = []
        for n in stored:     # one line per distinct column (aliases repeat values)
            if not any(np.array_equal(base.numeric(n), base.numeric(m), equal_nan=True) for m in kept):
                kept.append(n)
        for k, n in enumerate(kept):
            ax.plot(x, base.numeric(n), color=LINE_COLOURS[k], lw=1.3, label=f"{n} (stored column)")
    mapped = False
    if measures is not None:
        rows = run.step_rows(base) if base is not None else {}
        steps = list(measures["step"].astype(int))
        if base is not None and rows and all(s in rows for s in steps):
            xs = _x_values(base, drive)[[rows[s] for s in steps]]
            mapped = True
        else:
            xs = measures["step"]
        ax.plot(xs, measures["max"], "o", color="#A3223F", ms=4, mfc="none", label="largest $d$ (from the fields)")
        ax.plot(xs, measures["fraction"], "s", color="#16191D", ms=3.5, mfc="none",
                label="area or volume share with $d > 0.95$ (from the fields)")
        notes.append("Damage panel: markers are computed here from the stored damage at the stored steps; the share uses "
                     "cell means of nodal damage, weighted by cell size.")
    if base is None and measures is None:
        return []
    ax.set_xlabel(_col_label(base, drive) if (stored or mapped) else "stored step")
    ax.set_ylabel("damage measure")
    top = max([float(np.nanmax(measures["max"])) if measures is not None else 0.0]
              + [float(np.nanmax(base.numeric(n))) for n in stored if np.isfinite(base.numeric(n)).any()])
    if top > 0.1 or top == 0:
        ax.set_ylim(-0.02, 1.05)
    if not (stored or mapped):
        _integer_axis(ax, None)
    _legend_below(ax, ncol=1)
    ax.set_title("Damage", loc="left")
    return notes


def _residual_panel(run: RunData, ax) -> bool:
    for table in run.tables:
        names = [n for n in table.columns if (("residual" in n and "balance" not in n) or "kkt" in n)
                 and table.numeric(n) is not None and np.nanmax(np.abs(np.nan_to_num(table.numeric(n)))) > 0
                 and not set(np.unique(table.numeric(n)[np.isfinite(table.numeric(n))])) <= {0.0, 1.0}]  # no flags
        if not names:
            continue
        x_name = "increment" if "increment" in table.columns else "step" if "step" in table.columns else None
        x = _x_values(table, x_name)
        for k, n in enumerate(names[:5]):
            v = np.abs(table.numeric(n))
            ax.semilogy(x, np.where(v > 0, v, np.nan), marker="o" if table.n < 40 else None, ms=3, lw=1.1,
                        color=LINE_COLOURS[k % len(LINE_COLOURS)], ls="--" if n == "tolerance" else "-",
                        label=n)
        ax.set_xlabel(x_name or "history row")
        _integer_axis(ax, x_name)
        ax.set_ylabel("residual (as stored)")
        ax.yaxis.set_minor_locator(LogLocator(subs="auto"))       # label minor ticks on short log ranges
        ax.yaxis.set_minor_formatter(LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(1.5, 0.6)))
        ax.tick_params(axis="y", which="minor", length=2, labelsize=7)
        _legend_below(ax)
        ax.set_title("Solver residuals", loc="left")
        return True
    return False


def _panel_drawers(run: RunData, chosen, measures) -> list[tuple[str, Any]]:
    out = []
    if run.response() is not None:
        out.append(("response", lambda ax: (_response_panel(run, ax, chosen), True)))
    else:
        out.append(("other", lambda ax: ([], _other_panel(run, ax))))
    out.append(("energy", lambda ax: _energy_panel(run, ax)))
    if measures is not None or any(n in t.columns for t in run.tables for n in ("max_damage", "max_d")):
        out.append(("damage", lambda ax: (_damage_panel(run, ax, measures), True)))
    out.append(("residual", lambda ax: ([], _residual_panel(run, ax))))
    return out


def evolution_sheet(run: RunData, chosen, measures, path: Path, dpi: int) -> dict[str, Any] | None:
    # find the panels that have data first (drawn once on a scratch figure), then lay out only those
    available = []
    with figure(4, 3) as scratch:
        for name, draw in _panel_drawers(run, chosen, measures):
            ax = scratch.add_subplot(1, 1, 1)
            _, ok = draw(ax)
            scratch.delaxes(ax)
            if ok:
                available.append((name, draw))
    if not available:
        return None
    notes = run.notes_for("history", "steps")
    if available[0][0] not in ("response", "other"):
        notes.append("No reaction or other load-like history column is stored; the load and response panel "
                     "is omitted.")
    ncols = 3 if len(available) >= 5 else 2 if len(available) >= 2 else 1
    nrows = math.ceil(len(available) / ncols)
    width, height = 4.6 * ncols, 4.2 * nrows + 0.6
    with figure(width, height, layout="constrained") as fig:
        gs = fig.add_gridspec(nrows, ncols)
        for k, (name, draw) in enumerate(available):
            ax = fig.add_subplot(gs[k // ncols, k % ncols])
            more, _ = draw(ax)
            notes += [n for n in more if n not in notes]
            ax.grid(True, lw=0.4, alpha=0.35)
            ax.set_title(f"({'abcdefgh'[k]}) " + ax.get_title(loc="left"), loc="left")
        heading(fig, f"Evolution. {describe(run)}")
        footnote(fig, notes, width)
        w, h = save(fig, path, dpi)
    columns = sorted({n for t in run.tables for n in t.columns})
    return dict(file=path.name, kind="evolution", step=None, steps=[s for s, _ in chosen], fields=["damage"]
                if measures is not None else [], history_columns=columns[:200], width_px=w, height_px=h, notes=notes)
