"""Input-only material and boundary-condition diagrams for 2D teaching cases.

The caller supplies the mesh, material assignments, and boundary selections
resolved by the calculation. No displacement or damage solution is computed.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

from .specs import BoundaryConditionSpec, ProblemSpec


MATERIAL_COLOURS = {
    "material_1": "#8AB0C4",
    "weak_midline": "#D8D8D1",
    "material_2": "#D6A24B",
}
MATERIAL_LABELS = {
    "material_1": "Material 1: outer regions",
    "weak_midline": "Weak bulk layer",
    "material_2": "Material 2: circular region",
}


def save_material_setup(
    spec: ProblemSpec,
    nodes: np.ndarray,
    elements: np.ndarray,
    material_ids: np.ndarray,
    material_names: Sequence[str],
    node_regions: dict[str, np.ndarray],
    boundary_conditions: Sequence[tuple[BoundaryConditionSpec, Any]],
    output_path: str | Path,
) -> Path:
    """Draw the assigned T3 materials and selected nodes in reference coordinates.

    This diagram is intended for rectangular 2D teaching geometries. Material
    units are labelled for the N-mm convention when geometry units are mm;
    otherwise the table explicitly refers to the configuration's input units.
    Arrow lengths indicate direction only, not the displacement magnitude.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xmin, ymin = nodes.min(axis=0)[:2]
    xmax, ymax = nodes.max(axis=0)[:2]
    length, height = xmax - xmin, ymax - ymin
    units = spec.geometry.units if spec.geometry is not None else "input units"
    colours = [
        MATERIAL_COLOURS.get(name, f"C{index % 10}")
        for index, name in enumerate(material_names)
    ]
    material_by_name = {material.name: material for material in spec.materials}
    colour_map = ListedColormap(colours)
    colour_norm = BoundaryNorm(np.arange(len(colours) + 1) - 0.5, len(colours))

    # Apply the public figure style locally, without changing later field plots.
    with plt.rc_context():
        plt.style.use("default")
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "text.usetex": False,
            "axes.unicode_minus": False,
            "font.size": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": "black",
        })
        figure = plt.figure(figsize=(12, 8))
        figure.text(0.075, 0.95, "Layered DCB: materials and input conditions",
                    fontsize=21, weight="bold")
        figure.text(0.075, 0.912,
                    "Reference geometry and prescribed data, not a computed deformation or crack path.",
                    fontsize=12)
        axis = figure.add_axes((0.075, 0.49, 0.86, 0.34))
        axis.tripcolor(nodes[:, 0], nodes[:, 1], elements,
                       facecolors=material_ids, shading="flat",
                       cmap=colour_map, norm=colour_norm, rasterized=True)
        axis.plot([xmin, xmax, xmax, xmin, xmin],
                  [ymin, ymin, ymax, ymax, ymin], color="black", linewidth=0.8)

        # The black seed is drawn from the same node selection used by PhAST.
        for initial in spec.initial_conditions:
            if initial.field != "damage" or initial.region not in node_regions:
                continue
            indices = np.flatnonzero(node_regions[initial.region])
            if not indices.size:
                continue
            points = nodes[indices, :2]
            axis.scatter(points[:, 0], points[:, 1], s=8, color="black", zorder=4)
            span = np.ptp(points[:, 0])
            axis.annotate(
                f"Initial crack: $d={float(initial.value):g}$\n"
                f"Selected length = {span:g} {units}",
                xy=points.mean(axis=0),
                xytext=(xmin + 0.22 * length, ymin - 0.40 * height),
                ha="center", va="center", fontsize=10,
                arrowprops={"arrowstyle": "-", "color": "black", "lw": 0.8},
            )

        # Merge x/y constraints on the same edge into one readable annotation.
        fixed_regions: dict[str, set[int]] = {}
        vertical_values: list[float] = []
        for boundary, indices in boundary_conditions:
            components = (0, 1) if boundary.component is None else (boundary.component,)
            if boundary.kind == "fix":
                fixed_regions.setdefault(boundary.region, set()).update(components)
                continue
            if boundary.kind != "prescribe":
                continue
            points = nodes[np.asarray(indices, dtype=int), :2]
            centre = points.mean(axis=0)
            axis.scatter(points[:, 0], points[:, 1], s=9, color="black", zorder=4)
            value = float(boundary.value)
            for component in components:
                direction = 1.0 if value >= 0.0 else -1.0
                start = centre.copy()
                start[0] -= 0.035 * length
                end = start.copy()
                end[component] += direction * 0.22 * height
                axis.annotate("", xy=end, xytext=start,
                              arrowprops={"arrowstyle": "-|>", "lw": 1.8, "color": "black"})
                symbol = "x" if component == 0 else "y"
                axis.text(
                    xmin - 0.075 * length, centre[1] + direction * 0.14 * height,
                    f"{boundary.name or boundary.region}\n$u_{symbol}={value:+g}$ {units}",
                    ha="right", va="center", fontsize=11,
                )
                if component == 1:
                    vertical_values.append(value)
        for region, components in fixed_regions.items():
            points = nodes[np.flatnonzero(node_regions[region]), :2]
            centre = points.mean(axis=0)
            axis.scatter(points[:, 0], points[:, 1], marker="+", s=25,
                         linewidths=1.2, color="black", zorder=5)
            label = ", ".join(f"$u_{'x' if component == 0 else 'y'}=0$"
                              for component in sorted(components))
            axis.annotate(
                f"{region.replace('_', ' ')}\n{label}", xy=centre,
                xytext=(xmax + 0.055 * length, centre[1]), ha="left", va="center",
                fontsize=11, arrowprops={"arrowstyle": "-", "lw": 0.8, "color": "black"},
            )

        # Label the declared circular geometry; its material assignment remains
        # the element-centre selection shown by the coloured triangles.
        for region in spec.regions:
            selector = region.selector
            if selector.get("type") != "circle":
                continue
            centre = np.asarray(selector["center"], dtype=float)
            radius = float(selector["radius"])
            axis.annotate(
                f"Circular region: $R={radius:g}$ {units}", xy=centre,
                xytext=(centre[0] + 0.16 * length, ymax + 0.15 * height),
                ha="left", va="center", fontsize=10,
                arrowprops={"arrowstyle": "-", "lw": 0.8, "color": "black"},
            )
        dimension_y = ymax + 0.48 * height
        axis.annotate("", xy=(xmin, dimension_y), xytext=(xmax, dimension_y),
                      arrowprops={"arrowstyle": "<->", "lw": 0.8, "color": "black"})
        axis.text((xmin + xmax) / 2.0, dimension_y + 0.035 * height,
                  f"Length = {length:g} {units}", ha="center", va="bottom", fontsize=10)
        axis.set(xlim=(xmin - 0.32 * length, xmax + 0.26 * length),
                 ylim=(ymin - 0.60 * height, ymax + 0.75 * height),
                 xlabel=f"x [{units}]", ylabel=f"y [{units}]", aspect="equal")
        axis.set_yticks([ymin, (ymin + ymax) / 2.0, ymax])
        axis.spines[["top", "right"]].set_visible(False)

        table_axis = figure.add_axes((0.075, 0.275, 0.86, 0.155))
        table_axis.set_axis_off()
        modulus_label = "$E$ [MPa]" if units == "mm" else "$E$ [input units]"
        toughness_label = "$G_c$ [N/mm]" if units == "mm" else "$G_c$ [input units]"
        rows = []
        for name in material_names:
            parameters = material_by_name[name].parameters
            rows.append([
                MATERIAL_LABELS.get(name, name.replace("_", " ")),
                f"{float(parameters['E']):g}",
                f"{float(parameters['Gc']):g}",
            ])
        table = table_axis.table(cellText=rows,
                                 colLabels=["Material region", modulus_label, toughness_label],
                                 colWidths=[0.56, 0.22, 0.22], cellLoc="left",
                                 colLoc="left", bbox=(0, 0, 1, 1))
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        for (row, column), cell in table.get_celld().items():
            cell.set_edgecolor("white")
            cell.set_linewidth(2)
            if row == 0:
                cell.set_facecolor("#F2F2F0")
                cell.set_text_props(weight="bold")
            elif column == 0:
                cell.set_facecolor(colours[row - 1])

        first_material = spec.materials[0].parameters
        step = spec.analysis_steps[0]
        increments = max(int(step.controls.get("number_of_steps", 2)) - 1, 1)
        opening = ""
        if vertical_values and min(vertical_values) < 0 < max(vertical_values):
            opening = (f"Final relative opening: {max(vertical_values) - min(vertical_values):g} {units}. "
                       "Prescribed values above are at load factor 1.\n")
        figure.text(
            0.075, 0.229,
            f"Geometry: {length:g} x {height:g} {units}. "
            f"Mesh: {len(nodes):,} nodes; {len(elements):,} linear triangles.\n"
            f"Shared inputs: $\\nu={float(first_material['nu']):g}$; "
            f"$\\ell_0={float(first_material['l0']):g}$ {units}. "
            f"Analysis: {step.kind.replace('_', ' ')}; {increments} nominal load increments.",
            fontsize=11, va="top", linespacing=1.6,
        )
        figure.text(0.075, 0.136, opening +
                    "Material colours identify bulk regions, not damage. Unprescribed tractions are zero.",
                    fontsize=11, va="top", linespacing=1.6)
        figure.text(0.075, 0.045,
                    "Source: YAML selections resolved on the PhAST mesh. Arrow lengths are schematic. "
                    "No simulation was run to produce this diagram.", fontsize=9)
        figure.savefig(output_path, dpi=150, facecolor="white")
        plt.close(figure)
    return output_path
