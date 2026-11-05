"""Redraw the setup figure of a completed displacement-recovery run and update its manifest entry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_setup(fields: np.ndarray, elements: np.ndarray, config: dict):
    geometry = config["geometry"]
    width, height = geometry["width_mm"], geometry["height_mm"]
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    tri = mtri.Triangulation(fields[:, 1], fields[:, 2], elements)
    ax.triplot(tri, color="0.65", lw=0.25)
    ax.plot([0, geometry["notch_length_mm"]], [height / 2] * 2,
            color="black", lw=1.0)
    loading = config["loading"]
    top = loading.get("top_displacement_mm", loading.get("maximum_displacement_mm"))
    labels = [ax.text(width / 2, 1.21 * height,
                      rf"Top: $u_y={top:+g}$ mm", ha="center", va="center")]
    ax.annotate("", xy=(width / 2, 1.14 * height), xytext=(width / 2, height),
                arrowprops={"arrowstyle": "->", "color": "tab:blue"})
    if loading["type"] == "ramped_displacement":
        bottom = loading["bottom_displacement_mm"]
        ax.annotate("", xy=(width / 2, -0.14 * height), xytext=(width / 2, 0),
                    arrowprops={"arrowstyle": "->", "color": "tab:blue"})
        labels.append(ax.text(width / 2, -0.21 * height,
                              rf"Bottom: $u_y={bottom:+g}$ mm",
                              ha="center", va="center"))
        labels.extend(ax.text(x, height / 2, r"$u_x=0$", ha="center",
                              va="center", rotation=90)
                      for x in (-0.12 * width, 1.12 * width))
    elif loading["type"] == "quasistatic_displacement":
        ax.plot([0, width], [0, 0], color="tab:blue", lw=2)
        labels.append(ax.text(width / 2, -0.14 * height,
                              r"Bottom: $u_x=u_y=0$", ha="center", va="center"))
    else:
        plt.close(fig)
        raise ValueError("Only the documented displacement loading is supported")
    ax.set(xlim=(-0.22 * width, 1.22 * width),
           ylim=(-0.28 * height, 1.29 * height), aspect="equal",
           xlabel=r"$x$ (mm)", ylabel=r"$y$ (mm)")
    fig.suptitle("Original mesh and prescribed displacements", fontsize=12)
    return fig, labels


def replot(output: Path) -> None:
    summary = json.loads((output / "summary.json").read_text())
    if summary.get("success") is not True:
        raise ValueError("Replot only a completed successful result")
    if summary["example"] not in {"cross_mesh_toughness", "joint_toughness_length"}:
        raise ValueError("Only cross-mesh and joint displacement examples are supported")
    metadata = json.loads((output / "run_metadata.json").read_text())
    visuals_path = output / "visual_manifest.json"
    visuals = json.loads(visuals_path.read_text())
    matches = [row for row in visuals if row["file"] == "initial_conditions.png"]
    if len(matches) != 1:
        raise ValueError("The visual manifest must have exactly one setup entry")
    fields = np.loadtxt(output / "original_fields.csv", delimiter=",", skiprows=1,
                        ndmin=2)
    elements = np.loadtxt(output / "original_elements.csv", delimiter=",", skiprows=1,
                         dtype=int, ndmin=2)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix"})
    fig, _ = make_setup(fields, elements, metadata["config"])
    path = output / "initial_conditions.png"
    fig.savefig(path, dpi=160)
    dimensions = [round(value * 160) for value in fig.get_size_inches()]
    plt.close(fig)
    sources = ("original_fields.csv", "original_elements.csv", "run_metadata.json")
    matches[0].update({
        "dimensions": dimensions, "size_bytes": path.stat().st_size,
        "sha256": sha256(path), "source": "; ".join(sources),
        "source_sha256": {name: sha256(output / name) for name in sources},
        "generator": "../../../tools/replot_inverse_setup.py",
        "generator_sha256": sha256(Path(__file__)),
        "role": "setup_only_replot; numerical_run_identity_unchanged",
    })
    visuals_path.write_text(json.dumps(visuals, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    replot(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
