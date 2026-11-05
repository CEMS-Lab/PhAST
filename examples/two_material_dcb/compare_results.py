"""Replot retained DCB comparison data without running the FEM solver.

The archived pair has a common mesh but different accepted load increments.
Each response is therefore plotted against its own opening displacement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Circle
import numpy as np


def main() -> int:
    """Check the compact data and write the two comparison figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/dcb_comparison_replot"),
        help="New plot directory; the retained input data are not modified.",
    )
    args = parser.parse_args()
    data_dir = Path(__file__).resolve().parent / "results"
    record = json.loads((data_dir / "comparison_provenance.json").read_text())
    for name, expected in record["files"].items():
        digest = hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        if digest != expected["sha256"]:
            raise ValueError(f"Retained comparison data changed: {name}")

    tough = np.genfromtxt(
        data_dir / "coarse_tough_history.csv", delimiter=",", names=True,
    )
    uniform = np.genfromtxt(
        data_dir / "uniform_layer_history.csv", delimiter=",", names=True,
    )
    fields = np.genfromtxt(
        data_dir / "matched_control_fields.csv.gz", delimiter=",", names=True,
    )
    elements = np.loadtxt(
        data_dir / "matched_control_elements.csv.gz",
        delimiter=",", skiprows=1, dtype=np.int64,
    )
    for history in (tough, uniform):
        for name in ("opening", "reaction_force", "connected_crack_front_x"):
            if not np.isfinite(history[name]).all():
                raise ValueError(f"Non-finite retained history: {name}")
        if np.any(np.diff(history["opening"]) < 0):
            raise ValueError("Retained accepted openings are not monotonic.")
    for name in fields.dtype.names or ():
        if not np.isfinite(fields[name]).all():
            raise ValueError(f"Non-finite retained final field: {name}")
    if not np.array_equal(fields["node"], np.arange(len(fields))):
        raise ValueError("Final fields do not follow the stored node numbering.")
    if elements.ndim != 2 or elements.shape[1] != 3:
        raise ValueError("Expected the original three-node triangle connectivity.")
    if elements.min() < 0 or elements.max() >= len(fields):
        raise ValueError("Connectivity contains an out-of-range node index.")
    for name in ("damage_tough", "damage_uniform"):
        if np.any((fields[name] < 0) | (fields[name] > 1)):
            raise ValueError(f"Retained damage is outside [0, 1]: {name}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "stix",
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    labels = ("Tough circular region", "Uniform weak layer")
    colours = ("#9f3b31", "#285f87")
    styles = ("-", "--")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for history, label, colour, style in zip(
        (tough, uniform), labels, colours, styles,
    ):
        axes[0].plot(
            history["opening"], history["reaction_force"],
            linestyle=style, color=colour, label=label,
        )
        axes[1].plot(
            history["opening"], history["connected_crack_front_x"],
            linestyle=style, color=colour, label=label,
        )
    axes[0].set_ylabel("Opening reaction per unit thickness [N/mm]")
    axes[1].set_ylabel("Maximum connected x at d >= 0.80 [mm]")
    axes[1].axhline(2.0, color="0.5", linestyle=":", linewidth=1)
    axes[1].text(0.01, 2.02, "Left edge of circular region", fontsize=9)
    for axis in axes:
        axis.set_xlabel("Opening displacement [mm]")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    figure.suptitle(
        "Retained coarse-mesh comparison: identical prescribed loading, "
        "separate accepted increments",
        fontsize=11,
    )
    figure.savefig(args.output_dir / "matched_control_comparison.png", dpi=160)
    plt.close(figure)

    triangulation = mtri.Triangulation(fields["x_mm"], fields["y_mm"], elements)
    figure, axes = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)
    for axis, name, label, history in zip(
        axes, ("damage_tough", "damage_uniform"), labels, (tough, uniform),
    ):
        artist = axis.tripcolor(
            triangulation, fields[name], shading="gouraud",
            cmap="Reds", vmin=0, vmax=1,
        )
        # Geometry belongs to the frozen comparison, not an edited new input.
        axis.add_patch(Circle((2.2, 0), 0.2, fill=False, linestyle="--", color="0.25"))
        axis.axhline(-0.3, color="0.6", linestyle=":", linewidth=0.8)
        axis.axhline(0.3, color="0.6", linestyle=":", linewidth=0.8)
        axis.set(
            aspect="equal", xlabel="x [mm]", ylabel="y [mm]",
            title=f"{label}; final opening {history['opening'][-1]:.2f} mm",
        )
        figure.colorbar(artist, ax=axis, label="Damage d", shrink=0.8)
    figure.suptitle(
        "Final nodal damage on the original shared mesh; interpolated for display",
        fontsize=11,
    )
    figure.savefig(args.output_dir / "matched_control_damage.png", dpi=160)
    plt.close(figure)
    print(f"Replotted retained data in {args.output_dir}")
    print(
        "These figures show a qualitative confined-layer comparison. "
        "They do not establish penetration, completed bypass, or physical validation."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
