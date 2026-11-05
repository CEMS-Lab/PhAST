"""Full-field plots for displacement-based recovery results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from PIL import Image

from .plotting import read_fields
from .recovery import write_json


def plot_output(output: Path) -> None:
    plt.style.use("default")
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "axes.unicode_minus": False})
    summary = json.loads((output / "summary.json").read_text())
    config = json.loads((output / "run_metadata.json").read_text())["config"]
    original, target_tri = read_fields(output, "original")
    recovered, trial_tri = read_fields(output, "recovered")
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.6), layout="constrained")
    for ax, fields, tri, title in zip(axes, (original, recovered), (target_tri, trial_tri), ("Original run", "Recovered run")):
        image = ax.tripcolor(tri, fields[:, 3], shading="gouraud", cmap="Reds", vmin=0, vmax=1)
        ax.set(title=title, xlabel="x [mm]", ylabel="y [mm]", aspect="equal")
        fig.colorbar(image, ax=ax, label="Damage d")
    fig.savefig(output / "damage_comparison.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4, 3.6), layout="constrained")
    image = ax.tripcolor(trial_tri, recovered[:, 3], shading="gouraud", cmap="Reds", vmin=0, vmax=1)
    ax.set(xlabel="x [mm]", ylabel="y [mm]", aspect="equal", title="Recovered damage")
    fig.colorbar(image, ax=ax, label="Damage d")
    fig.savefig(output / "damage_final.png", dpi=150)
    fig.savefig(output / "thumbnail.png", dpi=100)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.6), layout="constrained")
    scale = max(float(np.linalg.norm(original[:, 4:6], axis=1).max()), float(np.linalg.norm(recovered[:, 4:6], axis=1).max()))
    for ax, fields, tri, title in zip(axes, (original, recovered), (target_tri, trial_tri), ("Original run", "Recovered run")):
        image = ax.tripcolor(tri, np.linalg.norm(fields[:, 4:6], axis=1), shading="gouraud",
                            cmap="plasma", vmin=0, vmax=scale)
        ax.set(title=title, xlabel="x [mm]", ylabel="y [mm]", aspect="equal")
        fig.colorbar(image, ax=ax, label="Displacement magnitude [mm]")
    fig.savefig(output / "displacement_comparison.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4, 3.6), layout="constrained")
    ax.triplot(target_tri, color="0.6", lw=.3)
    width, height = config["geometry"]["width_mm"], config["geometry"]["height_mm"]
    ax.annotate("Top: prescribed u_y", xy=(width / 2, height), xytext=(width / 2, height * 1.1),
                ha="center", fontsize=9, arrowprops={"arrowstyle": "->"})
    ax.set(xlabel="x [mm]", ylabel="y [mm]", aspect="equal", title="Mesh and loading")
    fig.savefig(output / "initial_conditions.png", dpi=160)
    plt.close(fig)
    history = np.genfromtxt(output / "history.csv", delimiter=",", names=True)
    history = np.atleast_1d(history)
    names = ["Gc"] if summary["example"] == "cross_mesh_toughness" else ["Gc", "l0"]
    fig, axes = plt.subplots(1, 1 + len(names), figsize=(8, 3), layout="constrained")
    axes[0].semilogy(history["evaluation"], np.maximum(history["loss"], 1e-30), "o-", color="black", ms=3)
    axes[0].set(xlabel="Objective evaluation", ylabel="Observation loss")
    for ax, name in zip(axes[1:], names):
        key = "Gc_N_per_mm" if name == "Gc" else "l0_mm"
        ax.plot(history["evaluation"], history[key], "o-", color="tab:blue", ms=3)
        ax.axhline(summary["truth"][name], color="tab:red", ls="--", label="Original value")
        label = r"$G_c$ [N/mm]" if name == "Gc" else r"$\ell_0$ [mm]"
        ax.set(xlabel="Objective evaluation", ylabel=label)
        ax.legend(fontsize=8, frameon=False)
    fig.savefig(output / "recovery_history.png", dpi=180)
    plt.close(fig)
    if (output / "recovered_snapshots.npz").exists():
        with np.load(output / "recovered_snapshots.npz") as arrays:
            frames = arrays["damage"]
        fig, ax = plt.subplots(figsize=(4, 3.6), layout="constrained")
        image = ax.tripcolor(trial_tri, frames[0], shading="gouraud", cmap="Reds", vmin=0, vmax=1)
        ax.set(xlabel="x [mm]", ylabel="y [mm]", aspect="equal")
        fig.colorbar(image, ax=ax, label="Damage d")

        def update(index: int) -> tuple[Any, ...]:
            image.set_array(frames[index])
            ax.set_title(f"Recovered state {index}/{len(frames) - 1}")
            return (image,)

        animation = FuncAnimation(fig, update, frames=len(frames), interval=150)
        animation.save(output / "damage_evolution.gif", writer=PillowWriter(fps=6), dpi=110)
        plt.close(fig)
    visuals = []
    for path in sorted(output.iterdir()):
        if path.suffix in {".png", ".gif"}:
            with Image.open(path) as image_file:
                dimensions = list(image_file.size)
            visuals.append({"file": path.name, "dimensions": dimensions,
                            "artifact_type": "animation" if path.suffix == ".gif" else "image",
                            "source": "recovered_snapshots.npz" if path.suffix == ".gif" else
                            "original_fields.csv; recovered_fields.csv; history.csv",
                            "role": "synthetic_recovery", "size_bytes": path.stat().st_size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "generator": "../_shared/current_plotting.py",
                            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    write_json(output / "visual_manifest.json", visuals)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replot displacement-based recovery outputs")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot_output(args.output_dir)
