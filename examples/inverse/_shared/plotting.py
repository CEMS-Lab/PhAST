"""Plot the fields and accepted optimisation history from a completed run."""

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
from matplotlib.tri import Triangulation
import numpy as np
from PIL import Image


def read_fields(output: Path, label: str) -> tuple[np.ndarray, Triangulation]:
    fields = np.loadtxt(output / f"{label}_fields.csv", delimiter=",", skiprows=1, ndmin=2)
    elements = np.loadtxt(output / f"{label}_elements.csv", delimiter=",", skiprows=1,
                          dtype=np.int64, ndmin=2)
    return fields, Triangulation(fields[:, 1], fields[:, 2], elements)


def plot_output(output: Path) -> None:
    plt.style.use("default")
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "axes.unicode_minus": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))["config"]
    original, target_tri = read_fields(output, "original")
    recovered, trial_tri = read_fields(output, "recovered")
    observations = np.loadtxt(output / "observations.csv", delimiter=",", skiprows=1, ndmin=2)
    maximum = min(1.0, max(float(original[:, 3].max()), float(recovered[:, 3].max())))
    maximum = max(maximum, 1.0e-6)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), layout="constrained")
    for ax, fields, triangulation, title in zip(
        axes[:2], (original, recovered), (target_tri, trial_tri), ("Original run", "Recovered run"),
    ):
        image = ax.tripcolor(triangulation, fields[:, 3], cmap="Reds", vmin=0.0,
                            vmax=maximum, shading="gouraud", rasterized=True)
        ax.set_title(f"{title}: max d = {fields[:, 3].max():.4g}")
        fig.colorbar(image, ax=ax, label="Damage d")
    difference = np.abs(observations[:, 3] - observations[:, 2])
    image = axes[2].scatter(observations[:, 0], observations[:, 1], c=difference, s=4,
                            cmap="plasma", vmin=0.0, vmax=max(float(difference.max()), 1.0e-12))
    axes[2].set_title("Absolute observation error")
    fig.colorbar(image, ax=axes[2], label="|d recovered - d original|")
    axes[2].set_xlim(0.0, config["geometry"]["width_mm"])
    axes[2].set_ylim(0.0, config["geometry"]["height_mm"])
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
    fig.savefig(output / "damage_comparison.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3), layout="constrained")
    image = ax.tripcolor(trial_tri, recovered[:, 3], cmap="Reds", shading="gouraud",
                        vmin=0.0, vmax=maximum, rasterized=True)
    ax.set(xlabel="x [mm]", ylabel="y [mm]", title="Recovered final damage")
    ax.set_aspect("equal")
    fig.colorbar(image, ax=ax, label="Damage d")
    fig.savefig(output / "damage_final.png", dpi=160)
    fig.savefig(output / "thumbnail.png", dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3), layout="constrained")
    ax.triplot(target_tri, color="0.65", lw=0.25)
    height = config["geometry"]["height_mm"]
    width = config["geometry"]["width_mm"]
    for y, sign in ((height, 1), (0.0, -1)):
        for x in (width * 0.2, width * 0.5, width * 0.8):
            ax.annotate("", xy=(x, y + sign * height * 0.12), xytext=(x, y),
                        arrowprops={"arrowstyle": "->", "color": "black"})
    ax.plot([0, 0], [0, height], color="black", lw=2, label="Left: u_x = 0")
    ax.set(xlabel="x [mm]", ylabel="y [mm]", title=f"SENT mesh; traction = {config['loading']['traction_MPa']:g} MPa")
    ax.set_ylim(-0.18 * height, 1.18 * height)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(output / "initial_conditions.png", dpi=160)
    plt.close(fig)
    history = np.loadtxt(output / "history.csv", delimiter=",", skiprows=1, ndmin=2)
    fig, axes = plt.subplots(1, 3 if "l0" in config["recover"] else 2,
                             figsize=(8, 3), layout="constrained")
    axes[0].semilogy(history[:, 0], np.maximum(history[:, 4], 1.0e-30), "o-", color="black", ms=3)
    axes[0].set(xlabel="Accepted iteration", ylabel="Mean squared damage error")
    for ax, name, column, unit in zip(axes[1:], config["recover"], (2, 3), ("N/mm", "mm")):
        ax.plot(history[:, 0], history[:, column], "o-", color="tab:blue", ms=3, label="Recovered")
        ax.axhline(config["parameters"][name]["truth"], color="tab:red", ls="--", label="Original value")
        label = r"$G_c$" if name == "Gc" else r"$\ell_0$"
        ax.set(xlabel="Accepted iteration", ylabel=f"{label} [{unit}]")
        ax.legend(fontsize=8, frameon=False)
    fig.savefig(output / "recovery_history.png", dpi=180)
    plt.close(fig)
    snapshots_path = output / "recovered_snapshots.npz"
    if snapshots_path.exists():
        with np.load(snapshots_path) as arrays:
            frames = arrays["damage"]
        fig, ax = plt.subplots(figsize=(6, 2.8), layout="constrained")
        image = ax.tripcolor(trial_tri, frames[0], cmap="Reds", vmin=0.0, vmax=maximum,
                            shading="gouraud")
        ax.set(xlabel="x [mm]", ylabel="y [mm]")
        ax.set_aspect("equal")
        fig.colorbar(image, ax=ax, label="Damage d")
        steps = np.r_[0, np.linspace(1, summary["steps"], min(20, summary["steps"]), dtype=int)]

        def update(frame: int) -> tuple[Any, ...]:
            image.set_array(frames[frame])
            ax.set_title(f"Recovered: t = {steps[frame] * summary['dt_s']:.3g} s")
            return (image,)

        animation = FuncAnimation(fig, update, frames=len(frames), interval=150)
        animation.save(output / "damage_evolution.gif", writer=PillowWriter(fps=6), dpi=110)
        plt.close(fig)
    rows = []
    for name, source in (("damage_comparison.png", "original_fields.csv; recovered_fields.csv; observations.csv"),
                         ("damage_final.png", "recovered_fields.csv"),
                         ("thumbnail.png", "recovered_fields.csv"),
                         ("initial_conditions.png", "original_elements.csv; config.yaml"),
                         ("recovery_history.png", "history.csv"),
                         ("damage_evolution.gif", "recovered_snapshots.npz")):
        path = output / name
        if path.exists():
            with Image.open(path) as image_file:
                dimensions = list(image_file.size)
            rows.append({"file": name, "artifact_type": "animation" if name.endswith("gif") else "image",
                         "source": source, "dimensions": dimensions, "role": "synthetic_recovery",
                         "size_bytes": path.stat().st_size,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "generator": "../_shared/plotting.py",
                         "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (output / "visual_manifest.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replot a synthetic recovery result directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot_output(args.output_dir)
