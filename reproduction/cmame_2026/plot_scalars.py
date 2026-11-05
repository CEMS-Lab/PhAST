"""Plot the recorded scalar inversions without importing the solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(material: str, output: Path) -> None:
    source = Path(__file__).resolve().parent / "scalar" / material / "recorded_result.json"
    record = json.loads(source.read_text())
    history = record["history"]
    accepted = np.asarray(history["iter"]) + 1
    parameter = np.r_[record["Gc_init"], history["Gc"]]
    iterations = np.r_[0, accepted]
    error = np.abs(parameter - record["Gc_true"]) / record["Gc_true"]
    # Glass saved the closure loss before the update; alumina saved the accepted loss.
    loss_iterations = accepted - 1 if material == "glass" else accepted
    scale = 1000.0
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), layout="constrained")
    axes[0].plot(iterations, parameter * scale, "o-", color="#0072b2", markersize=4)
    axes[0].axhline(record["Gc_true"] * scale, color="#d55e00", linestyle="--",
                    label="Original value")
    axes[0].set_ylabel(r"$G_c$ (J/m$^2$)")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].semilogy(loss_iterations, history["loss"], "o-", color="#d55e00", markersize=4)
    axes[1].set_ylabel("Mean squared damage error")
    axes[2].semilogy(iterations, np.maximum(error, np.finfo(float).tiny),
                     "o-", color="#009e73", markersize=4)
    axes[2].set_ylabel("Relative toughness error")
    for ax in axes:
        ax.set_xlabel("Optimiser iteration")
        ax.grid(alpha=0.2)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{material}_scalar_recovery.{suffix}", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"],
                         "mathtext.fontset": "stix", "font.size": 10})
    for material in ("glass", "alumina"):
        plot(material, args.output_dir)


if __name__ == "__main__":
    main()
