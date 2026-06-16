"""Static setup preview artifacts for workflow problems."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .workflow.specs import ProblemSpec


class SetupPreviewError(ValueError):
    """Raised when a setup preview cannot be generated."""


def write_setup_preview(spec: ProblemSpec, output: str | Path) -> Path:
    """Write a non-interactive PNG preview for a workflow setup."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SetupPreviewError("Setup preview requires matplotlib") from exc

    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        if spec.mesh is not None and spec.mesh.path is not None:
            _plot_mesh(ax, Path(spec.mesh.path))
        else:
            _plot_contract_summary(ax, spec)
        _annotate_regions(ax, spec)
        _annotate_boundary_conditions(ax, spec)
        ax.set_title(spec.name)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
    finally:
        plt.close(fig)
    return output_path


def _plot_mesh(ax: Any, path: Path) -> None:
    try:
        import meshio
        import numpy as np
    except ImportError as exc:
        raise SetupPreviewError("Mesh setup preview requires meshio and numpy") from exc

    if not path.is_file():
        raise FileNotFoundError(f"Mesh file does not exist: {path}")
    mesh = meshio.read(path)
    points = np.asarray(mesh.points)[:, :2]
    plotted = False
    for block in mesh.cells:
        cells = np.asarray(block.data)
        if block.type in {"triangle", "tri3"} and cells.size:
            ax.triplot(points[:, 0], points[:, 1], cells, color="0.45", linewidth=0.8)
            plotted = True
        elif block.type in {"line", "line3"} and cells.size:
            for edge in cells[:, :2]:
                xy = points[edge]
                ax.plot(xy[:, 0], xy[:, 1], color="#b23b3b", linewidth=1.4)
            plotted = True
    if not plotted:
        ax.scatter(points[:, 0], points[:, 1], s=12, color="0.35")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _plot_contract_summary(ax: Any, spec: ProblemSpec) -> None:
    ax.axis("off")
    lines = [
        f"geometry: {spec.geometry.kind if spec.geometry else 'none'}",
        f"materials: {', '.join(material.name for material in spec.materials)}",
        f"steps: {', '.join(step.name for step in spec.analysis_steps)}",
    ]
    ax.text(
        0.02,
        0.95,
        "\n".join(lines),
        va="top",
        ha="left",
        transform=ax.transAxes,
        fontsize=11,
    )


def _annotate_regions(ax: Any, spec: ProblemSpec) -> None:
    if not spec.regions:
        return
    labels = [
        f"{region.name}: {region.selector.get('from_mesh') or region.kind}"
        for region in spec.regions
    ]
    ax.text(
        0.02,
        0.02,
        "regions\n" + "\n".join(labels),
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.7"},
    )


def _annotate_boundary_conditions(ax: Any, spec: ProblemSpec) -> None:
    if not spec.boundary_conditions:
        return
    labels = [
        f"{bc.name or bc.kind}: {bc.kind} {bc.region}"
        for bc in spec.boundary_conditions
    ]
    ax.text(
        0.98,
        0.02,
        "boundary conditions\n" + "\n".join(labels),
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.7"},
    )
