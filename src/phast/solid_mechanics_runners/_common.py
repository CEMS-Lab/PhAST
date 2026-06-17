"""Shared helpers for promoted solid-mechanics runners."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


def parse_config_arg(description: str) -> Path | None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the example YAML config. Defaults to config.yaml beside run.py.",
    )
    return parser.parse_args().config


def load_config(path: str | Path | None, defaults: dict[str, Any]) -> dict[str, Any]:
    """Load a small YAML config if PyYAML is available; fall back to defaults."""
    if path is None:
        path = Path(__file__).resolve().parent / "config.yaml"
    path = Path(path)
    cfg = dict(defaults)
    if not path.exists():
        return cfg
    try:
        import yaml
    except Exception:
        return cfg
    data = yaml.safe_load(path.read_text()) or {}
    return _deep_update(cfg, data)


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def prepare_output_dir(example_file: str | Path, cfg: dict[str, Any]) -> Path:
    env_out = os.environ.get("PHAST_SOLID_MECH_OUTPUT_DIR")
    if env_out:
        path = Path(env_out)
        path.mkdir(parents=True, exist_ok=True)
        return path
    out = cfg.get("output", {}).get("directory", "outputs")
    path = Path(out)
    if not path.is_absolute():
        path = Path(example_file).resolve().parent / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def git_commit(repo_root: Path | None = None) -> str:
    cwd = repo_root or Path(__file__).resolve().parents[2]
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    return proc.stdout.strip()


def write_manifest(
    out_dir: Path,
    *,
    example: str,
    command: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    files: list[str],
    started_at: float,
) -> None:
    command = os.environ.get("PHAST_SOLID_MECH_COMMAND", command)
    manifest = {
        "schema_version": 1,
        "example": example,
        "command": command,
        "runtime_seconds": round(time.perf_counter() - started_at, 6),
        "git_commit": git_commit(),
        "config": config,
        "metrics": metrics,
        "files": files,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


def write_run_metadata(
    out_dir: Path,
    *,
    example: str,
    config_path: str | Path | None,
    config: dict[str, Any],
) -> None:
    """Write sanitized metadata for a promoted solid-mechanics example."""
    metadata = {
        "schema_version": 1,
        "example": example,
        "config_path": str(config_path) if config_path is not None else None,
        "git_commit": git_commit(),
        "generated_at_unix": time.time(),
        "dtype": "float64",
        "device": "cpu",
        "config_summary": {
            "mesh": config.get("mesh", {}),
            "material": config.get("material", {}),
            "loading": config.get("loading", {}),
            "solver": config.get("solver", {}),
        },
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))


def write_run_lockfile(
    out_dir: Path,
    *,
    config: dict[str, Any],
    command: str,
) -> None:
    """Write the resolved public example config used for this run."""
    lockfile = {
        "schema_version": 1,
        "command": os.environ.get("PHAST_SOLID_MECH_COMMAND", command),
        "resolved_config": config,
        "trajectory_format": "zarr",
    }
    (out_dir / "run_lockfile.json").write_text(json.dumps(lockfile, indent=2))


def _zarr_array(group: Any, name: str, data: Any) -> None:
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    arr = np.asarray(data)
    if name in group:
        del group[name]
    group.create_dataset(name, data=arr, shape=arr.shape, dtype=arr.dtype)


def write_solid_zarr(
    out_dir: Path,
    *,
    mesh: Any,
    steps: list[dict[str, Any]],
    fields: dict[str, Any],
) -> Path:
    """Write a compact tutorial trajectory store for solid-mechanics examples."""
    import zarr

    path = out_dir / "training_data.zarr"
    if path.exists():
        shutil.rmtree(path)
    root = zarr.open_group(str(path), mode="w")
    root.attrs["format"] = "phast.solid_mechanics.trajectory.zarr"
    root.attrs["schema_version"] = 1
    sim = root.create_group("simulation_data")
    mesh_group = sim.create_group("mesh")
    _zarr_array(mesh_group, "node_coordinates", mesh.nodes.detach().cpu().numpy())
    _zarr_array(mesh_group, "element_connectivity", mesh.elements.detach().cpu().numpy())

    traj = sim.create_group("trajectory")
    traj.attrs["count"] = len(steps)
    if steps:
        step_ids = np.asarray([row.get("step", i) for i, row in enumerate(steps)], dtype=np.int64)
        _zarr_array(traj, "step", step_ids)
        for key in sorted(steps[0]):
            if key == "step":
                continue
            values = [row[key] for row in steps if key in row]
            if len(values) == len(steps):
                _zarr_array(traj, key, values)
    field_group = sim.create_group("fields")
    for name, value in fields.items():
        _zarr_array(field_group, name, value)
    (out_dir / "zarr_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "store": "training_data.zarr",
                "format": "phast.solid_mechanics.trajectory.zarr",
                "steps": len(steps),
                "fields": sorted(fields),
            },
            indent=2,
        )
    )
    return path


def save_response_animation(
    out_dir: Path,
    *,
    csv_rows: list[tuple[float, float]],
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str = "response_evolution.mp4",
) -> None:
    """Animate the response curve as load/time steps accumulate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, PillowWriter

    xs = np.asarray([row[0] for row in csv_rows], dtype=float)
    ys = np.asarray([row[1] for row in csv_rows], dtype=float)
    if xs.size == 1:
        xs = np.asarray([0.0, xs[0]])
        ys = np.asarray([0.0, ys[0]])

    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=150)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.set_xlim(float(xs.min()), float(xs.max()) if xs.max() > xs.min() else float(xs.max() + 1.0))
    ymin, ymax = float(ys.min()), float(ys.max())
    span = max(ymax - ymin, abs(ymax), 1.0e-12)
    ax.set_ylim(ymin - 0.08 * span, ymax + 0.12 * span)
    line, = ax.plot([], [], marker="o", lw=1.6, color="#1f77b4")

    writer = FFMpegWriter(fps=6)
    try:
        with writer.saving(fig, out_dir / filename, dpi=150):
            for i in range(1, len(xs) + 1):
                line.set_data(xs[:i], ys[:i])
                writer.grab_frame()
    except Exception:
        fallback = out_dir / filename.replace(".mp4", ".gif")
        gif_writer = PillowWriter(fps=6)
        with gif_writer.saving(fig, fallback, dpi=150):
            for i in range(1, len(xs) + 1):
                line.set_data(xs[:i], ys[:i])
                gif_writer.grab_frame()
        if fallback.name != filename:
            shutil.copyfile(fallback, out_dir / filename)
    finally:
        plt.close(fig)


def save_field_animation(
    out_dir: Path,
    *,
    mesh: Any,
    nodal_fields: list[Any],
    title: str,
    colorbar_label: str,
    filename: str = "field_evolution.mp4",
    cmap: str = "viridis",
    displacements: list[Any] | None = None,
    deformation_scale: float | None = None,
) -> None:
    """Animate a scalar field, optionally on the deformed bending shape."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.animation import FFMpegWriter, PillowWriter

    nodes = mesh.nodes.detach().cpu().numpy()
    elems = mesh.elements.detach().cpu().numpy()
    values = [np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v, dtype=float) for v in nodal_fields]
    disp_values = None
    if displacements is not None:
        disp_values = [
            np.asarray(d.detach().cpu().numpy() if hasattr(d, "detach") else d, dtype=float)
            for d in displacements
        ]
    if len(values) == 1:
        values = [np.zeros_like(values[0]), values[0]]
        if disp_values is not None:
            disp_values = [np.zeros_like(disp_values[0]), disp_values[0]]
    if disp_values is not None and len(disp_values) != len(values):
        raise ValueError("displacements must have the same length as nodal_fields")
    vmin = min(float(np.nanmin(v)) for v in values)
    vmax = max(float(np.nanmax(v)) for v in values)
    if vmax <= vmin:
        vmax = vmin + 1.0
    if disp_values is not None and deformation_scale is None:
        span = max(float(np.ptp(nodes[:, 0])), float(np.ptp(nodes[:, 1])), 1.0)
        umax = max(float(np.linalg.norm(d, axis=1).max()) for d in disp_values)
        deformation_scale = 0.12 * span / max(umax, 1.0e-30)
    elif deformation_scale is None:
        deformation_scale = 0.0
    reference_tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elems)
    if disp_values is not None:
        all_nodes = np.vstack([nodes + float(deformation_scale) * d for d in disp_values])
    else:
        all_nodes = nodes
    xmin, xmax = float(np.min(all_nodes[:, 0])), float(np.max(all_nodes[:, 0]))
    ymin, ymax = float(np.min(all_nodes[:, 1])), float(np.max(all_nodes[:, 1]))
    xpad = 0.08 * max(xmax - xmin, 1.0)
    ypad = 0.12 * max(ymax - ymin, 1.0)
    fig, ax = plt.subplots(figsize=(5.8, 3.2), dpi=150)
    scalar = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=vmin, vmax=vmax),
        cmap=cmap,
    )
    cbar = fig.colorbar(scalar, ax=ax)
    cbar.set_label(colorbar_label)

    def draw_frame(field: np.ndarray, disp: np.ndarray | None) -> None:
        ax.clear()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        if disp is None:
            frame_nodes = nodes
            tri = reference_tri
        else:
            frame_nodes = nodes + float(deformation_scale) * disp
            tri = mtri.Triangulation(frame_nodes[:, 0], frame_nodes[:, 1], elems)
            ax.triplot(reference_tri, color="0.82", lw=0.35, alpha=0.7)
        ax.tricontourf(tri, field, levels=24, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.triplot(tri, color="0.25", lw=0.25, alpha=0.45)
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)
        if disp is not None:
            ax.text(
                0.02,
                0.02,
                f"deformation x{float(deformation_scale):.2g}",
                transform=ax.transAxes,
                fontsize=8,
                color="0.25",
            )

    writer = FFMpegWriter(fps=6)
    try:
        with writer.saving(fig, out_dir / filename, dpi=150):
            for i, field in enumerate(values):
                draw_frame(field, None if disp_values is None else disp_values[i])
                writer.grab_frame()
    except Exception:
        fallback = out_dir / filename.replace(".mp4", ".gif")
        gif_writer = PillowWriter(fps=6)
        with gif_writer.saving(fig, fallback, dpi=150):
            for i, field in enumerate(values):
                draw_frame(field, None if disp_values is None else disp_values[i])
                gif_writer.grab_frame()
        if fallback.name != filename:
            shutil.copyfile(fallback, out_dir / filename)
    finally:
        plt.close(fig)


def copy_thumbnail(out_dir: Path, source_name: str = "response.png") -> None:
    src = out_dir / source_name
    if src.exists():
        shutil.copyfile(src, out_dir / "thumbnail.png")


def write_solid_setup_preview(
    out_dir: Path,
    *,
    title: str,
    mesh: Any,
    config: dict[str, Any],
) -> None:
    """Write the standard setup preview for public solid-mechanics examples."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    nodes = mesh.nodes.detach().cpu().numpy()
    elems = mesh.elements.detach().cpu().numpy()
    xmin, ymin = nodes[:, :2].min(axis=0)
    xmax, ymax = nodes[:, :2].max(axis=0)
    material = config.get("material", {})
    loading = config.get("loading", {})

    fig, (ax, info) = plt.subplots(
        1,
        2,
        figsize=(9.5, 4.2),
        dpi=160,
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )
    for tri in elems:
        pts = nodes[tri[:3], :2]
        closed = np.vstack([pts, pts[0]])
        ax.plot(closed[:, 0], closed[:, 1], color="0.72", lw=0.35)
    ax.plot([xmin, xmin], [ymin, ymax], color="#2255cc", lw=3.0)
    ax.arrow(
        xmax,
        0.5 * (ymin + ymax),
        0.0,
        -0.18 * max(ymax - ymin, 1.0e-12),
        width=0.002 * max(xmax - xmin, ymax - ymin),
        head_width=0.035 * max(xmax - xmin, 1.0e-12),
        head_length=0.06 * max(ymax - ymin, 1.0e-12),
        length_includes_head=True,
        color="#cc3d3d",
    )
    ax.text(xmax, ymin + 0.72 * (ymax - ymin), "load", color="#cc3d3d", ha="right", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Geometry and boundary setup")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    info.axis("off")
    lines = [
        title,
        "",
        f"nodes: {mesh.n_nodes:,}",
        f"elements: {mesh.n_elems:,}",
        f"domain: {xmax - xmin:g} x {ymax - ymin:g}",
        f"material: E={material.get('E')}, nu={material.get('nu')}",
        f"loading: {loading}",
        "left edge: fixed/clamped",
    ]
    info.text(
        0.02,
        0.95,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        transform=info.transAxes,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "initial_conditions.png")
    plt.close(fig)


def write_diagnostic_setup_preview(
    out_dir: Path,
    *,
    title: str,
    config: dict[str, Any],
) -> None:
    """Write a setup preview for numerical-method diagnostic examples."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=160)
    ax.axis("off")
    lines = [title, "", "This is a numerical-method diagnostic, not a mesh-level FEA solve."]
    for key, value in config.items():
        if key in {"schema_version", "output"}:
            continue
        lines.append(f"{key}: {value}")
    ax.text(
        0.04,
        0.94,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "initial_conditions.png")
    plt.close(fig)
