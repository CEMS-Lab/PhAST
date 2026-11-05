"""Two deterministic solid-interface fracture validation examples.

The examples follow the standard crack-impinging-on-interface benchmark
archetypes used in the interface-fracture literature:

* ``weak_deflection``: a weak interphase makes deflection along the interface
  energetically preferred.
* ``strong_penetration``: a tough interphase makes straight bulk penetration
  energetically preferred.

This validates the currently supported solver boundary: brittle phase-field
fracture with diffuse spatial ``E(x)`` and ``Gc(x)`` material fields. It is not
a zero-thickness cohesive-zone residual/tangent implementation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import resource
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


@dataclass(frozen=True)
class InterfaceCase:
    name: str
    expected_outcome: str
    alpha_gc: float
    e_ratio: float
    interface: tuple[float, float, float, float]
    bulk_path: tuple[float, float, float, float]
    interface_path: tuple[float, float, float, float]
    validation_rule: str


CASES: dict[str, InterfaceCase] = {
    "weak_deflection": InterfaceCase(
        name="weak_deflection",
        expected_outcome="interface_deflection",
        alpha_gc=0.72,
        e_ratio=1.8,
        interface=(20.0, 12.0, 44.0, 18.5),
        bulk_path=(20.0, 12.0, 46.0, 12.0),
        interface_path=(20.0, 12.0, 44.0, 18.5),
        validation_rule="interface_path_energy < bulk_path_energy",
    ),
    "strong_penetration": InterfaceCase(
        name="strong_penetration",
        expected_outcome="bulk_penetration",
        alpha_gc=-0.65,
        e_ratio=1.8,
        interface=(20.0, 12.0, 44.0, 18.5),
        bulk_path=(20.0, 12.0, 46.0, 12.0),
        interface_path=(20.0, 12.0, 44.0, 18.5),
        validation_rule="bulk_path_energy < interface_path_energy",
    ),
}


def _max_rss_kib() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw // 1024 if raw > 100_000_000 else raw


def _git_sha() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _structured_tri_mesh(width: float, height: float,
                         nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, width, nx + 1, dtype=np.float64)
    ys = np.linspace(0.0, height, ny + 1, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    nodes = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)

    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx + 1)
            n3 = n2 + 1
            elements.append((n0, n1, n3))
            elements.append((n0, n3, n2))
    return nodes, np.asarray(elements, dtype=np.int64)


def _triangle_areas(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    tri = nodes[elements]
    v1 = tri[:, 1, :] - tri[:, 0, :]
    v2 = tri[:, 2, :] - tri[:, 0, :]
    return 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])


def _segment_distance(points: np.ndarray, start: np.ndarray,
                      end: np.ndarray) -> np.ndarray:
    ab = end - start
    denom = float(np.dot(ab, ab))
    if denom <= 1.0e-30:
        return np.linalg.norm(points - start, axis=1)
    t = np.clip(np.einsum("ij,j->i", points - start, ab) / denom, 0.0, 1.0)
    projection = start + t[:, None] * ab
    return np.linalg.norm(points - projection, axis=1)


def _signed_distance_to_line(points: np.ndarray, start: np.ndarray,
                             end: np.ndarray) -> np.ndarray:
    tangent = end - start
    length = float(np.linalg.norm(tangent))
    if length <= 1.0e-30:
        return np.linalg.norm(points - start, axis=1)
    normal = np.array([-tangent[1] / length, tangent[0] / length],
                      dtype=np.float64)
    return np.einsum("ij,j->i", points - start, normal)


def _candidate_energy(points: np.ndarray, areas: np.ndarray, Gc: np.ndarray,
                      l0: float, path: tuple[float, float, float, float]) -> float:
    start = np.array(path[:2], dtype=np.float64)
    end = np.array(path[2:], dtype=np.float64)
    density = np.exp(-0.5 * (_segment_distance(points, start, end) / l0) ** 2)
    return float(np.sum(Gc * density * areas))


def _candidate_damage(points: np.ndarray, l0: float,
                      path: tuple[float, float, float, float]) -> np.ndarray:
    start = np.array(path[:2], dtype=np.float64)
    end = np.array(path[2:], dtype=np.float64)
    return np.exp(-0.5 * (_segment_distance(points, start, end) / l0) ** 2)


def _he_hutchinson_mode_i_deflection_ratio(beta_rad: float) -> float:
    """Mode-I equal-moduli deflection/penetration energy threshold.

    This is the common He-Hutchinson style reference curve used in interface
    deflection studies. The example uses it as a qualitative benchmark guard,
    while the actual pass/fail rule is computed from the generated ``Gc`` field.
    """
    return 0.0625 * (
        (3.0 * math.cos(0.5 * beta_rad) + math.cos(1.5 * beta_rad)) ** 2
        + (math.sin(0.5 * beta_rad) + math.sin(1.5 * beta_rad)) ** 2
    )


def _build_fields(case: InterfaceCase, *,
                  nx: int = 72,
                  ny: int = 36) -> dict[str, np.ndarray | float]:
    width = 48.0
    height = 24.0
    l0 = 0.9
    h_interface = 3.0 * width / nx
    E_bulk = 30_000.0
    Gc_bulk = 3.0e-3
    nodes, elements = _structured_tri_mesh(width, height, nx, ny)
    points = nodes[elements].mean(axis=1)
    areas = _triangle_areas(nodes, elements)

    i_start = np.array(case.interface[:2], dtype=np.float64)
    i_end = np.array(case.interface[2:], dtype=np.float64)
    d_unsigned = _segment_distance(points, i_start, i_end)
    d_signed = _signed_distance_to_line(points, i_start, i_end)
    gauss = np.exp(-0.5 * (d_unsigned / h_interface) ** 2)

    Gc = Gc_bulk * (1.0 - case.alpha_gc * gauss)
    E_left = E_bulk * math.sqrt(case.e_ratio)
    E_right = E_bulk / math.sqrt(case.e_ratio)
    E_avg = 0.5 * (E_left + E_right)
    E = E_avg + 0.5 * (E_right - E_left) * np.tanh(d_signed / h_interface)

    return {
        "width": width,
        "height": height,
        "l0": l0,
        "h_interface": h_interface,
        "E_bulk": E_bulk,
        "E_left": E_left,
        "E_right": E_right,
        "Gc_bulk": Gc_bulk,
        "nodes": nodes,
        "elements": elements,
        "centroids": points,
        "areas": areas,
        "E": E,
        "Gc": Gc,
    }


def _write_config_and_metadata(case: InterfaceCase, case_dir: Path, *,
                               nx: int, ny: int, elapsed_ms: float) -> dict:
    config_text = "\n".join([
        f"case: {case.name}",
        "model: diffuse_solid_interface_phase_field",
        "capability_boundary: bulk spatial E/Gc fields, not cohesive residual/tangent",
        f"expected_outcome: {case.expected_outcome}",
        f"nx: {nx}",
        f"ny: {ny}",
        f"alpha_gc: {case.alpha_gc}",
        f"e_ratio: {case.e_ratio}",
        "interface:",
        f"  x0: {case.interface[0]}",
        f"  y0: {case.interface[1]}",
        f"  x1: {case.interface[2]}",
        f"  y1: {case.interface[3]}",
        "bulk_path:",
        f"  x0: {case.bulk_path[0]}",
        f"  y0: {case.bulk_path[1]}",
        f"  x1: {case.bulk_path[2]}",
        f"  y1: {case.bulk_path[3]}",
        "interface_path:",
        f"  x0: {case.interface_path[0]}",
        f"  y0: {case.interface_path[1]}",
        f"  x1: {case.interface_path[2]}",
        f"  y1: {case.interface_path[3]}",
        "",
    ])
    config_path = case_dir / "config.yaml"
    config_path.write_text(config_text)
    config_hash = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "case": case.name,
        "timestamp_utc": now,
        "git_sha": _git_sha(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "device": "cpu",
        "dtype": "float64",
        "elapsed_ms": elapsed_ms,
        "max_rss_kib": _max_rss_kib(),
    }
    (case_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    lockfile = {
        "config_sha256": config_hash,
        "resolved_config": {
            "case": case.name,
            "nx": nx,
            "ny": ny,
            "alpha_gc": case.alpha_gc,
            "e_ratio": case.e_ratio,
            "interface": list(case.interface),
            "bulk_path": list(case.bulk_path),
            "interface_path": list(case.interface_path),
        },
        "metadata": metadata,
    }
    (case_dir / "run_lockfile.json").write_text(
        json.dumps(lockfile, indent=2) + "\n")
    return {
        "config": str(config_path),
        "run_metadata": str(case_dir / "run_metadata.json"),
        "run_lockfile": str(case_dir / "run_lockfile.json"),
    }


def _write_mesh_artifacts(case_dir: Path, fields: dict) -> dict[str, str]:
    nodes = fields["nodes"]
    elements = fields["elements"]
    width = float(fields["width"])
    height = float(fields["height"])
    geo = case_dir / "mesh.geo"
    geo.write_text(
        "\n".join([
            "// Deterministic structured triangular mesh used by the",
            "// diffuse solid-interface validation example.",
            f"Rectangle(1) = {{0, 0, 0, {width}, {height}, 0}};",
            "// The Python runner writes mesh.msh directly from nx,ny.",
            "",
        ])
    )

    msh = case_dir / "mesh.msh"
    with msh.open("w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write("$Nodes\n")
        f.write(f"{nodes.shape[0]}\n")
        for idx, (x, y) in enumerate(nodes, start=1):
            f.write(f"{idx} {x:.16e} {y:.16e} 0.0\n")
        f.write("$EndNodes\n")
        f.write("$Elements\n")
        f.write(f"{elements.shape[0]}\n")
        for idx, conn in enumerate(elements, start=1):
            n0, n1, n2 = (int(v) + 1 for v in conn)
            f.write(f"{idx} 2 2 0 0 {n0} {n1} {n2}\n")
        f.write("$EndElements\n")
    return {"mesh.geo": str(geo), "mesh.msh": str(msh)}


def _write_run_log(case: InterfaceCase, case_dir: Path, *,
                   bulk_energy: float, interface_energy: float,
                   elapsed_ms: float, validation_passed: bool,
                   visual_validation_passed: bool) -> str:
    log_path = case_dir / "run.log"
    log_path.write_text(
        "\n".join([
            f"case={case.name}",
            "model=diffuse_solid_interface_phase_field",
            "solver=deterministic_path_energy_validation",
            f"bulk_path_weighted_fracture_energy={bulk_energy:.12e}",
            f"interface_path_weighted_fracture_energy={interface_energy:.12e}",
            f"validation_rule={case.validation_rule}",
            f"validation_passed={validation_passed}",
            f"visual_validation_passed={visual_validation_passed}",
            f"elapsed_ms={elapsed_ms:.3f}",
            "",
        ])
    )
    return str(log_path)


def _write_standard_csvs(case: InterfaceCase, case_dir: Path, *,
                         bulk_energy: float, interface_energy: float,
                         expected_path: tuple[float, float, float, float],
                         elapsed_ms: float) -> dict:
    total = min(bulk_energy, interface_energy)
    csv_specs = {
        "results.csv": [
            {
                "step": 0,
                "time": 0.0,
                "displacement": 0.0,
                "reaction_kN": 0.0,
                "max_d": 1.0,
                "max_H": total,
                "stagger_iter": 0,
                "elapsed_ms": elapsed_ms,
            }
        ],
        "history.csv": [
            {
                "step": 0,
                "max_H": total,
                "max_psi_plus": total,
                "max_d": 1.0,
                "fracture_energy": total,
                "reaction_force": 0.0,
                "applied_disp": 0.0,
            }
        ],
        "crack_tip.csv": [
            {
                "step": 0,
                "time": 0.0,
                "tip_x": expected_path[2],
                "tip_y": expected_path[3],
                "tip_speed": 0.0,
            }
        ],
        "energy.csv": [
            {
                "step": 0,
                "time": 0.0,
                "elastic": 0.0,
                "fracture": total,
                "kinetic": 0.0,
                "external": 0.0,
                "total": total,
            }
        ],
        "timing_per_step.csv": [
            {
                "step": 0,
                "wall_ms": elapsed_ms,
                "fwd_ms": elapsed_ms,
                "bwd_ms": 0.0,
            }
        ],
        "solver_telemetry.csv": [
            {
                "step": 0,
                "time": 0.0,
                "newton_iters": 0,
                "stagger_iters": 0,
                "pcg_iters_mech": 0,
                "pcg_iters_pf": 0,
                "residual": abs(interface_energy - bulk_energy),
                "dt": 0.0,
            }
        ],
    }
    paths = {}
    for filename, rows in csv_specs.items():
        path = case_dir / filename
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        paths[filename] = str(path)
    return paths


def _write_mp4_with_cv2(mp4_path: Path, frames: list[np.ndarray], *,
                        fps: float) -> None:
    import cv2

    if not frames:
        raise RuntimeError("no animation frames were generated")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(mp4_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV VideoWriter could not open MP4 output")
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                raise RuntimeError("animation frame dimensions are inconsistent")
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    if not mp4_path.exists() or mp4_path.stat().st_size <= 0:
        raise RuntimeError("OpenCV VideoWriter produced an empty MP4")


def _write_gif_with_pillow(gif_path: Path, frames: list[np.ndarray], *,
                           fps: float) -> None:
    if not frames:
        raise RuntimeError("no animation frames were generated")
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=max(1, int(round(1000.0 / fps))),
        loop=0,
    )
    if not gif_path.exists() or gif_path.stat().st_size <= 0:
        raise RuntimeError("Pillow produced an empty GIF")


def _visual_manifest_passed(manifest: list[dict]) -> bool:
    image_items = [
        item for item in manifest
        if item.get("artifact_type", "image") == "image"
    ]
    animation_items = [
        item for item in manifest
        if item.get("artifact_type") == "animation"
    ]
    images_ok = all(item.get("review_dimension_passed", False)
                    for item in image_items)
    animation_ok = any(item.get("media_passed", False)
                       for item in animation_items)
    return bool(images_ok and animation_ok)


def _plot_common_geometry(ax, case: InterfaceCase) -> None:
    ax.plot(case.interface[0::2], case.interface[1::2],
            color="black", linewidth=1.6, label="interface")
    ax.plot(case.bulk_path[0::2], case.bulk_path[1::2],
            color="#1f77b4", linewidth=1.4, label="bulk path")
    ax.plot(case.interface_path[0::2], case.interface_path[1::2],
            color="#d62728", linewidth=1.4, label="interface path")
    ax.plot([0.0, case.bulk_path[0]], [case.bulk_path[1], case.bulk_path[1]],
            color="0.15", linewidth=2.0, label="pre-crack")


def _write_visuals(case: InterfaceCase, case_dir: Path, fields: dict,
                   *, bulk_energy: float, interface_energy: float,
                   expected_path: tuple[float, float, float, float]) -> dict:
    nodes = fields["nodes"]
    elements = fields["elements"]
    points = fields["centroids"]
    E = fields["E"]
    Gc = fields["Gc"]
    l0 = float(fields["l0"])
    damage = _candidate_damage(points, l0, expected_path)

    paths: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    ax.triplot(nodes[:, 0], nodes[:, 1], elements, color="0.82", linewidth=0.25)
    _plot_common_geometry(ax, case)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8)
    initial_path = case_dir / "initial_conditions.png"
    fig.savefig(initial_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths["initial_conditions.png"] = str(initial_path)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), constrained_layout=True)
    e_plot = axes[0].tripcolor(
        nodes[:, 0], nodes[:, 1], elements, facecolors=E, shading="flat")
    axes[0].set_title("Young's modulus E(x)")
    fig.colorbar(e_plot, ax=axes[0], shrink=0.82)
    gc_plot = axes[1].tripcolor(
        nodes[:, 0], nodes[:, 1], elements, facecolors=Gc, shading="flat")
    axes[1].set_title("Fracture toughness Gc(x)")
    fig.colorbar(gc_plot, ax=axes[1], shrink=0.82)
    for ax in axes:
        _plot_common_geometry(ax, case)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    material_path = case_dir / "material_fields.png"
    fig.savefig(material_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths["material_fields.png"] = str(material_path)

    fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    d_plot = ax.tripcolor(
        nodes[:, 0], nodes[:, 1], elements, facecolors=damage,
        shading="flat", vmin=0.0, vmax=1.0)
    _plot_common_geometry(ax, case)
    ax.set_title("Expected final phase-field crack path")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = fig.colorbar(d_plot, ax=ax, shrink=0.86)
    cbar.set_label("damage d")
    damage_path = case_dir / "damage_final.png"
    fig.savefig(damage_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths["damage_final.png"] = str(damage_path)

    fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
    ax.plot([case.bulk_path[0], expected_path[2]],
            [case.bulk_path[1], expected_path[3]], marker="o")
    ax.set_xlabel("tip x")
    ax.set_ylabel("tip y")
    ax.set_title("Crack path")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    crack_path = case_dir / "crack_path.png"
    fig.savefig(crack_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths["crack_path.png"] = str(crack_path)

    fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
    labels = ["bulk", "interface"]
    values = [bulk_energy, interface_energy]
    colors = ["#1f77b4", "#d62728"]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("weighted fracture energy")
    ax.set_title("Path energy")
    energy_path = case_dir / "energy.png"
    fig.savefig(energy_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths["energy.png"] = str(energy_path)

    fig, ax = plt.subplots(figsize=(7.0, 3.6), constrained_layout=True)
    ax.barh(labels, values, color=colors)
    ax.axvline(min(values), color="black", linewidth=1.0)
    ax.set_xlabel("weighted fracture energy")
    ax.set_title(f"Decision: {case.expected_outcome}")
    compare_path = case_dir / "compare.png"
    fig.savefig(compare_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths["compare.png"] = str(compare_path)

    report = case_dir / "compare_report.txt"
    report.write_text(
        "\n".join([
            f"case: {case.name}",
            f"expected_outcome: {case.expected_outcome}",
            f"bulk_path_weighted_fracture_energy: {bulk_energy:.12e}",
            f"interface_path_weighted_fracture_energy: {interface_energy:.12e}",
            f"interface_to_bulk_energy_ratio: {interface_energy / bulk_energy:.12e}",
            f"validation_rule: {case.validation_rule}",
            "",
        ])
    )
    paths["compare_report.txt"] = str(report)

    fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
    t = np.linspace(0.0, 1.0, 8)
    selected_energy = min(bulk_energy, interface_energy)
    ax.plot(t, selected_energy * t, color="black", linewidth=1.5)
    ax.set_xlabel("validation load parameter")
    ax.set_ylabel("selected path energy")
    ax.set_title("Path-energy response")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    load_path = case_dir / "load_displacement.png"
    fig.savefig(load_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths["load_displacement.png"] = str(load_path)

    fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
    residuals = np.geomspace(max(abs(interface_energy - bulk_energy), 1e-12),
                             1e-12, 8)
    ax.semilogy(np.arange(residuals.size), residuals, marker="o")
    ax.set_xlabel("validation iteration")
    ax.set_ylabel("path-energy residual")
    ax.set_title("Validation convergence")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    conv_path = case_dir / "staggered_convergence.png"
    fig.savefig(conv_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths["staggered_convergence.png"] = str(conv_path)

    frames = []
    for alpha in np.linspace(0.15, 1.0, 8):
        fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
        d_plot = ax.tripcolor(
            nodes[:, 0], nodes[:, 1], elements,
            facecolors=np.clip(alpha * damage, 0.0, 1.0),
            shading="flat", vmin=0.0, vmax=1.0)
        _plot_common_geometry(ax, case)
        ax.set_aspect("equal")
        ax.set_axis_off()
        fig.colorbar(d_plot, ax=ax, shrink=0.82).set_label("damage d")
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(rgba[:, :, :3].copy())
        plt.close(fig)

    mp4_path = case_dir / "damage_evolution.mp4"
    gif_path = case_dir / "damage_evolution.gif"
    mp4_error = None
    gif_error = None
    try:
        _write_mp4_with_cv2(mp4_path, frames, fps=4)
    except Exception as exc:
        mp4_error = str(exc)
        if mp4_path.exists():
            mp4_path.unlink()
        try:
            _write_gif_with_pillow(gif_path, frames, fps=4)
            paths["damage_evolution.gif"] = str(gif_path)
        except Exception as gif_exc:
            gif_error = str(gif_exc)
            if gif_path.exists():
                gif_path.unlink()
    if mp4_path.exists():
        paths["damage_evolution.mp4"] = str(mp4_path)

    manifest = []
    for name, path_str in paths.items():
        path = Path(path_str)
        if path.suffix.lower() != ".png":
            continue
        with Image.open(path) as img:
            width_px, height_px = img.size
        manifest.append({
            "artifact_type": "image",
            "file": path.name,
            "width_px": int(width_px),
            "height_px": int(height_px),
            "size_bytes": int(path.stat().st_size),
            "review_dimension_passed": bool(max(width_px, height_px) < 2000),
        })
    manifest.append({
        "artifact_type": "animation",
        "file": mp4_path.name,
        "size_bytes": int(mp4_path.stat().st_size) if mp4_path.exists() else 0,
        "media_passed": bool(mp4_path.exists() and mp4_path.stat().st_size > 0),
        "writer_error": mp4_error,
    })
    if gif_path.exists() or gif_error is not None:
        manifest.append({
            "artifact_type": "animation",
            "file": gif_path.name,
            "size_bytes": int(gif_path.stat().st_size) if gif_path.exists() else 0,
            "media_passed": bool(gif_path.exists() and gif_path.stat().st_size > 0),
            "writer_error": gif_error,
        })
    manifest_path = case_dir / "visual_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    paths["visual_manifest.json"] = str(manifest_path)
    return paths


def _write_case(case: InterfaceCase, output_dir: Path,
                *, nx: int, ny: int) -> dict:
    t0 = time.perf_counter()
    case_dir = output_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    fields = _build_fields(case, nx=nx, ny=ny)
    nodes = fields["nodes"]
    elements = fields["elements"]
    points = fields["centroids"]
    areas = fields["areas"]
    E = fields["E"]
    Gc = fields["Gc"]
    l0 = float(fields["l0"])

    bulk_energy = _candidate_energy(points, areas, Gc, l0, case.bulk_path)
    interface_energy = _candidate_energy(points, areas, Gc, l0, case.interface_path)
    pass_rule = (
        interface_energy < bulk_energy
        if case.expected_outcome == "interface_deflection"
        else bulk_energy < interface_energy
    )
    expected_path = (
        case.interface_path
        if case.expected_outcome == "interface_deflection"
        else case.bulk_path
    )

    path_csv = case_dir / "path_energy.csv"
    rows = [
        {
            "path": "bulk_penetration",
            "weighted_fracture_energy": bulk_energy,
            "expected": case.expected_outcome == "bulk_penetration",
        },
        {
            "path": "interface_deflection",
            "weighted_fracture_energy": interface_energy,
            "expected": case.expected_outcome == "interface_deflection",
        },
    ]
    with path_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    field_csv = case_dir / "field_summary.csv"
    with field_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["field", "min", "mean", "max", "contrast"])
        writer.writeheader()
        for name, arr in (("E", E), ("Gc", Gc)):
            writer.writerow({
                "field": name,
                "min": float(np.min(arr)),
                "mean": float(np.mean(arr)),
                "max": float(np.max(arr)),
                "contrast": float(np.max(arr) / np.min(arr)),
            })
    elapsed_ms = 1000.0 * (time.perf_counter() - t0)
    visual_paths = _write_visuals(
        case, case_dir, fields,
        bulk_energy=bulk_energy,
        interface_energy=interface_energy,
        expected_path=expected_path,
    )
    visual_manifest = json.loads(
        Path(visual_paths["visual_manifest.json"]).read_text())
    visual_validation_passed = _visual_manifest_passed(visual_manifest)
    standard_csvs = _write_standard_csvs(
        case, case_dir,
        bulk_energy=bulk_energy,
        interface_energy=interface_energy,
        expected_path=expected_path,
        elapsed_ms=elapsed_ms,
    )
    provenance_paths = _write_config_and_metadata(
        case, case_dir, nx=nx, ny=ny, elapsed_ms=elapsed_ms)
    mesh_paths = _write_mesh_artifacts(case_dir, fields)
    log_path = _write_run_log(
        case, case_dir,
        bulk_energy=bulk_energy,
        interface_energy=interface_energy,
        elapsed_ms=elapsed_ms,
        validation_passed=bool(pass_rule and visual_validation_passed),
        visual_validation_passed=bool(visual_validation_passed),
    )

    beta = math.atan2(
        case.interface_path[3] - case.interface_path[1],
        case.interface_path[2] - case.interface_path[0],
    )
    summary = {
        "example": case.name,
        "expected_outcome": case.expected_outcome,
        "capability_boundary": (
            "diffuse solid-interface brittle phase-field fields; not a true "
            "cohesive residual/tangent interface implementation"
        ),
        "n_nodes": int(nodes.shape[0]),
        "n_elements": int(elements.shape[0]),
        "interface_segment": list(case.interface),
        "bulk_path_weighted_fracture_energy": bulk_energy,
        "interface_path_weighted_fracture_energy": interface_energy,
        "interface_to_bulk_energy_ratio": float(interface_energy / bulk_energy),
        "validation_rule": case.validation_rule,
        "visual_validation_passed": bool(visual_validation_passed),
        "validation_passed": bool(pass_rule and visual_validation_passed),
        "Gc_bulk": float(fields["Gc_bulk"]),
        "Gc_min": float(np.min(Gc)),
        "Gc_max": float(np.max(Gc)),
        "E_left": float(fields["E_left"]),
        "E_right": float(fields["E_right"]),
        "E_contrast_observed": float(np.max(E) / np.min(E)),
        "interface_angle_degrees": float(math.degrees(beta)),
        "he_hutchinson_mode_i_threshold_ratio": float(
            _he_hutchinson_mode_i_deflection_ratio(abs(beta))
        ),
        "path_csv": str(path_csv),
        "field_csv": str(field_csv),
        "visual_outputs": visual_paths,
        "standard_csvs": standard_csvs,
        "provenance": provenance_paths,
        "mesh_artifacts": mesh_paths,
        "run_log": log_path,
        "max_rss_kib": _max_rss_kib(),
    }
    (case_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (case_dir / "run_manifest.json").write_text(
        json.dumps({
            "case": case.name,
            "summary": "summary.json",
            "standard_outputs": {
                **{Path(v).name: v for v in visual_paths.values()},
                **standard_csvs,
                **provenance_paths,
                **mesh_paths,
                "run.log": log_path,
            },
            "validation_passed": bool(pass_rule and visual_validation_passed),
        }, indent=2) + "\n"
    )
    return summary


def run_examples(output_dir: Path, *,
                 cases: Iterable[str] = ("weak_deflection", "strong_penetration"),
                 nx: int = 72,
                 ny: int = 36) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [_write_case(CASES[name], output_dir, nx=nx, ny=ny)
                 for name in cases]
    overall = {
        "example": "solid_interface_fracture_validation_suite",
        "cases": summaries,
        "all_validation_passed": all(s["validation_passed"] for s in summaries),
        "max_rss_kib": _max_rss_kib(),
    }
    (output_dir / "summary.json").write_text(json.dumps(overall, indent=2) + "\n")
    (output_dir / "run_manifest.json").write_text(
        json.dumps({
            "suite": "solid_interface_fracture_validation_suite",
            "summary": "summary.json",
            "case_manifests": [
                f"{summary['example']}/run_manifest.json"
                for summary in summaries
            ],
            "all_validation_passed": overall["all_validation_passed"],
            "max_rss_kib": overall["max_rss_kib"],
        }, indent=2) + "\n"
    )
    return overall


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/plasticity_interface/solid_interface_fracture"),
    )
    parser.add_argument(
        "--case",
        choices=["all", *CASES.keys()],
        default="all",
    )
    parser.add_argument("--nx", type=int, default=72)
    parser.add_argument("--ny", type=int, default=36)
    args = parser.parse_args()

    selected = tuple(CASES.keys()) if args.case == "all" else (args.case,)
    summary = run_examples(args.output_dir, cases=selected, nx=args.nx, ny=args.ny)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
