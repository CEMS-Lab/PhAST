"""Coupled phase-field matrix damage plus cohesive-interface benchmark.

This customer-facing smoke example combines two failure mechanisms in one
staggered run:

* matrix cracking represented by an AT2 phase-field damage solve around a
  notched upper arm;
* interfacial debonding represented by zero-thickness cohesive elements on a
  bonded mid-plane interface.

The mechanics solve uses ``QuasiStaticSolver(cohesive_operator=...)`` so the
cohesive tangent is assembled into the Newton system.  After each mechanics
solve the phase-field damage problem is updated with
``PhaseFieldDamageSolver`` and the loop repeats until the load-step damage
increment is small.  This is a production-smoke validation of the coupled
workflow boundary, not a calibrated ASTM DCB or full PF-CZM release model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import resource
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import meshio
import numpy as np
import torch
from PIL import Image

from phast.cohesive_elements import (
    BilinearCohesiveLaw,
    CohesiveElement,
    CohesiveInterfaceOperator,
)
from phast.damage_solver import PhaseFieldDamageSolver
from phast.fem_operators import FEMOperators
from phast.material import Material
from phast.mechanics_solver import QuasiStaticSolver
from phast.mesh import FEMMesh


plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


@dataclass(frozen=True)
class CoupledGeometry:
    length: float = 4.8
    arm_height: float = 0.55
    nx: int = 16
    ny_per_arm: int = 2
    initial_crack_elements: int = 4
    matrix_notch_ix: int = 9

    @property
    def dx(self) -> float:
        return self.length / self.nx

    @property
    def initial_crack_length(self) -> float:
        return self.initial_crack_elements * self.dx

    @property
    def matrix_notch_x(self) -> float:
        return self.matrix_notch_ix * self.dx


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


def _build_mesh(
    geom: CoupledGeometry,
) -> tuple[np.ndarray, np.ndarray, list[CohesiveElement],
           dict[str, torch.Tensor], list[int], np.ndarray]:
    nodes: list[tuple[float, float]] = []
    top: list[list[int]] = []
    bottom: list[list[int]] = []

    for iy in range(geom.ny_per_arm + 1):
        row = []
        y = geom.arm_height * iy / geom.ny_per_arm
        for ix in range(geom.nx + 1):
            row.append(len(nodes))
            nodes.append((geom.dx * ix, y))
        top.append(row)

    for iy in range(geom.ny_per_arm + 1):
        row = []
        y = -geom.arm_height + geom.arm_height * iy / geom.ny_per_arm
        for ix in range(geom.nx + 1):
            row.append(len(nodes))
            nodes.append((geom.dx * ix, y))
        bottom.append(row)

    elements: list[list[int]] = []
    for grid in (top, bottom):
        for iy in range(geom.ny_per_arm):
            for ix in range(geom.nx):
                n00 = grid[iy][ix]
                n10 = grid[iy][ix + 1]
                n01 = grid[iy + 1][ix]
                n11 = grid[iy + 1][ix + 1]
                elements.append([n00, n10, n11])
                elements.append([n00, n11, n01])

    coords = np.asarray(nodes, dtype=np.float64)
    cohesives: list[CohesiveElement] = []
    for ix in range(geom.initial_crack_elements, geom.nx):
        n0_top = top[0][ix]
        n1_top = top[0][ix + 1]
        n0_bottom = bottom[-1][ix]
        n1_bottom = bottom[-1][ix + 1]
        edge = coords[n1_top] - coords[n0_top]
        length = float(np.linalg.norm(edge))
        tangent = edge / length
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        cohesives.append(
            CohesiveElement(
                nodes_top=(n0_top, n1_top),
                nodes_bottom=(n0_bottom, n1_bottom),
                normal=normal,
                tangent=tangent,
                length=length,
            )
        )

    notch_nodes = [
        top[geom.ny_per_arm][geom.matrix_notch_ix],
        top[geom.ny_per_arm - 1][geom.matrix_notch_ix],
        top[geom.ny_per_arm][geom.matrix_notch_ix + 1],
    ]
    notch_segment = np.array([
        [geom.matrix_notch_x, geom.arm_height],
        [geom.matrix_notch_x, 0.35 * geom.arm_height],
    ], dtype=np.float64)

    node_sets = {
        "right_clamp": torch.tensor(
            [top[iy][geom.nx] for iy in range(geom.ny_per_arm + 1)]
            + [bottom[iy][geom.nx] for iy in range(geom.ny_per_arm + 1)],
            dtype=torch.long,
        ),
        "top_load": torch.tensor(
            [top[iy][0] for iy in range(geom.ny_per_arm + 1)],
            dtype=torch.long,
        ),
        "bottom_load": torch.tensor(
            [bottom[iy][0] for iy in range(geom.ny_per_arm + 1)],
            dtype=torch.long,
        ),
        "matrix_notch": torch.tensor(notch_nodes, dtype=torch.long),
        "top_interface": torch.tensor(top[0], dtype=torch.long),
        "bottom_interface": torch.tensor(bottom[-1], dtype=torch.long),
    }
    return (
        coords,
        np.asarray(elements, dtype=np.int64),
        cohesives,
        node_sets,
        notch_nodes,
        notch_segment,
    )


def _segment_distance(points: np.ndarray, segment: np.ndarray) -> np.ndarray:
    start = segment[0]
    end = segment[1]
    ab = end - start
    denom = float(np.dot(ab, ab))
    if denom <= 1.0e-30:
        return np.linalg.norm(points - start, axis=1)
    t = np.clip(np.einsum("ij,j->i", points - start, ab) / denom, 0.0, 1.0)
    projection = start + t[:, None] * ab
    return np.linalg.norm(points - projection, axis=1)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_visual_manifest(output_dir: Path, paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg"}:
            with Image.open(path) as img:
                width, height = img.size
            rows.append({
                "artifact_type": "image",
                "path": path.name,
                "width_px": int(width),
                "height_px": int(height),
                "size_bytes": int(path.stat().st_size),
                "review_dimension_passed": bool(width >= 800 and height >= 500),
            })
        else:
            rows.append({
                "artifact_type": "animation",
                "path": path.name,
                "size_bytes": int(path.stat().st_size) if path.exists() else 0,
                "media_passed": bool(path.exists() and path.stat().st_size > 0),
            })
    (output_dir / "visual_manifest.json").write_text(
        json.dumps(rows, indent=2) + "\n")
    return rows


def _write_animation_gif(
    path: Path,
    frames: list[np.ndarray],
    *,
    fps: float = 3.0,
) -> None:
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=max(1, int(round(1000.0 / fps))),
        loop=0,
    )


def _write_config(output_dir: Path, geom: CoupledGeometry,
                  material: Material, law: BilinearCohesiveLaw,
                  openings: np.ndarray) -> str:
    text = "\n".join([
        "case: coupled_pf_matrix_damage_plus_cohesive_interface",
        "model: AT2_phase_field_matrix_damage + zero_thickness_cohesive_interface",
        "solver: staggered QuasiStaticSolver(cohesive_operator) + PhaseFieldDamageSolver",
        "capability_boundary: coupled validation smoke, not calibrated PF-CZM",
        f"length: {geom.length}",
        f"arm_height: {geom.arm_height}",
        f"nx: {geom.nx}",
        f"ny_per_arm: {geom.ny_per_arm}",
        f"initial_crack_length: {geom.initial_crack_length}",
        f"matrix_notch_x: {geom.matrix_notch_x}",
        "material:",
        f"  E: {material.E}",
        f"  nu: {material.nu}",
        f"  Gc: {material.Gc}",
        f"  l0: {material.l0}",
        f"  eta_residual: {material.eta_residual}",
        f"  energy_split: {material.energy_split}",
        f"  pf_model: {material.pf_model}",
        "cohesive_law:",
        f"  k_n: {law.k_n}",
        f"  k_t: {law.k_t}",
        f"  sigma_max: {law.sigma_max}",
        f"  delta_c: {law.delta_c}",
        f"  contact_stiffness: {law.contact_stiffness}",
        "load_openings:",
        *[f"  - {float(v)}" for v in openings],
        "",
    ])
    (output_dir / "config.yaml").write_text(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_mesh_files(output_dir: Path, nodes: np.ndarray,
                      elements: np.ndarray) -> dict[str, str]:
    geo = output_dir / "mesh.geo"
    geo.write_text(
        "\n".join([
            "// Coupled PF+cohesive validation smoke mesh.",
            "// mesh.msh is written directly by the Python runner.",
            "",
        ])
    )
    msh = output_dir / "mesh.msh"
    meshio.write(
        msh,
        meshio.Mesh(
            points=np.column_stack([
                nodes[:, 0], nodes[:, 1], np.zeros(nodes.shape[0])
            ]),
            cells=[("triangle", elements)],
        ),
    )
    return {"mesh.geo": str(geo), "mesh.msh": str(msh)}


def _write_standard_files(
    output_dir: Path,
    *,
    config_hash: str,
    elapsed_ms: float,
    artifacts: list[Path],
    n_steps: int,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "benchmark": "coupled_pf_cohesive_matrix_interface",
        "timestamp_utc": now,
        "git_sha": _git_sha(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "dtype": "float64",
        "elapsed_ms": float(elapsed_ms),
        "max_rss_kib": _max_rss_kib(),
    }
    lockfile = {
        "schema": "phast_run_lockfile_v1",
        "created_utc": now,
        "git_sha": metadata["git_sha"],
        "config_sha256": config_hash,
        "deterministic": True,
        "random_seed": 0,
        "n_load_steps": int(n_steps),
    }
    manifest = {
        "schema": "phast_run_manifest_v1",
        "benchmark": metadata["benchmark"],
        "artifacts": [path.name for path in artifacts],
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    (output_dir / "run_lockfile.json").write_text(
        json.dumps(lockfile, indent=2) + "\n")
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")


def _cohesive_front_x(op: CohesiveInterfaceOperator, nodes: np.ndarray,
                      damage_profile: np.ndarray, geom: CoupledGeometry) -> float:
    active = np.nonzero(damage_profile > 0.05)[0]
    if active.size == 0:
        return float(geom.initial_crack_length)
    front = max(nodes[op.cohesives[i].nodes_top[1], 0] for i in active)
    return float(max(geom.initial_crack_length, front))


def _opening_force(
    fem: FEMOperators,
    op: CohesiveInterfaceOperator,
    u: torch.Tensor,
    d: torch.Tensor,
    top_load: torch.Tensor,
    bottom_load: torch.Tensor,
) -> float:
    f_int = fem.internal_force(u, d) + op.internal_force(u, state=op.state)
    top = float(f_int[top_load, 1].sum().item())
    bottom = float(f_int[bottom_load, 1].sum().item())
    return 0.5 * (top - bottom)


def _plot_setup(output_dir: Path, nodes: np.ndarray, elements: np.ndarray,
                op: CohesiveInterfaceOperator, notch_nodes: list[int],
                geom: CoupledGeometry) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    ax.triplot(nodes[:, 0], nodes[:, 1], elements, color="0.72", linewidth=0.45)
    for ce in op.cohesives:
        x0 = nodes[ce.nodes_top[0], 0]
        x1 = nodes[ce.nodes_top[1], 0]
        ax.plot([x0, x1], [0.0, 0.0], color="tab:red", linewidth=2.0)
    ax.scatter(nodes[notch_nodes, 0], nodes[notch_nodes, 1],
               color="black", marker="x", s=50, label="PF matrix notch")
    ax.axvline(geom.initial_crack_length, color="0.25", linestyle="--",
               linewidth=1.0, label="initial interface crack")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Coupled PF matrix notch and cohesive interface")
    ax.legend(loc="best", fontsize=8)
    path = output_dir / "initial_conditions.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_step_damage_frame(
    nodes: np.ndarray,
    elements: np.ndarray,
    d: torch.Tensor,
    op: CohesiveInterfaceOperator,
    damage_profile: np.ndarray,
    opening: float,
) -> np.ndarray:
    d_elem = d.detach().cpu().numpy()[elements].mean(axis=1)
    fig, ax = plt.subplots(figsize=(7.0, 3.6), constrained_layout=True)
    tri = ax.tripcolor(
        nodes[:, 0], nodes[:, 1], elements, facecolors=d_elem,
        shading="flat", vmin=0.0, vmax=1.0, cmap="magma")
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    for ce, dmg in zip(op.cohesives, damage_profile):
        x0 = nodes[ce.nodes_top[0], 0]
        x1 = nodes[ce.nodes_top[1], 0]
        ax.plot([x0, x1], [0.0, 0.0], color=cmap(norm(float(dmg))),
                linewidth=2.8)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(f"opening = {opening:.3f}")
    fig.colorbar(tri, ax=ax, shrink=0.75).set_label("matrix damage d")
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


def _write_plots(
    output_dir: Path,
    *,
    rows: list[dict],
    energy_rows: list[dict],
    nodes: np.ndarray,
    elements: np.ndarray,
    mesh: FEMMesh,
    op: CohesiveInterfaceOperator,
    u: torch.Tensor,
    d: torch.Tensor,
    damage_profile: np.ndarray,
    geom: CoupledGeometry,
) -> list[Path]:
    openings = np.asarray([row["opening"] for row in rows], dtype=np.float64)
    forces = np.asarray([row["opening_force"] for row in rows], dtype=np.float64)
    pf_max = np.asarray([row["matrix_damage_max"] for row in rows], dtype=np.float64)
    pf_mean = np.asarray([row["matrix_damage_mean"] for row in rows], dtype=np.float64)
    cohesive_mean = np.asarray(
        [row["cohesive_damage_mean"] for row in rows], dtype=np.float64)
    front = np.asarray([row["cohesive_front_x"] for row in rows], dtype=np.float64)
    residual = np.asarray([row["mechanics_residual"] for row in rows],
                          dtype=np.float64)
    damage_delta = np.asarray([row["stagger_damage_delta"] for row in rows],
                              dtype=np.float64)
    seg_x = np.asarray([
        0.5 * (
            nodes[ce.nodes_top[0], 0] + nodes[ce.nodes_top[1], 0]
        )
        for ce in op.cohesives
    ], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax.plot(openings, forces, "o-", label="opening reaction")
    ax.set_xlabel("crack-mouth opening")
    ax.set_ylabel("opening force")
    ax.set_title("Load-displacement response")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    load_png = output_dir / "load_displacement.png"
    fig.savefig(load_png, dpi=160)
    plt.close(fig)

    d_elem = d.detach().cpu().numpy()[elements].mean(axis=1)
    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    tri = ax.tripcolor(
        nodes[:, 0], nodes[:, 1], elements, facecolors=d_elem,
        shading="flat", vmin=0.0, vmax=1.0, cmap="magma")
    for ce, dmg in zip(op.cohesives, damage_profile):
        x0 = nodes[ce.nodes_top[0], 0]
        x1 = nodes[ce.nodes_top[1], 0]
        ax.plot([x0, x1], [0.0, 0.0], color=plt.cm.viridis(float(dmg)),
                linewidth=2.8)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Final matrix PF damage and cohesive interface damage")
    fig.colorbar(tri, ax=ax, shrink=0.78).set_label("matrix damage d")
    damage_png = output_dir / "damage_final.png"
    fig.savefig(damage_png, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), constrained_layout=True)
    axes[0].plot(seg_x, damage_profile, "o-", color="tab:red")
    axes[0].axvline(geom.initial_crack_length, color="black", linestyle="--",
                    linewidth=1.0, label="initial interface crack")
    axes[0].set_xlabel("cohesive segment center x")
    axes[0].set_ylabel("final cohesive damage")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].plot(openings, front, "^-", label="cohesive front")
    axes[1].plot(openings, cohesive_mean, "o-", label="mean cohesive damage")
    axes[1].plot(openings, pf_mean, "s-", label="mean matrix damage")
    axes[1].set_xlabel("opening")
    axes[1].set_ylabel("front x / damage")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    front_png = output_dir / "cohesive_damage_front.png"
    fig.savefig(front_png, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    ax.plot(openings, [r["bulk_elastic_energy"] for r in energy_rows],
            "o-", label="bulk elastic")
    ax.plot(openings, [r["pf_fracture_energy"] for r in energy_rows],
            "s-", label="PF fracture")
    ax.plot(openings, [r["cohesive_dissipated_energy"] for r in energy_rows],
            "^-", label="cohesive dissipation")
    ax.plot(openings, [r["external_work"] for r in energy_rows],
            "k--", label="external work")
    ax.set_xlabel("opening")
    ax.set_ylabel("integrated energy")
    ax.set_title("Energy split")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    energy_png = output_dir / "energy_split.png"
    fig.savefig(energy_png, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax.semilogy(openings, np.maximum(residual, 1.0e-16), "o-",
                label="mechanics residual")
    ax.semilogy(openings, np.maximum(damage_delta, 1.0e-16), "s-",
                label="staggered damage increment")
    ax.set_xlabel("opening")
    ax.set_ylabel("convergence metric")
    ax.set_title("Staggered convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    convergence_png = output_dir / "convergence.png"
    fig.savefig(convergence_png, dpi=160)
    plt.close(fig)

    disp = u.detach().cpu().numpy()
    deformed = nodes + 3.5 * disp
    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    ax.triplot(nodes[:, 0], nodes[:, 1], elements, color="0.82", linewidth=0.35)
    ax.triplot(deformed[:, 0], deformed[:, 1], mesh.elements.cpu().numpy(),
               color="tab:blue", linewidth=0.45)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Undeformed mesh and amplified final deformation")
    mesh_png = output_dir / "mesh_deformed.png"
    fig.savefig(mesh_png, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax.plot(openings, pf_max, "o-", label="max matrix PF damage")
    ax.plot(openings, cohesive_mean, "s-", label="mean cohesive damage")
    ax.set_xlabel("opening")
    ax.set_ylabel("damage")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Matrix and interface damage histories")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    history_png = output_dir / "damage_history.png"
    fig.savefig(history_png, dpi=160)
    plt.close(fig)

    return [
        load_png,
        damage_png,
        front_png,
        energy_png,
        convergence_png,
        mesh_png,
        history_png,
    ]


def run_benchmark(output_dir: Path, *, n_steps: int = 9,
                  max_opening: float = 0.24,
                  max_stagger_iter: int = 20) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    torch.manual_seed(0)
    np.random.seed(0)

    geom = CoupledGeometry()
    nodes, elements, cohesives, node_sets, notch_nodes, notch_segment = (
        _build_mesh(geom)
    )
    material = Material(
        E=3_000.0,
        nu=0.30,
        Gc=0.10,
        l0=0.18,
        eta_residual=1.0e-6,
        energy_split="isotropic",
        pf_model="AT2",
    )
    law = BilinearCohesiveLaw(
        k_n=1_500.0,
        k_t=1_500.0,
        sigma_max=2.2,
        delta_c=0.075,
        contact_stiffness=1_500.0,
    )
    openings = np.linspace(0.0, max_opening, n_steps)
    config_hash = _write_config(output_dir, geom, material, law, openings)

    mesh = FEMMesh.from_tensors(
        torch.as_tensor(nodes, dtype=torch.float64),
        torch.as_tensor(elements, dtype=torch.long),
        node_sets=node_sets,
        device="cpu",
        dtype=torch.float64,
    )
    fem = FEMOperators(mesh, material)
    op = CohesiveInterfaceOperator(
        cohesives, law, n_nodes=mesh.n_nodes, device="cpu", dtype=torch.float64)
    mechanics = QuasiStaticSolver(
        fem,
        cohesive_operator=op,
        backend="auto",
        tol=1.0e-9,
        tol_rel=1.0e-8,
        max_iter=25,
        line_search=True,
    )
    damage_solver = PhaseFieldDamageSolver(
        fem,
        tol=1.0e-8,
        max_iter=300,
        use_multigrid=False,
        bounds_method="projected_cg",
    )

    centroids = nodes[elements].mean(axis=1)
    notch_distance = _segment_distance(centroids, notch_segment)
    matrix_notch_seed = torch.as_tensor(
        0.85 * material.Gc / material.l0
        * np.exp(-0.5 * (notch_distance / (0.75 * material.l0)) ** 2),
        dtype=torch.float64,
    )
    matrix_notch_seed = torch.where(
        torch.as_tensor(centroids[:, 1] > 0.0),
        matrix_notch_seed,
        torch.zeros_like(matrix_notch_seed),
    )

    d = torch.zeros(mesh.n_nodes, dtype=torch.float64)
    d[node_sets["matrix_notch"]] = 1.0
    pf_mask = torch.zeros(mesh.n_nodes, dtype=torch.bool)
    pf_mask[node_sets["matrix_notch"]] = True
    pf_vals = torch.zeros(mesh.n_nodes, dtype=torch.float64)
    pf_vals[node_sets["matrix_notch"]] = 1.0
    H_history = torch.zeros(mesh.n_elems, dtype=torch.float64)
    f_ext = torch.zeros((mesh.n_nodes, 2), dtype=torch.float64)
    top_load = node_sets["top_load"]
    bottom_load = node_sets["bottom_load"]
    right_clamp = node_sets["right_clamp"]
    u = None
    rows: list[dict] = []
    energy_rows: list[dict] = []
    timing_rows: list[dict] = []
    frames: list[np.ndarray] = []
    external_work = 0.0
    previous_opening = 0.0
    previous_force = 0.0
    damage_profile = np.zeros(len(op.cohesives), dtype=np.float64)

    setup_png = _plot_setup(output_dir, nodes, elements, op, notch_nodes, geom)
    mesh_artifacts = _write_mesh_files(output_dir, nodes, elements)

    for step, opening in enumerate(openings):
        step_start = time.perf_counter()
        bc_mask = torch.zeros((mesh.n_nodes, 2), dtype=torch.bool)
        bc_vals = torch.zeros((mesh.n_nodes, 2), dtype=torch.float64)
        bc_mask[right_clamp, :] = True
        bc_mask[top_load, 1] = True
        bc_vals[top_load, 1] = 0.5 * float(opening)
        bc_mask[bottom_load, 1] = True
        bc_vals[bottom_load, 1] = -0.5 * float(opening)

        converged = False
        n_iter = 0
        damage_delta = 0.0
        damage_residual_norm = 0.0
        for stagger_iter in range(1, max_stagger_iter + 1):
            u, converged, n_iter = mechanics.solve(
                d, f_ext, bc_mask, bc_vals, u_init=u)
            if not converged:
                raise RuntimeError(
                    f"Coupled mechanics solve failed at step {step}, "
                    f"stagger {stagger_iter}: residual={mechanics.last_residual}")
            strain = fem.compute_strain(u)
            psi = fem.compute_psi_plus(u, strain=strain)
            load_factor = 0.0 if max_opening == 0.0 else float(opening / max_opening)
            H_history = torch.maximum(
                H_history,
                psi + (load_factor ** 2) * matrix_notch_seed,
            )
            d_prev = d
            d = damage_solver.solve(
                H_history,
                d_prev,
                pf_dirichlet_mask=pf_mask,
                pf_dirichlet_values=pf_vals,
            )
            d[pf_mask] = pf_vals[pf_mask]
            damage_delta = float(torch.max(torch.abs(d - d_prev)).item())
            damage_residual = damage_solver.compute_residual(H_history, d)
            damage_residual[pf_mask] = 0.0
            damage_residual_norm = float(damage_residual.norm().item())
            if damage_delta < 5.0e-4:
                break

        force = _opening_force(fem, op, u, d, top_load, bottom_load)
        external_work += 0.5 * (force + previous_force) * (
            float(opening) - previous_opening)
        previous_opening = float(opening)
        previous_force = force

        damage_q = op.state.damage.detach().cpu().numpy()
        damage_profile = damage_q.mean(axis=1)
        cohesive_diss = float(op.integrated_dissipated_energy().item())
        energy_components = fem.compute_energy_components(u, d)
        fracture_surface, fracture_gradient = fem._fracture_energy_terms(d)
        front_x = _cohesive_front_x(op, nodes, damage_profile, geom)
        active_segments = int(np.count_nonzero(damage_profile > 0.05))
        wall_ms = 1000.0 * (time.perf_counter() - step_start)

        row = {
            "step": int(step),
            "opening": float(opening),
            "opening_force": float(force),
            "mechanics_converged": bool(converged),
            "mechanics_newton_iterations": int(n_iter),
            "mechanics_residual": float(mechanics.last_residual),
            "stagger_iterations": int(stagger_iter),
            "stagger_damage_delta": float(damage_delta),
            "damage_pcg_iterations": int(getattr(damage_solver, "last_iter", -1)),
            "damage_residual_norm": float(damage_residual_norm),
            "matrix_damage_min": float(d.min().item()),
            "matrix_damage_mean": float(d.mean().item()),
            "matrix_damage_max": float(d.max().item()),
            "cohesive_damage_mean": float(damage_q.mean()),
            "cohesive_damage_max": float(damage_q.max()),
            "active_cohesive_segments": active_segments,
            "cohesive_front_x": float(front_x),
            "wall_ms": float(wall_ms),
        }
        rows.append(row)
        energy_rows.append({
            "step": int(step),
            "opening": float(opening),
            "external_work": float(external_work),
            "bulk_elastic_energy": float(energy_components["elastic"]),
            "pf_fracture_surface_energy": float(fracture_surface),
            "pf_fracture_gradient_energy": float(fracture_gradient),
            "pf_fracture_energy": float(fracture_surface + fracture_gradient),
            "cohesive_dissipated_energy": float(cohesive_diss),
            "total_internal_energy": float(
                energy_components["elastic"]
                + fracture_surface
                + fracture_gradient
                + cohesive_diss
            ),
            "energy_balance_gap": float(abs(
                external_work
                - (
                    energy_components["elastic"]
                    + fracture_surface
                    + fracture_gradient
                    + cohesive_diss
                )
            )),
        })
        timing_rows.append({
            "step": int(step),
            "opening": float(opening),
            "wall_ms": float(wall_ms),
            "newton_iterations": int(n_iter),
            "damage_pcg_iterations": int(getattr(damage_solver, "last_iter", -1)),
            "stagger_iterations": int(stagger_iter),
        })
        frames.append(_plot_step_damage_frame(
            nodes, elements, d, op, damage_profile, float(opening)))

    for name in ("results.csv", "history.csv", "solver_telemetry.csv"):
        _write_csv(output_dir / name, rows)
    _write_csv(output_dir / "energy.csv", energy_rows)
    _write_csv(output_dir / "timing_per_step.csv", timing_rows)
    _write_csv(output_dir / "cohesive_front.csv", [
        {
            "step": row["step"],
            "opening": row["opening"],
            "cohesive_front_x": row["cohesive_front_x"],
            "active_cohesive_segments": row["active_cohesive_segments"],
            "cohesive_damage_mean": row["cohesive_damage_mean"],
            "cohesive_damage_max": row["cohesive_damage_max"],
        }
        for row in rows
    ])

    plot_paths = _write_plots(
        output_dir,
        rows=rows,
        energy_rows=energy_rows,
        nodes=nodes,
        elements=elements,
        mesh=mesh,
        op=op,
        u=u,
        d=d,
        damage_profile=damage_profile,
        geom=geom,
    )
    animation_path = output_dir / "damage_evolution.gif"
    _write_animation_gif(animation_path, frames)
    visual_manifest = _write_visual_manifest(
        output_dir, [setup_png, *plot_paths, animation_path])
    elapsed_ms = 1000.0 * (time.perf_counter() - t0)

    run_log = output_dir / "run.log"
    run_log.write_text(
        "\n".join([
            "example=coupled_pf_cohesive_matrix_interface",
            "solver=staggered QuasiStaticSolver(cohesive_operator)+PhaseFieldDamageSolver",
            f"n_steps={n_steps}",
            f"final_matrix_damage_max={rows[-1]['matrix_damage_max']:.12e}",
            f"final_cohesive_damage_max={rows[-1]['cohesive_damage_max']:.12e}",
            f"final_cohesive_front_x={rows[-1]['cohesive_front_x']:.12e}",
            f"max_damage_residual_norm={max(r['damage_residual_norm'] for r in rows):.12e}",
            f"elapsed_ms={elapsed_ms:.3f}",
            "",
        ])
    )
    artifacts = [
        output_dir / "summary.json",
        output_dir / "config.yaml",
        output_dir / "run_lockfile.json",
        output_dir / "run_metadata.json",
        output_dir / "run_manifest.json",
        run_log,
        output_dir / "mesh.geo",
        output_dir / "mesh.msh",
        output_dir / "results.csv",
        output_dir / "history.csv",
        output_dir / "solver_telemetry.csv",
        output_dir / "timing_per_step.csv",
        output_dir / "energy.csv",
        output_dir / "cohesive_front.csv",
        setup_png,
        *plot_paths,
        animation_path,
        output_dir / "visual_manifest.json",
    ]
    _write_standard_files(
        output_dir,
        config_hash=config_hash,
        elapsed_ms=elapsed_ms,
        artifacts=artifacts,
        n_steps=n_steps,
    )

    forces = np.asarray([row["opening_force"] for row in rows], dtype=np.float64)
    cohesive_diss = np.asarray(
        [row["cohesive_dissipated_energy"] for row in energy_rows],
        dtype=np.float64,
    )
    matrix_damage = np.asarray(
        [row["matrix_damage_max"] for row in rows], dtype=np.float64)
    max_energy_gap = float(max(row["energy_balance_gap"] for row in energy_rows))
    final_external_work = float(energy_rows[-1]["external_work"])
    max_energy_gap_fraction = max_energy_gap / max(abs(final_external_work), 1.0e-12)
    image_pass = all(
        item.get("review_dimension_passed", item.get("media_passed", False))
        for item in visual_manifest
    )
    summary = {
        "example": "coupled_pf_cohesive_matrix_interface",
        "capability": (
            "staggered AT2 phase-field matrix damage plus solver-coupled "
            "zero-thickness cohesive-interface delamination"
        ),
        "capability_boundary": (
            "coupled validation smoke with a notched matrix and cohesive "
            "interface; not a calibrated PF-CZM or ASTM DCB product workflow"
        ),
        "n_nodes": int(mesh.n_nodes),
        "n_elements": int(mesh.elements.shape[0]),
        "n_cohesive_elements": int(len(op.cohesives)),
        "n_load_steps": int(n_steps),
        "initial_crack_length": float(geom.initial_crack_length),
        "matrix_notch_x": float(geom.matrix_notch_x),
        "all_steps_converged": all(row["mechanics_converged"] for row in rows),
        "max_mechanics_residual": float(
            max(row["mechanics_residual"] for row in rows)),
        "max_damage_residual_norm": float(
            max(row["damage_residual_norm"] for row in rows)),
        "max_stagger_damage_delta": float(
            max(row["stagger_damage_delta"] for row in rows)),
        "final_matrix_damage_max": float(rows[-1]["matrix_damage_max"]),
        "final_matrix_damage_mean": float(rows[-1]["matrix_damage_mean"]),
        "matrix_damage_monotone": bool(np.all(np.diff(matrix_damage) >= -1.0e-12)),
        "final_cohesive_damage_max": float(rows[-1]["cohesive_damage_max"]),
        "final_cohesive_damage_mean": float(rows[-1]["cohesive_damage_mean"]),
        "final_active_cohesive_segments": int(rows[-1]["active_cohesive_segments"]),
        "final_cohesive_front_x": float(rows[-1]["cohesive_front_x"]),
        "front_advanced": bool(rows[-1]["cohesive_front_x"] > geom.initial_crack_length),
        "cohesive_dissipation_monotone": bool(np.all(np.diff(cohesive_diss) >= -1.0e-12)),
        "peak_opening_force": float(forces.max()),
        "final_opening_force": float(forces[-1]),
        "final_pf_fracture_energy": float(energy_rows[-1]["pf_fracture_energy"]),
        "final_cohesive_dissipated_energy": float(
            energy_rows[-1]["cohesive_dissipated_energy"]),
        "final_bulk_elastic_energy": float(energy_rows[-1]["bulk_elastic_energy"]),
        "final_external_work": final_external_work,
        "max_energy_balance_gap": max_energy_gap,
        "max_energy_balance_gap_fraction": float(max_energy_gap_fraction),
        "visual_validation_passed": bool(image_pass),
        "validation_passed": bool(
            all(row["mechanics_converged"] for row in rows)
            and max(row["mechanics_residual"] for row in rows) < 1.0e-7
            and max(row["damage_residual_norm"] for row in rows) < 5.0e-4
            and max(row["stagger_damage_delta"] for row in rows) < 5.0e-4
            and rows[-1]["matrix_damage_max"] >= 1.0 - 1.0e-12
            and rows[-1]["cohesive_damage_max"] > 0.80
            and rows[-1]["active_cohesive_segments"] >= 2
            and rows[-1]["cohesive_front_x"] > geom.initial_crack_length
            and np.all(np.diff(cohesive_diss) >= -1.0e-12)
            and energy_rows[-1]["pf_fracture_energy"] > 0.0
            and energy_rows[-1]["cohesive_dissipated_energy"] > 0.0
            and max_energy_gap_fraction < 0.50
            and image_pass
        ),
        "csv_outputs": [
            "results.csv",
            "history.csv",
            "solver_telemetry.csv",
            "timing_per_step.csv",
            "energy.csv",
            "cohesive_front.csv",
        ],
        "plots": [path.name for path in [setup_png, *plot_paths, animation_path]],
        "visual_manifest": visual_manifest,
        "mesh_artifacts": mesh_artifacts,
        "max_rss_kib": _max_rss_kib(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/plasticity_interface/coupled_pf_cohesive"),
    )
    parser.add_argument("--n-steps", type=int, default=9)
    parser.add_argument("--max-opening", type=float, default=0.24)
    args = parser.parse_args()
    print(json.dumps(
        run_benchmark(
            args.output_dir,
            n_steps=args.n_steps,
            max_opening=args.max_opening,
        ),
        indent=2,
    ))


if __name__ == "__main__":
    main()
