"""Structural DCB-style cohesive delamination benchmark.

This runner exercises the solver-coupled cohesive interface in a small
double-cantilever-beam-style specimen with a precrack, free internal bulk
degrees of freedom, and displacement-controlled Mode-I opening. It is a
structural validation smoke for the cohesive sparse Newton path, not an ASTM
D5528 data-reduction replacement.
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
class DCBGeometry:
    length: float = 6.0
    arm_height: float = 0.6
    nx: int = 16
    ny_per_arm: int = 2
    initial_crack_elements: int = 4

    @property
    def dx(self) -> float:
        return self.length / self.nx

    @property
    def initial_crack_length(self) -> float:
        return self.initial_crack_elements * self.dx


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


def _build_structural_mesh(
    geom: DCBGeometry,
) -> tuple[np.ndarray, np.ndarray, list[CohesiveElement], dict[str, torch.Tensor]]:
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

    cohesives: list[CohesiveElement] = []
    coords = np.asarray(nodes, dtype=np.float64)
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
        "top_interface": torch.tensor(top[0], dtype=torch.long),
        "bottom_interface": torch.tensor(bottom[-1], dtype=torch.long),
    }
    return coords, np.asarray(elements, dtype=np.int64), cohesives, node_sets


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_visual_manifest(output_dir: Path, image_paths: list[Path]) -> list[dict]:
    rows = []
    for path in image_paths:
        with Image.open(path) as img:
            width, height = img.size
        rows.append({
            "path": path.name,
            "width_px": int(width),
            "height_px": int(height),
            "review_dimension_passed": bool(width >= 800 and height >= 500),
        })
    (output_dir / "visual_manifest.json").write_text(
        json.dumps(rows, indent=2) + "\n")
    return rows


def _write_mesh_files(output_dir: Path, nodes: np.ndarray,
                      elements: np.ndarray, geom: DCBGeometry) -> None:
    geo = "\n".join([
        "// Structural DCB-style cohesive benchmark mesh parameters.",
        f"L = {geom.length};",
        f"h = {geom.arm_height};",
        f"nx = {geom.nx};",
        f"ny_per_arm = {geom.ny_per_arm};",
        f"a0 = {geom.initial_crack_length};",
        "// Mesh is generated in Python with separate coincident interface nodes.",
        "",
    ])
    (output_dir / "mesh.geo").write_text(geo)
    points = np.column_stack([nodes, np.zeros(nodes.shape[0])])
    tags = np.ones(elements.shape[0], dtype=np.int32)
    meshio.write(
        output_dir / "mesh.msh",
        meshio.Mesh(
            points=points,
            cells=[("triangle", elements)],
            cell_data={
                "gmsh:physical": [tags],
                "gmsh:geometrical": [tags],
            },
            field_data={"dcb_domain": np.array([1, 2], dtype=np.int32)},
        ),
        file_format="gmsh22",
    )


def _write_config(output_dir: Path, geom: DCBGeometry,
                  law: BilinearCohesiveLaw, openings: np.ndarray) -> str:
    text = "\n".join([
        "case: structural_dcb_cohesive_benchmark",
        "model: clamped_end_dcb_style_mode_I_cohesive_delamination",
        "solver: QuasiStaticSolver(cohesive_operator=CohesiveInterfaceOperator)",
        "capability_boundary: structural cohesive validation smoke, not ASTM D5528 data reduction",
        f"length: {geom.length}",
        f"arm_height: {geom.arm_height}",
        f"nx: {geom.nx}",
        f"ny_per_arm: {geom.ny_per_arm}",
        f"initial_crack_length: {geom.initial_crack_length}",
        "material:",
        "  E: 5000.0",
        "  nu: 0.3",
        "  energy_split: isotropic",
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
        "benchmark": "structural_dcb_cohesive_benchmark",
        "timestamp_utc": now,
        "git_sha": _git_sha(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "elapsed_ms": float(elapsed_ms),
        "max_rss_kib": _max_rss_kib(),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    (output_dir / "run_lockfile.json").write_text(
        json.dumps({
            "schema": "phast_run_lockfile_v1",
            "created_utc": now,
            "git_sha": metadata["git_sha"],
            "config_sha256": config_hash,
            "deterministic": True,
            "n_load_steps": int(n_steps),
        }, indent=2) + "\n")
    (output_dir / "run_manifest.json").write_text(
        json.dumps({
            "schema": "phast_run_manifest_v1",
            "benchmark": metadata["benchmark"],
            "artifacts": [path.name for path in artifacts],
        }, indent=2) + "\n")
    (output_dir / "run.log").write_text(
        "\n".join([
            f"{now} structural DCB-style cohesive benchmark started",
            f"{now} solver path: QuasiStaticSolver cohesive_operator hook",
            f"{now} completed {n_steps} displacement-controlled load steps",
            f"{now} elapsed_ms={elapsed_ms:.3f}",
            "",
        ]))


def _build_solver(
    geom: DCBGeometry,
    law: BilinearCohesiveLaw,
) -> tuple[QuasiStaticSolver, CohesiveInterfaceOperator, FEMMesh, np.ndarray]:
    nodes, elements, cohesives, node_sets = _build_structural_mesh(geom)
    mesh = FEMMesh.from_tensors(
        torch.as_tensor(nodes, dtype=torch.float64),
        torch.as_tensor(elements, dtype=torch.long),
        node_sets=node_sets,
        device="cpu",
        dtype=torch.float64,
    )
    material = Material(E=5_000.0, nu=0.3, energy_split="isotropic")
    fem = FEMOperators(mesh, material)
    op = CohesiveInterfaceOperator(
        cohesives, law, n_nodes=mesh.n_nodes, device="cpu",
        dtype=torch.float64)
    solver = QuasiStaticSolver(
        fem, cohesive_operator=op, backend="auto", tol=1.0e-9,
        max_iter=20, line_search=False)
    return solver, op, mesh, nodes


def _reaction_opening_force(
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


def _front_coordinate(op: CohesiveInterfaceOperator, nodes: np.ndarray,
                      damage: np.ndarray, threshold: float,
                      initial_front: float) -> float:
    """Return physical delamination-front coordinate including the precrack."""

    active = np.nonzero(damage > threshold)[0]
    if active.size == 0:
        return float(initial_front)
    damaged_front = max(nodes[op.cohesives[i].nodes_top[1], 0] for i in active)
    return float(max(initial_front, damaged_front))


def _write_plots(
    output_dir: Path,
    *,
    rows: list[dict],
    nodes: np.ndarray,
    elements: np.ndarray,
    mesh: FEMMesh,
    op: CohesiveInterfaceOperator,
    u: torch.Tensor,
    damage_profile: np.ndarray,
    geom: DCBGeometry,
) -> list[Path]:
    openings = np.asarray([row["opening"] for row in rows], dtype=np.float64)
    forces = np.asarray([row["opening_force"] for row in rows], dtype=np.float64)
    dissipated = np.asarray(
        [row["cohesive_dissipated_energy"] for row in rows], dtype=np.float64)
    bulk = np.asarray([row["bulk_elastic_energy"] for row in rows], dtype=np.float64)
    external = np.asarray([row["external_work"] for row in rows], dtype=np.float64)
    front = np.asarray([row["delamination_front_x"] for row in rows], dtype=np.float64)
    max_damage = np.asarray([row["damage_max"] for row in rows], dtype=np.float64)
    residual = np.asarray([row["residual_norm"] for row in rows], dtype=np.float64)
    seg_x = np.asarray([
        0.5 * (
            nodes[ce.nodes_top[0], 0] + nodes[ce.nodes_top[1], 0]
        )
        for ce in op.cohesives
    ], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), constrained_layout=True)
    axes[0].plot(openings, forces, "o-", label="opening reaction")
    axes[0].axvline(openings[int(np.argmax(forces))], color="0.4",
                    linestyle="--", linewidth=1.0, label="peak load")
    axes[0].set_xlabel("crack-mouth opening")
    axes[0].set_ylabel("opening force")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].semilogy(openings, np.maximum(residual, 1.0e-16), "s-")
    axes[1].set_xlabel("crack-mouth opening")
    axes[1].set_ylabel("free-DOF residual norm")
    axes[1].grid(True, which="both", alpha=0.3)
    load_png = output_dir / "structural_dcb_load_displacement.png"
    fig.savefig(load_png, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), constrained_layout=True)
    axes[0].plot(seg_x, damage_profile, "o-", color="tab:red")
    axes[0].axvline(geom.initial_crack_length, color="black",
                    linestyle="--", linewidth=1.0, label="initial crack")
    axes[0].set_xlabel("bonded-interface segment center x")
    axes[0].set_ylabel("final mean cohesive damage")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].plot(openings, front, "^-", label="front")
    axes[1].plot(openings, max_damage, "o-", label="max damage")
    axes[1].set_xlabel("crack-mouth opening")
    axes[1].set_ylabel("front x / damage")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    damage_png = output_dir / "structural_dcb_damage_front.png"
    fig.savefig(damage_png, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    ax.plot(openings, external, "o-", label="external work")
    ax.plot(openings, bulk, "s-", label="bulk elastic")
    ax.plot(openings, dissipated, "^-", label="cohesive dissipation")
    ax.plot(openings, bulk + dissipated, "k--", label="bulk + dissipation")
    ax.set_xlabel("crack-mouth opening")
    ax.set_ylabel("integrated energy")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    energy_png = output_dir / "structural_dcb_energy.png"
    fig.savefig(energy_png, dpi=160)
    plt.close(fig)

    disp = u.detach().cpu().numpy()
    deformed = nodes + 4.0 * disp
    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    ax.triplot(nodes[:, 0], nodes[:, 1], elements, color="0.82", linewidth=0.35)
    ax.triplot(
        deformed[:, 0], deformed[:, 1], mesh.elements.cpu().numpy(),
        color="tab:blue", linewidth=0.45)
    cmap = plt.cm.inferno
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    for ce, dmg in zip(op.cohesives, damage_profile):
        x0 = nodes[ce.nodes_top[0], 0]
        x1 = nodes[ce.nodes_top[1], 0]
        ax.plot([x0, x1], [0.0, 0.0], color=cmap(norm(float(dmg))),
                linewidth=2.5)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.78).set_label("mean cohesive damage")
    ax.axvline(geom.initial_crack_length, color="black",
               linestyle="--", linewidth=1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("undeformed mesh and amplified final deformation")
    mesh_png = output_dir / "structural_dcb_deformed_mesh.png"
    fig.savefig(mesh_png, dpi=160)
    plt.close(fig)

    return [load_png, damage_png, energy_png, mesh_png]


def run_benchmark(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    torch.manual_seed(0)
    np.random.seed(0)

    geom = DCBGeometry()
    law = BilinearCohesiveLaw(
        k_n=2_000.0,
        k_t=2_000.0,
        sigma_max=5.0,
        delta_c=0.080,
        contact_stiffness=2_000.0,
    )
    openings = np.linspace(0.0, 0.30, 11)
    config_hash = _write_config(output_dir, geom, law, openings)
    solver, op, mesh, nodes = _build_solver(geom, law)
    _write_mesh_files(output_dir, nodes, mesh.elements.cpu().numpy(), geom)

    d = torch.zeros(mesh.n_nodes, dtype=torch.float64)
    f_ext = torch.zeros((mesh.n_nodes, 2), dtype=torch.float64)
    top_load = mesh.node_sets["top_load"]
    bottom_load = mesh.node_sets["bottom_load"]
    right_clamp = mesh.node_sets["right_clamp"]
    u = None
    rows: list[dict] = []
    external_work = 0.0
    previous_opening = 0.0
    previous_force = 0.0
    damage_profile = np.zeros(len(op.cohesives), dtype=np.float64)

    for step, opening in enumerate(openings):
        bc_mask = torch.zeros((mesh.n_nodes, 2), dtype=torch.bool)
        bc_vals = torch.zeros((mesh.n_nodes, 2), dtype=torch.float64)
        bc_mask[right_clamp, :] = True
        bc_mask[top_load, 1] = True
        bc_vals[top_load, 1] = float(opening) * 0.5
        bc_mask[bottom_load, 1] = True
        bc_vals[bottom_load, 1] = -float(opening) * 0.5

        u, converged, n_iter = solver.solve(
            d, f_ext, bc_mask, bc_vals, u_init=u)
        force = _reaction_opening_force(
            solver.fem, op, u, d, top_load, bottom_load)
        if op._trial_state is not None:
            op.commit()
        external_work += 0.5 * (force + previous_force) * (
            float(opening) - previous_opening)
        previous_opening = float(opening)
        previous_force = force

        damage_q = op.state.damage.detach().cpu().numpy()
        damage_profile = damage_q.mean(axis=1)
        dissipated = float(op.integrated_dissipated_energy().item())
        bulk_elastic = float(solver.fem.compute_total_energy(u, d))
        front_x = _front_coordinate(
            op, nodes, damage_profile, threshold=0.05,
            initial_front=geom.initial_crack_length)
        active_segments = int(np.count_nonzero(damage_profile > 0.05))
        rows.append({
            "step": int(step),
            "opening": float(opening),
            "opening_force": float(force),
            "converged": bool(converged),
            "newton_iterations": int(n_iter),
            "residual_norm": float(solver.last_residual),
            "damage_max": float(damage_q.max()),
            "damage_mean": float(damage_q.mean()),
            "active_segments": active_segments,
            "delamination_front_x": float(front_x),
            "bulk_elastic_energy": bulk_elastic,
            "cohesive_dissipated_energy": dissipated,
            "external_work": float(external_work),
            "energy_balance_gap": float(
                abs(external_work - (bulk_elastic + dissipated))),
        })

    csv_path = output_dir / "structural_dcb_response.csv"
    _write_csv(csv_path, rows)
    image_paths = _write_plots(
        output_dir, rows=rows, nodes=nodes,
        elements=mesh.elements.cpu().numpy(), mesh=mesh, op=op, u=u,
        damage_profile=damage_profile, geom=geom)
    visual_manifest = _write_visual_manifest(output_dir, image_paths)
    elapsed_ms = 1_000.0 * (time.perf_counter() - start)
    artifacts = [
        output_dir / "summary.json",
        output_dir / "config.yaml",
        output_dir / "run_lockfile.json",
        output_dir / "run_metadata.json",
        output_dir / "run_manifest.json",
        output_dir / "run.log",
        output_dir / "mesh.geo",
        output_dir / "mesh.msh",
        csv_path,
        *image_paths,
        output_dir / "visual_manifest.json",
    ]
    _write_standard_files(
        output_dir, config_hash=config_hash, elapsed_ms=elapsed_ms,
        artifacts=artifacts, n_steps=len(rows))

    forces = np.asarray([row["opening_force"] for row in rows], dtype=np.float64)
    dissipated = np.asarray(
        [row["cohesive_dissipated_energy"] for row in rows], dtype=np.float64)
    front = np.asarray([row["delamination_front_x"] for row in rows], dtype=np.float64)
    peak_step = int(np.argmax(forces))
    initial_front = geom.initial_crack_length
    final_front = float(front[-1])
    max_energy_gap = float(max(row["energy_balance_gap"] for row in rows))
    final_external_work = float(rows[-1]["external_work"])
    max_energy_gap_fraction = max_energy_gap / max(final_external_work, 1.0e-12)
    energy_gap_tolerance = 0.15
    summary = {
        "example": "structural_dcb_cohesive_benchmark",
        "capability": (
            "DCB-style structural Mode-I cohesive delamination with free "
            "bulk DOFs through QuasiStaticSolver cohesive_operator hook"
        ),
        "capability_boundary": (
            "clamped-end structural validation smoke; not an ASTM D5528 "
            "fixture/data-reduction or analytical DCB calibration"
        ),
        "references": [
            "ASTM D5528 Mode-I DCB interlaminar fracture toughness standard",
            "Skec et al. 2018 analytical DCB with bilinear cohesive interface",
            "Krueger 2015 quasi-static delamination benchmark methodology",
        ],
        "n_nodes": int(mesh.n_nodes),
        "n_elements": int(mesh.elements.shape[0]),
        "n_cohesive_elements": int(len(op.cohesives)),
        "initial_crack_length": float(geom.initial_crack_length),
        "bonded_interface_length": float(geom.length - geom.initial_crack_length),
        "n_load_steps": int(len(rows)),
        "all_steps_converged": all(row["converged"] for row in rows),
        "max_residual_norm": float(max(row["residual_norm"] for row in rows)),
        "peak_opening_force": float(forces[peak_step]),
        "peak_force_step": peak_step,
        "final_opening_force": float(forces[-1]),
        "post_peak_softening": bool(peak_step < len(rows) - 1 and forces[-1] < forces[peak_step]),
        "final_damage_max": float(rows[-1]["damage_max"]),
        "final_damage_mean": float(rows[-1]["damage_mean"]),
        "final_active_segments": int(rows[-1]["active_segments"]),
        "final_delamination_front_x": final_front,
        "front_advanced": bool(final_front > initial_front),
        "cohesive_dissipation_monotone": bool(np.all(np.diff(dissipated) >= -1.0e-12)),
        "final_cohesive_dissipated_energy": float(dissipated[-1]),
        "fracture_energy_capacity": float(op.integrated_fracture_energy_capacity().item()),
        "final_bulk_elastic_energy": float(rows[-1]["bulk_elastic_energy"]),
        "final_external_work": final_external_work,
        "max_energy_balance_gap": max_energy_gap,
        "max_energy_balance_gap_fraction": float(max_energy_gap_fraction),
        "energy_balance_gap_tolerance": float(energy_gap_tolerance),
        "energy_balance_gap_pass": bool(max_energy_gap_fraction < energy_gap_tolerance),
        "energy_balance_note": (
            "Diagnostic displacement-control work balance using trapezoidal "
            "reaction work, bulk elastic energy, and cohesive dissipation."
        ),
        "validation_passed": bool(
            all(row["converged"] for row in rows)
            and max(row["residual_norm"] for row in rows) < 1.0e-8
            and peak_step < len(rows) - 1
            and forces[-1] < forces[peak_step]
            and rows[-1]["damage_max"] >= 1.0 - 1.0e-12
            and rows[-1]["active_segments"] >= 4
            and final_front > initial_front
            and np.all(np.diff(dissipated) >= -1.0e-12)
            and dissipated[-1] > 0.0
            and max_energy_gap_fraction < energy_gap_tolerance
            and all(item["review_dimension_passed"] for item in visual_manifest)
        ),
        "csv": csv_path.name,
        "plots": [path.name for path in image_paths],
        "visual_manifest": visual_manifest,
        "max_rss_kib": _max_rss_kib(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/plasticity_interface/structural_dcb_cohesive"),
    )
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
