"""Deterministic cohesive contact-compression benchmark.

This validation benchmark validates the optional normal-contact penalty in the
solver-coupled ``CohesiveInterfaceOperator``. The interface is fully
prescribed in compression, so the normal contact traction, zero damage growth,
and contact tangent can be checked against closed-form expectations while
still exercising the quasi-static cohesive hook.
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
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from phast.cohesive_elements import (
    BilinearCohesiveLaw,
    CohesiveInterfaceOperator,
    insert_cohesive_layer,
)
from phast.fem_operators import FEMOperators
from phast.material import Material
from phast.mechanics_solver import QuasiStaticSolver
from phast.mesh import FEMMesh


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


def _two_strip_mesh() -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    nodes = np.array(
        [[0.0, 0.0], [1.0, 0.0],
         [0.0, 1.0], [1.0, 1.0],
         [0.0, -1.0], [1.0, -1.0]],
        dtype=np.float64,
    )
    elements = np.array(
        [[0, 1, 3], [0, 3, 2],
         [4, 1, 0], [4, 5, 1]],
        dtype=np.int64,
    )
    return nodes, elements, [(0, 1)]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_visual_manifest(output_dir: Path, image_paths: list[Path]) -> list[dict]:
    manifest = []
    for path in image_paths:
        with Image.open(path) as img:
            width, height = img.size
        manifest.append({
            "path": path.name,
            "width_px": int(width),
            "height_px": int(height),
            "review_dimension_passed": bool(width >= 800 and height >= 500),
        })
    (output_dir / "visual_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    return manifest


def _write_config(output_dir: Path, compressions: list[float],
                  law: BilinearCohesiveLaw) -> str:
    lines = [
        "case: cohesive_contact_compression_benchmark",
        "model: elastic_bulk_plus_zero_thickness_cohesive_contact_interface",
        "solver: QuasiStaticSolver(cohesive_operator=CohesiveInterfaceOperator)",
        "mesh: two_strip_triangular_patch_with_one_inserted_cohesive_edge",
        "boundary_condition: fully_prescribed_uniform_normal_compression",
        "device: cpu",
        "dtype: float64",
        "cohesive_law:",
        f"  k_n: {law.k_n}",
        f"  k_t: {law.k_t}",
        f"  sigma_max: {law.sigma_max}",
        f"  delta_c: {law.delta_c}",
        f"  contact_stiffness: {law.contact_stiffness}",
        "compressive_jumps:",
    ]
    lines.extend(f"  - {jump}" for jump in compressions)
    text = "\n".join(lines) + "\n"
    (output_dir / "config.yaml").write_text(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_standard_run_files(
    output_dir: Path,
    *,
    config_hash: str,
    elapsed_ms: float,
    artifact_paths: list[Path],
    n_steps: int,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "benchmark": "cohesive_contact_compression_benchmark",
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
            "artifacts": [path.name for path in artifact_paths],
        }, indent=2) + "\n")
    (output_dir / "run.log").write_text(
        "\n".join([
            f"{now} cohesive contact-compression benchmark started",
            f"{now} solver path: QuasiStaticSolver cohesive_operator hook",
            f"{now} completed {n_steps} deterministic load steps",
            f"{now} elapsed_ms={elapsed_ms:.3f}",
            "",
        ]))


def _build_solver(law: BilinearCohesiveLaw) -> tuple[
    QuasiStaticSolver, CohesiveInterfaceOperator, FEMMesh, np.ndarray
]:
    nodes, elements, interface_edges = _two_strip_mesh()
    new_nodes, new_elements, cohesives = insert_cohesive_layer(
        nodes, elements, interface_edges)
    mesh = FEMMesh.from_tensors(
        torch.as_tensor(new_nodes, dtype=torch.float64),
        torch.as_tensor(new_elements, dtype=torch.long),
        device="cpu",
        dtype=torch.float64,
    )
    fem = FEMOperators(mesh, Material(energy_split="isotropic"))
    op = CohesiveInterfaceOperator(
        cohesives, law, n_nodes=mesh.n_nodes, device="cpu",
        dtype=torch.float64)
    solver = QuasiStaticSolver(
        fem,
        cohesive_operator=op,
        backend="auto",
        tol=1.0e-11,
        max_iter=6,
        line_search=False,
    )
    return solver, op, mesh, new_nodes


def _tangent_fd_error(op: CohesiveInterfaceOperator, u: torch.Tensor) -> float:
    du = torch.zeros_like(u)
    du[0:2, 1] = -3.0e-4
    K = op.assemble_tangent(u, state=op.state)
    action = torch.sparse.mm(K, du.reshape(-1, 1)).reshape_as(u)
    h = 1.0e-6
    f_plus = op.internal_force(u + h * du, state=op.state)
    f_minus = op.internal_force(u - h * du, state=op.state)
    fd = (f_plus - f_minus) / (2.0 * h)
    return float((action - fd).abs().max().item())


def run_benchmark(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    law = BilinearCohesiveLaw(
        k_n=1_000.0,
        k_t=500.0,
        sigma_max=10.0,
        delta_c=0.050,
        contact_stiffness=2_500.0,
    )
    compressions = [0.0, -0.001, -0.003, -0.006, -0.010]
    config_hash = _write_config(output_dir, compressions, law)
    solver, op, mesh, nodes = _build_solver(law)

    d = torch.zeros(mesh.n_nodes, dtype=torch.float64)
    f_ext = torch.zeros((mesh.n_nodes, 2), dtype=torch.float64)
    bc_mask = torch.ones((mesh.n_nodes, 2), dtype=torch.bool)
    rows = []

    for step, jump in enumerate(compressions):
        bc_vals = torch.zeros((mesh.n_nodes, 2), dtype=torch.float64)
        bc_vals[0:2, 1] = float(jump)
        u, converged, n_iter = solver.solve(d, f_ext, bc_mask, bc_vals)
        fd_error = 0.0 if jump == 0.0 else _tangent_fd_error(op, u)
        f_coh = op.internal_force(u, state=op.state)
        if op._trial_state is not None:
            op.commit()
        measured_contact = float(f_coh[0:2, 1].sum().item())
        expected_contact = float(law.contact_stiffness * jump)
        damage = float(op.state.damage.max().item())
        rows.append({
            "step": step,
            "normal_jump": float(jump),
            "converged": bool(converged),
            "newton_iterations": int(n_iter),
            "normal_contact_traction": measured_contact,
            "expected_normal_contact_traction": expected_contact,
            "abs_contact_traction_error": abs(measured_contact - expected_contact),
            "damage": damage,
            "tangent_fd_error": fd_error,
            "residual_norm": float(solver.last_residual),
        })

    csv_path = output_dir / "cohesive_contact_compression_response.csv"
    _write_csv(csv_path, rows)

    jumps = np.asarray([row["normal_jump"] for row in rows])
    tractions = np.asarray([row["normal_contact_traction"] for row in rows])
    expected = np.asarray(
        [row["expected_normal_contact_traction"] for row in rows])
    damage = np.asarray([row["damage"] for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), constrained_layout=True)
    axes[0].plot(jumps, tractions, "o-", label="solver")
    axes[0].plot(jumps, expected, "k--", label="contact reference")
    axes[0].set_xlabel("normal displacement jump")
    axes[0].set_ylabel("normal contact traction")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(jumps, damage, "s-", color="tab:red")
    axes[1].set_xlabel("normal displacement jump")
    axes[1].set_ylabel("committed cohesive damage")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)
    response_png = output_dir / "cohesive_contact_compression_response.png"
    fig.savefig(response_png, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    ax.triplot(nodes[:, 0], nodes[:, 1], mesh.elements.cpu().numpy(), color="0.65")
    ax.plot([0.0, 1.0], [0.0, 0.0], color="tab:red", linewidth=3.0,
            label="cohesive interface")
    ax.quiver([0.3, 0.7], [0.10, 0.10], [0.0, 0.0], [-0.20, -0.20],
              angles="xy", scale_units="xy", scale=1.0, color="tab:blue",
              width=0.008, label="prescribed compression")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8, loc="upper right")
    mesh_png = output_dir / "cohesive_contact_compression_mesh_and_bc.png"
    fig.savefig(mesh_png, dpi=160)
    plt.close(fig)

    visual_manifest = _write_visual_manifest(output_dir, [response_png, mesh_png])
    elapsed_ms = 1_000.0 * (time.perf_counter() - start)
    artifacts = [
        output_dir / "summary.json",
        output_dir / "config.yaml",
        output_dir / "run_lockfile.json",
        output_dir / "run_metadata.json",
        output_dir / "run_manifest.json",
        output_dir / "run.log",
        csv_path,
        response_png,
        mesh_png,
        output_dir / "visual_manifest.json",
    ]
    _write_standard_run_files(
        output_dir,
        config_hash=config_hash,
        elapsed_ms=elapsed_ms,
        artifact_paths=artifacts,
        n_steps=len(rows),
    )

    max_contact_error = max(row["abs_contact_traction_error"] for row in rows)
    max_damage = max(row["damage"] for row in rows)
    max_tangent_fd_error = max(row["tangent_fd_error"] for row in rows)
    summary = {
        "example": "cohesive_contact_compression_benchmark",
        "capability": (
            "normal-contact penalty for zero-thickness cohesive interface "
            "through QuasiStaticSolver cohesive_operator hook"
        ),
        "n_nodes": int(mesh.n_nodes),
        "n_elements": int(mesh.elements.shape[0]),
        "n_cohesive_elements": int(len(op.cohesives)),
        "n_load_steps": int(len(rows)),
        "all_steps_converged": all(row["converged"] for row in rows),
        "contact_stiffness": float(law.contact_stiffness),
        "max_abs_contact_traction_error": float(max_contact_error),
        "max_damage": float(max_damage),
        "max_tangent_fd_error": float(max_tangent_fd_error),
        "validation_passed": bool(
            all(row["converged"] for row in rows)
            and max_contact_error < 1.0e-10
            and max_damage < 1.0e-14
            and max_tangent_fd_error < 1.0e-8
            and all(item["review_dimension_passed"] for item in visual_manifest)
        ),
        "csv": csv_path.name,
        "plots": [response_png.name, mesh_png.name],
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
        default=Path("outputs/plasticity_interface/cohesive_contact_compression"),
    )
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
