"""Deterministic mixed-mode cohesive benchmark.

This customer-facing validation benchmark exercises the solver-coupled
``CohesiveInterfaceOperator`` in combined mode-I/mode-II opening. The
zero-thickness interface is fully prescribed, so the resultant normal and shear
tractions can be checked against the bilinear mixed-mode law while the solver
still assembles residuals, tangents, and cohesive history through the
``QuasiStaticSolver`` cohesive hook.
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


def _expected_response(
    normal_jump: float,
    shear_jump: float,
    *,
    law: BilinearCohesiveLaw,
    max_delta_eff: float,
) -> tuple[float, float, float, float]:
    delta_n = max(normal_jump, 0.0)
    delta_eff = float(np.sqrt(delta_n * delta_n + shear_jump * shear_jump))
    max_eff = max(max_delta_eff, delta_eff)
    if max_eff <= law.delta_0:
        damage = 0.0
    else:
        damage = float(np.clip(
            law.delta_c
            * (max_eff - law.delta_0)
            / (max_eff * (law.delta_c - law.delta_0)),
            0.0,
            1.0,
        ))
    t_n = 0.0 if normal_jump < 0.0 else (1.0 - damage) * law.k_n * normal_jump
    t_t = (1.0 - damage) * law.k_t * shear_jump
    return t_n, t_t, damage, max_eff


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


def _write_config(output_dir: Path, jumps: list[tuple[float, float]],
                  law: BilinearCohesiveLaw) -> str:
    lines = [
        "case: cohesive_mixed_mode_benchmark",
        "model: elastic_bulk_plus_zero_thickness_mixed_mode_cohesive_interface",
        "solver: QuasiStaticSolver(cohesive_operator=CohesiveInterfaceOperator)",
        "mesh: two_strip_triangular_patch_with_one_inserted_cohesive_edge",
        "boundary_condition: fully_prescribed_uniform_normal_and_shear_jump",
        "device: cpu",
        "dtype: float64",
        "cohesive_law:",
        f"  k_n: {law.k_n}",
        f"  k_t: {law.k_t}",
        f"  sigma_max: {law.sigma_max}",
        f"  delta_c: {law.delta_c}",
        "load_jumps:",
    ]
    lines.extend(f"  - normal: {dn}\n    shear: {dt}" for dn, dt in jumps)
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
        "benchmark": "cohesive_mixed_mode_benchmark",
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
            f"{now} cohesive mixed-mode benchmark started",
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
    if float(u.norm().item()) == 0.0:
        return 0.0
    du = torch.zeros_like(u)
    du[0:2, :] = u[0:2, :]
    K = op.assemble_tangent(u, state=op.state)
    action = torch.sparse.mm(K, du.reshape(-1, 1)).reshape_as(u)
    h = 1.0e-6
    f_base = op.internal_force(u, state=op.state)
    f_plus = op.internal_force(u + h * du, state=op.state)
    fd = (f_plus - f_base) / h
    return float((action - fd).abs().max().item())


def run_benchmark(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    torch.manual_seed(0)
    np.random.seed(0)

    law = BilinearCohesiveLaw(
        k_n=1_000.0,
        k_t=500.0,
        sigma_max=10.0,
        delta_c=0.050,
    )
    jumps = [
        (0.0, 0.0),
        (0.004, 0.002),
        (0.010, 0.006),
        (0.020, 0.012),
        (0.035, 0.020),
        (0.050, 0.030),
    ]
    config_hash = _write_config(output_dir, jumps, law)
    solver, op, mesh, nodes = _build_solver(law)
    d = torch.zeros(mesh.n_nodes, dtype=torch.float64)
    f_ext = torch.zeros((mesh.n_nodes, 2), dtype=torch.float64)
    bc_mask = torch.ones((mesh.n_nodes, 2), dtype=torch.bool)
    rows = []
    max_delta_eff = 0.0

    for step, (normal_jump, shear_jump) in enumerate(jumps):
        bc_vals = torch.zeros((mesh.n_nodes, 2), dtype=torch.float64)
        bc_vals[0:2, 0] = float(shear_jump)
        bc_vals[0:2, 1] = float(normal_jump)
        u, converged, n_iter = solver.solve(d, f_ext, bc_mask, bc_vals)
        tangent_fd_error = _tangent_fd_error(op, u)
        f_coh = op.internal_force(u, state=op.state)
        if op._trial_state is not None:
            op.commit()
        exp_n, exp_t, exp_damage, max_delta_eff = _expected_response(
            normal_jump, shear_jump, law=law, max_delta_eff=max_delta_eff)
        measured_n = float(f_coh[0:2, 1].sum().item())
        measured_t = float(f_coh[0:2, 0].sum().item())
        measured_damage = float(op.state.damage.max().item())
        rows.append({
            "step": step,
            "normal_jump": float(normal_jump),
            "shear_jump": float(shear_jump),
            "delta_eff_history": float(max_delta_eff),
            "converged": bool(converged),
            "newton_iterations": int(n_iter),
            "normal_traction": measured_n,
            "expected_normal_traction": exp_n,
            "abs_normal_traction_error": abs(measured_n - exp_n),
            "shear_traction": measured_t,
            "expected_shear_traction": exp_t,
            "abs_shear_traction_error": abs(measured_t - exp_t),
            "damage": measured_damage,
            "expected_damage": exp_damage,
            "abs_damage_error": abs(measured_damage - exp_damage),
            "tangent_fd_error": tangent_fd_error,
            "residual_norm": float(solver.last_residual),
        })

    csv_path = output_dir / "cohesive_mixed_mode_response.csv"
    _write_csv(csv_path, rows)

    normal_jumps = np.asarray([row["normal_jump"] for row in rows])
    shear_jumps = np.asarray([row["shear_jump"] for row in rows])
    normal_t = np.asarray([row["normal_traction"] for row in rows])
    normal_ref = np.asarray([row["expected_normal_traction"] for row in rows])
    shear_t = np.asarray([row["shear_traction"] for row in rows])
    shear_ref = np.asarray([row["expected_shear_traction"] for row in rows])
    damage = np.asarray([row["damage"] for row in rows])

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), constrained_layout=True)
    axes[0].plot(normal_jumps, normal_t, "o-", label="solver")
    axes[0].plot(normal_jumps, normal_ref, "k--", label="reference")
    axes[0].set_xlabel("normal jump")
    axes[0].set_ylabel("normal traction")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(shear_jumps, shear_t, "s-", label="solver")
    axes[1].plot(shear_jumps, shear_ref, "k--", label="reference")
    axes[1].set_xlabel("shear jump")
    axes[1].set_ylabel("shear traction")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(np.sqrt(normal_jumps ** 2 + shear_jumps ** 2), damage, "^-")
    axes[2].set_xlabel("effective jump")
    axes[2].set_ylabel("committed damage")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(True, alpha=0.3)
    response_png = output_dir / "cohesive_mixed_mode_response.png"
    fig.savefig(response_png, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    ax.triplot(nodes[:, 0], nodes[:, 1], mesh.elements.cpu().numpy(), color="0.65")
    ax.plot([0.0, 1.0], [0.0, 0.0], color="tab:red", linewidth=3.0,
            label="cohesive interface")
    ax.quiver([0.35, 0.70], [0.08, 0.08], [0.16, 0.16], [0.20, 0.20],
              angles="xy", scale_units="xy", scale=1.0, color="tab:blue",
              width=0.008, label="mixed-mode jump")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8, loc="upper right")
    mesh_png = output_dir / "cohesive_mixed_mode_mesh_and_bc.png"
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

    max_normal_error = max(row["abs_normal_traction_error"] for row in rows)
    max_shear_error = max(row["abs_shear_traction_error"] for row in rows)
    max_damage_error = max(row["abs_damage_error"] for row in rows)
    max_tangent_fd_error = max(row["tangent_fd_error"] for row in rows)
    summary = {
        "example": "cohesive_mixed_mode_benchmark",
        "capability": (
            "mixed-mode zero-thickness cohesive interface residual/tangent "
            "through QuasiStaticSolver cohesive_operator hook"
        ),
        "n_nodes": int(mesh.n_nodes),
        "n_elements": int(mesh.elements.shape[0]),
        "n_cohesive_elements": int(len(op.cohesives)),
        "n_load_steps": int(len(rows)),
        "all_steps_converged": all(row["converged"] for row in rows),
        "final_damage": float(rows[-1]["damage"]),
        "max_abs_normal_traction_error": float(max_normal_error),
        "max_abs_shear_traction_error": float(max_shear_error),
        "max_abs_damage_error": float(max_damage_error),
        "max_tangent_fd_error": float(max_tangent_fd_error),
        "validation_passed": bool(
            all(row["converged"] for row in rows)
            and max_normal_error < 1.0e-10
            and max_shear_error < 1.0e-10
            and max_damage_error < 1.0e-12
            and max_tangent_fd_error < 1.0e-4
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
        default=Path("outputs/plasticity_interface/cohesive_mixed_mode"),
    )
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
