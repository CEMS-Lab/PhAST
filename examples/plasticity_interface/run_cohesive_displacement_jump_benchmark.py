"""Deterministic cohesive displacement-jump benchmark.

This production-smoke benchmark exercises the solver-coupled
``CohesiveInterfaceOperator`` through the ``QuasiStaticSolver`` cohesive hook.
The case is intentionally small: a two-triangle strip above and below a
zero-thickness interface is opened in prescribed mode I. The displacement jump
is uniform, so the numerical response can be checked against the bilinear
traction-separation law exactly while still committing cohesive history through
the quasi-static solver path.
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


def _expected_bilinear_response(
    opening: float,
    *,
    law: BilinearCohesiveLaw,
) -> tuple[float, float, float]:
    delta_0 = law.delta_0
    if opening <= delta_0:
        damage = 0.0
    else:
        damage = float(np.clip(
            law.delta_c
            * (opening - delta_0)
            / (opening * (law.delta_c - delta_0)),
            0.0,
            1.0,
        ))
    traction = (1.0 - damage) * law.k_n * opening
    opening_t = torch.as_tensor(opening, dtype=torch.float64)
    damage_t = torch.as_tensor(damage, dtype=torch.float64)
    dissipated = float(
        law.dissipated_energy_density(opening_t, damage_t).item())
    return traction, damage, dissipated


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


def _write_config(output_dir: Path, *, openings: list[float],
                  law: BilinearCohesiveLaw) -> str:
    lines = [
        "case: cohesive_displacement_jump_mode_i",
        "model: elastic_bulk_plus_zero_thickness_cohesive_interface",
        "solver: QuasiStaticSolver(cohesive_operator=CohesiveInterfaceOperator)",
        "mesh: two_strip_triangular_patch_with_one_inserted_cohesive_edge",
        "boundary_condition: fully_prescribed_uniform_mode_i_opening",
        "device: cpu",
        "dtype: float64",
        "cohesive_law:",
        f"  k_n: {law.k_n}",
        f"  k_t: {law.k_t}",
        f"  sigma_max: {law.sigma_max}",
        f"  delta_c: {law.delta_c}",
        "load_openings:",
    ]
    lines.extend(f"  - {opening}" for opening in openings)
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
        "benchmark": "cohesive_displacement_jump_mode_i",
        "timestamp_utc": now,
        "git_sha": _git_sha(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "elapsed_ms": float(elapsed_ms),
        "max_rss_kib": _max_rss_kib(),
    }
    lockfile = {
        "schema": "phast_run_lockfile_v1",
        "created_utc": now,
        "git_sha": metadata["git_sha"],
        "config_sha256": config_hash,
        "deterministic": True,
        "random_seed": None,
        "n_load_steps": int(n_steps),
    }
    manifest = {
        "schema": "phast_run_manifest_v1",
        "benchmark": metadata["benchmark"],
        "artifacts": [path.name for path in artifact_paths],
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    (output_dir / "run_lockfile.json").write_text(
        json.dumps(lockfile, indent=2) + "\n")
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    (output_dir / "run.log").write_text(
        "\n".join([
            f"{now} cohesive displacement-jump benchmark started",
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
    openings = [0.0, 0.004, 0.010, 0.020, 0.035, 0.050]
    config_hash = _write_config(output_dir, openings=openings, law=law)
    solver, op, mesh, nodes = _build_solver(law)

    d = torch.zeros(mesh.n_nodes, dtype=torch.float64)
    f_ext = torch.zeros((mesh.n_nodes, 2), dtype=torch.float64)
    bc_mask = torch.ones((mesh.n_nodes, 2), dtype=torch.bool)
    rows = []

    for step, opening in enumerate(openings):
        bc_vals = torch.zeros((mesh.n_nodes, 2), dtype=torch.float64)
        # Original interface nodes are the top side. Duplicates created by
        # insert_cohesive_layer are the bottom side, so this gives a uniform
        # positive normal jump of exactly ``opening``.
        bc_vals[0:2, 1] = float(opening)
        u, converged, n_iter = solver.solve(d, f_ext, bc_mask, bc_vals)
        f_coh = op.internal_force(u)
        if op._trial_state is not None:
            op.commit()
        expected_traction, expected_damage, expected_dissipated = (
            _expected_bilinear_response(
                opening,
                law=law,
            )
        )
        measured_traction = float(f_coh[0:2, 1].sum().item())
        measured_damage = float(op.state.damage.max().item())
        measured_dissipated = float(op.integrated_dissipated_energy().item())
        rows.append({
            "step": step,
            "opening": float(opening),
            "converged": bool(converged),
            "newton_iterations": int(n_iter),
            "solver_backend": solver.last_backend or "not_needed_prescribed",
            "normal_traction": measured_traction,
            "expected_normal_traction": expected_traction,
            "abs_traction_error": abs(measured_traction - expected_traction),
            "damage": measured_damage,
            "expected_damage": expected_damage,
            "abs_damage_error": abs(measured_damage - expected_damage),
            "dissipated_energy": measured_dissipated,
            "expected_dissipated_energy": expected_dissipated,
            "abs_dissipated_energy_error": abs(
                measured_dissipated - expected_dissipated),
            "residual_norm": float(solver.last_residual),
        })

    evidence_csv = output_dir / "cohesive_response.csv"
    _write_csv(evidence_csv, rows)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2), constrained_layout=True)
    openings_np = np.asarray([row["opening"] for row in rows], dtype=np.float64)
    tractions = np.asarray([row["normal_traction"] for row in rows], dtype=np.float64)
    expected = np.asarray(
        [row["expected_normal_traction"] for row in rows], dtype=np.float64)
    damage = np.asarray([row["damage"] for row in rows], dtype=np.float64)
    dissipated = np.asarray(
        [row["dissipated_energy"] for row in rows], dtype=np.float64)
    expected_dissipated = np.asarray(
        [row["expected_dissipated_energy"] for row in rows], dtype=np.float64)
    axes[0].plot(openings_np, tractions, "o-", label="solver-coupled")
    axes[0].plot(openings_np, expected, "k--", label="bilinear reference")
    axes[0].set_xlabel("normal displacement jump")
    axes[0].set_ylabel("resultant normal traction")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(openings_np, damage, "s-", color="tab:red")
    axes[1].set_xlabel("normal displacement jump")
    axes[1].set_ylabel("committed cohesive damage")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(openings_np, dissipated, "o-", label="integrated")
    axes[2].plot(
        openings_np, expected_dissipated, "k--", label="bilinear energy")
    axes[2].axhline(
        op.integrated_fracture_energy_capacity().item(),
        color="tab:green",
        linestyle=":",
        label="capacity",
    )
    axes[2].set_xlabel("normal displacement jump")
    axes[2].set_ylabel("dissipated energy")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    response_png = output_dir / "cohesive_response.png"
    fig.savefig(response_png, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    ax.triplot(nodes[:, 0], nodes[:, 1], mesh.elements.cpu().numpy(), color="0.65")
    ax.plot([0.0, 1.0], [0.0, 0.0], color="tab:red", linewidth=3.0,
            label="cohesive interface")
    ax.quiver([0.2, 0.8], [0.05, 0.05], [0.0, 0.0], [0.2, 0.2],
              angles="xy", scale_units="xy", scale=1.0, color="tab:blue",
              width=0.008, label="prescribed opening")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8, loc="upper right")
    mesh_png = output_dir / "cohesive_mesh_and_bc.png"
    fig.savefig(mesh_png, dpi=160)
    plt.close(fig)

    visual_manifest = _write_visual_manifest(
        output_dir, [response_png, mesh_png])
    visual_validation_passed = all(
        item["review_dimension_passed"] for item in visual_manifest)
    elapsed_ms = 1_000.0 * (time.perf_counter() - start)
    artifacts = [
        output_dir / "summary.json",
        output_dir / "config.yaml",
        output_dir / "run_lockfile.json",
        output_dir / "run_metadata.json",
        output_dir / "run_manifest.json",
        output_dir / "run.log",
        evidence_csv,
        response_png,
        mesh_png,
        output_dir / "visual_manifest.json",
    ]
    _write_standard_run_files(
        output_dir,
        config_hash=config_hash,
        elapsed_ms=elapsed_ms,
        artifact_paths=artifacts,
        n_steps=len(openings),
    )

    max_traction_error = max(row["abs_traction_error"] for row in rows)
    max_damage_error = max(row["abs_damage_error"] for row in rows)
    max_dissipated_error = max(
        row["abs_dissipated_energy_error"] for row in rows)
    final_damage_error = abs(rows[-1]["damage"] - 1.0)
    final_capacity = float(op.integrated_fracture_energy_capacity().item())
    final_energy_error = abs(rows[-1]["dissipated_energy"] - final_capacity)
    summary = {
        "example": "cohesive_displacement_jump_mode_i",
        "capability": (
            "zero-thickness cohesive displacement-jump interface coupled "
            "through QuasiStaticSolver cohesive_operator hook"
        ),
        "n_nodes": int(mesh.n_nodes),
        "n_elements": int(mesh.elements.shape[0]),
        "n_cohesive_elements": int(len(op.cohesives)),
        "n_load_steps": int(len(rows)),
        "all_steps_converged": all(row["converged"] for row in rows),
        "final_opening": float(openings[-1]),
        "final_normal_traction": float(rows[-1]["normal_traction"]),
        "final_damage": float(rows[-1]["damage"]),
        "final_damage_error": float(final_damage_error),
        "final_dissipated_energy": float(rows[-1]["dissipated_energy"]),
        "fracture_energy_capacity": float(final_capacity),
        "final_dissipated_energy_error": float(final_energy_error),
        "max_abs_traction_error": float(max_traction_error),
        "max_abs_damage_error": float(max_damage_error),
        "max_abs_dissipated_energy_error": float(max_dissipated_error),
        "visual_validation_passed": bool(visual_validation_passed),
        "validation_passed": bool(
            all(row["converged"] for row in rows)
            and max_traction_error < 1.0e-10
            and max_damage_error < 1.0e-12
            and max_dissipated_error < 1.0e-12
            and final_damage_error < 1.0e-12
            and final_energy_error < 1.0e-12
            and visual_validation_passed
        ),
        "csv": evidence_csv.name,
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
        default=Path("outputs/plasticity_interface/cohesive_displacement_jump"),
    )
    args = parser.parse_args()
    summary = run_benchmark(args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
