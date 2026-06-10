"""Native Q4 sparse mechanics plus AT2 damage smoke benchmark.

This is the customer-facing smoke for issue #675. It exercises native Q4
mechanics without converting cells to T3, records the sparse backend selected
by ``QuasiStaticSolver`` (SciPy locally, MUMPS where PETSc/MUMPS is available),
and advances native Q4 AT2 damage from Gauss-point history.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

from phast.damage_solver import PhaseFieldDamageSolver
from phast.fem_operators import FEMOperators
from phast.material import Material
from phast.mechanics_solver import QuasiStaticSolver
from phast.mesh import FEMMesh
from phast.quad_elements import q4_to_triangles, structured_q4_mesh
from phast.sparse_solve import available_sparse_backends


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _max_rss_kib() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw // 1024 if raw > 100_000_000 else raw


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_config(output_dir: Path, *, nx: int, ny: int, n_steps: int,
                  max_disp: float, backend: str) -> str:
    text = "\n".join([
        "case: q4_sparse_at2_smoke",
        "mesh: structured_native_Q4",
        f"nx: {nx}",
        f"ny: {ny}",
        "element_type: Q4",
        "mechanics: QuasiStaticSolver sparse-direct isotropic Q4 assembly",
        "damage: PhaseFieldDamageSolver native Q4 AT2 matrix-free CG",
        f"requested_backend: {backend}",
        f"n_steps: {n_steps}",
        f"max_right_displacement: {max_disp}",
        "material:",
        "  E: 210000.0",
        "  nu: 0.30",
        "  Gc: 2.7",
        "  l0: 0.08",
        "  energy_split: isotropic",
        "  pf_model: AT2",
    ]) + "\n"
    (output_dir / "config.yaml").write_text(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_visual_manifest(output_dir: Path, image_paths: list[Path]) -> list[dict]:
    manifest = []
    for path in image_paths:
        with Image.open(path) as img:
            width, height = img.size
        manifest.append({
            "path": path.name,
            "artifact_type": "image",
            "width_px": int(width),
            "height_px": int(height),
            "review_dimension_passed": bool(width >= 800 and height >= 500),
        })
    (output_dir / "visual_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    return manifest


def _plot_damage(mesh: FEMMesh, damage: torch.Tensor, out: Path) -> None:
    tris = q4_to_triangles(mesh.elements.detach().cpu(), diagonal="02").numpy()
    x = mesh.nodes[:, 0].detach().cpu().numpy()
    y = mesh.nodes[:, 1].detach().cpu().numpy()
    z = damage.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    tpc = ax.tripcolor(x, y, tris, z, shading="gouraud", cmap="magma", vmin=0.0, vmax=1.0)
    ax.triplot(x, y, tris, color="white", linewidth=0.25, alpha=0.45)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Native Q4 AT2 damage")
    fig.colorbar(tpc, ax=ax, label="damage")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _plot_deformed_mesh(mesh: FEMMesh, u: torch.Tensor, out: Path) -> None:
    nodes = mesh.nodes.detach().cpu()
    u_cpu = u.detach().cpu()
    scale = 8.0
    deformed = nodes + scale * u_cpu
    quads = mesh.elements.detach().cpu().tolist()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    for quad in quads:
        loop = quad + [quad[0]]
        ax.plot(nodes[loop, 0], nodes[loop, 1], color="0.75", linewidth=0.6)
        ax.plot(deformed[loop, 0], deformed[loop, 1], color="#2563eb", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Q4 mesh and deformed mesh (x{scale:g})")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _plot_load_displacement(rows: list[dict], out: Path) -> None:
    disp = [float(row["right_displacement"]) for row in rows]
    reaction = [float(row["right_reaction"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.plot(disp, reaction, marker="o", linewidth=1.8)
    ax.set_xlabel("right displacement")
    ax.set_ylabel("reaction force")
    ax.set_title("Native Q4 sparse mechanics response")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _write_standard_files(output_dir: Path, *, config_hash: str,
                          elapsed_ms: float, artifact_paths: list[Path],
                          rows: list[dict], backend: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    backend_status = available_sparse_backends()
    metadata = {
        "benchmark": "q4_sparse_at2_smoke",
        "timestamp_utc": now,
        "git_sha": _git_sha(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "requested_backend": backend,
        "resolved_backends": sorted({row["resolved_backend"] for row in rows}),
        "sparse_backend_status": {
            "scipy": bool(backend_status.scipy),
            "petsc_mumps": bool(backend_status.petsc),
            "cudss": bool(backend_status.cudss),
        },
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
        "n_load_steps": len(rows),
    }
    manifest = {
        "schema": "phast_run_manifest_v1",
        "benchmark": "q4_sparse_at2_smoke",
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
            f"{now} q4_sparse_at2_smoke started",
            f"{now} requested backend: {backend}",
            f"{now} resolved backends: {metadata['resolved_backends']}",
            f"{now} elapsed_ms: {elapsed_ms:.3f}",
        ]) + "\n")


def run_smoke(output_dir: Path, *, nx: int = 32, ny: int = 16,
              n_steps: int = 5, backend: str = "scipy",
              max_disp: float = 2.5e-4) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config_hash = _write_config(
        output_dir, nx=nx, ny=ny, n_steps=n_steps,
        max_disp=max_disp, backend=backend)

    qmesh = structured_q4_mesh(width=1.0, height=0.5, nx=nx, ny=ny)
    mesh = FEMMesh.from_tensors(
        qmesh.nodes, qmesh.quads, node_sets=qmesh.node_sets,
        device="cpu", dtype=torch.float64, element_type="Q4")
    material = Material(
        E=210_000.0, nu=0.30, Gc=2.7, l0=0.08,
        energy_split="isotropic", pf_model="AT2")
    fem = FEMOperators(mesh, material)
    mech = QuasiStaticSolver(
        fem, backend=backend, max_iter=8, tol=1.0e-9, tol_rel=1.0e-10)
    damage_solver = PhaseFieldDamageSolver(
        fem, max_iter=500, tol=1.0e-9,
        use_multigrid=False, bounds_method="projected_cg")

    f_ext = torch.zeros((mesh.n_nodes, 2), dtype=torch.float64)
    d = torch.zeros(mesh.n_nodes, dtype=torch.float64)
    H_hist = torch.zeros((mesh.n_elems, 4), dtype=torch.float64)
    u = torch.zeros((mesh.n_nodes, 2), dtype=torch.float64)
    rows = []
    telemetry = []

    for step, right_disp in enumerate(torch.linspace(max_disp / n_steps, max_disp, n_steps), start=1):
        bc_mask = torch.zeros((mesh.n_nodes, 2), dtype=torch.bool)
        bc_vals = torch.zeros((mesh.n_nodes, 2), dtype=torch.float64)
        bc_mask[mesh.node_sets["left"], :] = True
        bc_mask[mesh.node_sets["right"], 0] = True
        bc_vals[mesh.node_sets["right"], 0] = float(right_disp.item())

        u, converged, n_iter = mech.solve(
            d, f_ext, bc_mask, bc_vals, u_init=u)
        H_hist = torch.maximum(H_hist, fem.compute_psi_plus(u))
        d = damage_solver.solve(H_hist, d)
        fint = fem.internal_force(u, d)
        right_reaction = float(fint[mesh.node_sets["right"], 0].sum().item())
        row = {
            "step": step,
            "right_displacement": float(right_disp.item()),
            "right_reaction": right_reaction,
            "mechanics_converged": bool(converged),
            "mechanics_newton_iter": int(n_iter),
            "requested_backend": backend,
            "resolved_backend": mech.last_backend,
            "mechanics_residual": float(mech.last_residual),
            "damage_pcg_iter": int(damage_solver.last_iter),
            "damage_residual": float(getattr(damage_solver, "last_residual", float("nan"))),
            "damage_max": float(d.max().item()),
            "damage_mean": float(d.mean().item()),
            "max_rss_kib": _max_rss_kib(),
        }
        rows.append(row)
        telemetry.append({
            "step": step,
            "mechanics_backend": mech.last_backend,
            "mechanics_converged": bool(converged),
            "mechanics_residual": float(mech.last_residual),
            "damage_iter": int(damage_solver.last_iter),
            "damage_residual": row["damage_residual"],
        })

    results_csv = output_dir / "results.csv"
    telemetry_csv = output_dir / "solver_telemetry.csv"
    backend_csv = output_dir / "backend_evidence.csv"
    _write_csv(results_csv, rows)
    _write_csv(telemetry_csv, telemetry)
    _write_csv(backend_csv, [{
        "requested_backend": backend,
        "resolved_backends": ";".join(sorted({row["resolved_backend"] for row in rows})),
        "n_nodes": mesh.n_nodes,
        "n_elements": mesh.n_elems,
        "n_dofs": 2 * mesh.n_nodes,
        "max_rss_kib": _max_rss_kib(),
    }])

    damage_png = output_dir / "damage_final.png"
    mesh_png = output_dir / "mesh_deformed.png"
    ld_png = output_dir / "load_displacement.png"
    _plot_damage(mesh, d, damage_png)
    _plot_deformed_mesh(mesh, u, mesh_png)
    _plot_load_displacement(rows, ld_png)
    visual_manifest = _write_visual_manifest(
        output_dir, [damage_png, mesh_png, ld_png])

    artifact_paths = [
        output_dir / "config.yaml",
        results_csv,
        telemetry_csv,
        backend_csv,
        damage_png,
        mesh_png,
        ld_png,
        output_dir / "visual_manifest.json",
    ]
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    _write_standard_files(
        output_dir,
        config_hash=config_hash,
        elapsed_ms=elapsed_ms,
        artifact_paths=artifact_paths,
        rows=rows,
        backend=backend,
    )

    summary = {
        "example": "q4_sparse_at2_smoke",
        "n_nodes": mesh.n_nodes,
        "n_elements": mesh.n_elems,
        "requested_backend": backend,
        "resolved_backends": sorted({row["resolved_backend"] for row in rows}),
        "all_mechanics_converged": all(row["mechanics_converged"] for row in rows),
        "max_mechanics_residual": max(float(row["mechanics_residual"]) for row in rows),
        "final_damage_max": float(d.max().item()),
        "final_damage_mean": float(d.mean().item()),
        "visual_manifest_passed": all(
            item["review_dimension_passed"] for item in visual_manifest),
        "artifacts": [path.name for path in artifact_paths],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/plasticity_interface/q4_sparse_at2_smoke"),
    )
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument("--backend", default="scipy", choices=("auto", "cg", "scipy", "mumps"))
    parser.add_argument("--max-disp", type=float, default=2.5e-4)
    args = parser.parse_args()
    print(json.dumps(
        run_smoke(
            args.output_dir,
            nx=args.nx,
            ny=args.ny,
            n_steps=args.n_steps,
            backend=args.backend,
            max_disp=args.max_disp,
        ),
        indent=2,
    ))


if __name__ == "__main__":
    main()
