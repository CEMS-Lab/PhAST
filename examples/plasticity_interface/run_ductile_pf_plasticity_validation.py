"""Mesh-level J2 plus ductile phase-field damage validation.

This example validates the first coupled ductile PF-plasticity slice:
mesh-level J2 state is updated/committed, accumulated plastic work enters the
phase-field driving force, and the bounded phase-field damage problem is solved
on that ductile history.

It is not yet a monolithic/staggered global PF-plasticity benchmark with
consistent elastoplastic damage tangent; that production gate remains separate.
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
import torch
from PIL import Image

from phast.damage_solver import PhaseFieldDamageSolver
from phast.fem_operators import FEMOperators
from phast.material import Material
from phast.mechanics_solver import QuasiStaticSolver
from phast.mesh import FEMMesh
from phast.plasticity import (
    DuctilePhaseFieldCoupling,
    MeshJ2Elastoplasticity,
)
from phast.sparse_solve import available_sparse_backends


plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


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


def _mesh() -> FEMMesh:
    nodes = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=torch.float64,
    )
    elements = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    node_sets = {
        "left": torch.tensor([0, 3], dtype=torch.long),
        "right": torch.tensor([1, 2], dtype=torch.long),
        "bottom": torch.tensor([0, 1], dtype=torch.long),
        "top": torch.tensor([2, 3], dtype=torch.long),
    }
    return FEMMesh.from_tensors(
        nodes, elements, node_sets=node_sets, device="cpu", dtype=torch.float64)


def _material(*, l0: float = 0.1) -> Material:
    return Material(
        E=210_000.0,
        nu=0.30,
        Gc=2.7,
        l0=l0,
        rho=7.8e-9,
        energy_split="amor",
        pf_model="AT2",
        plasticity_model="j2_isotropic",
        yield_stress=250.0,
        hardening_modulus=5_000.0,
        hardening_type="linear_iso",
        plane_stress=True,
    )


def _write_visual_manifest(output_dir: Path, names: list[str]) -> Path:
    rows = []
    for name in names:
        path = output_dir / name
        with Image.open(path) as img:
            width, height = img.size
        scope = "review"
        if name == "damage_final.png":
            scope = "diagnostic_damage_proxy_summary"
        rows.append({
            "file": name,
            "width_px": int(width),
            "height_px": int(height),
            "size_bytes": int(path.stat().st_size),
            "review_dimension_passed": bool(max(width, height) < 2000),
            "visual_scope": scope,
        })
    path = output_dir / "visual_manifest.json"
    path.write_text(json.dumps(rows, indent=2) + "\n")
    return path


def _write_mesh_artifacts(output_dir: Path, mesh: FEMMesh) -> dict[str, str]:
    geo = output_dir / "mesh.geo"
    geo.write_text(
        "\n".join([
            "// One-square two-triangle mesh for solver-level J2 validation.",
            "Point(1) = {0, 0, 0, 1};",
            "Point(2) = {1, 0, 0, 1};",
            "Point(3) = {1, 1, 0, 1};",
            "Point(4) = {0, 1, 0, 1};",
            "Line(1) = {1, 2};",
            "Line(2) = {2, 3};",
            "Line(3) = {3, 4};",
            "Line(4) = {4, 1};",
            "Curve Loop(1) = {1, 2, 3, 4};",
            "Plane Surface(1) = {1};",
            "",
        ])
    )
    msh = output_dir / "mesh.msh"
    nodes = mesh.nodes.detach().cpu().numpy()
    elements = mesh.elements.detach().cpu().numpy()
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


def _write_provenance(output_dir: Path, *, n_steps: int, max_strain: float,
                      l0: float, plastic_work_weight: float,
                      backend: str, resolved_backend: str,
                      backend_status, elapsed_ms: float) -> dict[str, str]:
    config_text = "\n".join([
        "example: ductile_pf_plasticity_validation",
        "model: sparse_quasistatic_j2_plus_bounded_ductile_phase_field_damage",
        "capability_boundary: operator-coupled validation patch, not full benchmark",
        f"n_steps: {n_steps}",
        f"max_strain: {max_strain}",
        f"plastic_work_weight: {plastic_work_weight}",
        f"requested_backend: {backend}",
        "material:",
        "  E: 210000.0",
        "  nu: 0.30",
        "  Gc: 2.7",
        f"  l0: {l0}",
        "  pf_model: AT2",
        "  plasticity_model: j2_isotropic",
        "  yield_stress: 250.0",
        "  hardening_modulus: 5000.0",
        "",
    ])
    config = output_dir / "config.yaml"
    config.write_text(config_text)
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "device": "cpu",
        "dtype": "float64",
        "elapsed_ms": elapsed_ms,
        "max_rss_kib": _max_rss_kib(),
        "backend_status": {
            "scipy": bool(backend_status.scipy),
            "petsc_mumps": bool(backend_status.petsc),
            "cudss": bool(backend_status.cudss),
        },
    }
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    lockfile_path = output_dir / "run_lockfile.json"
    lockfile_path.write_text(
        json.dumps({
            "config_sha256": hashlib.sha256(
                config_text.encode("utf-8")).hexdigest(),
            "resolved_config": {
                "n_steps": n_steps,
                "max_strain": max_strain,
                "l0": l0,
                "plastic_work_weight": plastic_work_weight,
                "solver": (
                    "QuasiStaticSolver(plasticity_operator=MeshJ2Elastoplasticity)"
                    "+PhaseFieldDamageSolver"
                ),
                "backend": backend,
                "resolved_backend": resolved_backend,
                "backend_status": {
                    "scipy": bool(backend_status.scipy),
                    "petsc_mumps": bool(backend_status.petsc),
                    "cudss": bool(backend_status.cudss),
                },
            },
            "metadata": metadata,
        }, indent=2) + "\n"
    )
    return {
        "config.yaml": str(config),
        "run_metadata.json": str(metadata_path),
        "run_lockfile.json": str(lockfile_path),
    }


def run_validation(output_dir: Path, *, n_steps: int = 48,
                   max_strain: float = 5.0e-3, l0: float = 0.1,
                   plastic_work_weight: float = 1.0,
                   backend: str = "auto") -> dict:
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = _mesh()
    material = _material(l0=l0)
    plasticity = MeshJ2Elastoplasticity(mesh, material)
    backend_status = available_sparse_backends()
    fem = FEMOperators(mesh, material)
    mechanics = QuasiStaticSolver(
        fem, plasticity_operator=plasticity, backend=backend,
        tol=1.0e-6, tol_rel=1.0e-7, max_iter=15)
    damage_solver = PhaseFieldDamageSolver(
        fem, tol=1.0e-9, max_iter=300, use_multigrid=False,
        bounds_method="projected_cg")
    coupling = DuctilePhaseFieldCoupling(
        fem=fem, plasticity=plasticity,
        plastic_work_weight=plastic_work_weight)

    rows = []
    energy_rows = []
    H_elastic = torch.zeros(mesh.n_elems, dtype=mesh.dtype)
    H_ductile = torch.zeros(mesh.n_elems, dtype=mesh.dtype)
    threshold = 3.0 * material.Gc / (16.0 * material.l0)
    u = torch.zeros((mesh.n_nodes, 2), dtype=mesh.dtype, device=mesh.device)
    d = torch.zeros(mesh.n_nodes, dtype=mesh.dtype, device=mesh.device)
    f_ext = torch.zeros((mesh.n_nodes, 2), dtype=mesh.dtype, device=mesh.device)
    left = mesh.node_sets["left"]
    right = mesh.node_sets["right"]
    for step in range(1, n_steps + 1):
        eps = max_strain * step / n_steps
        bc_mask = torch.zeros((mesh.n_nodes, 2), dtype=torch.bool)
        bc_vals = torch.zeros((mesh.n_nodes, 2), dtype=mesh.dtype)
        bc_mask[left, :] = True
        bc_mask[right, 0] = True
        bc_vals[right, 0] = eps
        u, converged, n_iter = mechanics.solve(
            d, f_ext, bc_mask, bc_vals, u_init=u)
        if not converged:
            raise RuntimeError(
                f"J2 quasi-static solve failed at step {step}: "
                f"residual={mechanics.last_residual}")
        psi = fem.compute_psi_plus(u)
        driving = coupling.driving_force(u, state=plasticity.state)
        H_elastic = torch.maximum(H_elastic, psi)
        H_ductile = torch.maximum(H_ductile, driving)
        d_elastic = torch.clamp(H_elastic / threshold, max=1.0)
        d_ductile = torch.clamp(H_ductile / threshold, max=1.0)
        d = damage_solver.solve(H_ductile, d)
        damage_residual = damage_solver.compute_residual(H_ductile, d)
        damage_residual_norm = float(damage_residual.norm().item())
        energy_components = fem.compute_energy_components(u, d, psi_plus=psi)
        fracture_surface, fracture_gradient = fem._fracture_energy_terms(d)
        plastic_work_total = float(
            (plasticity.state.plastic_work_density * mesh.areas).sum().item())
        elastic_driving_total = float((psi * mesh.areas).sum().item())
        ductile_driving_total = float((driving * mesh.areas).sum().item())
        fracture_total = float(fracture_surface + fracture_gradient)
        stored_plus_dissipated_total = float(
            energy_components["elastic"] + plastic_work_total + fracture_total)

        rows.append({
            "step": step,
            "eps_xx": float(eps),
            "mechanics_converged": bool(converged),
            "mechanics_newton_iter": int(n_iter),
            "mechanics_residual": float(mechanics.last_residual),
            "sigma_xx_mpa": float(plasticity.state.stress[:, 0].mean().item()),
            "eps_p_eq_mean": float(plasticity.state.eps_p_eq.mean().item()),
            "plastic_work_density_mean": float(
                plasticity.state.plastic_work_density.mean().item()),
            "elastic_driving_mean": float(psi.mean().item()),
            "ductile_driving_mean": float(driving.mean().item()),
            "damage_proxy_elastic_mean": float(d_elastic.mean().item()),
            "damage_proxy_ductile_mean": float(d_ductile.mean().item()),
            "damage_mean": float(d.mean().item()),
            "damage_max": float(d.max().item()),
            "damage_min": float(d.min().item()),
            "damage_pcg_iter": int(getattr(damage_solver, "last_iter", -1)),
            "damage_residual_norm": damage_residual_norm,
            "wall_ms": float(1000.0 * (time.perf_counter() - t0)),
        })
        energy_rows.append({
            "step": step,
            "eps_xx": float(eps),
            "elastic_driving_total": elastic_driving_total,
            "plastic_work_total": plastic_work_total,
            "ductile_driving_total": ductile_driving_total,
            "degraded_elastic_energy": float(energy_components["elastic"]),
            "fracture_surface_energy": float(fracture_surface),
            "fracture_gradient_energy": float(fracture_gradient),
            "fracture_total_energy": fracture_total,
            "kinetic_energy": float(energy_components["kinetic"]),
            "stored_plus_dissipated_total": stored_plus_dissipated_total,
            "damage_mean": float(d.mean().item()),
            "damage_residual_norm": damage_residual_norm,
        })

    for filename in ("results.csv", "history.csv",
                     "solver_telemetry.csv", "timing_per_step.csv"):
        path = output_dir / filename
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    energy_csv = output_dir / "energy.csv"
    with energy_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(energy_rows[0].keys()))
        writer.writeheader()
        writer.writerows(energy_rows)

    fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
    ax.triplot(
        mesh.nodes[:, 0].cpu(), mesh.nodes[:, 1].cpu(),
        mesh.elements.cpu(), color="0.55", linewidth=0.9)
    ax.scatter(mesh.nodes[left, 0].cpu(), mesh.nodes[left, 1].cpu(),
               marker="s", label="left fixed")
    ax.scatter(mesh.nodes[right, 0].cpu(), mesh.nodes[right, 1].cpu(),
               marker=">", label="right ux")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Boundary setup")
    ax.legend(loc="best", fontsize=8)
    initial_png = output_dir / "initial_conditions.png"
    fig.savefig(initial_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), constrained_layout=True)
    axes[0].plot([r["eps_xx"] for r in rows],
                 [r["sigma_xx_mpa"] for r in rows], linewidth=1.5)
    axes[0].set_xlabel("axial strain eps_xx")
    axes[0].set_ylabel("mean sigma_xx [MPa]")
    axes[0].set_title("Mesh-level J2 response")
    axes[0].grid(True, linewidth=0.4, alpha=0.4)

    axes[1].plot([r["eps_xx"] for r in rows],
                 [r["elastic_driving_mean"] for r in rows],
                 label="elastic")
    axes[1].plot([r["eps_xx"] for r in rows],
                 [r["ductile_driving_mean"] for r in rows],
                 label="elastic + plastic work")
    axes[1].set_xlabel("axial strain eps_xx")
    axes[1].set_ylabel("mean driving force")
    axes[1].set_title("Ductile PF driving force")
    axes[1].legend(loc="best")
    axes[1].grid(True, linewidth=0.4, alpha=0.4)
    response_png = output_dir / "ductile_pf_coupling.png"
    fig.savefig(response_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
    ax.plot([r["eps_xx"] for r in rows],
            [r["sigma_xx_mpa"] for r in rows], linewidth=1.5)
    ax.set_xlabel("applied strain")
    ax.set_ylabel("mean sigma_xx [MPa]")
    ax.set_title("Load response")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    load_png = output_dir / "load_displacement.png"
    fig.savefig(load_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
    ax.plot([r["eps_xx"] for r in energy_rows],
            [r["elastic_driving_total"] for r in energy_rows],
            label="elastic driving")
    ax.plot([r["eps_xx"] for r in energy_rows],
            [r["plastic_work_total"] for r in energy_rows],
            label="plastic work")
    ax.plot([r["eps_xx"] for r in energy_rows],
            [r["fracture_total_energy"] for r in energy_rows],
            label="fracture")
    ax.set_xlabel("applied strain")
    ax.set_ylabel("integrated energy")
    ax.set_title("Energy ledger")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    energy_png = output_dir / "energy.png"
    fig.savefig(energy_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    final = rows[-1]
    final_energy = energy_rows[-1]
    resolved_backend = getattr(mechanics, "last_backend", backend) or backend
    final_u_norm = float(u.norm().item())
    plastic_work_monotone = all(
        energy_rows[i]["plastic_work_total"]
        <= energy_rows[i + 1]["plastic_work_total"] + 1.0e-12
        for i in range(len(energy_rows) - 1)
    )
    fracture_energy_monotone = all(
        energy_rows[i]["fracture_total_energy"]
        <= energy_rows[i + 1]["fracture_total_energy"] + 1.0e-12
        for i in range(len(energy_rows) - 1)
    )
    finite_energy = all(
        torch.isfinite(torch.tensor([
            row["elastic_driving_total"],
            row["plastic_work_total"],
            row["ductile_driving_total"],
            row["degraded_elastic_energy"],
            row["fracture_surface_energy"],
            row["fracture_gradient_energy"],
            row["fracture_total_energy"],
            row["stored_plus_dissipated_total"],
        ], dtype=torch.float64)).all().item()
        for row in energy_rows
    )
    fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
    ax.bar(["elastic proxy", "ductile proxy", "PF solve"], [
        final["damage_proxy_elastic_mean"],
        final["damage_proxy_ductile_mean"],
        final["damage_mean"],
    ])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("mean damage/proxy d")
    ax.set_title("Final ductile damage proxy summary")
    damage_png = output_dir / "damage_final.png"
    fig.savefig(damage_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    visual_manifest = _write_visual_manifest(
        output_dir, [
            "initial_conditions.png",
            "ductile_pf_coupling.png",
            "load_displacement.png",
            "energy.png",
            "damage_final.png",
        ])

    elapsed_ms = 1000.0 * (time.perf_counter() - t0)
    provenance = _write_provenance(
        output_dir, n_steps=n_steps, max_strain=max_strain,
        l0=l0, plastic_work_weight=plastic_work_weight,
        backend=backend, resolved_backend=resolved_backend,
        backend_status=backend_status,
        elapsed_ms=elapsed_ms)
    mesh_artifacts = _write_mesh_artifacts(output_dir, mesh)
    run_log = output_dir / "run.log"
    run_log.write_text(
        "\n".join([
            "example=ductile_pf_plasticity_validation",
            "solver=QuasiStaticSolver+MeshJ2Elastoplasticity+PhaseFieldDamageSolver",
            f"requested_backend={backend}",
            f"resolved_backend={resolved_backend}",
            f"n_steps={n_steps}",
            f"yielded={bool(final['eps_p_eq_mean'] > 0.0)}",
            f"final_residual={final['mechanics_residual']:.12e}",
            f"final_damage_residual_norm={final['damage_residual_norm']:.12e}",
            f"final_plastic_work_total={final_energy['plastic_work_total']:.12e}",
            f"final_fracture_total_energy={final_energy['fracture_total_energy']:.12e}",
            f"plastic_work_monotone={plastic_work_monotone}",
            f"fracture_energy_monotone={fracture_energy_monotone}",
            f"elapsed_ms={elapsed_ms:.3f}",
            "",
        ])
    )
    summary = {
        "example": "ductile_pf_plasticity_validation",
        "capability_boundary": (
            "sparse quasi-static J2 plus bounded ductile phase-field damage "
            "solve on a validation patch; not yet a benchmark-matched "
            "monolithic/staggered ductile fracture workflow"
        ),
        "n_elements": mesh.n_elems,
        "n_steps": n_steps,
        "l0": l0,
        "plastic_work_weight": plastic_work_weight,
        "requested_backend": backend,
        "resolved_backend": resolved_backend,
        "backend_status": {
            "scipy": bool(backend_status.scipy),
            "petsc_mumps": bool(backend_status.petsc),
            "cudss": bool(backend_status.cudss),
        },
        "yielded": bool(final["eps_p_eq_mean"] > 0.0),
        "final_elastic_driving_mean": final["elastic_driving_mean"],
        "final_ductile_driving_mean": final["ductile_driving_mean"],
        "final_damage_proxy_elastic_mean": final["damage_proxy_elastic_mean"],
        "final_damage_proxy_ductile_mean": final["damage_proxy_ductile_mean"],
        "final_damage_mean": final["damage_mean"],
        "final_damage_max": final["damage_max"],
        "final_damage_residual_norm": final["damage_residual_norm"],
        "final_damage_pcg_iter": final["damage_pcg_iter"],
        "final_mechanics_newton_iter": final["mechanics_newton_iter"],
        "final_mechanics_residual": final["mechanics_residual"],
        "final_u_norm": final_u_norm,
        "final_sigma_xx_mpa": final["sigma_xx_mpa"],
        "final_elastic_driving_total": final_energy["elastic_driving_total"],
        "final_plastic_work_total": final_energy["plastic_work_total"],
        "final_ductile_driving_total": final_energy["ductile_driving_total"],
        "final_degraded_elastic_energy": final_energy["degraded_elastic_energy"],
        "final_fracture_surface_energy": final_energy["fracture_surface_energy"],
        "final_fracture_gradient_energy": final_energy["fracture_gradient_energy"],
        "final_fracture_total_energy": final_energy["fracture_total_energy"],
        "final_stored_plus_dissipated_total": (
            final_energy["stored_plus_dissipated_total"]
        ),
        "plastic_work_monotone": bool(plastic_work_monotone),
        "fracture_energy_monotone": bool(fracture_energy_monotone),
        "finite_energy_terms": bool(finite_energy),
        "max_rss_kib": _max_rss_kib(),
        "visual_manifest": str(visual_manifest),
        "provenance": provenance,
        "mesh_artifacts": mesh_artifacts,
        "run_log": str(run_log),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "run_manifest.json").write_text(
        json.dumps({
            "example": "ductile_pf_plasticity_validation",
            "summary": "summary.json",
            "standard_outputs": {
                "config.yaml": str(output_dir / "config.yaml"),
                "run_lockfile.json": str(output_dir / "run_lockfile.json"),
                "run_metadata.json": str(output_dir / "run_metadata.json"),
                "run.log": str(run_log),
                "mesh.geo": str(output_dir / "mesh.geo"),
                "mesh.msh": str(output_dir / "mesh.msh"),
                "results.csv": str(output_dir / "results.csv"),
                "history.csv": str(output_dir / "history.csv"),
                "energy.csv": str(output_dir / "energy.csv"),
                "solver_telemetry.csv": str(output_dir / "solver_telemetry.csv"),
                "timing_per_step.csv": str(output_dir / "timing_per_step.csv"),
                "initial_conditions.png": str(initial_png),
                "ductile_pf_coupling.png": str(response_png),
                "load_displacement.png": str(load_png),
                "energy.png": str(energy_png),
                "damage_final.png": str(damage_png),
                "visual_manifest.json": str(visual_manifest),
            },
            "yielded": summary["yielded"],
            "max_rss_kib": summary["max_rss_kib"],
        }, indent=2) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/plasticity_interface/ductile_pf_plasticity"),
    )
    parser.add_argument("--n-steps", type=int, default=48)
    parser.add_argument("--max-strain", type=float, default=5.0e-3)
    parser.add_argument("--l0", type=float, default=0.1)
    parser.add_argument("--plastic-work-weight", type=float, default=1.0)
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=("auto", "scipy", "mumps", "cudss", "cg"),
    )
    args = parser.parse_args()
    print(json.dumps(run_validation(
        args.output_dir,
        n_steps=args.n_steps,
        max_strain=args.max_strain,
        l0=args.l0,
        plastic_work_weight=args.plastic_work_weight,
        backend=args.backend,
    ), indent=2))


if __name__ == "__main__":
    main()
