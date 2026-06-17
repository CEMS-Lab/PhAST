"""Mesh-level J2 plasticity bar FEA example."""
from __future__ import annotations

import csv
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from phast.mesh import FEMMesh
from phast.material import Material
from phast.plasticity import MeshJ2Elastoplasticity, SparseJ2QuasiStaticSolver
from phast.plasticity.j2_vonmises import (
    _stress_dev_norm,
    _stress_deviator_voigt6,
)
from phast.visualization import (
    apply_publication_style,
    save_deformed_shape,
    save_field_plot,
    write_visual_manifest,
)
from ._common import (
    copy_thumbnail,
    load_config,
    parse_config_arg,
    prepare_output_dir,
    save_field_animation,
    write_run_lockfile,
    write_run_metadata,
    write_solid_setup_preview,
    write_manifest,
)

DEFAULT_CONFIG = {
    "schema_version": 1,
    "example": "solid_mechanics.j2_bar",
    "mesh": {
        "nx": 18,
        "ny": 6,
        "length": 1.0,
        "height": 0.25,
        "waist_depth": 0.35,
        "waist_width_fraction": 0.18,
    },
    "material": {
        "E": 210000.0,
        "nu": 0.3,
        "sigma_y0": 250.0,
        "hardening_modulus": 5000.0,
    },
    "loading": {"n_steps": 24, "max_strain_xx": 4.5e-3},
    "solver": {"tol": 1.0e-7, "tol_rel": 1.0e-6, "max_iter": 20, "backend": "auto"},
    "output": {"directory": "outputs"},
}


def von_mises(stress: torch.Tensor) -> torch.Tensor:
    """Return von Mises equivalent stress from Voigt-6 stress."""
    return torch.sqrt(torch.tensor(1.5, dtype=stress.dtype)) * _stress_dev_norm(
        _stress_deviator_voigt6(stress)
    )


def build_bar_mesh(
    nx: int,
    ny: int,
    length: float,
    height: float,
    *,
    waist_depth: float = 0.0,
    waist_width_fraction: float = 0.2,
) -> FEMMesh:
    xs = torch.linspace(0.0, length, nx + 1, dtype=torch.float64)
    eta = torch.linspace(-0.5, 0.5, ny + 1, dtype=torch.float64)
    width = max(float(waist_width_fraction) * length, 1.0e-12)
    local_height = height * (
        1.0
        - float(waist_depth)
        * torch.exp(-((xs - 0.5 * length) / width) ** 2)
    )
    nodes = []
    for i, x in enumerate(xs):
        for y_hat in eta:
            nodes.append([x, y_hat * local_height[i]])
    nodes = torch.tensor(nodes, dtype=torch.float64)
    nid = lambda i, j: i * (ny + 1) + j
    elems = []
    for i in range(nx):
        for j in range(ny):
            n00 = nid(i, j)
            n10 = nid(i + 1, j)
            n11 = nid(i + 1, j + 1)
            n01 = nid(i, j + 1)
            elems.append([n00, n10, n11])
            elems.append([n00, n11, n01])
    node_sets = {
        "left": torch.tensor([nid(0, j) for j in range(ny + 1)], dtype=torch.long),
        "right": torch.tensor([nid(nx, j) for j in range(ny + 1)], dtype=torch.long),
        "bottom_left": torch.tensor([nid(0, 0)], dtype=torch.long),
    }
    return FEMMesh.from_tensors(
        nodes,
        torch.tensor(elems, dtype=torch.long),
        node_sets=node_sets,
        device="cpu",
        dtype=torch.float64,
    )


def run(config_path: str | Path | None = None) -> dict[str, float]:
    started = time.perf_counter()
    cfg = load_config(config_path, DEFAULT_CONFIG)
    out_dir = prepare_output_dir(config_path or __file__, cfg)
    torch.set_default_dtype(torch.float64)

    sigma_y0 = float(cfg["material"]["sigma_y0"])
    hardening = float(cfg["material"]["hardening_modulus"])
    mesh_cfg = cfg["mesh"]
    mesh = build_bar_mesh(
        int(mesh_cfg["nx"]),
        int(mesh_cfg["ny"]),
        float(mesh_cfg["length"]),
        float(mesh_cfg["height"]),
        waist_depth=float(mesh_cfg.get("waist_depth", 0.0)),
        waist_width_fraction=float(mesh_cfg.get("waist_width_fraction", 0.2)),
    )
    mat = Material(
        E=float(cfg["material"]["E"]),
        nu=float(cfg["material"]["nu"]),
        plasticity_model="j2_isotropic",
        yield_stress=sigma_y0,
        hardening_modulus=hardening,
        hardening_type="linear_iso",
        plane_stress=True,
    )
    plasticity = MeshJ2Elastoplasticity(mesh, mat)
    solver_cfg = cfg["solver"]
    solver = SparseJ2QuasiStaticSolver(
        plasticity,
        tol=float(solver_cfg["tol"]),
        tol_rel=float(solver_cfg["tol_rel"]),
        max_iter=int(solver_cfg["max_iter"]),
        backend=str(solver_cfg["backend"]),
    )

    rows = []
    u_history = []
    vm_history = []
    eqp_history = []
    n_steps = int(cfg["loading"]["n_steps"])
    max_strain = float(cfg["loading"]["max_strain_xx"])
    left = mesh.node_sets["left"]
    right = mesh.node_sets["right"]
    bottom_left = mesh.node_sets["bottom_left"]
    u = torch.zeros((mesh.n_nodes, 2), dtype=mesh.dtype)
    for step in range(1, n_steps + 1):
        eps_xx = max_strain * step / n_steps
        bc_mask = torch.zeros((mesh.n_nodes, 2), dtype=torch.bool)
        bc_vals = torch.zeros((mesh.n_nodes, 2), dtype=mesh.dtype)
        bc_mask[left, 0] = True
        bc_mask[bottom_left, 1] = True
        bc_mask[right, 0] = True
        bc_vals[right, 0] = eps_xx * float(mesh_cfg["length"])
        u, converged, n_iter = solver.solve(bc_mask, bc_vals, u_init=u)
        if not converged:
            raise RuntimeError(
                f"J2 bar FEA failed at step {step}: {solver.last_failure}; "
                f"residual={solver.last_residual:.3e}")
        stress = plasticity.state.stress
        vm = von_mises(stress)
        eqp = plasticity.state.eps_p_eq
        reaction = plasticity.internal_force(state=plasticity.state)[right, 0].sum()
        yield_current = sigma_y0 + hardening * eqp.mean()
        rows.append((
            step,
            eps_xx,
            float(stress[:, 0].mean().item()),
            float(vm.mean().item()),
            float(vm.max().item()),
            float(eqp.mean().item()),
            float(eqp.max().item()),
            float(yield_current.item()),
            float(reaction.item()),
            int(n_iter),
            float(solver.last_residual),
        ))
        u_history.append(u.detach().clone())
        vm_history.append(mesh.elem_to_node(vm.detach()))
        eqp_history.append(mesh.elem_to_node(eqp.detach()))

    print("J2 plasticity bar FEA: displacement-controlled mesh solve")
    print(f"{'step':>4} {'eps_xx':>10} {'sigma_xx [MPa]':>16} "
          f"{'vm [MPa]':>10} {'eps_p_eq':>11} {'yield [MPa]':>12}")
    for step, eps_xx, sigma_xx, vm_mean, _vm_max, eqp_mean, _eqp_max, yield_current, *_ in rows[::5]:
        print(f"{step:4d} {eps_xx:10.4e} {sigma_xx:16.3f} "
              f"{vm_mean:10.3f} {eqp_mean:11.4e} {yield_current:12.3f}")

    last = rows[-1]
    plastic_steps = sum(1 for row in rows if row[5] > 0.0)
    final_vm = von_mises(plasticity.state.stress)
    final_yield = sigma_y0 + hardening * plasticity.state.eps_p_eq
    yielded = plasticity.state.eps_p_eq > 1.0e-10
    vm_error = float((final_vm[yielded] - final_yield[yielded]).abs().max().item())
    print(f"plastic steps: {plastic_steps}/{len(rows)}")
    print(f"final eps_p_eq: {last[5]:.6e}")
    print(f"final vm-yield residual: {vm_error:.3e} MPa")

    if plastic_steps == 0 or vm_error > 1.0:
        raise SystemExit("J2 consistency check failed")

    csv_path = out_dir / "response.csv"
    with csv_path.open("w", newline="") as fcsv:
        writer = csv.writer(fcsv, lineterminator="\n")
        writer.writerow([
            "step",
            "eps_xx",
            "sigma_xx_mean_MPa",
            "von_mises_mean_MPa",
            "von_mises_max_MPa",
            "eps_p_eq_mean",
            "eps_p_eq_max",
            "yield_mean_MPa",
            "reaction_x_MPa_mm",
            "newton_iterations",
            "residual",
        ])
        writer.writerows(rows)

    apply_publication_style()
    eps = [r[1] for r in rows]
    sig = [r[2] for r in rows]
    vm = [r[3] for r in rows]
    yld = [r[7] for r in rows]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.2, 2.8), dpi=180)
    ax0.plot(eps, sig, marker="o", ms=3, lw=1.4, color="#1f77b4")
    ax0.set_xlabel(r"axial strain $\varepsilon_{xx}$")
    ax0.set_ylabel(r"$\sigma_{xx}$ [MPa]")
    ax0.set_title("J2 bar response")
    ax0.grid(alpha=0.25)
    ax1.plot(eps, vm, label="von Mises", color="#d62728", lw=1.4)
    ax1.plot(eps, yld, "--", label="current yield", color="#2ca02c", lw=1.2)
    ax1.set_xlabel(r"axial strain $\varepsilon_{xx}$")
    ax1.set_ylabel("stress [MPa]")
    ax1.set_title("Yield consistency")
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    response = out_dir / "response.png"
    fig.savefig(response, bbox_inches="tight")
    plt.close(fig)

    u_nodes = u
    disp_mag = torch.linalg.norm(u_nodes, dim=1)
    vm_nodal = mesh.elem_to_node(von_mises(plasticity.state.stress))
    eqp_nodal = mesh.elem_to_node(plasticity.state.eps_p_eq)
    deformation_scale = save_deformed_shape(
        mesh,
        u_nodes,
        out_dir / "deformed_shape.png",
        title="J2 plasticity bar: deformed shape",
    )
    save_field_plot(
        mesh,
        disp_mag,
        out_dir / "displacement_magnitude.png",
        title="J2 plasticity bar: displacement magnitude",
        colorbar_label=r"$|u|$ [mm]",
        cmap="viridis",
        vmin=0.0,
    )
    save_field_plot(
        mesh,
        vm_nodal,
        out_dir / "von_mises.png",
        title="J2 plasticity bar: von Mises stress",
        colorbar_label=r"$\sigma_\mathrm{vm}$ [MPa]",
        cmap="magma",
        vmin=0.0,
    )
    save_field_plot(
        mesh,
        eqp_nodal,
        out_dir / "equivalent_plastic_strain.png",
        title="J2 plasticity bar: equivalent plastic strain",
        colorbar_label=r"$\bar{\varepsilon}^p$",
        cmap="plasma",
        vmin=0.0,
    )
    shutil.copyfile(out_dir / "displacement_magnitude.png", out_dir / "displacement_final.png")
    shutil.copyfile(out_dir / "von_mises.png", out_dir / "stress_final.png")
    shutil.copyfile(out_dir / "equivalent_plastic_strain.png", out_dir / "strain_final.png")
    shutil.copyfile(out_dir / "equivalent_plastic_strain.png", out_dir / "plastic_strain_final.png")
    save_field_animation(
        out_dir,
        mesh=mesh,
        nodal_fields=eqp_history,
        title="J2 plastic strain evolution",
        colorbar_label=r"$\bar{\varepsilon}^p$",
        cmap="plasma",
        displacements=u_history,
        deformation_scale=deformation_scale,
    )
    write_solid_setup_preview(
        out_dir,
        title="J2 plasticity bar setup",
        mesh=mesh,
        config=cfg,
    )
    copy_thumbnail(out_dir, source_name="equivalent_plastic_strain.png")
    visual_files = [
        "initial_conditions.png",
        "response.png",
        "deformed_shape.png",
        "displacement_magnitude.png",
        "displacement_final.png",
        "von_mises.png",
        "stress_final.png",
        "equivalent_plastic_strain.png",
        "strain_final.png",
        "plastic_strain_final.png",
        "field_evolution.mp4",
        "thumbnail.png",
    ]
    write_visual_manifest(out_dir, visual_files, visual_scope="solid_mechanics_fea")
    write_run_metadata(
        out_dir,
        example="solid_mechanics.j2_bar",
        config_path=config_path,
        config=cfg,
    )
    write_run_lockfile(
        out_dir,
        config=cfg,
        command="python examples/solid_mechanics_beta/j2_bar/run.py",
    )

    metrics = {
        "plastic_steps": plastic_steps,
        "final_eps_p_eq": last[5],
        "final_vm_yield_residual_MPa": vm_error,
        "max_von_mises_MPa": last[4],
        "max_equivalent_plastic_strain": last[6],
        "resolved_backend": solver.last_backend,
    }
    write_manifest(
        out_dir,
        example="solid_mechanics.j2_bar",
        command="python examples/solid_mechanics_beta/j2_bar/run.py",
        config=cfg,
        metrics=metrics,
        files=[
            "response.csv",
            "response.png",
            "initial_conditions.png",
            "deformed_shape.png",
            "displacement_magnitude.png",
            "displacement_final.png",
            "von_mises.png",
            "stress_final.png",
            "equivalent_plastic_strain.png",
            "strain_final.png",
            "plastic_strain_final.png",
            "field_evolution.mp4",
            "thumbnail.png",
            "visual_manifest.json",
            "run_metadata.json",
            "run_lockfile.json",
            "run_manifest.json",
            "fluent_setup.py",
        ],
        started_at=started,
    )
    return metrics


def main() -> None:
    run(parse_config_arg("Run the mesh-level J2 solid-mechanics FEA example."))


if __name__ == "__main__":
    main()
