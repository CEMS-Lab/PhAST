"""Compressible neo-Hookean cantilever via Newton + SparseSolveAutograd (#110, #105).

Plane-strain CST plate, clamped left, tip load right. Newton-Raphson outer loop
with the linear solve at each step delegated to ``sparse_solve.solve(...,
backend='auto')`` so the whole chain (including grad-through-solve) goes through
the #106 autograd primitive.
"""
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
from phast.sparse_solve import SparseSolveAutograd, solve
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
    "example": "solid_mechanics.neohookean_plate",
    "mesh": {"nx": 20, "ny": 10, "length": 1.0, "height": 0.2},
    "material": {"E": 2.1e11, "nu": 0.3},
    "loading": {"load_steps": 5, "target_linear_tip_displacement_fraction": 0.05, "load_scale": 0.5},
    "output": {"directory": "outputs"},
}


def build_mesh(nx, ny, L, H, dtype=torch.float64):
    xs = torch.linspace(0.0, L, nx + 1, dtype=dtype)
    ys = torch.linspace(0.0, H, ny + 1, dtype=dtype)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    coords = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    nid = lambda i, j: i * (ny + 1) + j
    elems = []
    for i in range(nx):
        for j in range(ny):
            n00, n10, n11, n01 = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
            elems += [[n00, n10, n11], [n00, n11, n01]]
    return coords, torch.tensor(elems, dtype=torch.long)


def cst_grads(coords_e):
    x1, y1 = coords_e[0]; x2, y2 = coords_e[1]; x3, y3 = coords_e[2]
    A2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    b = torch.stack([y2 - y3, y3 - y1, y1 - y2]) / A2
    c = torch.stack([x3 - x2, x1 - x3, x2 - x1]) / A2
    return b, c, 0.5 * A2


def elem_energy(u_e, coords_e, mu, lam):
    b, c, A = cst_grads(coords_e)
    ux, uy = u_e[0::2], u_e[1::2]
    F2 = torch.stack([torch.stack([1.0 + (b * ux).sum(), (c * ux).sum()]),
                      torch.stack([(b * uy).sum(), 1.0 + (c * uy).sum()])])
    F3 = torch.eye(3, dtype=u_e.dtype).clone(); F3[:2, :2] = F2
    J = torch.det(F3); I_C = (F3 * F3).sum()
    W = 0.5 * mu * (I_C - 3.0) - mu * torch.log(J) + 0.5 * lam * torch.log(J) ** 2
    return W * A


def element_kinematics(u, coords, elems, mu, lam):
    energies = []
    von_mises = []
    detF = []
    for tri in elems:
        idx = tri.tolist()
        dof = [
            2 * idx[0], 2 * idx[0] + 1,
            2 * idx[1], 2 * idx[1] + 1,
            2 * idx[2], 2 * idx[2] + 1,
        ]
        ce = coords[idx]
        u_e = u[dof]
        b, c, _ = cst_grads(ce)
        ux, uy = u_e[0::2], u_e[1::2]
        F2 = torch.stack([
            torch.stack([1.0 + (b * ux).sum(), (c * ux).sum()]),
            torch.stack([(b * uy).sum(), 1.0 + (c * uy).sum()]),
        ])
        F3 = torch.eye(3, dtype=u.dtype)
        F3[:2, :2] = F2
        J = torch.det(F3)
        B = F3 @ F3.T
        I = torch.eye(3, dtype=u.dtype)
        sigma = (mu / J) * (B - I) + (lam * torch.log(J) / J) * I
        sxx, syy, szz = sigma[0, 0], sigma[1, 1], sigma[2, 2]
        sxy = sigma[0, 1]
        vm = torch.sqrt(
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
            + 3.0 * sxy ** 2
        )
        W = (
            0.5 * mu * ((F3 * F3).sum() - 3.0)
            - mu * torch.log(J)
            + 0.5 * lam * torch.log(J) ** 2
        )
        energies.append(W.detach())
        von_mises.append(vm.detach())
        detF.append(J.detach())
    return torch.stack(energies), torch.stack(von_mises), torch.stack(detF)


def assemble(u, coords, elems, mu, lam, n_dof, keep_graph=False):
    """Return (r, K_indices, K_values). If keep_graph=True the returned tensors
    keep autograd dependency on (mu, lam); otherwise they're detached scalars."""
    rows, cols = [], []
    vals_list = []
    r_list = [torch.zeros((), dtype=torch.float64) for _ in range(n_dof)] if keep_graph \
             else None
    r_det = torch.zeros(n_dof, dtype=u.dtype)
    for tri in elems:
        idx = tri.tolist()
        ce = coords[idx]
        dof = [2*idx[0], 2*idx[0]+1, 2*idx[1], 2*idx[1]+1, 2*idx[2], 2*idx[2]+1]
        u_e = u[dof].detach().clone().requires_grad_(True)
        W = elem_energy(u_e, ce, mu, lam)
        f_e = torch.autograd.grad(W, u_e, create_graph=True)[0]
        for a in range(6):
            if keep_graph:
                r_list[dof[a]] = r_list[dof[a]] + f_e[a]
            else:
                r_det[dof[a]] = r_det[dof[a]] + f_e[a].detach()
            grad_a = torch.autograd.grad(f_e[a], u_e, retain_graph=True,
                                         create_graph=keep_graph)[0]
            for bb in range(6):
                rows.append(dof[a]); cols.append(dof[bb])
                vals_list.append(grad_a[bb] if keep_graph else grad_a[bb].item())
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.stack(vals_list) if keep_graph else torch.tensor(vals_list, dtype=u.dtype)
    r = torch.stack(r_list) if keep_graph else r_det
    return r, indices, values


def apply_dirichlet(idx, val, r, fixed, penalty, u):
    extra = torch.tensor(fixed, dtype=torch.long)
    idx_p = torch.cat([idx, torch.stack([extra, extra])], dim=1)
    val_p = torch.cat([val, torch.full((len(fixed),), penalty, dtype=val.dtype)])
    r_bc = r.clone()
    for d in fixed:
        r_bc[d] = penalty * u[d]
    return idx_p, val_p, r_bc


def newton_solve(u, coords, elems, mu, lam, fixed, f_ext, n_dof, tol=1e-6, max_it=30):
    free = torch.ones(n_dof, dtype=torch.bool); free[fixed] = False
    r0 = None; rn = float('inf')
    for it in range(max_it):
        r_int, idx, val = assemble(u, coords, elems, mu, lam, n_dof)
        r = r_int - f_ext
        penalty = val.abs().max().item() * 1e8
        idx_p, val_p, r_bc = apply_dirichlet(idx, val, r, fixed, penalty, u)
        rn = r_bc[free].norm().item()
        if r0 is None: r0 = max(rn, 1e-30)
        if rn / r0 < tol:
            return u, it, rn
        K = torch.sparse_coo_tensor(idx_p, val_p, size=(n_dof, n_dof)).coalesce()
        du = solve(K, -r_bc, backend='auto')
        u = u + du
        if du.abs().max().item() < 1e-9:
            return u, it + 1, rn
    return u, max_it, rn


def run(config_path: str | Path | None = None):
    started = time.perf_counter()
    cfg = load_config(config_path, DEFAULT_CONFIG)
    out_dir = prepare_output_dir(config_path or __file__, cfg)
    torch.set_default_dtype(torch.float64)
    nx, ny = int(cfg["mesh"]["nx"]), int(cfg["mesh"]["ny"])
    L, H = float(cfg["mesh"]["length"]), float(cfg["mesh"]["height"])
    nu = float(cfg["material"]["nu"])
    E_val = float(cfg["material"]["E"])
    n_steps = int(cfg["loading"]["load_steps"])
    coords, elems = build_mesh(nx, ny, L, H)
    n_dof = coords.shape[0] * 2
    mu_v = E_val / (2.0 * (1.0 + nu))
    lam_v = E_val * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    left = [i for i in range(coords.shape[0]) if coords[i, 0].item() == 0.0]
    fixed = [d for n in left for d in (2 * n, 2 * n + 1)]
    tip = nx * (ny + 1) + ny // 2

    I_sec = H ** 3 / 12.0
    # Pick P so linear EB tip deflection ~ 5% of beam length: enough geometric
    # nonlinearity to be visible, low enough for Newton on a coarse CST mesh.
    target_frac = float(cfg["loading"]["target_linear_tip_displacement_fraction"])
    load_scale = float(cfg["loading"]["load_scale"])
    P_lin = -3.0 * E_val * I_sec * (target_frac * L) / L ** 3
    delta_lin = P_lin * L ** 3 / (3.0 * E_val * I_sec)
    P_max = load_scale * P_lin

    u = torch.zeros(n_dof, dtype=torch.float64)
    rows = []
    u_history = []
    print(f"{'frac':>6}  {'P [N]':>11}  {'iters':>5}  {'residual':>12}  {'u_tip [m]':>12}")
    for k in range(1, n_steps + 1):
        frac = k / n_steps
        f_ext = torch.zeros(n_dof, dtype=torch.float64)
        f_ext[2 * tip + 1] = P_max * frac
        u, it, rn = newton_solve(u, coords, elems, mu_v, lam_v, fixed, f_ext, n_dof)
        print(f"{frac:>6.2f}  {P_max*frac:>11.3e}  {it:>5d}  {rn:>12.3e}  {u[2*tip+1].item():>12.4e}")
        rows.append((frac, P_max * frac, it, rn, u[2 * tip + 1].item()))
        u_history.append(u.detach().clone())
    print(f"linear EB at P_max={P_max:.3e}: {load_scale*delta_lin:.4e} m  (NL FE: {u[2*tip+1].item():.4e})")

    # Autograd: rebuild K, r at converged u with E differentiable, then drive
    # SparseSolveAutograd.apply so dE.grad flows through the #106 adjoint.
    E = torch.tensor(E_val, dtype=torch.float64, requires_grad=True)
    mu_g = E / (2.0 * (1.0 + nu))
    lam_g = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    f_ext = torch.zeros(n_dof, dtype=torch.float64); f_ext[2 * tip + 1] = P_max
    r_int, idx_g, val_g = assemble(u, coords, elems, mu_g, lam_g, n_dof, keep_graph=True)
    penalty = val_g.detach().abs().max().item() * 1e8
    idx_p, val_p, r_bc = apply_dirichlet(idx_g, val_g, r_int - f_ext, fixed, penalty, u)
    du = SparseSolveAutograd.apply(idx_p, val_p, -r_bc, n_dof)
    u_tip = u[2 * tip + 1].detach() + du[2 * tip + 1]
    u_tip.abs().sum().backward()
    print(f"d|u_tip|/dE = {E.grad.item():.6e}  (finite, non-zero)")

    csv_path = out_dir / "response.csv"
    with csv_path.open("w", newline="") as fcsv:
        writer = csv.writer(fcsv, lineterminator="\n")
        writer.writerow(["load_fraction", "force_N", "newton_iterations", "residual", "tip_displacement_m"])
        writer.writerows(rows)

    apply_publication_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.2, 2.8), dpi=180)
    forces = [abs(r[1]) for r in rows]
    disp = [abs(r[4]) * 1e3 for r in rows]
    iters = [r[2] for r in rows]
    ax0.plot(disp, forces, marker="o", color="#1f77b4", lw=1.5)
    ax0.set_xlabel(r"$|u_y^\mathrm{tip}|$ [mm]")
    ax0.set_ylabel(r"$|P|$ [N]")
    ax0.set_title("Neo-Hookean load response")
    ax0.grid(alpha=0.25)
    ax1.bar([r[0] for r in rows], iters, width=0.12, color="#ff7f0e", edgecolor="black", linewidth=0.6)
    ax1.set_xlabel("load fraction")
    ax1.set_ylabel("Newton iterations")
    ax1.set_title("Newton convergence")
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    response = out_dir / "response.png"
    fig.savefig(response, bbox_inches="tight")
    plt.close(fig)

    u_nodes = u.reshape(-1, 2)
    disp_mag = torch.linalg.norm(u_nodes, dim=1)
    mesh = FEMMesh.from_tensors(coords, elems, device="cpu", dtype=torch.float64)
    energy_elem, vm_elem, detF_elem = element_kinematics(
        u, coords, elems, torch.tensor(mu_v), torch.tensor(lam_v))
    energy_nodal = mesh.elem_to_node(energy_elem)
    vm_nodal = mesh.elem_to_node(vm_elem)
    detF_nodal = mesh.elem_to_node(detF_elem)
    deformation_scale = save_deformed_shape(
        mesh,
        u_nodes,
        out_dir / "deformed_shape.png",
        title="Neo-Hookean cantilever: deformed shape",
    )
    save_field_plot(
        mesh,
        disp_mag,
        out_dir / "displacement_magnitude.png",
        title="Neo-Hookean cantilever: displacement magnitude",
        colorbar_label=r"$|u|$ [m]",
        cmap="viridis",
        vmin=0.0,
    )
    save_field_plot(
        mesh,
        vm_nodal,
        out_dir / "von_mises.png",
        title="Neo-Hookean cantilever: Cauchy von Mises stress",
        colorbar_label=r"$\sigma_\mathrm{vm}$ [Pa]",
        cmap="magma",
        vmin=0.0,
    )
    save_field_plot(
        mesh,
        energy_nodal,
        out_dir / "strain_energy.png",
        title="Neo-Hookean cantilever: strain-energy density",
        colorbar_label=r"$W$ [J/m$^3$]",
        cmap="plasma",
        vmin=0.0,
    )
    save_field_plot(
        mesh,
        detF_nodal,
        out_dir / "jacobian.png",
        title="Neo-Hookean cantilever: deformation Jacobian",
        colorbar_label=r"$J$",
        cmap="cividis",
    )
    shutil.copyfile(out_dir / "displacement_magnitude.png", out_dir / "displacement_final.png")
    shutil.copyfile(out_dir / "von_mises.png", out_dir / "stress_final.png")
    shutil.copyfile(out_dir / "strain_energy.png", out_dir / "strain_final.png")
    vm_history = []
    energy_history = []
    detf_history = []
    disp_history = []
    displacement_history = []
    for u_step in u_history:
        u_step_nodes = u_step.reshape(-1, 2)
        e_step, vm_step, detf_step = element_kinematics(
            u_step, coords, elems, torch.tensor(mu_v), torch.tensor(lam_v))
        displacement_history.append(u_step_nodes)
        disp_history.append(torch.linalg.norm(u_step_nodes, dim=1))
        vm_history.append(mesh.elem_to_node(vm_step))
        energy_history.append(mesh.elem_to_node(e_step))
        detf_history.append(mesh.elem_to_node(detf_step))
    save_field_animation(
        out_dir,
        mesh=mesh,
        nodal_fields=vm_history,
        title="Neo-Hookean stress evolution",
        colorbar_label=r"$\sigma_\mathrm{vm}$ [Pa]",
        cmap="magma",
        displacements=displacement_history,
        deformation_scale=deformation_scale,
    )
    write_solid_setup_preview(
        out_dir,
        title="Neo-Hookean cantilever setup",
        mesh=mesh,
        config=cfg,
    )
    copy_thumbnail(out_dir, source_name="von_mises.png")
    visual_files = [
        "initial_conditions.png",
        "response.png",
        "deformed_shape.png",
        "displacement_magnitude.png",
        "displacement_final.png",
        "von_mises.png",
        "stress_final.png",
        "strain_energy.png",
        "strain_final.png",
        "jacobian.png",
        "field_evolution.mp4",
        "thumbnail.png",
    ]
    write_visual_manifest(out_dir, visual_files, visual_scope="solid_mechanics_fea")
    write_run_metadata(
        out_dir,
        example="solid_mechanics.neohookean_plate",
        config_path=config_path,
        config=cfg,
    )
    write_run_lockfile(
        out_dir,
        config=cfg,
        command="python examples/solid_mechanics_beta/neohookean_plate/run.py",
    )

    metrics = {
        "final_tip_displacement_m": rows[-1][4],
        "linear_eb_tip_displacement_m": load_scale * delta_lin,
        "max_newton_iterations": max(iters),
        "d_abs_tip_displacement_dE": E.grad.item(),
        "max_displacement_m": float(disp_mag.max().item()),
        "max_von_mises_Pa": float(vm_elem.max().item()),
        "min_detF": float(detF_elem.min().item()),
        "max_strain_energy_density": float(energy_elem.max().item()),
    }
    write_manifest(
        out_dir,
        example="solid_mechanics.neohookean_plate",
        command="python examples/solid_mechanics_beta/neohookean_plate/run.py",
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
            "strain_energy.png",
            "strain_final.png",
            "jacobian.png",
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


def main():
    run(parse_config_arg("Run the nonlinear neo-Hookean solid-mechanics FEA example."))


if __name__ == "__main__":
    main()
