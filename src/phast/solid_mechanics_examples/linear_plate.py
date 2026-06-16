"""Linear elastic cantilever via SparseSolveAutograd (#110).

Minimal demo proving the #106 building block works for non-fracture problems:
plane-strain CST mesh, clamped-left + tip-load right, autograd-through-solve.
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
from phast.sparse_solve import SparseSolveAutograd
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
    save_response_animation,
    write_run_lockfile,
    write_run_metadata,
    write_solid_setup_preview,
    write_solid_zarr,
    write_manifest,
)

DEFAULT_CONFIG = {
    "schema_version": 1,
    "example": "solid_mechanics.linear_plate",
    "mesh": {"nx": 20, "ny": 10, "length": 1.0, "height": 0.2},
    "material": {"E": 2.1e11, "nu": 0.3},
    "loading": {"tip_force_y": -1.0e3},
    "output": {"directory": "outputs"},
}


def build_mesh(nx: int, ny: int, L: float, H: float, dtype=torch.float64):
    xs = torch.linspace(0.0, L, nx + 1, dtype=dtype)
    ys = torch.linspace(0.0, H, ny + 1, dtype=dtype)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    coords = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    nid = lambda i, j: i * (ny + 1) + j
    elems = []
    for i in range(nx):
        for j in range(ny):
            n00, n10, n11, n01 = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
            elems.append([n00, n10, n11])
            elems.append([n00, n11, n01])
    return coords, torch.tensor(elems, dtype=torch.long)


def plane_strain_D(E: torch.Tensor, nu: float):
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    D = torch.zeros(3, 3, dtype=E.dtype if E.ndim else torch.float64)
    D[0, 0] = c * (1.0 - nu); D[1, 1] = c * (1.0 - nu)
    D[0, 1] = c * nu;        D[1, 0] = c * nu
    D[2, 2] = c * (0.5 - nu)
    return D


def cst_BA(coords_e: torch.Tensor):
    x1, y1 = coords_e[0]; x2, y2 = coords_e[1]; x3, y3 = coords_e[2]
    A2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    A = 0.5 * A2
    b = torch.stack([y2 - y3, y3 - y1, y1 - y2]) / A2
    c = torch.stack([x3 - x2, x1 - x3, x2 - x1]) / A2
    B = torch.zeros(3, 6, dtype=coords_e.dtype)
    for k in range(3):
        B[0, 2 * k] = b[k]
        B[1, 2 * k + 1] = c[k]
        B[2, 2 * k] = c[k]; B[2, 2 * k + 1] = b[k]
    return B, A


def assemble(coords, elems, E, nu):
    D = plane_strain_D(E, nu)
    n_dof = coords.shape[0] * 2
    rows, cols, vals = [], [], []
    for tri in elems:
        idx = tri.tolist()
        ce = coords[idx]
        B, A = cst_BA(ce)
        Ke = (B.T @ D @ B) * A
        dof = [2 * idx[0], 2 * idx[0] + 1, 2 * idx[1], 2 * idx[1] + 1, 2 * idx[2], 2 * idx[2] + 1]
        for a in range(6):
            for b in range(6):
                rows.append(dof[a]); cols.append(dof[b]); vals.append(Ke[a, b])
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.stack(vals)
    return indices, values, n_dof


def element_stress_strain(coords, elems, u, E, nu):
    D = plane_strain_D(E, nu)
    strains = []
    stresses = []
    for tri in elems:
        idx = tri.tolist()
        dof = [
            2 * idx[0], 2 * idx[0] + 1,
            2 * idx[1], 2 * idx[1] + 1,
            2 * idx[2], 2 * idx[2] + 1,
        ]
        B, _ = cst_BA(coords[idx])
        eps = B @ u[dof]
        sig = D @ eps
        strains.append(eps.detach())
        stresses.append(sig.detach())
    return torch.stack(strains), torch.stack(stresses)


def von_mises_plane_strain(stress, nu):
    sxx = stress[:, 0]
    syy = stress[:, 1]
    sxy = stress[:, 2]
    szz = nu * (sxx + syy)
    return torch.sqrt(
        0.5 * (
            (sxx - syy) ** 2
            + (syy - szz) ** 2
            + (szz - sxx) ** 2
        )
        + 3.0 * sxy ** 2
    )


def apply_dirichlet_penalty(indices, values, n_dof, fixed_dofs, penalty):
    extra_r, extra_c, extra_v = [], [], []
    for d in fixed_dofs:
        extra_r.append(d); extra_c.append(d); extra_v.append(torch.tensor(penalty, dtype=values.dtype))
    new_indices = torch.cat([indices, torch.tensor([extra_r, extra_c], dtype=torch.long)], dim=1)
    new_values = torch.cat([values, torch.stack(extra_v)])
    return new_indices, new_values


def run(config_path: str | Path | None = None):
    started = time.perf_counter()
    cfg = load_config(config_path, DEFAULT_CONFIG)
    out_dir = prepare_output_dir(config_path or __file__, cfg)
    torch.set_default_dtype(torch.float64)
    nx, ny = int(cfg["mesh"]["nx"]), int(cfg["mesh"]["ny"])
    L, H = float(cfg["mesh"]["length"]), float(cfg["mesh"]["height"])
    nu = float(cfg["material"]["nu"])
    P = float(cfg["loading"]["tip_force_y"])
    E = torch.tensor(float(cfg["material"]["E"]), requires_grad=True)

    coords, elems = build_mesh(nx, ny, L, H)
    indices, values, n_dof = assemble(coords, elems, E, nu)

    left = [i for i in range(coords.shape[0]) if coords[i, 0] == 0.0]
    fixed = []
    for n in left:
        fixed.extend([2 * n, 2 * n + 1])
    penalty = values.abs().max().item() * 1e8
    indices_p, values_p = apply_dirichlet_penalty(indices, values, n_dof, fixed, penalty)

    tip_node = nx * (ny + 1) + ny // 2
    f = torch.zeros(n_dof, dtype=torch.float64)
    f[2 * tip_node + 1] = P

    u = SparseSolveAutograd.apply(indices_p, values_p, f, n_dof)
    u_tip = u[2 * tip_node + 1]

    I = H ** 3 / 12.0
    delta_eb = P * L ** 3 / (3.0 * E.detach().item() * I)
    err = (u_tip.detach().item() - delta_eb) / delta_eb * 100.0

    print(f"tip displacement (FE)         : {u_tip.detach().item():.6e} m")
    print(f"Euler-Bernoulli analytical    : {delta_eb:.6e} m  (error {err:+.1f}%)")

    loss = u_tip.abs()
    loss.backward()
    print(f"d|u_tip|/dE                    : {E.grad.item():.6e}  (finite, non-zero)")

    csv_path = out_dir / "response.csv"
    with csv_path.open("w", newline="") as fcsv:
        writer = csv.writer(fcsv, lineterminator="\n")
        writer.writerow(["quantity", "value", "unit"])
        writer.writerow(["tip_displacement_fe", u_tip.detach().item(), "m"])
        writer.writerow(["tip_displacement_euler_bernoulli", delta_eb, "m"])
        writer.writerow(["relative_error_percent", err, "%"])
        writer.writerow(["d_abs_tip_displacement_dE", E.grad.item(), "m/Pa"])

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=180)
    labels = ["FE", "Euler-Bernoulli"]
    vals = [abs(u_tip.detach().item()) * 1e6, abs(delta_eb) * 1e6]
    bars = ax.bar(labels, vals, color=["#1f77b4", "#ff7f0e"], edgecolor="black", linewidth=0.7)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(r"$|u_y^\mathrm{tip}|$ [$\mu$m]")
    ax.set_title("Linear elastic cantilever")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    response = out_dir / "response.png"
    fig.savefig(response, bbox_inches="tight")
    plt.close(fig)

    u_nodes = u.reshape(-1, 2)
    strains, stresses = element_stress_strain(coords, elems, u, E.detach(), nu)
    mesh = FEMMesh.from_tensors(coords, elems, device="cpu", dtype=torch.float64)
    disp_mag = torch.linalg.norm(u_nodes, dim=1)
    vm_elem = von_mises_plane_strain(stresses, nu)
    vm_nodal = mesh.elem_to_node(vm_elem)
    strain_energy_density = 0.5 * torch.sum(strains * stresses, dim=1)
    energy_nodal = mesh.elem_to_node(strain_energy_density)

    save_deformed_shape(
        mesh,
        u_nodes,
        out_dir / "deformed_shape.png",
        title="Linear elastic plate: deformed shape",
    )
    save_field_plot(
        mesh,
        disp_mag,
        out_dir / "displacement_magnitude.png",
        title="Linear elastic plate: displacement magnitude",
        colorbar_label=r"$|u|$ [m]",
        cmap="viridis",
        vmin=0.0,
    )
    save_field_plot(
        mesh,
        vm_nodal,
        out_dir / "von_mises.png",
        title="Linear elastic plate: von Mises stress",
        colorbar_label=r"$\sigma_\mathrm{vm}$ [Pa]",
        cmap="magma",
        vmin=0.0,
    )
    save_field_plot(
        mesh,
        energy_nodal,
        out_dir / "strain_energy.png",
        title="Linear elastic plate: strain energy density",
        colorbar_label=r"$\psi$ [J/m$^3$]",
        cmap="plasma",
        vmin=0.0,
    )
    shutil.copyfile(out_dir / "displacement_magnitude.png", out_dir / "displacement_final.png")
    shutil.copyfile(out_dir / "von_mises.png", out_dir / "stress_final.png")
    shutil.copyfile(out_dir / "strain_energy.png", out_dir / "strain_final.png")
    ramp = torch.linspace(0.0, 1.0, 9, dtype=torch.float64)
    vm_steps = [factor * vm_nodal for factor in ramp]
    save_response_animation(
        out_dir,
        csv_rows=[
            (abs(float(factor * u_tip.detach().item())) * 1.0e6, abs(float(factor * P)))
            for factor in ramp
        ],
        xlabel=r"$|u_y^\mathrm{tip}|$ [$\mu$m]",
        ylabel=r"$|P|$ [N]",
        title="Linear plate response evolution",
    )
    save_field_animation(
        out_dir,
        mesh=mesh,
        nodal_fields=vm_steps,
        title="Linear plate stress evolution",
        colorbar_label=r"$\sigma_\mathrm{vm}$ [Pa]",
        cmap="magma",
    )
    write_solid_zarr(
        out_dir,
        mesh=mesh,
        steps=[
            {
                "step": int(i),
                "load_fraction": float(factor),
                "tip_displacement_m": float(factor * u_tip.detach().item()),
                "force_N": float(factor * P),
            }
            for i, factor in enumerate(ramp)
        ],
        fields={
            "displacement": torch.stack([factor * u_nodes for factor in ramp]),
            "displacement_magnitude": torch.stack([factor * disp_mag for factor in ramp]),
            "von_mises": torch.stack(vm_steps),
            "strain_energy": torch.stack([factor * energy_nodal for factor in ramp]),
        },
    )
    write_solid_setup_preview(
        out_dir,
        title="Linear elastic cantilever setup",
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
        "response_evolution.mp4",
        "field_evolution.mp4",
        "thumbnail.png",
    ]
    write_visual_manifest(out_dir, visual_files, visual_scope="solid_mechanics_fea")
    write_run_metadata(
        out_dir,
        example="solid_mechanics.linear_plate",
        config_path=config_path,
        config=cfg,
    )
    write_run_lockfile(
        out_dir,
        config=cfg,
        command="python examples/solid_mechanics/linear_plate/run.py",
    )

    metrics = {
        "tip_displacement_fe_m": u_tip.detach().item(),
        "tip_displacement_euler_bernoulli_m": delta_eb,
        "relative_error_percent": err,
        "d_abs_tip_displacement_dE": E.grad.item(),
        "max_displacement_m": float(disp_mag.max().item()),
        "max_von_mises_Pa": float(vm_elem.max().item()),
    }
    write_manifest(
        out_dir,
        example="solid_mechanics.linear_plate",
        command="python examples/solid_mechanics/linear_plate/run.py",
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
            "response_evolution.mp4",
            "field_evolution.mp4",
            "training_data.zarr",
            "zarr_manifest.json",
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
    run(parse_config_arg("Run the linear elastic solid-mechanics FEA example."))


if __name__ == "__main__":
    main()
