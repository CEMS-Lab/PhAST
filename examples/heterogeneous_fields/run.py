#!/usr/bin/env python3
"""Run the elementwise E(x)/Gc(x) AT2 teaching problem."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from phast.damage_solver import PhaseFieldDamageSolver
from phast.fem_operators import FEMOperators
from phast.material import Material
from phast.mesh import FEMMesh
from phast.visualization import write_visual_manifest


def build_t3_mesh(nx: int, ny: int, length: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    if nx < 2 or ny < 2:
        raise ValueError("mesh.nx and mesh.ny must both be at least 2")
    xs = np.linspace(0.0, length, nx + 1)
    ys = np.linspace(0.0, height, ny + 1)
    nodes = np.array([(x, y) for y in ys for x in xs], dtype=np.float64)
    elements: list[tuple[int, int, int]] = []
    stride = nx + 1
    for j in range(ny):
        for i in range(nx):
            n00 = j * stride + i
            n10 = n00 + 1
            n01 = n00 + stride
            n11 = n01 + 1
            elements.extend(((n00, n10, n11), (n00, n11, n01)))
    return nodes, np.asarray(elements, dtype=np.int64)


def material_fields(cfg: dict, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    material = cfg["material"]
    heterogeneity = cfg["heterogeneity"]
    inclusion = heterogeneity["soft_inclusion"]
    band = heterogeneity["weak_band"]

    center = np.asarray(inclusion["center"], dtype=np.float64)
    inside = np.linalg.norm(centroids - center, axis=1) <= float(inclusion["radius"])
    E = np.full(centroids.shape[0], float(material["E"]), dtype=np.float64)
    E[inside] *= float(inclusion["E_ratio"])

    in_band = np.abs(centroids[:, 0] - float(band["x_center"])) <= 0.5 * float(band["width"])
    Gc = np.full(centroids.shape[0], float(material["Gc"]), dtype=np.float64)
    Gc[in_band] *= float(band["Gc_ratio"])
    return E, Gc


def run(config_path: Path, output_dir: Path) -> dict:
    started = time.perf_counter()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_cfg = cfg["mesh"]
    nodes_np, elements_np = build_t3_mesh(
        int(mesh_cfg["nx"]), int(mesh_cfg["ny"]),
        float(mesh_cfg["length"]), float(mesh_cfg["height"]),
    )
    centroids = nodes_np[elements_np].mean(axis=1)
    E_np, Gc_np = material_fields(cfg, centroids)

    nodes = torch.tensor(nodes_np, dtype=torch.float64)
    elements = torch.tensor(elements_np, dtype=torch.long)
    mesh = FEMMesh.from_tensors(nodes, elements, device="cpu", dtype=torch.float64)
    material_cfg = cfg["material"]
    material = Material(
        E=float(material_cfg["E"]), nu=float(material_cfg["nu"]),
        Gc=float(material_cfg["Gc"]), l0=float(material_cfg["l0"]), rho=1.0,
        energy_split=str(material_cfg["energy_split"]), pf_model="AT2",
        plane_stress=bool(material_cfg["plane_stress"]),
    )
    fem = FEMOperators(mesh, material)
    fem.diff_E_field = torch.tensor(E_np, dtype=torch.float64)

    strain_yy = float(cfg["loading"]["strain_yy"])
    displacement = torch.zeros((mesh.n_nodes, 2), dtype=torch.float64)
    displacement[:, 0] = -float(material_cfg["nu"]) * strain_yy * nodes[:, 0]
    displacement[:, 1] = strain_yy * nodes[:, 1]
    history = fem.compute_psi_plus(displacement)

    solver_cfg = cfg["solver"]
    damage_solver = PhaseFieldDamageSolver(
        fem, tol=float(solver_cfg["tolerance"]),
        max_iter=int(solver_cfg["max_iterations"]),
        bounds_method=str(solver_cfg["bounds_method"]), use_multigrid=False,
    )
    Gc_field = torch.tensor(Gc_np, dtype=torch.float64)
    damage = damage_solver.solve(
        history, torch.zeros(mesh.n_nodes, dtype=torch.float64),
        Gc_field=Gc_field,
    )
    if not torch.isfinite(damage).all():
        raise RuntimeError("damage solution contains non-finite values")
    damage_elem = damage[elements].mean(dim=1).detach().cpu().numpy()

    with (output_dir / "material_fields.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["element", "centroid_x", "centroid_y", "E", "Gc", "history"])
        for e, (xy, E_value, Gc_value, H_value) in enumerate(
                zip(centroids, E_np, Gc_np, history.detach().cpu().numpy())):
            writer.writerow([e, xy[0], xy[1], E_value, Gc_value, H_value])

    with (output_dir / "damage.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["node", "x", "y", "damage"])
        for node, (xy, value) in enumerate(zip(nodes_np, damage.detach().cpu().numpy())):
            writer.writerow([node, xy[0], xy[1], value])

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    for ax, values, title in ((axes[0], E_np, "Young's modulus E(x)"),
                              (axes[1], Gc_np, "Fracture toughness Gc(x)")):
        image = ax.tripcolor(nodes_np[:, 0], nodes_np[:, 1], elements_np,
                             facecolors=values, shading="flat")
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.82)
    fig.savefig(output_dir / "material_fields.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 3.6), constrained_layout=True)
    image = ax.tripcolor(nodes_np[:, 0], nodes_np[:, 1], elements_np,
                         facecolors=damage_elem, shading="flat", vmin=0.0, vmax=1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Bounded AT2 damage under imposed strain")
    fig.colorbar(image, ax=ax, shrink=0.84, label="damage d")
    fig.savefig(output_dir / "damage_final.png", dpi=160)
    plt.close(fig)

    resolved_text = yaml.safe_dump(cfg, sort_keys=False)
    (output_dir / "parameters_resolved.yaml").write_text(resolved_text, encoding="utf-8")
    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "example": "heterogeneous_fields.elementwise_E_Gc_AT2",
        "capability_boundary": "imposed-strain AT2 damage subproblem; not a coupled benchmark",
        "element_order": "row e of mesh.elements corresponds to E_field[e] and Gc_field[e]",
        "n_nodes": int(mesh.n_nodes),
        "n_elements": int(mesh.n_elems),
        "E_min": float(E_np.min()),
        "E_max": float(E_np.max()),
        "Gc_min": float(Gc_np.min()),
        "Gc_max": float(Gc_np.max()),
        "damage_min": float(damage.min().item()),
        "damage_max": float(damage.max().item()),
        "damage_iterations": int(getattr(damage_solver, "last_iter", 0) or 0),
        "damage_converged": bool(getattr(damage_solver, "last_converged", True) is not False),
        "elapsed_seconds": elapsed,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "schema_version": 1,
        "example": summary["example"],
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "device": "cpu",
        "dtype": "float64",
        "runtime_seconds": elapsed,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    lockfile = {
        "schema_version": 1,
        "config_sha256": hashlib.sha256(resolved_text.encode("utf-8")).hexdigest(),
        "resolved_parameters": cfg,
        "element_order": summary["element_order"],
    }
    (output_dir / "run_lockfile.json").write_text(json.dumps(lockfile, indent=2) + "\n", encoding="utf-8")
    write_visual_manifest(
        output_dir, ["material_fields.png", "damage_final.png"],
        visual_scope="heterogeneous_fields_teaching",
    )
    files = [
        "parameters_resolved.yaml", "material_fields.csv", "damage.csv",
        "material_fields.png", "damage_final.png", "summary.json",
        "run_metadata.json", "run_lockfile.json", "visual_manifest.json",
        "run_manifest.json",
    ]
    manifest = {
        "schema_version": 1,
        "example": summary["example"],
        "command": (
            "python examples/heterogeneous_fields/run.py --config "
            "examples/heterogeneous_fields/parameters.yaml --output-dir <output-dir>"
        ),
        "runtime_seconds": elapsed,
        "metrics": summary,
        "files": files,
        "notes": summary["capability_boundary"],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
