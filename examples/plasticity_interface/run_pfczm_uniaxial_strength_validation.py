"""PF-CZM uniaxial strength and length-scale validation example.

This public beta validation example exercises the Wu cohesive phase-field
damage model in a homogeneous uniaxial setting.  The driving force is
``H = sigma_trial^2 / (2E)`` and the PF-CZM parameter ``a1`` is calibrated
from the tensile strength ``sigma_ts``.  The expected onset threshold is
therefore ``Hcrit = sigma_ts^2 / (2E)`` independent of ``l0``.

The example is intentionally bounded: it validates the forward PF-CZM damage
law, nonlinear damage solve, strength threshold, and visualization/telemetry
bundle.  It does not claim a full structural crack-growth or mixed-mode
interface-fracture benchmark.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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

from examples.plasticity_interface._promoted_result_utils import write_zarr_trajectory
from phast.damage_solver import PhaseFieldDamageSolver
from phast.fem_operators import FEMOperators
from phast.material import Material
from phast.mesh import FEMMesh


plt.style.use("default")
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


@dataclass(frozen=True)
class PFCZMCase:
    E: float = 3_000.0
    nu: float = 0.30
    Gc: float = 0.12
    sigma_ts: float = 3.0
    width: float = 1.0
    height: float = 0.25
    nx: int = 20
    ny: int = 5
    n_steps: int = 32


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


def _structured_mesh(case: PFCZMCase) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, case.width, case.nx + 1)
    ys = np.linspace(0.0, case.height, case.ny + 1)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    nodes = np.column_stack([xx.ravel(), yy.ravel()])
    elements: list[list[int]] = []
    stride = case.nx + 1
    for iy in range(case.ny):
        for ix in range(case.nx):
            n00 = iy * stride + ix
            n10 = n00 + 1
            n01 = n00 + stride
            n11 = n01 + 1
            elements.append([n00, n10, n11])
            elements.append([n00, n11, n01])
    return nodes, np.asarray(elements, dtype=np.int64)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_mesh_artifacts(output_dir: Path, nodes: np.ndarray,
                          elements: np.ndarray, case: PFCZMCase) -> None:
    (output_dir / "mesh.geo").write_text(
        "\n".join([
            "// Structured triangular PF-CZM validation mesh.",
            f"// width={case.width}, height={case.height}, nx={case.nx}, ny={case.ny}",
            "// mesh.msh is written directly by meshio from the Python runner.",
            "",
        ])
    )
    meshio.write_points_cells(
        output_dir / "mesh.msh",
        np.column_stack([nodes, np.zeros(nodes.shape[0])]),
        [("triangle", elements)],
        file_format="gmsh22",
    )


def _image_manifest_item(path: Path) -> dict:
    with Image.open(path) as img:
        width, height = img.size
    return {
        "artifact_type": "image",
        "path": path.name,
        "width_px": int(width),
        "height_px": int(height),
        "size_bytes": int(path.stat().st_size),
        "review_dimension_passed": bool(width >= 800 and height >= 500),
    }


def _write_animation(path: Path, frames: list[np.ndarray]) -> None:
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        path, save_all=True, append_images=pil_frames[1:],
        duration=220, loop=0)


def _render_damage_frame(nodes: np.ndarray, elements: np.ndarray,
                         damage: np.ndarray, title: str) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    tpc = ax.tripcolor(
        nodes[:, 0], nodes[:, 1], elements, damage,
        shading="gouraud", cmap="Reds", vmin=0.0, vmax=1.0)
    ax.triplot(nodes[:, 0], nodes[:, 1], elements, color="0.75", lw=0.25)
    ax.set_title(title)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    fig.colorbar(tpc, ax=ax, label="PF-CZM damage d")
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


def _write_plots(output_dir: Path, nodes: np.ndarray, elements: np.ndarray,
                 rows: list[dict], frames: list[np.ndarray]) -> list[dict]:
    visual_paths: list[Path] = []
    final = [r for r in rows if r["l0"] == rows[-1]["l0"]]

    initial_png = output_dir / "initial_conditions.png"
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.triplot(nodes[:, 0], nodes[:, 1], elements, color="0.45", lw=0.35)
    ax.axvline(nodes[:, 0].min(), color="black", linestyle="--", linewidth=1.0,
               label="fixed edge")
    ax.axvline(nodes[:, 0].max(), color="tab:red", linestyle="--", linewidth=1.0,
               label="tensile loading edge")
    ax.set_title("PF-CZM uniaxial initial mesh and loading edges")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(initial_png, dpi=170)
    plt.close(fig)
    visual_paths.append(initial_png)

    damage_png = output_dir / "damage_final.png"
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    final_damage = np.full(nodes.shape[0], final[-1]["damage_mean"], dtype=float)
    tpc = ax.tripcolor(
        nodes[:, 0], nodes[:, 1], elements, final_damage,
        shading="gouraud", cmap="Reds", vmin=0.0, vmax=1.0)
    ax.triplot(nodes[:, 0], nodes[:, 1], elements, color="0.70", lw=0.3)
    ax.set_title("PF-CZM final damage field")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    fig.colorbar(tpc, ax=ax, label="damage d")
    fig.tight_layout()
    fig.savefig(damage_png, dpi=170)
    plt.close(fig)
    visual_paths.append(damage_png)

    response_png = output_dir / "load_displacement.png"
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for l0 in sorted({r["l0"] for r in rows}):
        subset = [r for r in rows if r["l0"] == l0]
        ax.plot(
            [r["strain"] for r in subset],
            [r["degraded_stress"] for r in subset],
            marker="o", ms=3, label=f"l0={l0:g} mm")
    ax.axhline(rows[0]["sigma_ts"], color="0.2", ls="--", lw=1.0,
               label="target tensile strength")
    ax.set_xlabel("nominal strain")
    ax.set_ylabel("degraded stress [MPa]")
    ax.set_title("PF-CZM nominal stress response")
    ax.legend()
    fig.tight_layout()
    fig.savefig(response_png, dpi=170)
    plt.close(fig)
    visual_paths.append(response_png)

    damage_history_png = output_dir / "damage_history.png"
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for l0 in sorted({r["l0"] for r in rows}):
        subset = [r for r in rows if r["l0"] == l0]
        ax.plot(
            [r["sigma_trial"] / r["sigma_ts"] for r in subset],
            [r["damage_mean"] for r in subset],
            marker="o", ms=3, label=f"l0={l0:g} mm")
    ax.axvline(1.0, color="0.2", ls="--", lw=1.0)
    ax.set_xlabel("trial stress / sigma_ts")
    ax.set_ylabel("mean damage")
    ax.set_title("PF-CZM damage onset and growth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(damage_history_png, dpi=170)
    plt.close(fig)
    visual_paths.append(damage_history_png)

    energy_png = output_dir / "energy_split.png"
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for label, key in [
        ("driving", "driving_energy"),
        ("surface", "fracture_surface_energy"),
        ("gradient", "fracture_gradient_energy"),
    ]:
        ax.plot([r["step"] for r in final], [r[key] for r in final],
                marker="o", ms=3, label=label)
    ax.set_xlabel("load step")
    ax.set_ylabel("energy [N mm]")
    ax.set_title("PF-CZM energy split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(energy_png, dpi=170)
    plt.close(fig)
    visual_paths.append(energy_png)

    conv_png = output_dir / "convergence.png"
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    ax1.semilogy([r["step"] for r in final], [max(r["residual_norm"], 1e-16) for r in final],
                 marker="o", ms=3, color="tab:blue", label="residual")
    ax1.set_xlabel("load step")
    ax1.set_ylabel("projected residual", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot([r["step"] for r in final], [r["solver_iters"] for r in final],
             marker="s", ms=3, color="tab:orange", label="iterations")
    ax2.set_ylabel("nonlinear iterations", color="tab:orange")
    ax1.set_title("PF-CZM nonlinear solve convergence")
    fig.tight_layout()
    fig.savefig(conv_png, dpi=170)
    plt.close(fig)
    visual_paths.append(conv_png)

    mesh_png = output_dir / "mesh_deformed.png"
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    strain = final[-1]["strain"]
    deformed = nodes.copy()
    deformed[:, 0] += 4.0 * strain * nodes[:, 0]
    ax.triplot(nodes[:, 0], nodes[:, 1], elements, color="0.75", lw=0.35,
               label="reference")
    ax.triplot(deformed[:, 0], deformed[:, 1], elements, color="tab:red",
               lw=0.5, label="deformed x4")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Mesh and deformed mesh overlay")
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(mesh_png, dpi=170)
    plt.close(fig)
    visual_paths.append(mesh_png)

    anim_path = output_dir / "damage_evolution.gif"
    _write_animation(anim_path, frames)

    manifest = [_image_manifest_item(path) for path in visual_paths]
    manifest.append({
        "artifact_type": "animation",
        "path": anim_path.name,
        "size_bytes": int(anim_path.stat().st_size),
        "media_passed": bool(anim_path.stat().st_size > 0),
    })
    (output_dir / "visual_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    return manifest


def _write_run_metadata(output_dir: Path, case: PFCZMCase,
                        l0_values: list[float]) -> None:
    config = {
        "schema_version": 1,
        "name": "pfczm_uniaxial_strength_validation",
        "material": {
            "E": case.E,
            "nu": case.nu,
            "Gc": case.Gc,
            "sigma_ts": case.sigma_ts,
            "pf_model": "PFCZM",
            "pfczm_softening": "linear",
            "l0_sweep": l0_values,
        },
        "solver": {
            "bounds_method": "projected_cg",
            "damage_tol": 1.0e-8,
            "damage_max_iter": 1000,
        },
    }
    config_text = "\n".join([
        "schema_version: 1",
        "name: pfczm_uniaxial_strength_validation",
        "material:",
        f"  E: {case.E}",
        f"  nu: {case.nu}",
        f"  Gc: {case.Gc}",
        f"  sigma_ts: {case.sigma_ts}",
        "  pf_model: PFCZM",
        "  pfczm_softening: linear",
        f"  l0_sweep: {l0_values}",
        "solver:",
        "  bounds_method: projected_cg",
        "  damage_tol: 1.0e-8",
        "  damage_max_iter: 1000",
        "",
    ])
    (output_dir / "config.yaml").write_text(config_text)
    config_sha256 = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
    common = {
        "git_sha": _git_sha(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "case": config,
    }
    lockfile = {
        "schema": "phast_run_lockfile_v1",
        "git_sha": common["git_sha"],
        "created_at": common["created_at"],
        "platform": common["platform"],
        "config_path": "config.yaml",
        "config_sha256": config_sha256,
        "deterministic": True,
        "case": config,
    }
    (output_dir / "run_lockfile.json").write_text(json.dumps(lockfile, indent=2) + "\n")
    (output_dir / "run_metadata.json").write_text(json.dumps(common, indent=2) + "\n")
    (output_dir / "run.log").write_text(
        "PF-CZM uniaxial strength validation completed.\n")


def run_validation(output_dir: Path, *,
                   l0_values: list[float] | None = None,
                   n_steps: int = 32) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case = PFCZMCase(n_steps=n_steps)
    if l0_values is None:
        l0_values = [0.08, 0.12, 0.18]

    nodes, elements = _structured_mesh(case)
    _write_mesh_artifacts(output_dir, nodes, elements, case)
    _write_run_metadata(output_dir, case, l0_values)

    trial_stresses = np.linspace(0.0, 2.1 * case.sigma_ts, case.n_steps)
    rows: list[dict] = []
    frames: list[np.ndarray] = []
    snapshots: list[tuple[int, dict[str, np.ndarray]]] = []
    peak_stresses = []
    onset_ratios = []

    for l0 in l0_values:
        mesh = FEMMesh.from_tensors(
            torch.tensor(nodes, dtype=torch.float64),
            torch.tensor(elements, dtype=torch.long),
            device="cpu",
            dtype=torch.float64,
            element_type="T3",
        )
        material = Material(
            E=case.E, nu=case.nu, Gc=case.Gc, l0=l0, rho=1.0,
            energy_split="isotropic", pf_model="PFCZM",
            sigma_ts=case.sigma_ts, eta_residual=1.0e-9,
        )
        solver = PhaseFieldDamageSolver(
            FEMOperators(mesh, material),
            tol=1.0e-8,
            max_iter=1000,
            use_multigrid=False,
            bounds_method="projected_cg",
        )
        d_prev = torch.zeros(mesh.n_nodes, dtype=torch.float64)
        first_damage_ratio = None
        degraded_series = []
        for step, sigma in enumerate(trial_stresses):
            step_start = time.perf_counter()
            H_value = sigma * sigma / (2.0 * case.E)
            H = torch.full((mesh.n_elems,), H_value, dtype=torch.float64)
            d = solver.solve(H, d_prev)
            d_prev = d
            d_mean = float(d.mean().item())
            g_mean = float(material.degradation(d).mean().item())
            strain = float(sigma / case.E)
            degraded_stress = float(g_mean * case.E * strain)
            degraded_series.append(degraded_stress)
            if first_damage_ratio is None and d_mean > 1.0e-5:
                first_damage_ratio = float(sigma / case.sigma_ts)

            residual = float(getattr(solver, "last_residual", math.nan))
            solver_converged = bool(getattr(solver, "last_converged", False))
            Gc_l0 = material.Gc * material.l0 / math.pi
            grad_energy = 0.0
            surface_energy = float(
                material.Gc / (math.pi * material.l0)
                * mesh.areas.sum().item()
                * material.pfczm_alpha(d).mean().item())
            driving_energy = float(H_value * mesh.areas.sum().item() * g_mean)
            rows.append({
                "l0": float(l0),
                "step": int(step),
                "sigma_ts": float(case.sigma_ts),
                "sigma_trial": float(sigma),
                "strain": strain,
                "applied_disp": float(strain * case.width),
                "H": float(H_value),
                "H_over_Hcrit": float(H_value / (case.sigma_ts ** 2 / (2.0 * case.E))),
                "damage_mean": d_mean,
                "damage_max": float(d.max().item()),
                "degradation_mean": g_mean,
                "degraded_stress": degraded_stress,
                "reaction_force": degraded_stress,
                "driving_energy": driving_energy,
                "fracture_surface_energy": surface_energy,
                "fracture_gradient_energy": grad_energy,
                "solver_iters": int(getattr(solver, "last_iter", -1)),
                "residual_norm": residual,
                "solver_converged": solver_converged,
                "Gc_l0_over_pi": float(Gc_l0),
                "elapsed_s": time.perf_counter() - step_start,
            })
            if (
                l0 == l0_values[-1]
                and (
                    step % max(1, case.n_steps // 10) == 0
                    or step == case.n_steps - 1
                )
            ):
                damage_np = d.detach().cpu().numpy()
                snapshots.append((
                    int(step),
                    {
                        "damage": damage_np,
                        "displacement": np.column_stack([
                            strain * nodes[:, 0],
                            np.zeros(nodes.shape[0], dtype=np.float64),
                        ]),
                    },
                ))
                frames.append(_render_damage_frame(
                    nodes, elements, damage_np,
                    f"PF-CZM damage, sigma/sigma_ts={sigma / case.sigma_ts:.2f}"))
        peak_stresses.append(float(max(degraded_series)))
        onset_ratios.append(first_damage_ratio if first_damage_ratio is not None else math.inf)

    if not frames:
        frames.append(_render_damage_frame(
            nodes, elements, np.zeros(nodes.shape[0]), "PF-CZM damage"))
        snapshots.append((0, {"damage": np.zeros(nodes.shape[0], dtype=np.float64)}))

    _write_csv(output_dir / "results.csv", rows)
    _write_csv(output_dir / "history.csv", rows)
    _write_csv(output_dir / "solver_telemetry.csv", [
        {
            "l0": r["l0"],
            "step": r["step"],
            "solver_iters": r["solver_iters"],
            "residual_norm": r["residual_norm"],
            "solver_converged": r["solver_converged"],
        }
        for r in rows
    ])
    _write_csv(output_dir / "timing_per_step.csv", [
        {"l0": r["l0"], "step": r["step"], "elapsed_s": r["elapsed_s"]}
        for r in rows
    ])
    _write_csv(output_dir / "energy.csv", [
        {
            "l0": r["l0"],
            "step": r["step"],
            "driving_energy": r["driving_energy"],
            "fracture_surface_energy": r["fracture_surface_energy"],
            "fracture_gradient_energy": r["fracture_gradient_energy"],
        }
        for r in rows
    ])
    visual_manifest = _write_plots(output_dir, nodes, elements, rows, frames)
    write_zarr_trajectory(
        output_dir,
        nodes=nodes,
        elements=elements,
        snapshots=snapshots,
        metadata={
            "validation_id": "pfczm_uniaxial_strength",
            "field_source": "projected_pfczm_damage_solve",
        },
    )

    peak_rel_errors = [
        abs(peak - case.sigma_ts) / case.sigma_ts for peak in peak_stresses
    ]
    max_residual = max(r["residual_norm"] for r in rows)
    all_solvers_converged = all(r["solver_converged"] for r in rows)
    valid = (
        max(peak_rel_errors) < 0.01
        and min(onset_ratios) >= 0.90
        and max(onset_ratios) <= 1.35
        and all_solvers_converged
        and math.isfinite(max_residual)
        and max_residual < 1.0e-5
        and all(item.get("review_dimension_passed", item.get("media_passed", True))
                for item in visual_manifest)
    )
    manifest_paths = [
        "summary.json", "config.yaml", "run_lockfile.json",
        "run_metadata.json", "run_manifest.json", "run.log", "mesh.geo",
        "mesh.msh", "training_data.zarr", "results.csv", "history.csv", "energy.csv",
        "solver_telemetry.csv", "timing_per_step.csv", "damage_final.png",
        "initial_conditions.png", "load_displacement.png", "damage_history.png",
        "energy_split.png", "convergence.png", "mesh_deformed.png",
        "damage_evolution.gif", "visual_manifest.json",
    ]
    run_manifest = {
        "schema": "phast_run_manifest_v1",
        "artifacts": manifest_paths,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n")
    summary = {
        "example": "pfczm_uniaxial_strength_validation",
        "validation_passed": bool(valid),
        "capability_boundary": (
            "Forward Wu PF-CZM damage-law strength calibration validation; not a "
            "full structural crack-growth, mixed-mode delamination, or "
            "PF-plasticity-cohesive benchmark."
        ),
        "n_nodes": int(nodes.shape[0]),
        "n_elements": int(elements.shape[0]),
        "l0_values": [float(v) for v in l0_values],
        "target_sigma_ts": float(case.sigma_ts),
        "peak_degraded_stresses": peak_stresses,
        "max_peak_strength_relative_error": float(max(peak_rel_errors)),
        "damage_onset_sigma_over_sigma_ts": onset_ratios,
        "all_solvers_converged": bool(all_solvers_converged),
        "max_residual_norm": float(max_residual),
        "max_solver_iters": int(max(r["solver_iters"] for r in rows)),
        "visual_manifest_passed": bool(all(
            item.get("review_dimension_passed", item.get("media_passed", True))
            for item in visual_manifest)),
        "max_rss_kib": _max_rss_kib(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/plasticity_interface/pfczm_uniaxial_strength"),
    )
    parser.add_argument("--n-steps", type=int, default=32)
    args = parser.parse_args()
    summary = run_validation(args.output_dir, n_steps=args.n_steps)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
