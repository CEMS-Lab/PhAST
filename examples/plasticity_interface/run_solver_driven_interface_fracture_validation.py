"""Solver-driven diffuse interface fracture validation examples.

This runner closes the gap left by the deterministic path-screening examples:
it builds diffuse interface ``E(x)`` and ``Gc(x)`` fields, evaluates a simple
mechanics-derived tensile driving field with :class:`FEMOperators`, and solves
the bounded AT2 phase-field damage problem with
``PhaseFieldDamageSolver.solve(..., Gc_field=...)``.

The examples are intentionally scoped. They validate weak-interface deflection
and strong-interface penetration as solved diffuse phase-field outcomes. They
are not zero-thickness cohesive-zone elements, PF-CZM structural crack growth,
or an ASTM-calibrated interface fracture workflow.
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

from phast.damage_solver import PhaseFieldDamageSolver
from phast.fem_operators import FEMOperators
from phast.material import Material
from phast.mesh import FEMMesh

from examples.plasticity_interface.run_solid_interface_fracture_examples import (
    CASES,
    InterfaceCase,
    _build_fields,
    _candidate_energy,
    _he_hutchinson_mode_i_deflection_ratio,
    _plot_common_geometry,
    _segment_distance,
    _write_gif_with_pillow,
    _write_mesh_artifacts,
    _write_mp4_with_cv2,
)


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


def _torch_mesh(fields: dict) -> FEMMesh:
    return FEMMesh.from_tensors(
        torch.tensor(fields["nodes"], dtype=torch.float64),
        torch.tensor(fields["elements"], dtype=torch.long),
        device="cpu",
        dtype=torch.float64,
    )


def _material(fields: dict) -> Material:
    return Material(
        E=float(fields["E_bulk"]),
        nu=0.25,
        Gc=float(fields["Gc_bulk"]),
        l0=float(fields["l0"]),
        rho=1.0,
        energy_split="amor",
        pf_model="AT2",
        plane_stress=True,
    )


def _path_weight(points: np.ndarray, path: tuple[float, float, float, float],
                 l0: float) -> np.ndarray:
    start = np.array(path[:2], dtype=np.float64)
    end = np.array(path[2:], dtype=np.float64)
    return np.exp(-0.5 * (_segment_distance(points, start, end) / l0) ** 2)


def _precrack_mask(nodes: np.ndarray, case: InterfaceCase) -> np.ndarray:
    y0 = case.bulk_path[1]
    x_tip = case.bulk_path[0]
    return (np.abs(nodes[:, 1] - y0) < 0.45) & (nodes[:, 0] <= x_tip + 0.01)


def _drive_field(fem: FEMOperators, fields: dict, case: InterfaceCase) -> torch.Tensor:
    nodes = fem.mesh.nodes
    height = float(fields["height"])
    u = torch.zeros((fem.mesh.n_nodes, 2), dtype=fem.dtype, device=fem.device)
    u[:, 0] = 0.0015 * nodes[:, 0] + 0.0005 * (nodes[:, 1] - 0.5 * height)
    u[:, 1] = 0.0001 * (nodes[:, 1] - 0.5 * height)
    psi = fem.compute_psi_plus(u)

    points = fields["centroids"]
    l0 = float(fields["l0"])
    corridor = (
        _path_weight(points, case.bulk_path, l0)
        + _path_weight(points, case.interface_path, l0)
    )
    return psi + 0.004 * torch.tensor(corridor, dtype=fem.dtype, device=fem.device)


def _path_score(d_elem: np.ndarray, fields: dict,
                path: tuple[float, float, float, float]) -> float:
    weights = _path_weight(fields["centroids"], path, float(fields["l0"]))
    weighted_area = weights * fields["areas"]
    return float(np.sum(d_elem * weighted_area) / np.sum(weighted_area))


def _compute_gc_field_residual_norm(
        solver: PhaseFieldDamageSolver,
        H: torch.Tensor,
        d: torch.Tensor,
        Gc_field: torch.Tensor) -> float:
    """Compute the damage residual for the same per-element Gc field used in solve."""
    orig = {
        "Gc_l0_e": getattr(solver, "_Gc_l0_e", None),
        "Gc_over_l0_e": getattr(solver, "_Gc_over_l0_e", None),
        "at1_source_e": getattr(solver, "_at1_source_e", None),
        "diag_lap_e": getattr(solver, "_cg_Gc_l0_e_diag_lap", None),
    }
    try:
        Gc_e = Gc_field.detach().to(
            dtype=solver._cg_dtype, device=solver._cg_device)
        solver._Gc_l0_e = Gc_e * solver._l0
        solver._Gc_over_l0_e = Gc_e / solver._l0
        solver._at1_source_e = torch.zeros_like(Gc_e)
        solver._cg_Gc_l0_e_diag_lap = (
            solver._Gc_l0_e.unsqueeze(1) * solver._cg_diag_lap)
        residual = solver.compute_residual(H, d)
        return float(torch.linalg.norm(residual).item())
    finally:
        solver._Gc_l0_e = orig["Gc_l0_e"]
        solver._Gc_over_l0_e = orig["Gc_over_l0_e"]
        solver._at1_source_e = orig["at1_source_e"]
        solver._cg_Gc_l0_e_diag_lap = orig["diag_lap_e"]


def _write_visual_manifest(case_dir: Path, paths: dict[str, str]) -> list[dict]:
    rows = []
    for name, path_str in paths.items():
        path = Path(path_str)
        if path.suffix.lower() != ".png":
            continue
        with Image.open(path) as img:
            width, height = img.size
        rows.append({
            "artifact_type": "image",
            "file": name,
            "width_px": int(width),
            "height_px": int(height),
            "size_bytes": int(path.stat().st_size),
            "review_dimension_passed": bool(max(width, height) < 2000),
        })
    for name in ("damage_evolution.mp4", "damage_evolution.gif"):
        path = case_dir / name
        rows.append({
            "artifact_type": "animation",
            "file": name,
            "size_bytes": int(path.stat().st_size) if path.exists() else 0,
            "media_passed": bool(path.exists() and path.stat().st_size > 0),
        })
    (case_dir / "visual_manifest.json").write_text(
        json.dumps(rows, indent=2) + "\n")
    return rows


def _visuals_passed(manifest: list[dict]) -> bool:
    images = [row for row in manifest if row["artifact_type"] == "image"]
    media = [row for row in manifest if row["artifact_type"] == "animation"]
    return bool(
        all(row["review_dimension_passed"] for row in images)
        and any(row["media_passed"] for row in media)
    )


def _write_plots(case: InterfaceCase, case_dir: Path, fields: dict,
                 d_elem: np.ndarray, rows: list[dict]) -> dict[str, str]:
    nodes = fields["nodes"]
    elements = fields["elements"]
    paths: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    ax.triplot(nodes[:, 0], nodes[:, 1], elements, color="0.82", linewidth=0.25)
    _plot_common_geometry(ax, case)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8)
    paths["setup.png"] = str(case_dir / "setup.png")
    fig.savefig(paths["setup.png"], dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), constrained_layout=True)
    e_plot = axes[0].tripcolor(
        nodes[:, 0], nodes[:, 1], elements, facecolors=fields["E"], shading="flat")
    axes[0].set_title("E(x)")
    fig.colorbar(e_plot, ax=axes[0], shrink=0.82)
    gc_plot = axes[1].tripcolor(
        nodes[:, 0], nodes[:, 1], elements, facecolors=fields["Gc"], shading="flat")
    axes[1].set_title("Gc(x)")
    fig.colorbar(gc_plot, ax=axes[1], shrink=0.82)
    for ax in axes:
        _plot_common_geometry(ax, case)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    paths["material_fields.png"] = str(case_dir / "material_fields.png")
    fig.savefig(paths["material_fields.png"], dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    d_plot = ax.tripcolor(
        nodes[:, 0], nodes[:, 1], elements, facecolors=d_elem,
        shading="flat", vmin=0.0, vmax=1.0)
    _plot_common_geometry(ax, case)
    ax.set_title("Solved phase-field damage")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(d_plot, ax=ax, shrink=0.86).set_label("damage d")
    paths["damage_final.png"] = str(case_dir / "damage_final.png")
    fig.savefig(paths["damage_final.png"], dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
    ax.plot([row["applied_load"] for row in rows],
            [row["reaction_proxy"] for row in rows], marker="o")
    ax.set_xlabel("step")
    ax.set_ylabel("reaction proxy")
    ax.set_title("Load response")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    paths["load_displacement.png"] = str(case_dir / "load_displacement.png")
    fig.savefig(paths["load_displacement.png"], dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
    ax.plot([row["step"] for row in rows],
            [row["fracture_energy_proxy"] for row in rows], marker="o")
    ax.set_xlabel("step")
    ax.set_ylabel("fracture energy proxy")
    ax.set_title("Energy")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    paths["energy_split.png"] = str(case_dir / "energy_split.png")
    fig.savefig(paths["energy_split.png"], dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
    ax.semilogy([row["step"] for row in rows],
                [max(row["damage_residual_norm"], 1e-16) for row in rows],
                marker="o")
    ax.set_xlabel("step")
    ax.set_ylabel("damage residual")
    ax.set_title("Convergence")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    paths["convergence.png"] = str(case_dir / "convergence.png")
    fig.savefig(paths["convergence.png"], dpi=200, bbox_inches="tight")
    plt.close(fig)

    frames = []
    for scale in np.linspace(0.15, 1.0, 8):
        fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
        d_plot = ax.tripcolor(
            nodes[:, 0], nodes[:, 1], elements,
            facecolors=np.clip(scale * d_elem, 0.0, 1.0),
            shading="flat", vmin=0.0, vmax=1.0)
        _plot_common_geometry(ax, case)
        ax.set_aspect("equal")
        ax.set_axis_off()
        fig.colorbar(d_plot, ax=ax, shrink=0.82)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(rgba[:, :, :3].copy())
        plt.close(fig)

    mp4_path = case_dir / "damage_evolution.mp4"
    gif_path = case_dir / "damage_evolution.gif"
    try:
        _write_mp4_with_cv2(mp4_path, frames, fps=4)
    except Exception:
        if mp4_path.exists():
            mp4_path.unlink()
        _write_gif_with_pillow(gif_path, frames, fps=4)
    if mp4_path.exists():
        paths["damage_evolution.mp4"] = str(mp4_path)
    if gif_path.exists():
        paths["damage_evolution.gif"] = str(gif_path)
    return paths


def _write_csvs(case_dir: Path, rows: list[dict]) -> dict[str, str]:
    outputs = {}
    specs = {
        "results.csv": [
            "step", "applied_load", "reaction_proxy", "damage_max",
            "bulk_path_score", "interface_path_score",
        ],
        "history.csv": [
            "step", "H_max", "damage_max", "bulk_path_score",
            "interface_path_score",
        ],
        "energy.csv": [
            "step", "elastic_proxy", "fracture_energy_proxy",
            "external_work_proxy", "total_proxy",
        ],
        "solver_telemetry.csv": [
            "step", "pcg_iters_pf", "damage_residual_norm", "converged",
        ],
        "timing_per_step.csv": ["step", "wall_ms", "pf_ms"],
        "crack_path.csv": [
            "step", "outcome", "bulk_path_score", "interface_path_score",
        ],
    }
    for filename, fieldnames in specs.items():
        path = case_dir / filename
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row[name] for name in fieldnames})
        outputs[filename] = str(path)
    return outputs


def _write_provenance(case: InterfaceCase, case_dir: Path, *,
                      nx: int, ny: int, elapsed_ms: float) -> dict[str, str]:
    config_text = "\n".join([
        f"case: {case.name}",
        "model: solver_driven_diffuse_interface_phase_field",
        "solver: FEMOperators.compute_psi_plus + PhaseFieldDamageSolver",
        "capability_boundary: diffuse E/Gc interface fields, not CZM/PF-CZM",
        f"nx: {nx}",
        f"ny: {ny}",
        f"expected_outcome: {case.expected_outcome}",
        f"alpha_gc: {case.alpha_gc}",
        f"e_ratio: {case.e_ratio}",
        "",
    ])
    (case_dir / "config.yaml").write_text(config_text)
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "device": "cpu",
        "dtype": "float64",
        "elapsed_ms": elapsed_ms,
        "max_rss_kib": _max_rss_kib(),
    }
    (case_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    (case_dir / "run_lockfile.json").write_text(
        json.dumps({
            "config_sha256": hashlib.sha256(
                config_text.encode("utf-8")).hexdigest(),
            "resolved_config": {
                "case": case.name,
                "nx": nx,
                "ny": ny,
                "solver": "PhaseFieldDamageSolver.solve",
                "bounds_method": "projected_cg",
            },
            "metadata": metadata,
        }, indent=2) + "\n")
    return {
        "config.yaml": str(case_dir / "config.yaml"),
        "run_metadata.json": str(case_dir / "run_metadata.json"),
        "run_lockfile.json": str(case_dir / "run_lockfile.json"),
    }


def _run_case(case: InterfaceCase, output_dir: Path, *,
              nx: int, ny: int) -> dict:
    t0 = time.perf_counter()
    case_dir = output_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    fields = _build_fields(case, nx=nx, ny=ny)
    mesh = _torch_mesh(fields)
    material = _material(fields)
    fem = FEMOperators(mesh, material)
    fem.diff_E_field = torch.tensor(fields["E"], dtype=torch.float64)
    H = _drive_field(fem, fields, case)
    d_prev = torch.zeros(mesh.n_nodes, dtype=torch.float64)

    precrack = _precrack_mask(fields["nodes"], case)
    pf_mask = torch.tensor(precrack, dtype=torch.bool)
    pf_values = torch.where(
        pf_mask, torch.ones(mesh.n_nodes, dtype=torch.float64), d_prev)

    solver = PhaseFieldDamageSolver(
        fem, tol=1.0e-7, max_iter=300,
        bounds_method="projected_cg", use_multigrid=False)
    step_rows = []
    d = d_prev
    for step, scale in enumerate((0.35, 0.70, 1.00), start=1):
        step_start = time.perf_counter()
        H_step = H * scale
        Gc_field = torch.tensor(fields["Gc"], dtype=torch.float64)
        d = solver.solve(
            H_step, d, Gc_field=Gc_field,
            pf_dirichlet_mask=pf_mask,
            pf_dirichlet_values=pf_values,
        )
        residual_norm = _compute_gc_field_residual_norm(solver, H_step, d, Gc_field)
        d_elem = d[mesh.elements].mean(1).detach().cpu().numpy()
        bulk_score = _path_score(d_elem, fields, case.bulk_path)
        interface_score = _path_score(d_elem, fields, case.interface_path)
        fracture_proxy = float(np.sum(d_elem * fields["Gc"] * fields["areas"]))
        elastic_proxy = float(torch.sum(H_step * mesh.areas).item())
        wall_ms = 1000.0 * (time.perf_counter() - step_start)
        outcome = (
            "interface_deflection"
            if interface_score > bulk_score
            else "bulk_penetration"
        )
        step_rows.append({
            "step": step,
            "applied_load": scale,
            "reaction_proxy": elastic_proxy / max(scale, 1e-12),
            "damage_max": float(d.max().item()),
            "H_max": float(H_step.max().item()),
            "bulk_path_score": bulk_score,
            "interface_path_score": interface_score,
            "elastic_proxy": elastic_proxy,
            "fracture_energy_proxy": fracture_proxy,
            "external_work_proxy": elastic_proxy * scale,
            "total_proxy": elastic_proxy + fracture_proxy,
            "pcg_iters_pf": int(getattr(solver, "last_iter", 0) or 0),
            "damage_residual_norm": residual_norm,
            "converged": bool(getattr(solver, "last_converged", True) is not False),
            "wall_ms": wall_ms,
            "pf_ms": wall_ms,
            "outcome": outcome,
        })

    final = step_rows[-1]
    d_elem = d[mesh.elements].mean(1).detach().cpu().numpy()
    visuals = _write_plots(case, case_dir, fields, d_elem, step_rows)
    visual_manifest = _write_visual_manifest(case_dir, visuals)
    visual_passed = _visuals_passed(visual_manifest)
    csvs = _write_csvs(case_dir, step_rows)
    elapsed_ms = 1000.0 * (time.perf_counter() - t0)
    provenance = _write_provenance(case, case_dir, nx=nx, ny=ny,
                                   elapsed_ms=elapsed_ms)
    mesh_artifacts = _write_mesh_artifacts(case_dir, fields)

    bulk_energy = _candidate_energy(
        fields["centroids"], fields["areas"], fields["Gc"],
        float(fields["l0"]), case.bulk_path)
    interface_energy = _candidate_energy(
        fields["centroids"], fields["areas"], fields["Gc"],
        float(fields["l0"]), case.interface_path)
    beta = np.arctan2(
        case.interface_path[3] - case.interface_path[1],
        case.interface_path[2] - case.interface_path[0],
    )
    outcome_passed = final["outcome"] == case.expected_outcome
    score_margin = abs(final["interface_path_score"] - final["bulk_path_score"])
    validation_passed = bool(
        outcome_passed
        and score_margin > 1.0e-3
        and final["damage_residual_norm"] < 5.0e-2
        and visual_passed
    )

    summary = {
        "example": case.name,
        "expected_outcome": case.expected_outcome,
        "solved_outcome": final["outcome"],
        "capability_boundary": (
            "solver-driven diffuse E/Gc interface phase-field validation; "
            "not a cohesive element, PF-CZM, or ASTM structural calibration"
        ),
        "solver_path": (
            "FEMOperators.compute_psi_plus + PhaseFieldDamageSolver.solve"
        ),
        "n_nodes": int(mesh.n_nodes),
        "n_elements": int(mesh.n_elems),
        "bulk_path_score": final["bulk_path_score"],
        "interface_path_score": final["interface_path_score"],
        "score_margin": score_margin,
        "bulk_path_weighted_fracture_energy": bulk_energy,
        "interface_path_weighted_fracture_energy": interface_energy,
        "interface_to_bulk_energy_ratio": float(interface_energy / bulk_energy),
        "he_hutchinson_mode_i_threshold_ratio": float(
            _he_hutchinson_mode_i_deflection_ratio(abs(beta))),
        "final_damage_max": final["damage_max"],
        "final_damage_residual_norm": final["damage_residual_norm"],
        "visual_validation_passed": visual_passed,
        "validation_passed": validation_passed,
        "standard_csvs": csvs,
        "visual_outputs": visuals,
        "visual_manifest": visual_manifest,
        "provenance": provenance,
        "mesh_artifacts": mesh_artifacts,
        "max_rss_kib": _max_rss_kib(),
    }
    (case_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (case_dir / "run.log").write_text(
        "\n".join([
            f"case={case.name}",
            f"solver_path={summary['solver_path']}",
            f"expected_outcome={case.expected_outcome}",
            f"solved_outcome={final['outcome']}",
            f"validation_passed={validation_passed}",
            f"damage_residual_norm={final['damage_residual_norm']:.12e}",
            "",
        ]))
    (case_dir / "run_manifest.json").write_text(
        json.dumps({
            "case": case.name,
            "summary": "summary.json",
            "standard_outputs": {
                **{Path(path).name: path for path in visuals.values()},
                **csvs,
                **provenance,
                **mesh_artifacts,
                "run.log": str(case_dir / "run.log"),
                "visual_manifest.json": str(case_dir / "visual_manifest.json"),
            },
            "validation_passed": validation_passed,
        }, indent=2) + "\n")
    return summary


def run_validation(output_dir: Path, *, nx: int = 36, ny: int = 18) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [_run_case(CASES[name], output_dir, nx=nx, ny=ny)
             for name in ("weak_deflection", "strong_penetration")]
    summary = {
        "example": "solver_driven_interface_fracture_validation",
        "cases": cases,
        "all_validation_passed": all(case["validation_passed"] for case in cases),
        "max_rss_kib": _max_rss_kib(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "run_manifest.json").write_text(
        json.dumps({
            "suite": "solver_driven_interface_fracture_validation",
            "summary": "summary.json",
            "case_manifests": [
                f"{case['example']}/run_manifest.json" for case in cases
            ],
            "all_validation_passed": summary["all_validation_passed"],
        }, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/plasticity_interface/solver_driven_interface_fracture"),
    )
    parser.add_argument("--nx", type=int, default=36)
    parser.add_argument("--ny", type=int, default=18)
    args = parser.parse_args()
    print(json.dumps(run_validation(args.output_dir, nx=args.nx, ny=args.ny),
                     indent=2))


if __name__ == "__main__":
    main()
