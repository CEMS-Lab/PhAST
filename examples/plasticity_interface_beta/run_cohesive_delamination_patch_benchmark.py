"""Multi-element cohesive delamination patch benchmark.

This benchmark exercises a short zero-thickness cohesive interface with four
cohesive segments and a tapered mixed-mode displacement profile. It is still a
deterministic prescribed-displacement validation, but unlike the single-edge
mode-I/contact validation cases it checks multi-element residual assembly, localized
damage, delamination-front metrics, and tangent consistency across a small
interface patch.
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


def _strip_mesh(n_segments: int = 4) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    xs = np.linspace(0.0, float(n_segments), n_segments + 1)
    interface = np.column_stack([xs, np.zeros_like(xs)])
    top = np.column_stack([xs, np.ones_like(xs)])
    bottom = np.column_stack([xs, -np.ones_like(xs)])
    nodes = np.vstack([interface, top, bottom]).astype(np.float64)
    top_offset = n_segments + 1
    bottom_offset = 2 * (n_segments + 1)
    elements: list[list[int]] = []
    for i in range(n_segments):
        j = i + 1
        ti = top_offset + i
        tj = top_offset + j
        bi = bottom_offset + i
        bj = bottom_offset + j
        elements.append([i, j, tj])
        elements.append([i, tj, ti])
        elements.append([bi, j, i])
        elements.append([bi, bj, j])
    interface_edges = [(i, i + 1) for i in range(n_segments)]
    return nodes, np.asarray(elements, dtype=np.int64), interface_edges


def _profile(xs: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    xi = xs / xs.max()
    taper = np.clip(1.0 - xi, 0.0, 1.0)
    normal = scale * 0.060 * taper ** 1.3
    shear = scale * 0.035 * taper ** 1.1
    return normal, shear


def _expected_q_response(
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


def _expected_resultants(
    cohesives,
    normal_nodes: np.ndarray,
    shear_nodes: np.ndarray,
    law: BilinearCohesiveLaw,
    history: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    if any(max(ce.nodes_top) >= len(normal_nodes) for ce in cohesives):
        raise ValueError(
            "cohesive top-node ids must index the prescribed interface profile"
        )
    gauss = np.asarray([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    weights = np.ones(2)
    damage = np.zeros_like(history)
    new_history = history.copy()
    normal_resultant = 0.0
    shear_resultant = 0.0
    for e, ce in enumerate(cohesives):
        n0, n1 = ce.nodes_top
        for q, xi in enumerate(gauss):
            N0 = 0.5 * (1.0 - xi)
            N1 = 0.5 * (1.0 + xi)
            dn = float(N0 * normal_nodes[n0] + N1 * normal_nodes[n1])
            dt = float(N0 * shear_nodes[n0] + N1 * shear_nodes[n1])
            tn, tt, d, h = _expected_q_response(
                dn, dt, law=law, max_delta_eff=float(history[e, q]))
            damage[e, q] = d
            new_history[e, q] = h
            factor = 0.5 * ce.length * weights[q]
            normal_resultant += tn * factor
            shear_resultant += tt * factor
    return normal_resultant, shear_resultant, damage, new_history


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


def _write_config(output_dir: Path, scales: list[float],
                  law: BilinearCohesiveLaw) -> str:
    text = "\n".join([
        "case: cohesive_delamination_patch_benchmark",
        "model: multi_element_zero_thickness_mixed_mode_cohesive_patch",
        "solver: QuasiStaticSolver(cohesive_operator=CohesiveInterfaceOperator)",
        "mesh: two_strip_triangular_patch_with_four_inserted_cohesive_edges",
        "boundary_condition: fully_prescribed_tapered_mixed_mode_profile",
        "device: cpu",
        "dtype: float64",
        "cohesive_law:",
        f"  k_n: {law.k_n}",
        f"  k_t: {law.k_t}",
        f"  sigma_max: {law.sigma_max}",
        f"  delta_c: {law.delta_c}",
        f"  contact_stiffness: {law.contact_stiffness}",
        "load_scales:",
        *[f"  - {scale}" for scale in scales],
        "",
    ])
    (output_dir / "config.yaml").write_text(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_standard_files(
    output_dir: Path,
    *,
    config_hash: str,
    elapsed_ms: float,
    artifacts: list[Path],
    n_steps: int,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "benchmark": "cohesive_delamination_patch_benchmark",
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
            "artifacts": [path.name for path in artifacts],
        }, indent=2) + "\n")
    (output_dir / "run.log").write_text(
        "\n".join([
            f"{now} cohesive delamination patch benchmark started",
            f"{now} solver path: QuasiStaticSolver cohesive_operator hook",
            f"{now} completed {n_steps} deterministic load steps",
            f"{now} elapsed_ms={elapsed_ms:.3f}",
            "",
        ]))


def _build_solver(law: BilinearCohesiveLaw) -> tuple[
    QuasiStaticSolver, CohesiveInterfaceOperator, FEMMesh, np.ndarray
]:
    nodes, elements, interface_edges = _strip_mesh()
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
        fem, cohesive_operator=op, backend="auto", tol=1.0e-11,
        max_iter=6, line_search=False)
    return solver, op, mesh, new_nodes


def _segment_center_x(ce, nodes: np.ndarray) -> float:
    n0, n1 = ce.nodes_top
    return float(0.5 * (nodes[n0, 0] + nodes[n1, 0]))


def _segment_tip_x(ce, nodes: np.ndarray) -> float:
    n0, n1 = ce.nodes_top
    return float(max(nodes[n0, 0], nodes[n1, 0]))


def _tangent_fd_error(op: CohesiveInterfaceOperator, u: torch.Tensor) -> float:
    du = torch.zeros_like(u)
    taper = torch.linspace(1.0, 0.2, 5, device=u.device, dtype=u.dtype)
    du[0:5, 0] = 0.001 * taper
    du[0:5, 1] = 0.0015 * taper
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
        contact_stiffness=2_500.0,
    )
    scales = [0.0, 0.25, 0.50, 0.75, 1.00]
    config_hash = _write_config(output_dir, scales, law)
    solver, op, mesh, nodes = _build_solver(law)
    xs = nodes[:5, 0]
    history = np.zeros((len(op.cohesives), 2), dtype=np.float64)
    d = torch.zeros(mesh.n_nodes, dtype=torch.float64)
    f_ext = torch.zeros((mesh.n_nodes, 2), dtype=torch.float64)
    bc_mask = torch.ones((mesh.n_nodes, 2), dtype=torch.bool)
    rows: list[dict] = []
    final_damage_profile = None

    for step, scale in enumerate(scales):
        normal_nodes, shear_nodes = _profile(xs, scale)
        bc_vals = torch.zeros((mesh.n_nodes, 2), dtype=torch.float64)
        bc_vals[:5, 0] = torch.as_tensor(shear_nodes, dtype=torch.float64)
        bc_vals[:5, 1] = torch.as_tensor(normal_nodes, dtype=torch.float64)
        u, converged, n_iter = solver.solve(d, f_ext, bc_mask, bc_vals)
        tangent_error = 0.0 if scale == 0.0 else _tangent_fd_error(op, u)
        f_coh = op.internal_force(u, state=op.state)
        if op._trial_state is not None:
            op.commit()
        exp_n, exp_t, exp_damage, history = _expected_resultants(
            op.cohesives, normal_nodes, shear_nodes, law, history)
        measured_n = float(f_coh[:5, 1].sum().item())
        measured_t = float(f_coh[:5, 0].sum().item())
        damage_q = op.state.damage.detach().cpu().numpy()
        final_damage_profile = damage_q.mean(axis=1)
        active_segments = int(np.count_nonzero(final_damage_profile > 0.05))
        front_x = 0.0
        if active_segments:
            damaged = np.nonzero(final_damage_profile > 0.05)[0]
            front_x = max(_segment_tip_x(op.cohesives[i], nodes) for i in damaged)
        rows.append({
            "step": step,
            "load_scale": float(scale),
            "converged": bool(converged),
            "newton_iterations": int(n_iter),
            "normal_resultant": measured_n,
            "expected_normal_resultant": exp_n,
            "abs_normal_error": abs(measured_n - exp_n),
            "shear_resultant": measured_t,
            "expected_shear_resultant": exp_t,
            "abs_shear_error": abs(measured_t - exp_t),
            "damage_max": float(damage_q.max()),
            "damage_mean": float(damage_q.mean()),
            "active_segments": active_segments,
            "delamination_front_x": front_x,
            "tangent_fd_error": tangent_error,
            "residual_norm": float(solver.last_residual),
        })

    csv_path = output_dir / "cohesive_delamination_patch_response.csv"
    _write_csv(csv_path, rows)
    damage_profile = np.asarray(final_damage_profile, dtype=np.float64)
    seg_x = np.asarray([
        _segment_center_x(ce, nodes) for ce in op.cohesives
    ], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), constrained_layout=True)
    load = np.asarray([row["load_scale"] for row in rows])
    axes[0].plot(load, [row["normal_resultant"] for row in rows], "o-", label="normal")
    axes[0].plot(load, [row["shear_resultant"] for row in rows], "s-", label="shear")
    axes[0].set_xlabel("load scale")
    axes[0].set_ylabel("interface resultant")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(load, [row["damage_max"] for row in rows], "^-", label="max")
    axes[1].plot(load, [row["damage_mean"] for row in rows], "o-", label="mean")
    axes[1].set_xlabel("load scale")
    axes[1].set_ylabel("cohesive damage")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    response_png = output_dir / "cohesive_delamination_patch_response.png"
    fig.savefig(response_png, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    ax.plot(seg_x, damage_profile, "o-", color="tab:red")
    ax.set_xlabel("interface segment center x")
    ax.set_ylabel("final mean cohesive damage")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    damage_png = output_dir / "cohesive_delamination_patch_damage_profile.png"
    fig.savefig(damage_png, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    ax.triplot(nodes[:, 0], nodes[:, 1], mesh.elements.cpu().numpy(), color="0.70")
    ax.plot(xs, np.zeros_like(xs), color="tab:red", linewidth=3.0,
            label="cohesive patch")
    normal_nodes, shear_nodes = _profile(xs, scales[-1])
    ax.quiver(xs, 0.08 * np.ones_like(xs), shear_nodes, normal_nodes,
              angles="xy", scale_units="xy", scale=1.0, color="tab:blue",
              width=0.006, label="tapered mixed-mode jump")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8, loc="upper right")
    mesh_png = output_dir / "cohesive_delamination_patch_mesh_and_bc.png"
    fig.savefig(mesh_png, dpi=160)
    plt.close(fig)

    visual_manifest = _write_visual_manifest(
        output_dir, [response_png, damage_png, mesh_png])
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
        damage_png,
        mesh_png,
        output_dir / "visual_manifest.json",
    ]
    _write_standard_files(
        output_dir, config_hash=config_hash, elapsed_ms=elapsed_ms,
        artifacts=artifacts, n_steps=len(rows))

    max_normal_error = max(row["abs_normal_error"] for row in rows)
    max_shear_error = max(row["abs_shear_error"] for row in rows)
    max_tangent_error = max(row["tangent_fd_error"] for row in rows)
    summary = {
        "example": "cohesive_delamination_patch_benchmark",
        "capability": (
            "multi-element mixed-mode cohesive delamination patch through "
            "QuasiStaticSolver cohesive_operator hook"
        ),
        "n_nodes": int(mesh.n_nodes),
        "n_elements": int(mesh.elements.shape[0]),
        "n_cohesive_elements": int(len(op.cohesives)),
        "n_load_steps": int(len(rows)),
        "all_steps_converged": all(row["converged"] for row in rows),
        "final_damage_max": float(rows[-1]["damage_max"]),
        "final_damage_mean": float(rows[-1]["damage_mean"]),
        "final_active_segments": int(rows[-1]["active_segments"]),
        "final_delamination_front_x": float(rows[-1]["delamination_front_x"]),
        "max_abs_normal_resultant_error": float(max_normal_error),
        "max_abs_shear_resultant_error": float(max_shear_error),
        "max_tangent_fd_error": float(max_tangent_error),
        "validation_passed": bool(
            all(row["converged"] for row in rows)
            and max_normal_error < 1.0e-10
            and max_shear_error < 1.0e-10
            and max_tangent_error < 1.0e-4
            and rows[-1]["damage_max"] > 0.95
            and 1 <= rows[-1]["active_segments"] < len(op.cohesives)
            and all(item["review_dimension_passed"] for item in visual_manifest)
        ),
        "csv": csv_path.name,
        "plots": [response_png.name, damage_png.name, mesh_png.name],
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
        default=Path("outputs/plasticity_interface/cohesive_delamination_patch"),
    )
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
