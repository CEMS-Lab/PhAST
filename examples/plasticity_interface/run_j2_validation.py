"""Standalone J2 plasticity validation example.

This is a customer-facing validation of the current supported plasticity
boundary: the material-point return-mapping kernel. It is not a coupled
phase-field plasticity solve.
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
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from examples.plasticity_interface._promoted_result_utils import (
    merge_run_manifest_artifacts,
    write_csv_rows,
    write_zarr_trajectory,
)
from phast.material import Material
from phast.plasticity import J2Plasticity, J2State
from phast.plasticity.j2_vonmises import (
    _stress_dev_norm,
    _stress_deviator_voigt6,
)
from phast.visualization import write_visual_manifest


SQRT_3_2 = math.sqrt(1.5)


plt.style.use("default")
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


def _von_mises(stress: torch.Tensor) -> torch.Tensor:
    return SQRT_3_2 * _stress_dev_norm(_stress_deviator_voigt6(stress))


def _max_rss_kib() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes. Values above 100 million are
    # bytes for this smoke-scale example.
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


def _structured_bar_mesh(nx: int = 24, ny: int = 6) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, 1.0, nx + 1)
    ys = np.linspace(0.0, 0.2, ny + 1)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    nodes = np.column_stack([xx.ravel(), yy.ravel()])
    elements: list[list[int]] = []
    stride = nx + 1
    for iy in range(ny):
        for ix in range(nx):
            n00 = iy * stride + ix
            n10 = n00 + 1
            n01 = n00 + stride
            n11 = n01 + 1
            elements.append([n00, n10, n11])
            elements.append([n00, n11, n01])
    return nodes, np.asarray(elements, dtype=np.int64)


def _element_to_nodal(
    n_nodes: int, elements: np.ndarray, element_values: np.ndarray
) -> np.ndarray:
    nodal = np.zeros(n_nodes, dtype=np.float64)
    counts = np.zeros(n_nodes, dtype=np.float64)
    for elem, value in zip(elements, element_values):
        nodal[elem] += float(value)
        counts[elem] += 1.0
    return nodal / np.maximum(counts, 1.0)


def _image_manifest_item(path: Path, *, artifact_type: str = "image") -> dict:
    if artifact_type == "animation":
        return {
            "artifact_type": "animation",
            "path": path.name,
            "file": path.name,
            "visual_scope": "plasticity_interface_beta",
            "size_bytes": int(path.stat().st_size),
            "media_passed": bool(path.stat().st_size > 0),
        }
    with Image.open(path) as img:
        width, height = img.size
    return {
        "artifact_type": "image",
        "path": path.name,
        "file": path.name,
        "visual_scope": "plasticity_interface_beta",
        "width_px": int(width),
        "height_px": int(height),
        "size_bytes": int(path.stat().st_size),
        "review_dimension_passed": bool(width >= 800 and height >= 500),
    }


def _plot_field(
    path: Path,
    nodes: np.ndarray,
    elements: np.ndarray,
    values: np.ndarray,
    title: str,
    label: str,
    *,
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    tpc = ax.tripcolor(
        nodes[:, 0], nodes[:, 1], elements, values,
        shading="gouraud", cmap=cmap)
    ax.triplot(nodes[:, 0], nodes[:, 1], elements, color="0.75", lw=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.colorbar(tpc, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _render_field_frame(
    nodes: np.ndarray, elements: np.ndarray, values: np.ndarray, title: str
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.tripcolor(
        nodes[:, 0], nodes[:, 1], elements, values,
        shading="gouraud", cmap="plasma")
    ax.triplot(nodes[:, 0], nodes[:, 1], elements, color="0.78", lw=0.25)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


def _write_config(output_dir: Path, *, n_load: int, n_unload: int) -> str:
    text = "\n".join([
        "schema_version: 1",
        "example: standalone_j2_plasticity_validation",
        "source_contract: configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml",
        "script: examples/plasticity_interface/run_j2_validation.py",
        "parameters:",
        f"  n_load: {n_load}",
        f"  n_unload: {n_unload}",
        "visual_requirements: docs/visualisation_requirements.md",
        "outputs:",
        "  - summary.json",
        "  - j2_stress_strain.csv",
        "  - j2_stress_strain.png",
        "  - visual_manifest.json",
        "",
    ])
    (output_dir / "config.yaml").write_text(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_validation(output_dir: Path, *, n_load: int = 48,
                   n_unload: int = 18) -> dict:
    start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_hash = _write_config(output_dir, n_load=n_load, n_unload=n_unload)
    nodes, elements = _structured_bar_mesh()

    sigma_y0 = 250.0
    hardening = 5_000.0
    material = Material(
        E=210_000.0,
        nu=0.30,
        plasticity_model="j2_isotropic",
        yield_stress=sigma_y0,
        hardening_modulus=hardening,
        hardening_type="linear_iso",
        plane_stress=True,
    )
    kernel = J2Plasticity(material, plane_stress=True)
    state = J2State.zeros((1,), dtype=torch.float64)
    strain = torch.zeros((1, 6), dtype=torch.float64)

    max_strain = 4.0e-3
    strain_program = []
    for i in range(n_load):
        strain_program.append(max_strain * float(i + 1) / float(n_load))
    for i in range(n_unload):
        strain_program.append(max_strain * (1.0 - float(i + 1) / float(n_unload)))

    rows = []
    snapshots: list[tuple[int, dict[str, np.ndarray]]] = []
    frames: list[np.ndarray] = []
    n_elems = elements.shape[0]
    plastic_residuals = []
    first_yield_step = None
    for step, eps_xx in enumerate(strain_program, start=1):
        step_start = time.perf_counter()
        eps_p_eq_n = float(state.eps_p_eq[0].item())
        next_strain = strain.clone()
        next_strain[..., 0] = eps_xx
        stress, plastic_strain, eps_p_eq = kernel.step(
            strain,
            next_strain,
            state.stress,
            state.plastic_strain,
            state.eps_p_eq,
        )
        state = J2State(stress, plastic_strain, eps_p_eq)
        strain = next_strain

        vm = float(_von_mises(stress)[0].item())
        eqp = float(eps_p_eq[0].item())
        yield_current = sigma_y0 + hardening * eqp
        is_new_plastic_flow = eqp > eps_p_eq_n + 1.0e-12
        residual = vm - yield_current if is_new_plastic_flow else 0.0
        if is_new_plastic_flow:
            plastic_residuals.append(abs(residual))
            if first_yield_step is None:
                first_yield_step = step
        rows.append({
            "step": step,
            "eps_xx": float(eps_xx),
            "sigma_xx_mpa": float(stress[0, 0].item()),
            "sigma_yy_mpa": float(stress[0, 1].item()),
            "sigma_vm_mpa": vm,
            "eps_p_eq": eqp,
            "yield_stress_mpa": yield_current,
            "yield_residual_mpa": residual,
            "elapsed_s": time.perf_counter() - step_start,
        })
        elem_vm = np.full(n_elems, vm, dtype=np.float64)
        elem_eqp = np.full(n_elems, eqp, dtype=np.float64)
        nodal_eqp = _element_to_nodal(nodes.shape[0], elements, elem_eqp)
        disp = np.zeros((nodes.shape[0], 2), dtype=np.float64)
        disp[:, 0] = eps_xx * nodes[:, 0]
        stress_field = np.zeros((n_elems, 3), dtype=np.float64)
        stress_field[:, 0] = float(stress[0, 0].item())
        stress_field[:, 1] = float(stress[0, 1].item())
        if step in {1, len(strain_program)} or step % max(1, len(strain_program) // 8) == 0:
            snapshots.append((
                step,
                {
                    "displacement": disp,
                    "stress": stress_field,
                    "von_mises": elem_vm,
                    "plastic_strain": nodal_eqp,
                },
            ))
            frames.append(_render_field_frame(
                nodes, elements, nodal_eqp,
                f"J2 equivalent plastic strain, step {step}"))

    csv_path = output_dir / "j2_stress_strain.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot([r["eps_xx"] for r in rows],
            [r["sigma_xx_mpa"] for r in rows],
            marker="o", markersize=2.8, linewidth=1.5,
            label="sigma_xx")
    ax.plot([r["eps_xx"] for r in rows],
            [r["sigma_vm_mpa"] for r in rows],
            linewidth=1.5, label="von Mises")
    ax.set_xlabel("axial strain eps_xx")
    ax.set_ylabel("stress [MPa]")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    fig_path = output_dir / "j2_stress_strain.png"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)
    (output_dir / "stress_strain.png").write_bytes(fig_path.read_bytes())

    final_eqp = _element_to_nodal(
        nodes.shape[0], elements,
        np.full(elements.shape[0], rows[-1]["eps_p_eq"], dtype=np.float64))
    final_vm = np.full(elements.shape[0], rows[-1]["sigma_vm_mpa"], dtype=np.float64)
    initial_png = output_dir / "initial_conditions.png"
    _plot_field(
        initial_png, nodes, elements, np.zeros(nodes.shape[0]),
        "J2 validation initial equivalent plastic strain",
        "equivalent plastic strain", cmap="plasma")
    eqp_png = output_dir / "equivalent_plastic_strain.png"
    _plot_field(
        eqp_png, nodes, elements, final_eqp,
        "Final equivalent plastic strain", "eps_p_eq", cmap="plasma")
    vm_png = output_dir / "von_mises.png"
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    tpc = ax.tripcolor(
        nodes[:, 0], nodes[:, 1], elements, facecolors=final_vm,
        edgecolors="0.75", linewidth=0.25, cmap="viridis")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_title("Final von Mises stress")
    fig.colorbar(tpc, ax=ax, label="MPa")
    fig.tight_layout()
    fig.savefig(vm_png, dpi=170)
    plt.close(fig)
    plastic_work_png = output_dir / "plastic_work.png"
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot([r["step"] for r in rows], [r["eps_p_eq"] * hardening for r in rows], "o-")
    ax.set_xlabel("step")
    ax.set_ylabel("plastic work proxy [MPa]")
    ax.set_title("J2 plastic work proxy")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(plastic_work_png, dpi=170)
    plt.close(fig)
    mesh_png = output_dir / "mesh_deformed.png"
    deformed = nodes.copy()
    deformed[:, 0] += rows[-1]["eps_xx"] * nodes[:, 0] * 10.0
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.triplot(nodes[:, 0], nodes[:, 1], elements, color="0.75", lw=0.3, label="reference")
    ax.triplot(deformed[:, 0], deformed[:, 1], elements, color="tab:red", lw=0.45, label="deformed x10")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("J2 bar mesh and amplified final deformation")
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(mesh_png, dpi=170)
    plt.close(fig)
    anim_path = output_dir / "field_evolution.gif"
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(anim_path, save_all=True, append_images=pil_frames[1:], duration=180, loop=0)
    visual_manifest = [
        _image_manifest_item(initial_png),
        _image_manifest_item(eqp_png),
        _image_manifest_item(vm_png),
        _image_manifest_item(output_dir / "stress_strain.png"),
        _image_manifest_item(plastic_work_png),
        _image_manifest_item(mesh_png),
        _image_manifest_item(anim_path, artifact_type="animation"),
    ]
    (output_dir / "visual_manifest.json").write_text(json.dumps(visual_manifest, indent=2) + "\n")

    history_rows = [
        {
            "step": r["step"],
            "load": r["eps_xx"],
            "reaction_force": r["sigma_xx_mpa"],
            "max_equivalent_plastic_strain": r["eps_p_eq"],
            "max_von_mises": r["sigma_vm_mpa"],
        }
        for r in rows
    ]
    write_csv_rows(output_dir / "history.csv", history_rows)
    write_csv_rows(output_dir / "results.csv", history_rows)
    write_csv_rows(output_dir / "solver_telemetry.csv", [
        {
            "step": r["step"],
            "yield_residual_mpa": r["yield_residual_mpa"],
            "plastic_step": bool(r["eps_p_eq"] > 1.0e-12),
        }
        for r in rows
    ])
    write_csv_rows(output_dir / "timing_per_step.csv", [
        {"step": r["step"], "elapsed_s": r["elapsed_s"]}
        for r in rows
    ])
    write_zarr_trajectory(
        output_dir,
        nodes=nodes,
        elements=elements,
        snapshots=snapshots,
        metadata={"validation_id": "j2_validation", "field_source": "vectorized_j2_kernel"},
    )
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "example": "plasticity_interface.j2_validation",
        "validation_id": "j2_validation",
        "timestamp_utc": now,
        "git_sha": _git_sha(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mesh": {"n_nodes": int(nodes.shape[0]), "n_elements": int(elements.shape[0])},
        "elapsed_ms": 1_000.0 * (time.perf_counter() - start),
        "max_rss_kib": _max_rss_kib(),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output_dir / "run_lockfile.json").write_text(json.dumps({
        "schema": "phast_run_lockfile_v1",
        "created_utc": now,
        "git_sha": metadata["git_sha"],
        "config_sha256": config_hash,
        "deterministic": True,
    }, indent=2) + "\n")
    (output_dir / "run.log").write_text(
        f"{now} J2 validation completed with {len(rows)} load steps.\n")
    merge_run_manifest_artifacts(output_dir, [
        "summary.json", "config.yaml", "run_lockfile.json",
        "run_metadata.json", "run_manifest.json", "run.log",
        "training_data.zarr", "j2_stress_strain.csv", "j2_stress_strain.png",
        "stress_strain.png", "results.csv", "history.csv",
        "solver_telemetry.csv", "timing_per_step.csv",
        "initial_conditions.png", "equivalent_plastic_strain.png",
        "von_mises.png", "plastic_work.png", "mesh_deformed.png",
        "field_evolution.gif", "visual_manifest.json",
    ])

    summary = {
        "example": "standalone_j2_plasticity_validation",
        "capability_boundary": (
            "standalone material-point return mapping; not coupled PF-plasticity"
        ),
        "sigma_y0_mpa": sigma_y0,
        "hardening_modulus_mpa": hardening,
        "n_steps": len(rows),
        "n_plastic_steps": sum(1 for r in rows if r["eps_p_eq"] > 1.0e-12),
        "first_yield_step": first_yield_step,
        "max_abs_yield_residual_mpa": max(plastic_residuals) if plastic_residuals else 0.0,
        "csv": str(csv_path),
        "plot": str(fig_path),
        "config": str(output_dir / "config.yaml"),
        "config_sha256": config_hash,
        "visual_manifest": str(output_dir / "visual_manifest.json"),
        "visual_manifest_passed": bool(
            all(item.get("review_dimension_passed", item.get("media_passed", True))
                for item in visual_manifest)),
        "validation_passed": bool(
            sum(1 for r in rows if r["eps_p_eq"] > 1.0e-12) > 0
            and (max(plastic_residuals) if plastic_residuals else 0.0) < 1.0e-5
            and all(item.get("review_dimension_passed", item.get("media_passed", True))
                    for item in visual_manifest)
        ),
        "max_rss_kib": _max_rss_kib(),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/plasticity_interface/j2_validation"),
    )
    args = parser.parse_args()
    summary = run_validation(args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
