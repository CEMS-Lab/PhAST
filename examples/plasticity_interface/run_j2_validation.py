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
import resource
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from phast.material import Material
from phast.plasticity import J2Plasticity, J2State
from phast.plasticity.j2_vonmises import (
    _stress_dev_norm,
    _stress_deviator_voigt6,
)


SQRT_3_2 = math.sqrt(1.5)


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


def _write_visual_manifest(output_dir: Path, paths: list[Path]) -> list[dict]:
    manifest = []
    for path in paths:
        with Image.open(path) as img:
            width, height = img.size
        manifest.append({
            "path": path.name,
            "artifact_type": "image",
            "width_px": int(width),
            "height_px": int(height),
            "size_bytes": int(path.stat().st_size),
            "review_dimension_passed": bool(max(width, height) < 2000),
        })
    (output_dir / "visual_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    return manifest


def run_validation(output_dir: Path, *, n_load: int = 48,
                   n_unload: int = 18) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_hash = _write_config(output_dir, n_load=n_load, n_unload=n_unload)

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
    plastic_residuals = []
    first_yield_step = None
    for step, eps_xx in enumerate(strain_program, start=1):
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
        })

    csv_path = output_dir / "j2_stress_strain.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
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
    visual_manifest = _write_visual_manifest(output_dir, [fig_path])

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
            all(item["review_dimension_passed"] for item in visual_manifest)),
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
