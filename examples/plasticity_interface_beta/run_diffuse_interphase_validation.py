"""Diffuse interphase phase-field validation example.

The example validates the current supported interface/interphase boundary:
spatial ``E(x)`` and ``Gc(x)`` fields generated for a brittle bulk phase-field
model. It is not a cohesive displacement-jump interface formulation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from phast.dataset_benchmark.generators import InterfaceFractureGenerator
from phast.visualization import write_visual_manifest


plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


def _centroids(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    return nodes[elements].mean(axis=1)


def _segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if not np.isfinite(denom) or denom <= 1.0e-30:
        return np.linalg.norm(points - a, axis=1)
    numer = np.einsum("ij,j->i", points - a, ab)
    t = np.clip(numer / denom, 0.0, 1.0)
    projection = a + t[:, None] * ab
    return np.linalg.norm(points - projection, axis=1)


def _crack_density(distance: np.ndarray, l0: float) -> np.ndarray:
    # A normalized Gaussian process-zone proxy. The absolute normalization is
    # less important than using the same density for all candidate paths.
    return np.exp(-0.5 * (distance / l0) ** 2)


def _weighted_fracture_energy(points: np.ndarray, Gc: np.ndarray,
                              area: np.ndarray, l0: float,
                              start: np.ndarray, end: np.ndarray) -> float:
    density = _crack_density(_segment_distance(points, start, end), l0)
    return float(np.sum(Gc * density * area))


def _triangle_areas(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    tri = nodes[elements]
    v1 = tri[:, 1, :] - tri[:, 0, :]
    v2 = tri[:, 2, :] - tri[:, 0, :]
    return 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])


def _max_rss_kib() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes. Values above 100 million are
    # bytes for this reduced-size example.
    return raw // 1024 if raw > 100_000_000 else raw


def _write_config(output_dir: Path, *, seed: int, max_nodes: int) -> str:
    text = "\n".join([
        "schema_version: 1",
        "example: diffuse_interphase_phase_field_validation",
        "source_contract: configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml",
        "script: examples/plasticity_interface_beta/run_diffuse_interphase_validation.py",
        "parameters:",
        f"  seed: {seed}",
        f"  max_nodes: {max_nodes}",
        "visual_requirements: docs/user_guide/example_contract.md",
        "outputs:",
        "  - summary.json",
        "  - diffuse_interphase_path_energy.csv",
        "  - diffuse_interphase_field_summary.csv",
        "  - diffuse_interphase_fields.png",
        "  - visual_manifest.json",
        "",
    ])
    (output_dir / "config.yaml").write_text(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_validation(output_dir: Path, *, seed: int = 11,
                   max_nodes: int = 1_800) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_hash = _write_config(output_dir, seed=seed, max_nodes=max_nodes)

    gen = InterfaceFractureGenerator(
        W_range=(48.0, 48.0),
        H_range=(24.0, 24.0),
        a_frac_range=(0.45, 0.45),
        E_range=(30_000.0, 30_000.0),
        nu_range=(0.24, 0.24),
        Gc_range=(3.0e-3, 3.0e-3),
        l0_range=(0.9, 0.9),
        rho_range=(2.4e-9, 2.4e-9),
        n_interfaces_choices=(1,),
        n_interfaces_weights=(1.0,),
        h_i_elem_widths_range=(3.0, 3.0),
        alpha_Gc_range=(0.65, 0.65),
        E_contrast_prob=1.0,
        E_ratio_range=(1.8, 1.8),
        nx_per_l0=1.2,
        ny_per_l0=1.2,
        max_nodes=max_nodes,
    )
    out = gen.generate(seed=seed)

    nodes = np.asarray(out.mesh["nodes"], dtype=np.float64)
    elements = np.asarray(out.mesh["elements"], dtype=np.int64)
    points = _centroids(nodes, elements)
    areas = _triangle_areas(nodes, elements)
    E = np.asarray(out.material["E"], dtype=np.float64)
    Gc = np.asarray(out.material["Gc"], dtype=np.float64)
    params = out.bcs["_params"]
    l0 = float(params["l0"])
    W = float(params["W"])
    H = float(params["H"])
    segment = np.asarray(out.bcs["_class_meta"]["interface_segments"][0],
                         dtype=np.float64)
    interface_start = segment[:2]
    interface_end = segment[2:]
    bulk_start = np.array([float(params["a"]), 0.5 * H], dtype=np.float64)
    bulk_end = np.array([W, 0.5 * H], dtype=np.float64)

    interface_energy = _weighted_fracture_energy(
        points, Gc, areas, l0, interface_start, interface_end)
    bulk_energy = _weighted_fracture_energy(
        points, Gc, areas, l0, bulk_start, bulk_end)
    energy_ratio = interface_energy / bulk_energy

    rows = [
        {
            "path": "bulk_horizontal_from_precrack",
            "start_x": bulk_start[0],
            "start_y": bulk_start[1],
            "end_x": bulk_end[0],
            "end_y": bulk_end[1],
            "weighted_fracture_energy": bulk_energy,
        },
        {
            "path": "interface_following",
            "start_x": interface_start[0],
            "start_y": interface_start[1],
            "end_x": interface_end[0],
            "end_y": interface_end[1],
            "weighted_fracture_energy": interface_energy,
        },
    ]
    csv_path = output_dir / "diffuse_interphase_path_energy.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    field_csv = output_dir / "diffuse_interphase_field_summary.csv"
    with field_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["field", "min", "mean", "max", "contrast"])
        writer.writeheader()
        for name, arr in (("E", E), ("Gc", Gc)):
            writer.writerow({
                "field": name,
                "min": float(arr.min()),
                "mean": float(arr.mean()),
                "max": float(arr.max()),
                "contrast": float(arr.max() / arr.min()),
            })

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), constrained_layout=True)
    e_plot = axes[0].tripcolor(
        nodes[:, 0], nodes[:, 1], elements, facecolors=E, shading="flat")
    axes[0].plot([interface_start[0], interface_end[0]],
                 [interface_start[1], interface_end[1]], "k-", linewidth=1.5)
    axes[0].set_title("E(x) diffuse contrast")
    fig.colorbar(e_plot, ax=axes[0], shrink=0.85)

    gc_plot = axes[1].tripcolor(
        nodes[:, 0], nodes[:, 1], elements, facecolors=Gc, shading="flat")
    axes[1].plot([bulk_start[0], bulk_end[0]],
                 [bulk_start[1], bulk_end[1]], color="white", linewidth=1.4,
                 label="bulk path")
    axes[1].plot([interface_start[0], interface_end[0]],
                 [interface_start[1], interface_end[1]], "k-", linewidth=1.5,
                 label="interface path")
    axes[1].set_title("Gc(x) interphase reduction")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.colorbar(gc_plot, ax=axes[1], shrink=0.85)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig_path = output_dir / "diffuse_interphase_fields.png"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)
    visual_manifest = write_visual_manifest(
        output_dir, [fig_path.name], visual_scope="plasticity_interface_beta")

    summary = {
        "example": "diffuse_interphase_phase_field_validation",
        "capability_boundary": (
            "bulk brittle phase-field with spatial E/Gc fields; not a cohesive "
            "displacement-jump interface law"
        ),
        "seed": int(seed),
        "n_nodes": int(nodes.shape[0]),
        "n_elements": int(elements.shape[0]),
        "l0": l0,
        "interface_segment": [float(x) for x in segment],
        "Gc_bulk": float(params["Gc_bulk"]),
        "Gc_min": float(Gc.min()),
        "Gc_reduction_fraction_observed": float(1.0 - Gc.min() / params["Gc_bulk"]),
        "E_left": float(params["E_left"]),
        "E_right": float(params["E_right"]),
        "E_contrast_observed": float(E.max() / E.min()),
        "bulk_path_weighted_fracture_energy": bulk_energy,
        "interface_path_weighted_fracture_energy": interface_energy,
        "interface_to_bulk_energy_ratio": float(energy_ratio),
        "csv": str(csv_path),
        "field_csv": str(field_csv),
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
        default=Path("outputs/plasticity_interface/diffuse_interphase"),
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-nodes", type=int, default=1_800)
    args = parser.parse_args()
    summary = run_validation(
        args.output_dir, seed=args.seed, max_nodes=args.max_nodes)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
