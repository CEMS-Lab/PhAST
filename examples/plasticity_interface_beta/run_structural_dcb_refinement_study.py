"""Structural DCB cohesive refinement study.

This runner compares the promoted DCB-style cohesive validation case against a
finer mesh/load-step variant. It is a public promotion aid: it demonstrates
that the cohesive result is not a single accidental run, while keeping the
claim bounded to a lightweight refinement trend rather than ASTM D5528
calibration or a mesh-converged engineering benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from examples.plasticity_interface_beta.run_structural_dcb_cohesive_benchmark import (
    run_benchmark,
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


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_response(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        {
            "opening": float(row["opening"]),
            "opening_force": float(row["opening_force"]),
            "energy_balance_gap": float(row["energy_balance_gap"]),
            "delamination_front_x": float(row["delamination_front_x"]),
        }
        for row in rows
    ]


def _manifest_item(path: Path) -> dict:
    with Image.open(path) as img:
        width, height = img.size
    return {
        "artifact_type": "image",
        "path": path.name,
        "width_px": int(width),
        "height_px": int(height),
        "size_bytes": int(path.stat().st_size),
        "review_dimension_passed": bool(width >= 800 and height >= 500),
        "visual_scope": "plasticity_interface_beta",
    }


def _write_plots(output_dir: Path, cases: list[dict]) -> list[dict]:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for case in cases:
        rows = _read_response(output_dir / case["case_dir"] / "structural_dcb_response.csv")
        ax.plot(
            [row["opening"] for row in rows],
            [row["energy_balance_gap"] for row in rows],
            marker="o",
            label=case["label"],
        )
    ax.set_xlabel("crack-mouth opening")
    ax.set_ylabel("absolute energy-balance gap")
    ax.set_title("DCB cohesive refinement: energy-balance diagnostic")
    ax.grid(True, alpha=0.3)
    ax.legend()
    gap_png = output_dir / "dcb_refinement_energy_gap.png"
    fig.savefig(gap_png, dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for case in cases:
        rows = _read_response(output_dir / case["case_dir"] / "structural_dcb_response.csv")
        ax.plot(
            [row["opening"] for row in rows],
            [row["opening_force"] for row in rows],
            marker="o",
            label=case["label"],
        )
    ax.set_xlabel("crack-mouth opening")
    ax.set_ylabel("opening force")
    ax.set_title("DCB cohesive refinement: load-displacement response")
    ax.grid(True, alpha=0.3)
    ax.legend()
    response_png = output_dir / "dcb_refinement_load_displacement.png"
    fig.savefig(response_png, dpi=170)
    plt.close(fig)

    manifest = [_manifest_item(gap_png), _manifest_item(response_png)]
    (output_dir / "visual_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    return manifest


def _write_standard_files(output_dir: Path, cases: list[dict], elapsed_ms: float) -> None:
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "benchmark": "structural_dcb_refinement_study",
        "timestamp_utc": now,
        "git_sha": _git_sha(),
        "platform": platform.platform(),
        "python": platform.python_version(),
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
            "deterministic": True,
            "cases": [
                {
                    "label": case["label"],
                    "nx": case["nx"],
                    "n_load_steps": case["n_load_steps"],
                    "case_dir": case["case_dir"],
                }
                for case in cases
            ],
        }, indent=2) + "\n")
    (output_dir / "config.yaml").write_text(
        "\n".join([
            "case: structural_dcb_refinement_study",
            "model: clamped_end_dcb_style_mode_I_cohesive_delamination",
            "capability_boundary: lightweight refinement trend, not ASTM D5528 calibration",
            "cases:",
            *[
                f"  - label: {case['label']}\n"
                f"    nx: {case['nx']}\n"
                f"    n_load_steps: {case['n_load_steps']}\n"
                f"    case_dir: {case['case_dir']}"
                for case in cases
            ],
            "",
        ])
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps({
            "schema": "phast_run_manifest_v1",
            "benchmark": metadata["benchmark"],
            "artifacts": [
                "summary.json",
                "config.yaml",
                "run_lockfile.json",
                "run_metadata.json",
                "run_manifest.json",
                "run.log",
                "dcb_refinement_summary.csv",
                "dcb_refinement_energy_gap.png",
                "dcb_refinement_load_displacement.png",
                "visual_manifest.json",
            ],
            "case_manifests": [
                f"{case['case_dir']}/run_manifest.json" for case in cases
            ],
        }, indent=2) + "\n")
    (output_dir / "run.log").write_text(
        "\n".join([
            f"{now} structural DCB refinement study started",
            f"{now} completed {len(cases)} cases",
            f"{now} elapsed_ms={elapsed_ms:.3f}",
            "",
        ]))


def run_study(output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    cases = [
        {
            "label": "baseline nx=16, 11 steps",
            "case_dir": "baseline",
            "nx": 16,
            "n_load_steps": 11,
        },
        {
            "label": "refined nx=32, 41 steps",
            "case_dir": "refined",
            "nx": 32,
            "n_load_steps": 41,
        },
    ]

    summaries = []
    for case in cases:
        summary = run_benchmark(
            output_dir / case["case_dir"],
            nx=case["nx"],
            n_load_steps=case["n_load_steps"],
        )
        summaries.append(summary)

    rows = []
    for case, summary in zip(cases, summaries):
        rows.append({
            "label": case["label"],
            "case_dir": case["case_dir"],
            "nx": case["nx"],
            "n_load_steps": case["n_load_steps"],
            "n_nodes": summary["n_nodes"],
            "n_elements": summary["n_elements"],
            "n_cohesive_elements": summary["n_cohesive_elements"],
            "max_residual_norm": summary["max_residual_norm"],
            "max_energy_balance_gap_fraction": summary[
                "max_energy_balance_gap_fraction"
            ],
            "front_advanced": summary["front_advanced"],
            "post_peak_softening": summary["post_peak_softening"],
            "validation_passed": summary["validation_passed"],
        })
    _write_csv(output_dir / "dcb_refinement_summary.csv", rows)
    visual_manifest = _write_plots(output_dir, cases)
    elapsed_ms = 1_000.0 * (time.perf_counter() - start)
    _write_standard_files(output_dir, cases, elapsed_ms)

    baseline, refined = summaries
    visual_passed = all(row["review_dimension_passed"] for row in visual_manifest)
    validation_passed = bool(
        baseline["validation_passed"]
        and refined["validation_passed"]
        and refined["max_energy_balance_gap_fraction"]
        < baseline["max_energy_balance_gap_fraction"]
        and refined["all_steps_converged"]
        and refined["front_advanced"]
        and visual_passed
    )
    summary = {
        "example": "structural_dcb_refinement_study",
        "validation_passed": validation_passed,
        "capability_boundary": (
            "Lightweight DCB cohesive refinement trend; not a mesh-converged "
            "ASTM D5528 validation or material-property data-reduction."
        ),
        "n_cases": len(cases),
        "baseline_case": baseline,
        "refined_case": refined,
        "gap_fraction_reduction": float(
            baseline["max_energy_balance_gap_fraction"]
            - refined["max_energy_balance_gap_fraction"]
        ),
        "summary_csv": "dcb_refinement_summary.csv",
        "plots": [
            "dcb_refinement_energy_gap.png",
            "dcb_refinement_load_displacement.png",
        ],
        "visual_manifest": visual_manifest,
        "visual_manifest_passed": bool(visual_passed),
        "max_rss_kib": _max_rss_kib(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/plasticity_interface/structural_dcb_refinement"),
    )
    args = parser.parse_args()
    print(json.dumps(run_study(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
