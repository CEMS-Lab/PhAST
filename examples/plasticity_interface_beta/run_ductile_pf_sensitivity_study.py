"""Ductile PF-plasticity length-scale and plastic-work sensitivity study.

This runner packages a small, deterministic public beta validation study
for the current ductile phase-field plasticity slice. It compares an
elastic-driving reference against ductile plastic-work driving at several
regularization lengths, writes tabular acceptance evidence, and produces
review-sized figures.

The study advances the ductile validation gate but does not claim a
benchmark-matched SENT or TPB fracture calibration.
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
from PIL import Image

try:
    from examples.plasticity_interface_beta.run_ductile_pf_plasticity_validation import (
        run_validation,
    )
except ModuleNotFoundError:
    from run_ductile_pf_plasticity_validation import run_validation


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


def _write_visual_manifest(output_dir: Path, image_names: list[str]) -> list[dict]:
    rows = []
    for name in image_names:
        path = output_dir / name
        with Image.open(path) as img:
            width, height = img.size
        rows.append({
            "file": name,
            "width_px": int(width),
            "height_px": int(height),
            "size_bytes": int(path.stat().st_size),
            "review_dimension_passed": bool(max(width, height) < 2000),
        })
    (output_dir / "visual_manifest.json").write_text(
        json.dumps(rows, indent=2) + "\n")
    return rows


def _case_rows(cases: list[dict]) -> list[dict]:
    elastic_ref = next(row for row in cases if row["plastic_work_weight"] == 0.0)
    rows = []
    for row in cases:
        rows.append({
            "case": row["case"],
            "l0": row["l0"],
            "plastic_work_weight": row["plastic_work_weight"],
            "yielded": row["yielded"],
            "final_elastic_driving_mean": row["final_elastic_driving_mean"],
            "final_ductile_driving_mean": row["final_ductile_driving_mean"],
            "driving_lift_vs_elastic_only": (
                row["final_ductile_driving_mean"]
                - row["final_elastic_driving_mean"]
            ),
            "damage_lift_vs_elastic_reference": (
                row["final_damage_mean"]
                - elastic_ref["final_damage_mean"]
            ),
            "final_damage_mean": row["final_damage_mean"],
            "final_damage_max": row["final_damage_max"],
            "final_damage_residual_norm": row["final_damage_residual_norm"],
            "final_damage_pcg_iter": row["final_damage_pcg_iter"],
            "max_rss_kib": row["max_rss_kib"],
            "case_dir": row["case_dir"],
        })
    return rows


def run_study(output_dir: Path, *, n_steps: int = 36,
              max_strain: float = 5.0e-3,
              l0_values: tuple[float, ...] = (0.075, 0.10, 0.15)) -> dict:
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases: list[dict] = []
    reference_l0 = l0_values[len(l0_values) // 2]
    plan = [("elastic_reference_l0_0p10", reference_l0, 0.0)]
    plan.extend(
        (f"ductile_l0_{str(l0).replace('.', 'p')}", l0, 1.0)
        for l0 in l0_values
    )
    for case_name, l0, weight in plan:
        case_dir = output_dir / case_name
        summary = run_validation(
            case_dir,
            n_steps=n_steps,
            max_strain=max_strain,
            l0=l0,
            plastic_work_weight=weight,
        )
        summary["case"] = case_name
        summary["case_dir"] = case_name
        cases.append(summary)

    rows = _case_rows(cases)
    table_path = output_dir / "ductile_sensitivity_table.csv"
    with table_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ductile_rows = [row for row in rows if row["plastic_work_weight"] > 0.0]
    fig, ax = plt.subplots(figsize=(5.8, 3.6), constrained_layout=True)
    ax.plot(
        [row["l0"] for row in ductile_rows],
        [row["final_damage_mean"] for row in ductile_rows],
        marker="o",
        linewidth=1.5,
        label="ductile driving",
    )
    ax.axhline(
        rows[0]["final_damage_mean"],
        color="0.35",
        linestyle="--",
        linewidth=1.0,
        label="elastic-driving reference",
    )
    ax.set_xlabel("regularization length l0")
    ax.set_ylabel("final mean damage")
    ax.set_title("Ductile PF length-scale sensitivity")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend(loc="best", fontsize=8)
    damage_png = output_dir / "ductile_damage_sensitivity.png"
    fig.savefig(damage_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 3.6), constrained_layout=True)
    labels = [row["case"].replace("_", "\n") for row in rows]
    ax.bar(labels, [row["driving_lift_vs_elastic_only"] for row in rows])
    ax.set_ylabel("final driving-force lift")
    ax.set_title("Plastic-work contribution to PF driving force")
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(7)
    lift_png = output_dir / "ductile_driving_lift.png"
    fig.savefig(lift_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    visual_manifest = _write_visual_manifest(
        output_dir,
        ["ductile_damage_sensitivity.png", "ductile_driving_lift.png"],
    )

    all_residuals_pass = all(
        row["final_damage_residual_norm"] < 1.0e-8 for row in rows)
    all_yielded = all(row["yielded"] for row in rows)
    ductile_lift_pass = all(
        row["driving_lift_vs_elastic_only"] > 0.0 for row in ductile_rows)
    ductile_damage_pass = all(
        row["damage_lift_vs_elastic_reference"] >= -1.0e-12
        for row in ductile_rows
    )
    validation_passed = bool(
        all_residuals_pass and all_yielded and ductile_lift_pass
        and ductile_damage_pass
        and all(item["review_dimension_passed"] for item in visual_manifest)
    )

    config_text = "\n".join([
        "example: ductile_pf_sensitivity_study",
        "capability_boundary: validation study, not benchmark-matched SENT/TPB",
        f"n_steps: {n_steps}",
        f"max_strain: {max_strain}",
        "l0_values: [" + ", ".join(str(v) for v in l0_values) + "]",
        "reference: elastic-driving same-mesh case with plastic_work_weight=0",
        "",
    ])
    (output_dir / "config.yaml").write_text(config_text)
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "elapsed_ms": 1000.0 * (time.perf_counter() - t0),
        "max_rss_kib": _max_rss_kib(),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    (output_dir / "run_lockfile.json").write_text(
        json.dumps({
            "config_sha256": hashlib.sha256(
                config_text.encode("utf-8")).hexdigest(),
            "resolved_config": {
                "n_steps": n_steps,
                "max_strain": max_strain,
                "l0_values": list(l0_values),
                "case_count": len(rows),
            },
            "metadata": metadata,
        }, indent=2) + "\n"
    )
    (output_dir / "run.log").write_text(
        "\n".join([
            "example=ductile_pf_sensitivity_study",
            f"validation_passed={validation_passed}",
            f"case_count={len(rows)}",
            f"max_damage_residual_norm={max(row['final_damage_residual_norm'] for row in rows):.12e}",
            f"elapsed_ms={metadata['elapsed_ms']:.3f}",
            "",
        ])
    )

    summary = {
        "example": "ductile_pf_sensitivity_study",
        "capability_boundary": (
            "ductile plastic-work driving-force and length-scale validation "
            "study; not a benchmark-matched SENT/TPB fracture calibration"
        ),
        "n_cases": len(rows),
        "n_steps": n_steps,
        "max_strain": max_strain,
        "l0_values": list(l0_values),
        "validation_passed": validation_passed,
        "all_residuals_pass": bool(all_residuals_pass),
        "all_yielded": bool(all_yielded),
        "ductile_lift_pass": bool(ductile_lift_pass),
        "ductile_damage_pass": bool(ductile_damage_pass),
        "max_damage_residual_norm": max(
            row["final_damage_residual_norm"] for row in rows),
        "max_final_damage": max(row["final_damage_max"] for row in rows),
        "table": table_path.name,
        "plots": ["ductile_damage_sensitivity.png", "ductile_driving_lift.png"],
        "visual_manifest": visual_manifest,
        "max_rss_kib": _max_rss_kib(),
        "cases": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "run_manifest.json").write_text(
        json.dumps({
            "example": "ductile_pf_sensitivity_study",
            "summary": "summary.json",
            "standard_outputs": {
                "config.yaml": "config.yaml",
                "run_lockfile.json": "run_lockfile.json",
                "run_metadata.json": "run_metadata.json",
                "run.log": "run.log",
                "ductile_sensitivity_table.csv": table_path.name,
                "ductile_damage_sensitivity.png": damage_png.name,
                "ductile_driving_lift.png": lift_png.name,
                "visual_manifest.json": "visual_manifest.json",
            },
            "case_directories": [row["case_dir"] for row in rows],
            "validation_passed": validation_passed,
        }, indent=2) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/plasticity_interface/ductile_pf_sensitivity"),
    )
    parser.add_argument("--n-steps", type=int, default=36)
    parser.add_argument("--max-strain", type=float, default=5.0e-3)
    parser.add_argument(
        "--l0-values",
        type=float,
        nargs="+",
        default=[0.075, 0.10, 0.15],
    )
    args = parser.parse_args()
    print(json.dumps(run_study(
        args.output_dir,
        n_steps=args.n_steps,
        max_strain=args.max_strain,
        l0_values=tuple(args.l0_values),
    ), indent=2))


if __name__ == "__main__":
    main()
