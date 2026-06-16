"""Ductile PF-plasticity backend promotion harness.

Runs the backend-selectable ductile PF-plasticity validation through the
requested sparse backends and records a comparison bundle. This is the
backend-promotion counterpart to the ductile operator-coupled validation
example and is intended to feed the remaining #659 evidence gap.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from phast.sparse_solve import available_sparse_backends

from examples.plasticity_interface.run_ductile_pf_plasticity_validation import (
    run_validation,
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _max_rss_kib() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw // 1024 if raw > 100_000_000 else raw


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _backend_status_dict(status) -> dict[str, bool]:
    return {
        "scipy": bool(status.scipy),
        "petsc_mumps": bool(status.petsc),
        "cudss": bool(status.cudss),
    }


def _write_config(output_dir: Path, backends: tuple[str, ...], *,
                  n_steps: int, max_strain: float, l0: float,
                  plastic_work_weight: float) -> str:
    text = "\n".join([
        "case: ductile_pf_backend_promotion",
        "example: ductile_pf_plasticity_validation",
        "mesh: one_square_two_triangle",
        "mechanics: QuasiStaticSolver + MeshJ2Elastoplasticity",
        "phase_field: PhaseFieldDamageSolver + DuctilePhaseFieldCoupling",
        "requested_backends:",
        *[f"  - {backend}" for backend in backends],
        f"n_steps: {n_steps}",
        f"max_strain: {max_strain}",
        f"l0: {l0}",
        f"plastic_work_weight: {plastic_work_weight}",
        "material:",
        "  E: 210000.0",
        "  nu: 0.30",
        "  Gc: 2.7",
        "  pf_model: AT2",
        "  plasticity_model: j2_isotropic",
        "  yield_stress: 250.0",
        "  hardening_modulus: 5000.0",
        "  hardening_type: linear_iso",
    ]) + "\n"
    (output_dir / "config.yaml").write_text(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row_delta(row: dict, baseline: dict, key: str) -> float:
    return abs(float(row[key]) - float(baseline[key]))


def _run_backend(
    requested_backend: str,
    *,
    output_dir: Path,
    n_steps: int,
    max_strain: float,
    l0: float,
    plastic_work_weight: float,
    backend_status,
) -> dict:
    started = time.perf_counter()
    case_dir = output_dir / requested_backend
    case_dir.mkdir(parents=True, exist_ok=True)
    try:
        summary = run_validation(
            case_dir,
            n_steps=n_steps,
            max_strain=max_strain,
            l0=l0,
            plastic_work_weight=plastic_work_weight,
            backend=requested_backend,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        resolved_backend = str(summary.get("resolved_backend", requested_backend))
        fallback_used = requested_backend != resolved_backend
        if fallback_used:
            if requested_backend == "mumps" and not backend_status.petsc:
                fallback_reason = "petsc_mumps_unavailable"
            elif requested_backend == "cudss" and not backend_status.cudss:
                fallback_reason = "cudss_unavailable"
            elif requested_backend == "auto":
                fallback_reason = "auto_resolver_selected_available_backend"
            else:
                fallback_reason = "backend_fallback_or_retry"
        else:
            fallback_reason = ""
        converged = bool(
            summary["final_mechanics_residual"] <= 1.0e-6
            and summary["final_damage_residual_norm"] <= 1.0e-8
            and summary["plastic_work_monotone"]
            and summary["fracture_energy_monotone"]
            and summary["finite_energy_terms"]
        )
        return {
            "requested_backend": requested_backend,
            "resolved_backend": resolved_backend,
            "case_dir": str(case_dir),
            "fallback_used": bool(fallback_used),
            "fallback_reason": fallback_reason,
            "converged": converged,
            "yielded": bool(summary["yielded"]),
            "final_mechanics_newton_iter": int(summary["final_mechanics_newton_iter"]),
            "final_mechanics_residual": float(summary["final_mechanics_residual"]),
            "final_u_norm": float(summary["final_u_norm"]),
            "final_damage_mean": float(summary["final_damage_mean"]),
            "final_damage_max": float(summary["final_damage_max"]),
            "final_damage_residual_norm": float(summary["final_damage_residual_norm"]),
            "final_plastic_work_total": float(summary["final_plastic_work_total"]),
            "final_fracture_total_energy": float(summary["final_fracture_total_energy"]),
            "elapsed_ms": elapsed_ms,
            "max_rss_kib": _max_rss_kib(),
        }
    except Exception as exc:
        return {
            "requested_backend": requested_backend,
            "resolved_backend": "error",
            "case_dir": str(case_dir),
            "fallback_used": False,
            "fallback_reason": "",
            "converged": False,
            "yielded": False,
            "final_mechanics_newton_iter": -1,
            "final_mechanics_residual": float("nan"),
            "final_u_norm": float("nan"),
            "final_damage_mean": float("nan"),
            "final_damage_max": float("nan"),
            "final_damage_residual_norm": float("nan"),
            "final_plastic_work_total": float("nan"),
            "final_fracture_total_energy": float("nan"),
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "max_rss_kib": _max_rss_kib(),
            "error": str(exc),
        }


def _write_standard_files(
    output_dir: Path,
    *,
    config_hash: str,
    rows: list[dict],
    backends: tuple[str, ...],
    elapsed_ms: float,
    backend_status,
    artifact_paths: list[Path],
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "benchmark": "ductile_pf_backend_promotion",
        "timestamp_utc": now,
        "git_sha": _git_sha(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "requested_backends": list(backends),
        "backend_status": _backend_status_dict(backend_status),
        "elapsed_ms": float(elapsed_ms),
        "max_rss_kib": _max_rss_kib(),
    }
    lockfile = {
        "schema": "phast_run_lockfile_v1",
        "created_utc": now,
        "git_sha": metadata["git_sha"],
        "config_sha256": config_hash,
        "deterministic": True,
    }
    manifest = {
        "schema": "phast_run_manifest_v1",
        "benchmark": "ductile_pf_backend_promotion",
        "artifacts": [path.name for path in artifact_paths],
        "case_dirs": [row.get("case_dir", "") for row in rows],
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    (output_dir / "run_lockfile.json").write_text(
        json.dumps(lockfile, indent=2) + "\n")
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    log_lines = [
        f"{now} ductile PF backend promotion started",
        f"{now} requested backends: {', '.join(backends)}",
        f"{now} backend status: {metadata['backend_status']}",
        f"{now} elapsed_ms: {elapsed_ms:.3f}",
    ]
    for row in rows:
        log_lines.append(
            f"{now} backend {row['requested_backend']} -> "
            f"{row['resolved_backend']}; converged={row['converged']}; "
            f"residual={row['final_mechanics_residual']:.3e}; "
            f"damage_residual={row['final_damage_residual_norm']:.3e}"
        )
    (output_dir / "run.log").write_text("\n".join(log_lines) + "\n")


def run_promotion(
    output_dir: Path,
    *,
    backends: tuple[str, ...] = ("auto", "scipy", "mumps", "cudss"),
    n_steps: int = 24,
    max_strain: float = 5.0e-3,
    l0: float = 0.1,
    plastic_work_weight: float = 1.0,
) -> dict:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_hash = _write_config(
        output_dir, backends,
        n_steps=n_steps, max_strain=max_strain, l0=l0,
        plastic_work_weight=plastic_work_weight)
    backend_status = available_sparse_backends()

    rows = []
    for backend in backends:
        rows.append(_run_backend(
            backend,
            output_dir=output_dir,
            n_steps=n_steps,
            max_strain=max_strain,
            l0=l0,
            plastic_work_weight=plastic_work_weight,
            backend_status=backend_status,
        ))

    csv_path = output_dir / "backend_promotion.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [row for row in rows if row["converged"]]
    baseline = ok_rows[0] if ok_rows else None
    max_metric_delta = 0.0
    if baseline is not None:
        metrics = (
            "final_u_norm",
            "final_damage_mean",
            "final_damage_max",
            "final_mechanics_residual",
            "final_damage_residual_norm",
            "final_plastic_work_total",
            "final_fracture_total_energy",
        )
        for row in ok_rows:
            for key in metrics:
                max_metric_delta = max(max_metric_delta, _row_delta(row, baseline, key))

    summary = {
        "example": "ductile_pf_backend_promotion",
        "requested_backends": list(backends),
        "backend_status": _backend_status_dict(backend_status),
        "rows": rows,
        "n_converged": len(ok_rows),
        "max_metric_delta_vs_first_converged": max_metric_delta,
        "all_converged_rows_match_baseline": bool(
            baseline is not None and max_metric_delta <= 1.0e-8),
        "csv": str(csv_path),
        "standard_artifacts": [
            "config.yaml",
            "run_lockfile.json",
            "run_metadata.json",
            "run_manifest.json",
            "run.log",
            "backend_promotion.csv",
            "summary.json",
        ],
        "max_rss_kib": _max_rss_kib(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    _write_standard_files(
        output_dir,
        config_hash=config_hash,
        rows=rows,
        backends=backends,
        elapsed_ms=elapsed_ms,
        backend_status=backend_status,
        artifact_paths=[
            output_dir / "config.yaml",
            output_dir / "backend_promotion.csv",
            output_dir / "summary.json",
            output_dir / "run_lockfile.json",
            output_dir / "run_metadata.json",
            output_dir / "run_manifest.json",
            output_dir / "run.log",
        ],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/plasticity_interface/ductile_pf_backend_promotion"),
    )
    parser.add_argument(
        "--backend",
        action="append",
        dest="backends",
        default=None,
        help="Requested backend; repeat for multiple backends.",
    )
    parser.add_argument("--n-steps", type=int, default=24)
    parser.add_argument("--max-strain", type=float, default=5.0e-3)
    parser.add_argument("--l0", type=float, default=0.1)
    parser.add_argument("--plastic-work-weight", type=float, default=1.0)
    args = parser.parse_args()
    backends = tuple(args.backends) if args.backends else (
        "auto", "scipy", "mumps", "cudss")
    print(json.dumps(run_promotion(
        args.output_dir,
        backends=backends,
        n_steps=args.n_steps,
        max_strain=args.max_strain,
        l0=args.l0,
        plastic_work_weight=args.plastic_work_weight,
    ), indent=2))


if __name__ == "__main__":
    main()
