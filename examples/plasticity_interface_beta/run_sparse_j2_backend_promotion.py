"""Sparse J2 backend promotion harness.

Runs the same small plane-strain J2 patch through the sparse backend resolver
for requested backends and writes a comparison report. This is the local/CI
counterpart to the HPC promotion evidence requested in #659.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from phast.fem_operators import FEMOperators
from phast.material import Material
from phast.mechanics_solver import QuasiStaticSolver
from phast.mesh import FEMMesh
from phast.plasticity import MeshJ2Elastoplasticity
from phast.sparse_solve import available_sparse_backends


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


def _write_config(output_dir: Path, backends: tuple[str, ...]) -> str:
    text = "\n".join([
        "case: sparse_j2_backend_promotion",
        "mesh: two_triangle_unit_square",
        "mechanics: QuasiStaticSolver + MeshJ2Elastoplasticity",
        "plasticity_model: j2_isotropic",
        "hardening: linear_iso",
        "requested_backends:",
        *[f"  - {backend}" for backend in backends],
        "material:",
        "  E: 210000.0",
        "  nu: 0.30",
        "  yield_stress: 250.0",
        "  hardening_modulus: 5000.0",
        "boundary_condition: left fixed, right ux=0.004",
    ]) + "\n"
    (output_dir / "config.yaml").write_text(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_standard_files(output_dir: Path, *, config_hash: str,
                          rows: list[dict], backends: tuple[str, ...],
                          elapsed_ms: float, artifact_paths: list[Path],
                          backend_status) -> None:
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "benchmark": "sparse_j2_backend_promotion",
        "timestamp_utc": now,
        "git_sha": _git_sha(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "scipy": _package_version("scipy"),
        "petsc4py": _package_version("petsc4py"),
        "nvmath": _package_version("nvmath-python"),
        "cupy": _package_version("cupy"),
        "requested_backends": list(backends),
        "resolved_backends": sorted({
            str(row.get("resolved_backend", "unknown")) for row in rows
        }),
        "backend_status": {
            "scipy": bool(backend_status.scipy),
            "petsc_mumps": bool(backend_status.petsc),
            "cudss": bool(backend_status.cudss),
        },
        "elapsed_ms": float(elapsed_ms),
        "max_rss_kib": _max_rss_kib(),
    }
    lockfile = {
        "schema": "phast_run_lockfile_v1",
        "created_utc": now,
        "git_sha": metadata["git_sha"],
        "config_sha256": config_hash,
        "deterministic": True,
        "random_seed": None,
    }
    manifest = {
        "schema": "phast_run_manifest_v1",
        "benchmark": "sparse_j2_backend_promotion",
        "artifacts": [path.name for path in artifact_paths],
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    (output_dir / "run_lockfile.json").write_text(
        json.dumps(lockfile, indent=2) + "\n")
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    log_lines = [
        f"{now} sparse J2 backend promotion started",
        f"{now} requested backends: {', '.join(backends)}",
        f"{now} backend status: {metadata['backend_status']}",
        f"{now} resolved backends: {metadata['resolved_backends']}",
        f"{now} elapsed_ms: {elapsed_ms:.3f}",
    ]
    for row in rows:
        log_lines.append(
            f"{now} backend {row['requested_backend']} -> "
            f"{row['resolved_backend']}; converged={row['converged']}; "
            f"residual={row['residual']}")
    (output_dir / "run.log").write_text("\n".join(log_lines) + "\n")


def _mesh() -> FEMMesh:
    nodes = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=torch.float64,
    )
    elements = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    return FEMMesh.from_tensors(nodes, elements, device="cpu", dtype=torch.float64)


def _material() -> Material:
    return Material(
        E=210_000.0,
        nu=0.30,
        Gc=2.7,
        l0=0.1,
        rho=7.8e-9,
        energy_split="amor",
        plasticity_model="j2_isotropic",
        yield_stress=250.0,
        hardening_modulus=5_000.0,
        hardening_type="linear_iso",
        plane_stress=False,
    )


def _run_backend(requested_backend: str, backend_status) -> dict:
    started = time.perf_counter()
    mesh = _mesh()
    material = _material()
    fem = FEMOperators(mesh, material)
    plasticity = MeshJ2Elastoplasticity(mesh, material)
    solver = QuasiStaticSolver(
        fem,
        plasticity_operator=plasticity,
        backend=requested_backend,
        tol=1.0e-6,
        tol_rel=1.0e-7,
        max_iter=12,
    )
    bc_mask = torch.zeros((mesh.n_nodes, 2), dtype=torch.bool)
    bc_vals = torch.zeros((mesh.n_nodes, 2), dtype=mesh.dtype)
    left = torch.tensor([0, 3], dtype=torch.long)
    right = torch.tensor([1, 2], dtype=torch.long)
    bc_mask[left, :] = True
    bc_mask[right, 0] = True
    bc_vals[right, 0] = 4.0e-3
    d = torch.zeros(mesh.n_nodes, dtype=mesh.dtype)
    f_ext = torch.zeros((mesh.n_nodes, 2), dtype=mesh.dtype)
    u, converged, n_iter = solver.solve(d, f_ext, bc_mask, bc_vals)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    fallback_used = requested_backend != solver.last_backend
    fallback_reason = ""
    if fallback_used:
        if requested_backend == "mumps" and not backend_status.petsc:
            fallback_reason = "petsc_mumps_unavailable"
        elif requested_backend == "cudss" and not backend_status.cudss:
            fallback_reason = "cudss_unavailable"
        elif requested_backend == "auto":
            fallback_reason = "auto_resolver_selected_available_backend"
        else:
            fallback_reason = "backend_fallback_or_retry"
    return {
        "requested_backend": requested_backend,
        "resolved_backend": solver.last_backend,
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
        "converged": bool(converged),
        "newton_iter": int(n_iter),
        "residual": float(solver.last_residual),
        "residual_history_json": json.dumps([float(solver.last_residual)]),
        "u_norm": float(u.norm().item()),
        "eps_p_eq_mean": float(plasticity.state.eps_p_eq.mean().item()),
        "plastic_work_density_mean": float(
            plasticity.state.plastic_work_density.mean().item()),
        "elapsed_ms": elapsed_ms,
        "max_rss_kib": _max_rss_kib(),
    }


def run_promotion(output_dir: Path,
                  backends: tuple[str, ...] = ("auto", "scipy")) -> dict:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_hash = _write_config(output_dir, backends)
    backend_status = available_sparse_backends()
    rows = []
    for backend in backends:
        try:
            rows.append(_run_backend(backend, backend_status))
        except Exception as exc:
            rows.append({
                "requested_backend": backend,
                "resolved_backend": "error",
                "fallback_used": False,
                "fallback_reason": "",
                "converged": False,
                "newton_iter": -1,
                "residual": float("nan"),
                "residual_history_json": json.dumps([]),
                "u_norm": float("nan"),
                "eps_p_eq_mean": float("nan"),
                "plastic_work_density_mean": float("nan"),
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                "max_rss_kib": _max_rss_kib(),
                "error": str(exc),
            })

    csv_path = output_dir / "backend_promotion.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [row for row in rows if row["converged"]]
    baseline = ok_rows[0] if ok_rows else None
    max_u_norm_delta = 0.0
    if baseline is not None:
        max_u_norm_delta = max(
            abs(float(row["u_norm"]) - float(baseline["u_norm"]))
            for row in ok_rows
        )
    summary = {
        "example": "sparse_j2_backend_promotion",
        "requested_backends": list(backends),
        "backend_status": {
            "scipy": bool(backend_status.scipy),
            "petsc_mumps": bool(backend_status.petsc),
            "cudss": bool(backend_status.cudss),
        },
        "rows": rows,
        "all_requested_converged_or_reported": True,
        "n_converged": len(ok_rows),
        "max_u_norm_delta_vs_first_converged": max_u_norm_delta,
        "all_converged_rows_match_baseline": bool(
            baseline is not None and max_u_norm_delta <= 1.0e-10),
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
        artifact_paths=[
            output_dir / "config.yaml",
            output_dir / "backend_promotion.csv",
            output_dir / "summary.json",
            output_dir / "run_lockfile.json",
            output_dir / "run_metadata.json",
            output_dir / "run_manifest.json",
            output_dir / "run.log",
        ],
        backend_status=backend_status,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/plasticity_interface/sparse_j2_backend_promotion"),
    )
    parser.add_argument(
        "--backend",
        action="append",
        dest="backends",
        default=None,
        help="Requested backend; repeat for multiple backends.",
    )
    args = parser.parse_args()
    backends = tuple(args.backends) if args.backends else ("auto", "scipy")
    print(json.dumps(run_promotion(args.output_dir, backends=backends), indent=2))


if __name__ == "__main__":
    main()
