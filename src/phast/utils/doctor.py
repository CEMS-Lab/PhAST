"""Environment diagnostics for customer installs."""

from __future__ import annotations

import argparse
import importlib.util
import platform
import shutil
import sys


def _module_version(name: str) -> str:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return "missing"
    try:
        module = __import__(name)
    except Exception as exc:  # pragma: no cover - depends on local env
        return f"import failed: {exc}"
    return str(getattr(module, "__version__", "installed"))


def _torch_status() -> tuple[str, str, str]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local env
        return f"import failed: {exc}", "False", "False"
    cuda = bool(torch.cuda.is_available())
    mps = bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
    return str(torch.__version__), str(cuda), str(mps)


def _backend_status() -> tuple[object, str, str]:
    from ..solvers.sparse_solve import available_sparse_backends

    status = available_sparse_backends()
    if status.petsc:
        selected = "mumps"
        note = "PETSc/MUMPS is available and will be preferred by backend='auto'."
    elif status.scipy:
        selected = "scipy"
        note = "SciPy SuperLU is available; MUMPS is not active in this environment."
    elif status.cudss:
        selected = "cudss"
        note = "cuDSS is available for CUDA sparse direct solves."
    else:
        selected = "cg"
        note = "No sparse-direct backend is active; matrix-free CG remains available."
    return status, selected, note


def build_report() -> str:
    torch_ver, cuda, mps = _torch_status()
    status, selected, note = _backend_status()
    lines = [
        "PhAST environment doctor",
        "=" * 34,
        f"Python:   {sys.version.split()[0]} ({platform.platform()})",
        f"PyTorch:  {torch_ver}",
        f"CUDA:     {cuda}",
        f"MPS:      {mps}",
        "",
        "Core packages:",
        f"  numpy:  {_module_version('numpy')}",
        f"  scipy:  {_module_version('scipy')}",
        f"  gmsh:   {_module_version('gmsh')}",
        f"  meshio: {_module_version('meshio')}",
        f"  zarr:   {_module_version('zarr')}",
        f"  ffmpeg: {'found' if shutil.which('ffmpeg') else 'missing'}",
        f"  pyvista fast viz: {_module_version('pyvista')}",
        "",
        "Optional sparse-direct backends:",
        f"  scipy SuperLU: {status.scipy}",
        f"  PETSc/MUMPS:   {status.petsc}",
        f"  cuDSS/nvmath:  {status.cudss}",
        "",
        f"backend='auto' on CPU will select: {selected}",
        note,
        "",
        "Backend meanings:",
        "  scipy = SciPy SuperLU sparse-direct LU",
        "  mumps = PETSc/MUMPS sparse-direct LU through petsc4py",
        "  cudss = NVIDIA cuDSS/nvmath CUDA sparse direct solve",
        "  cg    = matrix-free preconditioned CG",
        "",
        "Recommended problem-class defaults:",
        "  explicit dynamics: solver_type=explicit, dt_safety=0.8, damage_every=1 for reference validation",
        "  explicit throughput: after validation, damage_every=2 or 3 for subcycling sensitivity runs",
        "  quasi-static fracture: solver_type=quasi_static, backend=auto, preconditioner=jacobi",
        "  spectral/Amor QS on CPU/HPC: install PETSc/MUMPS so backend=auto can select mumps",
        "  cohesive contact: sparse quasi-static backend, backend=auto, normal-contact penalty only when configured",
        "  J2 plasticity: sparse quasi-static backend, backend=auto, guarded supported material combinations",
        "  dataset/deep learning: Zarr trajectory stores; MP4/raster visualisation when animations are requested",
        "",
        "Install guidance:",
        "  Start with: pip install -e .",
        "  Add workflow extras as needed: .[amg], .[viz-fast], .[dataset], .[hpc]",
        "  For HPC MUMPS: install petsc petsc4py mumps-mpi from conda-forge",
        "  Then rerun: python -m phast doctor",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check PhAST install and solver backend status.")
    parser.parse_args(argv)
    print(build_report())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
