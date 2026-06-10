# Installation

## From source

```bash
git clone https://github.com/CEMS-Lab/PhAST
cd phast
pip install -e .
python -m phast doctor
```

## Optional extras

| Extra | Purpose |
| --- | --- |
| `amg` | PyAMG hierarchy setup for the AMG-preconditioned CG path |
| `viz-fast` | PyVista + zstd writer for fast `.pv` output |
| `hpc` | `pyamg` + `pymetis` + `cupy-cuda12x` |
| `dev` | `pytest` for the unit suite |
| `petsc` | PETSc/petsc4py build path for MUMPS-capable environments |

```bash
pip install -e ".[amg]"
pip install -e ".[hpc]"
```

## Workflow backend policy

The default customer setting for implicit and quasi-static problems is
`backend: auto`. It performs runtime smoke tests and selects the strongest
available backend for the problem:

1. PETSc/MUMPS sparse-direct LU when `petsc4py` and MUMPS are functional.
2. SciPy SuperLU sparse-direct LU when SciPy is available.
3. Matrix-free CG for large or explicitly iterative configurations.

For machines with `mamba` or `conda`, the repo installer attempts the
PETSc/MUMPS stack automatically:

```bash
bash install.sh            # tries safe optional workflow packages
bash install.sh --no-direct  # skip the PETSc/MUMPS attempt
```

If automatic installation is not possible, use conda-forge explicitly:

```bash
mamba install -c conda-forge petsc petsc4py mumps-mpi
python -m phast doctor
```

PARDISO and SPOOLES are commercial/legacy sparse-direct solvers discussed for
context in COMSOL comparisons. They are not called by `phast`.

Recommended problem-class defaults are:

| Problem class | Recommended default |
| --- | --- |
| Explicit dynamics | `solver_type: explicit`, `dt_safety: 0.8`, `damage_every: 1` for validation; use `damage_every: 2` or `3` only after sensitivity checks. |
| Quasi-static phase-field fracture | `solver_type: quasi_static`, `backend: auto`, `preconditioner: jacobi`, `stagger_criterion: linf`, `stagger_tol: 1e-6`. |
| Spectral/Amor implicit mechanics | Install PETSc/MUMPS where possible; otherwise `auto` falls back to SciPy SuperLU or CG. |
| Cohesive contact | Sparse quasi-static backend with `backend: auto`; enable contact penalty only for contact cases. |
| J2 plasticity | Guarded sparse quasi-static plasticity path with `backend: auto`. |

Fast `.pv` visualisation output is attempted by `install.sh` through
`pyvista`/`pyvista-zstd`. If it is unavailable, the solver still writes
standard VTU files and MP4/raster animations; visual accuracy is unchanged.

## Hardware notes

- **macOS**: run with `--device cpu --no-compile`. MPS float64 transfers and
  `torch.compile` warmups are too costly for the typical PF workload.
- **CUDA**: use `--device cuda`. `torch.compile` is beneficial for sustained
  runs.
