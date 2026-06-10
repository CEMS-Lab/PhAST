# Getting Started

This page gives the shortest path from a clone to a validated run. Use the
deeper user guide when you need to design a new geometry, choose solver
backends, or interpret benchmark outputs.

## 1. Install

```bash
git clone https://github.com/CEMS-Lab/PhAST.git
cd phast
pip install -e .
```

Core dependencies are PyTorch, NumPy, SciPy, matplotlib, h5py, meshio, gmsh,
Pillow, and PyYAML. Optional solver/backend packages include `pyamg`,
`pymetis`, AmgX, PETSc/MUMPS, cuDSS, Zarr, and PyVista fast visualisation.
PARDISO and SPOOLES are not called by this repository; they are mentioned only
when comparing with commercial solver menus.

For documentation builds:

```bash
pip install -r requirements-docs.txt
```

## 2. Check the Environment

```bash
python -c "import phast, torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -m phast doctor
```

`doctor` is the first command to run on a new workstation or cluster login. It
prints the optional sparse-direct backend status and the backend that
`backend: auto` will choose for CPU implicit and quasi-static workflows.

Platform notes:

| Platform | Practical note |
|---|---|
| Linux + CUDA | Preferred for larger production runs. Use a PyTorch wheel matching the CUDA driver. |
| CPU | Good for validation, small examples, and reproducibility checks. |
| Apple Silicon | MPS can be useful for selected paths, but CPU float64 is safer for verification and spectral/eigenvalue-sensitive cases. |
| Windows | WSL2 is the recommended route for CUDA workflows. |

## 3. Validate a Config

Most workflows start from YAML:

```bash
python -m phast run configs/benchmarks/dynamic/B3_dynamic_sent.yaml --validate-only
```

Validation catches schema errors before mesh generation or solver allocation.
For a readable summary of a setup:

```bash
python -m phast explain-config configs/benchmarks/dynamic/B3_dynamic_sent.yaml
```

## 4. Run a Forward Problem

```bash
python -m phast run configs/benchmarks/dynamic/B3_dynamic_sent.yaml
```

Typical shipped configs cover dynamic fracture and quasistatic benchmarks. Use
`configs/BENCHMARK_RUN_MAP.md` to find the current recommended config for a
benchmark, and use `examples/README.md` to locate runnable example workflows.

Recommended defaults by workflow:

| Workflow | Default setup |
|---|---|
| Dynamic explicit fracture | `solver_type: explicit`, `dt_safety: 0.8`, and `damage_every: 1` for reference validation. Increase `damage_every` only after a subcycling sensitivity check. |
| Quasi-static fracture | `solver_type: quasi_static`, `backend: auto`, `preconditioner: jacobi`, `stagger_criterion: linf`, `stagger_tol: 1e-6`. |
| Spectral/Amor implicit QS | Install PETSc/MUMPS where possible; `backend: auto` selects it after a runtime smoke test, with SciPy SuperLU and CG fallbacks. |
| Cohesive contact | Sparse quasi-static backend with `backend: auto`; normal-contact penalty only for contact cases. |
| J2 plasticity | Guarded sparse quasi-static plasticity path with `backend: auto`; unsupported combinations fail early. |

Each normal run writes standard outputs plus provenance files such as
`config.yaml` and `run_lockfile.json`. See `docs/STANDARD_OUTPUTS.md` for the
output contract.

## 5. Understand the Main Workflows

| Workflow | What to read next |
|---|---|
| Forward dynamic or static fracture | `docs/user_guide/problem_types.md`, `docs/user_guide/physics.md`, and `docs/user_guide/configuration.md` |
| Quasistatic benchmark reproduction | `docs/benchmarks/catalogue.md`, `docs/benchmarks/examples.md`, and benchmark configs under `configs/benchmarks/` |
| Solver/backend selection | `docs/user_guide/sparse_solve.md`, `docs/user_guide/performance.md`, and API notes under `docs/api/` |

## 6. Build the Docs

```bash
sphinx-build -b html docs docs/_build/html
open docs/_build/html/index.html
```

The hosted site is published at:

https://cems-lab.github.io/PhAST/

GitHub workflows are manual-only. Start CI, docs, install, wheel, or Claude
jobs from the GitHub Actions tab when you explicitly want them to run.
