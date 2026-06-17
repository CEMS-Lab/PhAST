# Getting Started

This page gives the shortest path from a clone to a validated run. Use the
deeper user guide when you need to design a new geometry, choose solver
backends, or interpret benchmark outputs.

Use the fluent `phast.Problem` API to author new models. Use YAML configurations for public examples, reproducibility, batch/HPC runs, and sharing exact simulations.

## 1. Install

```bash
git clone https://github.com/CEMS-Lab/PhAST.git
cd PhAST
pip install -e .
```

Core dependencies are PyTorch, NumPy, SciPy, matplotlib, h5py, meshio, gmsh,
Pillow, and PyYAML. Optional solver/backend packages include `pyamg`,
`pymetis`, AmgX, PETSc/MUMPS, cuDSS, Zarr, and PyVista fast visualisation.

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

## 3. Author or Validate a Setup

Author new forward models with the fluent Python API when you are designing a
problem interactively:

```python
import phast

problem = (
    phast.Problem("linear plate")
    .geometry("structured_grid", nx=20, ny=10, length=1.0, height=0.2)
    .region("body", kind="domain")
    .material("steel", region="body", E=2.1e11, nu=0.3)
    .analysis_step("load", kind="solid_mechanics", controls={"tip_force_y": -1.0e3})
    .solver("solid_mechanics", example="solid_mechanics.linear_plate")
    .outputs(fields=["displacement", "von_mises"], histories=["response"])
)

problem.validate_setup()
result = problem.run(output_dir="runs/linear_plate", return_result=True)
print(result.metadata())
```

For public examples, reproducible sharing, CI, and HPC queues, validate the
YAML declarative configuration that will be run:

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
python -m phast run configs/benchmarks/dynamic/B3_dynamic_sent.yaml --device cpu --num_steps 20 --no-plots --output_dir runs/b3_dynamic_sent
```

Typical shipped configs cover dynamic fracture and quasi-static benchmarks. Use
`examples/README.md` and the example gallery to locate runnable public
workflows.

Recommended defaults by workflow:

| Workflow | Default setup |
|---|---|
| Dynamic explicit fracture | `solver_type: explicit`, `dt_safety: 0.8`, and `damage_every: 1` for reference validation. Increase `damage_every` only after a subcycling sensitivity check. |
| Quasi-static fracture | `solver_type: quasi_static`, `backend: auto`, `preconditioner: jacobi`, `stagger_criterion: linf`, `stagger_tol: 1e-6`. |
| Spectral/Amor implicit QS | Install PETSc/MUMPS where possible; `backend: auto` selects it after a runtime verification, with SciPy SuperLU and CG fallbacks. |
| Cohesive contact | Sparse quasi-static backend with `backend: auto`; normal-contact penalty only for contact cases. |
| J2 plasticity | Guarded sparse quasi-static plasticity path with `backend: auto`; unsupported combinations fail early. |
| Trajectory datasets | Zarr stores for forward-run trajectories and lightweight GIF/raster animations for visualisation. |

Each normal run writes standard outputs plus provenance files such as
`config.yaml` and `run_lockfile.json`. See
`docs/user_guide/example_contract.md` for the promoted-example contract and
the [output standards](output_standards/index.md) page for visualization and
result artifact conventions.

## 5. Inspect the Result

Run a promoted example into an explicit directory, then inspect it without
rerunning the solver:

```bash
python -m phast run examples/solid_mechanics_beta/linear_plate/config.yaml --output_dir runs/linear_plate
```

```python
import phast

result = phast.load_result("runs/linear_plate")
print(result.metadata())
print(result.manifest())
print(result.history_names())
print(result.visuals())
```

`Result` is read-only. It reports stored manifests, histories, visuals, mesh
metadata, and raw fields where the run actually wrote a trajectory store. It
does not silently derive missing postprocessed fields.

## 6. Understand the Main Workflows

| Workflow | What to read next |
|---|---|
| Fluent Python authoring | `docs/user_guide/python_api.md` and `docs/tutorial/problem_setup_walkthrough.ipynb` |
| Forward dynamic or static fracture | `docs/user_guide/physics.md`, `docs/user_guide/configuration.md`, and `docs/user_guide/capability_matrix.md` |
| Quasistatic benchmark reproduction | `docs/benchmarks/catalogue.md` and benchmark configs under `configs/benchmarks/` |
| Dataset visualisation | `docs/example-gallery.md`, dynamic benchmark outputs, and standard Zarr-first output conventions |
| Solver/backend selection | `docs/user_guide/sparse_solve.md`, `docs/user_guide/configuration.md`, and `docs/performance_reproducibility/index.md` |

## 7. Advanced Installation And Platform Setup

### macOS (Apple Silicon or Intel)
* **Execution Device:** Prefer `--device cpu`.
* **Hardware note:** While Apple Silicon GPU (MPS) is supported, MPS currently lacks native `float64` operations, forcing the damage solver to CPU. High-precision mechanics and eigenvalue/spectral-sensitive operations are more stable and faster running directly on the CPU.
* Run with `--no-compile` as PyTorch JIT warmup overhead dominates typical 2D meshes on Mac.

### Linux & WSL2 (CUDA)
* **Execution Device:** Use `--device cuda`.
* Ensure you install PyTorch matching your system CUDA version:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu121  # Adjust cuXXX for your driver
  pip install -e .
  ```
* `torch.compile` is supported and recommended for long-horizon or large-batch runs.

### HPC Clusters (SLURM)
```bash
module load python/3.11 cuda/12.1
git clone https://github.com/CEMS-Lab/PhAST.git
cd phast
pip install --user -e ".[hpc]"
```

For high-resolution quasi-static validation where direct factorization is needed, configure and validate the optional **PETSc/MUMPS** stack.

#### Clean PETSc/MUMPS validation
The most reproducible route is a fresh environment with PETSc, petsc4py, and MUMPS from the same conda-forge solve:

```bash
mamba create -n phast-petsc -c conda-forge \
  python=3.11 numpy scipy pytorch petsc petsc4py mumps-mpi
mamba activate phast-petsc

pip install -e .
python -m phast doctor
python - <<'PY'
from phast.sparse_solve import available_sparse_backends
print(available_sparse_backends())
PY
python -m phast run configs/benchmarks/quasistatic/QS_notched_holed_plate.yaml --validate-only
```

On clusters with a site PETSc module, build petsc4py against the loaded PETSc instead of mixing unrelated binaries:

```bash
module load petsc/<site-petsc-with-mumps>
export PETSC_DIR=/path/to/site/petsc
export PETSC_ARCH=<site-arch-if-required>

python -m pip install --no-binary=:all: petsc4py
python -m phast doctor
```

Use a direct backend check before claiming PETSc/MUMPS support for a machine:

```bash
python - <<'PY'
from phast.sparse_solve import available_sparse_backends
backends = available_sparse_backends()
print(backends)
raise SystemExit(0 if backends.petsc else 1)
PY
```

If a stale `libpetsc` inside an environment shadows the intended PETSc module,
remove the conflicting package or rebuild the environment cleanly.

For issue and pull-request validation, run:

```bash
python -m phast doctor
sphinx-build -W -b html docs docs/_build/html
python -m phast run configs/benchmarks/dynamic/B3_dynamic_sent.yaml --validate-only
```

### Optional Extras And Packages

You can append extras during installation to enable accelerated preconditioning, fast export, or cluster support:

| Extra Group | Install Command | Purpose / Target |
|---|---|---|
| `[dev]` | `pip install -e ".[dev]"` | Development and Sphinx documentation tools. |
| `[amg]` | `pip install -e ".[amg]"` | `pyamg` hierarchy setup for multi-grid preconditioning on CPU/GPU. |
| `[amgx]` | `pip install pyamgx` | NVIDIA AmgX wrapper (requires `module load amgx` on CUDA clusters). |
| `[metis]` | `pip install -e ".[metis]"` | `pymetis` for domain-decomposition/multi-GPU execution. |
| `[petsc]` | `pip install -e ".[petsc]"` | Compiled PETSc/petsc4py stack for MUMPS direct solver on CPU/HPC. |
| `[viz-fast]` | `pip install -e ".[viz-fast]"` | PyVista + zstd compression for fast `.pv` time-series files. |
| `[dataset]` | `pip install -e ".[dataset]"` | Zarr + numcodecs trajectory data extraction support. |
| `[hpc]` | `pip install -e ".[hpc]"` | Bundle package including `pyamg` + `pymetis` + `cupy`. |

### Workflow Backend Policy

The framework uses an automated backend routing policy (`backend: auto` inside configurations) to select the most efficient linear solver for implicit mechanics and quasi-static damage solves:

1. **`mumps`:** PETSc/MUMPS sparse-direct solver. Chosen if the environment passes the `petsc4py` runtime verification. Best for large, highly-unstable quasi-static crack initiation on clusters.
2. **`scipy`:** SciPy SuperLU sparse-direct solver. Portable CPU fallback used when PETSc is unavailable.
3. **`cg`:** Matrix-free Conjugate Gradient solver. Best for massive GPU-bound meshes.

## 8. Build The Docs

```bash
sphinx-build -b html docs docs/_build/html
open docs/_build/html/index.html
```

The hosted site is published at:

https://cems-lab.github.io/PhAST/
