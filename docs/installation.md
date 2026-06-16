# Installation & Verification

This guide outlines the standard paths for installing, configuring, and verifying the **PhAST** framework across local workstations and HPC clusters.

---

## 1. Quick Install

The package is pure Python. Standard dependencies (PyTorch, NumPy, SciPy, Matplotlib, h5py, meshio, Gmsh, Pillow, and PyYAML) are installed automatically.

```bash
pip install phast
python -m phast doctor
```

---

## 2. Auto-Detecting Installer

If you are setting up from a local checkout, use the auto-detecting installer script at the repository root. It detects your operating system, GPU architecture, and attempts to install optimized solver libraries:

```bash
bash install.sh            # Auto-detect platform + GPU + best solver libs
bash install.sh cuda       # Force NVIDIA CUDA environment
bash install.sh mps        # Force macOS Apple Silicon
bash install.sh cpu        # Force CPU-only environment
bash install.sh --no-direct  # Skip the automatic PETSc/MUMPS compile attempt
```

---

## 3. Manual Source Install

For development or direct workspace customization:

```bash
git clone https://github.com/CEMS-Lab/PhAST.git
cd phast
pip install -e .
python -m phast doctor
```

---

## 4. Platform-Specific Setup

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
pytest tests -q
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
pytest tests -q
```

---

## 5. Optional Extras & Packages

You can append extras during installation to enable accelerated preconditioning, fast export, or cluster support:

| Extra Group | Install Command | Purpose / Target |
|---|---|---|
| `[dev]` | `pip install -e ".[dev]"` | Unit testing (`pytest`) and Sphinx documentation tools. |
| `[amg]` | `pip install -e ".[amg]"` | `pyamg` hierarchy setup for multi-grid preconditioning on CPU/GPU. |
| `[amgx]` | `pip install pyamgx` | NVIDIA AmgX wrapper (requires `module load amgx` on CUDA clusters). |
| `[metis]` | `pip install -e ".[metis]"` | `pymetis` for domain-decomposition/multi-GPU execution. |
| `[petsc]` | `pip install -e ".[petsc]"` | Compiled PETSc/petsc4py stack for MUMPS direct solver on CPU/HPC. |
| `[viz-fast]` | `pip install -e ".[viz-fast]"` | PyVista + zstd compression for fast `.pv` time-series files. |
| `[dataset]` | `pip install -e ".[dataset]"` | Zarr + numcodecs trajectory data extraction support. |
| `[hpc]` | `pip install -e ".[hpc]"` | Bundle package including `pyamg` + `pymetis` + `cupy`. |

---

## 6. Verification and Smoke Tests

### Test 1: Doctor Check
The doctor tool inspects execution backends and validates what solver `backend: auto` will default to:
```bash
python -m phast doctor
```

### Test 2: Autograd + Sparse Solver Sanity
Execute this one-liner to verify that the sparse solver is correctly linked and that autograd can propagate gradients through the sparse solver:
```bash
python -c "from phast.sparse_solve import solve; import torch; \
i = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 2, 3, 3], [0, 1, 0, 1, 2, 1, 2, 3, 2, 3]], dtype=torch.long); \
v = torch.tensor([2., -1., -1., 2., -1., -1., 2., -1., -1., 2.], dtype=torch.float64, requires_grad=True); \
K = torch.sparse_coo_tensor(i, v, (4, 4)).coalesce(); b = torch.tensor([1., 0., 0., 1.], dtype=torch.float64); \
x = solve(K, b); x.sum().backward(); print('Sol:', x.detach().numpy(), '\nGrad_v:', v.grad.detach().numpy())"
```
* **Expected Output:** Solution vector should be `[1., 1., 1., 1.]` and gradients `Grad_v` should print successfully.

### Test 3: Run Validation and Dry Run
Verify config parser schema validation and execute a short dry-run (20 explicit steps on CPU without plots):
```bash
# Validate config schema
python -m phast run configs/benchmarks/dynamic/B3_dynamic_sent.yaml --validate-only

# Short execution test
python -m phast run configs/benchmarks/dynamic/B3_dynamic_sent.yaml --device cpu --num_steps 20 --no-plots
```
Confirm the output directory contains the run metadata, solved parameters, and telemetry: `config.yaml`, `run_lockfile.json`, and `results.csv`.

---

## 7. Workflow Backend Policy

The framework uses an automated backend routing policy (`backend: auto` inside configurations) to select the most efficient linear solver for implicit mechanics and quasi-static damage solves:

1. **`mumps`:** PETSc/MUMPS sparse-direct solver. Chosen if the environment passes the `petsc4py` runtime smoke test. Best for large, highly-unstable quasi-static crack initiation on clusters.
2. **`scipy`:** SciPy SuperLU sparse-direct solver. Portable CPU fallback used when PETSc is unavailable.
3. **`cg`:** Matrix-free Conjugate Gradient solver. Best for massive GPU-bound meshes.

---

## 8. Expected Parameter Defaults by Workflow

For literature comparisons and verification parity, maintain the following defaults in your YAML files unless specifically testing alternatives:

| Workflow Class | Solver Type | Backend | Preconditioner | Stagger Criterion | Step Safety / Cadence |
|---|---|---|---|---|---|
| **Explicit Dynamics** | `explicit` | N/A | N/A | N/A | `dt_safety: 0.8`, `damage_every: 1` |
| **Quasi-static Implicit** | `quasi_static` | `auto` | `jacobi` | `linf` (L_inf norm) | `stagger_tol: 1e-6`, `max_stagger: 50` |
| **Cohesive Interface** | `quasi_static` | `auto` | `jacobi` | `relative` | Contact penalty active only on contact boundaries |
| **Ductile J2 Plasticity** | `quasi_static` | `auto` | `jacobi` | `relative` | Guarded material-point return mapping active |
| **Trajectory Datasets** | N/A | N/A | N/A | N/A | Zarr trajectory output active |
