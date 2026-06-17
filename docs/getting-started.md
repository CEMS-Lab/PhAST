# Getting Started

This page gives the shortest path from a clone to a validated run. Use the
deeper user guide when you need to design a new geometry, choose solver
backends, or interpret benchmark outputs.

Use the fluent `phast.Problem` API to author new models. Use YAML decks for public examples, reproducibility, batch/HPC runs, and sharing exact simulations.

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
YAML input deck that will be run:

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
| Spectral/Amor implicit QS | Install PETSc/MUMPS where possible; `backend: auto` selects it after a runtime smoke test, with SciPy SuperLU and CG fallbacks. |
| Cohesive contact | Sparse quasi-static backend with `backend: auto`; normal-contact penalty only for contact cases. |
| J2 plasticity | Guarded sparse quasi-static plasticity path with `backend: auto`; unsupported combinations fail early. |
| Trajectory datasets | Zarr stores for forward-run trajectories and MP4/raster animations for visualisation. |

Each normal run writes standard outputs plus provenance files such as
`config.yaml` and `run_lockfile.json`. See
`docs/user_guide/example_contract.md` for the promoted-example contract and
the [output standards](output_standards/index.md) page for visualization and
result artifact conventions.

## 5. Inspect the Result

Run a promoted example into an explicit directory, then inspect it without
rerunning the solver:

```bash
python -m phast run configs/benchmarks/solid_mechanics/linear_plate.yaml --output_dir runs/linear_plate
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
| Fluent Python authoring | `docs/tutorial/fluent_authoring_guide.md` and `docs/user_guide/python_api.md` |
| Forward dynamic or static fracture | `docs/user_guide/problem_types.md`, `docs/user_guide/physics.md`, and `docs/user_guide/configuration.md` |
| Quasistatic benchmark reproduction | `docs/benchmarks/catalogue.md`, `docs/benchmarks/examples.md`, and benchmark configs under `configs/benchmarks/` |
| Dataset visualisation | `docs/example-gallery.md`, dynamic benchmark outputs, and standard Zarr-first output conventions |
| Solver/backend selection | `docs/user_guide/sparse_solve.md`, `docs/user_guide/performance.md`, and API notes under `docs/api/` |

## 7. Build the Docs

```bash
sphinx-build -b html docs docs/_build/html
open docs/_build/html/index.html
```

The hosted site is published at:

https://cems-lab.github.io/PhAST/
