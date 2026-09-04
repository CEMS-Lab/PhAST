# Getting Started

This is the canonical installation and first-run route. The README and tutorial
index summarize this page rather than defining separate installation contracts.

## Installation Route Selector

PhAST is a Python/PyTorch project. There is no native C++ library or CMake build
to compile. "Source installation" means installing the Python package from this
repository.

### Conda

From the repository root:

```bash
conda env create -f environment.yml
conda activate phast
python run_sanitizer.py
```

The supplied environment is a portable Python 3.11 CPU baseline. PETSc/MUMPS,
CUDA, AmgX, and site-specific MPI stacks remain optional installations and are
not implied by this environment.

### CPU Docker image

Docker Desktop or Docker Engine can build the same Linux CPU reference image on
Linux, macOS, or Windows hosts:

```bash
docker build -t phast:cpu .
docker run --rm phast:cpu
```

The image runs `python run_sanitizer.py` by default. It does not claim CUDA,
Apple MPS, PETSc/MUMPS, MPI, or host-native performance.

### Editable Python source installation

The virtual-environment commands below install `pip install -e .`. This route is
appropriate for students changing PhAST source or documentation; no separate
native compilation step is required.

This guide takes a new user from a source checkout to a validated configuration,
a small completed simulation, and programmatic result inspection. No optional
HPC or sparse-direct backend is required for the basic workflow.

## 1. Prerequisites

PhAST currently supports Python 3.10-3.12 and requires Git. Python 3.11 is
recommended for a first installation. The base installation obtains
PyTorch, NumPy, SciPy, Gmsh, meshio, Matplotlib, YAML support, and the standard
result-storage dependencies from `pyproject.toml`.

PhAST itself does not require a separate CMake build. PETSc/MUMPS, AmgX, cuDSS,
PyAMG, and other optional backends should be installed only for workflows that
explicitly require them.

## 2. Create An Environment And Install

On Linux or macOS:

```bash
git clone https://github.com/CEMS-Lab/PhAST.git
cd PhAST
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows PowerShell:

```powershell
git clone https://github.com/CEMS-Lab/PhAST.git
cd PhAST
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

An editable installation is appropriate for a source repository: changes made
under `src/phast/` are immediately available in the active environment.

Before creating the environment, check `python3.11 --version` (or the selected
Python 3.10-3.12 interpreter). Do not rely on an unqualified `python3` when it
selects a newer, untested interpreter.

## 3. Verify The Environment

Run the environment report:

```bash
python -m phast doctor
```

The report identifies Python and PyTorch versions, CPU/GPU visibility, required
packages, optional sparse backends, and the backend selected by `backend: auto`.
An unavailable optional backend is not an installation failure.

Next, validate a fracture configuration without generating a mesh or allocating
the full solver:

```bash
python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml --validate-only
```

Expected output:

```text
OK: examples/dynamic/B2_kalthoff_winkler/config.yaml passes schema validation.
```

This message means that the YAML satisfies the schema and the implemented
semantic preflight checks. It does not run a fracture solve and does not prove
mesh convergence, benchmark reproduction, or physical validity. Review the
example README, `explain-config` warnings, retained comparison evidence, and
mesh-to-length-scale ratio before making a scientific claim.

For a readable summary of the model before execution:

```bash
python -m phast explain-config examples/dynamic/B2_kalthoff_winkler/config.yaml
```

The [installation verification page](verify-install.md) explains common
`doctor` outcomes.

For a shorter installation-only route, see [Install](install.md). Continue
with the [example gallery](example-gallery.md) to choose between a runnable
solve and a validate-only configuration.

## 4. Know Which YAML Files Are Runnable

PhAST contains several kinds of YAML file:

| YAML category | Runnable command | Purpose |
|---|---|---|
| `examples/<family>/<case>/config.yaml` | `python -m phast run <path>` | Recommended starting point; a complete example-local solver input. |
| `configs/benchmarks/<family>/<case>.yaml` | `python -m phast run <path>` | Complete benchmark solver input. |
| `configs/REFERENCE.yaml` | Not intended for execution | Field-by-field reference and template. |
| `examples/PUBLIC_EXAMPLES_CONTRACT.yaml` | Not a solver input | Documentation and artifact inventory for public examples. |
| `configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml` | Requires `--validation-id` | Dispatcher manifest for beta validation scripts, not a single fracture deck. |
| `configs/phast.schema.json` | Not YAML and not executable | JSON Schema for editors and external validation. |

If a file is not named `config.yaml` and is described as a contract, manifest,
schema, or reference template, consult its README before passing it to
`python -m phast run`.

## 5. Run A Small End-To-End Example

Use the linear-elastic plate to verify mesh construction, solver execution,
artifact writing, and result loading:

```bash
python -m phast run examples/solid_mechanics_beta/linear_plate/config.yaml \
  --output_dir runs/linear_plate
```

The example is a supporting solid-mechanics check rather than a phase-field
fracture validation case. It is used here because it provides a compact first
execution. It writes response data, metadata, manifests, and field plots; it
does not retain reloadable displacement or stress arrays in a trajectory store.
The expected finite-element tip displacement is approximately
`-2.024e-6 m`, about `-14.98%` relative to the documented Euler-Bernoulli
estimate. Reproducing that recorded value indicates that the compact example
followed its expected route; it is not a general accuracy criterion.

Inspect the completed result without rerunning the solver:

```python
import phast

result = phast.load_result("runs/linear_plate")
print(result.metadata())
print(result.manifest())
print(result.history_names())
print(result.visuals())
```

`phast.load_result` reports only artifacts present in the result directory. It
does not synthesize fields that were not written.

## 6. Run A Phase-Field Fracture Example

Begin by validating and explaining the selected example:

```bash
python -m phast run examples/quasistatic/miehe_tension/config.yaml --validate-only
python -m phast explain-config examples/quasistatic/miehe_tension/config.yaml
```

Then execute it into a separate result directory:

```bash
python -m phast run examples/quasistatic/miehe_tension/config.yaml \
  --output_dir runs/miehe_tension
```

Fracture examples may require substantially more time and memory than the
linear-elastic installation check. Review the example README, mesh resolution,
device, and output settings before starting a complete benchmark rerun.

## 7. Understand The Solver Sequence

For a fracture simulation, PhAST:

1. parses and validates the YAML or `phast.Problem` definition;
2. constructs or imports a two-dimensional finite-element mesh;
3. evaluates element-level mechanical quantities with PyTorch tensor operators;
4. advances explicit dynamics or solves quasi-static mechanical equilibrium;
5. updates the tensile history field used to enforce crack irreversibility;
6. solves the AT1, AT2, or documented beta damage formulation;
7. enforces damage bounds and configured boundary conditions; and
8. writes fields, histories, manifests, configuration provenance, and visuals.

Read [User Guide Overview](user_guide/overview.md) for the software pathway and
[Physics, Units, and Formulation](user_guide/physics.md) for the governing
equations and staggered algorithm.

## 8. Create A New Setup

To generate a starter YAML:

```bash
python -m phast new my_benchmark --type quasi_static --material pmma_bleyer
```

Validate and inspect it before execution:

```bash
python -m phast run my_benchmark.yaml --validate-only
python -m phast explain-config my_benchmark.yaml
```

Alternatively, use the fluent `phast.Problem` interface while constructing a
model programmatically. The [problem setup guide](user_guide/setup_problems.md)
maps common finite-element concepts to both YAML and the Python API.

## 9. Platform And Optional Backend Notes

| Platform | Guidance |
|---|---|
| CPU | Recommended for installation verification, small examples, and reproducibility checks. |
| Linux with CUDA | Install a PyTorch wheel compatible with the local driver before installing PhAST. Use CUDA for cases whose documented pathway supports it. |
| Apple Silicon | MPS is visible to PyTorch, but CPU `float64` remains the conservative choice for verification-sensitive fracture calculations. |
| Windows | The base Python installation may be used directly; WSL2 is often more convenient for CUDA and Unix-oriented research workflows. |
| HPC systems | Use site-provided modules and install optional PETSc/MUMPS or CUDA libraries only after confirming binary compatibility. |

Backend availability is machine-dependent. Do not infer PETSc/MUMPS, cuDSS, or
AmgX support from package installation alone; confirm it with
`python -m phast doctor` on the target machine.

## 10. If You Become Stuck

Consult [Troubleshooting](troubleshooting.md), then
[open a GitHub issue](https://github.com/CEMS-Lab/PhAST/issues/new/choose) if
the problem remains or the instructions are unclear. Include:

- the exact command;
- the YAML path;
- operating system and Python/PyTorch versions;
- relevant `python -m phast doctor` output; and
- the first warning or traceback.

Students and first-time users are explicitly welcome to ask installation and
usage questions. An unclear step is a documentation defect worth reporting.

## 11. Build The Documentation

```bash
python -m pip install -r requirements-docs.txt
sphinx-build -W -b html docs docs/_build/html
```

The hosted documentation is available at
<https://cems-lab.github.io/PhAST/>.
