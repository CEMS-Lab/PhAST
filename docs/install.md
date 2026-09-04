# Install PhAST

PhAST supports Python 3.10, 3.11, and 3.12. The base installation includes
the dependencies needed by the public solver; PETSc/MUMPS, GPU direct solvers,
and other HPC backends remain optional.

## Tested platform matrix

| Platform | Python | Public CI status | Recommended first route |
|---|---:|---|---|
| Ubuntu | 3.10, 3.11, 3.12 | Package and public tests | CPU float64 |
| macOS | 3.11 | Package and public tests | CPU float64; inspect MPS separately |
| Windows | 3.11 | Package and public tests | CPU float64 |
| CUDA Linux | Environment-dependent | Not part of the portable CI matrix | Run `doctor`, sanitizer, then a bounded case |
| HPC optional backends | Site-dependent | Not part of the portable CI matrix | Validate against the site module and scheduler environment |

"Tested" describes the public GitHub Actions matrix. It does not imply that
every optional accelerator, compiler, MPI stack, or sparse backend is tested on
that platform.

## Virtual environment with pip

From a clone of the repository:

```bash
python3 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

For an isolated conda environment, the existing route is:

```bash
conda env create --file environment.yml
conda activate phast
```

Install documentation tooling separately when building the site:

```bash
python -m pip install -r requirements-docs.txt
```

## Platform activation

On Linux and macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use a process-scoped policy rather than a
machine-wide change:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

## Docker

The repository image is a portable CPU route for the bounded installation
check. It does not start a fracture simulation:

```bash
git clone https://github.com/CEMS-Lab/PhAST.git
cd PhAST
docker build --tag phast:local .
docker run --rm phast:local
```

Run the first completed example and retain its output on the host:

```bash
mkdir -p runs
docker run --rm \
  --volume "$PWD/runs:/opt/phast/runs" \
  phast:local \
  python -m phast run examples/solid_mechanics_beta/linear_plate/config.yaml \
  --output_dir runs/linear_plate
```

Open an interactive shell when inspecting the container environment:

```bash
docker run --rm -it --entrypoint /bin/sh phast:local
```

## Verify before running a simulation

Use the [verification ladder](verify-install.md): doctor, sanitizer,
`--validate-only`, then a deliberately selected completed run. Use an explicit
output directory for completed runs and inspect the resulting manifest and
metadata. A preflight pass does not establish runtime or scientific validity.

If installation fails, retain the first traceback and consult
[Troubleshooting](troubleshooting.md). Optional backend installation should be
deferred until the doctor shows that the intended workflow needs it.

## First-error recovery

| First error | Check |
|---|---|
| `python: command not found` | Install Python 3.10-3.12 and use `python3` on Unix or `py -3.11` on Windows. |
| `ModuleNotFoundError: phast` | Activate the environment and rerun `python -m pip install -e .` from the repository root. |
| Gmsh import or library error | Confirm that the environment contains one consistent Gmsh installation; avoid mixing Conda and Homebrew binary stacks. |
| CUDA/PyTorch incompatibility | Install the PyTorch build matching the local CUDA driver, then confirm it with `python -m phast doctor`. |
| Example path not found | Run repository-relative commands from the repository root. |
| Permission denied for output | Select a writable `--output_dir`; do not write into the installed package directory. |

Preserve the first traceback. Opening a question issue is appropriate whenever
the documented recovery step does not resolve it.
