# Contributing to PhAST

Welcome! PhAST is a PyTorch-native differentiable finite element framework for
phase-field fracture mechanics. Contributions are welcome to improve
performance, broaden physics coverage, and refine documentation. Please follow
the rigorous academic and engineering standards outlined below.

## 1. Development Setup

To set up your local development environment:

```bash
git clone https://github.com/CEMS-Lab/PhAST.git
cd PhAST
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For workstation or HPC validation, install optional extras only when the machine supports them:

```bash
pip install -e ".[hpc,dataset]"
```

PETSc, MUMPS, cuDSS, AmgX, and vendor solvers are optional backend checks and are not required for the default CPU confidence suite.

## 2. Coding Standards

- **Type Hinting**: All new Python functions must use strict type hints.
- **Device Safety**: Ensure tensor operations are device-agnostic (`cpu`/`cuda`).
- **Autograd Compatibility**: All new physics kernels must remain fully differentiable and support PyTorch's `autograd`.

## 3. Pull Request Lifecycle

- PhAST is maintained under the CEMS Lab public repository. Contributions are
  welcome, but public changes are merged only after maintainer review and
  approval.
- Run the relevant validation commands locally.
- Keep high-fidelity volumetric datasets (Zarr/H5), internal logs, and generated heavy media out of git.
- Provide parity checks against established analytical or commercial benchmarks when modifying physics kernels.
- Update relevant documentation, such as `README.md`, YAML schemas, example
  READMEs, and capability pages, with user-facing changes.

## 4. Validation

Run the fastest relevant checks before opening a pull request:

```bash
PYTHONPATH=src python -m phast doctor
sphinx-build -W -b html docs docs/_build/html
```

For changed YAML examples or benchmark configs, also run:

```bash
PYTHONPATH=src python -m phast run <config.yaml> --validate-only
```

For generated visuals or retained example artifacts, inspect the output folder
and update the relevant README or public contract file. The public repository
does not ship the full internal regression suite; run any project-specific test
commands documented in the pull request or issue that motivated the change.

## 5. Documentation Contributions

Documentation source lives in `docs/` and is built with Sphinx/MyST. Example
folders also contain public-facing `README.md` files, so changes to example
commands, inputs, outputs, or visuals usually require both docs and example
README updates.

Install the documentation dependencies:

```bash
pip install -r requirements-docs.txt
```

Build the documentation locally:

```bash
sphinx-build -W -b html docs docs/_build/html
```

Open the local build:

```bash
open docs/_build/html/index.html
```

Good documentation pull requests are small and verifiable. Prefer one topic
per PR: a broken command, a clearer explanation, a missing example note, a
fixed figure reference, or a capability-boundary correction. If a page documents
a runnable command, validate the command or state why it was not run.

When editing promoted examples, use `docs/user_guide/example_contract.md` as
the source of truth for required files, README content, visuals, and artifact
conventions.

AI-assisted contributions are welcome when they follow `AGENTS.md`, `llms.txt`,
`.cursorrules`, and `docs/agent-contribution-guide.md`. Agents should verify
commands where possible and must not invent solver capabilities, benchmark
results, paper metadata, or local/HPC provenance.

## 6. Adding Examples

To promote a local simulation to the public `examples/` gallery, follow
`docs/user_guide/example_contract.md`. In short, promoted examples need a flat
folder with `README.md`, `config.yaml`, a fluent Python companion when
available, manifests, lightweight CSV outputs, setup/final-state visuals, and
an evolution animation appropriate to the physics.

The README should document the problem definition, exact run command, expected
artifacts, claim boundary, and result-inspection snippet. Do not commit raw HPC
run trees, large H5/Zarr stores, or unpublished diagnostic archives.
