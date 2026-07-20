# Contributing to PhAST

PhAST is a finite-element framework implemented in PyTorch for phase-field
fracture mechanics. Contributions from researchers, students,
scientific-software developers, and users are welcome. Useful contributions
include clearer documentation, reproducible examples, bug reports, numerical
verification, performance analysis, and carefully scoped solver improvements.
If any instruction or example is unclear, open a GitHub issue; questions from
new users are valuable documentation feedback.

## 1. Development Setup

To set up your local development environment:

```bash
git clone https://github.com/CEMS-Lab/PhAST.git
cd PhAST
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Install optional extras only when they are required and supported by the
machine:

```bash
pip install -e ".[dataset]"
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
- Keep high-fidelity volumetric datasets (Zarr/H5), local diagnostic logs, and generated heavy media out of git.
- Provide parity checks against established analytical or commercial benchmarks when modifying physics kernels.
- Update relevant documentation, such as `README.md`, YAML schemas, example
  READMEs, and capability pages, with user-facing changes.

## 4. Validation

Run the narrowest relevant checks before opening a pull request:

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
includes the tests intended for public review. Run any additional,
project-specific checks documented in the pull request or issue that motivated
the change.

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

Good documentation pull requests are focused and verifiable. Prefer one topic
per pull request: a broken command, a clearer explanation, a missing example
note, a fixed figure reference, or a capability-boundary correction. If a page
documents a runnable command, validate the command or state why it was not run.

When editing curated examples, use `docs/user_guide/example_contract.md` as
the source of truth for required files, README content, visuals, and artifact
conventions.

AI-assisted contributions are welcome when they follow `AGENTS.md`, `llms.txt`,
`.cursorrules`, and `docs/agent-contribution-guide.md`. Agents should verify
commands where possible and must not invent solver capabilities, benchmark
results, paper metadata, or local/HPC provenance.

## 6. Adding Examples

To add a simulation to the public `examples/` gallery, follow
`docs/user_guide/example_contract.md`. In short, curated examples need a flat
folder with `README.md`, `config.yaml`, a fluent Python companion when
available, manifests, lightweight CSV outputs, setup/final-state visuals, and
an evolution animation appropriate to the physics.

The README should document the problem definition, exact run command, expected
artifacts, evidence boundary, and result-inspection snippet. Do not commit raw
HPC run trees, large H5/Zarr stores, or unpublished diagnostic archives.

## 7. Asking For Help

Open an issue if you are unsure how to install PhAST, interpret a configuration,
run an example, or contribute a change. A useful help request includes the
command, configuration path, operating system, PyTorch version, and the first
warning or traceback. It is acceptable to open an issue before diagnosing the
solver internals.
