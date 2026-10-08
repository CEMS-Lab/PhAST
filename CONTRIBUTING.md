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

PETSc, MUMPS, cuDSS, AmgX, and vendor solvers are optional. They are not
required for the standard CPU test suite.

## 2. First Contribution

Before changing solver behavior, a first contribution can be documentation,
an example clarification, a focused test, or an actionable bug report. A
minimal contribution route is:

1. Fork the repository and create a descriptively named branch.
2. Reproduce the documented behavior on the smallest relevant example.
3. Change one coherent concern and update adjacent documentation.
4. Run the narrowest checks listed below.
5. Open a draft pull request and state commands not run and why.

All participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## 3. Coding Standards

- **Type hinting**: All new Python functions must use strict type hints.
- **Device safety**: Ensure tensor operations are device-agnostic (`cpu`/`cuda`).
- **Autograd compatibility**: Preserve and document autograd compatibility for
  supported smooth tensor pathways. Identify nonsmooth history updates,
  projections, bounds, active sets, and external sparse-backend boundaries.

## 4. Pull Request Lifecycle

- PhAST is maintained under the CEMS Lab public repository. Contributions are
  welcome, but public changes are merged only after maintainer review and
  approval.
- Run the relevant validation commands locally.
- Keep large volumetric datasets (Zarr/H5), local debugging logs, and generated
  media out of git.
- Compare changes to physics kernels against an appropriate analytical,
  numerical, or published reference problem.
- Update relevant documentation, such as `README.md`, YAML schemas, example
  READMEs, and capability pages, with user-facing changes.

## 5. Validation

Run the narrowest relevant checks before opening a pull request:

```bash
python -m pytest -q tests
PYTHONPATH=src python -m phast doctor
sphinx-build -W -b html docs docs/_build/html
```

For changed YAML examples or benchmark configs, also run:

```bash
PYTHONPATH=src python -m phast run <config.yaml> --validate-only
```

For generated visualisations or example results, inspect the output folder
and update the relevant README or public contract file. The public repository
includes the tests intended for public review. Run any additional,
project-specific checks documented in the pull request or issue that motivated
the change.

## 6. Documentation Contributions

Documentation source lives in `docs/` and is built with Sphinx/MyST. Example
folders also contain public-facing `README.md` files, so changes to example
commands, inputs, outputs, or visuals usually require both docs and example
README updates.

All documentation, README, tutorial, and notebook prose must follow
[`DOCUMENTATION_STYLE.md`](DOCUMENTATION_STYLE.md). The guide defines the
project's academic tone, terminology, capability labels, command conventions,
and checks for avoiding internal development language in public text.

New YAML inputs, example READMEs, and tutorials must additionally follow
[`CONFIGURATION_STYLE.md`](CONFIGURATION_STYLE.md). Keep the same schema-2
section order across supported student problems, define units and editable
choices, and use the explain/check/run sequence. A common layout does not
imply an execution adapter: require adapter and command checks before
promotion. Preserve numerical inputs in documentation-only edits. Frozen
course assets and separate 3D research are not migration targets.

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

AI-assisted contributions are welcome when they follow `DOCUMENTATION_STYLE.md`,
`AGENTS.md`, `llms.txt`,
`.cursorrules`, and `docs/agent-contribution-guide.md`. Agents should verify
commands where possible and must not invent solver capabilities, benchmark
results, paper metadata, or local/HPC provenance.

## 7. Adding Examples

To add a simulation to the public `examples/` gallery, follow
`docs/user_guide/example_contract.md`. In short, curated examples need a flat
folder with `README.md`, `config.yaml`, a fluent Python companion when
available, manifests, lightweight CSV outputs, setup/final-state visuals, and
an evolution animation appropriate to the physics.

The README should document the problem definition, exact run command, expected
results, scope of validation, and result-inspection snippet. Do not commit raw
HPC run trees, large H5/Zarr stores, or unpublished diagnostic archives.

## 8. Trajectory Storage Convention

New trajectory and dataset writers, example configurations, notebooks, and
documentation must use HDF5 (`training_data.h5`) by default. Zarr is an explicit
opt-in, not a fallback. Keep existing Zarr readers and previously generated
artifacts intact. Match the selected format in output manifests and follow
the [example contract](docs/user_guide/example_contract.md).

## 9. Asking For Help

Open an issue if you are unsure how to install PhAST, interpret a configuration,
run an example, or contribute a change. A useful help request includes the
command, configuration path, PhAST version or commit, operating system, Python
and PyTorch versions, and the complete first warning or traceback. It is
acceptable to open an issue before diagnosing the solver internals.
