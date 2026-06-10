# Contributing

This project is moving toward a customer-ready engineering solver, so changes
need to preserve both software behavior and the mechanics assumptions behind
the examples.

## Development Setup

```bash
git clone https://github.com/CEMS-Lab/PhAST.git
cd phast
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For workstation or HPC validation, install optional extras only when the
machine supports them:

```bash
pip install -e ".[hpc,dataset]"
```

PETSc, MUMPS, cuDSS, AmgX, and vendor solvers are optional backend checks.
They should not be required for the default CPU confidence suite.

## Pull Request Checklist

- Keep raw HPC outputs, meeting packs, private notes, and generated heavy media
  out of git.
- Prefer Zarr for new solver trajectories. H5 is legacy input/conversion
  support only unless a compatibility test explicitly needs it.
- Update `CHANGELOG.md` for user-visible behavior, benchmark status, config
  migrations, or documentation changes.
- Update the relevant README or doc page when changing a workflow, CLI flag,
  YAML field, benchmark assumption, or output artifact.
- For benchmark YAMLs, record acceptance metadata and add the config to
  `configs/status/`.
- For physics changes, document the reference implementation, paper, or
  manufactured check used to justify the setup.

## Validation Tiers

Run the fastest relevant tier before opening a PR:

```bash
pytest -q
```

For benchmark or HPC changes, use explicit tiers instead of making normal CI
run long jobs:

```bash
pytest -q -m benchmark
pytest -q -m hpc
```

If a change touches output formats, run a small Zarr-producing example and
verify post-processing reads the same directory without manual conversion.

## Issue Hygiene

When closing or rewriting issues, state whether the fix is implemented,
documented, deferred to an optional backend, or superseded by a newer design.
Avoid closing stale wording silently when the underlying physics or validation
claim is still unresolved.
