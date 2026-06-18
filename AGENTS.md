# Agent Guide

This repository welcomes agent-assisted contributions to documentation,
examples, validation scripts, post-processing utilities, performance hygiene,
and solver code. Public scientific claims must remain traceable to the
checked-in code, examples, capability matrix, and citation metadata.

## Start Here

- Read `llms.txt` for repository orientation.
- Read `.cursorrules` before editing solver code or generated visuals.
- Read `CONTRIBUTING.md` before opening a pull request.
- For contribution lanes, prompt templates, and validation commands, follow
  `docs/agent-contribution-guide.md`.
- For promoted examples, follow `docs/user_guide/example_contract.md`.

## Good Agent Tasks

- Fix broken commands, stale imports, dead links, and renamed files.
- Expand theory documentation with clear equations, symbol tables, and links
  from mathematical notation to YAML keys and fluent API calls.
- Improve examples by adding clearer YAML, fluent API companions, README
  explanations, lightweight visuals, or validation commands.
- Improve post-processing and result-inspection ergonomics without changing
  solver claims.
- Add focused validation checks for public contracts, config parsing, result
  loading, and promoted example structure.
- Refactor clear duplication when behavior is covered by an explicit validation
  command.
- Improve performance only when tensor semantics, dtype/device behavior, and
  scientific equivalence are preserved.

## Guardrails

- Prefer small, reviewable pull requests.
- Do not invent benchmark results, solver capabilities, timings, references,
  or paper/publication details.
- Keep the distinction between production, beta, scaffold, and unsupported
  workflows exactly aligned with `docs/user_guide/capability_matrix.md`.
- If a command appears in documentation or an example README, run it or state
  why it was not run.
- If an example input, output, visual, or command changes, update the
  corresponding example `README.md`.
- Do not add raw HPC folders, heavy H5/Zarr stores, unpublished notes,
  proprietary solver files, or local absolute paths.
- Do not change mechanics kernels, damage evolution, material laws, or solver
  convergence behavior without a validation plan.
- Do not add new public APIs without documenting the intended workflow and
  capability boundary.

## Validation

Use the narrowest check that covers the change:

```bash
sphinx-build -W -b html docs docs/_build/html
PYTHONPATH=src python -m phast doctor
PYTHONPATH=src python -m phast run <config.yaml> --validate-only
```

For solver or tensor-code changes, also check dtype/device safety and avoid
breaking autograd-sensitive paths. Follow `.cursorrules` for PyTorch-specific
constraints.

`docs/_build/`, `runs/`, raw trajectory stores, and local solver outputs are
generated artifacts and must not be committed.
