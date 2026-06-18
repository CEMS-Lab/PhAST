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

## Agentic Contribution Loop

Every agent contribution must follow this sequence.

1. **Acquire context.** Read `llms.txt`, `.cursorrules`, `CONTRIBUTING.md`, and
   the relevant documentation or example contract before editing.
2. **Classify the lane.** Identify the narrowest applicable lane:
   documentation, examples and visuals, validation and contracts, solver and
   performance, or public API.
3. **Inspect the current files.** Review the exact source, documentation, and
   example files that will be changed. Do not overwrite files without reading
   them first.
4. **Make a small change.** Prefer one coherent edit set per contribution.
   Preserve scientific meaning, public claim boundaries, and existing output
   contracts.
5. **Synchronize adjacent materials.** If a command, output, figure, example
   input, or public API changes, update the linked README, tutorial, gallery,
   or contract file in the same contribution.
6. **Verify.** Run the narrowest command set that covers the change. If a
   command is documented, run it or state why it could not be run.
7. **Review the result.** Check that the change is consistent with the
   capability matrix, that generated artifacts are not committed, and that the
   repository remains scientifically defensible.

### Lane Guide

| Lane | Typical scope | Minimum verification |
|---|---|---|
| Documentation | theory pages, tutorials, landing pages, README text | `sphinx-build -W -b html docs docs/_build/html` |
| Examples and visuals | example READMEs, YAML decks, lightweight figures, manifests | `PYTHONPATH=src python -m phast run <config.yaml> --validate-only` plus `sphinx-build -W -b html docs docs/_build/html` when docs links change |
| Validation and contracts | config checks, manifest checks, result loading, public example rules | focused `--validate-only` commands and a targeted Python check |
| Solver and performance | tensor code, device or dtype hygiene, allocation reduction | one affected public YAML deck and any required numerical sanity check |
| Public API | fluent interfaces, result loading, documented entry points | affected tests and documentation updates |

### Final Self-Check

- Confirm that no generated build output, trajectory store, or local scratch
  directory is staged for commit.
- Confirm that every new command or example reference is reflected in the
  relevant documentation.
- Confirm that no public-facing text overstates supported capabilities.
- Confirm that the change is small enough to review without reconstructing
  hidden context from issue history or private notes.

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

## Tone and Language
- **Academic and Formal**: All documentation, comments, and public-facing text must use a formal, academic tone suitable for a State-Of-The-Art (SOTA) FEM solver.
- **No Internal Slang**: Do NOT use software-development slang (e.g., "ships with", "hack", "we need to fix this later", "epic").
- **No Internal References**: Do NOT reference internal GitHub issues (e.g., "#107", "PR #146"), Jira tickets, or private workflow phases. Refer to features generically (e.g. "The framework supports...").
- **No Emoticons**: Do not use emojis or emoticons in the documentation or source code.
