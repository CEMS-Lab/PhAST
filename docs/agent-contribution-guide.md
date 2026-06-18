# Agent Contribution Guide

This guide is for AI agents and automated tools that propose improvements to
PhAST. Contributions are welcome, but this is scientific software: changes must
preserve reproducibility, numerical meaning, and public claim boundaries.

## What Agents Can Improve

- Documentation clarity, broken commands, stale links, and renamed files.
- Example READMEs, YAML decks, fluent API companions, and lightweight visuals.
- Validation scripts for public examples, config parsing, result loading, and
  artifact contracts.
- Post-processing utilities, plotting consistency, result manifests, and
  result-inspection workflows.
- Small, behavior-preserving refactors that remove duplication.
- Performance improvements that preserve PyTorch autograd, dtype, device, and
  solver semantics.
- Packaging, install, CI, issue templates, and contributor ergonomics.

## What Agents Must Not Invent

- New benchmark pass/fail claims.
- New timing or accuracy numbers.
- Unsupported solver capabilities.
- Paper DOI, arXiv identifiers, journal details, or publication status.
- Local HPC paths, raw cluster logs, proprietary solver artifacts, or heavy
  trajectory stores.

## Required Context

Before editing, inspect the relevant source of truth:

| Topic | Source of truth |
|---|---|
| Supported physics and maturity | `docs/user_guide/capability_matrix.md` |
| Example folder contents | `docs/user_guide/example_contract.md` |
| YAML execution | `docs/user_guide/yaml_workflow.md` |
| Python/fluent API | `docs/user_guide/python_api.md` |
| Result loading | `docs/user_guide/public_api_reference.md` |
| Public examples | `examples/README.md` and each example `README.md` |
| PyTorch/autograd constraints | `.cursorrules` |
| Citation | `CITATION.cff` |

## Solver-Code Guardrails

- Preserve differentiability unless the value is strictly telemetry or
  post-processing.
- Do not insert `.item()`, `.numpy()`, `.tolist()`, or CPU/GPU transfers in
  differentiable forward paths.
- Preserve float64 reference behavior unless a documented mixed-precision path
  explicitly allows otherwise.
- Avoid in-place tensor mutation in autograd-sensitive code.
- Do not change constitutive models, damage evolution, boundary-condition
  semantics, or convergence behavior without an explicit validation plan.
- If a refactor touches shared solver behavior, validate at least one relevant
  public YAML deck with `--validate-only` and state what was not run.

## Documentation and Example Style

- Use precise scientific language.
- Prefer “supported”, “beta”, “scaffold”, or “unsupported” over vague claims.
- Prefer “reproduce this public example” over “guarantees validation” unless a
  comparison report and acceptance metric are present.
- Keep instructions command-first and runnable from the repository root.
- Avoid internal process language such as “customer”, “private archive”,
  “smoke workflow”, or local machine paths.
- When changing example inputs, outputs, visuals, or commands, update the
  example README.

## Checks

For documentation-only changes:

```bash
sphinx-build -W -b html docs docs/_build/html
```

For docs or examples that mention a YAML config:

```bash
PYTHONPATH=src python -m phast run <config.yaml> --validate-only
```

For environment or packaging changes:

```bash
PYTHONPATH=src python -m phast doctor
```

If a command cannot be run because it is expensive or requires optional
hardware, say that explicitly in the pull request.
