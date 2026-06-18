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

## Contribution Lanes

Use one lane per pull request unless the changes are tightly coupled.

| Lane | Good changes | Required checks |
|---|---|---|
| Documentation polish | clearer wording, equations, theory-to-YAML tables, figure links, dead-link fixes | `sphinx-build -W -b html docs docs/_build/html` |
| Example maintenance | README updates, YAML cleanup, fluent companion scripts, visual manifests, lightweight PNG/GIF refreshes | `python -m phast run <example>/config.yaml --validate-only` plus docs build when README links change |
| Validation ergonomics | config checks, manifest checks, result-loading checks, clearer error messages | relevant `--validate-only` commands and focused Python checks |
| Post-processing | plotting consistency, figure generation, result readers, artifact summaries | run the changed script on a retained lightweight example or explain why data is unavailable |
| Solver performance | vectorization, allocation reduction, dtype/device cleanup, cached quantities | validate an affected public YAML deck and document any numerical tolerance used |
| Solver physics | constitutive updates, damage evolution, boundary conditions, convergence behavior | requires a validation plan, capability-matrix update, and reviewer attention from a human maintainer |

## High-Value Agent Tasks

Agents are especially useful for bounded improvements that are easy to verify:

- Expand theory pages with MyST/LaTeX equations, short derivations, and links
  from each symbol to YAML keys and fluent API calls.
- Add visual examples to documentation using existing public PNG/GIF assets.
- Compare each example README against `docs/user_guide/example_contract.md`.
- Check that every documented command runs from the repository root.
- Remove stale internal wording, local paths, generated caches, and unsupported
  claims from public-facing files.
- Improve diagnostics so invalid configs fail with actionable messages.
- Add small validation utilities that catch broken manifests, missing media,
  or stale README commands.

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

## Ready-to-Use Prompts

These prompts are intentionally narrow. Replace the paths in angle brackets.

### Documentation theory page

```text
Improve <docs/page.md> as a public Sphinx/MyST documentation page for PhAST.
Preserve all scientifically correct content, remove redundancy, and add clear
equations where helpful. Link theory symbols to YAML keys and phast.Problem API
terms. Use only public examples and public docs as evidence. Do not invent
benchmark results, timings, paper metadata, or unsupported capabilities. Run
sphinx-build -W -b html docs docs/_build/html before finishing.
```

### Example folder audit

```text
Audit <examples/...> against docs/user_guide/example_contract.md. The folder
must be flat and public-facing, with README.md, config.yaml, fluent Python where
available, lightweight setup/final-state visuals, appropriate animation, and
valid manifests. Remove or flag files outside the contract. Run
PYTHONPATH=src python -m phast run <examples/...>/config.yaml --validate-only
and update the README only with verified commands.
```

### Solver-code review

```text
Review <src/phast/...> for correctness, device/dtype safety, autograd safety,
unnecessary tensor allocation, avoidable CPU/GPU transfers, and duplicated
logic. Do not change physics behavior unless there is a validation plan. Prefer
small patches. Validate with PYTHONPATH=src python -m phast doctor and at least
one affected public config using --validate-only.
```

### Post-processing and visuals

```text
Improve <script_or_example> public visuals. Use existing public result files or
document the missing input data. Keep figures lightweight, white-background,
readable on GitHub, and aligned with .cursorrules. Update visual_manifest.json
when outputs change. Do not commit raw H5/Zarr trajectories.
```

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

## Pull Request Notes for Agents

Every agent-authored pull request should state:

- the contribution lane;
- the source-of-truth files that were checked;
- the exact commands run;
- any commands not run and why;
- whether scientific claims, public examples, or output artifacts changed.
