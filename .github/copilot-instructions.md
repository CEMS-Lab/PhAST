# Copilot Instructions for PhAST

PhAST is a scientific Python/PyTorch finite-element solver. Treat code,
examples, documentation, and post-processing edits as part of the
reproducibility surface.

- Follow `AGENTS.md`, `llms.txt`, and `CONTRIBUTING.md`.
- For agent contribution guidance, follow `docs/agent-contribution-guide.md`.
- For PyTorch solver code, follow `.cursorrules`.
- Do not add claims that are not supported by the code, examples, or
  `docs/user_guide/capability_matrix.md`.
- Keep example documentation aligned with
  `docs/user_guide/example_contract.md`.
- Prefer commands using `python -m phast ...` from the repository root.
- Do not commit `docs/_build/`, `runs/`, raw H5/Zarr stores, or local absolute
  paths.
- When changing docs, run:

```bash
sphinx-build -W -b html docs docs/_build/html
```

- When changing configs or examples, run:

```bash
PYTHONPATH=src python -m phast run <config.yaml> --validate-only
```

- When changing solver code, preserve autograd, dtype, and device semantics.
