## Summary

- 

## User-Facing Impact

- [ ] No user-facing behavior change
- [ ] CLI/config behavior changed
- [ ] Solver physics or numerical method changed
- [ ] Output artifacts or file formats changed
- [ ] Documentation/examples changed

## Validation

- [ ] `python -m pytest -q tests`
- [ ] `PYTHONPATH=src python -m phast doctor`
- [ ] `sphinx-build -W -b html docs docs/_build/html`
- [ ] Relevant configs passed `python -m phast run <config> --validate-only`
- [ ] Targeted solver/example check:
- [ ] Visual/output artifacts checked:

## Documentation Checklist

- [ ] User-facing behavior changes are reflected in `README.md`, `docs/`, or
      the relevant example `README.md`.
- [ ] New or changed commands in documentation were run, or the reason for not
      running them is stated in this PR.
- [ ] Example documentation follows `docs/user_guide/example_contract.md` when
      curated example files or visuals changed.
- [ ] Capability wording remains aligned with the supported, beta,
      experimental, and scaffold classifications.
- [ ] Validation-only commands are not presented as completed simulations.

## Physics and Benchmark Notes

For solver, material, element, benchmark, or post-processing changes, describe
the reference equation, paper, codebase, or manufactured test used to justify
the implementation.

## Artifact Hygiene

- [ ] No raw HPC result bundles, unpublished meeting material, or heavy generated
      media were added to git.
- [ ] New trajectory output is Zarr-first, or legacy H5 use is explicitly
      compatibility-only.
- [ ] New/changed benchmark configs are documented in the relevant README or
      contract.
