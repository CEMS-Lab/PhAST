# User Guide Overview

`phast` is a 2D phase-field fracture framework built on PyTorch. It
is intended for reproducible fracture simulations, benchmark comparisons,
and trajectory/dataset generation from validated forward runs.

## What the Project Is

The project combines:

| Component | Role |
|---|---|
| Tensor solver code | Mechanics, damage, time stepping, and selected differentiable solver operations in PyTorch. |
| YAML workflows | Reproducible benchmark and experiment definitions with validation and explain commands. |
| Provenance outputs | Standard result folders with resolved configs, lockfiles, diagnostics, and comparison artifacts. |
| Documentation/tests | Capability boundaries, examples, regression tests, and benchmark notes. |

The codebase is research-oriented but keeps production/beta/experimental status
visible. Use the capability matrix before relying on a path for paper claims,
automation, or external comparison.

## What It Is Not

`phast` is not a PINN package. It solves discretised finite-element
phase-field equations and can expose supported tensor computations to autograd,
but the public release is scoped to forward fracture workflows rather than
neural surrogates.

It is also not a full 3D fracture platform. Current production workflows target
2D triangulated meshes.

## Major Workflows

| Workflow | Summary |
|---|---|
| Forward solver | Run explicit dynamic or staggered static/quasistatic phase-field fracture simulations from YAML. |
| Quasistatic benchmarks | Reproduce and compare benchmark cases using standard configs, compare scripts, and documented output conventions. |
| Trajectory datasets | Write Zarr-first trajectory and visualization outputs from reproducible forward runs. |

## Solver Coupling

Most fracture workflows alternate between mechanics and damage updates:

1. Given the current damage field, solve/update displacement and stress.
2. Given the current mechanical state and history field, solve/update damage.
3. Enforce irreversibility and boundary/loading rules.
4. Write outputs and continue until the load or time schedule is complete.

This staggered structure is easier to inspect, compare, and debug than a single
large monolithic solve. Experimental monolithic paths may exist, but
the staggered workflows remain the clearest default for benchmarked fracture
runs.

## Where to Go Next

| Need | Page or path |
|---|---|
| Install and first run | `docs/getting-started.md` |
| Supported/unsupported status | `docs/user_guide/capability_matrix.md` |
| Problem and physics setup | `docs/user_guide/setup_problems.md`, `docs/user_guide/physics.md`, `docs/user_guide/configuration.md` |
| Mesh and geometry notes | `docs/user_guide/meshes.md` |
| Sparse/direct backend notes | `docs/user_guide/sparse_solve.md` and `docs/api/sparse_solve.md` |
| Benchmarks | `docs/benchmarks/catalogue.md` and the public example folders |
| Examples | `docs/user_guide/example_contract.md`, `examples/README.md`, and `docs/example-gallery.md` |

GitHub workflows are currently manual-only. Run CI, docs, install checks, or
wheel builds from the GitHub Actions tab when needed.
