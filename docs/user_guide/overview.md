# User Guide Overview

`phast` is a matrix-free, differentiable PyTorch solver for 2D phase-field
fracture, explicit dynamics, and FEM benchmark workflows. It is intended for
reproducible fracture simulations, benchmark comparisons, and trajectory/dataset
generation from validated forward runs.

The core workflow is:

`YAML / phast.Problem` -> `Mesh` -> `Operators` -> `Solver` -> `Result bundle`

## Module Map

The public package is organized around the same workflow. The arrows show the
data path a typical public run follows, not Python import direction:

```text
YAML / phast.Problem
        |
        v
config + workflow adapters
        |
        v
mesh + geometry
        |
        v
physics operators
        |
        v
mechanics / damage / sparse solvers
        |
        v
result API + visualization + provenance
```

| Area | Main modules | Role |
|---|---|---|
| Authoring | `phast.core.problem`, `phast.workflow` | Fluent `Problem` objects and workflow contracts. |
| Configuration | `phast.config` | YAML parsing, schema validation, `run`, `doctor`, and `explain-config` entry points. |
| Mesh and geometry | `phast.core.mesh`, `phast.core.geometry_dsl`, `phast.core.geometry_compiler` | Mesh loading, declarative geometry parsing, Gmsh-backed geometry compilation, and mesh inspection. |
| Physics | `phast.physics`, `phast.core.fem_operators` | Materials, initial conditions, boundary conditions, fracture mechanics, and FEM tensor operators. |
| Solvers | `phast.solvers` | Mechanics, damage, staggered solves, sparse/direct solves, time integrators, and optional acceleration paths. |
| Results | `phast.result`, `phast.utils.visualization`, `phast.utils.provenance` | Result loading, visual outputs, metadata, manifests, and reproducibility records. |

## Design Principles

- **YAML for reproducibility:** public examples should be rerunnable from a
  declarative `config.yaml` and validated with `--validate-only`.
- **Fluent Python for authoring:** use `phast.Problem` when exploring a new
  setup, then preserve durable studies as YAML.
- **Tensor-first numerics:** mechanics and damage kernels stay close to PyTorch
  tensor operations so device, dtype, and autograd behavior remain inspectable.
- **Explicit evidence boundaries:** supported, beta, scaffold, optional-backend,
  and unsupported paths are documented in the capability matrix.
- **Result bundles over hidden state:** examples should retain the config,
  lockfile, manifests, CSV histories, and lightweight visuals needed to inspect
  the run.

## What the Project Is

The project combines:

| Component | Role |
|---|---|
| Tensor solver code | Mechanics, damage, time stepping, and selected differentiable solver operations in PyTorch. |
| YAML workflows | Reproducible benchmark and experiment definitions with validation and explain commands. |
| Provenance outputs | Standard result folders with resolved configs, lockfiles, diagnostics, and comparison artifacts. |
| Documentation/tests | Capability boundaries, examples, regression tests, and benchmark notes. |

The codebase is research-oriented and records supported, beta, experimental,
and unsupported status explicitly. Consult the capability matrix before using
a pathway for a publication, automated study, or external comparison.

## What It Is Not

`phast` is not a PINN package. It solves discretised finite-element
phase-field equations and can expose supported tensor computations to autograd,
but the public release is scoped to forward fracture workflows rather than
neural surrogates.

It is also not a full 3D fracture platform. The principal supported fracture
workflows use two-dimensional triangulated meshes.

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

## Solver Sequence

The public fracture pathway follows the same numerical sequence whether the
model originated from YAML or `phast.Problem`:

1. **Configuration and mesh.** PhAST validates the model, constructs or imports
   the mesh, resolves named regions, and checks material and boundary data.
2. **Element operators.** PyTorch gather-compute-scatter kernels evaluate
   strain, stress, internal force, energy density, and damage-operator actions.
3. **Mechanics.** Explicit runs advance displacement, velocity, and
   acceleration with the documented central-difference/Velocity-Verlet pathway.
   Quasi-static runs solve mechanical equilibrium with the configured
   matrix-free or sparse backend.
4. **Crack driving force.** The selected tensile/compressive energy split
   defines the crack-driving energy. A history field retains the maximum
   admissible driving state required by the irreversibility treatment.
5. **Damage.** The configured AT1, AT2, or documented beta formulation solves
   for the regularized damage field. Bounds, fixed-damage conditions, and
   irreversibility are applied according to the selected solver pathway.
6. **Coupling and convergence.** Quasi-static fracture alternates mechanics and
   damage until the configured staggered criterion is satisfied. Explicit
   dynamics updates damage at the configured cadence.
7. **Results and provenance.** The run writes the resolved configuration,
   lockfile, histories, requested fields, manifests, and lightweight visuals.

"Matrix-free" describes operator application on the principal pathway; optional
sparse-direct solvers and some preconditioners may assemble sparse matrices.
"Differentiable" applies only to supported tensor operations. Nonsmooth maximum
history updates, clipping, active-set changes, and external sparse libraries
require case-specific treatment when derivatives are interpreted.

## Where to Go Next

| Need | Page or path |
|---|---|
| Install and first run | `docs/getting-started.md` |
| Supported/unsupported status | `docs/user_guide/capability_matrix.md` |
| Problem and physics setup | `docs/user_guide/setup_problems.md`, `docs/user_guide/physics.md`, `docs/user_guide/configuration.md` |
| Theory-first visual guide | [Visual glossary](../tutorial/02_visual_glossary.md) | Read AT1/AT2, energy-split, and length-scale summaries before the equations. |
| Mesh and geometry notes | `docs/user_guide/meshes.md`, `docs/user_guide/geometry_gallery.md` |
| Sparse/direct backend notes | `docs/user_guide/sparse_solve.md` and `docs/api/sparse_solve.md` |
| Benchmarks | `docs/example-gallery.md` and the public example folders |
| Examples | `docs/user_guide/example_contract.md`, `examples/README.md`, and `docs/example-gallery.md` |

Documentation publication is initiated through the repository's GitHub Actions
workflow. Contributors should run the documented local checks before requesting
review.
