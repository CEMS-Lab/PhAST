# Modular fracture problems and learned damage updates

This tutorial describes how the main components of a phase-field fracture
problem fit together in PhAST. It also introduces the experimental
learned-damage plug-in without assuming prior knowledge of the codebase.

## 1. The finite-element problem

A runnable problem combines:

1. A mesh defining geometry, nodes, elements, and physical boundary groups.
2. A material defining elasticity, fracture resistance, regularization length,
   density, crack-density model, degradation law, and energy split.
3. Initial conditions defining displacement, velocity, damage, and any initial
   crack.
4. Boundary conditions constraining mechanical or phase-field degrees of
   freedom.
5. Loading prescribing the time-dependent displacement or force history.
6. Solver settings choosing integration, linear algebra, and damage update.
7. Output settings selecting fields and sampling frequency.

Keep reusable configurations under `configs/`. Schema, manifest, contract, and
template files are machine-readable specifications rather than simulations.

## 2. Distinguish the fracture-model choices

- `pf_model: AT1` or `AT2` selects the crack-density functional.
- `energy_split` selects the elastic-energy decomposition. Implementations
  include `spectral`, `spectral_stress`,
  `spectral_plane_stress_condensed`, `isotropic`, `amor`, and `star_convex`.
- `degradation_type` selects the stiffness-degradation family.

The supported coupled AT1/AT2 route uses `degradation_type: standard`.
PF-CZM is experimental and constructs its rational law from its own
parameters. Although the material layer contains additional degradation
functions, PhAST does not present cubic or rational degradation as supported
coupled AT1/AT2 damage solves.

## 3. Define a classical problem

This fragment illustrates the responsibilities. Start from a complete example
because mesh and physical-group names are geometry-specific.

```yaml
schema_version: 1

problem:
  name: introductory_fracture_problem

geometry:
  mesh_path: meshes/specimen.msh

material:
  E: 210000.0
  nu: 0.30
  rho: 7.8e-9
  Gc: 2.7
  l0: 0.02
  pf_model: AT2
  degradation_type: standard
  energy_split: spectral

boundary_conditions:
  - type: fix
    nodes: bottom
    component: 0
    value: 0.0
  - type: fix
    nodes: bottom
    component: 1
    value: 0.0

loading:
  protocol: simple
  num_steps: 100

solver:
  solver_type: quasi_static
  backend: auto
  damage_update: classical

device:
  device: null

output:
  print_every: 10
  trajectory: true
```

Validate and run a complete configuration:

```bash
python -m phast run path/to/problem.yaml --validate-only
python -m phast run path/to/problem.yaml \
  --output_dir runs/introductory_fracture_problem
```

## 4. How automatic route selection works

Automatic device and sparse-backend settings select a deterministic route from
the backends available in the current environment. This is a portability
policy, not a universal fastest-solver guarantee. Performance depends on
problem size, sparsity, hardware, installed libraries, and transfer costs.

At startup and completion, PhAST reports the device, configured and resolved
sparse backend, damage-update mode, predictor, and learned-update
acceptance/fallback counts. Preserve this report with benchmark results.

```bash
python -m phast doctor
```

## 5. Use a learned model as a proposal

```yaml
solver:
  damage_update: learned_proposal
  damage_predictor: examples.learned_damage.predictor_plugin:create_predictor
  damage_checkpoint: path/to/model.pt
  damage_predictor_options:
    representation: damage_increment
```

The predictor receives a `DamageStepContext` containing coordinates,
connectivity, displacement, velocity, element and nodal history, previous
damage, material metadata, formulation choices, time, and load factor. The
reference TorchScript adapter uses:

```text
[x, y, history field, previous damage, displacement x, displacement y]
```

This six-column matrix is only the reference adapter. The public protocol does
not impose one architecture or tensor layout. A plug-in can access raw
coordinates and connectivity, call `context.graph_edge_index()`, construct
node-element incidence operators, retain temporal state, project fields to a
regular grid, and apply checkpoint-specific normalization. This accommodates
graph, coordinate-conditioned, temporal, grid-operator, and reduced-basis
research without placing Paper-specific models in the solver core.

A `DamagePrediction` declares `representation="damage"` or
`representation="damage_increment"`. The latter is added to the previous
accepted damage before projection and audit.

PhAST projects the proposal onto damage bounds, irreversibility, and prescribed
phase-field values, then uses it as the initial guess for the classical damage
solve. The converged classical update remains authoritative.

## 6. Experimental learned replacement

```yaml
solver:
  damage_update: learned_replacement
  damage_predictor: examples.learned_damage.predictor_plugin:create_predictor
  damage_checkpoint: path/to/model.pt
  damage_residual_rtol: 1.0e-3
  damage_residual_atol: 1.0e-8
  damage_bound_tolerance: 1.0e-8
  damage_fallback: true
```

Replacement is accepted only after shape, finite-value, damage-bound,
irreversibility, phase-field Dirichlet, and projected-residual checks. If a
check fails, the default route is an exact classical fallback. Disabling
fallback turns rejection into an error and should be reserved for debugging.

These safeguards are not evidence that a learned model is accurate or
generalizes. Scientific claims require independent comparisons against matched
finite-element trajectories and complete reporting of acceptance, rejection,
and fallback events.

## 7. A defensible student workflow

1. Run an existing classical example without changing its formulation.
2. Modify one component at a time: mesh, material, loading, or energy split.
3. Examine the route report and retained displacement, damage, and history.
4. Generate training data with the constitutive and discretization choices
   intended at inference.
5. Introduce a learned model through `learned_proposal`.
6. Compare the corrected trajectory with an independent classical reference.
7. Study `learned_replacement` only after fallback reporting is understood.

The plug-in is architecture-neutral. Students and developers are invited to
propose adapters, tutorials, and audited examples. If an installation step,
configuration field, or solver message is unclear, open a GitHub issue with
the configuration, environment report, and minimal reproduction.
