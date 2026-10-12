# Modular fracture problems

This tutorial describes how the main components of a phase-field fracture
problem fit together in PhAST without assuming prior knowledge of the
codebase.

| Lesson item | Scope |
|---|---|
| Prerequisites | Complete the installation checks and read the phase-field primer. |
| Route | Quasi-static staggered phase-field fracture through a checked-in YAML configuration. |
| Hardware | CPU is sufficient for configuration inspection and the bounded example route. |
| Verification | Inspect the selected route, result manifest, and convergence history. |

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
6. Solver settings choosing integration, linear algebra, and the damage
   subproblem.
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

## 3. Begin from a complete problem

Mesh groups, loading controls, and output requests are geometry-specific. Use
the checked-in Miehe tension configuration rather than a fragment containing
placeholder paths:

```bash
python -m phast explain-config examples/quasistatic/miehe_tension/config.yaml
python -m phast run examples/quasistatic/miehe_tension/config.yaml --validate-only
python -m phast run examples/quasistatic/miehe_tension/config.yaml \
  --num_steps 5 --output_dir runs/miehe_tension_five_step
```

The five-step calculation is a bounded workflow exercise, not a crack-growth
validation result. Read the example README before undertaking the full
calculation; it records the retained runtime and evidence boundary.

## 4. How automatic route selection works

Automatic device and sparse-backend settings select a deterministic route from
the backends available in the current environment. This is a portability
policy, not a universal fastest-solver guarantee. Performance depends on
problem size, sparsity, hardware, installed libraries, and transfer costs.

At startup and completion, PhAST reports the device, precision, element type,
mechanics solver, configured and resolved sparse backend, fracture model,
energy split, history update, damage-bound treatment, and preconditioner.
Preserve this report with benchmark results.

```bash
python -m phast doctor
```

## 5. A defensible student workflow

1. Run an existing example without changing its formulation.
2. Modify one component at a time: mesh, material, loading, or energy split.
3. Examine the route report and retained displacement, damage, and history.
4. Compare each modified run with the unmodified reference calculation.

Students and developers are invited to propose tutorials and examples. If an
installation step, configuration field, or solver message is unclear, open a
GitHub issue with the configuration, environment report, and minimal
reproduction.

Continue with [Heterogeneous material fields](05_heterogeneous_material_fields.md)
for an element-ordered material-field teaching problem, or return to the
[example gallery](../example-gallery.md) for complete workflows.
