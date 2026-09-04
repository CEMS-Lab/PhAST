# Controlled exploration experiments

These experiments are intended for students who have completed the
[Getting Started](../getting-started.md) workflow and can inspect a PhAST result
directory. Each experiment changes one documented input while keeping the
remaining configuration fixed.

The experiments are study designs, not predicted benchmark outcomes. Record the
resolved configuration, selected solver route, runtime, convergence history,
field outputs, and comparison metric for every variant. Long dynamic examples
should first be checked with `--validate-only`; configuration acceptance does
not execute the simulation.

| Lesson item | Scope |
|---|---|
| Prerequisites | Completed installation checks and familiarity with YAML. |
| Baseline | An unchanged checked-in example and a separate copied configuration for each variant. |
| Hardware | CPU is sufficient for preflight; full dynamic runs can be expensive. |
| Outputs | Separate output directory, metadata, manifests, histories, and requested fields for each variant. |
| Verification | Compare one variable at a time against the unchanged baseline. |

All commands assume the repository root as the working directory.

## Prepare configuration copies

Create a dedicated directory before editing any example:

```bash
mkdir -p runs/config_variants
cp configs/benchmarks/dynamic/B7_dynamic_crack_branching_comsol.yaml \
  runs/config_variants/B7_baseline.yaml
cp configs/benchmarks/dynamic/B3_dynamic_sent.yaml \
  runs/config_variants/B3_baseline.yaml
```

Copy the relevant baseline again for each variant, change only the named YAML
field, and retain the edited file with its output. Do not edit the checked-in
baseline.

## 1. Poisson-ratio sensitivity

Change `material.overrides.nu` in copies of `B7_baseline.yaml`. Use values
that are admissible for the selected constitutive assumption and retain the
same mesh, load, time integration, output cadence, and solver controls.

```bash
python -m phast run runs/config_variants/B7_nu010.yaml --validate-only
python -m phast run runs/config_variants/B7_nu030.yaml --validate-only
python -m phast run runs/config_variants/B7_nu040.yaml --validate-only
```

For completed runs, compare crack-path geometry, branching time, energy
histories, stable time step, and wall-clock time. Do not assume the direction or
magnitude of a trend before measuring it. Values near incompressibility require
particular care, and negative Poisson ratios are not categorically unphysical;
their admissibility depends on the material model and study.

## 2. Energy-split comparison

Create copies of `B3_baseline.yaml` and change only
`material.overrides.energy_split` to an implemented choice such as
`isotropic`, `amor`, `spectral`, or `star_convex`. Consult the
[capability matrix](../user_guide/capability_matrix.md) before selecting a
specialized route.

```bash
python -m phast run runs/config_variants/B3_isotropic.yaml --validate-only
python -m phast run runs/config_variants/B3_amor.yaml --validate-only
python -m phast run runs/config_variants/B3_spectral.yaml --validate-only
python -m phast run runs/config_variants/B3_star_convex.yaml --validate-only
```

For completed runs, compare the tensile driving field, damage onset, crack
path, energy balance, and solver telemetry. Differences must be reported as
case-specific observations rather than general rankings of the splits.

## 3. Resolution relative to the regularization length

Create variants that change `geometry.parameters.h_crack` and
`material.overrides.l0` while retaining all other settings. The commonly used
starting diagnostic `h <= l0 / 2` is a resolution guideline, not a universal
convergence guarantee.

Before a full FEM study, use the
[mesh-resolution notebook](notebook_mesh_resolution.ipynb) to understand how
sampling changes with `h/l0`. A defensible numerical study should then compare
at least:

- the actual element-size distribution near the crack;
- damage-band width and orientation;
- load-displacement or energy response;
- crack path;
- solver convergence and computational cost.

The analytical notebook is not a replacement for mesh-convergence evidence.

## 4. Output-cadence comparison

Output cadence changes disk usage and can change wall-clock time. Compare dense
and sparse output on the same short execution window:

```bash
python -m phast run examples/dynamic/B3_dynamic_sent/config.yaml \
  --device cpu --num_steps 50 --output_dir runs/sent_dense
python -m phast run examples/dynamic/B3_dynamic_sent/config.yaml \
  --device cpu --num_steps 50 --h5_every 10 --output_dir runs/sent_sparse
```

Record total bytes, number of stored snapshots, runtime, and whether the
retained cadence resolves the event being studied. Output cadence should not
change the governing update, but output and transfer costs must be measured
rather than assumed.

## 5. Inspect result artifacts

A result directory should remain self-describing:

```bash
python - <<'PY'
import phast

for path in ("runs/sent_dense", "runs/sent_sparse"):
    result = phast.load_result(path)
    print(path)
    print(result.metadata())
    print(result.manifest())
    print(result.visuals())
    print(result.history_names())
PY
```

A visual file is not a reloadable numerical trajectory. Confirm that the
configuration, metadata, manifests, histories, mesh information, and requested
trajectory fields required by the comparison are present.

## 6. Damage-update cadence in explicit dynamics

The `solver.damage_every` setting controls how often the implicit damage
subproblem is updated relative to explicit mechanics steps. Create copied
configurations with values such as 1, 3, and 5, then validate each file before
running:

```bash
python -m phast run runs/config_variants/B3_damage_every_1.yaml --validate-only
python -m phast run runs/config_variants/B3_damage_every_3.yaml --validate-only
python -m phast run runs/config_variants/B3_damage_every_5.yaml --validate-only
```

Use `damage_every: 1` as the comparison baseline. For completed runs, measure
damage-solve time, total time, onset time, crack morphology, energy history, and
the number of accepted damage updates. No fixed speedup or accuracy equivalence
is implied; both depend on the problem, mesh, time step, and fracture model.

## 7. Anderson acceleration for staggered convergence

For a quasi-static copied configuration, compare the unaccelerated staggered
iteration against selected `anderson_depth` values:

```yaml
solver:
  solver_type: quasi_static
  anderson_depth: 3
```

Record staggered iterations, rejected or restarted updates, linear-solver work,
wall-clock time, and the final response. Anderson acceleration can help or harm
a particular fixed-point sequence. PhAST therefore does not assign a universal
percentage reduction or preferred depth.

## Reporting the experiment

For each comparison, retain:

1. The unchanged baseline configuration.
2. Every copied variant configuration.
3. The exact command and PhAST revision.
4. Device, dtype, resolved backend, solver and preconditioner.
5. Mesh and `h/l0` information.
6. Runtime and iteration-level telemetry.
7. Matched field, history, and signed-difference plots where applicable.
8. A statement distinguishing configuration preflight, completed execution,
   numerical verification, and scientific validation.

Continue with [Results and visualization](../user_guide/results_visualization.md)
for result interpretation and [Performance and reproducibility](../performance-reproducibility.md)
before publishing timing comparisons.
