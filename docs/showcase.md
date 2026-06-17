# Showcase

Representative public workflows and solver components. Performance figures are
included only when their source CSVs and provenance are retained.

## Dynamic timing comparison

![Dynamic timing comparison](../assets/dynamic_timing_comparison.png)

This figure is regenerated from the current retained SENT and Kalthoff-Winkler
timing CSVs using Akantu, FEniCS, and PhAST final timing traces. Treat it as a
reproducibility artifact for Paper-1 performance discussion, not as a universal
hardware-independent claim. The source summary CSV is kept at
`assets/dynamic_timing_comparison.csv`.

## Sparse direct vs CG inner solve

| Workflow | Public route | Evidence to keep |
| -------- | ------------ | ---------------- |
| Miehe tension | `python -m phast run examples/quasistatic/miehe_tension/config.yaml` | run manifests, CSV histories, damage animation |
| Notched-holed plate | `python -m phast run examples/quasistatic/notched_holed_plate/config.yaml` | setup preview, response plot, final damage |

Driver: [`sparse_solve`](api/sparse_solve.md). Speedup is wall-time of the
sparse direct path versus the matrix-free CG inner-solve path on the same mesh,
tolerance, backend stack, and output settings. See
[`Performance and Reproducibility`](performance_reproducibility/index.md) for
the reporting checklist before publishing numbers.

## Mixed-precision CG with iterative refinement

[`mixed_precision_cg`](api/mixed_precision_cg.md) runs the inner Krylov loop
in float32 and refines the residual in float64. Final residual beats the
pure-float64 baseline on every problem in the demo.

See `examples/solid_mechanics/mixed_precision_cg/run.py` and
`examples/solid_mechanics/mixed_precision_cg/response.png`.

## Hulbert-Chung generalized-alpha

[`time_integrators`](api/time_integrators.md) ships a Hulbert-Chung
generalized-alpha integrator with explicit rho-infinity control, suitable
for small dynamic reference problems. Production YAML dynamics currently uses
`central_difference` / Velocity-Verlet; `generalized_alpha` is tracked as the
future COMSOL-style implicit path in #570.

See `examples/solid_mechanics/generalized_alpha_oscillator/run.py` and
`examples/solid_mechanics/generalized_alpha_oscillator/response.png`.
