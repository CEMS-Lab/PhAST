# Showcase

This page highlights public capabilities. Cross-code timing ratios are under
release refresh and are not listed until the benchmark environments are rerun
and archived.

## Sparse direct vs CG inner solve

| Benchmark | DOFs  | Public status |
| --------- | ----- | ------- |
| Miehe SENT | 16,000 | Refresh pending |
| Miehe SENS | 26,000 | Refresh pending |

Driver: [`sparse_solve`](api/sparse_solve.md). Any future speedup claim should
state the backend, device, thread count, mesh, tolerance, and run metadata.

## Mixed-precision CG with iterative refinement

[`mixed_precision_cg`](api/mixed_precision_cg.md) runs the inner Krylov loop
in float32 and refines the residual in float64. Final residual beats the
pure-float64 baseline on every problem in the demo.

See `examples/solid_mechanics/mixed_precision_cg_demo.py` and the rendered
plot.

## Hulbert-Chung generalized-alpha

[`time_integrators`](api/time_integrators.md) ships a Hulbert-Chung
generalized-alpha integrator with explicit rho-infinity control, suitable
for small dynamic reference problems. Production YAML dynamics currently uses
`central_difference` / Velocity-Verlet; `generalized_alpha` is tracked as the
future COMSOL-style implicit path in #570.

See `examples/solid_mechanics/dynamic_oscillator_genalpha.py`.
