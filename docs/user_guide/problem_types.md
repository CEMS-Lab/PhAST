# Problem types & solver selection

Solvers shipped by phast, when each is the right choice, and which
problem classes are supported beyond phase-field fracture.


The YAML run loop streams a `results.csv` with columns
`step, time, displacement, reaction_kN, max_d, max_H, stagger_iter,
elapsed_ms`, line-buffered so a killed run preserves usable data.
`compare.py` consumes this directly.

## Supported Problem Types

The solver handles several problem classes beyond phase-field fracture:

| Problem Type | Solver | `solver_type` | Status | Use Case |
|-------------|--------|---------------|--------|----------|
| Nonlinear quasi-static | `QuasiStaticSolver` | `quasi_static` | **Primary** — current default for new quasi-static benchmarks | Newton-Raphson with matrix-free CG or sparse-direct mechanics; spectral direct uses a frozen-state secant tangent |
| Nonlinear quasi-static | `SecantCGSolver` | `quasi_static_legacy` | Retained for the standalone `examples/quasistatic/*` benchmark drivers | Frozen-secant CG; supports spectral/amor without conjugacy break |
| Explicit dynamics | `ExplicitDynamics` | `explicit` | **Active** — used by all dynamic benchmarks | Impact, wave-driven fracture, rapid data generation |
| Linear static equilibrium | `StaticSolver` | `static` | Internal — pre-strain initialization only | Single load step with d=0 (called by StaggeredSolver) |
| Nonlinear quasi-static | `LBFGSSolver` | `lbfgs` | Available — not used by benchmarks | Gradient-only minimization; useful when matvec is unavailable or expensive |

### Solver Selection Guide

**Which solver should I use?**

For displacement-controlled quasi-static fracture (SENT, SENS, TPB, L-shaped
panel, and most standard benchmarks in the literature), use the default
**`QuasiStaticSolver`** via `solver_type='quasi_static'` (Newton-Raphson + CG
with the autograd-JVP spectral-split tangent landed in PR #170). For
configurations that depended on the older frozen-secant CG path,
**`SecantCGSolver`** is retained verbatim under `solver_type='quasi_static_legacy'`
(repointed 2026-04-29; see `staggered_solver.py:297-310`).

Use **`ExplicitDynamics`** only when inertial effects are physically relevant
(impact loading, blast, Kalthoff-Winkler). Explicit dynamics (Velocity-Verlet,
equivalent to Newmark-β with β=0, γ=½, also called central difference) introduces
stress-wave artifacts in quasi-static problems and requires CFL-limited timesteps
(~1000+ steps vs ~150 for quasi-static).

**Why two quasi-static paths?** `QuasiStaticSolver` uses a standard
Newton-Raphson + CG with the spectral-split tangent assembled via autograd-JVP,
matching FD to ~1e-7 across all four energy splits (PR #170). `SecantCGSolver`
freezes eigenvector projectors and runs a perfectly linear CG matvec, which has
historically been the more robust path on stiff spectral/amor problems; it is
also the path that carries the iterative-CG `rigid_connector` MPC support
(PR #182).

**Why not `LBFGSSolver`?** L-BFGS uses only gradient evaluations (no matvec),
which is appealing when assembling the tangent is expensive. However, for the
well-conditioned linear subproblems AT2 produces after secant linearization,
CG with Jacobi/multigrid preconditioning converges faster. L-BFGS is retained
for potential use in problems where the tangent operator is unavailable.


