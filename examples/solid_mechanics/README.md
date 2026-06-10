# Solid Mechanics Tutorial Primitives

These are minimal end-to-end examples of the autograd-enabled
`SparseSolveAutograd` and `time_integrators` primitives, intended for
tutorial use rather than as validation benchmarks. They show the
shortest path from a problem statement to a gradient w.r.t. material
parameters or to a dissipation-controlled time march. Validation-quality
benchmarks (Cook's membrane, Cantilever, MMS convergence, etc.) are
tracked under #110 follow-ups.

## Demos

- **`linear_plate.py`** — plane-strain CST cantilever solved with
  `SparseSolveAutograd`. Run:
  ```
  python linear_plate.py
  ```
  Headline output: tip displacement, % match against the
  Euler–Bernoulli closed form, and `d(tip_disp)/dE` via autograd
  through the sparse solve.

- **`neohookean_plate.py`** — compressible neo-Hookean plate driven by
  load-stepped Newton iteration on top of `SparseSolveAutograd`. Run:
  ```
  python neohookean_plate.py
  ```
  Headline output: ~3 Newton iterations per load step, and a working
  gradient through 5 load increments (end-to-end autograd through the
  outer Newton loop).

- **`dynamic_oscillator_genalpha.py`** — 2-DOF spring-mass system
  integrated with the Hulbert–Chung generalized-α scheme. Run:
  ```
  python dynamic_oscillator_genalpha.py
  ```
  Headline output: with `ρ_∞ = 0.5` the high-frequency mode
  (`ω = 1000`) is dissipated to ~`1e-25` while the physical low mode
  (`ω = 1`) is preserved. PNG of the response is committed alongside
  (`dynamic_oscillator_genalpha.png`).

- **`j2_plasticity_bar.py`** — standalone material-point J2/von-Mises
  plasticity with linear isotropic hardening. Run:
  ```
  python j2_plasticity_bar.py
  ```
  Headline output: the equivalent von-Mises stress tracks
  `sigma_y0 + H * eps_p_eq` after yield, demonstrating the radial-return
  kernel. This is a constitutive demo only; coupled PF-plasticity remains
  a separate product-hardening item.

## Required imports

```python
from phast.sparse_solve import solve, SparseSolveAutograd
from phast.mixed_precision_cg import cg_mixed_precision
from phast.time_integrators import gen_alpha_step, gen_alpha_params
from phast.plasticity import J2Plasticity, J2State
```

## Related

- `DOCUMENTATION.md`, section "Sparse linear solver" — API reference
  for `SparseSolveAutograd` and the SuperLU backend.
- Issues:
  - #105 — solid-mechanics primitives epic
  - #106 — SciPy SuperLU sparse-solve with autograd-enabled adjoint
  - #110 — linear + nonlinear plate demos (this directory)
  - #102 — Hulbert–Chung generalized-α time integrator
  - #118 — mixed-precision CG
- Validation benchmarks: tracked under #110 follow-ups (not yet
  landed; these tutorial demos are not a substitute).
