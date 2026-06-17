# Implicit Solver Extension — Scoping Document

Issue: #260 (parent epic #105). Tracks the implicit-solver roadmap for
`phast`: full-Newton consistent tangent (#170 follow-up),
matrix-free geometric multigrid (#115/#116), and implicit-dynamic
generalised-α (#102). Companion sub-tickets: #106 SciPy SuperLU baseline,
#107 PETSc/MUMPS, #108 cuDSS, #117 p-multigrid, #118 mixed-precision
Krylov, #241 MUMPS symbolic-stage caching.

---

## §1 Current state diagnosis

The solver stack lives in three modules:

* `mechanics_solver.py::QuasiStaticSolver` (lines 543–960). Outer Newton
  loop on `R(u; d) = f_ext − f_int(u, d)` with the damage field frozen.
  Inner step is dispatched by `_resolve_backend`: SciPy SuperLU / PETSc
  MUMPS sparse-direct, or matrix-free preconditioned CG. The CG matvec
  uses two paths (lines 824–870): for the *isotropic* split,
  `f_int(·, d)` is linear in `u` so the operator action equals the
  consistent tangent action and CG is exact. For *spectral / amor /
  star_convex*, the matvec is an autograd JVP through `f_int` at the
  current iterate `u_lin = u.detach()` (PR #170, issue #114). The JVP
  *is* the consistent tangent at `u`; it differs from the secant
  operator (`fem.secant_matvec`) by the eigenvector-rotation term that
  arises when the spectral projectors `P_i(eps)` depend on `u`.
* `mechanics_solver.py::SecantCGSolver` (lines 1470–2040). Newton outer
  loop, but the inner CG matvec is *always* `fem.secant_matvec(p,
  state)` where `state = fem.freeze_secant_state(u, d)` (line 1660).
  This is the secant tangent. It supports the rigid-connector MPC
  (`_solve_impl_mpc`, line 1818) used by the cantilever / SENT meshes.
  Re-linearisation happens once per outer Newton iterate, and the
  multigrid preconditioner is updated via `self._multigrid.update(d,
  secant_state=state)`.
* `mechanics_solver.py::DirectSolver` (lines 1030–1470). Assembles the
  full secant stiffness on CPU float64 (`_assemble_stiffness`, called
  at line 1429) and routes through `_spsolve_auto` (CuPy on CUDA, SciPy
  on CPU). Newton outer loop with absolute (`tol`) and relative
  (`rtol`) checks (line 1414) and a stall guard at 0.999× monotonic
  decrease (line 1422).
* `sparse_solve.py::SparseSolveAutograd` /
  `_MumpsSparseSolveAutograd`. `torch.autograd.Function` wrappers
  exposing the SciPy and PETSc/MUMPS solves to autograd via the adjoint
  identity `∂L/∂A = −x ⊗ (Aᵀ\\∂L/∂x)` (#106).

**Differential–algebraic state.** The staggered driver
(`staggered_solver.py::StaggeredSolver.step_full`, line 551) alternates
mechanics and damage solves. The full residual the implicit path is
trying to drive to zero is

    R_u(u, d, λ)    = f_int(u, d) − λ f_ext(t)                    (1a)
    R_d(u, d, H)    = (G_c/ℓ) (d − ℓ² Δd) − 2(1−d) H(u)            (1b)
    R_H(u, H)       = max(H_prev, ψ⁺(u)) − H                      (1c)

with `λ` the load multiplier. (1a) is the mechanical equilibrium
solved by `*Solver.solve()`; (1b) is the AT2 damage equation
(`StaggeredSolver.step_solve_damage`, line 482); (1c) is the
Miehe–Welschinger–Hofacker history projector. The staggered loop fixes
`d` while solving (1a), then fixes `u` while solving (1b), iterating
until the dual residual `‖Δu‖ + ‖Δd‖ < tol` (line 619).

**Where the consistent tangent is missing.** The inner CG matvec in
`SecantCGSolver` is the *secant* operator `K_sec(u, d) = ∑_e g(d_e) ∫
B^T C_sec(eps_e) B`, with `C_sec` evaluated from the *frozen*
eigenvalue signs and projectors. The full Jacobian of `f_int` w.r.t.
`u` includes a term

    ∂P_i / ∂eps : ∂eps / ∂u                                      (2)

— the eigenvector-rotation contribution — that the secant drops. PR
#170 added this term back in `QuasiStaticSolver` via autograd-JVP, but
**only inside the inner CG matvec, not in any preconditioner / MG
operator / MPC reduction**. None of `SecantCGSolver`, `DirectSolver`,
or the multigrid path uses it. The full coupled `(u, d)` Jacobian
required for a *monolithic* Newton step on (1a)+(1b) is also not
assembled anywhere — the staggered split sidesteps that block.

---

## §2 What COMSOL and Abaqus do

### COMSOL — damped Newton (stationary nonlinear solver)

The COMSOL Multiphysics blog *"Solving Nonlinear Static Finite Element
Problems"* documents the algorithm: damped Newton–Raphson with the
update `u_{i+1} = u_i − α [f'(u_i)]⁻¹ f(u_i)`, where `0 < α ≤ 1` is
the damping factor. When the residual fails to decrease the solver
*backtracks* — α is reduced until `‖f(u_damped)‖` is smaller than the
previous estimate. The Jacobian `f'` is the **consistent tangent**;
COMSOL re-evaluates it every iteration by default ("constant Newton"
freezes it across iterations and is reserved for nearly-linear
problems). Termination is on a **scaled relative tolerance** (default
`1e-3`) measured on the *scaled* solution vector (not absolute), or on
the user iteration cap. The Damage interface (COMSOL 6.2 docs,
`sme_ug_theory.06.035.html`, Eq. 3-92) wraps the variational fracture
energy and is delivered to the same damped Newton driver — there is no
phase-field-specific solver, only the standard nonlinear stationary
study with optional load ramping. On divergence, recovery is via
parametric continuation (load ramping) or via tightening the damping
preset (`Highly nonlinear` increases the minimum damping factor).

### Abaqus/Standard — full Newton with cutbacks (no line search by default)

The Abaqus Analysis User's Guide §"Convergence criteria for nonlinear
problems" (mirror at
`abaqus-docs.mit.edu/2017/.../simaanl-c-convergecontrol.htm`)
specifies:

* Update strategy: **full Newton**, consistent tangent re-assembled
  every iteration. Modified Newton is reserved for the *quasi-Newton*
  step type.
* Convergence criteria (defaults):
  * Force residual ratio `Rnα = 5 × 10⁻³` — largest residual / time-
    averaged force norm `q̃α`.
  * Solution correction ratio `Cnα = 10⁻²` — largest correction /
    largest incremental solution value.
* Iteration thresholds: `I₀ = 4` (start checking residual non-monotone
  growth), `I_R = 8` (start checking quadratic convergence rate).
* Line search: `Nls > 0` activates it but **only during quasi-Newton
  steps**; full Newton steps run without it.
* Restart: automatic time-step **cutback** if the iteration count
  exceeds a limit or the residual grows monotonically — Abaqus halves
  Δt and restarts the increment from the converged state at `t_n`.

The Abaqus Theory Guide §2.2.1 (1992 reference manual) gives the same
update with the consistent material Jacobian as the standard FE
linearisation and notes the radius-of-convergence argument for cutback
control.

### Implication for `phast`

Both vendors converge on the same recipe: **full Newton + consistent
tangent + cutback / damping**. Abaqus uses a relative-residual check
normalised by a *time-averaged* force; COMSOL uses a scaled relative
tolerance. Our current implementation uses an *absolute* residual
norm in `QuasiStaticSolver.solve()` (line 811) and a mixed
abs/rel-with-stall-guard in `DirectSolver.solve()` (lines 1414–1424).
Aligning on a consistent-tangent Newton + relative criterion is the
minimum-surface change to look professional next to either reference.

---

## §3 Implementation plan

### Sub-task A — Full-Newton consistent tangent for the inner solve

*Estimate: 150–300 LOC. Builds on PR #170.*

1. Promote the autograd-JVP matvec from `QuasiStaticSolver._cg_solve`
   into a free-function `tangent_matvec(fem, u, d, p)` in
   `fem_operators.py` so `SecantCGSolver`, `DirectSolver`, and the MPC
   path can all use it.
2. Add a `tangent='secant'|'consistent'` knob to `SolverConfig`. When
   `'consistent'`, the inner CG matvec calls `tangent_matvec`. The
   outer Newton loop is unchanged — re-linearisation still happens
   once per Newton iterate.
3. Add a relative-residual termination `‖R‖ < rtol · ‖R_0‖ ∨ atol` to
   match Abaqus / COMSOL conventions (default `rtol = 1e-4`, `atol =
   1e-10`).
4. Implement Armijo backtracking on the Newton step length: try `α =
   1, 1/2, 1/4, …` until `‖R(u + α du)‖ < (1 − c α) ‖R(u)‖` with
   `c = 1e-4`. This is COMSOL's damped-Newton recipe.

The composition with `_AdjointDamageSolveScalar/Field` is clean
because the consistent-tangent matvec is a function only of `(u, d)`
which are already detached at the Newton entry — autograd through the
sparse-solve still factorises into the same VJP as the secant path.

### Sub-task B — Matrix-free GMG preconditioner

*Estimate: 500–800 LOC.*

1. Build a hierarchy of T3 unstructured meshes by edge-collapse
   coarsening (Borden 2014, §3.2; Gravouil & Combescure 2001 for the
   transfer operators). Cache the prolongation `P` as a sparse map
   from coarse nodes to fine nodes via piecewise-linear interpolation
   along the edge.
2. Smoother: 3-step **damped Jacobi** (the simplest matrix-free
   smoother that vectorises on GPU) on the *consistent-tangent* matvec
   from Sub-task A. Symmetric Gauss-Seidel needs colored element
   patches; defer until coloring is in place.
3. V-cycle: pre-smooth → restrict residual via `Pᵀ` → coarse solve
   (recurse, or Sub-task A's MUMPS at the coarsest level) → prolong
   correction → post-smooth.
4. Plug into `SecantCGSolver._multigrid` slot — the existing class
   already takes a `secant_state`; extend the contract to accept a
   `tangent_matvec` callable.
5. Patch-coloring (#116) and p-multigrid (#117) are deferred — coloring
   is needed for symmetric GS smoothers, p-multigrid waits on P2.

### Sub-task C — Implicit-dynamic via Hulbert–Chung generalised-α

*Estimate: 300–500 LOC. Issue #102.*

1. Implement the gen-α update formulae (Chung & Hulbert 1993, Eqs
   16–17) parametrised by `ρ_∞ ∈ [0, 1]` (spectral radius at infinity
   — the documented numerical-damping knob).
2. Effective tangent at iteration `k` is `K_eff = (1 − α_f) K_t + (1 −
   α_m) M / (β Δt²) + (1 − α_f) γ C / (β Δt)`. With
   `tangent_matvec`, this is a sum of three matvecs and reuses
   Sub-task A.
3. Monolithic-coupling option in `staggered_solver.py::step_full`'s
   `monolithic` branch (line 599) so the gen-α driver can replace the
   staggered iterate when the user asks for tight `(u, d)` coupling.
4. Adaptive Δt control via the Hulbert–Hughes 1987 local-truncation
   estimator.

---

## §4 Acceptance gates

* **A.** B5 PMMA quasi-static benchmark (`benchmarks/B5_pmma`):
  Newton iteration count must be ≤ 1.5× SecantCG at the same outer
  tolerance. h-convergence study at `h ∈ {h₀, h₀/2, h₀/4}` must show
  residual reduction at quadratic-or-better rate (≥ 1.8 in the log-log
  fit). Wall-clock target: not slower than current secant-CG by more
  than 20 % at `h = h₀/4`.
* **B.** Synthetic 1 M-DOF Poisson-like elasticity test problem (B1
  Borden mesh, refined ×4): GMG-V(2,2) preconditioned CG should reach
  `‖r‖/‖r₀‖ < 1e-8` in ≤ 30 iterations versus ≥ 300 for Jacobi-CG;
  Bourdin (2014) reports 10× reduction at fine meshes.
* **C.** Borden et al. 2012 §5 cyclic-load benchmark (CMAME 217–220):
  with `ρ_∞ = 0.5`, the high-frequency content should be damped with
  the Chung–Hulbert documented decay; energy drift over 1000 cycles
  must be ≤ 1 %. Spectral-radius verification at the linearised level
  (eigenvalues of the amplification matrix) on a 2-DOF spring-mass
  test.

---

## §5 Risks and gotchas

* **Newton basin in damaged regions.** Where `g(d) ≈ 0` the consistent
  tangent has rank deficiency and the JVP returns near-zero
  components. Practical fix: an ϵ-floor on `g(d)` (already
  `eta = 1e-6` in the codebase) is sufficient for the matvec but the
  Newton update will still take giant steps in those nodes. Mitigation:
  **L-BFGS rescue** — if Armijo backtracking fails three times,
  switch to a 5-vector L-BFGS step on `‖R‖²` for that iterate, then
  resume Newton. (Not full L-BFGS-Newton hybrid; just a rescue.)
* **Memory at 238 k nodes.** `_assemble_K_isotropic` (line 662) builds
  the COO triplet array `(rows, cols, vals)` of length `36 · n_elem`
  ≈ 17 M entries → 400 MB float64. Conversion to CSR doubles this
  briefly. Matrix-free is the only path at the next mesh refinement
  (≥ 1 M nodes ⇒ > 4 GB triplets). The MUMPS symbolic-stage cache
  (#241) helps only for repeated solves with the same sparsity, not
  for assembly.
* **Autograd-JVP × adjoint-implicit-diff.** `_AdjointDamageSolveScalar`
  / `_AdjointDamageSolveField` differentiate through the staggered
  loop's damage solve. If Sub-task A's consistent-tangent matvec is
  used, the outer adjoint reuses the *same* tangent — that's the
  forward Jacobian, and the adjoint VJP is `Jᵀ v` not `J v`. We need
  to verify `tangent_matvec` is symmetric (it is, when assembled from
  `B^T C B`; the spectral-projector eigenvector-rotation term breaks
  symmetry slightly at coincident eigenvalues — this is the same
  asymmetry Miehe et al. 2010 §4.3 flags). Recommend a symmetric
  projection `K_sym = ½(K + Kᵀ)` for the adjoint path until the
  asymmetry is bounded numerically.

---

## §6 References

* Miehe C., Hofacker M., Welschinger F. (2010). *A phase field model
  for rate-independent crack propagation.* CMAME **199**, 2765–2778.
  DOI: 10.1016/j.cma.2010.04.011.
* Borden M.J., Hughes T.J.R., Landis C.M., Verhoosel C.V. (2014). *A
  higher-order phase-field model for brittle fracture.* CMAME **273**,
  100–118. DOI: 10.1016/j.cma.2014.01.016.
* Borden M.J., Verhoosel C.V., Scott M.A., Hughes T.J.R., Landis C.M.
  (2012). *A phase-field description of dynamic brittle fracture.*
  CMAME **217–220**, 77–95. DOI: 10.1016/j.cma.2012.01.008.
* Chung J., Hulbert G.M. (1993). *A time integration algorithm for
  structural dynamics with improved numerical dissipation: the
  generalized-α method.* J. Appl. Mech. **60**, 371–375.
  DOI: 10.1115/1.2900803.
* Simo J.C., Hughes T.J.R. (1992/1998). *Computational Inelasticity.*
  Springer. ISBN 978-0-387-22763-4.
* Bourdin B. (2014). *Numerical implementation of the variational
  formulation for quasi-static brittle fracture.* Interfaces & Free
  Boundaries **9**, 411–430. DOI: 10.4171/IFB/171.
* Hulbert G.M., Hughes T.J.R. (1987). *Space-time finite element
  methods for second-order hyperbolic equations.* CMAME **84**,
  327–348.
* Abaqus 2017 Analysis User's Guide. *Convergence criteria for
  nonlinear problems.* `abaqus-docs.mit.edu/2017/English/
  SIMACAEANLRefMap/simaanl-c-convergecontrol.htm` (accessed
  2026-05-06).
* COMSOL Multiphysics 6.2 Reference. *Damage theory.* COMSOL doc
  `sme_ug_theory.06.035.html` (accessed 2026-05-06).
* COMSOL Multiphysics blog. *Solving Nonlinear Static Finite Element
  Problems.* `comsol.com/blogs/solving-nonlinear-static-finite-
  element-problems` (accessed 2026-05-06).
