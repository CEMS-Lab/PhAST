# Configuration guide

This guide summarizes the configuration choices that most strongly affect
stability, convergence, and reproducibility in PhAST.

```python
mat = create_material('glass_borden', eta_residual=1e-6)
```

---

## Configuration Guide

### Preconditioner Selection Logic

For implicit solver types, an unspecified preconditioner defaults to `jacobi`.
Other preconditioners are optional paths whose availability and suitability
depend on the selected solver, device, mesh, and installed dependencies. Check
the capability matrix before treating one as part of a reproducible workflow.

```
The available vocabulary includes `jacobi`, `gmg`, and optional `amg`/`amgx`
backends. Automatic selection is environment-dependent; no device-specific
performance ranking is implied here.
```

### Anderson Acceleration Guide

`anderson_depth` enables an optional Type-II fixed-point acceleration for the
staggered damage update. Its effect is problem-dependent; no iteration-saving
percentage or generally preferred depth is asserted here. Leave it at its
default unless the selected case has been evaluated with that option.

### Energy Split Decision Tree

```
Is the problem pure Mode I tension?
  YES → isotropic (lowest split complexity)
  NO  →
    Is crack path expected to curve?
      YES → spectral (Miehe 2010, eigenvalue decomposition)
      NO  →
        Is compression significant?
          YES → amor (volumetric-deviatoric, robust default)
          NO  → isotropic

    Studying crack nucleation without pre-crack?
      YES → star_convex (Kumar & Lopez-Pamies 2020)
```

### Convergence Criterion Comparison

| Criterion | Formula | Relative cost | Interpretation | Typical use |
|-----------|---------|-------|------------|----------|
| `relative` | `\|\|Δd\|\|/\|\|d\|\| < tol` | Fast | Medium | **Default**, general use |
| `absolute` | `\|\|Δd\|\| < tol` | Fast | Varies | Simple problems, debugging |
| `linf` | `max\|Δd\| < tol` | Fast | **Strictest** | Publication-quality results |
| `residual` | `\|\|R_u\|\| + \|\|R_d\|\| < tol` | Slow | Very strict | Validation, reference solutions |
| `am_energy` | `\|ΔE\|/\|E\| < tol` | Medium | Energy-based | Energy-sensitive studies |

### Solver Type Selection

| Problem class | Solver | `solver_type` | Status | Use case |
|---|---|---|---|---|
| Nonlinear quasi-static | `QuasiStaticSolver` | `quasi_static` | Primary path for new quasi-static fracture runs | Newton-Raphson with sparse-direct or matrix-free mechanics; the matrix-free mechanics action uses an automatic-differentiation JVP. |
| Nonlinear quasi-static | `SecantCGSolver` | `quasi_static_legacy` | Compatibility path for older validated runs | Frozen-secant CG for older accepted runs and selected iterative-CG connector support. |
| Explicit dynamics | `ExplicitDynamics` | `explicit` | Active dynamic-fracture path | Impact, wave-driven fracture, branching, and rapid trajectory generation. |
| Linear static equilibrium | `StaticSolver` | `static` | Supporting path | Single load step with `d=0`, used by selected mechanics setup paths. |
| Nonlinear minimisation | `LBFGSSolver` | `lbfgs` | Available for specialized studies | Gradient-only minimisation when tangent matvecs are unavailable or expensive. |

| Problem | Solver | Why |
|---------|--------|-----|
| Dynamic fracture | `explicit` (ExplicitDynamics) | CFL-limited explicit time integration |
| Quasi-static SENT/SENS/TPB | `quasi_static` (QuasiStaticSolver) | Standard staggered scheme; mechanics action via automatic-differentiation JVP |
| Benchmark comparison | `quasi_static` + `linf` criterion | Strictest, matches PhaseFieldX |
| Frozen-secant CG fallback | `quasi_static_legacy` (SecantCGSolver) | Required for iterative-CG `rigid_connector` MPC |
| Energy minimisation | `monolithic` (MonolithicSolver) | Joint (u,d), no stagger |
| Linear elastic equilibrium | `static` (StaticSolver) | Pre-strain, simple problems |

Use `explicit` only when inertia and stress waves are part of the physics
question, for example Kalthoff-Winkler impact or dynamic branching. Explicit
dynamics is CFL-limited and is not a shortcut for quasi-static loading.

`quasi_static` is the default for displacement-controlled SENT, SENS, TPB,
L-shaped panel, and comparable slow-loading fracture problems. Keep
`quasi_static_legacy` only when reproducing an older accepted run that depends
on the frozen-secant CG path.

### Explicit-dynamics performance knobs

For explicit dynamics, the damage equation can be solved every N-th time step
instead of every step. Since damage propagates more slowly than the elastic
wave speed used by the CFL condition, `damage_every: 2` or `3` can be useful
after a sensitivity check.

```yaml
solver:
  damage_every: 1   # reference validation: solve damage every explicit step
  # damage_every: 2 # throughput sensitivity run
  # damage_every: 3 # aggressive throughput run; document as non-reference
```

The first explicit steps still solve damage every step to capture crack
nucleation. Treat subcycling as a performance setting, not as a change to the
benchmark definition.

Smooth load ramps reduce high-frequency oscillations from instantaneous
tractions. In YAML, prefer `ramp_type: smooth_step` with an explicit `t_ramp`
for dynamic traction loads that should approximate a finite rise time.

Residual stiffness `eta_residual` is the numerical floor in the degradation
function. It prevents fully damaged regions from producing zero-stiffness rows;
use the benchmark value unless you are deliberately studying sensitivity.

### Recommended Settings per Benchmark

| Benchmark | Energy Split | Preconditioner | Steps | Stagger Tol | damage_every |
|-----------|-------------|----------------|-------|-------------|-------------|
| Representative cases | case-dependent | case-dependent | case-dependent | case-dependent | case-dependent |

### Benchmark Acceptance Metadata

primary public and paper-facing configs should include a top-level
`acceptance:` block. It is structured but extensible metadata for reviewers and
post-processing scripts; it does not change the solve. PhAST validates the
standard fields while preserving custom benchmark-specific keys. Use it to
record the reference figure/table, required artifacts, metric targets,
tolerances, units, and caveats:

```yaml
acceptance:
  status: beta                  # scaffold | beta | production | validated | diagnostic | experimental
  reference_result: "Ambati et al. (2015), L-panel peak load"
  required_outputs: [run_lockfile.json, config.yaml, load_displacement.png]
  metrics:
    peak_reaction_kN:
      target: 16.0
      tolerance: 0.15
      units: kN
    crack_path:
      target: "corner-to-top-edge arc"
      tolerance: visual
  notes: "Reaction convention and extraction script must be cited here."
```

`python -m phast explain-config <config.yaml>` prints this block, and YAML runs
preserve it in the resolved config and run lockfile. Keep the status below
`validated` until the reference extraction and output artifacts are stored in
the repository or linked from the public record.

### Material Presets Quick Reference

| Preset | E (MPa) | nu | Gc (N/mm) | l0 (mm) | Split | eta | Use |
|--------|---------|-----|-----------|---------|-------|-----|-----|
| `miehe_tension` | 210000 | 0.3 | 2.7 | 0.015 | isotropic | 1e-7 | QS SENT |
| `miehe_shear` | 210000 | 0.3 | 2.7 | 0.06 | spectral | 1e-7 | QS SENS |
| `glass_borden` | 32000 | 0.2 | 3e-3 | 0.25 | spectral | 1e-7 | B1, B3, B4 |
| `maraging_steel_kw` | 190000 | 0.3 | 22.13 | 0.195 | spectral | 1e-7 | B2 Kalthoff |
| `pmma_bleyer` | 3090 | 0.35 | 0.3 | 0.1 | amor | 1e-7 | B5, B6 (plane stress) |
| `l_shaped_concrete` | 25850 | 0.18 | 0.089 | 1.1875 | spectral | 1e-7 | L-panel (Ambati) |
| `l_shaped_glass` | 70000 | 0.23 | 0.008 | 0.4 | spectral | 1e-7 | L-panel glass |
| `alumina_kumar` | 335000 | 0.25 | 0.0268 | 0.04 | star_convex | 1e-7 | Nucleation studies |

### Troubleshooting

**CG diverges or very slow:**
- Check `pf_model`: AT1 has a nucleation threshold below which CG converges to d=0 trivially
- Keep quasi-static/static jobs on `--preconditioner jacobi`; use `gmg`/`amg`
  only when validating that preconditioner path explicitly
- Increase `--damage_cg_tol` from 1e-5 to 1e-4 if convergence is too slow
- For spectral/amor splits, both `QuasiStaticSolver` (`quasi_static`, default
  ) and `SecantCGSolver` (`quasi_static_legacy`) are supported;
  fall back to `quasi_static_legacy` if the new spectral-split tangent
  exhibits stalls on a particular configuration
- Optional AMG backends can be unavailable or can reject a problem during
  setup. Treat such messages as backend-specific diagnostics and compare the
  resulting route and outputs explicitly; they do not by themselves establish
  that a simulation result is unaffected.
