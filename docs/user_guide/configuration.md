# Configuration guide

Preconditioner auto-selection, Anderson acceleration, energy-split decision
tree, convergence criteria, recommended per-benchmark settings, and the YAML
configuration system.

```python
mat = create_material('glass_borden', eta_residual=1e-6)  # old value
```

---

## Configuration Guide

### Preconditioner Selection Logic

For explicit dynamics, `preconditioner=None` with `use_multigrid=True` resolves
through the auto-selection chain below. For `static`, `quasi_static`,
`quasi_static_legacy`, `lbfgs`, and `monolithic`, unresolved preconditioners
default to `jacobi` to avoid fragile AMG/GMG behavior in reaction-dominated
implicit endgame states. Use `preconditioner='auto'` or `'amg'` only for
explicit validation of that path.

```
CUDA HPC (A100/H100/V100):
  pyamgx installed? → amgx (fastest, GPU-native AMG)
  pyamg installed?  → amg (CPU setup, GPU solve)
  else              → gmg (2-level geometric, always available)

CUDA Workstation (RTX 3090/4090/A6000):
  pyamg installed?  → amg
  else              → gmg

CUDA Consumer (RTX 2080/3060):
  → gmg (low VRAM, AMG setup overhead not worth it)

Apple MPS:
  → gmg (no CUDA, no AMG; float32 with CPU float64 fallback for CG)

CPU:
  pyamg installed?  → amg (fastest CPU option)
  else              → gmg
```

### Anderson Acceleration Guide

Anderson acceleration (Type II, Walker & Ni 2011) reduces stagger iterations by 30-50%.

| Depth | Memory | Convergence | Recommended For |
|-------|--------|-------------|-----------------|
| 0 | None | Baseline | Debugging, simple problems |
| 3 | 3 vectors | Good | General use, default recommendation |
| 5 | 5 vectors | Best | Large meshes, tight tolerances |
| 7+ | 7+ vectors | Diminishing returns | Rarely needed |

Usage: `SolverConfig(anderson_depth=3)` or `--anderson_depth 3`

### Energy Split Decision Tree

```
Is the problem pure Mode I tension?
  YES → isotropic (fastest, simplest)
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

| Criterion | Formula | Speed | Strictness | Best For |
|-----------|---------|-------|------------|----------|
| `relative` | `\|\|Δd\|\|/\|\|d\|\| < tol` | Fast | Medium | **Default**, general use |
| `absolute` | `\|\|Δd\|\| < tol` | Fast | Varies | Simple problems, debugging |
| `linf` | `max\|Δd\| < tol` | Fast | **Strictest** | Publication-quality results |
| `residual` | `\|\|R_u\|\| + \|\|R_d\|\| < tol` | Slow | Very strict | Validation, reference solutions |
| `am_energy` | `\|ΔE\|/\|E\| < tol` | Medium | Energy-based | Energy-sensitive studies |

### Solver Type Selection

| Problem | Solver | Why |
|---------|--------|-----|
| Dynamic fracture | `explicit` (ExplicitDynamics) | O(N) per step, CFL-limited |
| Quasi-static SENT/SENS/TPB | `quasi_static` (QuasiStaticSolver) | Standard staggered scheme; spectral tangent via autograd-JVP |
| Benchmark comparison | `quasi_static` + `linf` criterion | Strictest, matches PhaseFieldX |
| Frozen-secant CG fallback | `quasi_static_legacy` (SecantCGSolver) | Required for iterative-CG `rigid_connector` MPC |
| Energy minimisation | `monolithic` (MonolithicSolver) | Joint (u,d), no stagger |
| Linear elastic equilibrium | `static` (StaticSolver) | Pre-strain, simple problems |

### Recommended Settings per Benchmark

| Benchmark | Energy Split | Preconditioner | Anderson | Steps | Stagger Tol | damage_every |
|-----------|-------------|----------------|----------|-------|-------------|-------------|
| SENT (QS) | isotropic | auto | 3 | 150 | 1e-6 | N/A |
| SENS (QS) | spectral | auto | 5 | 200 | 1e-6 | N/A |
| TPB (QS) | spectral | auto | 3 | 200 | 1e-6 | N/A |
| L-panel (QS) | spectral | auto | 5 | 1400 | 1e-6 | N/A |
| B1 Branching | spectral | auto | N/A | ~1500 | N/A | 1 |
| B2 Kalthoff | spectral | auto | N/A | ~2000 | N/A | 3 |
| B3 SENT (dyn) | spectral | auto | N/A | ~1500 | N/A | 3 |
| B5 PMMA | amor (PS) | jacobi (AT1) | N/A | ~19000 | N/A | 3 |

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

`python -m phast explain-config <config.yaml>` prints this block,
and YAML runs preserve it in the resolved config and run lockfile. Keep the
status below `validated` until the reference extraction and output artifacts
are stored in the repo or linked from the issue.

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
  since PR #170) and `SecantCGSolver` (`quasi_static_legacy`) are supported;
  fall back to `quasi_static_legacy` if the new spectral-split tangent
  exhibits stalls on a particular configuration
- **v0.11.2 fix:** If CG reports 5000 iterations every step, update to v0.11.2.
  Three bugs caused the damage CG to spin at max_iter: (1) absolute tolerance
  unreachable for small RHS at low damage, (2) AMG preconditioner returning
  non-descent directions for reaction-dominated systems, (3) AMG hierarchy
  rebuilt every step (100ms waste). Fix: relative tolerance, AMG quality
  detection with Jacobi fallback, cached AMG hierarchy. Result: CG=5-18
  iterations instead of 5000, a 2800× per-step speedup on CUDA
  (before vs. after v0.11.2 on the same hardware — internal bugfix
  gain, not a cross-code speedup claim; for the latter see the CMAME
  paper's FEniCS/Akantu comparison).

**`[AMG] pyAMG failed (array must not contain infs or NaNs), skipping
hierarchy rebuild` warnings during crack transition:**

*Benign — simulation results are unaffected.* These warnings fire during the
narrow window where damage jumps from `max(d) ≈ 0.6` to `max(d) = 1.0` at
crack nucleation. The reaction coefficient `reaction_coeff = (2H + Gc/l0)
· area/12` passed into `_assemble_sparse_cpu` is itself finite (guarded at
`multigrid.py:875` and would emit a separate `[AMG] WARNING: N non-finite
reaction_coeff entries, clamping` message if it weren't). The inf/NaN
appears *inside* `pyamg.smoothed_aggregation_solver(A_csr)` setup, where
operations like Jacobi relaxation-weight estimation (`1/diag`), Lanczos

