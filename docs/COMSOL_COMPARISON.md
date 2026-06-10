# COMSOL vs phast — Detailed Comparison

Based on COMSOL 6.4 documentation, Application Library examples, and release notes.

Status note, 2026-06-10: this comparison is a capability-boundary document,
not a claim of COMSOL parity. `phast` now has beta validation slices
for sparse J2 plasticity, cohesive elements, PF-CZM uniaxial strength smoke,
and coupled brittle PF+cohesive examples. It does not yet provide a mature
commercial-style coupled plasticity + PF-CZM + cohesive-interface production
workflow.

## Architecture

| Feature | COMSOL 6.4 | phast |
|---------|-----------|-----------------|
| **Assembly** | Sparse matrix (CSR/CSC) | **Matrix-free** (scatter_add_) |
| **GPU support** | cuDSS sparse direct solver (6.4), explicit acoustics multi-GPU | **Native CUDA** via PyTorch tensors |
| **GPU for phase-field** | Not mentioned in GPU docs | ✓ Full GPU support (all operations) |
| **Differentiability** | None | ✓ Native PyTorch autograd |
| **Language** | Proprietary (Java/C++) | Python/PyTorch (open source) |
| **License** | Commercial ($$$) | Open source |
| **3D support** | ✓ Full 3D | 2D only (current) |
| **Parallel** | MPI + GPU (cuDSS) | GPU batching |

**Key finding: COMSOL does NOT do matrix-free GPU.** COMSOL 6.4 uses NVIDIA cuDSS — a sparse **direct solver** on GPU. This is fundamentally different from our matrix-free approach. COMSOL still assembles sparse matrices, then factorizes them on GPU. We never assemble a matrix at all.

## Phase-Field Models

| Feature | COMSOL 6.4 | phast |
|---------|-----------|-----------------|
| **AT1** | ✓ (w(φ)=φ, c_w=8/3) | ✓ (same formulation) |
| **AT2** | ✓ (w(φ)=φ², c_w=2) | ✓ (same formulation) |
| **PF-CZM** | ✓ (cohesive zone model) | Beta smoke: Wu PF-CZM forward damage solves with uniaxial strength/`l0` validation; structural crack growth and mixed-mode benchmarks remain gated |
| **Energy splits** | Volumetric, Spectral (stress), Spectral (strain), None | Isotropic, Amor (vol-dev), Spectral (strain), Star-convex |
| **Degradation** | Power law (m=2), Cubic (Borden), User-defined | Standard (1-d)², Cubic, Rational |
| **Crack driving force** | Strain energy density or Principal stress criterion | Strain energy density (ψ⁺); principal-stress criterion scaffold exists for selected validation cases |
| **Irreversibility** | History field H = max(ψ⁺, H_prev) | Same + projected CG for AT1 |
| **AT1 threshold** | 3Gc/(8l₀) = 2W_c0 = Eε_c² | Same (auto-computed) |
| **η (residual stiffness)** | 1e-7 in the dynamic branching application | Configurable; COMSOL-parity configs use 1e-7 |

### Still not COMSOL-equivalent in phast:
- **PF-CZM production parity** — current support is a forward Wu PF-CZM
  strength-calibration smoke, not a full structural/mixed-mode PF-CZM product.
- **Principal stress criterion production path** — scaffold exists, but
  strain-energy driving force remains the validated default for most brittle
  fracture workflows.
- **Fully coupled plasticity + PF-CZM + cohesive interfaces** — unsupported as
  a single calibrated customer workflow.

## Solver / Time Integration

| Feature | COMSOL 6.4 | phast |
|---------|-----------|-----------------|
| **Explicit dynamics** | ✓ Velocity-Verlet with mass lumping | ✓ Same |
| **Implicit dynamics** | ✓ Generalized-α, Newmark, BDF | Forward-only generalized-alpha validation path; differentiable adjoint and production hardening still open |
| **Quasi-static** | ✓ Newton-Raphson | ✓ `QuasiStaticSolver` Newton path (`quasi_static`); legacy SecantCG kept as `quasi_static_legacy` |
| **Stagger scheme** | Segregated, fixed # iterations (e.g., 3) | Segregated, convergence-based + Anderson AA |
| **CFL time step** | Auto (based on c_p) | Auto (based on c_p, dt_safety factor) |
| **Phase field subcycling** | ✓ Solves damage every 2nd step | ✓ `damage_every` for explicit dynamics |
| **Mass lumping** | ✓ Row-sum lumping | ✓ Same |
| **Linear solver backend** | Direct sparse solvers including MUMPS, PARDISO, SPOOLES, and cuDSS where supported | Matrix-free CG, SciPy SuperLU sparse direct, optional PETSc/MUMPS sparse direct |
| **Preconditioner** | Direct factorisation or iterative preconditioners, selected through the COMSOL solver tree | Jacobi by default for QS damage; GMG/AMG/AmgX are experimental for QS fracture |
| **Adaptive time stepping** | ✓ Built-in | partial explicit-dynamics support |
| **Smooth step loading** | ✓ Built-in function | ✓ YAML smooth-step ramp |
| **Symmetry exploitation** | ✓ Half-model | ✓ Node-set symmetry/fix BCs; B7 half-plate configs exist |

### Key COMSOL advantages we should adopt:
1. **Implicit dynamics hardening** — production generalized-alpha/Newmark with adjoints and robust preconditioning
2. **Adaptive time stepping** — broaden cutback/growth controls and validation
3. **Higher-order / quad elements** — improve accuracy per DOF for customer meshes
4. **Integrated commercial-style postprocessing** — richer derived fields and report templates

### Direct solver terminology

COMSOL exposes several sparse direct solvers. MUMPS is a multifrontal sparse
direct solver that also supports distributed-memory MPI builds. PARDISO is a
high-performance sparse direct solver from the Intel MKL/PARDISO family.
SPOOLES is an older sparse direct solver based on multifrontal sparse
factorisation. These are not preconditioners in the usual iterative-solver
sense; they factorise the assembled sparse system directly.

`phast` currently calls two CPU sparse-direct backends:

- `mumps`: PETSc/MUMPS through `petsc4py`, selected by `backend='auto'` when
  the runtime smoke test passes.
- `scipy`: SciPy SuperLU through `scipy.sparse.linalg`, used as the portable
  CPU fallback.

On environments where the PETSc/MUMPS smoke test passes, SENS/TPB jobs
launched with `backend='auto'` use MUMPS for the mechanics linear solve.
PARDISO and SPOOLES are not wired into this repository.

## Mesh and Elements

| Feature | COMSOL 6.4 | phast |
|---------|-----------|-----------------|
| **Element types** | Tri, Quad, Tet, Hex, Serendipity | Production Tri3; native Q4 isotropic mechanics + AT2 damage is beta; higher-order primitives are scaffold |
| **Element order** | Linear, Quadratic, Cubic | Linear production; higher-order shape functions are not globally dispatched |
| **Adaptive mesh refinement** | ✓ Built-in (API-based control) | ✓ NVB (newest-vertex bisection) |
| **h_e guideline** | h_e ≤ l_int (h_e = l_int/4 in examples) | Same (h_crack = l0/4 typical) |
| **Mesh generators** | Built-in parametric | Gmsh (external) |

### Still gated in phast:
- **Quadratic production elements** — higher-order primitives exist, but global
  solver dispatch and benchmark coverage remain gated.
- **Broad quad production support** — Q4 isotropic mechanics + AT2 damage has a
  beta smoke path; Q4 PF-CZM, plasticity, cohesive-coupled damage, and
  differentiable damage adjoints remain gated.

## Postprocessing

| Feature | COMSOL 6.4 | phast |
|---------|-----------|-----------------|
| **Crack front tracking** | φ·∂φ/∂t indicator (heuristic) | d > 0.5 threshold scan |
| **Damaged stress** | von Mises × g(d), auto-computed | Computed in postprocessing |
| **Crack removal in plots** | d(φ) = 1 - g(φ) > 0.95 removed | Not implemented (show full field) |
| **Energy balance** | Built-in integration | Computed from H5 |
| **Crack velocity** | Model method (Windows only!) | From crack_tip.csv or H5 |

### Better in phast:
- **Automated postprocessing** — single command generates all plots
- **H5 training data** — ready for ML pipelines

## Material Models

| Feature | COMSOL 6.4 | phast |
|---------|-----------|-----------------|
| **Linear elastic** | ✓ | ✓ |
| **Hyperelastic** | ✓ (Neo-Hookean, Mooney-Rivlin, etc.) | ✗ |
| **Plasticity** | ✓ (J2, Drucker-Prager, etc.) | Beta J2: material-point and sparse quasi-static validation slices; Drucker-Prager/Mohr-Coulomb and benchmark-matched ductile fracture remain gated |
| **Viscoplastic** | ✓ | ✗ |
| **Creep** | ✓ | ✗ |
| **Plane stress** | ✓ | ✓ (v0.12.0) |
| **Plane strain** | ✓ | ✓ |
| **Thermoelastic coupling** | ✓ | ✗ |

## Benchmark Comparison

### Dynamic Crack Branching (same problem: Borden 2011)

| Parameter | COMSOL | phast |
|-----------|--------|-----------------|
| Model | AT1 | AT2 (B1) and AT1 (B5) |
| E | 32 GPa | 32 GPa ✓ |
| ν | 0.2 | 0.2 ✓ |
| Gc | 3 J/m² | 3 J/m² ✓ |
| l_int | 0.5 mm | 0.5 mm ✓ |
| c_R | 2125 m/s | 2125 m/s ✓ |
| η | 1e-7 | 1e-7 in COMSOL-parity configs |
| h_e | l_int/4 = 0.125 mm | l_int/4 in current B7 COMSOL-parity configs |
| Symmetry | 100 x 20 mm half model (`height/2`) | Half-model and full-plate-equivalent configs |
| Subcycling | Every 2nd step | Configurable via `damage_every` |
| Branching | ~33 µs in the COMSOL application; Ren/Borden-style references differ | Under active B7 parity rerun; see #570/#576 |
| Max velocity | < 0.6 c_R | < 0.6 c_R ✓ |

### Holed Plate (Ambati 2015 reference)

| Parameter | COMSOL | phast |
|-----------|--------|-----------------|
| Material | Cement mortar | N/A (different benchmark) |
| E | 6 GPa | — |
| Gc | 2280 J/m² | — |
| l_int | 0.25 mm | — |
| Solver | Quasi-static, segregated, 3 iterations | — |
| Plane stress | ✓ | — |

## Feature Roadmap vs COMSOL

*Last verified against v0.15.2 (2026-04-16). See CHANGELOG.md for version history.*

### Implemented (v0.13+)
- [x] **Phase field subcycling** — `SolverConfig(damage_every=3)` (v0.13.0, ~45% speedup)
- [x] **Smooth step loading** — Hermite ramp via `smooth_step()` (v0.13.0)
- [x] **Lower default η** — `eta_residual=1e-6` (configurable per-material)
- [x] **AT1 model with projected CG** — bound-constrained solve (v0.12.0)
- [x] **Plane stress** — full support with Amor split (v0.12.0)

### Implemented as beta / smoke
- [x] **Sparse J2 validation slice** — per-element state, return mapping,
  commit/rollback, internal force, sparse quasi-static dispatch, and
  plastic-work accounting are covered by validation examples.
- [x] **Cohesive element validation slices** — bilinear residual/tangent,
  mixed-mode/contact/delamination/structural smoke examples, and visual
  manifests exist.
- [x] **PF-CZM smoke** — Wu PF-CZM uniaxial strength/length-scale validation
  bundle exists with residual/convergence/visual gates.

### Not yet production-ready
- [ ] **PF-CZM structural/mixed-mode product** — full TSL families,
  He-Hutchinson, Camanho mixed-mode, PPR, and structural crack-growth
  validations remain open.
- [ ] **Principal stress criterion production validation** — alternative driving force
- [x] **Symmetry BCs** — half-model support via node-set fix/symmetry constraints
- [ ] **Adaptive time stepping** — increase dt when Δd < threshold
- [ ] **Implicit dynamics production hardening** — forward-only generalized-alpha exists; adjoint/preconditioner/customer gates remain open
- [x] **Higher-order element primitives** — T6 plus Q4/Q8/Q9 shape functions,
  quadrature, and single-element stiffness are tested; production solver
  dispatch remains a separate hardening step.
- [ ] **Hyperelastic materials** — Neo-Hookean + damage
- [ ] **3D support** — tetrahedral elements
- [ ] **Thermoelastic coupling** — temperature-dependent fracture

For full benchmark parameter tables, see `papers/paper/BENCHMARK_SETTINGS.md`.

## Our Unique Advantages over COMSOL

1. **Matrix-free assembly** — O(N) memory vs O(nnz) sparse
2. **Native autograd** — ∂(any output)/∂(any input) in one backward pass
3. **GPU-native** — all operations on GPU, no CPU↔GPU transfer for assembly
4. **Neural operator coupling** — zero-serialization tensor handoff
5. **Open source** — modifiable, extensible, reproducible
6. **Python ecosystem** — PyTorch, PyG, DeepONet integration
7. **YAML config** — declarative problem definition
8. **Automated postprocessing** — one command for all plots

Sources:
- [COMSOL 6.4 GPU Acceleration](https://www.comsol.com/release/6.4/gpu-acceleration)
- [COMSOL 6.4 Geomechanics Module](https://www.comsol.com/release/6.4/geomechanics-module)
- [COMSOL 6.4 Nonlinear Structural Materials](https://www.comsol.com/release/6.4/nonlinear-structural-materials-module)
- [COMSOL Phase-Field Damage Documentation (5.6)](https://doc.comsol.com/5.6/doc/com.comsol.help.sme/sme_ug_solid.07.029.html)
- [COMSOL Phase Field Theory (6.3)](https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_theory.06.170.html)
- [COMSOL GPU Press Release](https://www.comsol.com/press-release/comsol-speeds-simulation-with-expanded-nvidia-gpu-support-for-comsol-multiphysics-version-64-14612)
