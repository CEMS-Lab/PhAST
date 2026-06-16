# Customer-Readiness Plan

This document is the customer-facing engineering gate for requests such as
"implicit phase-field fracture with plasticity, microstructures, and cohesive
interfaces". The target is not feature naming parity with Abaqus or COMSOL; the
target is that every advertised coupled physics path has a defensible
formulation, regression tests, reference benchmarks, and failure-mode handling.

For the current technical-preview package, see
[`plasticity_cohesive_beta_release.md`](plasticity_cohesive_beta_release.md).

## Current Product Boundary

| Area | Status | Customer-safe claim |
| --- | --- | --- |
| Implicit/quasi-static brittle phase-field fracture | Implemented, validation still expanding | Usable for AT1/AT2 brittle fracture studies with benchmark-specific validation and documented endgame limits. |
| Heterogeneous elastic and fracture fields | Implemented | Per-element `E(x)` and `Gc(x)` fields support inclusions, weak zones, void proxies, and dataset microstructures. |
| Solver-driven diffuse interface fracture | Beta smoke | Weak-interface deflection and strong-interface penetration are available as solved AT2 phase-field damage examples with spatial `E(x)`/`Gc(x)` fields, standard visuals, CSV telemetry, provenance, and manifest checks; this is not a cohesive displacement-jump law or PF-CZM structural calibration. |
| Geometric microstructures | Implemented for data generation | Random microstructures, Voronoi phases, stiff inclusions, degraded void proxies, and gmsh-carved circular voids are available as benchmark/data generators. |
| Dataset / trajectory storage | Implemented, parity still expanding | New neural-operator and large dataset workflows are Zarr-first; H5 is legacy compatibility for existing paper artifacts and post-processing bridges. |
| J2 plasticity | Quasi-static mechanics beta | Material-point return mapping, per-element mesh state, commit/rollback, internal-force assembly, sparse quasi-static Newton dispatch, and plastic-work accounting are implemented for validation examples. |
| Ductile PF-plasticity coupling layer | Beta | Guarded quasi-static T3 `j2_isotropic` + AT2 staggered integration is available with sparse-direct mechanics, bounded damage validation, sensitivity evidence, and an integrated elastic/plastic/fracture energy ledger; benchmark-matched ductile fracture remains pending. |
| Cohesive elements | Beta | Node-doubling, metadata-preserving T3/Q4 cohesive-layer insertion, true bilinear traction-separation state, residual/tangent assembly, scalar dissipated-energy history, optional normal-contact penalty, sparse solver smoke, mode-I displacement-jump, mixed-mode tangent, contact-compression, multi-element delamination patch, and DCB-style structural cohesive benchmarks are available. |
| Coupled brittle PF + cohesive elements | Beta smoke | A notched-matrix AT2 phase-field damage plus zero-thickness cohesive-interface delamination example runs in one staggered loop with cohesive tangent mechanics, PF damage updates, full visuals, CSV telemetry, and manifest checks; calibrated PF-CZM remains pending. |
| PF-CZM cohesive phase-field | Beta smoke | Wu PF-CZM forward damage solves are implemented with tensile-strength-calibrated rational degradation, element-wise gamma-corrected calibration, and a uniaxial strength/length-scale validation bundle with visuals, CSV telemetry, residual/convergence gates, and manifest checks; structural crack-growth and coupled PF-plasticity-cohesive workflows remain pending. |
| Coupled implicit PF + plasticity + cohesive interfaces | Not implemented | Do not sell as available. This is the product-hardening track below. |

`StaggeredSolver` has a guarded ductile path for quasi-static T3
`j2_isotropic` + AT2 with sparse-direct mechanics. Unsupported plasticity
combinations are rejected at construction so plasticity cannot be silently
ignored in a coupled phase-field analysis. `QuasiStaticSolver` also rejects
simultaneous `plasticity_operator` and `cohesive_operator` inputs at
construction, because the current J2 and cohesive paths have separate
state-update and rollback contracts.

## Current Validation Evidence

| Evidence | Scope | Result |
| --- | --- | --- |
| PR #666 / issue #550 | Base plasticity/interface examples | Focused validation suite and visual-manifest evidence retained through issue/PR records. |
| PR #667 / issue #553 | Production hardening: guarded J2+AT2 path, Q4 mechanics first slice, cohesive displacement-jump benchmark | Focused validation suite passed; durable claims are scoped through issue evidence and manifests. |
| PR #668 / issue #553 | Ductile PF-plasticity sensitivity study | Focused validation suite passed; benchmark-matched ductile fracture remains gated. |
| PR #669 / issue #554 | Mixed-mode cohesive tangent benchmark | Focused suite passed with tangent and traction checks. |
| PR #670 / issue #554 | Cohesive contact-compression benchmark | Focused suite passed with contact traction and no-damage-growth checks. |
| PR #672 / issue #554 | True-bilinear cohesive law, multi-element delamination patch, DCB-style structural smoke, and ductile energy ledger | Focused suites passed; public claims remain beta/smoke until calibrated structural validation closes. |
| PR #678 / issues #553/#659 | Sparse J2 and ductile PF backend-promotion harnesses | Backend promotion evidence exists for SciPy/MUMPS paths; cuDSS remains gated on a functional GPU-node promotion environment. |

Retained customer-facing visual evidence currently includes:

- `plasticity_interface_43131`: 21 PNGs across ductile PF-plasticity,
  weak-interface deflection, and strong-interface penetration; all three
  visual manifests pass their review-dimension checks.
- `plasticity_interface_prod_43186`: 2 cohesive displacement-jump PNGs; visual
  manifest passes. The merged #554 follow-up strengthens this benchmark with a
  true bilinear TSL correction and a mode-I dissipated-energy capacity check.
- The merged #554 follow-up adds a mixed-mode cohesive benchmark with
  normal/shear traction checks, scalar damage history, and cohesive tangent
  finite-difference evidence.
- The merged #554 contact follow-up adds an optional normal-contact penalty
  and a pure-compression benchmark that checks contact traction, zero damage
  growth, and contact tangent finite-difference error.
- The merged #554 delamination follow-up adds a four-segment mixed-mode
  cohesive patch benchmark with closed-form resultant checks, localized damage
  and delamination-front metrics, and three review-dimension-checked plots.
- The merged #554 structural follow-up adds a DCB-style precracked cohesive
  benchmark with free bulk DOFs, load-displacement/post-peak softening,
  damage-front metrics, energy plots, mesh artifacts, and four
  review-dimension-checked plots.
- `structural_dcb_pr672_latest`: retained DCB-style structural cohesive bundle
  for the structural benchmark implementation, with `validation_passed=true`,
  front advance from initial crack `x=1.5` to `x=3.75`, post-peak softening,
  bounded diagnostic energy-balance gap, four review-dimension-checked plots,
  and focused suite `104 passed`.
- `cohesive_delamination_patch_pr672_latest`: cohesive law and delamination
  patch bundle with mode-I dissipated-energy capacity check, 3 cohesive
  delamination patch PNGs, visual manifest pass, and focused suite
  `103 passed`.
- `ductile_pf_sensitivity_43754`: 2 ductile sensitivity PNGs plus retained
  child bundles; visual manifest passes. The merged follow-up strengthens the
  base ductile validation bundle with separated elastic driving, plastic work,
  degraded elastic, fracture surface/gradient, and total
  stored-plus-dissipated energy outputs.
- `ductile_pf_energy_pr672_latest`: retained merged-stack ductile energy
  ledger bundle with finite-term and monotonicity checks.
- PR #678 adds a solver-driven diffuse-interface validation smoke for #674:
  weak-interface deflection and strong-interface penetration are classified
  from solved AT2 damage fields using `PhaseFieldDamageSolver.solve` with
  per-element `Gc(x)`, standard visual bundles, animation media, CSV telemetry,
  and provenance manifests. Local focused pytest coverage checks both outcomes
  and artifact integrity.
- PR #678 also hardens the native Q4 slice for #675: isotropic Q4 mechanics now
  assembles through the sparse-direct `QuasiStaticSolver` path, and Q4 AT2
  damage uses matrix-free 2x2-Gauss quadrature with Gauss-point history. The
  customer-safe Q4 claim remains beta and excludes Q4 PF-CZM, AT1, plasticity,
  cohesive-coupled damage, direct damage assembly, and differentiable damage
  adjoints.

The June 2026 plasticity/interface stack has been manually integrated into
`main` through PR #666 and the redundant stacked PRs #667, #668, #669, #670,
and #672 were closed after their content was incorporated. Maintainer-only
handoff notes remain in the private development repository and are not part of
the staged CEMS-Lab public payload.

Current issue-level status is tracked in GitHub issues #550, #553, and #554
and in the stacked PR comments. Do not promote any additional benchmark from
"running" to "validated" until the logs, summaries, visual manifests, and
compare artifacts are attached to the relevant GitHub issue or PR.

## Release Gates

### Gate 0 - Honest API Boundary

- Reject unsupported coupled plasticity at solver construction.
- Keep the docs explicit about which cohesive paths are solver-coupled and
  which remain roadmap work.
- Maintain a public capability matrix in docs and README.

### Gate 1 - Elastoplastic Mechanics Core

Deliverable: implicit small-strain J2 mechanics without phase-field damage.

Current branch status: the per-element state layer, return-mapping call,
internal-force assembly, plastic-work accounting, commit/rollback API, and
`QuasiStaticSolver` sparse J2 dispatch are implemented and tested. The current
J2 tangent is assembled from element algorithmic tangents of the return map,
and the sparse backend promotion harness records standard provenance. Recent
backend-promotion evidence shows `backend='auto'` and explicit MUMPS resolving
to MUMPS when PETSc/MUMPS is available, SciPy parity on the tested cases, and
`backend='cudss'` falling back cleanly when cuDSS is unavailable. The remaining
backend gate is a functional cuDSS promotion environment plus larger
production-scale J2 studies.

- Add per-element plastic state: stress, plastic strain, equivalent plastic
  strain, trial copies, commit/rollback.
- Wire `J2Plasticity.step()` into mechanics stress evaluation.
- Replace the current return-map differentiated element tangent with a
  closed-form Simo-Taylor elastoplastic tangent for the promoted production
  path.
- Validate uniaxial tension, unload/reload, cyclic loading, plane stress, and
  plane strain against closed-form or Abaqus/COMSOL reference curves.

Exit criteria:

- Newton convergence is mesh-independent on elastic-plastic patch tests.
- Consistent tangent matches finite-difference Jacobian on a one-element test.
- State rollback is correct after failed line-search/Newton attempts.

### Gate 2 - Phase-Field Ductile Fracture

Deliverable: PF + plasticity coupling with a documented energy split.

Current branch status: the first coupling mode is implemented as
`elastic tensile energy + accumulated plastic work` in the ductile
phase-field driving force. The branch contains a runnable solver-level J2
mechanics plus bounded AT2 damage example, guarded T3 J2+AT2 staggered smoke
coverage, and a retained elastic-reference/length-scale sensitivity study.
The validation example now writes an integrated energy ledger with finite-term
and monotonicity checks. Ductile SENT/TPB benchmark matching remains open.

- Choose one formulation for the first release: Ambati/Borden/Miehe ductile PF
  or a newer length-scale-insensitive PF-CZM-plasticity formulation.
- Add plastic-work or plastic-free-energy contribution to the damage driving
  force.
- Define exactly what is degraded: elastic energy only, plastic energy, yield
  stress, or separate compliance/damage variables.
- Validate against ductile SENT and three-point-bend references.

Exit criteria:

- Load-displacement and crack path match reference data within documented
  tolerances.
- Mesh and length-scale sensitivity are quantified.
- Energy accounting separates elastic, plastic, fracture, external, and kinetic
  terms.

### Gate 3 - Cohesive Interfaces

Deliverable: zero-thickness cohesive-zone interfaces coupled to the mechanics
residual.

Merged stack status: bilinear cohesive residual and tangent assembly are
solver-coupled through the quasi-static cohesive hook. The mode-I benchmark
checks integrated dissipated energy against the analytical bilinear fracture
energy capacity. Mode-I displacement-jump, mixed-mode tangent, normal
contact-compression, a multi-element delamination patch benchmark, and a
DCB-style structural cohesive smoke exist; ASTM-calibrated structural
delamination remains open. A coupled brittle PF+cohesive smoke now exercises
AT2 matrix damage and cohesive-interface delamination in one staggered loop.

Completed in the current stack:

- Turn `CohesiveElement` side data into residual and tangent contributions.
- Implement true bilinear traction-separation with scalar mixed-mode damage
  history, optional normal contact penalty, and scalar dissipated-energy state.
- Validate single-interface opening, mixed-mode response, contact compression,
  multi-element delamination patch, and DCB-style structural smoke tests.
- Validate a coupled brittle PF+cohesive smoke with a notched matrix,
  cohesive front tracking, energy split, convergence plot, animation, CSV
  telemetry, and visual manifest.

Remaining production-hardening work:

- Add PPR/Camanho-style mixed-mode law variants and richer unloading/reloading
  policies.
- Add ASTM/analytical DCB calibration and structural mesh refinement studies.
- Couple cohesive interfaces into full staggered phase-field workflows where
  required by customer problems.

Exit-criteria status:

- Achieved: global Newton includes the cohesive tangent in the sparse
  quasi-static hook.
- Achieved for the mode-I validation path: integrated cohesive dissipation
  equals the analytical bilinear fracture-energy capacity at complete
  separation.
- Achieved for the T3/Q4 array insertion helpers:
  `insert_cohesive_layer_with_metadata` preserves node sets, side-specific
  interface node sets, element sets, and per-element material/region arrays
  for T3 and Q4 connectivity;
  `insert_cohesive_layer_meshio` extracts a named line-cell interface set from
  a single-block T3 or Q4 `meshio.Mesh`, extends point data onto duplicate
  nodes, preserves cell metadata, and can write the updated mesh through
  meshio.
- Remaining: multi-element-family import studies and larger structural
  regression studies before advertising arbitrary external cohesive meshes as
  production-ready.

### Gate 4 - Combined Microstructure Workflows

Deliverable: production examples combining heterogeneity with one nonlinear
failure mechanism at a time, then the full coupled stack.

- Heterogeneous brittle PF: already largely present.
- Heterogeneous elastoplastic mechanics: after Gate 1.
- Heterogeneous PF-plasticity: after Gate 2.
- Heterogeneous cohesive interfaces and coupled brittle PF+cohesive smoke:
  after Gate 3.
- Combined PF-plasticity-cohesive microstructure: final integration scenario.

Exit criteria:

- Each example has a reproducible config, reference output, convergence report,
  and runtime/memory envelope.
- HPC and CPU paths agree within tolerances for small cases.

## Customer Messaging Until Gates Land

Use:

> The solver supports implicit brittle phase-field fracture with heterogeneous
> material fields and microstructure-style generators. Plasticity and cohesive
> elements are active development tracks: sparse quasi-static J2 mechanics,
> bounded ductile AT2 damage validation, cohesive residual/tangent smoke
> coverage, and a coupled brittle PF+cohesive validation smoke are available,
> but the fully coupled implicit PF-plasticity-CZM benchmark workflow is not
> yet released.

Avoid:

> Abaqus/COMSOL-equivalent coupled elastoplastic cohesive fracture.

That statement becomes acceptable only after Gates 1-4 have benchmark evidence.
