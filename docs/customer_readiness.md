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
| Heterogeneous elastic and fracture fields | Implemented | Per-element `E(x)` and `Gc(x)` fields support inclusions, weak zones, void proxies, and microstructure-style forward studies. |
| Geometric microstructures | Implemented for forward studies | Random microstructures, Voronoi phases, stiff inclusions, degraded void proxies, and gmsh-carved circular voids are available for forward fracture benchmarks and examples. |
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

The public repository keeps validation evidence reproducible through checked-in
examples, tests, manifests, CSV/JSON summaries, and lightweight documentation
figures. Internal CI/HPC job roots, private review bundles, and raw generated
outputs are not part of the public evidence surface.

| Evidence class | Public scope | Release status |
| --- | --- | --- |
| Base plasticity/interface examples | J2 material-point checks and ductile PF-plasticity validation | Beta validation examples; not a full coupled product claim. |
| Cohesive operator examples | Mode-I opening, mixed-mode tangent, contact compression, multi-element delamination patch, and DCB-style structural smoke | Beta validation examples with analytical or finite-difference checks where applicable. |
| Ductile PF-plasticity energy accounting | Elastic driving, plastic work, fracture terms, and bounded AT2 damage residuals | Beta evidence for the guarded T3 `j2_isotropic` + AT2 path only. |
| Sparse backend promotion | SciPy baseline plus optional PETSc/MUMPS/cuDSS dispatch checks where available | Optional-backend evidence; backend availability must be verified on the target machine. |
| Native Q4 mechanics and AT2 damage slice | Q4 mesh admission, isotropic mechanics assembly, scalar Laplacian/mass, and matrix-free AT2 damage with Gauss-point history | Beta; Q4 PF-CZM, AT1, plasticity, cohesive-coupled damage, direct damage assembly, and differentiable damage adjoints remain gated. |

Before promoting any additional benchmark from "running" to "validated",
attach enough public evidence for a reviewer to reproduce the claim: command or
YAML, solver/device/backend settings, logs or summary tables, visual manifests,
compare artifacts, and known caveats. Private retained evidence may support
internal review, but it must not be the only basis for a public claim.

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
and the sparse backend promotion harness now records standard provenance.
Internal backend-promotion runs have shown SciPy/PETSc-MUMPS parity for the
sparse J2 and coupled ductile validation paths on the tested CPU stack, while
cuDSS remains a clean fallback when the GPU sparse-direct environment is not
available. The remaining backend gate is a functional cuDSS promotion
environment plus larger production-scale J2 studies.

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
