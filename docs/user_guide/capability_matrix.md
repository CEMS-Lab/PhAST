# Capability matrix

This page is the public boundary for what the solver can be used for today.
It separates production paths from beta, experimental, optional-backend, and
unsupported work so a YAML file or tutorial does not imply more physics than
the code actually solves.

For the current plasticity/cohesive/PF-CZM technical-preview release boundary,
see [`../plasticity_cohesive_beta_release.md`](../plasticity_cohesive_beta_release.md).

Status meanings:

| Status | Meaning |
|---|---|
| Production | Available through the normal Python/YAML path with tests and at least one benchmark or smoke workflow. |
| Beta | Implemented and useful, but validation coverage or ergonomics still need hardening before customer delivery. |
| Experimental | Research path. Use for comparisons or development, not for customer commitments. |
| Optional backend | Code path exists, but depends on external solver libraries or HPC environment validation. |
| Scaffold | Data structures or helper kernels exist, but the feature is not coupled into the production solve. |
| Unsupported | Do not advertise as available. |

## Simulation physics

| Capability | Status | Customer-safe statement |
|---|---|---|
| Small-strain 2D linear elasticity | Production | Supported through the mechanics kernels and quasi-static/static examples. |
| Brittle phase-field fracture, AT2 | Production | Supported for explicit dynamics and staggered quasi-static/static solves. |
| Brittle phase-field fracture, AT1 | Beta | Supported with projected damage solve and AT1 threshold fields; benchmark coverage is still being expanded. |
| Heterogeneous elastic fields `E(x)` | Production | Per-element fields support inclusions, weak/strong bands, and inverse calibration demos. |
| Heterogeneous fracture fields `Gc(x)` | Production | Per-element fields support weak zones, spatial recovery, and microstructure-style studies. |
| Diffuse interface fracture validation | Beta smoke | Solver-driven weak-interface deflection and strong-interface penetration examples use spatial `E(x)`/`Gc(x)` fields plus AT2 damage solves with visuals and telemetry; discrete cohesive and PF-CZM structural calibration remain separate capabilities. |
| Plane strain | Production | Default 2D constitutive setting. |
| Plane stress | Beta | Available via `material.plane_stress`; benchmark coverage is narrower than plane strain. |
| Spectral / Amor / isotropic energy splits | Production | Available through `material.energy_split` for brittle phase-field workflows. Plane-stress `spectral` is a reduced 2D in-plane projection, not a fully condensed 3D plane-stress spectral decomposition; prefer plane-strain `spectral` or plane-stress `amor` for mature validated paths. |
| `spectral_stress` split | Experimental | Opt-in COMSOL-parity research path; do not use as the default customer claim. |
| Monolithic `(u,d)` L-BFGS solve | Experimental | Research comparison only until the bound-constrained irreversibility work closes. |
| Sparse quasi-static J2 elastoplasticity | Beta | Per-element state, return mapping, commit/rollback, internal force, sparse `QuasiStaticSolver` dispatch, and plastic-work accounting are available through the plasticity API and validation example. Smoke evidence exists through PR #667/#668; large-mesh backend promotion remains gated. |
| Ductile PF-plasticity validation | Beta | Elastic tensile energy plus accumulated plastic-work coupling is implemented, with a bounded AT2 phase-field damage solve, separated elastic/plastic/fracture energy ledger, guarded quasi-static T3 J2+AT2 staggered support, and a retained elastic-reference/`l0` sensitivity study; benchmark-matched ductile fracture remains gated. |
| Cohesive elements / discrete CZM | Beta | Stateful true-bilinear cohesive residual/tangent assembly, scalar dissipated-energy history, optional normal-contact penalty, sparse quasi-static smoke coverage, metadata-preserving T3/Q4 array and single-block meshio cohesive-layer insertion, mode-I/mixed-mode/contact/delamination/structural smoke benchmarks, and visual manifests are available; ASTM-calibrated structural delamination and mixed-family external mesh import remain gated. |
| Coupled brittle PF + cohesive elements | Beta | Staggered AT2 matrix damage plus zero-thickness cohesive-interface delamination smoke exists with matrix notch damage, cohesive front metrics, energy split, convergence, animation, CSV telemetry, and visual manifest; calibrated PF-CZM and structural validation studies remain gated. |
| PF-CZM | Beta smoke | Wu PF-CZM is available as `pf_model: PFCZM` for forward nonlinear damage solves with tensile-strength-calibrated rational degradation, element-wise gamma-corrected calibration, residual/convergence telemetry, and a uniaxial strength/`l0` validation bundle; structural crack-growth, mixed-mode delamination, differentiable adjoints, and PF-plasticity-cohesive coupling remain gated. |
| Coupled PF + plasticity + cohesive interfaces | Unsupported | Do not sell as available. This is a staged product-hardening track. |
| 3D fracture | Unsupported | Current production elements are 2D triangles. |
| P2 / Q8 / Q9 element primitives | Scaffold | Shape functions, quadrature, and single-element stiffness tests exist for higher-order families; global production solver dispatch remains gated. |
| Native Q4 isotropic mechanics + AT2 damage | Beta | Structured Q4 mesh helpers, native Q4 mesh admission, 2x2-Gauss isotropic mechanics, SciPy/MUMPS sparse-direct stiffness assembly, scalar Laplacian/mass, and matrix-free Q4 AT2 damage with Gauss-point history are tested. Q4 PF-CZM, AT1, plasticity, cohesive-coupled damage, direct damage assembly, and differentiable damage adjoints remain gated. |

## Solvers and backends

| Capability | Status | Customer-safe statement |
|---|---|---|
| Explicit dynamics, Velocity Verlet | Production | Main validated dynamic fracture path. |
| Staggered quasi-static/static solve | Production | Customer-facing implicit brittle-fracture path, with `jacobi` as the conservative damage preconditioner. Matrix-free CG and sparse-direct mechanics backends are available; non-isotropic sparse direct uses a frozen-state secant tangent for validation robustness. |
| `quasi_static_legacy` secant path | Beta | Retained for compatibility and selected MPC/frozen-secant workflows. |
| SciPy SuperLU sparse direct baseline | Production | Always-available sparse direct baseline where SciPy is installed. |
| PETSc/MUMPS | Optional backend | Runtime smoke-guarded; when available, `backend='auto'` chooses this as the CPU sparse-direct mechanics backend before SciPy SuperLU. |
| PARDISO / SPOOLES | Not wired | Commercial sparse direct solvers exposed by COMSOL; not currently called by this repository. |
| cuDSS / nvmath | Optional backend | Runtime smoke-guarded; GPU sparse-direct path requires current nvmath/cuDSS validation on the target GPU stack. |
| AMG / AmgX / GMG damage preconditioning | Experimental for QS fracture | Useful performance paths, but quasi-static customer runs should default to Jacobi unless validating the preconditioner itself. |
| Anderson acceleration | Beta | Available for staggered iterations; use with benchmark-specific validation. |

## YAML and workflows

| Capability | Status | Customer-safe statement |
|---|---|---|
| YAML problem definition | Production | Canonical entry point: `python -m phast run configs/benchmarks/dynamic/B3_dynamic_sent.yaml`; use `python -m phast explain-config <config.yaml>` before long runs. |
| YAML schema validation | Production | `--validate-only` catches schema errors with line-numbered messages. |
| `explain-config` dry-run review | Production | Prints selected physics, solver path, outputs, provenance, and setup warnings without meshing/running. |
| `schema_version` | Production | Shipped configs declare `schema_version: 1`; older files still load but receive review warnings. |
| Resolved run lockfile | Production | YAML runs write `run_lockfile.json` with config hash, post-CLI resolved config, CLI args, git state, dependencies, and resolved object summaries. |
| Internal workflow capability registry | Production foundation | The internal `ProblemSpec` layer records public solver, step, material-model, and boundary-condition names for compatibility checks; strict runtime enforcement remains deferred. |
| Internal workflow execution plan | Production foundation | Supported `ProblemSpec` instances route to existing compatibility runners without changing solver loops; direct `ProblemSpec.run()` remains deferred. |
| Internal workflow region validation | Production foundation | Contract-level material, initial-condition, boundary-condition, and history-output region references are checked against declared regions without changing YAML keys. |
| Built-in geometry generators | Production | Supported through `geometry.type` and `geometry.parameters`. |
| External meshes | Beta | Supported via `geometry.mesh_path`; node-set compatibility is mesh-format dependent. |
| Declarative primitive geometry DSL | Beta | Parsed and used by current benchmark configs; continue validating new domain/mesh recipes before customer delivery. |
| Config inheritance/includes/sweeps | Unsupported | Not implemented yet; use copied YAMLs or scripts for parameter sweeps. |
| JSON Schema export / IDE autocomplete | Production | `python -m phast schema` exports the checked-in schema generated from the dataclass config model, enum tables, and numeric ranges. |

### Registered workflow contract names

These names are validated against the internal workflow capability registry.
They are not a separate solver dispatch table; the current runners remain the
compatibility boundary.

| Category | Name | Status | Description |
|---|---|---|---|
| `analysis_step` | `explicit` | Production | Dynamic fracture step represented by existing loading config. |
| `analysis_step` | `quasi_static` | Production | Quasi-static fracture step represented by existing loading config. |
| `analysis_step` | `quasi_static_legacy` | Beta | Legacy quasi-static step represented by existing loading config. |
| `analysis_step` | `solid_mechanics` | Production | Solid-mechanics example load step. |
| `analysis_step` | `validation_script` | Beta | Curated validation-script step for beta physics contracts. |
| `boundary_condition` | `fix` | Production | Dirichlet zero-displacement boundary condition. |
| `boundary_condition` | `neumann` | Production | Neumann force/traction boundary condition. |
| `boundary_condition` | `pf_dirichlet` | Production | Phase-field Dirichlet boundary condition. |
| `boundary_condition` | `prescribe` | Production | Dirichlet prescribed displacement boundary condition. |
| `boundary_condition` | `rigid_connector` | Beta | Compatibility rigid-connector boundary condition. |
| `boundary_condition` | `symmetry` | Production | Symmetry-plane boundary condition used by public examples. |
| `boundary_condition` | `traction` | Production | Boundary traction/load condition. |
| `field_output` | `acceleration` | Production | Dynamic acceleration field output. |
| `field_output` | `damage` | Production | Damage field output. |
| `field_output` | `displacement` | Production | Displacement field output. |
| `field_output` | `equivalent_plastic_strain` | Beta | Promoted J2 solid-mechanics plastic-strain visual artifact. |
| `field_output` | `higher_order_element_fields` | Scaffold | Placeholder for higher-order element field output; helper kernels exist but workflow output is not coupled. |
| `field_output` | `history_field` | Production | Phase-field history variable output. |
| `field_output` | `history_field_nodal` | Production | Nodal phase-field history variable output. |
| `field_output` | `jacobian` | Beta | Promoted nonlinear solid-mechanics Jacobian visual artifact. |
| `field_output` | `psi_plus` | Production | Positive strain-energy density output. |
| `field_output` | `strain` | Production | Stored strain field output. |
| `field_output` | `strain_energy` | Beta | Promoted solid-mechanics strain-energy visual artifact. |
| `field_output` | `stress` | Production | Stored stress field output. |
| `field_output` | `trajectory` | Production | Stored Zarr/H5 trajectory field output. |
| `field_output` | `velocity` | Production | Dynamic velocity field output. |
| `field_output` | `von_mises` | Beta | Promoted solid-mechanics von Mises output artifact. |
| `field_output` | `vtu` | Beta | VTU/PyVista-style visualization field output. |
| `history_output` | `energy` | Production | Energy history output. |
| `history_output` | `load_displacement` | Production | Canonical load-displacement response history. |
| `history_output` | `max_damage` | Production | Maximum damage scalar history. |
| `history_output` | `reaction` | Production | Reaction history for load-displacement workflows. |
| `history_output` | `reaction_force` | Production | Canonical reaction-force history. |
| `history_output` | `response` | Production | Promoted solid-mechanics response output. |
| `history_output` | `solver_telemetry` | Production | Solver telemetry history. |
| `history_output` | `timing_per_step` | Production | Per-step timing history. |
| `material_model` | `cohesive_interface` | Beta | Curated cohesive-interface validation contract material. |
| `material_model` | `diffuse_interface` | Beta | Curated diffuse-interface validation contract material. |
| `material_model` | `ductile_phase_field` | Beta | Curated ductile phase-field/plasticity validation material. |
| `material_model` | `j2_plasticity` | Beta | Curated J2 plasticity validation contract material. |
| `material_model` | `phase_field` | Production | Current brittle phase-field material contract model. |
| `material_model` | `solid_mechanics` | Production | Promoted solid-mechanics material contract model. |
| `material_model` | `validation_artifact` | Beta | Fallback material marker for curated validation artifacts. |
| `postprocess` | `animation` | Beta | Shared animation postprocess. |
| `postprocess` | `damage_final` | Beta | Final damage visual artifact. |
| `postprocess` | `energy` | Beta | Energy plot visual artifact. |
| `postprocess` | `initial_conditions` | Beta | Initial-condition visual artifact. |
| `postprocess` | `plots` | Beta | Shared plot generation postprocess. |
| `postprocess` | `thumbnail` | Beta | Thumbnail visual artifact. |
| `solver` | `coupled_pf_plasticity_cohesive` | Unsupported | Future coupled plasticity, phase-field, and cohesive-interface solver; not available as a public workflow. |
| `solver` | `explicit` | Production | Velocity-Verlet explicit dynamics fracture path. |
| `solver` | `quasi_static` | Production | Staggered quasi-static/static brittle-fracture path. |
| `solver` | `quasi_static_legacy` | Beta | Compatibility secant path retained for selected workflows. |
| `solver` | `solid_mechanics` | Production | Promoted solid-mechanics YAML runner path. |
| `solver` | `validation_script` | Beta | Curated plasticity/interface reproducibility-contract route. |

## Outputs and validation artifacts

| Capability | Status | Customer-safe statement |
|---|---|---|
| Zarr trajectory stores | Production for dataset generation | Primary reusable format for new trajectory and large dataset workflows; chunked, appendable, and compatible with parallel readers. |
| HDF5 snapshots | Legacy compatibility | Supported for existing archived artifacts, benchmark post-processing, and import/export bridges; do not use as the default for new large training corpora. |
| VTU/PyVista-style visualization output | Beta | Available via output settings; exact format support depends on optional visualization dependencies. |
| GIF/plot generation | Beta | Useful for demos and reports; verify generated artifacts before using in papers. |
| Reaction-force logging | Production for QS benchmarks | Set `output.reaction_node_set` and `output.reaction_component` for load-displacement comparisons. |
| Benchmark `compare.py` scripts | Beta | Available for selected examples; tolerance coverage is still expanding across the full suite. |
| CPU/HPC portability | Production for core CPU/CUDA paths | CPU is valid for small and quasi-static validation; large dynamic runs are GPU/HPC-oriented. |

## Customer messaging

Use:

> The solver supports YAML-driven 2D brittle phase-field fracture with
> differentiable PyTorch mechanics, explicit dynamics, staggered
> quasi-static/static solves, heterogeneous `E(x)`/`Gc(x)` fields, sparse
> quasi-static J2 plasticity validation APIs, a bounded ductile AT2 damage
> validation example plus sensitivity study, solver-driven diffuse
> solid-interface examples,
> cohesive displacement-jump, mixed-mode, contact-compression, and
> delamination patch benchmarks plus DCB-style structural and coupled brittle
> PF+cohesive smoke examples, and review tooling such as `validate-only` and
> `explain-config`.

Avoid:

> Abaqus/COMSOL-equivalent coupled elastoplastic cohesive fracture.

That claim becomes defensible only after the plasticity, cohesive interface,
and PF-CZM tracks have coupled residual/tangent implementations plus benchmark
evidence.
