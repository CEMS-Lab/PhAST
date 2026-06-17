# Capability matrix

This page is the public boundary for what the solver can be used for today.
It separates production paths from beta, experimental, optional-backend, and
unsupported work so a YAML file or tutorial does not imply more physics than
the code actually solves.

For the current plasticity/cohesive/PF-CZM technical-preview release boundary,
see the [plasticity/interface supported-workflow page](../supported_workflows/plasticity_interface_beta.md).

Status meanings:

| Status | Meaning |
|---|---|
| Production | Available through the normal Python/YAML path with tests and at least one benchmark or baseline validation workflow. |
| Beta | Implemented and useful, but validation coverage or ergonomics still need hardening before customer delivery. |
| Experimental | Research path. Use for comparisons or development, not for customer commitments. |
| Optional backend | Code path exists, but depends on external solver libraries or HPC environment validation. |
| Scaffold | Data structures or helper kernels exist, but the feature is not coupled into the production solve. |
| Unsupported | Do not advertise as available. |

## Simulation physics

| Capability | Status | Public statement |
|---|---|---|
| Small-strain 2D linear elasticity | Production | Supported through the mechanics kernels and quasi-static/static examples. |
| Brittle phase-field fracture, AT2 | Production | Supported for explicit dynamics and staggered quasi-static/static solves. |
| Brittle phase-field fracture, AT1 | Beta | Supported with projected damage solve and AT1 threshold fields; benchmark coverage is still being expanded. |
| Heterogeneous elastic fields `E(x)` | Production | Per-element fields support inclusions, weak/strong bands, and inverse calibration demos. |
| Heterogeneous fracture fields `Gc(x)` | Production | Per-element fields support weak zones, spatial recovery, and microstructure-style studies. |
| Diffuse interface fracture validation | Beta validation | Solver-driven weak-interface deflection and strong-interface penetration examples use spatial `E(x)`/`Gc(x)` fields plus AT2 damage solves with visuals and telemetry; discrete cohesive and PF-CZM structural calibration remain separate capabilities. |
| Plane strain | Production | Default 2D constitutive setting. |
| Plane stress | Beta | Available via `material.plane_stress`; benchmark coverage is narrower than plane strain. |
| Spectral / Amor / isotropic energy splits | Production | Available through `material.energy_split` for brittle phase-field workflows. Plane-stress `spectral` is a reduced 2D in-plane projection, not a fully condensed 3D plane-stress spectral decomposition; prefer plane-strain `spectral` or plane-stress `amor` for mature validated paths. |
| `spectral_stress` split | Experimental | Opt-in COMSOL-parity research path; do not use as the default customer claim. |
| Monolithic `(u,d)` L-BFGS solve | Experimental | Research comparison only until the bound-constrained irreversibility work closes. |
| Sparse quasi-static J2 elastoplasticity | Beta | Per-element state, return mapping, commit/rollback, internal force, sparse `QuasiStaticSolver` dispatch, and plastic-work accounting are available through the plasticity API and validation example; large-mesh backend promotion remains gated. |
| Ductile PF-plasticity validation | Beta | Elastic tensile energy plus accumulated plastic-work coupling is implemented, with a bounded AT2 phase-field damage solve, separated elastic/plastic/fracture energy ledger, guarded quasi-static T3 J2+AT2 staggered support, and a retained elastic-reference/`l0` sensitivity study; benchmark-matched ductile fracture remains gated. |
| Cohesive elements / discrete CZM | Beta | Stateful true-bilinear cohesive residual/tangent assembly, scalar dissipated-energy history, optional normal-contact penalty, sparse quasi-static validation coverage, metadata-preserving T3/Q4 array and single-block meshio cohesive-layer insertion, mode-I/mixed-mode/contact/delamination/structural validation benchmarks, and visual manifests are available; ASTM-calibrated structural delamination and mixed-family external mesh import remain gated. |
| Coupled brittle PF + cohesive elements | Beta | Staggered AT2 matrix damage plus zero-thickness cohesive-interface delamination validation exists with matrix notch damage, cohesive front metrics, energy split, convergence, animation, CSV telemetry, and visual manifest; calibrated PF-CZM and structural validation studies remain gated. |
| PF-CZM | Beta validation | Wu PF-CZM is available as `pf_model: PFCZM` for forward nonlinear damage solves with tensile-strength-calibrated rational degradation, element-wise gamma-corrected calibration, residual/convergence telemetry, and a uniaxial strength/`l0` validation bundle; structural crack-growth, mixed-mode delamination, differentiable adjoints, and PF-plasticity-cohesive coupling remain gated. |
| Coupled PF + plasticity + cohesive interfaces | Unsupported | Not part of the public workflow surface. |
| 3D fracture | Unsupported | Current production elements are 2D triangles. |
| P2 / Q8 / Q9 element primitives | Scaffold | Shape functions, quadrature, and single-element stiffness tests exist for higher-order families; global production solver dispatch remains gated. |
| Native Q4 isotropic mechanics + AT2 damage | Beta | Structured Q4 mesh helpers, native Q4 mesh admission, 2x2-Gauss isotropic mechanics, SciPy/MUMPS sparse-direct stiffness assembly, scalar Laplacian/mass, and matrix-free Q4 AT2 damage with Gauss-point history are tested. Q4 PF-CZM, AT1, plasticity, cohesive-coupled damage, direct damage assembly, and differentiable damage adjoints remain gated. |

## Solvers and backends

| Capability | Status | Public statement |
|---|---|---|
| Explicit dynamics, Velocity Verlet | Production | Main validated dynamic fracture path. |
| Staggered quasi-static/static solve | Production | Customer-facing implicit brittle-fracture path, with `jacobi` as the conservative damage preconditioner. Matrix-free CG and sparse-direct mechanics backends are available; non-isotropic sparse direct uses a frozen-state secant tangent for validation robustness. |
| `quasi_static_legacy` secant path | Beta | Retained for compatibility and selected MPC/frozen-secant workflows. |
| SciPy SuperLU sparse direct baseline | Production | Always-available sparse direct baseline where SciPy is installed. |
| PETSc/MUMPS | Optional backend | Runtime-verification guarded; when available, `backend='auto'` chooses this as the CPU sparse-direct mechanics backend before SciPy SuperLU. |
| cuDSS / nvmath | Optional backend | Runtime-verification guarded; GPU sparse-direct path requires current nvmath/cuDSS validation on the target GPU stack. |
| AMG / AmgX / GMG damage preconditioning | Experimental for QS fracture | Useful performance paths, but quasi-static customer runs should default to Jacobi unless validating the preconditioner itself. |
| Anderson acceleration | Beta | Available for staggered iterations; use with benchmark-specific validation. |

## YAML and workflows

| Capability | Status | Public statement |
|---|---|---|
| YAML problem definition | Production | Canonical entry point: `python -m phast run configs/benchmarks/dynamic/B3_dynamic_sent.yaml`; use `python -m phast explain-config <config.yaml>` before long runs. |
| YAML schema validation | Production | `--validate-only` catches schema errors with line-numbered messages. |
| `explain-config` dry-run review | Production | Prints selected physics, solver path, outputs, provenance, and setup warnings without meshing/running. |
| `schema_version` | Production | Shipped configs declare `schema_version: 1`; older files still load but receive review warnings. |
| Resolved run lockfile | Production | YAML runs write `run_lockfile.json` with config hash, post-CLI resolved config, CLI args, git state, dependencies, and resolved object summaries. |
| Built-in geometry generators | Production | Supported through `geometry.type` and `geometry.parameters`. |
| External meshes | Beta | Supported via `geometry.mesh_path`; node-set compatibility is mesh-format dependent. |
| Declarative primitive geometry DSL | Beta | Parsed and used by current benchmark configs; continue validating new domain/mesh recipes before customer delivery. |
| Config inheritance/includes/parametric studies | Unsupported | Not implemented yet; use copied YAMLs or scripts for parameter studies. |
| JSON Schema export / IDE autocomplete | Production | `python -m phast schema` exports the checked-in schema generated from the config model, enum tables, and numeric ranges. |

## Outputs and validation artifacts

| Capability | Status | Public statement |
|---|---|---|
| Zarr trajectory stores | Production for dataset generation | Primary reusable format for new trajectory and large dataset workflows; chunked, appendable, and compatible with parallel readers. |
| HDF5 snapshots | Legacy compatibility | Supported for existing archived artifacts, benchmark post-processing, and import/export bridges; do not use as the default for new large training corpora. |
| VTU/PyVista-style visualization output | Beta | Available via output settings; exact format support depends on optional visualization dependencies. |
| GIF/plot generation | Beta | Useful for demos and reports; verify generated artifacts before using in papers. |
| Reaction-force logging | Production for QS benchmarks | Set `output.reaction_node_set` and `output.reaction_component` for load-displacement comparisons. |
| Benchmark `compare.py` scripts | Beta | Available for selected examples; tolerance coverage is still expanding across the full suite. |
| CPU/HPC portability | Production for core CPU/CUDA paths | CPU is valid for small and quasi-static validation; large dynamic runs are GPU/HPC-oriented. |
