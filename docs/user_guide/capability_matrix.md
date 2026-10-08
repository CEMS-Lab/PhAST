# Capability Matrix

This capability matrix states which PhAST features are supported and how far
they have been tested. It distinguishes established use from beta,
experimental, optional-backend, scaffold, and unsupported features.

For the present scope of plasticity, cohesive-interface, and PF-CZM support,
see the [plasticity and interface beta workflow](../supported_workflows/plasticity_interface_beta.md).

## Status Definitions

| Status | Meaning |
|---|---|
| **Supported** | Implemented in the documented solver interface and covered by repository tests or a reproducible example. Support remains specific to the stated problem class. |
| **Beta** | Mathematically implemented and functional, but robustness or full-domain validation is still under review. |
| **Experimental** | Active research pathway with insufficient evidence for general research claims. |
| **Optional backend** | Implemented but requires external solver libraries or a compatible computing environment. |
| **Scaffold** | Foundational data structures or kernels exist, but the feature is not yet coupled into the global solve. |
| **Unsupported** | Feature is mathematically or computationally unsupported. |

## Simulation Physics

| Capability | Status | Public Statement |
|---|---|---|
| Small-strain 2D linear elasticity | Supported | Covered by the core mechanics tests and documented static and quasi-static examples. |
| Small-strain 1D axial elasticity | Beta | Separate CPU float64 Python API: `line_mesh`, `LineMesh`, and `solve_bar`. Uniform material/area, one left displacement constraint and a right-end force; analytical fields and first-order gradients are tested. See the [1D bar API](one_dimensional_bar.md). |
| Brittle phase-field fracture, AT2 | Supported | Available for explicit dynamics and staggered quasi-static/static solves within the documented examples. |
| Brittle phase-field fracture, AT1 | Beta | Available through projected damage solves and AT1 threshold fields. Validation coverage is narrower than for AT2. |
| Heterogeneous elastic fields `E(x)` | Supported programmatic path | Per-element fields support structural inclusions and weak/strong bands. The element ordering and low-level authoring route are demonstrated in the [heterogeneous-fields teaching example](https://github.com/CEMS-Lab/PhAST/tree/main/examples/heterogeneous_fields); arbitrary field maps are not currently a general YAML feature. |
| Heterogeneous fracture fields `Gc(x)` | Supported programmatic path | Per-element fields support weak zones and microstructure-style forward studies through the same [heterogeneous-fields teaching example](https://github.com/CEMS-Lab/PhAST/tree/main/examples/heterogeneous_fields). The example is not a coupled benchmark or material-calibration claim. |
| Schema-2 multi-material quasi-static fracture | Beta | DCB and heterogeneous SENT use a structured T3 rectangle, CPU float64, elementwise `E`/`Gc`, common `nu`/`l0`, shared Amor AT2 settings, and Dirichlet conditions. Centroid-based bulk assignments do not provide an independent interface law. |
| Schema-2 dynamic multi-material fracture | Unsupported | A common input layout and a single-material dynamic adapter do not provide dynamic heterogeneous execution. Do not switch the DCB input to `explicit`. |
| Diffuse-interface fracture | Beta | Weak-interface deflection and strong-interface penetration examples use spatial `E(x)`/`Gc(x)` fields with AT2 damage solves. Discrete cohesive and PF-CZM formulations are separate models. |
| Plane strain | Supported | Default 2D constitutive setting. |
| Plane stress | Beta | Legacy inputs use `material.plane_stress`; schema-2 inputs use the named material's `parameters.plane_stress`, with common settings on the multi-material route. Formal benchmark coverage is narrower than plane strain. |
| Spectral / Amor / isotropic energy splits | Supported, route-specific | Legacy inputs select `material.energy_split`. The documented schema-2 multi-material route requires common `parameters.energy_split: amor`. Plane-stress `spectral` is a reduced 2D in-plane projection; do not infer every split is supported by every adapter. |
| `spectral_stress` split | Experimental | Opt-in comparison pathway; do not present it as a generally supported formulation. |
| Monolithic `(u,d)` L-BFGS solve | Experimental | Research comparison only until the bound-constrained irreversibility algorithms are formalized. |
| Sparse quasi-static J2 elastoplasticity | Beta | Per-element state variables, return mapping, state commit/rollback, internal force, sparse solver selection, and plastic-work accounting are available. Large-mesh backend validation is incomplete. |
| Ductile PF-plasticity | Beta | Elastic tensile energy and accumulated plastic work can drive bounded AT2 damage in a staggered solve. Validation against established ductile-fracture benchmarks is incomplete. |
| Cohesive elements / discrete CZM | Beta | True-bilinear cohesive residual and tangent assembly, dissipated-energy history, optional normal-contact penalty, and single-block meshio cohesive-layer insertion are available. Validation for calibrated structural delamination is incomplete. |
| Coupled brittle PF + cohesive elements | Beta | Staggered AT2 matrix damage can be combined with zero-thickness cohesive-interface delamination. Calibrated structural validation remains incomplete. |
| PF-CZM | Beta | Wu PF-CZM is available through `pf_model: PFCZM` for nonlinear forward damage solves with tensile-strength-calibrated rational degradation and uniaxial tests. Structural crack-growth validation is incomplete, and PF-plasticity-cohesive coupling is not supported. |
| Coupled PF + plasticity + cohesive interfaces | Unsupported | This combined formulation is not available through the documented solver interface. |
| Learned damage proposal | Experimental | A user-supplied predictor may provide a projected initial guess; the classical damage solve remains authoritative. No trained model is distributed. |
| Audited learned damage replacement | Experimental | A predicted damage state may replace one damage solve only after bound, irreversibility, phase-field boundary-condition, and projected-residual checks. Rejection uses the classical fallback by default. |
| 3D fracture | Unsupported | The documented fracture element pathways are two-dimensional. |
| P2 / Q8 / Q9 element primitives | Scaffold | Shape functions, quadrature, and single-element stiffness tests exist for higher-order families; global solver dispatch is not supported. |
| Native Q4 isotropic mechanics + AT2 damage | Beta | Structured Q4 mesh helpers, native Q4 mesh input, 2x2-Gauss isotropic mechanics, SciPy/MUMPS sparse-direct stiffness assembly, and matrix-free Q4 AT2 damage are tested. Q4 PF-CZM, AT1, and plasticity are not supported. |

## Solvers and Backends

| Capability | Status | Public Statement |
|---|---|---|
| Explicit dynamics, Velocity Verlet | Supported | Principal documented pathway for dynamic impact and fracture simulations. |
| Staggered quasi-static/static solve | Supported | Main documented implicit brittle-fracture algorithm. It uses `jacobi` as the default damage preconditioner; matrix-free CG and sparse-direct mechanics solvers are available. |
| Rotation-free connectors in `quasi_static` | Beta | The staggered call forwards active connectors to the existing reduced mechanics solve. CPU float64 isotropic elastic tests check connector kinematics, reaction and moment balance. Nonlinear fracture trajectories require separate case-specific verification. |
| `quasi_static_legacy` secant path | Beta | Available for compatibility and selected MPC or frozen-secant calculations. |
| SciPy SuperLU sparse direct baseline | Supported | Portable sparse-direct baseline when SciPy is installed. |
| PETSc/MUMPS | Optional backend | When available, `backend='auto'` prioritises this backend for CPU sparse-direct mechanics. |
| cuDSS / nvmath | Optional backend | GPU sparse-direct solution requires a compatible nvmath/cuDSS installation and validation on the target hardware. |
| AMG / AmgX / GMG damage preconditioning | Experimental | Performance-oriented pathways for quasi-static fracture; use Jacobi unless the alternative preconditioner is itself part of the study. |
| Anderson acceleration | Beta | Available for staggered iterations; use only with benchmark-specific validation. |

## Inverse Workflows

| Capability | Status | Public Statement |
|---|---|---|
| Differentiable forward sensitivities | Beta | Supported tensor operations can participate in PyTorch autograd, but nonsmooth history, bounds, and active-set switches require case-specific interpretation. |
| Public inverse-analysis examples | Scaffold | `examples/inverse_problems_beta/` is a README-only landing zone until promoted inverse examples include configs, losses, retained lightweight outputs, and validation notes. |
| General-purpose inverse-calibration framework | Unsupported | The public release does not provide a turnkey inverse-problem framework for arbitrary observations, priors, or optimizers. |

Selected paper-specific inverse figures have separate
[retained-data reproduction commands](../paper-reproduction.md). Their
catalogue distinguishes replotting saved results from running the inverse
calculation. These utilities preserve the scope of the recorded studies while
the complete companion archive is prepared.

## Declarative YAML Workflows

| Capability | Status | Public Statement |
|---|---|---|
| YAML problem definition | Supported | Primary execution entry point (e.g., `python -m phast run config.yaml`). |
| YAML input validation | Supported within each adapter | `run CONFIG --validate-only` checks implemented schema/workflow constraints without solving. The multi-material route constructs the geometric mesh to check selections and assignments. It is not numerical or scientific validation. |
| `explain-config` review | Supported for the standard examples | Explains legacy and schema-2 inputs without solving. Explanation does not establish numerical convergence or validate a physical model. |
| `schema_version` | Supported, route-specific | Existing schema-1 benchmarks retain their layout. Standard schema-2 inputs use explicit regions/materials/assignments, analysis steps, and outputs; a common layout does not establish an execution adapter. |
| Small single-material standard exercises | Supported | `examples/standard_workflow/quasistatic_sent/config.yaml` and `examples/standard_workflow/dynamic_sent/config.yaml` execute through the existing quasi-static and explicit-dynamic runners. These short setup exercises are not crack-growth benchmarks. |
| Resolved run record | Supported, route-specific content | Runs write provenance/lockfile records; recorded fields depend on the adapter. A configuration hash or later public-availability fingerprint does not establish missing historical generating-source identity. |
| Built-in geometry generators | Supported | Available through `geometry.type` and `geometry.parameters`. |
| External meshes | Beta | Supported via `geometry.mesh_path`; node-set capability relies on the specific mesh format standard. |
| Declarative primitive geometry DSL | Beta | Parsed by selected benchmark configs; broader geometry coverage remains under evaluation. |
| Config inheritance / sweeps | Unsupported | Parameter sweeps currently require external scripting or duplicated declarative files. |
| JSON Schema export / IDE autocomplete | Supported | `python -m phast schema` exports the schema used for editor assistance and external validation. |

New YAML/README/tutorial authors must follow the
{download}`configuration style guide <../../CONFIGURATION_STYLE.md>` and
[standard simulation tutorial](../tutorial/07_standard_simulation_workflow.md).
Legacy `precheck` is not a universal schema-2 command. Quasi-static analysis
omits inertia; dynamics includes density and physical time, and explicit
integration has an additional CFL restriction. A smaller step is not validation.

Retained DCB evidence shows qualitative layered-material interaction: no
seed-connected nodes with `d >= 0.80` enter the disk, and both the tough
disk and no-contrast disk control fork early. Neither completed bypass nor
experimental/interface-law validation is demonstrated. Independent review
of the fixed stored fields found numerical acceptability, not general model
validation; input guards and integrated command checks remain separate.

Frozen course assets and separate 3D research are not automatically migrated
into the public schema; their adapters and evidence must exist first.

## Outputs and Validation Artifacts

| Capability | Status | Public Statement |
|---|---|---|
| HDF5 snapshots | Supported | Default single-file format for new trajectory workflows. |
| Zarr trajectory stores | Supported | Explicit opt-in for directory-based trajectory storage; existing stores remain readable. |
| VTU / PyVista-style visualization | Beta | Available via declarative output settings; format fidelity relies on optional visualization dependencies. |
| Automated visualization generation | Beta | Generates documentation-ready GIFs and plots; artifacts must be manually verified before academic publication. |
| Reaction-force logging | Supported | Available for load-displacement verification through `output.reaction_node_set` and `output.reaction_component`. |
| Benchmark comparison scripts | Beta | Selected examples include `compare.py` routines that evaluate stated tolerances against published reference results. |
| CPU execution and optional accelerator use | Supported | Core CPU execution is exercised by public checks. Practical device choice for larger simulations depends on mesh size, precision, backend availability, and the documented workflow. |
