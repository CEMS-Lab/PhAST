# Capability Matrix

This capability matrix defines the documented evidence boundaries of PhAST. It
separates supported and benchmarked pathways from beta, experimental,
optional-backend, scaffold, and unsupported features.

For the current plasticity, cohesive interface, and PF-CZM technical-preview boundaries, see the [plasticity/interface beta workflow](../supported_workflows/plasticity_interface_beta.md).

## Status Definitions

| Status | Meaning |
|---|---|
| **Supported** | Implemented in the public execution pathway and represented by checked-in tests or a reproducible example. The cited evidence remains specific to the documented problem class. |
| **Beta** | Mathematically implemented and functional, but robustness or full-domain validation is still under review. |
| **Experimental** | Active research pathway with insufficient evidence for general research claims. |
| **Optional backend** | Code path exists but requires external solver libraries or specific HPC hardware environments. |
| **Scaffold** | Foundational data structures or kernels exist, but the feature is not yet coupled into the global solve. |
| **Unsupported** | Feature is mathematically or computationally unsupported. |

## Simulation Physics

| Capability | Status | Public Statement |
|---|---|---|
| Small-strain 2D linear elasticity | Supported | Exercised through the core mechanics kernels and documented static and quasi-static examples. |
| Brittle phase-field fracture, AT2 | Supported | Available for explicit dynamics and staggered quasi-static/static solves within the documented examples. |
| Brittle phase-field fracture, AT1 | Beta | Supported via projected damage solves and AT1 threshold fields; benchmark coverage is expanding. |
| Heterogeneous elastic fields `E(x)` | Supported | Per-element fields support structural inclusions and weak/strong bands in documented forward studies. |
| Heterogeneous fracture fields `Gc(x)` | Supported | Per-element fields support weak zones and microstructure-style morphology studies in forward simulations. |
| Diffuse interface fracture | Beta | Solver-driven weak-interface deflection and strong-interface penetration examples use spatial `E(x)`/`Gc(x)` fields plus AT2 damage solves; discrete cohesive and PF-CZM calibration remain separate. |
| Plane strain | Supported | Default 2D constitutive setting. |
| Plane stress | Beta | Available via `material.plane_stress`; formal benchmark coverage is narrower than plane strain. |
| Spectral / Amor / isotropic energy splits | Supported | Available through `material.energy_split` for brittle phase-field workflows. Plane-stress `spectral` is a reduced 2D in-plane projection; prefer plane-strain `spectral` or plane-stress `amor` where benchmark evidence is required. |
| `spectral_stress` split | Experimental | Opt-in comparison pathway; do not present it as a generally supported formulation. |
| Monolithic `(u,d)` L-BFGS solve | Experimental | Research comparison only until the bound-constrained irreversibility algorithms are formalized. |
| Sparse quasi-static J2 elastoplasticity | Beta | Per-element state, return mapping, commit/rollback, internal force, sparse dispatch, and plastic-work accounting are available; large-mesh backend promotion remains gated. |
| Ductile PF-plasticity | Beta | Elastic tensile energy plus accumulated plastic-work coupling is implemented, featuring bounded AT2 damage, separated energy ledgers, and guarded staggered support. Benchmark-matched ductile fracture remains gated. |
| Cohesive elements / discrete CZM | Beta | Stateful true-bilinear cohesive residual/tangent assembly, dissipated-energy history, optional normal-contact penalty, and single-block meshio cohesive-layer insertion are available. ASTM-calibrated structural delamination remains gated. |
| Coupled brittle PF + cohesive elements | Beta | Staggered AT2 matrix damage plus zero-thickness cohesive-interface delamination exists with matrix notch damage, convergence telemetry, and visual manifests. Calibrated PF-CZM structural validation studies remain gated. |
| PF-CZM | Beta | Wu PF-CZM is available via `pf_model: PFCZM` for forward nonlinear damage solves with tensile-strength-calibrated rational degradation and uniaxial validation. Structural crack-growth and PF-plasticity-cohesive coupling remain gated. |
| Coupled PF + plasticity + cohesive interfaces | Unsupported | Not currently supported within the public workflow surface. |
| 3D fracture | Unsupported | The documented fracture element pathways are two-dimensional. |
| P2 / Q8 / Q9 element primitives | Scaffold | Shape functions, quadrature, and single-element stiffness tests exist for higher-order families; global solver dispatch is not supported. |
| Native Q4 isotropic mechanics + AT2 damage | Beta | Structured Q4 mesh helpers, native Q4 mesh admission, 2x2-Gauss isotropic mechanics, SciPy/MUMPS sparse-direct stiffness assembly, and matrix-free Q4 AT2 damage are tested. Q4 PF-CZM, AT1, and plasticity remain gated. |

## Solvers and Backends

| Capability | Status | Public Statement |
|---|---|---|
| Explicit dynamics, Velocity Verlet | Supported | Principal documented pathway for dynamic impact and fracture simulations. |
| Staggered quasi-static/static solve | Supported | Principal documented implicit brittle-fracture pathway, using `jacobi` as the conservative damage preconditioner. Matrix-free CG and sparse-direct mechanics backends are available. |
| `quasi_static_legacy` secant path | Beta | Retained strictly for compatibility and selected MPC/frozen-secant workflows. |
| SciPy SuperLU sparse direct baseline | Supported | Portable sparse-direct baseline when SciPy is installed. |
| PETSc/MUMPS | Optional backend | Runtime-guarded. When available, `backend='auto'` prioritizes this for CPU sparse-direct mechanics. |
| cuDSS / nvmath | Optional backend | Runtime-guarded. GPU sparse-direct path requires current nvmath/cuDSS validation on the target hardware stack. |
| AMG / AmgX / GMG damage preconditioning | Experimental | Performance-oriented pathways for quasi-static fracture; use Jacobi unless the alternative preconditioner is itself part of the study. |
| Anderson acceleration | Beta | Available for staggered iterations; use only with benchmark-specific validation. |

## Inverse Workflows

| Capability | Status | Public Statement |
|---|---|---|
| Differentiable forward sensitivities | Beta | Supported tensor operations can participate in PyTorch autograd, but nonsmooth history, bounds, and active-set switches require case-specific interpretation. |
| Public inverse-analysis examples | Scaffold | `examples/inverse_problems_beta/` is a README-only landing zone until promoted inverse examples include configs, losses, retained lightweight outputs, and validation notes. |
| General-purpose inverse-calibration framework | Unsupported | The public release does not provide a turnkey inverse-problem framework for arbitrary observations, priors, or optimizers. |

## Declarative YAML Workflows

| Capability | Status | Public Statement |
|---|---|---|
| YAML problem definition | Supported | Primary execution entry point (e.g., `python -m phast run config.yaml`). |
| YAML schema validation | Supported | `--validate-only` enforces schema compliance and reports line-numbered semantic errors. |
| `explain-config` review | Supported | Prints selected physics, solver paths, provenance, and setup warnings without initiating meshing or solving. |
| `schema_version` | Supported | Current runnable configs declare `schema_version: 1`; legacy files trigger review warnings. |
| Resolved run lockfile | Supported | Execution generates `run_lockfile.json` capturing the config hash, resolved objects, CLI arguments, and Git state. |
| Built-in geometry generators | Supported | Available through `geometry.type` and `geometry.parameters`. |
| External meshes | Beta | Supported via `geometry.mesh_path`; node-set capability relies on the specific mesh format standard. |
| Declarative primitive geometry DSL | Beta | Parsed by selected benchmark configs; broader geometry coverage remains under evaluation. |
| Config inheritance / sweeps | Unsupported | Parameter sweeps currently require external scripting or duplicated declarative files. |
| JSON Schema export / IDE autocomplete | Supported | `python -m phast schema` exports the schema used for editor assistance and external validation. |

## Outputs and Validation Artifacts

| Capability | Status | Public Statement |
|---|---|---|
| Zarr trajectory stores | Supported | Principal high-fidelity format for trajectory workflows; chunked and appendable. |
| HDF5 snapshots | Legacy | Maintained for backward compatibility and benchmark extraction; deprecated for new large-scale training corpora. |
| VTU / PyVista-style visualization | Beta | Available via declarative output settings; format fidelity relies on optional visualization dependencies. |
| Automated visualization generation | Beta | Generates documentation-ready GIFs and plots; artifacts must be manually verified before academic publication. |
| Reaction-force logging | Supported | Available for load-displacement verification through `output.reaction_node_set` and `output.reaction_component`. |
| Strict-parity benchmark scripts | Beta | `compare.py` routines available for selected examples to verify tolerances against reference literature. |
| CPU execution and optional accelerator use | Supported | Core CPU execution is exercised by public checks. Practical device choice for larger simulations depends on mesh size, precision, backend availability, and the documented workflow. |
