# Capability Matrix

This capability matrix formally defines the validated boundaries of the PhAST framework. It strictly separates production paths from beta, experimental, optional-backend, and unsupported features to ensure all solver claims remain reproducible and academically rigorous.

For the current plasticity, cohesive interface, and PF-CZM technical-preview boundaries, see the [plasticity/interface beta workflow](../supported_workflows/plasticity_interface_beta.md).

## Status Definitions

| Status | Meaning |
|---|---|
| **Production** | Fully validated through extensive testing and represented by at least one reproducible benchmark. |
| **Beta** | Mathematically implemented and functional, but robustness or full-domain validation is still under review. |
| **Experimental** | Active research path. Designed for internal comparison and development, not for reproducible claims. |
| **Optional backend** | Code path exists but requires external solver libraries or specific HPC hardware environments. |
| **Scaffold** | Foundational data structures or kernels exist, but the feature is not yet coupled into the global solve. |
| **Unsupported** | Feature is mathematically or computationally unsupported. |

## Simulation Physics

| Capability | Status | Public Statement |
|---|---|---|
| Small-strain 2D linear elasticity | Production | Supported through the core mechanics kernels and validated via quasi-static and static examples. |
| Brittle phase-field fracture, AT2 | Production | Supported for explicit dynamics and staggered quasi-static/static solves. |
| Brittle phase-field fracture, AT1 | Beta | Supported via projected damage solves and AT1 threshold fields; benchmark coverage is expanding. |
| Heterogeneous elastic fields `E(x)` | Production | Per-element fields support structural inclusions, weak/strong bands, and inverse calibration studies. |
| Heterogeneous fracture fields `Gc(x)` | Production | Per-element fields support weak zones, spatial recovery, and microstructure-style morphology studies. |
| Diffuse interface fracture | Beta | Solver-driven weak-interface deflection and strong-interface penetration examples use spatial `E(x)`/`Gc(x)` fields plus AT2 damage solves; discrete cohesive and PF-CZM calibration remain separate. |
| Plane strain | Production | Default 2D constitutive setting. |
| Plane stress | Beta | Available via `material.plane_stress`; formal benchmark coverage is narrower than plane strain. |
| Spectral / Amor / isotropic energy splits | Production | Available through `material.energy_split` for brittle phase-field workflows. Plane-stress `spectral` is a reduced 2D in-plane projection; prefer plane-strain `spectral` or plane-stress `amor` for mature validated paths. |
| `spectral_stress` split | Experimental | Opt-in COMSOL-parity research path; do not use as the default production claim. |
| Monolithic `(u,d)` L-BFGS solve | Experimental | Research comparison only until the bound-constrained irreversibility algorithms are formalized. |
| Sparse quasi-static J2 elastoplasticity | Beta | Per-element state, return mapping, commit/rollback, internal force, sparse dispatch, and plastic-work accounting are available; large-mesh backend promotion remains gated. |
| Ductile PF-plasticity | Beta | Elastic tensile energy plus accumulated plastic-work coupling is implemented, featuring bounded AT2 damage, separated energy ledgers, and guarded staggered support. Benchmark-matched ductile fracture remains gated. |
| Cohesive elements / discrete CZM | Beta | Stateful true-bilinear cohesive residual/tangent assembly, dissipated-energy history, optional normal-contact penalty, and single-block meshio cohesive-layer insertion are available. ASTM-calibrated structural delamination remains gated. |
| Coupled brittle PF + cohesive elements | Beta | Staggered AT2 matrix damage plus zero-thickness cohesive-interface delamination exists with matrix notch damage, convergence telemetry, and visual manifests. Calibrated PF-CZM structural validation studies remain gated. |
| PF-CZM | Beta | Wu PF-CZM is available via `pf_model: PFCZM` for forward nonlinear damage solves with tensile-strength-calibrated rational degradation and uniaxial validation. Structural crack-growth and PF-plasticity-cohesive coupling remain gated. |
| Coupled PF + plasticity + cohesive interfaces | Unsupported | Not currently supported within the public workflow surface. |
| 3D fracture | Unsupported | Current production element primitives are strictly 2D triangles. |
| P2 / Q8 / Q9 element primitives | Scaffold | Shape functions, quadrature, and single-element stiffness tests exist for higher-order families; global production solver dispatch remains gated. |
| Native Q4 isotropic mechanics + AT2 damage | Beta | Structured Q4 mesh helpers, native Q4 mesh admission, 2x2-Gauss isotropic mechanics, SciPy/MUMPS sparse-direct stiffness assembly, and matrix-free Q4 AT2 damage are tested. Q4 PF-CZM, AT1, and plasticity remain gated. |

## Solvers and Backends

| Capability | Status | Public Statement |
|---|---|---|
| Explicit dynamics, Velocity Verlet | Production | Primary validated path for dynamic impact and fracture simulations. |
| Staggered quasi-static/static solve | Production | Primary implicit brittle-fracture path, utilizing `jacobi` as the conservative damage preconditioner. Matrix-free CG and sparse-direct mechanics backends are available. |
| `quasi_static_legacy` secant path | Beta | Retained strictly for compatibility and selected MPC/frozen-secant workflows. |
| SciPy SuperLU sparse direct baseline | Production | Robust, always-available sparse direct baseline provided SciPy is installed. |
| PETSc/MUMPS | Optional backend | Runtime-guarded. When available, `backend='auto'` prioritizes this for CPU sparse-direct mechanics. |
| cuDSS / nvmath | Optional backend | Runtime-guarded. GPU sparse-direct path requires current nvmath/cuDSS validation on the target hardware stack. |
| AMG / AmgX / GMG damage preconditioning | Experimental | Performance optimization paths for quasi-static fracture; production runs should default to Jacobi unless validating the preconditioner itself. |
| Anderson acceleration | Beta | Available for staggered iterations; use only with benchmark-specific validation. |

## Declarative YAML Workflows

| Capability | Status | Public Statement |
|---|---|---|
| YAML problem definition | Production | Primary execution entry point (e.g., `python -m phast run config.yaml`). |
| YAML schema validation | Production | `--validate-only` strictly enforces schema compliance and reports line-numbered semantic errors. |
| `explain-config` dry-run review | Production | Prints selected physics, solver paths, provenance, and setup warnings without initiating meshing or solving. |
| `schema_version` | Production | Modern configs declare `schema_version: 1`; legacy files trigger review warnings. |
| Resolved run lockfile | Production | Execution generates `run_lockfile.json` capturing the config hash, resolved objects, CLI arguments, and exact Git state. |
| Built-in geometry generators | Production | Supported natively via `geometry.type` and `geometry.parameters`. |
| External meshes | Beta | Supported via `geometry.mesh_path`; node-set capability relies on the specific mesh format standard. |
| Declarative primitive geometry DSL | Beta | Parsed successfully by benchmark configs; continued validation required before widespread production usage. |
| Config inheritance / sweeps | Unsupported | Parameter sweeps currently require external scripting or duplicated declarative files. |
| JSON Schema export / IDE autocomplete | Production | `python -m phast schema` exports the formal schema derived from the internal model, ensuring IDE autocomplete support. |

## Outputs and Validation Artifacts

| Capability | Status | Public Statement |
|---|---|---|
| Zarr trajectory stores | Production | Primary high-fidelity format for trajectory and dataset workflows; chunked, appendable, and designed for parallel I/O. |
| HDF5 snapshots | Legacy | Maintained for backward compatibility and benchmark extraction; deprecated for new large-scale training corpora. |
| VTU / PyVista-style visualization | Beta | Available via declarative output settings; format fidelity relies on optional visualization dependencies. |
| Automated visualization generation | Beta | Generates documentation-ready GIFs and plots; artifacts must be manually verified before academic publication. |
| Reaction-force logging | Production | Critical for load-displacement verification; triggered via `output.reaction_node_set` and `output.reaction_component`. |
| Strict-parity benchmark scripts | Beta | `compare.py` routines available for selected examples to verify tolerances against reference literature. |
| CPU / HPC portability | Production | Core CPU execution is validated for small/quasi-static checks; large-scale dynamic runs are strictly GPU/HPC-oriented. |
