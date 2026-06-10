# Customer Tutorial Suite

This page is the customer-facing map of runnable tutorials and showcase
workflows. It intentionally mirrors the
[capability matrix](user_guide/capability_matrix.md): tutorials marked
production or beta can be demonstrated today, while scaffold and unsupported
rows are roadmap items and must not be sold as complete solver capability.

## Quick Selection

| Workflow | Readiness | Entry point | Expected outputs |
|---|---|---|---|
| Fresh install validation check | Production | `python -m phast run configs/benchmarks/dynamic/B3_dynamic_sent.yaml --validate-only` | Config validation report |
| Dynamic brittle fracture | Production | `python -m phast run configs/benchmarks/dynamic/B3_dynamic_sent.yaml --device cpu --num_steps 20 --plots` | Damage snapshots, telemetry, config copy |
| Microstructured fracture | Production | `python -m phast run configs/benchmarks/dynamic/B6_perforated_30holes.yaml --device cpu --num_steps 20 --plots` | Crack-hole damage snapshots and run metadata |
| Solid mechanics: linear elasticity | Production | `python examples/solid_mechanics/linear_plate.py` | Tip displacement and autograd sensitivity |
| Solid mechanics: nonlinear elasticity | Beta | `python examples/solid_mechanics/neohookean_plate.py` | Newton convergence table and sensitivity |
| Solid mechanics: J2 plasticity material point | Beta | `python examples/solid_mechanics/j2_plasticity_bar.py` | von-Mises/yield consistency table |
| Sparse quasi-static ductile PF-plasticity validation | Beta | `python examples/plasticity_interface/run_ductile_pf_plasticity_validation.py --backend scipy` | J2 stress-strain plot, Newton residual telemetry, requested/resolved backend status, ductile driving-force plot, elastic/plastic/fracture energy ledger, bounded AT2 damage residual, CSVs, visual manifest |
| Ductile PF-plasticity sensitivity study | Beta | `python examples/plasticity_interface/run_ductile_pf_sensitivity_study.py` | Elastic-driving reference, `l0` sensitivity table, plastic-work driving lift plot, damage sensitivity plot, retained child bundles |
| Solid interface fracture validation | Beta | `python examples/plasticity_interface/run_solid_interface_fracture_examples.py` | Weak-deflection and strong-penetration visual bundles |
| Solver-driven diffuse interface fracture validation | Beta | `python examples/plasticity_interface/run_solver_driven_interface_fracture_validation.py` | Weak-interface deflection and strong-interface penetration from solved AT2 damage fields, material/setup/damage plots, animation, CSV telemetry, provenance |
| Mixed-mode cohesive interface validation | Beta | `python examples/plasticity_interface/run_cohesive_mixed_mode_benchmark.py` | Normal/shear traction tables, cohesive damage history, tangent finite-difference check, visual manifest |
| Cohesive contact-compression validation | Beta | `python examples/plasticity_interface/run_cohesive_contact_compression_benchmark.py` | Normal contact-traction table, zero-damage compression check, tangent finite-difference check, visual manifest |
| Cohesive delamination patch validation | Beta | `python examples/plasticity_interface/run_cohesive_delamination_patch_benchmark.py` | Four-segment mixed-mode cohesive patch, resultant checks, damage-front plot, tangent finite-difference check, visual manifest |
| Structural DCB-style cohesive validation | Beta | `python examples/plasticity_interface/run_structural_dcb_cohesive_benchmark.py` | Precracked two-arm cohesive structure, load-displacement plot, damage-front plot, energy plot, mesh artifacts, visual manifest |
| Coupled PF matrix damage + cohesive interface validation | Beta | `python examples/plasticity_interface/run_coupled_pf_cohesive_benchmark.py` | Notched matrix AT2 damage plus cohesive-interface delamination in one staggered run, damage/front plots, energy split, convergence, animation, CSV telemetry, visual manifest |
| Mixed-precision CG numerics | Production | `python examples/solid_mechanics/mixed_precision_cg_demo.py` | `mixed_precision_cg_demo.png` |
| Generalized-alpha dynamics | Beta | `python examples/solid_mechanics/dynamic_oscillator_genalpha.py` | `dynamic_oscillator_genalpha.png` |

## Standard Tutorial Contract

Every customer-facing tutorial should provide:

| Requirement | Rule |
|---|---|
| One command | The command should run from the repository root. |
| Runtime expectation | State whether it is a CPU check, laptop-scale, or HPC-scale. |
| Output files | List plots, CSVs, Zarr/H5 stores, and metadata files. |
| Physics scope | Say exactly which equations, material model, and solver path are active. |
| Validation reference | Link to a benchmark, analytical check, or explicit validation invariant. |
| Failure modes | Mention common setup or convergence failures and how they appear. |

## What Not To Claim Yet

The following are active product-hardening tracks, not customer-ready
tutorials:

| Workflow | Current state |
|---|---|
| Large-scale production J2 with closed-form tangent and PETSc/MUMPS promotion | Sparse `QuasiStaticSolver` J2 dispatch plus a safe closed-form tangent slice exist for validation; backend promotion evidence and large-mesh benchmarks remain hardening work. |
| Production coupled PF + plasticity | Guarded quasi-static T3 J2+AT2 staggered path and validation example exist; benchmark-matched ductile fracture remains pending. |
| Discrete cohesive elements | Stateful cohesive residual/tangent operator, sparse solver validation, mode-I displacement-jump benchmark, mixed-mode tangent benchmark, normal contact-compression benchmark, multi-element delamination patch benchmark, and DCB-style structural cohesive validation exist; ASTM-calibrated structural delamination remains pending. |
| Coupled brittle PF + cohesive elements | Staggered AT2 matrix damage plus cohesive-interface delamination smoke exists; calibrated PF-CZM and structural benchmark studies remain pending. |
| PF-CZM | Forward nonlinear damage-law smoke exists for `pf_model: PFCZM`; structural crack-growth and mixed-mode calibration remain pending. |
| Native Q4 global fracture solves | Native Q4 mesh admission, isotropic mechanics/scalar operator dispatch, sparse-direct Q4 mechanics, and matrix-free Q4 AT2 damage with Gauss-point history exist; Q4 PF-CZM, AT1, plasticity, cohesive-coupled damage, direct damage assembly, and differentiable damage adjoints remain gated. |
| Abaqus/COMSOL-equivalent coupled ductile cohesive fracture | Not defensible until the above tracks pass benchmark evidence. |

## Retained Plasticity/Interface Evidence

The June 2026 plasticity/interface validation stack is retained as internal
release evidence and is not shipped in the public repository. Public users
should rely on the runnable example commands above and their generated local
manifests. Internal HPC roots and review bundles are not part of the public
release.

## Showcase Assets

The README montage is curated under
[`docs/readme_showcase/`](readme_showcase/README.md). Those images are
lightweight documentation assets copied from existing outputs; raw HPC result
folders and large media packs should stay out of git.
