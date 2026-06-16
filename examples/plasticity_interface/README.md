# Plasticity and Diffuse-Interface Validation Examples

These examples package the current customer-safe boundary for plasticity and
interface/interphase fracture work.

## Current Capability Boundary

| Example | What it validates | Readiness |
| --- | --- | --- |
| `run_j2_validation.py` | Standalone J2/von-Mises material-point return mapping with linear isotropic hardening | Runnable kernel validation |
| `run_ductile_pf_plasticity_validation.py` | Sparse quasi-static J2 mechanics plus bounded AT2 phase-field damage solved on a ductile plastic-work history | Gate 1 solver validation plus Gate 2 operator-coupled damage validation; backend-selectable for promotion runs; guarded staggered T3 J2+AT2 path exists, benchmark matching pending |
| `run_ductile_pf_sensitivity_study.py` | Elastic-driving reference plus ductile plastic-work length-scale sensitivity table and plots | Customer-facing ductile validation study; not a SENT/TPB calibration |
| `run_diffuse_interphase_validation.py` | Diffuse interface/interphase fields in a brittle phase-field setting using spatial `E(x)` and `Gc(x)` | Runnable field/path validation |
| `run_solid_interface_fracture_examples.py` | Two deterministic crack-impinging-on-interface path-energy benchmarks: weak-interface deflection and strong-interface penetration | Diffuse solid-interface field/path screening examples; not solved crack-evolution runs |
| `run_solver_driven_interface_fracture_validation.py` | Weak-interface deflection and strong-interface penetration classified from solved AT2 phase-field damage with spatial `E(x)`/`Gc(x)` fields | Solver-driven diffuse interface fracture smoke; not a cohesive-zone or ASTM calibration |
| `run_cohesive_displacement_jump_benchmark.py` | Zero-thickness cohesive displacement-jump response coupled through `QuasiStaticSolver(cohesive_operator=...)` | Production-smoke cohesive operator benchmark |
| `run_cohesive_mixed_mode_benchmark.py` | Zero-thickness mixed-mode cohesive response with residual/tangent finite-difference evidence | Production-smoke mixed-mode cohesive operator benchmark |
| `run_cohesive_contact_compression_benchmark.py` | Zero-thickness normal-compression contact penalty response with no damage growth | Production-smoke cohesive contact benchmark |
| `run_cohesive_delamination_patch_benchmark.py` | Four-segment zero-thickness mixed-mode cohesive patch with localized damage/front metrics and closed-form resultant checks | Production-smoke cohesive delamination patch benchmark |
| `run_structural_dcb_cohesive_benchmark.py` | DCB-style Mode-I structural cohesive delamination with a precrack, free bulk DOFs, post-peak softening, damage-front metrics, and energy plots | Structural cohesive validation smoke; not ASTM D5528 data reduction |
| `run_coupled_pf_cohesive_benchmark.py` | AT2 phase-field matrix damage around a notch plus zero-thickness cohesive delamination on an embedded interface in one staggered run | Coupled brittle PF+CE validation smoke; not calibrated PF-CZM |
| `run_pfczm_uniaxial_strength_validation.py` | Wu PF-CZM cohesive phase-field damage-law strength calibration and length-scale sweep with nonlinear bounded damage solve | Forward PF-CZM validation smoke; not a full structural crack-growth or PF-plasticity-cohesive benchmark |

These examples do **not** claim a benchmark-matched fully coupled staggered
ductile phase-field-plasticity solver or ASTM-calibrated cohesive
delamination workflow. Those remain tracked under GitHub issues #553 and #554.

## Run

From the repository root:

The canonical reproducibility contract for this validation suite is:

```text
configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml
```

That YAML lists every validation runner, its exact launcher command, required
CSV/JSON/mesh/visual artifacts, and the claim boundary for customer-facing
use. Each promoted run should also retain the generated `config.yaml`,
`run_manifest.json` when available, and `visual_manifest.json` for plots or
animations according to `docs/visualisation_requirements.md`.

The first promoted #708 validation slice is available through the curated YAML
dispatcher:

| Validation id | Role | Retained result | Fluent setup |
| --- | --- | --- | --- |
| `j2_validation` | Standard J2 return-map validation example for the promoted plasticity slice. | `examples/plasticity_interface/results/j2_validation` | `examples/plasticity_interface/fluent_setups/j2_validation.py` |
| `structural_dcb_cohesive` | DCB-style structural cohesive benchmark for the promoted cohesive/interface slice. | `examples/plasticity_interface/results/structural_dcb_cohesive` | `examples/plasticity_interface/fluent_setups/structural_dcb_cohesive.py` |
| `structural_dcb_refinement` | Lightweight DCB cohesive mesh/load-step refinement trend for the promoted cohesive/interface slice. | `examples/plasticity_interface/results/structural_dcb_refinement` | Script-contract runner; flat fluent setup pending tutorial promotion. |
| `pfczm_uniaxial_strength` | One-dimensional PF-CZM strength calibration smoke for the promoted PF-CZM slice. | `examples/plasticity_interface/results/pfczm_uniaxial_strength` | `examples/plasticity_interface/fluent_setups/pfczm_uniaxial_strength.py` |

```bash
python -m phast run configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml --validation-id j2_validation
python -m phast run configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml --validation-id structural_dcb_cohesive
python -m phast run configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml --validation-id structural_dcb_refinement
python -m phast run configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml --validation-id pfczm_uniaxial_strength
```

To regenerate the retained result folders in place:

```bash
python -m phast run configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml \
  --validation-id j2_validation \
  --output_dir examples/plasticity_interface/results/j2_validation

python -m phast run configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml \
  --validation-id structural_dcb_cohesive \
  --output_dir examples/plasticity_interface/results/structural_dcb_cohesive

python -m phast run configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml \
  --validation-id structural_dcb_refinement \
  --output_dir examples/plasticity_interface/results/structural_dcb_refinement

python -m phast run configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml \
  --validation-id pfczm_uniaxial_strength \
  --output_dir examples/plasticity_interface/results/pfczm_uniaxial_strength
```

The retained #708 result index is
`examples/plasticity_interface/results/issue_708_promoted_results.yaml`. Each
retained result folder includes the YAML-resolved `config.yaml`, provenance
JSON, `run_manifest.json`, `visual_manifest.json`, CSV histories, setup and
final-state PNGs, a non-empty GIF animation, and `training_data.zarr` for
read-only `phast.Result` field inspection:

```python
import phast

result = phast.load_result("examples/plasticity_interface/results/structural_dcb_cohesive")
print(result.field_names())
print(result.history_names())
print(result.visuals())
```

```bash
python examples/plasticity_interface/run_j2_validation.py \
  --output-dir outputs/plasticity_interface/j2_validation

python examples/plasticity_interface/run_ductile_pf_plasticity_validation.py \
  --output-dir outputs/plasticity_interface/ductile_pf_plasticity \
  --backend scipy

python examples/plasticity_interface/run_ductile_pf_sensitivity_study.py \
  --output-dir outputs/plasticity_interface/ductile_pf_sensitivity

python examples/plasticity_interface/run_diffuse_interphase_validation.py \
  --output-dir outputs/plasticity_interface/diffuse_interphase

python examples/plasticity_interface/run_solid_interface_fracture_examples.py \
  --output-dir outputs/plasticity_interface/solid_interface_fracture

python examples/plasticity_interface/run_solver_driven_interface_fracture_validation.py \
  --output-dir outputs/plasticity_interface/solver_driven_interface_fracture

python examples/plasticity_interface/run_cohesive_displacement_jump_benchmark.py \
  --output-dir outputs/plasticity_interface/cohesive_displacement_jump

python examples/plasticity_interface/run_cohesive_mixed_mode_benchmark.py \
  --output-dir outputs/plasticity_interface/cohesive_mixed_mode

python examples/plasticity_interface/run_cohesive_contact_compression_benchmark.py \
  --output-dir outputs/plasticity_interface/cohesive_contact_compression

python examples/plasticity_interface/run_cohesive_delamination_patch_benchmark.py \
  --output-dir outputs/plasticity_interface/cohesive_delamination_patch

python examples/plasticity_interface/run_structural_dcb_cohesive_benchmark.py \
  --output-dir outputs/plasticity_interface/structural_dcb_cohesive

python examples/plasticity_interface/run_structural_dcb_refinement_study.py \
  --output-dir outputs/plasticity_interface/structural_dcb_refinement

python examples/plasticity_interface/run_coupled_pf_cohesive_benchmark.py \
  --output-dir outputs/plasticity_interface/coupled_pf_cohesive

python examples/plasticity_interface/run_pfczm_uniaxial_strength_validation.py \
  --output-dir outputs/plasticity_interface/pfczm_uniaxial_strength
```

The compact script-contract examples write:

- `summary.json` with validation metrics and memory use.
- One CSV table with the numerical evidence.
- PNG figures suitable for a customer-facing validation note.

The solid-interface benchmark runner writes the fuller standard-output bundle
used for customer review of the diffuse-field/path-energy validation boundary:

- `config.yaml`, `run_lockfile.json`, `run_metadata.json`, `run_manifest.json`,
  `run.log`, `mesh.geo`, and `mesh.msh`.
- `initial_conditions.png`, `material_fields.png`, `damage_final.png`,
  `crack_path.png`, `energy.png`, `load_displacement.png`,
  `staggered_convergence.png`, `compare.png`, `compare_report.txt`, and
  `damage_evolution.mp4` when MP4 encoding is available, otherwise a
  non-empty `damage_evolution.gif` fallback recorded in `visual_manifest.json`.
- `results.csv`, `history.csv`, `crack_tip.csv`, `energy.csv`,
  `timing_per_step.csv`, and `solver_telemetry.csv`.

These two solid-interface cases are deterministic energetic path-screening
examples over spatial `E(x)` and `Gc(x)` fields. They are useful for
communicating weak-interface deflection versus strong-interface penetration,
but they are not time/load-stepped phase-field crack-evolution solves.

The solver-driven interface fracture runner uses the same weak/strong
geometry family, computes a mechanics-derived tensile driving field, and calls
`PhaseFieldDamageSolver.solve(..., Gc_field=...)` for load-stepped AT2 damage
updates. It writes `setup.png`, `material_fields.png`, `damage_final.png`,
`load_displacement.png`, `energy_split.png`, `convergence.png`,
`damage_evolution.mp4` or GIF fallback, standard CSV telemetry, mesh artifacts,
and provenance files. The smoke gate classifies weak-interface deflection and
strong-interface penetration from the solved damage field. The claim remains a
diffuse interface/interphase validation, not a zero-thickness cohesive element
or PF-CZM structural crack-growth benchmark.

The native Q4 sparse AT2 smoke runner exercises the guarded Q4 production slice
without converting cells to T3:

```bash
python examples/plasticity_interface/run_q4_sparse_at2_smoke.py \
  --output-dir outputs/plasticity_interface/q4_sparse_at2_smoke \
  --nx 32 --ny 16 --n-steps 5 --backend scipy
```

On an HPC environment with PETSc/MUMPS available, use `--backend mumps` to
record MUMPS backend evidence in `backend_evidence.csv` and
`run_metadata.json`. The bundle includes `results.csv`,
`solver_telemetry.csv`, `damage_final.png`, `mesh_deformed.png`,
`load_displacement.png`, `visual_manifest.json`, and standard run provenance.
The Q4 claim is limited to isotropic sparse mechanics plus matrix-free AT2
damage with Gauss-point history; Q4 PF-CZM, AT1, plasticity, cohesive-coupled
damage, direct damage assembly, and differentiable damage adjoints remain
gated.

The cohesive displacement-jump benchmark writes a compact standard bundle:

- `summary.json`, `config.yaml`, `run_lockfile.json`, `run_metadata.json`,
  `run_manifest.json`, and `run.log`.
- `cohesive_response.csv` with load-step opening, solver convergence,
  committed damage, traction, scalar dissipated energy, and closed-form
  bilinear-law error checks.
- `cohesive_response.png`, `cohesive_mesh_and_bc.png`, and
  `visual_manifest.json`.

For T3 or Q4 array meshes that need a zero-thickness cohesive layer inserted,
use `phast.cohesive_elements.insert_cohesive_layer_with_metadata`
or `insert_cohesive_layer_meshio` for a single-block T3 or Q4 `meshio.Mesh`
with a named line-cell set marking the interface. These helpers return doubled nodes,
updated bulk connectivity, cohesive side-data, preserved point/cell sets,
side-specific interface point sets, and copied per-element material/region
arrays. The meshio path can write the updated mesh back through meshio;
multi-element-family production studies remain gated.

The cohesive mixed-mode benchmark writes the same compact standard bundle with
`cohesive_mixed_mode_response.csv`,
`cohesive_mixed_mode_response.png`,
`cohesive_mixed_mode_mesh_and_bc.png`, and a tangent finite-difference error in
`summary.json`.

The cohesive contact-compression benchmark writes
`cohesive_contact_compression_response.csv`,
`cohesive_contact_compression_response.png`,
`cohesive_contact_compression_mesh_and_bc.png`, and checks the optional normal
contact penalty without damage growth.

The cohesive delamination patch benchmark writes
`cohesive_delamination_patch_response.csv`,
`cohesive_delamination_patch_response.png`,
`cohesive_delamination_patch_damage_profile.png`,
`cohesive_delamination_patch_mesh_and_bc.png`, and
`visual_manifest.json`. It uses four cohesive segments under a tapered
mixed-mode prescribed displacement jump, checks assembled normal/shear
resultants against the closed-form bilinear cohesive law at two Gauss points
per segment, records localized damage, and reports the delamination-front
coordinate.

The structural DCB-style cohesive benchmark writes `mesh.geo`, `mesh.msh`,
`structural_dcb_response.csv`, `structural_dcb_load_displacement.png`,
`structural_dcb_damage_front.png`, `structural_dcb_energy.png`,
`structural_dcb_deformed_mesh.png`, and `visual_manifest.json`. It uses a
clamped-end two-arm specimen with a precrack and a bonded zero-thickness
interface, solves free internal bulk DOFs with the sparse cohesive Newton
path, and checks convergence, post-peak load softening, monotone cohesive
dissipation, front advance from the initial precrack, a bounded diagnostic
reaction-work energy gap, and review-safe visual dimensions.

## Expected Evidence

For J2 plasticity, `summary.json` reports the number of plastic steps and the
maximum residual in the consistency relation
`sigma_vm = sigma_y0 + H * eps_p_eq` after yielding.

For ductile PF-plasticity, `summary.json` reports solver-level J2 yielding,
Newton residual telemetry, ductile driving-force growth, the requested and
resolved sparse backend, and the bounded AT2 phase-field damage solve
residual. The runner writes `config.yaml`, `run_lockfile.json`,
`run_metadata.json`, `run_manifest.json`, `run.log`, `mesh.geo`, `mesh.msh`,
standard CSVs, setup/load/energy/damage figures, and `visual_manifest.json`.
This is the validated operator-coupled mechanics/damage layer used by the
guarded quasi-static T3 J2+AT2 staggered path. `energy.csv` is an integrated
energy ledger with elastic driving, plastic work, degraded elastic energy,
fracture surface/gradient energy, and total stored-plus-dissipated energy;
`summary.json` records finite and monotonicity checks plus backend status.
Full benchmark-matched ductile fracture remains pending.

The ductile sensitivity study writes `ductile_sensitivity_table.csv`,
`ductile_damage_sensitivity.png`, `ductile_driving_lift.png`, and one retained
child output bundle per case. It compares an elastic-driving reference against
ductile plastic-work driving for several `l0` values and checks yielding,
damage residuals, plastic-work driving lift, and review-safe figure dimensions.

For diffuse interphase fracture, `summary.json` reports the generated material
contrast, interface toughness reduction, and crack-density weighted fracture
energy for bulk-crossing versus interface-following candidate phase-field crack
paths. The interface-following path should be cheaper when the generated
interphase has reduced `Gc`.

For the two solid-interface examples:

- `weak_deflection` is a He-Hutchinson-style crack-impinging benchmark where a
  weak interphase makes interface deflection cheaper than bulk penetration.
- `strong_penetration` uses the same geometry but a tough interphase, so the
  straight bulk penetration path is cheaper than interface deflection.

For the cohesive displacement-jump benchmark, `summary.json` reports the
solver-coupled zero-thickness interface response for a fully prescribed mode-I
opening path. The committed cohesive damage and resultant normal traction are
checked against the bilinear traction-separation law at every load step. At
complete separation, the integrated dissipated energy is checked against the
analytical cohesive fracture-energy capacity `0.5 * sigma_max * delta_c`.

For the mixed-mode cohesive benchmark, `summary.json` reports both normal and
shear traction errors, committed scalar damage, and the maximum directional
finite-difference error for the assembled cohesive tangent along the active
loading path.

For the contact-compression benchmark, `summary.json` reports the normal
contact-traction error, verifies zero damage under pure compression, and records
the maximum tangent finite-difference error for the contact branch.

For the delamination patch benchmark, `summary.json` reports multi-element
normal/shear resultant errors, maximum cohesive tangent finite-difference
error, final damage extrema, active cohesive segment count, delamination-front
coordinate, visual-manifest status, and memory use.

For the structural DCB-style cohesive benchmark, `summary.json` reports
structural load-displacement response, maximum free-DOF residual, peak force
step, post-peak softening status, active cohesive segments, front advance,
integrated cohesive dissipation, bulk elastic energy, external work, a
diagnostic trapezoidal reaction-work energy-balance gap with a bounded
fractional tolerance, visual manifest status, and memory use. The runner cites
DCB/interlaminar delamination validation standards and analytical cohesive DCB
references, but keeps the claim scoped to a solver-coupled structural smoke
rather than ASTM D5528 material-property data reduction.

For the coupled PF+cohesive benchmark, `summary.json` reports a staggered
AT2 matrix damage plus cohesive-interface delamination run. The smoke gate
excludes pinned phase-field Dirichlet nodes from the free-DOF damage residual
and requires bounded damage residual and final staggered damage increment. The
bundle writes
`mesh.geo`, `mesh.msh`, `results.csv`, `history.csv`,
`solver_telemetry.csv`, `timing_per_step.csv`, `energy.csv`,
`cohesive_front.csv`, `initial_conditions.png`, `damage_final.png`,
`cohesive_damage_front.png`, `load_displacement.png`, `energy_split.png`,
`convergence.png`, `mesh_deformed.png`, `damage_history.png`,
`damage_evolution.gif`, and `visual_manifest.json`. The claim is scoped to
coupled brittle PF+CE smoke validation, not a calibrated PF-CZM, ASTM DCB, or
PF-plasticity-cohesive product workflow.

For the PF-CZM uniaxial strength benchmark, `summary.json` reports the
strength-calibrated Wu cohesive phase-field response for several `l0` values.
When gamma correction is enabled in PF-CZM solver paths, the degradation
parameter is calibrated from the same element-wise effective `Gc` used in the
fracture terms so the tensile-strength threshold is preserved.
The bundle writes `mesh.geo`, `mesh.msh`, `results.csv`, `history.csv`,
`energy.csv`, `solver_telemetry.csv`, `timing_per_step.csv`,
`damage_final.png`, `load_displacement.png`, `damage_history.png`,
`energy_split.png`, `convergence.png`, `mesh_deformed.png`,
`damage_evolution.gif`, and `visual_manifest.json`. The validation checks
that the degraded stress peak matches the target tensile strength, damage
onset occurs near `sigma_ts`, nonlinear residuals are finite/bounded, and
all review visuals pass dimension/media checks. The claim is scoped to a
forward PF-CZM damage-law calibration smoke, not a full structural
crack-growth, mixed-mode delamination, or ductile PF-plasticity-cohesive
workflow.

The sparse J2 backend-promotion harness records backend availability and
fallback behavior for the same elastoplastic patch:

```bash
python examples/plasticity_interface/run_sparse_j2_backend_promotion.py \
  --output-dir outputs/plasticity_interface/sparse_j2_backend_promotion \
  --backend auto --backend scipy --backend mumps --backend cudss
```

It writes `config.yaml`, `run_lockfile.json`, `run_metadata.json`,
`run_manifest.json`, `run.log`, `backend_promotion.csv`, and `summary.json`.
Retained backend-promotion evidence shows PETSc/MUMPS can resolve
`auto -> mumps` and explicit `mumps -> mumps`, matching the SciPy baseline on
the tested patch. A `cudss` request must either resolve on a configured
nvmath/cuDSS environment or record a clean fallback reason. cuDSS remains a
separate open promotion gate until `backend='cudss'` runs without fallback.

The ductile PF-plasticity backend-promotion harness exercises the coupled
ductile validation under the same backend-selection contract:

```bash
python examples/plasticity_interface/run_ductile_pf_backend_promotion.py \
  --output-dir outputs/plasticity_interface/ductile_pf_backend_promotion \
  --backend auto --backend scipy --backend mumps --backend cudss
```

It writes the same retained evidence bundle plus `backend_promotion.csv`
and one child bundle per requested backend. The current local proof point is
the SciPy path; PETSc/MUMPS and cuDSS still depend on the active HPC/backend
environment and remain the open promotion target for #659. Site-specific CPU
and GPU scheduler launchers live outside the public release payload.

## HPC/Memory Notes

The scripts are CPU-smoke examples and are intentionally small. They report
`max_rss_kib` in `summary.json` so local and cluster runs can be compared
without extra profiling tools.

Figures follow `docs/visualisation_requirements.md`: STIX-style serif fonts,
review-safe dimensions, labelled colorbars, and explicit geometry/path overlays.
