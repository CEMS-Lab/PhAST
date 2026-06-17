# Standard outputs for a fracture-FEM run

The canonical promoted-example contract is
[`docs/user_guide/example_contract.md`](../user_guide/example_contract.md). Use
that page first when deciding what a public PhAST example folder must contain.
The first-class output overview is
[`docs/output_standards/index.md`](../output_standards/index.md). This file is
retained as the compatibility checklist for solver-run output families and
historical benchmark audits.

This is the compatibility checklist for what every public run (forward dynamic,
quasistatic, solid-mechanics, and validation examples) should produce. Built
2026-05-09 and updated 2026-05-14 from auditing existing benchmark dirs
(B7, Kalthoff, SENT, TPB, L-shape, notched-holed-plate, Miehe
shear/tension) plus consensus from the PF-fracture literature
(Bourdin/Francfort/Marigo, Miehe et al., Borden et al., Bleyer
2017/2025, Kosin et al. 2024, COMSOL App Library) and common conventions
from open phase-field fracture examples such as PhaseFieldX, PhaFiDyn,
FEniCS-Explicit-PF, and COMET.

Publication sizing, font, GIF, and review-dimension rules are tracked in the
public [example contract](../user_guide/example_contract.md). A run is not
paper-ready until both this artifact checklist and the visual contract pass.

## Pattern (consensus across the field)

Every reference codebase follows the same 2-stage pattern:

1. **During the run**: store raw fields per timestep when requested (Zarr for new training/dataset workflows; legacy HDF5 / XDMF / VTU for compatibility and visualization) + numerical trajectory CSVs (load, energy, crack tip, telemetry) + convergence log. Raw field snapshots are the source of truth when trajectory tensors are requested; otherwise the canonical per-step CSVs (`results.csv`, `energy.csv`, `timing_per_step.csv`, `solver_telemetry.csv`) are the validation/audit source of truth.
2. **Post-processing** (separate scripts): read the raw + CSVs, render publication PNGs/GIFs/comparison figures.

PhaseFieldX example: `top.reaction`, `bottom.reaction`, `top.dof`,
`total.energy`, `phasefieldx.conv`, `paraview-solutions_vtu/` per-step.
Plot scripts such as `plot_1711.py`, `plot_1712.py`, and `plot_1714.py`
then generate publication figures: force-displacement, fracture energy,
Gamma, final phase-field, displacement, and stagger-iteration curves.

PhaFiDyn example: `XDMFFile` per-step + `.data` tab-separated trajectory + commented-out matplotlib in same script.

Our setup matches this pattern via Zarr-first dataset stores for new
neural-operator/large-corpus work, optional legacy `training_data.h5` for
existing benchmark post-processing, per-step CSVs, `compare.py`, and paper
figure scripts.

Keep raw H5/Zarr archives outside git and mirror only curated summaries,
manifests, or regenerated figures into the repository.

## 1. Configuration + provenance

| File | Required | What |
|---|---|---|
| `config.yaml` | ✅ | The exact config used (post-resolution, all defaults expanded) |
| `run_lockfile.json` | ✅ | Reproducibility contract: input config SHA-256, post-CLI resolved config, CLI args, git state, dependency versions, hostname/platform, resolved mesh/material/solver/device summaries |
| `run_metadata.json` | ✅ | git SHA, CLI args, timestamp, hostname, device, dtype, n_threads |
| `mesh.geo` | ✅ | gmsh source for reproducibility |
| `mesh.msh` | ✅ | mesh as run (so re-meshing isn't required to re-run) |

## 2. Pre-run visualisation

| File | Required | What |
|---|---|---|
| `initial_conditions.png` | ✅ | mesh + boundary tags + pre-crack viz; the "what was set up" image |

## 3. Per-step CSVs (numerical trajectory)

| File | Required | Columns |
|---|---|---|
| `results.csv` | ✅ when reaction output is configured | `step,time,displacement,reaction_kN,max_d,max_H,stagger_iter,elapsed_ms` |
| `history.csv` | ✅ | step, max(H), max(psi_plus), max(d), energy terms if available, reaction_force, applied_disp |
| `crack_tip.csv` | ✅ (when crack present) | step, time, tip_x, tip_y, tip_speed (mm/s) |
| `energy.csv` | ✅ | `step,time,elastic,fracture,kinetic,external,total` |
| `timing_per_step.csv` | ✅ | step, wall_ms, fwd_ms, bwd_ms (if grad) |
| `solver_telemetry.csv` | ✅ (per #300) | step, time/load parameter, newton/stagger_iters, pcg_iters_mech, pcg_iters_pf, residual, relative_residual, mechanics_residual, mechanics_relative_residual, dt |

For quasi-static validation problems, `results.csv` must use the same
reaction side and sign convention as the reference implementation:
PhaseFieldX 1711 uses `-bottom.reaction["Ry"]`, 1712 uses
`-bottom.reaction["Rx"]`, and 1714 uses `-top.reaction["Ry"]`.
L-shaped-panel references without a digitised load-displacement curve still
must write the driven boundary reaction so later digitisation can be scored
without re-running the solve.

## 4. Final-state plots (PNGs)

| File | Required | What |
|---|---|---|
| `damage_final.png` | ✅ | d(x) colourmap at end of run |
| `load_displacement.png` | ✅ when displacement/load control exists | F vs u curve; overlay the digitised/reference curve when available |
| `energy.png` | ✅ | E_elastic / E_dissipated / E_kinetic / E_external vs t (stacked or 4-panel) |
| `crack_path.png` | ✅ (dynamic) | crack tip x(t), y(t), speed(t) |
| `staggered_convergence.png` | ✅ (quasistatic) | staggered iterations, residual, and mechanics/phase-field linear iterations per load step |

Problem-specific PhaseFieldX-derived plots should be emitted whenever the
quantity exists in the reference setup:

| Problem family | Required validation plots |
|---|---|
| Miehe SENT/SENS/TPB and similar QS benchmarks | `load_displacement.png`, `staggered_convergence.png`, `damage_final.png`, `compare.png`, `compare_report.txt` |
| L-shaped panel | `load_displacement.png`, `staggered_convergence.png`, `damage_final.png`, crack-path/kink metric figure when reference geometry is available |
| Dynamic crack benchmarks | damage snapshots/GIF, crack-tip trajectory, energy balance, reference timing/position panels |

## 5. Animations (GIFs)

| File | Required | What |
|---|---|---|
| `damage_evolution.gif` | ✅ | d(x, t) over time, lightweight public animation |
| `deformation_evolution.gif` | ✅ if requested | \|u\|(x, t) coloured by magnitude |
| `stress_evolution.gif` | ✅ if requested | von Mises stress |
| `strain_evolution.gif` | optional | strain trace |
| `psi_plus_evolution.gif` | optional | tensile elastic energy density (driving force) |

## 6. Comparison vs reference

| File | Required | What |
|---|---|---|
| `compare.png` | ✅ (if ref exists, post-run compare step) | side-by-side: ours vs reference (panel grid) |
| `compare_report.txt` | ✅ (if ref exists, post-run compare step) | numerical metrics + PASS/FAIL gates |

## 7. Trajectory data (for ML / re-analysis)

| File | Required | Layout |
|---|---|---|
| `sample_*.zarr` / `training_data.zarr` / run-level `.zarr` store | ✅ for new neural-operator, replay-buffer, and large dataset-generation workflows | canonical chunked trajectory layout from `dataset_benchmark/schema.py` for packaged samples. Solver-run `training_data.zarr` stores keep both `simulation_data/steps/step_####` legacy groups and dense step-major `simulation_data/trajectory/*` arrays for fast post-processing and restart loading. Fields include mesh, material, keyframe times, `damage_nodal`, `displacement`, `psi_plus`, `psi_minus` where available, `H_elem`, `H_nodal`, strain/stress, telemetry, and validation metadata. Use chunking/compression appropriate to the reader; do not claim Zarr is always smaller than H5. |
| `training_data.h5` | legacy read/convert compatibility only | historical per-step or per-keyframe solver archives; verified existing schema (Kalthoff job2) per-step group: `H_elem`, `H_nodal`, `acceleration` (N,2), `damage_nodal`, `displacement` (N,2), `psi_plus`, `strain` (E,3), `stress` (E,3), `velocity` (N,2). New `--h5` / `output.h5` runs now write `training_data.zarr`; convert old H5 archives with maintained release tooling before publishing them. |
| `paraview/*.xdmf` | optional | mirror of trajectory fields in XDMF format for ParaView users (PhaseFieldX / PhaFiDyn convention). Not currently emitted; can convert from legacy H5 or Zarr post-hoc once converters exist. |

Readers should prefer `simulation_data/trajectory/*` when present and fall
back to `simulation_data/steps/step_####` for older stores. Publication
post-processing now accepts `training_data.zarr` directly; H5 remains a legacy
compatibility path.

## 8. Logging

| File | Required | What |
|---|---|---|
| `run.log` | ✅ | full stdout (redirected; not just the slurm `.o`) |

---

| `recovered_vs_truth_compare.png` | ❌ MISSING — side-by-side panel |

## Slurm-side outputs

| File | Required | What |
|---|---|---|
| `logs/<jobname>_<jobid>.o` | ✅ slurm | stdout (mirrors `run.log`) |
| `logs/<jobname>_<jobid>.e` | ✅ slurm | stderr |

---

## Audit status (2026-05-14)

| Job category | Coverage of canonical set |
|---|---|
| **B7 dynamic crack branching** (`b7_branching_47961`) | accepted bundle complete for public evidence; raw `training_data.zarr` retained in a private HPC archive (`98G`, verified 2026-06-13). Stress/strain fields are present in the raw trajectory; stress/strain GIFs remain optional derived media, not a validation blocker. |
| **Kalthoff** (e.g. `job2_*/B2_kalthoff_mesh1`) | 10/14 (missing stress/strain GIFs, energy.png, telemetry CSV) |
| **QS Miehe SENT** (e.g. `examples/quasistatic/miehe_tension/reference_runs/qs_sent_37992/`) | standalone runner writes telemetry, `timing_per_step.csv`, `energy.csv`, `energy.png`, Zarr trajectory output, MP4 damage animation, load-displacement and staggered-convergence plots; remaining gaps are problem-specific derived crack-path outputs |
| **QS notched-holed plate** | YAML runner writes telemetry, timing, `energy.csv`, `results.csv` when `output.reaction_node_set` is set; compare artifacts are produced by `compare.py` |

## How to apply

When designing a new benchmark or public example:
1. Use this checklist to decide what the run should emit.
2. If a category is unchecked, file an issue (umbrella: #485) before declaring the run complete.
3. Reference this doc in slurm header + demo CLI `--help`.
