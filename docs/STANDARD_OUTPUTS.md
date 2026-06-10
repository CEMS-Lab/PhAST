# Standard outputs for a fracture-FEM run

This is the canonical checklist for what every public forward run (dynamic,
quasistatic, plasticity/cohesive validation, and PF-CZM validation) should
produce. Built
2026-05-09 and updated 2026-05-14 from auditing existing benchmark dirs
(B7, Kalthoff, SENT, TPB, L-shape, notched-holed-plate, Miehe
shear/tension) plus consensus from the PF-fracture literature
(Bourdin/Francfort/Marigo, Miehe et al., Borden et al., Bleyer
2017/2025, Kosin et al. 2024, COMSOL App Library) and reference-code
conventions in `reference_codes/` (PhaseFieldX, PhaFiDyn,
FEniCS-Explicit-PF, COMET).

Publication sizing, font, GIF, particle-overlay, and review-dimension rules
are tracked in `docs/visualisation_requirements.md`. A run is not paper-ready
until both this artifact checklist and the visualisation checklist pass.

## Pattern (consensus across the field)

Every reference codebase follows the same 2-stage pattern:

1. **During the run**: store raw fields per timestep when requested (Zarr, legacy HDF5, XDMF, or VTU depending on the configured output path) + numerical trajectory CSVs (load, energy, crack tip, telemetry) + convergence log. Raw field snapshots are the source of truth when trajectory tensors are requested; otherwise the canonical per-step CSVs (`results.csv`, `energy.csv`, `timing_per_step.csv`, `solver_telemetry.csv`) are the validation/audit source of truth.
2. **Post-processing** (separate scripts): read the raw + CSVs, render publication PNGs/GIFs/comparison figures.

PhaseFieldX example: `top.reaction`, `bottom.reaction`, `top.dof`,
`total.energy`, `phasefieldx.conv`, `paraview-solutions_vtu/` per-step.
Plot scripts such as `plot_1711.py`, `plot_1712.py`, and `plot_1714.py`
then generate publication figures: force-displacement, fracture energy,
Gamma, final phase-field, displacement, and stagger-iteration curves.

PhaFiDyn example: `XDMFFile` per-step + `.data` tab-separated trajectory + commented-out matplotlib in same script.

Our setup matches this pattern via optional Zarr/HDF5 trajectory stores,
per-step CSVs, `compare.py`, and figure scripts.

The legacy HPC H5 generator-to-figure map is tracked in
`docs/hpc_h5_provenance_index.md`. Keep raw H5 archives on HPC scratch and
mirror only curated summaries, manifests, or regenerated figures into git.

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
| `damage_evolution.mp4` | ✅ | d(x, t) over time, default fast raster MP4; GIF/APNG available by explicit animation format |
| `displacement_evolution.mp4` | ✅ if requested | \|u\|(x, t) coloured by magnitude; GIF/APNG available by explicit animation format |
| `stress_evolution.mp4` | ✅ if requested | von Mises stress; GIF/APNG available by explicit animation format |
| `strain_evolution.gif` | optional | strain trace |
| `psi_plus_evolution.gif` | optional | tensile elastic energy density (driving force) |

## 6. Comparison vs reference

| File | Required | What |
|---|---|---|
| `compare.png` | ✅ (if ref exists, post-run compare step) | side-by-side: ours vs reference (panel grid) |
| `compare_report.txt` | ✅ (if ref exists, post-run compare step) | numerical metrics + PASS/FAIL gates |

## 7. Trajectory data and re-analysis

| File | Required | Layout |
|---|---|---|
| `training_data.zarr` / run-level `.zarr` store | optional when configured | Chunked trajectory output for forward-run post-processing and large artifact inspection. Solver-run `training_data.zarr` stores keep both `simulation_data/steps/step_####` legacy groups and dense step-major `simulation_data/trajectory/*` arrays where available. Fields may include mesh, material, keyframe times, `damage_nodal`, `displacement`, `psi_plus`, `psi_minus`, `H_elem`, `H_nodal`, strain/stress, telemetry, and validation metadata. Use chunking/compression appropriate to the reader; do not claim Zarr is always smaller than H5. |
| `training_data.h5` | legacy read/convert compatibility only | historical per-step or per-keyframe solver archives; verified existing schema (Kalthoff job2) per-step group: `H_elem`, `H_nodal`, `acceleration` (N,2), `damage_nodal`, `displacement` (N,2), `psi_plus`, `strain` (E,3), `stress` (E,3), `velocity` (N,2). New `--h5` / `output.h5` runs now write `training_data.zarr`; use `scripts/h5_to_zarr.py` for old H5 archives. |
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

## Slurm-side outputs

| File | Required | What |
|---|---|---|
| `logs/<jobname>_<jobid>.o` | ✅ slurm | stdout (mirrors `run.log`) |
| `logs/<jobname>_<jobid>.e` | ✅ slurm | stderr |

---

## Audit status (2026-05-14)

| Job category | Coverage of canonical set |
|---|---|
| **B7 dynamic crack branching** (e.g. `b7_branching_29760`) | 12/14 (missing stress/strain GIFs) |
| **Kalthoff** (e.g. `job2_*/B2_kalthoff_mesh1`) | 10/14 (missing stress/strain GIFs, energy.png, telemetry CSV) |
| **QS Miehe SENT** (e.g. `examples/quasistatic/miehe_tension/reference_runs/qs_sent_37992/`) | standalone runner writes telemetry, `timing_per_step.csv`, `energy.csv`, `energy.png`, Zarr trajectory output, MP4 damage animation, load-displacement and staggered-convergence plots; remaining gaps are problem-specific derived crack-path outputs |
| **QS notched-holed plate** | YAML runner writes telemetry, timing, `energy.csv`, `results.csv` when `output.reaction_node_set` is set; compare artifacts are produced by `compare.py` |
| **Inversion (Pair A)** (e.g. `inv_pA_h_sweep_*/hard_max/`) | partial — has truth+init viz, JSON convergence; missing recovered viz + loss/pos_err PNG |
| **ES (Pair A)** (e.g. `es_pair_A_cpu_smoke_30024`) | partial — has loss curve + JSON; missing recovered viz |

## How to apply

When designing a new benchmark or validation example:
1. Use this checklist to decide what the run should emit.
2. If a category is unchecked, file an issue (umbrella: #485) before declaring the run complete.
3. Reference this doc in slurm header + demo CLI `--help`.
