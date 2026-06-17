# Quasistatic Benchmark Configs

This directory contains the source-controlled launch definitions for
quasistatic validation.

Keep generated run outputs under the chosen run directory, HPC scratch,
or a promoted lightweight reference folder, not beside these YAMLs.

## Direct Problem Configs

These files are standard `phast run` configs:

| Config | Purpose | Typical command |
|---|---|---|
| `QS_lshaped_concrete.yaml` | L-shaped concrete panel, Ambati/Winkler-style validation | `python -m phast run configs/benchmarks/quasistatic/QS_lshaped_concrete.yaml` |
| `QS_notched_holed_plate.yaml` | COMSOL holed-plate baseline | `python -m phast run configs/benchmarks/quasistatic/QS_notched_holed_plate.yaml` |
| `QS_notched_holed_plate_comsol_strict.yaml` | COMSOL holed-plate stricter parity setup | `python -m phast run configs/benchmarks/quasistatic/QS_notched_holed_plate_comsol_strict.yaml` |

Top-level compatibility copies such as `configs/QS_*.yaml` have been removed.
Use the canonical `configs/benchmarks/quasistatic/...` paths in documentation,
scripts, and paper supplement commands.

## Command Manifests

The command manifests describe runs whose entry point is a specialised Python
example module plus a comparator. They are consumed by Slurm wrappers and review
scripts rather than by `python -m phast run` directly. They are kept under
`docs/internal/config_manifests/quasistatic/manifests/` because they are
orchestration records, not public solver problem configs.

| Manifest | Purpose | Current status |
|---|---|---|
| `QS_miehe_tension_repro.yaml` | Single-case reproducible Miehe SENT local verification run | Active |
| `QS_miehe_sens_repro.yaml` | Single-case reproducible Miehe SENS peak-window verification run | Active |
| `QS_miehe_tpb_repro.yaml` | Single-case reproducible Miehe TPB peak-window verification run | Active |
| `QS_sens_tpb_peak_window_corrected.yaml` | Clean SENS/TPB peak-window validation, PETSc/MUMPS mechanics, Zarr/MP4 outputs | Internal HPC validation manifest |
| `QS_mesh_convergence_arc_length.yaml` | SENT/SENS/TPB mesh-convergence and Riks-style arc-length studies | Internal post-peak/convergence manifest |
| `QS_sens_tpb_rescue_visuals.yaml` | Targeted SENS/TPB visual rerun cells | Internal rerun manifest |

The active HPC launcher for the peak-window manifest is maintained as a
site-specific scheduler template outside the public repository. It requires a
functional PETSc/MUMPS backend by default and should fail fast if the runtime
validation test does not select `mumps`.

## Reproducibility Index

Use this compact list when re-running each quasistatic benchmark family.

| Contract | Launcher |
|---|---|
| `QS_notched_holed_plate.yaml` | `python -m phast run configs/benchmarks/quasistatic/QS_notched_holed_plate.yaml` |
| `QS_notched_holed_plate_comsol_strict.yaml` | `python -m phast run configs/benchmarks/quasistatic/QS_notched_holed_plate_comsol_strict.yaml` |
| `QS_lshaped_concrete.yaml` | `python -m phast run configs/benchmarks/quasistatic/QS_lshaped_concrete.yaml` |
| `QS_sens_tpb_peak_window_corrected.yaml` | Site scheduler template |
| `QS_mesh_convergence_arc_length.yaml` | Site scheduler template |
| `QS_miehe_tension_repro.yaml` | Local `python -m phast` substitute via direct module runner; see example section |
| `QS_miehe_sens_repro.yaml` | Local module runner for Miehe SENS reproducibility |
| `QS_miehe_tpb_repro.yaml` | Local module runner for Miehe TPB reproducibility |
| `QS_sens_tpb_rescue_visuals.yaml` | Site scheduler template |

A machine-readable contract file mirrors this table and can be used for scripted
reruns or validation checks:

- `docs/internal/config_manifests/quasistatic/reproducibility_contracts.yaml`

For a user-facing catalogue of all quasi-static rerun commands and required
artifacts, use:

- `examples/quasistatic/reproducibility_catalog.yaml`

## Cleanup Rule

Do not commit `training_data.zarr`, H5 files, VTU files, raw Slurm logs, or
scratch result folders. Promote only reviewed compare reports, small figures,
and documented reference evidence.
