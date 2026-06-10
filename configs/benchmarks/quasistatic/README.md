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

Root-level legacy compatibility copies may exist in `configs/QS_*.yaml` for
historical commands. Check the contract file for what is canonical today.

## Diagnostics

Diagnostic-only YAMLs are kept under:

`configs/benchmarks/quasistatic/diagnostics/`

They are used for targeted code-path experiments (for example, pin-MPC
diagnostics in `QS_notched_holed_plate_welded.yaml`) and are not part of the
paper benchmark matrix by default.

## Command Manifests

The `manifests/` files describe runs whose entry point is a specialised Python
example module plus a comparator. They are consumed by Slurm wrappers and review
scripts rather than by `python -m phast run` directly.

| Manifest | Purpose | Current status |
|---|---|---|
| `manifests/QS_miehe_tension_repro.yaml` | Single-case reproducible Miehe SENT local verification run | Active |
| `manifests/QS_miehe_sens_repro.yaml` | Single-case reproducible Miehe SENS peak-window verification run | Active |
| `manifests/QS_miehe_tpb_repro.yaml` | Single-case reproducible Miehe TPB peak-window verification run | Active |
| `manifests/QS_sens_tpb_peak_window_corrected.yaml` | Clean SENS/TPB peak-window validation, PETSc/MUMPS mechanics, Zarr/MP4 outputs | Active HPC validation manifest |
| `manifests/QS_mesh_convergence_arc_length.yaml` | SENT/SENS/TPB mesh-convergence and Riks-style arc-length diagnostics | Post-peak and convergence diagnostics |
| `manifests/QS_sens_tpb_rescue_visuals.yaml` | Older targeted SENS/TPB visual rescue cells | Diagnostic, use only when the issue thread names it |

The active HPC launcher for the peak-window manifest is:

```bash
scripts/slurm/benchmarks/hpc_qs_sens_tpb_peak_window.slurm
```

It requires a functional PETSc/MUMPS backend by default and fails fast if the
runtime smoke test does not select `mumps`.

## Reproducibility Index

Use this compact list when re-running each quasistatic benchmark family.

| Contract | Launcher |
|---|---|
| `QS_notched_holed_plate.yaml` | `python -m phast run configs/benchmarks/quasistatic/QS_notched_holed_plate.yaml` |
| `QS_notched_holed_plate_comsol_strict.yaml` | `python -m phast run configs/benchmarks/quasistatic/QS_notched_holed_plate_comsol_strict.yaml` |
| `QS_lshaped_concrete.yaml` | `python -m phast run configs/benchmarks/quasistatic/QS_lshaped_concrete.yaml` |
| `manifests/QS_sens_tpb_peak_window_corrected.yaml` | `scripts/slurm/benchmarks/hpc_qs_sens_tpb_peak_window.slurm` |
| `manifests/QS_mesh_convergence_arc_length.yaml` | `scripts/slurm/benchmarks/hpc_qs_mesh_convergence_arc.slurm` |
| `manifests/QS_miehe_tension_repro.yaml` | Local `python -m phast` substitute via direct module runner; see example section |
| `manifests/QS_miehe_sens_repro.yaml` | Local module runner for Miehe SENS reproducibility |
| `manifests/QS_miehe_tpb_repro.yaml` | Local module runner for Miehe TPB reproducibility |
| `manifests/QS_sens_tpb_rescue_visuals.yaml` | `scripts/slurm/benchmarks/hpc_qs_sens_tpb_rescue.slurm` |

A machine-readable contract file mirrors this table and can be used for scripted
reruns or smoke checks:

- `configs/benchmarks/quasistatic/reproducibility_contracts.yaml`

For a user-facing catalogue of all quasi-static rerun commands and required
artifacts, use:

- `examples/quasistatic/reproducibility_catalog.yaml`

## Cleanup Rule

Do not commit `training_data.zarr`, H5 files, VTU files, raw Slurm logs, or
scratch result folders. Promote only reviewed compare reports, small figures,
and documented reference evidence.
