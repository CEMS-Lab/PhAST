# Benchmark Config To Run Map

This file is the source-of-truth index for shipped YAML configs, archived run
folders, and their benchmark role. Keep result-local `config.yaml` files inside
run folders as provenance snapshots; do not treat them as canonical launch
configs.

## Config Classes

| Class | Paths | Rule |
|---|---|---|
| Canonical launch configs | `configs/benchmarks/dynamic/*.yaml`, `configs/benchmarks/quasistatic/*.yaml` | User-facing configs for reruns and documentation. Root `configs/B*.yaml` and `configs/QS*.yaml` are compatibility symlinks. |
| Diagnostic launch configs | `configs/benchmarks/dynamic/diagnostics/**/*.yaml`, `configs/benchmarks/quasistatic/diagnostics/*.yaml` | Kept for discrepancy analysis, timing, or literature-mesh sensitivity. Do not cite as the primary benchmark unless the paper/issue explicitly names them. |
| Command manifests | `configs/benchmarks/plasticity_interface/manifests/*.yaml` | Reproducibility manifests for workflows whose forward model is built by a specialised Python module rather than by `python -m phast run`. |
| Run snapshots | `examples/**/run_metadata.json`, `examples/**/config.yaml`, `papers/**/results/**/config.yaml` | Immutable provenance from completed runs. Use to identify what actually produced a figure/result. |
| Non-problem manifests | `configs/benchmarks/quasistatic/manifests/*.yaml` | Slurm/visual orchestration, not solver problem configs. Root `configs/QS_*manifest*.yaml` paths are compatibility symlinks when present. |
| Templates | `configs/REFERENCE*.yaml`, `configs/phast.schema.json` | Documentation/schema only. |

## Canonical Dynamic Benchmarks

| Benchmark | Canonical config | Correct archived runs / evidence | Notes |
|---|---|---|---|
| B1 glass branching, Borden/Liu | `configs/benchmarks/dynamic/B1_branching_glass.yaml` | Historical paper-1 runs: `examples/dynamic/hpc_jobs_raw/job2_20260401_104228_gpu03_job8586/B1_branching_mesh{1,2,3}`. Current corrected config uses `l0=0.25 mm`, `h_crack=0.125 mm` by default; Mesh 3 paper-quality rerun should override `h_crack=0.0625 mm`. | The older job2 raw runs used `l0=0.5 mm`; keep them as historical Paper 1 evidence only. Use `configs/benchmarks/dynamic/diagnostics/B1_branching_glass_liu_structured.yaml` for Liu-style structured split-quad diagnostics. |
| B2 Kalthoff-Winkler | `configs/benchmarks/dynamic/B2_kalthoff_winkler.yaml` | Current reference/timing runs: `examples/dynamic/kalthoff/reference_runs/B2_kalthoff_mesh{1..5}` from `examples/dynamic/timing_comparisons/kalthoff/config.yaml`. Historical raw runs: `examples/dynamic/hpc_jobs_raw/job2_20260401_104228_gpu03_job8586/B2_kalthoff_mesh{1,2}`. | The `reference_runs/` tree is the clean comparison set; `hpc_jobs_raw/job2` is historical. |
| B3 dynamic SENT | `configs/benchmarks/dynamic/B3_dynamic_sent.yaml` | Paper-1 figure/run: `examples/dynamic/hpc_jobs_raw/job2_20260401_104228_gpu03_job8586/B3_dynamic_sent`; downloaded working copy without H5: `examples/dynamic/sent/hpc_results/job2_20260401_104228_gpu03_job8586/B3_dynamic_sent`. | Config and run are spectral/AT2, `L=H=40 mm`, `a=20 mm`, `l0=0.5 mm`, `h_crack=0.25 mm`, 1091 nodes. |
| B3 clean timing comparison | `configs/benchmarks/dynamic/diagnostics/B3_sent_clean_timing.yaml` | `examples/dynamic/timing_comparisons/sent/torch/output` plus FEniCS sibling. | Timing/provenance config only, not the Paper-1 B3 SENT figure config. |
| B4 crack coalescence | `configs/benchmarks/dynamic/B4_coalescence.yaml` | Historical raw run: `examples/dynamic/hpc_jobs_raw/job2_20260401_104228_gpu03_job8586/B4_coalescence`. | Historical demo; not currently a primary acceptance benchmark. |
| B5 PMMA branching | `configs/benchmarks/dynamic/B5_pmma_branching.yaml`; sweep configs `configs/benchmarks/dynamic/B5_pmma_branching_dU*.yaml` | Raw sweep evidence: `examples/dynamic/hpc_jobs_raw/job1_20260401_104226_gpu04_job8585/B5a_pmma_dU0.05`, `examples/dynamic/hpc_jobs_raw/job3_20260401_154602_gpu04_job8632/B5b_pmma_dU0.04`, `B5c_pmma_dU0.035`. | Correct physics: Bleyer PMMA, AT1, Amor/volumetric-deviatoric split, two-step prestrain + dynamic release. |
| B6 perforated PMMA | `configs/benchmarks/dynamic/B6_perforated_{1hole_near,1hole_far,10holes,30holes}.yaml` | Raw evidence: `examples/dynamic/hpc_jobs_raw/job1_20260401_104226_gpu04_job8585/B6{a,b,c,d}_...`; extra 15-hole offsets in `job3_20260401_154602_gpu04_job8632/B6{e,f,g,h}_...`. | Dynamic Bleyer perforated-plate benchmark, not quasi-static. The 15-hole offset cases are extra diagnostics and currently do not have root canonical YAMLs. |
| B7 COMSOL dynamic branching | `configs/benchmarks/dynamic/B7_dynamic_crack_branching_comsol.yaml` | Local evidence: `examples/dynamic/crack_branching_comsol/run_b7_mac_cpu_20260504_181618`. Debug variants under `configs/benchmarks/dynamic/diagnostics/B7_debug/`. | Root config is the stable public entry point. The `B7_debug` files are structured diagnostics for COMSOL parity, not replacements until accepted. |

## Canonical Quasi-Static Benchmarks

| Benchmark | Canonical config | Correct archived runs / evidence | Notes |
|---|---|---|---|
| QS L-shaped concrete | `configs/benchmarks/quasistatic/QS_lshaped_concrete.yaml` | Historical glass evidence: `examples/quasistatic/l_shaped_panel/run_glass_mac_cpu_20260429_130539`; matrix sweep evidence: `examples/quasistatic/l_shaped_panel/run_qs_matrix40_33819_lshape_*`. | Use the config for Ambati/Winkler concrete reruns. The `run_glass_*` folder is historical and not the canonical material case. |
| QS Miehe SENT | `configs/benchmarks/quasistatic/manifests/QS_mesh_convergence_arc_length.yaml` | Promoted examples under `examples/quasistatic/miehe_tension/reference_runs/`. | SENT is driven by `examples.quasistatic.miehe_tension.run` plus `compare.py`; the manifest records the reproducible CLI arguments. |
| QS Miehe shear/SENS | `configs/benchmarks/quasistatic/manifests/QS_sens_tpb_peak_window_corrected.yaml`; convergence diagnostics in `QS_mesh_convergence_arc_length.yaml` | Current clean HPC rerun: Slurm array `45999`; promotion pending compare/artifact checks. | SENS is a command-manifest workflow, not a direct `phast run` YAML. The active clean setup uses PETSc/MUMPS, no diagnostic damage viscosity, and Zarr/MP4 outputs. |
| QS three-point bending | `configs/benchmarks/quasistatic/manifests/QS_sens_tpb_peak_window_corrected.yaml`; post-peak diagnostics in `QS_mesh_convergence_arc_length.yaml` | Current clean HPC rerun: Slurm array `45999`; promotion pending compare/artifact checks. | TPB is a command-manifest workflow. The active peak-window setup uses PETSc/MUMPS and a high stagger cap; full snap-back remains an arc-length diagnostic track. |
| QS COMSOL notched holed plate | `configs/benchmarks/quasistatic/QS_notched_holed_plate.yaml` | `examples/quasistatic/notched_holed_plate/hpc_results/job32465_notched_holed/run`. | Isotropic AT2 baseline. |
| QS COMSOL notched holed plate strict | `configs/benchmarks/quasistatic/QS_notched_holed_plate_comsol_strict.yaml` | `examples/quasistatic/notched_holed_plate/hpc_results/job32761_qs_comsol_strict/notched_holed_voldev/run`; matrix sweep `run_qs_matrix40_33819_notched_holed_*`. | Amor/vol-dev, `eta=1e-5`, closer COMSOL parity. Prefer for COMSOL strict comparisons. |
| QS notched holed welded diagnostic | `configs/QS_notched_holed_plate_welded.yaml` | No accepted canonical run. | Diagnostic for replacing rigid connectors with welded/prescribed BCs; do not cite as benchmark. |

## Plasticity, Cohesive, And Contact Examples

| Config | Role | Correct run family |
|---|---|---|
| `configs/benchmarks/plasticity_interface/manifests/customer_validation_examples.yaml` | Command manifest for J2 plasticity, ductile PF-plasticity, cohesive mode-I, mixed-mode, contact-compression, delamination patch, structural DCB-style cohesive, and diffuse-interface examples | The runnable entry points live under `examples/plasticity_interface/`; retained evidence is summarised in `docs/customer_readiness.md`. |

## Cleanup Rules

1. New user-facing benchmark: add exactly one canonical
   `configs/benchmarks/<family>/<ID>_<name>.yaml`, then add its accepted run
   folder to this file. Add a root compatibility symlink only when an existing
   command line, test, or historical document still needs the old path.
2. Parameter sweeps: keep sweep configs only when each file is directly
   runnable and maps to a named paper/acceptance cell.
3. Debug variants: place under `configs/benchmarks/*/diagnostics/` and
   label them diagnostic in comments and this map.
4. Result-local configs must remain in their run directories; they are
   provenance snapshots, not launch recommendations.
5. H5/HDF5/large NPZ output files must not be committed. Keep large result
   payloads on HPC/object storage.
