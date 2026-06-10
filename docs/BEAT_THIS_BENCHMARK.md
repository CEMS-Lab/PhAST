# Beat this benchmark

This page is the stable target linked from the README. It records
solver timing snapshots and points to the reproducible artefacts that
back them.

These are engineering benchmarks, not fixed manuscript claims. The
numbers depend on solver versions, optional backends, hardware,
threading, tolerances, mesh regeneration, and run configuration.
`phast`, Akantu, FEniCS/PhaFiDyn, COMSOL, PETSc, cuDSS, and
other relevant software stacks are all under active development. Rerun
the scripts before using these numbers in a paper, proposal, or public
comparison.

## Current Status

The cross-code timing table is under refresh for the public release. Earlier
Akantu comparison runs may have used a debug-mode build/configuration, so those
numbers are intentionally not published here as headline claims.

The public benchmark target is now the protocol, not a stale ratio:

| Benchmark | Required comparison state before publishing a ratio |
| --- | --- |
| Miehe SENT clean-IC tension | PhAST, Akantu, and FEniCS all rerun from clean environments with documented build type, device, threads, mesh, tolerances, and damage cadence. |
| Miehe SENS shear-notched | Same shared-mesh protocol, with Release-mode external solvers and archived run metadata. |
| Kalthoff-Winkler impact | PhAST and FEniCS rerun for the spectral split; Akantu included only for a compatible Amor/vol-dev variant. |

## Paper-1 Timing Refresh

This page is the home for the Paper 1 performance comparison once the refreshed
release-mode benchmark jobs are complete. Until then, do not quote old Akantu or
cross-code speedup values from development notes, commit history, screenshots,
or unpublished run folders.

Important caveats:

- The target harnesses are explicit dynamic timing harnesses. They are not
  quasi-static timing claims.
- Phase-field subcycling is disabled in this table so every solver
  performs one damage solve per explicit step.
- SENT should use AT2/Amor so Akantu can participate. Kalthoff-Winkler uses
  the Miehe spectral split, so Akantu is omitted there.
- Quasi-static runs may use matrix-free CG/JVP mechanics or an
  assembled sparse-direct backend. Always check the archived
  `run_metadata.json` or `run_lockfile.json`.

## Reproduce

Historical paper benchmark settings were tracked under
`papers/paper/BENCHMARK_SETTINGS.md` before the Paper 1 artifact cleanup.
Current timing comparison artefacts are under
`examples/dynamic/timing_comparisons/`.

For fresh benchmark work, prefer the YAML entry point:

```bash
python -m phast run configs/benchmarks/dynamic/B2_kalthoff_winkler.yaml --device cuda
python -m phast run configs/benchmarks/dynamic/B3_dynamic_sent.yaml --device cuda
```

Use the benchmark-specific `compare.py` scripts under
`examples/{dynamic,quasistatic}/<name>/` to regenerate acceptance plots and text
reports from a completed run directory.
