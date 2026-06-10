# Quasistatic Examples

Canonical landing area for quasistatic drivers, comparison helpers, and
reference data. Generated run folders should stay outside the repository unless
they have been explicitly promoted as lightweight reference evidence.

## Problem Folders

| Folder | Benchmark | Canonical rerun entry point |
|---|---|---|
| `miehe_tension/` | Miehe SENT tension, PhaseFieldX 1711 | `configs/benchmarks/quasistatic/manifests/QS_mesh_convergence_arc_length.yaml` |
| `miehe_shear/` | Miehe SENS shear, PhaseFieldX 1712 | `configs/benchmarks/quasistatic/manifests/QS_sens_tpb_peak_window_corrected.yaml` |
| `three_point_bending/` | Miehe three-point bending, PhaseFieldX 1714 | `configs/benchmarks/quasistatic/manifests/QS_sens_tpb_peak_window_corrected.yaml` |
| `l_shaped_panel/` | L-shaped panel, Ambati/Winkler/Rudshaug references | `configs/benchmarks/quasistatic/QS_lshaped_concrete.yaml` |
| `notched_holed_plate/` | COMSOL holed-plate fracture | `configs/benchmarks/quasistatic/QS_notched_holed_plate_comsol_strict.yaml` |
| `galvis_validation/` | Galvis et al. cross-check helpers | comparator-only until the external reference curves are promoted |

## Reproducible Runs

Quasistatic YAMLs live under `configs/benchmarks/quasistatic/`.

Direct problem configs are runnable through the package CLI:

```bash
python -m phast run configs/benchmarks/quasistatic/QS_notched_holed_plate_comsol_strict.yaml
python -m phast run configs/benchmarks/quasistatic/QS_lshaped_concrete.yaml
```

SENT / SENS / TPB continue to use dedicated run scripts and manifests
because they have special post-processing and optional continuation workflows:

```bash
python -u examples/quasistatic/miehe_tension/run.py --help
python -u examples/quasistatic/miehe_shear/run.py --help
python -u examples/quasistatic/three_point_bending/run.py --help
```

Use the manifest files for the exact accepted/HPC argument lists:

- `configs/benchmarks/quasistatic/manifests/QS_sens_tpb_peak_window_corrected.yaml`
- `configs/benchmarks/quasistatic/manifests/QS_mesh_convergence_arc_length.yaml`
- `configs/benchmarks/quasistatic/manifests/QS_sens_tpb_rescue_visuals.yaml`
- `configs/benchmarks/quasistatic/manifests/QS_miehe_tension_repro.yaml`
- `configs/benchmarks/quasistatic/manifests/QS_miehe_sens_repro.yaml`
- `configs/benchmarks/quasistatic/manifests/QS_miehe_tpb_repro.yaml`

### Reproducible entry points

| Benchmark | Primary config / manifest | Run command |
|---|---|---|
| Miehe SENT (PhaseFieldX 1711) | `configs/benchmarks/quasistatic/manifests/QS_mesh_convergence_arc_length.yaml` | `scripts/slurm/benchmarks/hpc_qs_mesh_convergence_arc.slurm` |
| Miehe SENS (PhaseFieldX 1712) | `configs/benchmarks/quasistatic/manifests/QS_sens_tpb_peak_window_corrected.yaml` | `scripts/slurm/benchmarks/hpc_qs_sens_tpb_peak_window.slurm` |
| Miehe TPB (PhaseFieldX 1714) | `configs/benchmarks/quasistatic/manifests/QS_sens_tpb_peak_window_corrected.yaml` | `scripts/slurm/benchmarks/hpc_qs_tpb_postpeak.slurm` |
| Miehe SENT (local repro) | `configs/benchmarks/quasistatic/manifests/QS_miehe_tension_repro.yaml` | `python -u examples/quasistatic/miehe_tension/run.py` |
| Miehe SENS (local repro) | `configs/benchmarks/quasistatic/manifests/QS_miehe_sens_repro.yaml` | `python -u examples/quasistatic/miehe_shear/run.py` |
| Miehe TPB (local repro) | `configs/benchmarks/quasistatic/manifests/QS_miehe_tpb_repro.yaml` | `python -u examples/quasistatic/three_point_bending/run.py` |
| Notched holed plate (COMSOL) | `configs/benchmarks/quasistatic/QS_notched_holed_plate_comsol_strict.yaml` | `python -m phast run configs/benchmarks/quasistatic/QS_notched_holed_plate_comsol_strict.yaml` |
| L-shaped panel | `configs/benchmarks/quasistatic/QS_lshaped_concrete.yaml` | `python -m phast run configs/benchmarks/quasistatic/QS_lshaped_concrete.yaml` |
| Galvis helper set | example-specific scripts | `python -u examples/quasistatic/galvis_validation/run.py` |

## Reference packages in-repo (lightweight promotion)

| Benchmark | Promoted package |
|---|---|
| `miehe_tension` | `examples/quasistatic/miehe_tension/reference_runs/qs_sent_41278_coarse` and `qs_sent_41278_medium` |
| `miehe_shear` | none in-repo (run via manifest or direct local run folder) |
| `three_point_bending` | none in-repo (run via manifest or direct local run folder) |
| `notched_holed_plate` | none in-repo (compare against `hpc_results` or local output) |
| `l_shaped_panel` | pending revalidation; no in-repo lightweight package currently promoted |

Do not place raw run dumps directly in `papers/` or `examples/`. Promote only
small, reviewed evidence folders and manuscript-ready figures.
