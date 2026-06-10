# README Visual Showcase

This folder contains small, curated images for the repository README. They
are documentation assets, not raw benchmark outputs. Keep large HPC mirrors,
paper result packs, videos, and generated run directories out of git unless a
specific result is promoted into a lightweight documentation panel here.

## Panels

| File | Capability | Source |
|---|---|---|
| `dynamic_sent_damage.png` | Dynamic phase-field fracture | `examples/dynamic/sent/reference_runs/B1_dynamic_sent/figures/damage_multipanel.png` |
| `perforated_microstructure_damage.png` | Microstructured fracture | `examples/dynamic/perforated_plate/reference_runs/B4a_perforated_30holes/figures/damage_multipanel.png` |
| `qs_notched_holed_damage.png` | Quasi-static implicit fracture | `docs/qs_hpc_results/job32465_notched_holed/run/figures/damage_multipanel.png` |
| `qs_force_displacement.png` | Standard engineering outputs | `docs/qs_hpc_results/job32465_notched_holed/run/figures/force_displacement.png` |
| `solid_mechanics_materials.png` | Solid-mechanics material kernels | Generated from the linear-elastic, neo-Hookean, and J2 demo equations/kernels |
| `autograd_cost.png` | Gradient verification and scaling | `docs/autograd_tutorial/figures/02_fd_vs_autograd_cost.png` |
| `differentiable_graph.png` | Differentiable forward pipeline | `docs/autograd_tutorial/figures/01_computational_graph.png` |
| `timing_comparison.png` | Benchmark performance | `examples/dynamic/kalthoff/timing_comparison/timing_comparison.png` |
| `generalized_alpha.png` | Implicit dynamics | `examples/solid_mechanics/dynamic_oscillator_genalpha.png` |
| `mixed_precision_cg.png` | Solver numerics | `examples/solid_mechanics/mixed_precision_cg_demo.png` |
| `solver_showcase_montage.png` | README summary graphic | Generated from the twelve panels above |

## Update Rule

When replacing a panel, keep the file lightweight enough for README rendering
and update this provenance table in the same change. Avoid adding raw Molinari
packs, full HPC result folders, or long videos directly to the repository.
