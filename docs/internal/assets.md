# Visual Assets

This folder contains small, curated images for the repository README and
documentation. They are lightweight presentation assets, not raw benchmark
outputs or trajectory stores. Keep large run folders, proprietary files, heavy
Zarr/H5 stores, and paper-specific evidence packs out of the public repository.

## Panels

| File | Capability | Source |
|---|---|---|
| `kalthoff_winkler_long_crack.gif` | Dynamic phase-field fracture | Lightweight Kalthoff-Winkler fixed-viewport README GIF |
| `b7_crack_branching_evolution.gif` | Dynamic crack branching | Lightweight B7 dynamic branching damage-evolution GIF |
| `perforated_microstructure_damage.png` | Microstructured fracture | Curated perforated-plate damage visual |
| `qs_notched_holed_damage.png` | Quasi-static implicit fracture | Curated retained notched-holed-plate output |
| `qs_force_displacement.png` | Standard engineering outputs | Curated retained notched-holed-plate force-displacement output |
| `solid_mechanics_materials.png` | Solid-mechanics material kernels | Generated material-kernel showcase for linear-elastic, neo-Hookean, and J2 responses |
| `dynamic_timing_comparison.png` | Dynamic fracture timing | Lightweight timing comparison panel; companion data in `dynamic_timing_comparison.csv` |

## Update Rule

When replacing a panel, keep the file lightweight enough for README rendering
and update this provenance table in the same change. Avoid adding raw result
packs, full cluster result folders, proprietary solver files, or long videos
directly to the repository.
