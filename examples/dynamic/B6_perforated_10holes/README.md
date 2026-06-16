# B6 Perforated 10-Hole Plate

Curated PMMA perforated-plate dynamic fracture variant with ten holes.

Run:

```bash
python -m phast run examples/dynamic/B6_perforated_10holes/config.yaml
```

Public bundle status: candidate. This folder is flat by design and contains the YAML input deck plus curated lightweight outputs from HPC job 8585. Raw trajectories and raw HPC folders are intentionally not included in the public examples tree.

Expected public visuals are `initial_conditions.png`, `thumbnail.png`, `damage_multipanel.png`, `damage_final.png`, and diagnostic response plots listed in `visual_manifest.json`. The retained CSV files are `history.csv`, `energy.csv`, and `crack_tip.csv`.
