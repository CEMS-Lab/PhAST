# B6 Perforated 1-Hole Near Plate

Curated PMMA dynamic fracture variant with a single near crack-path hole.

Run:

```bash
python -m phast run examples/dynamic/B6_perforated_1hole_near/config.yaml
```

Public bundle status: candidate. This folder is flat by design and contains the YAML input deck plus curated lightweight outputs from reviewed reference run 8585. Raw trajectories and full run directories are intentionally not included in the public examples tree.

Expected public visuals are `initial_conditions.png`, `thumbnail.png`, `damage_multipanel.png`, `damage_final.png`, and diagnostic response plots listed in `visual_manifest.json`. The retained CSV files are `history.csv`, `energy.csv`, and `crack_tip.csv`.
