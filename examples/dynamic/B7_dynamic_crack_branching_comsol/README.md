# B7 Dynamic Crack Branching COMSOL Cross-Check

Accepted dynamic crack branching comparison package. This folder includes the
runnable YAML, curated PNG/CSV/report metadata, and excludes the COMSOL binary
model and vendor PDF.

Run:

```bash
python -m phast run examples/dynamic/B7_dynamic_crack_branching_comsol/config.yaml
```

Use `initial_conditions.png`, `thumbnail.png`, `damage_final.png`,
`damage_evolution.gif`, `damage_evolution.mp4`, `energy.png`, `compare.png`,
and `compare_report.txt` as the public evidence set. The animation was
regenerated from the retained reference trajectory for run
`b7_branching_47961`. The raw trajectory and internal branching montage are
not distributed with the public repository.
