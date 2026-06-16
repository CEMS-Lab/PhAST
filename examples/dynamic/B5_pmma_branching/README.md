# B5 PMMA Branching

Selected PMMA dynamic crack-path sweep from the Bleyer-style setup. This public
folder retains the representative runnable deck and curated visual-evidence
bundle; raw sweep archives and trajectory stores remain outside git.

Run:

```bash
python -m phast run examples/dynamic/B5_pmma_branching/config.yaml
```

Use `initial_conditions.png`, `damage_final.png`, `damage_multipanel.png`,
`damage_profiles_multi.png`, `space_time_diagram.png`, `compare.png`, and the
CSV files as the lightweight public visual/evidence set.

The archived metadata did not record an automatic `branch_step`; keep that
provenance visible in `run_manifest.json` and use the promoted damage figures as
the visual evidence rather than claiming a detected branch time.
