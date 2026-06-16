# Dynamic Fracture

Dynamic fracture examples use YAML input decks for explicit phase-field runs.
They are useful for crack propagation, impact, branching, and dataset-style
trajectories, but each example carries its own status in the public contract.

## Start here

```bash
python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml --validate-only
python -m phast run examples/dynamic/B3_dynamic_sent/config.yaml --validate-only
```

For full production reruns, prefer CPU nodes when the GPU queue is busy and the
case fits CPU memory; otherwise use CUDA for large trajectory-producing runs.
Keep `dtype: float64` where the validation deck requires it.

## Expected outputs

Promoted dynamic examples should include `run_manifest.json`,
`visual_manifest.json`, `initial_conditions.png`, `thumbnail.png`, damage
snapshots or multipanels, `history.csv`, and `energy.csv`. Examples with
reference comparisons should also include `compare.png` and
`compare_report.txt`.

## Boundary

B1 branching remains a placeholder until the accepted quad rerun is promoted.
B7 is a public candidate without COMSOL binaries or raw trajectory stores.
Large Zarr/H5 trajectories belong in private storage or release artifacts, not
inside lightweight public example folders.
