# B2 Kalthoff-Winkler

Dynamic Kalthoff-Winkler impact example. This folder contains the runnable YAML
plus lightweight curated mesh-1 outputs. The private `training_data.h5` source
store is intentionally not copied into this public-candidate folder.

Run:

```bash
python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml
```

Expected public visuals are `initial_conditions.png`, `damage_final.png`,
`thumbnail.png`, and the CSV history files in this folder.
