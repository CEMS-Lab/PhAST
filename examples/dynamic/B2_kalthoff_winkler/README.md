# B2 Kalthoff-Winkler

## What This Example Solves

Dynamic Kalthoff-Winkler impact example with a flat public bundle of curated
mesh-1 outputs. The checked-in YAML deck is the public reproduction path;
large trajectory stores are not distributed with the public repository.

## Run

Run commands from the repository root:

```bash
python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml --validate-only
python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml --output_dir runs/B2_kalthoff_winkler
```

## YAML And Manual Setup

`config.yaml` defines the geometry, material parameters, dynamic loading,
explicit solver controls, and requested public outputs. No equivalent
`run_fluent.py` is currently promoted for this example; use the YAML deck as
the public reproduction path.

## Promoted Result

| Initial conditions | Damage field |
| --- | --- |
| <img src="initial_conditions.png" width="360"> | <img src="damage_final.png" width="360"> |

Required public artifacts are `config.yaml`, `run_manifest.json`,
`visual_manifest.json`, `initial_conditions.png`, `thumbnail.png`,
`damage_final.png`, `damage_evolution.mp4`, `response_evolution.mp4`,
`history.csv`, `energy.csv`, `crack_tip.csv`, and `timing_per_step.csv`.

## Claim Boundary

This is a public candidate dynamic-fracture example with curated lightweight
outputs. Regenerate local trajectory stores from the YAML deck when field-level
post-processing is needed.
