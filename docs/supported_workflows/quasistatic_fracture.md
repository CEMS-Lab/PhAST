# Quasi-static Fracture

Quasi-static fracture is the main public validation path for implicit
phase-field examples. Curated cases are listed in
`examples/PUBLIC_EXAMPLES_CONTRACT.yaml` and follow the flat example contract.

## Start here

```bash
python -m phast run examples/quasistatic/miehe_tension/config.yaml --validate-only
python -m phast run examples/quasistatic/miehe_tension/config.yaml --output_dir runs/miehe_tension
python -m phast run examples/quasistatic/notched_holed_plate/config.yaml --output_dir runs/notched_holed_plate
```

Use `python -m phast explain-config <config.yaml>` before launching longer
runs. For HPC or batch work, submit the same YAML configuration and keep the resulting
metadata, manifests, and comparison artifacts together.

## Expected outputs

The curated examples produce setup previews, CSV histories, damage fields,
comparison artifacts when a reference exists, and visual manifests. Required
files include `initial_conditions.png`, `results.csv`, `history.csv`,
`energy.csv`, `solver_telemetry.csv`, `timing_per_step.csv`,
`damage_final.png`, `damage_evolution.gif`, `compare.png`, and
`compare_report.txt` where applicable.

## Boundary

Do not treat deferred SENS, TPB, or L-shaped panel material as public examples
until those folders have the same YAML, visual, manifest, and test coverage as
the curated cases.
