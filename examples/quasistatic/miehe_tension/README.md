# Quasistatic Miehe Tension (SENT)

Canonical artifact folder for the Miehe single-edge-notched tension benchmark.

Run locally:

```bash
python -u examples/quasistatic/miehe_tension/run.py \
  --backend auto --preconditioner jacobi --all_outputs \
  --output_dir examples/quasistatic/miehe_tension/run_local
```

Run comparison:

```bash
python -u examples/quasistatic/miehe_tension/compare.py \
  --run-dir examples/quasistatic/miehe_tension/run_local
```

Recommended accepted setup for benchmark parity:

- `--h_crack 0.001875`
- `--num_steps 350` (or equivalent production sweep entrypoints)
- `--backend auto --preconditioner jacobi`
- `--trajectory --trajectory_format zarr --animation_format mp4` for diagnostics
