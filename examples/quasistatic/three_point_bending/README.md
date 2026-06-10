# Quasistatic Three-Point Bending

Canonical artifact folder for the Miehe TPB benchmark.

Run locally:

```bash
python -u examples/quasistatic/three_point_bending/run.py \
  --backend auto --preconditioner jacobi --all_outputs \
  --output_dir examples/quasistatic/three_point_bending/run_local
```

Run comparison:

```bash
python -u examples/quasistatic/three_point_bending/compare.py \
  --run-dir examples/quasistatic/three_point_bending/run_local
```

Configurable variants are controlled through CLI:

- `--num_steps`, `--h_crack` for mesh schedule
- `--backend`, `--preconditioner` for mechanical solver
- `--trajectory` / `--zarr` via `--trajectory` + `--trajectory_format`
- `--animation_format mp4` for faster animation export
