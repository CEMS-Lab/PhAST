# Quasi-static L-Shaped Panel

Canonical artifact folder for the quasistatic L-shaped-panel benchmark.

Canonical problem config:

- `configs/benchmarks/quasistatic/QS_lshaped_concrete.yaml`

Run locally:

```bash
python -m phast run configs/benchmarks/quasistatic/QS_lshaped_concrete.yaml \
  --output_dir examples/quasistatic/l_shaped_panel/run_local
```

Compare outputs with:

```bash
python -u examples/quasistatic/l_shaped_panel/compare.py \
  --run-dir examples/quasistatic/l_shaped_panel/run_local
```
