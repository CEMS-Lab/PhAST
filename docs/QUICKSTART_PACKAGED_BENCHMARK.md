# Quickstart

Run a packaged benchmark with the same module path users get after
`pip install`:

```bash
python examples/quasistatic/miehe_tension/run.py \
  --backend auto --preconditioner jacobi --all_outputs
```

The driver writes plots, VTU/PV output, timing, and energy/history files
under the selected benchmark run directory.

For a longer walkthrough see [examples/solid_mechanics](examples/solid_mechanics.md).
