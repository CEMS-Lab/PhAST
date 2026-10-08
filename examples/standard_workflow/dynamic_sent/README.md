# Dynamic single-edge-notched tension: setup exercise

This small input introduces the same configuration sequence with inertia,
explicit mechanical time integration, and an implicit damage update.
Its short duration checks execution; it does not establish a complete crack
trajectory or a mesh- and time-step-converged dynamic solution.

From the repository root:

```bash
python -m phast run examples/standard_workflow/dynamic_sent/config.yaml --validate-only
python -m phast run examples/standard_workflow/dynamic_sent/config.yaml --output_dir runs/first_dynamic
```

Edit [config.yaml](config.yaml), following the
[common instructions](../README.md). Density, loading time, mesh size, and
time-step stability all matter. Do not reuse a quasi-static load increment
as a physical time step without examining these choices.
