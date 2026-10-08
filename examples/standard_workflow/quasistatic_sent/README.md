# Quasi-static single-edge-notched tension: setup exercise

This small input introduces the standard configuration sequence for a
quasi-static phase-field calculation. It neglects inertia and uses a staggered
displacement-damage solution. The short loading sequence is an execution
exercise, not a fracture-validation result.

From the repository root:

```bash
python -m phast run examples/standard_workflow/quasistatic_sent/config.yaml --validate-only
python -m phast run examples/standard_workflow/quasistatic_sent/config.yaml --output_dir runs/first_quasistatic
```

Edit [config.yaml](config.yaml), following the
[common instructions](../README.md). Use a new output directory for each
variation and keep the configuration with the resulting fields and histories.
