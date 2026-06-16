# Generalized-Alpha Oscillator

Two-degree-of-freedom oscillator integrated with the Hulbert-Chung generalized-alpha method. The example shows high-frequency numerical dissipation for `rho_inf = 0.5` while preserving the low-frequency mode.

Run from the repository root:

```bash
python examples/solid_mechanics/generalized_alpha_oscillator/run.py
```

Promoted outputs are checked in flat beside the config. Use `--output_dir` for
scratch reruns when you do not want to overwrite the promoted bundle:

- `response.csv`
- `response.png`
- `initial_conditions.png`
- `thumbnail.png`
- `visual_manifest.json`
- `run_manifest.json`
