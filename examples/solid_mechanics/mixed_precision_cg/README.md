# Mixed-Precision CG

Conjugate-gradient numerics demo on a sparse one-dimensional Laplacian. The example compares float64, float32, and mixed-precision solve behavior.

Run from the repository root:

```bash
python examples/solid_mechanics/mixed_precision_cg/run.py
```

Promoted outputs are checked in flat beside the config. Use `--output_dir` for
scratch reruns when you do not want to overwrite the promoted bundle:

- `response.csv`
- `response.png`
- `initial_conditions.png`
- `thumbnail.png`
- `visual_manifest.json`
- `run_manifest.json`
