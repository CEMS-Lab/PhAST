# Linear Elastic Plate

Plane-strain CST cantilever solved with PhAST's sparse autograd linear-solve path. The example compares the finite-element tip displacement against the Euler-Bernoulli beam estimate, differentiates the tip displacement with respect to Young's modulus, and writes standard FEA field visualisations.

Run from the repository root:

Recommended journey: inspect the fluent authoring shape when useful -> run the
checked-in YAML deck with
`python -m phast run examples/solid_mechanics/linear_plate/config.yaml` ->
standard result directory. The checked-in `config.yaml` is the canonical public
input deck for exact reproduction.

```bash
python -m phast run examples/solid_mechanics/linear_plate/config.yaml
```

The direct script wrapper is still supported:
`python examples/solid_mechanics/linear_plate/run.py --config examples/solid_mechanics/linear_plate/config.yaml`.

## Equivalent Fluent API

```python
import phast

result = (
    phast.Problem("Linear elastic plate")
    .geometry("structured_grid", nx=20, ny=10, length=1.0, height=0.2)
    .region("body", kind="domain")
    .material("steel", region="body", E=2.1e11, nu=0.3)
    .analysis_step("load", kind="solid_mechanics",
                   controls={"tip_force_y": -1.0e3})
    .solver("solid_mechanics", example="solid_mechanics.linear_plate")
    .outputs(fields=["displacement", "von_mises", "strain_energy"],
             histories=["response"], plots=True)
    .run(output_dir="runs/linear_plate", return_result=True)
)
```

The config.yaml is the canonical public input deck; the snippet documents
the equivalent fluent authoring surface and output intent.
The same authoring shape is available as `fluent_setup.py`.

Promoted outputs are checked in flat beside the config. Use `--output_dir` for
scratch reruns when you do not want to overwrite the promoted bundle:

- `config.yaml`
- `fluent_setup.py`
- `response.csv`
- `response.png`
- `initial_conditions.png`
- `deformed_shape.png`
- `displacement_magnitude.png`
- `displacement_final.png`
- `von_mises.png`
- `stress_final.png`
- `strain_energy.png`
- `strain_final.png`
- `response_evolution.mp4`
- `field_evolution.mp4`
- `training_data.zarr`
- `zarr_manifest.json`
- `thumbnail.png`
- `visual_manifest.json`
- `run_metadata.json`
- `run_lockfile.json`
- `run_manifest.json`
