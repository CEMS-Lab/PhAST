# J2 Bar

Mesh-level displacement-controlled J2/von-Mises plasticity with linear isotropic hardening. The example uses PhAST's `FEMMesh`, `MeshJ2Elastoplasticity`, and `SparseJ2QuasiStaticSolver` path to solve a mildly waisted bar problem and writes stress/plastic-strain fields.

Run from the repository root:

YAML-first journey: validate the checked-in deck, run
`python -m phast run examples/solid_mechanics/j2_bar/config.yaml`, and inspect
the standard result directory. YAML is canonical for this example until the
fluent path is promoted because the current public example uses the
plasticity-specific mesh J2 runner surface and should not be presented as a
generic fluent material compiler.

```bash
python -m phast run examples/solid_mechanics/j2_bar/config.yaml
```

The direct script wrapper is still supported:
`python examples/solid_mechanics/j2_bar/run.py --config examples/solid_mechanics/j2_bar/config.yaml`.

`fluent_setup.py` documents and validates the public `phast.Problem` authoring
shape for this promoted runner while keeping `config.yaml` as the canonical
execution deck.

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
- `equivalent_plastic_strain.png`
- `plastic_strain_final.png`
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
