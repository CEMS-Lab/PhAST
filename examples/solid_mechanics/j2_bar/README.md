# J2 Bar

## What This Example Solves

Mesh-level displacement-controlled J2/von-Mises plasticity with linear isotropic hardening. The example solves a mildly waisted bar, tracks equivalent plastic strain, and writes stress/plastic-strain fields for the reference public run.

## Files

| File | Purpose |
| --- | --- |
| `config.yaml` | Canonical YAML deck for the public run. |
| `run_fluent.py` | Equivalent Python/manual authoring setup using `phast.Problem`. |
| `mesh.geo` | Public Gmsh recipe for the Gaussian-waisted bar geometry. |
| `run.py` | Compatibility wrapper used by the YAML runner. |
| `fluent_setup.py` | Legacy name retained for compatibility with earlier drafts. |
| `response.csv`, `response.png` | Stress-strain and plasticity response summary. |
| `initial_conditions.png` | Geometry and boundary-condition preview. |
| `deformed_shape.png`, `displacement_magnitude.png`, `displacement_final.png` | Displacement-field visual evidence. |
| `von_mises.png`, `stress_final.png`, `strain_final.png`, `equivalent_plastic_strain.png`, `plastic_strain_final.png` | Final stress, strain, and plasticity fields. |
| `response_evolution.mp4`, `field_evolution.mp4` | Lightweight reference animations. |
| `training_data.zarr`, `zarr_manifest.json` | Compact retained trajectory and manifest. |
| `thumbnail.png`, `visual_manifest.json` | Gallery thumbnail and visual QA metadata. |
| `run_manifest.json` | Reproducibility metadata for the reference output. |

## Run The Canonical YAML Deck

Run commands from the repository root:

```bash
python -m phast run examples/solid_mechanics/j2_bar/config.yaml --validate-only
python -m phast run examples/solid_mechanics/j2_bar/config.yaml
python examples/solid_mechanics/j2_bar/run_fluent.py
```

Use `--output_dir <path>` with `python -m phast run` for local reruns that should not overwrite the reference outputs.

## How The YAML Is Used

`mesh` defines the structured waisted bar, `material` defines the elastic constants, initial yield stress, and hardening modulus, and `loading` defines the number of displacement increments and maximum axial strain. The workflow lowers those blocks to the mesh J2 runner, which performs the elastoplastic solve and records the response and final plasticity fields.

## Run Without YAML

```bash
python examples/solid_mechanics/j2_bar/run_fluent.py --run --output-dir runs/j2_bar
```

`run_fluent.py` builds the same problem manually with `phast.Problem`: geometry, region, material, load step, solver selection, and requested outputs are all declared in Python before the workflow contract is validated.

## How Manual Setup Works

The manual setup mirrors the YAML fields directly: `.geometry(...)` maps to `mesh`, `.material(...)` maps to `material`, `.analysis_step(...)` maps to `loading`, `.solver(...)` selects `solid_mechanics.j2_bar`, and `.outputs(...)` requests the response and field artifacts.

## Reference Result

| Initial conditions | Plastic strain field |
| --- | --- |
| <img src="initial_conditions.png" width="360"> | <img src="equivalent_plastic_strain.png" width="360"> |

| Metric | Reference value |
| --- | ---: |
| Plastic steps | `25` |
| Final equivalent plastic strain | `3.278e-03` |
| Maximum von Mises stress | `325.65 MPa` |
| Maximum equivalent plastic strain | `1.513e-02` |
