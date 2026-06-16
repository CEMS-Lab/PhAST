# Solid Mechanics Examples

These examples are compact solid-mechanics workflows with a common public output contract. The first three are actual mesh-level finite-element simulations and now run through `python -m phast run <config.yaml>`; the remaining two are solver diagnostics retained for numerical-method checks. Each promoted FEA leaf folder contains `README.md`, `config.yaml`, `fluent_setup.py`, `run.py`, a compact Zarr trajectory, response animations, final field plots, and a flat promoted result bundle.

Run commands from the repository root.

| Example | Physics | Command | Outputs |
| --- | --- | --- | --- |
| Linear elastic plate | Plane-strain CST cantilever, sparse autograd solve, displacement and stress fields | `python -m phast run examples/solid_mechanics/linear_plate/config.yaml` | `response.csv`, final field PNGs, MP4 animations, `training_data.zarr`, manifests |
| Neo-Hookean cantilever | Load-stepped nonlinear hyperelastic FEA with Newton iterations | `python -m phast run examples/solid_mechanics/neohookean_plate/config.yaml` | `response.csv`, final field PNGs, MP4 animations, `training_data.zarr`, manifests |
| J2 plasticity bar | Mesh-level displacement-controlled von-Mises plasticity solve | `python -m phast run examples/solid_mechanics/j2_bar/config.yaml` | `response.csv`, stress/plastic-strain PNGs, MP4 animations, `training_data.zarr`, manifests |
| Mixed-precision CG | Sparse Krylov precision comparison diagnostic | `python examples/solid_mechanics/mixed_precision_cg/run.py` | `response.csv`, `response.png`, `thumbnail.png`, `run_manifest.json` |
| Generalized-alpha oscillator | Hulbert-Chung high-frequency dissipation diagnostic | `python examples/solid_mechanics/generalized_alpha_oscillator/run.py` | `response.csv`, `response.png`, `thumbnail.png`, `run_manifest.json` |

The old script paths are retained as compatibility wrappers:

```bash
python examples/solid_mechanics/linear_plate/run.py --config examples/solid_mechanics/linear_plate/config.yaml
python examples/solid_mechanics/neohookean_plate/run.py --config examples/solid_mechanics/neohookean_plate/config.yaml
python examples/solid_mechanics/j2_bar/run.py --config examples/solid_mechanics/j2_bar/config.yaml
python examples/solid_mechanics/linear_plate.py
python examples/solid_mechanics/neohookean_plate.py
python examples/solid_mechanics/j2_plasticity_bar.py
python examples/solid_mechanics/mixed_precision_cg_demo.py
python examples/solid_mechanics/dynamic_oscillator_genalpha.py
```

## Output Contract

The three promoted FEA examples write:

- `fluent_setup.py` with the equivalent public authoring setup.
- `response.csv` with numerical result data.
- `response.png` with the primary response visual.
- `deformed_shape.png`.
- `displacement_magnitude.png`.
- `displacement_final.png`.
- `von_mises.png`.
- `stress_final.png`.
- `equivalent_plastic_strain.png` for the J2 example, or `strain_energy.png` for elastic/hyperelastic examples.
- `strain_final.png`.
- `response_evolution.mp4` and `field_evolution.mp4`.
- `training_data.zarr` plus `zarr_manifest.json`.
- `thumbnail.png` for gallery use.
- `visual_manifest.json` with image/video dimensions, sizes, and review gate status.
- `run_manifest.json`, `run_metadata.json`, and `run_lockfile.json` with command, config, runtime, metrics, and output list.

The diagnostic examples keep the lighter `response.csv` / `response.png` /
`thumbnail.png` / `run_manifest.json` contract.
