# Solid Mechanics Examples

These examples are compact solid-mechanics workflows with a common public output contract. The first three are mesh-level finite-element simulations and run through `python -m phast run <config.yaml>`; the remaining two are solver diagnostics retained for numerical-method checks. The J2 bar is marked beta because it exercises the developing plasticity material-model path. Each reference FEA leaf folder contains `README.md`, `config.yaml`, `run_fluent.py`, `mesh.geo`, `run.py`, response animations, final field plots, and a flat reference result bundle.

Run commands from the repository root.

| Example | Status | Physics | Command | Outputs |
| --- | --- | --- | --- | --- |
| Linear elastic plate | Supporting solver example | Plane-strain CST cantilever, sparse autograd solve, displacement and stress fields | `python -m phast run examples/solid_mechanics/linear_plate/config.yaml` | `response.csv`, final field PNGs, MP4 animations, manifests |
| Neo-Hookean cantilever | Supporting solver example | Load-stepped nonlinear hyperelastic FEA with Newton iterations | `python -m phast run examples/solid_mechanics/neohookean_plate/config.yaml` | `response.csv`, final field PNGs, MP4 animations, manifests |
| J2 plasticity bar | Beta material-model example | Mesh-level displacement-controlled von-Mises plasticity solve | `python -m phast run examples/solid_mechanics/j2_bar/config.yaml` | `response.csv`, stress/plastic-strain PNGs, MP4 animations, manifests |
| Mixed-precision CG | Numerical diagnostic | Sparse Krylov precision comparison diagnostic | `python examples/solid_mechanics/mixed_precision_cg/run.py` | `response.csv`, `response.png`, `thumbnail.png`, `run_manifest.json` |
| Generalized-alpha oscillator | Numerical diagnostic | Hulbert-Chung high-frequency dissipation diagnostic | `python examples/solid_mechanics/generalized_alpha_oscillator/run.py` | `response.csv`, `response.png`, `thumbnail.png`, `run_manifest.json` |

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

Each public leaf folder also includes a Python/manual entry point:

```bash
python examples/solid_mechanics/linear_plate/run_fluent.py
python examples/solid_mechanics/neohookean_plate/run_fluent.py
python examples/solid_mechanics/j2_bar/run_fluent.py
python examples/solid_mechanics/mixed_precision_cg/run_fluent.py
python examples/solid_mechanics/generalized_alpha_oscillator/run_fluent.py
```

## Output Contract

The three reference FEA examples write:

- `run_fluent.py` with the equivalent public Python/manual setup.
- `mesh.geo` with the public Gmsh geometry recipe for mesh-based examples.
- `response.csv` with numerical result data.
- `response.png` with the primary response visual.
- `deformed_shape.png`.
- `displacement_magnitude.png`.
- `displacement_final.png`.
- `von_mises.png`.
- `stress_final.png`.
- `equivalent_plastic_strain.png` for the J2 example, or `strain_energy.png` for elastic/hyperelastic examples.
- `strain_final.png`.
- `thumbnail.png` for gallery use.
- `visual_manifest.json` with image/video dimensions, sizes, and review gate status.
- `run_manifest.json` with command, config, runtime, metrics, and output list.

The diagnostic examples keep the lighter `response.csv` / `response.png` /
`thumbnail.png` / `run_manifest.json` contract and include `run_fluent.py` as
the public Python/manual runner. They do not include `mesh.geo` because they are
not mesh-generation examples.
