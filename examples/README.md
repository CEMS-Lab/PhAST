# PhAST Examples Gallery

This directory contains runnable PhAST examples for dynamic fracture,
quasi-static fracture, solid mechanics, beta plasticity/interface workflows,
and a placeholder for future inverse-analysis examples. The promoted examples
are designed to be reproducible, self-contained, and easy to inspect on GitHub.

For Paper-1 reproduction, start with the dynamic and quasi-static fracture
examples. Solid-mechanics folders provide supporting solver checks, while
plasticity/interface folders are beta validation material with a narrower
capability claim. `inverse_problems_beta/` is currently a README-only landing
zone, not a promoted runnable benchmark.

Most public examples use declarative YAML input configurations. Validate and run them
from the repository root:

```bash
python -m phast run examples/<family>/<case>/config.yaml --validate-only
python -m phast run examples/<family>/<case>/config.yaml --output_dir runs/<case>
```

Use the fluent `phast.Problem` API when authoring new models. Where an example
includes `run_fluent.py`, that script shows the equivalent manual Python setup;
`config.yaml` remains the primary reproducibility configuration file.

For detailed artifact and promotion rules, see the
[example contract](../docs/user_guide/example_contract.md).

## Dynamic Fracture

Dynamic examples use the explicit-dynamics YAML runner for impact, branching,
and rapid crack-growth scenarios.

| Example | Status | Physics / Scenario | Command |
| :--- | :--- | :--- | :--- |
| [`B2_kalthoff_winkler`](dynamic/B2_kalthoff_winkler/) | Paper-1 public fracture example | Kalthoff-Winkler impact | `python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml` |
| [`B3_dynamic_sent`](dynamic/B3_dynamic_sent/) | Paper-1 baseline verification | Dynamic single-edge-notched tension baseline verification case | `python -m phast run examples/dynamic/B3_dynamic_sent/config.yaml` |
| [`B5_pmma_branching`](dynamic/B5_pmma_branching/) | Paper-1 public fracture example | PMMA dynamic crack branching | `python -m phast run examples/dynamic/B5_pmma_branching/config.yaml` |
| [`B6_perforated_30holes`](dynamic/B6_perforated_30holes/) | Public fracture extension | Perforated PMMA plate | `python -m phast run examples/dynamic/B6_perforated_30holes/config.yaml` |
| [`B7_dynamic_crack_branching_comsol`](dynamic/B7_dynamic_crack_branching_comsol/) | External comparison example | Dynamic branching comparison case | `python -m phast run examples/dynamic/B7_dynamic_crack_branching_comsol/config.yaml` |

## Quasi-Static Fracture

These examples cover stable crack growth under slowly applied loads and include
comparison plots plus CSV histories.

| Example | Physics / Scenario | Validation | Command |
| :--- | :--- | :--- | :--- |
| [`miehe_tension`](quasistatic/miehe_tension/) | Miehe single-edge-notched tension | PASS against PhaseFieldX-style reference | `python -m phast run examples/quasistatic/miehe_tension/config.yaml --output_dir runs/miehe_tension` |
| [`notched_holed_plate`](quasistatic/notched_holed_plate/) | Notched plate with holes | PASS strict-parity comparison | `python -m phast run examples/quasistatic/notched_holed_plate/config.yaml --output_dir runs/notched_holed_plate` |

## Solid Mechanics

These examples are mesh-level finite-element simulations through the common
YAML runner.

| Example | Status | Physics / Scenario | Command |
| :--- | :--- | :--- | :--- |
| [`linear_plate`](solid_mechanics_beta/linear_plate/) | Supporting solver example | Linear elastic cantilever with sparse autograd solve | `python -m phast run examples/solid_mechanics_beta/linear_plate/config.yaml` |
| [`neohookean_plate`](solid_mechanics_beta/neohookean_plate/) | Supporting solver example | Nonlinear neo-Hookean cantilever | `python -m phast run examples/solid_mechanics_beta/neohookean_plate/config.yaml` |
| [`j2_bar`](solid_mechanics_beta/j2_bar/) | Beta material-model example | Mesh-level J2 plasticity bar | `python -m phast run examples/solid_mechanics_beta/j2_bar/config.yaml` |

For the YAML-first solid examples, `run_fluent.py` provides the equivalent
manual Python setup:

```bash
python examples/solid_mechanics_beta/linear_plate/run_fluent.py
python examples/solid_mechanics_beta/neohookean_plate/run_fluent.py
python examples/solid_mechanics_beta/j2_bar/run_fluent.py
```

## Beta Plasticity And Interface

All folders and retained results under `examples/plasticity_interface_beta/` are
beta validation workflows for J2 plasticity, cohesive/interface operators,
PF-CZM calibration, and diffuse interphase screening. They are not Paper-1
production fracture examples. Their reproducibility manifest is:

```text
configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml
```

Run retained validations through the dispatcher:

```bash
python -m phast run configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml --validation-id j2_validation
python -m phast run configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml --validation-id structural_dcb_cohesive
python -m phast run configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml --validation-id structural_dcb_refinement
python -m phast run configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml --validation-id pfczm_uniaxial_strength
```

See [`plasticity_interface/README.md`](plasticity_interface_beta/) for the full beta
capability boundary and direct script commands.

## Inverse Problems Beta

[`inverse_problems_beta/`](inverse_problems_beta/) is a placeholder for future
differentiable inverse-analysis examples. Do not treat it as a promoted
runnable example until a configuration, loss definition, retained lightweight
outputs, and validation notes are added.

## Common CLI Flags

| Flag | Output |
| :--- | :--- |
| `--validate-only` | Check the YAML configuration without launching the solve. |
| `--output_dir DIR` | Write results to a custom directory. |
| `--plots` | Generate PNG figures when supported. |
| `--gif` | Generate an animated GIF when supported. |
| `--vtu` | Write VTU snapshots for ParaView. |
| `--trajectory --trajectory-format zarr` | Write a Zarr trajectory store for inspection or reuse. |
| `--device cpu/cuda` | Select the compute device. |

## Expected Folder Contents

Public example folders are intentionally compact. Typical files include:

- `README.md` with the physics, commands, and result summary.
- `config.yaml` for YAML-first examples.
- `run_fluent.py` when an equivalent Python setup is provided.
- `mesh.geo` for mesh-based examples when a Gmsh recipe is available.
- `initial_conditions.png`, final field plots, response plots, and lightweight animations.
- `response.csv`, `results.csv`, `history.csv`, or other small numerical evidence.
- `run_manifest.json` and `visual_manifest.json` when generated by the runner.

Large raw trajectories, scratch run directories, and machine-specific logs are
not part of the lightweight example folders. Generate fresh results with
`--output_dir runs/<case>` when you need full local output for inspection.
