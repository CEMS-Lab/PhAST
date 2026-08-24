# PhAST Examples Gallery

This directory contains the runnable PhAST examples grouped by physics family:
dynamic fracture, quasi-static fracture, solid mechanics, heterogeneous-field teaching, beta
plasticity/interface workflows, and a documentation-only location for possible
future inverse-analysis
examples. Each public example is kept compact so that a new user can inspect
the input deck, run it, and compare the output bundle without searching through
solver internals.

Start with the dynamic and quasi-static fracture examples if you are learning
the solver. Solid-mechanics folders provide supporting finite-element checks,
while plasticity/interface folders are beta validation workflows with a narrower
evidence boundary. `inverse_problems_beta/` contains documentation only and is
not a runnable inverse-analysis example.

Most public examples use declarative YAML input configurations. Validate and
run them from the repository root:

```bash
python -m phast run examples/<family>/<case>/config.yaml --validate-only
python -m phast run examples/<family>/<case>/config.yaml --output_dir runs/<case>
```

Use the fluent `phast.Problem` API when authoring new models. Where an example
includes `run_fluent.py`, that script shows the equivalent Python setup;
`config.yaml` remains the primary reproducibility configuration file.

For detailed artifact and curation rules, see the
[example contract](../docs/user_guide/example_contract.md).

## Dynamic Fracture

Dynamic examples use the explicit-dynamics YAML runner for impact, branching,
and rapid crack-growth scenarios.

| Example | Status | Physics / Scenario | Command |
| :--- | :--- | :--- | :--- |
| [`B2_kalthoff_winkler`](dynamic/B2_kalthoff_winkler/) | Public fracture example | Kalthoff-Winkler impact | `python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml` |
| [`B3_dynamic_sent`](dynamic/B3_dynamic_sent/) | Public fracture baseline | Dynamic single-edge-notched tension | `python -m phast run examples/dynamic/B3_dynamic_sent/config.yaml` |
| [`B5_pmma_branching`](dynamic/B5_pmma_branching/) | Public fracture example | PMMA dynamic crack branching | `python -m phast run examples/dynamic/B5_pmma_branching/config.yaml` |
| [`B6_perforated_30holes`](dynamic/B6_perforated_30holes/) | Public fracture extension | Perforated PMMA plate | `python -m phast run examples/dynamic/B6_perforated_30holes/config.yaml` |
| [`B7_dynamic_crack_branching_comsol`](dynamic/B7_dynamic_crack_branching_comsol/) | Reference comparison example | Dynamic branching comparison case | `python -m phast run examples/dynamic/B7_dynamic_crack_branching_comsol/config.yaml` |

## Quasi-Static Fracture

These examples cover stable crack growth under slowly applied loads and include
comparison plots plus CSV histories.

| Example | Physics / Scenario | Validation | Command |
| :--- | :--- | :--- | :--- |
| [`miehe_tension`](quasistatic/miehe_tension/) | Single-edge-notched tension | Public comparison reference | `python -m phast run examples/quasistatic/miehe_tension/config.yaml --output_dir runs/miehe_tension` |
| [`notched_holed_plate`](quasistatic/notched_holed_plate/) | Notched plate with holes | Public comparison reference | `python -m phast run examples/quasistatic/notched_holed_plate/config.yaml --output_dir runs/notched_holed_plate` |

## Solid Mechanics

These examples are mesh-level finite-element simulations through the common
YAML runner.

| Example | Status | Physics / Scenario | Command |
| :--- | :--- | :--- | :--- |
| [`linear_plate`](solid_mechanics_beta/linear_plate/) | Supporting solver example | Linear elastic cantilever | `python -m phast run examples/solid_mechanics_beta/linear_plate/config.yaml` |
| [`neohookean_plate`](solid_mechanics_beta/neohookean_plate/) | Supporting solver example | Nonlinear neo-Hookean cantilever | `python -m phast run examples/solid_mechanics_beta/neohookean_plate/config.yaml` |
| [`j2_bar`](solid_mechanics_beta/j2_bar/) | Beta material-model example | Mesh-level J2 plasticity bar | `python -m phast run examples/solid_mechanics_beta/j2_bar/config.yaml` |

For the YAML-first solid examples, `run_fluent.py` provides the equivalent
manual Python setup:

```bash
python examples/solid_mechanics_beta/linear_plate/run_fluent.py
python examples/solid_mechanics_beta/neohookean_plate/run_fluent.py
python examples/solid_mechanics_beta/j2_bar/run_fluent.py
```

## Heterogeneous Material Fields

[`heterogeneous_fields/`](heterogeneous_fields/) is a script-contract teaching
example for element-ordered `E(x)` and `Gc(x)` arrays. It is intentionally not a
general YAML workflow:

```bash
python examples/heterogeneous_fields/run.py \
  --config examples/heterogeneous_fields/parameters.yaml \
  --output-dir runs/heterogeneous_fields
```

The example solves a bounded AT2 damage subproblem under an imposed affine
strain field. It does not claim coupled microstructure-fracture validation.

## Beta Plasticity And Interface

All folders and validation bundles under `examples/plasticity_interface_beta/`
are beta workflows for J2 plasticity, cohesive/interface operators, PF-CZM
calibration, and diffuse interphase screening. Their reproducibility manifest
is:

```text
configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml
```

Run the validation workflows through the dispatcher:

```bash
python -m phast run configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml --validation-id j2_validation
python -m phast run configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml --validation-id structural_dcb_cohesive
python -m phast run configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml --validation-id structural_dcb_refinement
python -m phast run configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml --validation-id pfczm_uniaxial_strength
```

See [`plasticity_interface/README.md`](plasticity_interface_beta/) for the full
beta capability boundary and direct script commands.

## Inverse Problems Beta

[`inverse_problems_beta/`](inverse_problems_beta/) documents requirements for a
possible future differentiable inverse-analysis example. It must not be
described as runnable unless a configuration, loss definition, lightweight
outputs, and validation notes are added.

## Common CLI Flags

| Flag | Output |
| :--- | :--- |
| `--validate-only` | Check YAML schema and semantic consistency without launching the solve; not scientific or mesh-convergence validation. |
| `--output_dir DIR` | Write results to a custom directory. |
| `--plots` | Generate PNG figures when supported. |
| `--gif` | Generate an animated GIF when supported. |
| `--trajectory --trajectory-format zarr` | Write a Zarr trajectory store for inspection or reuse. |
| `--device cpu/cuda` | Select the compute device. |

## Expected Folder Contents

Public example folders are intentionally compact. Typical files include:

- `README.md` with the physics, commands, and result summary.
- `config.yaml` for YAML-first examples, or a clearly labelled parameter file and entry-point script for script-contract examples.
- `run_fluent.py` when an equivalent Python setup is provided.
- `mesh.geo` for mesh-based examples when a Gmsh recipe is available.
- `initial_conditions.png`, final field plots, response plots, and lightweight animations.
- `response.csv`, `results.csv`, `history.csv`, or other small numerical evidence.
- `run_manifest.json` and `visual_manifest.json` when generated by the runner.

Large raw trajectories, scratch run directories, and machine-specific logs are
not part of the lightweight example folders. Generate fresh results with
`--output_dir runs/<case>` when you need full local output for inspection.

If an example command, YAML field, or expected output is unclear, open a
[GitHub issue](https://github.com/CEMS-Lab/PhAST/issues/new/choose). Questions
from students and first-time users are welcome and help identify missing
documentation.
