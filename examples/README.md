# PhAST Examples

This directory contains runnable PhAST examples for phase-field fracture,
solid mechanics, and beta plasticity/interface workflows. Each public example
keeps its inputs, lightweight reference outputs, and README in a flat folder so
the case can be inspected directly on GitHub and rerun from the repository
root.

Use YAML decks for reproducible examples:

```bash
python -m phast run examples/<family>/<case>/config.yaml --validate-only
python -m phast run examples/<family>/<case>/config.yaml --output_dir runs/<case>
```

Use the fluent `phast.Problem` API when authoring new models. Where a public
example includes `run_fluent.py`, that script shows the equivalent manual
Python setup while `config.yaml` remains the canonical reproducibility deck.

For the full artifact contract, see
[`docs/user_guide/example_contract.md`](../docs/user_guide/example_contract.md).

## Dynamic Fracture

Dynamic examples are YAML-first phase-field fracture cases with lightweight
visual and numerical evidence checked into each example folder.

| Example | Physics | Command |
| --- | --- | --- |
| [`dynamic/B2_kalthoff_winkler`](dynamic/B2_kalthoff_winkler/) | Kalthoff-Winkler impact | `python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml` |
| [`dynamic/B3_dynamic_sent`](dynamic/B3_dynamic_sent/) | Dynamic single-edge-notched tension smoke case | `python -m phast run examples/dynamic/B3_dynamic_sent/config.yaml` |
| [`dynamic/B5_pmma_branching`](dynamic/B5_pmma_branching/) | PMMA dynamic crack branching | `python -m phast run examples/dynamic/B5_pmma_branching/config.yaml` |
| [`dynamic/B6_perforated_30holes`](dynamic/B6_perforated_30holes/) | Perforated PMMA plate | `python -m phast run examples/dynamic/B6_perforated_30holes/config.yaml` |
| [`dynamic/B7_dynamic_crack_branching_comsol`](dynamic/B7_dynamic_crack_branching_comsol/) | Dynamic branching comparison case | `python -m phast run examples/dynamic/B7_dynamic_crack_branching_comsol/config.yaml` |

## Quasi-Static Fracture

These examples provide validated quasi-static fracture workflows with
comparison plots and CSV histories.

| Example | Physics | Validation | Command |
| --- | --- | --- | --- |
| [`quasistatic/miehe_tension`](quasistatic/miehe_tension/) | Miehe single-edge-notched tension | PASS against PhaseFieldX-style reference | `python -m phast run examples/quasistatic/miehe_tension/config.yaml --output_dir runs/miehe_tension` |
| [`quasistatic/notched_holed_plate`](quasistatic/notched_holed_plate/) | Notched plate with holes | PASS strict-parity comparison | `python -m phast run examples/quasistatic/notched_holed_plate/config.yaml --output_dir runs/notched_holed_plate` |

Validate before running a full solve:

```bash
python -m phast run examples/quasistatic/miehe_tension/config.yaml --validate-only
python -m phast run examples/quasistatic/notched_holed_plate/config.yaml --validate-only
```

## Solid Mechanics

The first three examples are mesh-level finite-element simulations through the
common YAML runner. The final two are compact numerical-method diagnostics.

| Example | Physics | Command |
| --- | --- | --- |
| [`solid_mechanics/linear_plate`](solid_mechanics/linear_plate/) | Linear elastic cantilever with sparse autograd solve | `python -m phast run examples/solid_mechanics/linear_plate/config.yaml` |
| [`solid_mechanics/neohookean_plate`](solid_mechanics/neohookean_plate/) | Nonlinear neo-Hookean cantilever | `python -m phast run examples/solid_mechanics/neohookean_plate/config.yaml` |
| [`solid_mechanics/j2_bar`](solid_mechanics/j2_bar/) | Mesh-level J2 plasticity bar | `python -m phast run examples/solid_mechanics/j2_bar/config.yaml` |
| [`solid_mechanics/mixed_precision_cg`](solid_mechanics/mixed_precision_cg/) | Krylov precision diagnostic | `python examples/solid_mechanics/mixed_precision_cg/run.py` |
| [`solid_mechanics/generalized_alpha_oscillator`](solid_mechanics/generalized_alpha_oscillator/) | Generalized-alpha time-integration diagnostic | `python examples/solid_mechanics/generalized_alpha_oscillator/run.py` |

For the YAML-first solid examples, `run_fluent.py` provides the equivalent
manual Python setup:

```bash
python examples/solid_mechanics/linear_plate/run_fluent.py
python examples/solid_mechanics/neohookean_plate/run_fluent.py
python examples/solid_mechanics/j2_bar/run_fluent.py
```

## Plasticity And Interface Beta

The plasticity/interface examples are beta validation workflows for J2
plasticity, cohesive/interface operators, PF-CZM calibration, and diffuse
interphase screening. Their reproducibility manifest is:

```text
configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml
```

Run a retained validation through the dispatcher:

```bash
python -m phast run configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml --validation-id j2_validation
python -m phast run configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml --validation-id structural_dcb_cohesive
python -m phast run configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml --validation-id structural_dcb_refinement
python -m phast run configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml --validation-id pfczm_uniaxial_strength
```

See [`plasticity_interface/README.md`](plasticity_interface/) for the full beta
capability boundary and direct script commands.

## Common CLI Flags

| Flag | Output |
| --- | --- |
| `--validate-only` | Check the YAML deck without launching the solve. |
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
