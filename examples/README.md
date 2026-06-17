# PhAST Examples Gallery

Welcome to the PhAST examples gallery! This directory contains runnable, validated simulations demonstrating PhAST's capabilities across dynamic fracture, quasi-static fracture, and foundational solid mechanics.

Our examples are designed to be **reproducible, self-contained, and auditable**. We prioritize declarative YAML configurations (`config.yaml`) for most public examples. This allows you to easily validate inputs, run simulations locally, submit them to HPC clusters, and share exact configurations.

*For detailed policies on example promotion and output contracts, see the [Example Contract](../docs/user_guide/example_contract.md).*

---

## 🚀 Quick Start

You can launch any YAML-based example from the repository root using the PhAST CLI:

```bash
# Run a full simulation and save outputs to a specific directory
python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml --output_dir runs/kalthoff_winkler

# Quickly validate a configuration without running the compute-heavy solve
python -m phast run examples/quasistatic/notched_holed_plate/config.yaml --validate-only
```

---

## 🗂️ Example Categories

### 1. Dynamic Fracture
Dynamic examples are flat YAML packages. They use the explicit-dynamics YAML runner to simulate rapid crack propagation and impact scenarios.

| Example | Physics / Scenario | Command |
| :--- | :--- | :--- |
| `B2_kalthoff_winkler` | Kalthoff-Winkler impact | `python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml` |
| `B3_dynamic_sent` | Dynamic SENT smoke/example | `python -m phast run examples/dynamic/B3_dynamic_sent/config.yaml` |
| `B5_pmma_branching` | PMMA branching selected sweep | `python -m phast run examples/dynamic/B5_pmma_branching/config.yaml` |
| `B6_perforated_30holes` | Perforated PMMA plate | `python -m phast run examples/dynamic/B6_perforated_30holes/config.yaml` |
| `B7_dynamic_crack_branching_comsol` | Dynamic branching cross-check | `python -m phast run examples/dynamic/B7_dynamic_crack_branching_comsol/config.yaml` |

### 2. Quasi-Static Fracture
These examples focus on stable crack growth under slowly applied loads. *Note: Authoring snippets can be found in individual example folders, but the checked-in `config.yaml` remains the canonical public input deck.*

| Example | Physics / Scenario | Command |
| :--- | :--- | :--- |
| [`miehe_tension`](quasistatic/miehe_tension/) | Miehe SENT tension | `python -m phast run examples/quasistatic/miehe_tension/config.yaml` |
| [`notched_holed_plate`](quasistatic/notched_holed_plate/) | COMSOL notched-holed plate | `python -m phast run examples/quasistatic/notched_holed_plate/config.yaml` |

### 3. Solid Mechanics
Foundational FEA simulations. The first three run through the common YAML runner, while the final two are numerical-method diagnostic scripts.

| Example | Physics / Scenario | Command |
| :--- | :--- | :--- |
| `linear_plate` | Linear elastic cantilever | `python -m phast run examples/solid_mechanics/linear_plate/config.yaml` |
| `neohookean_plate` | Nonlinear neo-Hookean cantilever | `python -m phast run examples/solid_mechanics/neohookean_plate/config.yaml` |
| `j2_bar` | Mesh-level J2 plasticity bar | `python -m phast run examples/solid_mechanics/j2_bar/config.yaml` |
| `mixed_precision_cg` | Krylov precision diagnostic | `python examples/solid_mechanics/mixed_precision_cg/run.py` |
| `generalized_alpha_oscillator` | Time-integration diagnostic | `python examples/solid_mechanics/generalized_alpha_oscillator/run.py` |

*(Note: Legacy solid FEA script entrypoints remain available as compatibility wrappers, e.g., `python examples/solid_mechanics/linear_plate/run.py --config ...`)*

### 4. Plasticity & Interface (Beta)
The `plasticity_interface/` folder contains beta workflows that use Python script contracts instead of YAML. The canonical reproducibility manifest for this family is:
```text
configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml
```
These workflows test capabilities (like J2 material validation and cohesive operators) that are not yet fully expressible through the generic `phast run` YAML runner. Once standardized, they will be promoted to the flat YAML gallery.

---

## 🛠️ Common CLI Flags

Customize your simulation outputs using standard CLI flags:

| Flag | Description |
| :--- | :--- |
| `--plots` | Generates PNG figures (when supported by the YAML runner). |
| `--vtu` | Exports VTU snapshots for visualization in ParaView. |
| `--gif` | Creates an animated GIF showing damage evolution. |
| `--trajectory --trajectory-format zarr` | Stores reusable solver outputs in Zarr trajectory format. |
| `--h5` | *Legacy compatibility* trajectory output (prefer Zarr for new work). |
| `--all_outputs` | Generates VTU, GIF, plots, and profiler data (does not imply H5). |
| `--device cpu/cuda` | Sets compute device (e.g., use `cpu` on Mac for float64 solves). |
| `--output_dir DIR` | Overrides the default custom output directory. |

A standard YAML run typically produces `config.yaml`, `run_lockfile.json`, `run_metadata.json`, scalar CSVs, and requested visual artifacts in the output directory.

---

## 🤝 For Contributors: Adding or Promoting an Example

To contribute a new example or promote an existing one to the public gallery, follow these guidelines:

1. **Keep it flat:** Place the example in a flat leaf folder containing a `README.md`, the `config.yaml` (or `run.py`), and lightweight output assets.
2. **Include manifests:** Provide a `run_manifest.json`, and if plots/animations are generated, include a `visual_manifest.json`.
3. **Update contracts:** Add the new example to `PUBLIC_EXAMPLES_CONTRACT.yaml`.
4. **Add tests:** Ensure tests are updated so the contract fails if the example's output drifts over time.
5. **Keep heavy data private:** Maintain raw HPC data, large Zarr/H5 stores (like the 98 GB Zarr trajectories), and paper-specific scratch work in the private archive. Only lightweight artifacts belong in the public PhAST export.
