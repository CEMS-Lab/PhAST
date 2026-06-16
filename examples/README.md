# PhAST Examples

This folder is the public-example staging area. The release boundary is
defined for humans in
[`docs/user_guide/example_contract.md`](../docs/user_guide/example_contract.md)
and recorded for tests in `PUBLIC_EXAMPLES_CONTRACT.yaml`. Customer-facing examples live only
in `dynamic/`, `quasistatic/`, `solid_mechanics/`, and
`plasticity_interface/`. Non-promoted tracked example folders are classified in
`EXAMPLES_SCOPE.yaml` and stored in a private development archive so private, legacy,
inverse, diagnostic, raw-HPC, and paper-specific material is not confused with
customer-runnable examples. Both files are checked by
`tests/test_public_examples_contract.py`.

Use the fluent `phast.Problem` API to author new models. Use YAML decks for public examples, reproducibility, batch/HPC runs, and sharing exact simulations.

The important rule is simple: public examples must be runnable, flat, and
auditable. Dynamic, quasi-static, and promoted solid-mechanics FEA examples
are YAML-first because `config.yaml` is the exact input deck users can
validate, run locally, submit on HPC, and compare in CI. Plasticity/interface
examples remain script-contract beta where the current capability is not yet
expressible through the generic YAML runner.

The canonical tier definitions live in
[`docs/user_guide/example_contract.md`](../docs/user_guide/example_contract.md).
This page lists the promoted folders and their user-facing commands without
duplicating the contract table.

Heavy trajectory stores (`training_data.zarr`, legacy `training_data.h5`), raw
HPC folders, private COMSOL binaries, old diagnostic sweeps, and inverse/hybrid
paper work do not belong in public PhAST examples. In this private development
repository they may be retained in the private archive for provenance; the
public CEMS-Lab/PhAST export must exclude that archive.

## Dynamic Fracture

Dynamic examples are flat YAML packages. Run them from the repository root.
They remain YAML-first until explicit-dynamics fluent/schema-v2 lowering is
validated for the public examples.

| Example | Physics | Status | Command |
| --- | --- | --- | --- |
| `dynamic/B2_kalthoff_winkler` | Kalthoff-Winkler impact | Public candidate, private H5 retained outside release | `python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml` |
| `dynamic/B3_dynamic_sent` | Dynamic SENT smoke/example | Qualitative smoke until finer result is promoted | `python -m phast run examples/dynamic/B3_dynamic_sent/config.yaml` |
| `dynamic/B5_pmma_branching` | PMMA branching selected sweep | Public candidate | `python -m phast run examples/dynamic/B5_pmma_branching/config.yaml` |
| `dynamic/B6_perforated_30holes` | Perforated PMMA plate | Public B6 name; old B4 source names stay private | `python -m phast run examples/dynamic/B6_perforated_30holes/config.yaml` |
| `dynamic/B7_dynamic_crack_branching_comsol` | Dynamic branching cross-check | Public candidate without COMSOL binary or 98 GB Zarr | `python -m phast run examples/dynamic/B7_dynamic_crack_branching_comsol/config.yaml` |

## Quasi-Static Fracture

Only the promoted quasi-static examples are public release examples right now.
Additional Miehe shear, three-point-bending, and L-shaped-panel work remains
private/deferred until it has the same flat contract.
Where the fluent surface is already promoted, the example README shows an
authoring snippet; the checked-in `config.yaml` remains the canonical public
input deck.

| Example | Physics | Validation | Command |
| --- | --- | --- | --- |
| [quasistatic/miehe_tension](quasistatic/miehe_tension/) | Miehe SENT tension | PASS against PhaseFieldX-style reference | `python -m phast run examples/quasistatic/miehe_tension/config.yaml --output_dir runs/miehe_tension` |
| [quasistatic/notched_holed_plate](quasistatic/notched_holed_plate/) | COMSOL notched-holed plate | PASS strict-parity comparison | `python -m phast run examples/quasistatic/notched_holed_plate/config.yaml --output_dir runs/notched_holed_plate` |

Validate a YAML deck before launching a full run:

```bash
python -m phast run examples/quasistatic/miehe_tension/config.yaml --validate-only
python -m phast run examples/quasistatic/notched_holed_plate/config.yaml --validate-only
```

## Solid Mechanics

The first three solid mechanics examples are real mesh-level FEA simulations
and run through the common YAML runner. Fluent snippets are authoring examples
only; the checked-in YAML deck and flat promoted result bundle remain the public
reproducibility contract. The remaining two are retained as numerical-method
diagnostics.

| Example | Physics | Command |
| --- | --- | --- |
| `solid_mechanics/linear_plate` | Linear elastic cantilever, sparse autograd solve | `python -m phast run examples/solid_mechanics/linear_plate/config.yaml` |
| `solid_mechanics/neohookean_plate` | Nonlinear neo-Hookean cantilever | `python -m phast run examples/solid_mechanics/neohookean_plate/config.yaml` |
| `solid_mechanics/j2_bar` | Mesh-level J2 plasticity bar | `python -m phast run examples/solid_mechanics/j2_bar/config.yaml` |
| `solid_mechanics/mixed_precision_cg` | Krylov precision diagnostic | `python examples/solid_mechanics/mixed_precision_cg/run.py` |
| `solid_mechanics/generalized_alpha_oscillator` | Generalized-alpha time-integration diagnostic | `python examples/solid_mechanics/generalized_alpha_oscillator/run.py` |

The legacy solid FEA script entrypoints remain as compatibility wrappers:
`python examples/solid_mechanics/linear_plate/run.py --config examples/solid_mechanics/linear_plate/config.yaml`,
`python examples/solid_mechanics/neohookean_plate/run.py --config examples/solid_mechanics/neohookean_plate/config.yaml`,
and `python examples/solid_mechanics/j2_bar/run.py --config examples/solid_mechanics/j2_bar/config.yaml`.

## Plasticity And Interface Beta

The plasticity/interface folder is a beta script-contract family, not a
YAML-first public gallery. Its canonical reproducibility manifest is:

```text
configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml
```

That manifest lists every runner, launcher command, required artifacts, and
claim boundary. This is deliberate: J2 material validation, ductile
phase-field evidence, cohesive operator smoke tests, PF-CZM validation, and
diffuse-interface screening exercise capabilities that are not all available
through the generic `phast run` YAML path yet.

When one of these workflows becomes customer-standard, promote it by updating
`PUBLIC_EXAMPLES_CONTRACT.yaml`, adding a flat leaf folder if needed, and
adding a regression test for the expected artifacts.

## Common CLI Flags

| Flag | Output |
| --- | --- |
| `--plots` | PNG figures when supported by the YAML runner |
| `--vtu` | VTU snapshots for ParaView |
| `--gif` | Animated GIF of damage evolution |
| `--trajectory --trajectory-format zarr` | Zarr trajectory store for reusable solver outputs |
| `--h5` | Legacy compatibility trajectory output; prefer Zarr for new work |
| `--all_outputs` | VTU + GIF + plots + profiler; does not imply legacy H5 |
| `--device cpu/cuda` | Compute device; use CPU on Mac for float64 fracture solves |
| `--output_dir DIR` | Custom output directory |

YAML runs should produce provenance and reusable trajectory outputs in the run
directory, including `config.yaml`, `run_lockfile.json`, `run_metadata.json`,
mesh provenance, scalar CSVs, plots, and `training_data.zarr/` when trajectory
output is requested. The heavy `training_data.zarr/` directory is retained for
private regeneration/HPC evidence in the private archive and is
not part of the lightweight public example payload. See
`docs/user_guide/example_contract.md` for the full promoted-example contract.
The older `docs/STANDARD_OUTPUTS.md`, `docs/visualisation_requirements.md`,
and `docs/visualization-output.md` pages are compatibility/narrow-backend
references linked from that canonical page.

## Adding Or Promoting An Example

1. Put the example in a flat leaf folder with `README.md`, `config.yaml` or
   `run.py`, and lightweight outputs.
2. Include `run_manifest.json`; include `visual_manifest.json` when plots or
   animations are generated.
3. Update `PUBLIC_EXAMPLES_CONTRACT.yaml`.
4. Add or update tests so the contract fails if the example drifts.
5. Keep raw HPC data, large Zarr/H5 stores, diagnostics, and paper-specific
   scratch work private unless they are promoted as named benchmark artifacts.
