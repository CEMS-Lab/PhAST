# YAML benchmark configs

These YAML files are runnable problem definitions for the canonical
configuration driver:

```bash
python -m phast run configs/benchmarks/<family>/<name>.yaml
```

They are intended to be both input files and reviewable documentation:
geometry, material, loading, boundary conditions, solver settings, and
outputs live in one file. `configs/REFERENCE.yaml` is the generated
field-by-field schema reference.

## Quick Start

```bash
# Validate schema only; does not generate meshes or run a solve.
python -m phast run configs/benchmarks/dynamic/B3_dynamic_sent.yaml --validate-only

# Explain the selected model, solver path, outputs, and setup warnings.
python -m phast explain-config configs/benchmarks/dynamic/B3_dynamic_sent.yaml

# Export editor autocomplete / JSON Schema support.
python -m phast schema --output configs/phast.schema.json

# Run on CPU.
python -m phast run configs/benchmarks/dynamic/B3_dynamic_sent.yaml --device cpu

# Run on CUDA and request plots/GIF where configured.
python -m phast run configs/benchmarks/dynamic/B7_dynamic_crack_branching_comsol.yaml \
    --device cuda --gif --plots

# Scaffold a new starter config.
python -m phast new my_benchmark --type quasi_static --material pmma_bleyer
```

CLI flags override YAML values at run time. For example, `--device cpu`
overrides the `device:` block, and output flags such as `--gif` or
`--plots` can enable additional artifacts for a run.

## Layout

- `configs/benchmarks/dynamic/`: canonical dynamic benchmark configs.
- `configs/benchmarks/dynamic/diagnostics/`: dynamic timing, structured-mesh,
  and debug variants.
- `configs/benchmarks/quasistatic/`: canonical quasi-static benchmark configs.
- `configs/benchmarks/quasistatic/diagnostics/`: quasi-static diagnostic
  variants.
- `configs/benchmarks/quasistatic/manifests/`: Slurm/benchmark orchestration
  manifests labeled `manifest_type: command_manifest`; they are not solver
  problem configs.
- `configs/benchmarks/plasticity_interface/manifests/`: command manifests for
  J2 plasticity, cohesive, contact, and diffuse-interface validation examples.
- `configs/benchmarks/*/reproducibility_contracts.yaml`: validation artifact
  contracts labeled `manifest_type: reproducibility_contract`; they document
  expected outputs and rerun evidence, but are not runnable problem configs.
- `configs/REFERENCE*.yaml`: schema reference and template.
- `configs/B*.yaml`, `configs/QS*.yaml`: compatibility symlinks to preserve
  old command lines, tests, and historical docs.

Each run writes both `config.yaml` and `run_lockfile.json` into the output
directory. `config.yaml` is the post-CLI resolved problem definition;
`run_lockfile.json` adds the input config SHA-256, CLI arguments, git state,
dependency versions, hostname/platform, and resolved mesh/material/solver/device
summaries for reproducibility audits.

## Current Benchmark Index

| YAML | Solver class | Problem |
|---|---|---|
| `benchmarks/dynamic/B2_kalthoff_winkler.yaml` | explicit dynamics | Kalthoff-Winkler impact |
| `benchmarks/dynamic/B3_dynamic_sent.yaml` | explicit dynamics | Dynamic SENT straight crack |
| `benchmarks/dynamic/diagnostics/B3_sent_clean_timing.yaml` | explicit dynamics | Clean SENT timing/regression config |
| `benchmarks/dynamic/B5_pmma_branching.yaml` | explicit dynamics | PMMA pre-strained branching, Bleyer 2017 |
| `benchmarks/dynamic/B5_pmma_branching_dU*.yaml` | explicit dynamics | PMMA branching displacement variants |
| `benchmarks/dynamic/B6_perforated_10holes.yaml` | explicit dynamics | Perforated plate, 10 holes |
| `benchmarks/dynamic/B6_perforated_30holes.yaml` | explicit dynamics | Perforated plate, 30 holes |
| `benchmarks/dynamic/B6_perforated_1hole_near.yaml` | explicit dynamics | One hole near the crack path |
| `benchmarks/dynamic/B6_perforated_1hole_far.yaml` | explicit dynamics | One hole far from the crack path |
| `benchmarks/dynamic/B7_dynamic_crack_branching_comsol.yaml` | explicit dynamics | COMSOL 6.4 PMMA dynamic branching cross-check |
| `benchmarks/quasistatic/QS_lshaped_concrete.yaml` | quasi-static | L-shaped concrete panel, Ambati 2015 |
| `benchmarks/quasistatic/QS_notched_holed_plate.yaml` | quasi-static | COMSOL notched-holed plate, Ambati 2015 |
| `benchmarks/quasistatic/diagnostics/QS_notched_holed_plate_welded.yaml` | quasi-static | Welded-control variant of the holed plate |
| `benchmarks/plasticity_interface/manifests/customer_validation_examples.yaml` | command manifest | J2 plasticity, cohesive contact, and structural cohesive validation examples |

Some historical configs retain an `example:` field for provenance or for
older demo wrappers. It is accepted by the schema, but it is not required
for the canonical `python -m phast run ...` path.

Benchmark and customer-validation configs should also include an
`acceptance:` block. This is structured but extensible metadata for the
reference result, required artifacts, numerical/visual metrics, tolerances, and
known caveats. The standard fields are validated: `status` must be one of
`scaffold`, `beta`, `production`, `validated`, `diagnostic`, or
`experimental`; `required_outputs` must be a list of artifact filenames; and
`metrics` must be a mapping of named metric metadata. Custom
benchmark-specific keys are still preserved. The solver does not enforce these
tolerances during the run; it preserves them in the resolved config and
`run_lockfile.json`, and `explain-config` prints the block so reviewers can see
what a successful run is expected to demonstrate.

## Schema And Validation

Use these checks before spending CPU/GPU time:

```bash
python -m phast run configs/benchmarks/<family>/<name>.yaml --validate-only
python -m phast explain-config configs/benchmarks/<family>/<name>.yaml
```

`--validate-only` catches schema errors, missing external mesh files, and
bad node-set references in imported meshes with line-numbered messages.
Relative `geometry.mesh_path` values are checked against the YAML file's
directory first, then the current working directory.
Command manifests and reproducibility contracts intentionally fail
`--validate-only` with a short message explaining that they are orchestration
artifacts rather than runnable problem configs.
`explain-config` is a dry-run reviewer: it prints the selected geometry
source, material/fracture model, AT1/AT2 fields, solver/backend choices,
loading, boundary conditions, outputs, device, and known warnings without
generating a mesh.

Compatibility checks also reject combinations that are known to fail or be
ignored, such as dynamic-only `time_integrator` settings on quasi-static
runs, or explicit `generalized_alpha` / `fresh_d_in_corrector` with
`rigid_connector` MPCs.

Non-fatal advisories are printed for physically risky but legal setups,
including missing `schema_version` and declared fracture meshes with
`h/l0 > 0.5` near the crack path. These warnings do not fail
`--validate-only`, but they should be resolved before customer validation
or paper-quality runs.

For IDE autocomplete and external tooling, use the generated JSON Schema:

```bash
python -m phast schema --output configs/phast.schema.json
```

The schema is generated from the same dataclasses, enum tables, and numeric
ranges as the runtime validator, so editor hints stay aligned with the
canonical YAML loader.

## Units

Bare numeric values use the solver's internal unit convention:

| Quantity | Internal unit |
|---|---|
| Length | mm |
| Time | s |
| Force | N |
| Stress / modulus | MPa |
| Boundary traction | N/mm |
| Density | tonne/mm^3 |
| Fracture toughness `Gc` | N/mm |

Quoted SI suffixes are accepted for material quantities, key loading
fields, and boundary-condition displacement/traction/ramp values, for example:

```yaml
material:
  E: "32 GPa"
  Gc: "3 J/m^2"
  l0: "0.25 mm"
  rho: "2450 kg/m^3"
loading:
  t_total: "80 us"
boundary_conditions:
  - {nodes: top, type: prescribe, component: 1, value: "0.01 mm"}
  - {nodes: right, type: traction, component: 1,
     value: "1 MPa", ramp_type: smooth_step, t_ramp: "10 us"}
```

Boundary tractions are stored as force per boundary length (`N/mm`) in the
2D weak form. Stress suffixes such as `MPa` assume unit out-of-plane
thickness, which matches the shipped 2D benchmark convention.

See `configs/REFERENCE.yaml` and `phast/units.py` for the
complete supported suffix list.

## New Config Checklist

- Start from `python -m phast new ...` or copy the closest
  existing benchmark.
- Keep material values inline unless a shared preset is genuinely useful.
- Prefer `geometry.primitives` / `domain` / `named_groups` when the
  problem is easier to review as geometry than as generator parameters.
- Set `output.reaction_node_set` and `output.reaction_component` for
  quasi-static load-displacement comparisons.
- Run `--validate-only` and `explain-config` before submitting a long run.
- For quasi-static fracture, use the current safe defaults:
  `solver_type: quasi_static`, `backend: auto`, and `preconditioner:
  jacobi` unless a backend-specific validation issue says otherwise.
