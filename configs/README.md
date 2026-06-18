# Configuration Files

This directory contains the public YAML configuration surface for PhAST.
Runnable benchmark configs live under `configs/benchmarks/`; schema and
reference files live at the top level.

```bash
python -m phast run configs/benchmarks/<family>/<name>.yaml
```

The YAML files are both solver inputs and reviewable documentation. They keep
geometry, material data, loading, boundary conditions, solver settings, and
output requests in one place. `configs/REFERENCE.yaml` is the field-by-field
reference, and `configs/phast.schema.json` provides editor autocomplete and
external validation support.

## Quick Start

```bash
# Validate schema and referenced files without running the solve.
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
overrides the `device:` block, and output flags such as `--gif` or `--plots`
can enable additional artifacts for a run.

## Public Layout

| Path | Contents |
|---|---|
| `configs/benchmarks/dynamic/` | Reference dynamic fracture benchmark configs. |
| `configs/benchmarks/quasistatic/` | Reference quasi-static benchmark configs. |
| `configs/benchmarks/plasticity_interface/` | Beta plasticity/interface validation manifests and contracts. |
| `configs/REFERENCE.yaml` | Human-readable reference for supported config fields. |
| `configs/phast.schema.json` | Generated JSON Schema for editor and tooling support. |

Top-level compatibility aliases such as `configs/B2_kalthoff_winkler.yaml` and
`configs/QS_notched_holed_plate.yaml` have been removed from the public tree.
Use the reference `configs/benchmarks/...` paths in examples, documentation,
CI, and papers.

Command manifests that are not directly runnable by `python -m phast run` are
not part of the public config tree.

Each run writes both `config.yaml` and `run_lockfile.json` into the output
directory. `config.yaml` is the post-CLI resolved problem definition;
`run_lockfile.json` records the input config hash, CLI arguments, git state,
dependency versions, host/platform metadata, and resolved
mesh/material/solver/device summaries for reproducibility audits.

## Benchmark Index

| YAML | Solver class | Problem |
|---|---|---|
| `benchmarks/dynamic/B2_kalthoff_winkler.yaml` | explicit dynamics | Kalthoff-Winkler impact |
| `benchmarks/dynamic/B3_dynamic_sent.yaml` | explicit dynamics | Dynamic SENT straight crack |
| `benchmarks/dynamic/B5_pmma_branching.yaml` | explicit dynamics | PMMA pre-strained branching, Bleyer 2017 |
| `benchmarks/dynamic/B5_pmma_branching_dU*.yaml` | explicit dynamics | PMMA branching displacement variants |
| `benchmarks/dynamic/B6_perforated_10holes.yaml` | explicit dynamics | Perforated plate, 10 holes |
| `benchmarks/dynamic/B6_perforated_30holes.yaml` | explicit dynamics | Perforated plate, 30 holes |
| `benchmarks/dynamic/B6_perforated_1hole_near.yaml` | explicit dynamics | One hole near the crack path |
| `benchmarks/dynamic/B6_perforated_1hole_far.yaml` | explicit dynamics | One hole far from the crack path |
| `benchmarks/dynamic/B7_dynamic_crack_branching_comsol.yaml` | explicit dynamics | COMSOL 6.4 PMMA dynamic branching cross-check |
| `benchmarks/quasistatic/QS_lshaped_concrete.yaml` | quasi-static | L-shaped concrete panel, Ambati 2015 |
| `benchmarks/quasistatic/QS_notched_holed_plate.yaml` | quasi-static | COMSOL notched-holed plate, Ambati 2015 |
| `benchmarks/quasistatic/QS_notched_holed_plate_comsol_strict.yaml` | quasi-static | COMSOL holed-plate strict parity setup |

Some historical configs retain an `example:` field for provenance or older
demo wrappers. It is accepted by the schema, but it is not required for the
reference `python -m phast run ...` path.

Benchmark and researcher-validation configs should include an `acceptance:`
block. This structured metadata documents the reference result, required
artifacts, numerical/visual metrics, tolerances, and known caveats. The solver
preserves the block in the resolved config and `run_lockfile.json`, and
`explain-config` prints it for review. It does not enforce these tolerances
during the solve.

## Schema And Validation

Use these checks before spending CPU/GPU time:

```bash
python -m phast run configs/benchmarks/<family>/<name>.yaml --validate-only
python -m phast explain-config configs/benchmarks/<family>/<name>.yaml
```

`--validate-only` catches schema errors, missing external mesh files, and bad
node-set references in imported meshes with line-numbered messages. Relative
`geometry.mesh_path` values are checked against the YAML file's directory first,
then the current working directory.

Command manifests and reproducibility contracts intentionally fail
`--validate-only` because they describe orchestration or expected artifacts
rather than a single solver problem.

For IDE autocomplete and external tooling, use the generated JSON Schema:

```bash
python -m phast schema --output configs/phast.schema.json
```

The schema is generated from the same dataclasses, enum tables, and numeric
ranges as the runtime validator, so editor hints stay aligned with the
YAML loader.

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

Quoted SI suffixes are accepted for material quantities, key loading fields,
and boundary-condition displacement/traction/ramp values, for example:

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
  - {nodes: right, type: traction, component: 1, value: "1 MPa",
     ramp_type: smooth_step, t_ramp: "10 us"}
```

Boundary tractions are stored as force per boundary length (`N/mm`) in the 2D
weak form. Stress suffixes such as `MPa` assume unit out-of-plane thickness,
which matches the shipped 2D benchmark convention.

See `configs/REFERENCE.yaml` and `phast/units.py` for the complete supported
suffix list.

## New Config Checklist

- Start from `python -m phast new ...` or copy the closest existing benchmark.
- Keep material values inline unless a shared preset is genuinely useful.
- Prefer `geometry.primitives` / `domain` / `named_groups` when the problem is
  easier to review as geometry than as generator parameters.
- Set `output.reaction_node_set` and `output.reaction_component` for
  quasi-static load-displacement comparisons.
- Run `--validate-only` and `explain-config` before submitting a long run.
- For quasi-static fracture, use the current safe defaults:
  `solver_type: quasi_static`, `backend: auto`, and `preconditioner: jacobi`
  unless a backend-specific validation issue says otherwise.
