# Schema v2 Migration

`schema_version: 2` is an additive workflow-contract schema. It lets PhAST
compile a customer-style input deck into the internal `ProblemSpec` contract
used by YAML adapters, the Python API, validation, and result planning.

v1 YAML remains supported. Do not rename existing v1 YAML keys when maintaining
or running current examples. The v2 shape is for migration tooling, new
contract validation, and future workflow productization.

## Current Boundary

Use v2 for contract validation:

```bash
python -m phast run path/to/schema_v2.yaml --validate-only
```

Direct schema-v2 execution is available for two deliberately narrow slices:
supported quasi-static phase-field fracture decks and promoted solid-mechanics
examples that declare `solver.type: solid_mechanics` plus a supported
`solver.example`, such as `solid_mechanics.linear_plate`. PhAST validates the
v2 contract and lowers it to the existing v1 compatibility runner shape.
Unsupported fracture v2 decks, including explicit/dynamic v2 decks, remain
validate-only until each v2-to-runner adapter is deliberately implemented and
tested.

The v2 deck is not an arbitrary weak-form compiler. It also does not promote beta plasticity/cohesive/interface workflows into a general public schema. Those
beta paths remain allowlisted validation contracts or explicit scripts until
their solver support and output contracts are promoted separately.

## Mapping From v1 Concepts

The migration target is the internal workflow contract:

| v1/current concept | v2 contract concept |
|---|---|
| `geometry` / `geometry.mesh_path` | `geometry` with either generated parameters or mesh path |
| named node sets and physical groups | `regions` |
| one material preset/inline material | `materials` plus assignments |
| `initial_conditions.preseed_*` | `initial_conditions` |
| `boundary_conditions` entries | `boundary_conditions` with stable names |
| `loading` plus solver type | `analysis_steps` and `solver` |
| `output.trajectory` / `output.vtu` | `outputs.fields` |
| reaction, energy, telemetry CSVs | `outputs.history` |
| plots, GIFs, thumbnails | `outputs.visuals` |

The split between materials plus assignments is intentional. Material
definitions describe constitutive parameters; assignments say which region
receives each material. This keeps future region-wise material decks explicit
without changing current v1 behavior.

## Programmatic Migration

Existing YAML can be compiled to a `ProblemSpec` and serialized to the v2
dictionary shape:

```python
import yaml
from phast.workflow import problem_spec_from_yaml, problem_spec_to_schema_v2_dict

spec = problem_spec_from_yaml("examples/dynamic/B3_dynamic_sent/config.yaml")
payload = problem_spec_to_schema_v2_dict(spec)
print(yaml.safe_dump(payload, sort_keys=False))
```

`problem_spec_to_schema_v2_dict()` is a migration helper, not a broad promise
that the emitted v2 deck is directly executable. Validate the result with
`python -m phast run path/to/schema_v2.yaml --validate-only`; supported
quasi-static phase-field fracture decks and promoted solid-mechanics decks can
then run through their compatibility lowering adapters.

Public contract tests roundtrip the YAML-first dynamic, quasi-static, and
promoted solid-mechanics examples through this schema-v2 dictionary shape
without running solvers. This keeps the migration surface synchronized with
current public examples while preserving the existing v1 YAML behavior.

## Validation

The schema-v2 validator checks:

- unsupported solver, material, boundary-condition, output, and postprocess
  contract names;
- missing region references from materials, initial conditions, boundary
  conditions, and histories;
- duplicate region, material, analysis-step, boundary-condition, field-output,
  history-output, and postprocess names;
- duplicate material assignments to the same named region;
- ambiguous multi-material specs that omit explicit region assignments;
- duplicate mesh-to-region mappings;
- invalid component indices and conflicting displacement Dirichlet conditions;
- active boundary-condition references from analysis steps;
- unsupported solver/material/analysis-step/boundary-condition/output family
  combinations;
- unsupported execution routes.

These checks run before solver construction and do not rewrite solver loops,
postprocessors, output formats, or public examples.
