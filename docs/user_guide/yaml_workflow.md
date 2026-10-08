# Declarative YAML workflows

YAML records the geometry, mesh, materials, constraints, loading, numerical
controls, and outputs needed to share a PhAST model. Start with the
[standard simulation tutorial](../tutorial/07_standard_simulation_workflow.md)
for source installation and the three-problem student route: small
quasi-static SENT, small dynamic SENT, and full layered DCB.

The fluent `phast.Problem` API remains useful for programmatic authoring.
Neither a common input format nor a Python object makes unsupported physics
available.

## Explain, check, run

Run from the installed repository root. Replace `CONFIG` with a complete
YAML path and `RUN` with a new result directory.

```bash
python -m phast explain-config CONFIG
python -m phast run CONFIG --validate-only
python -m phast run CONFIG --output_dir RUN
```

The first command explains the input, the second checks the implemented
schema/workflow constraints without solving, and the third executes it.
Schema-2 explanation dispatch and the new single-material adapters require
combined CLI checks before the new sequences are labelled verified.
Legacy `precheck --config` is not a universal schema-2 replacement.

For the full DCB reference:

```bash
python -m phast explain-config examples/two_material_dcb_beta/config.yaml
python -m phast run examples/two_material_dcb_beta/config.yaml --validate-only
python -m phast run examples/two_material_dcb_beta/config.yaml --output_dir runs/dcb_fine_reference
```

Preflight stops before meshing and solving. It does not establish non-empty
mesh selections, numerical convergence, crack growth, mesh independence, or
experimental agreement. Use the selected runner's documented scope.

## Schema-2 order

New standard-workflow inputs must follow the
{download}`configuration authoring rules <../../CONFIGURATION_STYLE.md>`:

| Order | Key | Contents |
|---|---|---|
| 1 | `schema_version` | `2` for this layout. |
| 2 | `name` | Descriptive problem name. |
| 3 | `reference` | Source or teaching scope, when applicable. |
| 4 | `geometry` | Generator and supported mesh parameters. |
| 5 | `regions` | Named node and element selections. |
| 6 | `materials` | Named models and material parameters. |
| 7 | `assignments` | Material-to-element-region mapping. |
| 8 | `initial_conditions` | Initial fields or route-specific maintained damage. |
| 9 | `boundary_conditions` | Named constraints and prescribed values. |
| 10 | `analysis_steps` | Physical analysis, active conditions, loading controls. |
| 11 | `solver` | Numerical algorithm, tolerances, iteration limits. |
| 12 | `outputs` | Directory, fields, histories, visuals. |

Use the {download}`complete DCB input <../../examples/two_material_dcb_beta/config.yaml>`
or {download}`heterogeneous SENT input <../../examples/heterogeneous_sent_beta/config.yaml>`,
not a mixture of unrelated partial examples. The companion single-material
inputs under `examples/standard_workflow/` require their documented adapter
checks before promotion.

### Geometry and regions

The documented multi-material route uses `geometry.type: structured_rectangle`
with `length`, `height`, `origin`, `nx`, and `ny` in
`geometry.parameters`. Cells are divided into T3 triangles. Material regions
select element centroids; crack constraints and supports select nodes.

Changing dimensions does not automatically move boundaries, cracks, layers,
or complementary disk selectors. Update those dependencies together and
cover every element exactly once. Mesh changes can make a narrow node
selection empty or alter an inclusion's discrete boundary.

### Materials and analysis

`materials.<name>.parameters.E` controls stiffness;
`materials.<name>.parameters.Gc` controls fracture resistance. Vary them
independently before attributing a response change to one property.
Use a coherent unit system; `geometry.units` does not convert arbitrary
bare material/loading numbers.

Quasi-static analysis omits inertia. The DCB
`analysis_steps[0].controls.number_of_steps: 121` gives 121 nominal levels
including zero: 120 nominal increments, with more possible after cutbacks.
`solver` separately controls convergence.

Dynamics includes density and physical time. Explicit integration needs a
CFL-limited time step. Final time, maximum steps, solver iterations, and
saved-output cadence are different controls; use the actual adapter keys.
Neither a smaller step nor one `h/l0` ratio establishes validation.

## Compatibility layouts and capability limits

Schema-1 benchmarks retain singular `material`, `loading`, and `output`
sections and `solver.solver_type`. Schema 2 uses `materials`,
`assignments`, `analysis_steps`, `outputs`, and `solver.type`.
Do not mix layouts or migrate by changing only the version number.

The documented multi-material scope is structured T3 rectangles, CPU float64,
quasi-static Amor AT2, elementwise `E`/`Gc`, shared `nu`/`l0` and
constitutive settings, and Dirichlet conditions. It does not cover dynamic
multi-material execution or independent interface laws. Other legacy
generators and imported meshes belong to their specific documented routes,
not automatically to this one. Consult the [capability matrix](capability_matrix.md).

Frozen course inputs and separate 3D research remain separate until adapters
and evidence exist. A common writing style is not a migration instruction.

## Inputs, templates, and complete comparison decks

| File class | Interpretation |
|---|---|
| Example `config.yaml` | Solver input for its documented adapter/status, not proof of a completed run. |
| DCB `comparisons/tough_region.yaml` and `uniform_layer.yaml` | Complete coarse inputs for the same CLI; only disk `Gc` differs between the pair. |
| Benchmark input under `configs/benchmarks/` | Follow its exact schema and example command. |
| `configs/REFERENCE.yaml` | Annotated compatibility reference, not a universal schema-2 input. |
| `examples/PUBLIC_EXAMPLES_CONTRACT.yaml` | Artifact inventory, not a solver problem. |
| Plasticity/interface reproduction contract | Dispatcher manifest; follow its documented `--validation-id` route. |

Compact complete comparison decks duplicate model data, not solver code.
The [standard tutorial](../tutorial/07_standard_simulation_workflow.md)
provides exact coarse-pair commands and distinguishes their results from
the fine DCB reference. Use separate output directories.

## HDF5 trajectories

This **partial schema-2 excerpt** adds a trajectory request to the existing
`outputs.fields` list; preserve the other outputs:

```yaml
outputs:
  fields:
    - {name: trajectory, format: h5, every: 1}
```

HDF5 writes `training_data.h5`. The multi-material runner supports HDF5
only and refuses to overwrite an existing trajectory. DCB requests it;
heterogeneous SENT leaves it opt-in.

The **legacy schema-1** equivalent uses different keys:

```yaml
output:
  trajectory: true
  trajectory_format: h5
  h5_every: 5
```

Legacy routes that support `zarr` or `both` require explicit selection;
do not assume those formats work in every schema-2 adapter. No format is a
silent fallback. Existing stores remain unchanged. Close a file before
transferring it; a single file does not make concurrent synchronised writes
safe. Keep large raw stores out of public example payloads.

## Inspect results

A successful run writes requested artifacts to `RUN`: configuration and
provenance, manifests, histories, lightweight visuals, and requested fields,
as supported by the route. Inspect the manifest rather than assuming every
small exercise produces a propagating crack.

```python
import phast

result = phast.load_result("runs/dcb_fine_reference")
print(result.metadata())
print(result.history_names())
print(result.visuals())
if result.has_field("damage"):
    damage = result.field("damage", step=-1)
```

The result interface reads stored quantities. A PNG is not a field
trajectory; retained evidence is not a new run. See the
[example contract](example_contract.md).

If a documented step fails,
[open a GitHub issue](https://github.com/CEMS-Lab/PhAST/issues/new/choose)
with the configuration, exact command, OS, Python/PhAST version, and full error.
