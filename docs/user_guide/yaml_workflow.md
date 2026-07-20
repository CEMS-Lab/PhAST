# Declarative YAML Workflows

YAML is the primary configuration format for reproducible simulations in
PhAST. Declarative configurations record the geometric, physical, numerical,
and output choices needed to review and repeat an academic simulation.

The fluent `phast.Problem` API is useful for programmatic model construction;
YAML is preferred when a complete setup must be shared and reviewed.

YAML is the public reproduction format because it is explicit, machine-readable,
and stable across reruns. A validated configuration makes it possible to
recreate the same geometry, material parameters, loading history, and output
contract without reinterpreting interactive Python state.

## Configuration Scope

A declarative YAML configuration explicitly defines:
- **Geometry**: Built-in primitives or paths to external meshes.
- **Constitutive Models**: Material definitions, physics presets, and specific parameters.
- **Boundary Conditions**: Kinematic constraints and loading protocols.
- **Solver Execution**: Mathematical backend, temporal discretization, tolerances, and hardware device.
- **Artifact Generation**: Desired volumetric fields, CSV histories, visualization rendering, and trajectory storage formats.

## Execution Workflow

PhAST provides three primary CLI entry points for interacting with YAML configurations:

```bash
# 1. Validate schema constraints and semantic logic without solving
python -m phast run examples/quasistatic/notched_holed_plate/config.yaml --validate-only

# 2. Inspect the parsed configuration graph and hardware placement
python -m phast explain-config examples/quasistatic/notched_holed_plate/config.yaml

# 3. Execute the simulation and specify an artifact output directory
python -m phast run examples/quasistatic/notched_holed_plate/config.yaml --output_dir runs/notched_holed_plate
```

For batch processing and HPC environments, this identical sequence ensures configurations are mathematically valid on the head node before consuming cluster compute resources.

This matters because the same configuration file can be shared across local
development, continuous integration, and HPC submission workflows without
changing the physics definition.

## Schema Structure

```yaml
problem:
  name: notched_holed_plate

geometry:
  mesh_path: mesh.msh

material:
  preset: miehe_tension
  overrides:
    l0: 0.015
    pf_model: AT2

boundary_conditions:
  - {nodes: bottom, type: fix, component: 0}
  - {nodes: bottom, type: fix, component: 1}
  - {nodes: top, type: prescribe, component: 1, value: 0.001}

loading:
  protocol: simple
  num_steps: 10

solver:
  solver_type: quasi_static
  backend: auto
  preconditioner: jacobi

output:
  plots: true
  trajectory: true
  trajectory_format: zarr
```

Public examples contain the keys required by their documented execution
pathways. Begin with the nearest example and modify one physical or numerical
choice at a time.

## Runnable Decks, Templates, And Contracts

Not every YAML file in the repository represents one solver problem:

| File class | Directly runnable? | Interpretation |
|---|---|---|
| `examples/<family>/<case>/config.yaml` | Yes | Complete example-local solver input. |
| `configs/benchmarks/<family>/<case>.yaml` | Yes | Complete benchmark solver input. |
| `configs/REFERENCE.yaml` | No | Annotated field reference and template. |
| `examples/PUBLIC_EXAMPLES_CONTRACT.yaml` | No | Example documentation and artifact inventory. |
| `configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml` | Only with `--validation-id` | Dispatcher manifest for beta validation programs. |

Contracts and manifests describe a collection of programs or expected
artifacts. They are not interchangeable with a single fracture `config.yaml`.

## External Meshes and Provenance

For built-in examples, structural geometry can be declared directly in YAML and PhAST will generate the underlying computational mesh via Gmsh. For custom domains, generate a format-compliant mesh (e.g., `.msh`) and reference it:

```yaml
geometry:
  mesh_path: meshes/custom_domain.msh
```

External meshes must preserve named physical groups for every boundary or domain referenced in the YAML configuration. You can inspect parsed mesh groups programmatically:

```python
import phast

summary = phast.inspect_mesh("meshes/custom_domain.msh")
print(summary["named_groups"])
```

## High-Fidelity Volumetric Storage

PhAST defaults to chunked, parallel-friendly `zarr` stores for recording volumetric field trajectories. The legacy `h5` format is maintained strictly for compatibility with older external post-processing scripts.

```yaml
output:
  trajectory: true
  trajectory_format: zarr  # Options: zarr, h5, both
  h5_every: 5
```

Trajectory formats can also be dynamically overridden via the CLI without modifying the underlying configuration file:

```bash
python -m phast run config.yaml --trajectory --trajectory-format zarr
```

## Standardized Artifact Directory

A successfully executed configuration writes an artifact directory such as
`runs/notched_holed_plate/`. Depending on the selected output settings and
runner, the directory may include:
- The exact `config.yaml` used for execution.
- `run_lockfile.json` capturing the parsed state, Git hashes, and dependencies.
- CSV histories (response, kinetic energy, nonlinear convergence).
- High-fidelity Zarr trajectories when trajectory output is enabled.
- Visual manifests and pre-rendered plots.

## Result Inspection

The public `Result` interface treats the existing artifact directory as
read-only and can be queried programmatically:

```python
import phast

result = phast.load_result("runs/notched_holed_plate")
print(result.metadata())
print(result.history_names())

if result.has_field("damage"):
    damage = result.field("damage", step=-1)
```

The `Result` object exposes only quantities that were explicitly written during the simulation; it does not silently synthesize derived fields.

## Extensibility Boundary

YAML configurations route to explicitly implemented PhAST execution pathways.
They do not compile arbitrary weak-form PDEs. The capability matrix defines the
documented support boundary. Schema validation detects unsupported combinations
that are represented in the validator, but it is not a substitute for checking
the numerical evidence associated with a model.
