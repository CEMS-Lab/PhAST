# Declarative YAML Workflows

YAML is the canonical configuration format for reproducible simulations in PhAST. Declarative configurations should be used for sharing exact geometric and physical setups, submitting batch/HPC jobs, and enforcing strict reproducibility for academic publications.

While the fluent `phast.Problem` API is ideal for interactive model design, the final implementation should be serialized to YAML to ensure durability.

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

*Note: Public validation examples contain the exact keys required by their execution pathways. It is recommended to use existing examples as a foundation for novel configurations.*

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

A successfully executed configuration writes an immutable artifact directory (e.g., `runs/notched_holed_plate/`). For promoted academic examples, this directory guarantees:
- The exact `config.yaml` used for execution.
- `run_lockfile.json` capturing the parsed state, Git hashes, and dependencies.
- CSV histories (response, kinetic energy, nonlinear convergence).
- High-fidelity Zarr trajectories.
- Visual manifests and pre-rendered plots.

## Result Inspection

The resulting artifact directory is strictly read-only and can be queried programmatically:

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

YAML configurations strictly route to validated PhAST solver execution pathways. They do not execute arbitrary Python scripts or compile arbitrary weak-form PDEs. If a configuration requests physics outside the supported [Capability Matrix](capability_matrix.md), the validation phase will cleanly intercept and report the violation before any compute is allocated.
