# YAML Workflow

YAML is the canonical input-deck format for reproducible PhAST runs. Use YAML
when you want to share an exact setup, submit a batch/HPC job, run CI checks,
or reproduce a public example without writing a Python driver.

Use the fluent `phast.Problem` API while designing a model. Move the final
setup into YAML when the run needs to be durable and repeatable.

## Basic Commands

Validate a configuration without running the solver:

```bash
python -m phast run examples/quasistatic/notched_holed_plate/config.yaml --validate-only
```

Explain a configuration in a readable form:

```bash
python -m phast explain-config examples/quasistatic/notched_holed_plate/config.yaml
```

Run the case:

```bash
python -m phast run examples/quasistatic/notched_holed_plate/config.yaml \
  --output_dir runs/notched_holed_plate
```

## What a YAML Deck Controls

A YAML deck can define:

- geometry or imported mesh path,
- material model and parameters,
- initial conditions such as seeded damage,
- boundary conditions and loading protocol,
- solver type, tolerances, backend, device, and time stepping,
- requested fields, histories, plots, animations, and trajectory stores,
- acceptance metadata for curated validation examples.

The deck is intended to behave like a conventional FEM input file: validate it,
run it, and inspect the standard result directory.

## Minimal Shape

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

Public examples contain the exact keys required by their runner. Start from an
existing example when creating a related case.

## Geometry and Meshes

For built-in examples, geometry can be described directly in YAML and PhAST
will generate a Gmsh mesh. For custom geometry, generate a `.msh` file and
reference it:

```yaml
geometry:
  mesh_path: meshes/my_plate.msh
```

The mesh should preserve named physical groups for every boundary or domain
used by the deck. Inspect external mesh groups before writing the YAML:

```python
import phast

summary = phast.inspect_mesh("meshes/my_plate.msh")
print(summary["named_groups"])
```

## Trajectory Output

Use Zarr for new runs:

```yaml
output:
  trajectory: true
  trajectory_format: zarr
  h5_every: 5
```

Use H5 only for legacy tools:

```yaml
output:
  trajectory: true
  trajectory_format: h5
  h5_every: 5
```

Write both stores only when comparing old and new post-processing:

```yaml
output:
  trajectory: true
  trajectory_format: both
  h5_every: 5
```

The CLI can override trajectory settings:

```bash
python -m phast run case.yaml --trajectory --trajectory-format zarr
```

## Standard Result Directory

A normal run writes a directory such as `runs/notched_holed_plate/`. Promoted
examples should include:

- copied or resolved `config.yaml`,
- `run_lockfile.json`,
- `run_metadata.json`,
- `run_manifest.json` or `visual_manifest.json`,
- mesh provenance where applicable,
- CSV histories such as response, energy, timing, or convergence,
- final field plots,
- setup images and animations where meaningful,
- trajectory stores when requested.

See [Example result contract](example_contract.md) and
the repository visualization-output guide for output conventions.

## Inspecting a Completed Run

```python
import phast

result = phast.load_result("runs/notched_holed_plate")
print(result.metadata())
print(result.mesh())
print(result.history_names())
print(result.visuals())
print(result.field_names())

if result.has_field("damage"):
    damage = result.field("damage", step=-1)
```

`Result` is read-only. It exposes stored manifests, metadata, CSV histories,
visual artifacts, mesh metadata, and raw trajectory fields when present. It
does not silently synthesize derived quantities that were not written by the
run.

## Batch and HPC Use

YAML is the recommended surface for batch execution because the same file can
be validated locally, committed with an example, submitted to a queue, and
loaded later with the Result API.

Typical sequence:

```bash
python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml --validate-only
python -m phast explain-config examples/dynamic/B2_kalthoff_winkler/config.yaml
python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml \
  --device cpu \
  --output_dir runs/B2_kalthoff_winkler
```

For larger dynamic or quasi-static runs, use the same `config.yaml` inside the
cluster submission script and keep the output directory intact.

## Current Boundary

YAML decks route to supported PhAST runners. They do not evaluate arbitrary
Python launcher strings or arbitrary weak-form equations. If a deck requests a
workflow outside the supported capability matrix, validation should report that
boundary before the solver is launched.

Next pages:

- [Setting up new problems](setup_problems.md)
- [Python API](python_api.md)
- [Results API](results_api.md)
- [Example result contract](example_contract.md)
- [Capability matrix](capability_matrix.md)
