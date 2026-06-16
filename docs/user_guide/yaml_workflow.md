# YAML workflow

Use the fluent `phast.Problem` API to author new models. Use YAML decks for public examples, reproducibility, batch/HPC runs, and sharing exact simulations.

YAML remains the exact public input-deck format. Users should not need to write
a Python driver to reproduce a promoted example, submit a cluster job, share an
exact setup, or rerun CI validation.

## Basic run

Validate a configuration without running the solver:

```bash
python -m phast run examples/quasistatic/notched_holed_plate/config.yaml --validate-only
```

Run the case:

```bash
python -m phast run examples/quasistatic/notched_holed_plate/config.yaml \
  --output_dir runs/notched_holed_plate
```

The output directory follows `docs/user_guide/example_contract.md`, the
canonical promoted-example contract. The older compatibility references remain:
`docs/STANDARD_OUTPUTS.md` for run-file details,
`docs/visualisation_requirements.md` for promoted figures and animations, and
`docs/visualization-output.md` for VTU/PV visualization format guidance. A
normal run contains the resolved `config.yaml`, `run_lockfile.json`,
`run_metadata.json`, `mesh.msh`, `mesh.geo` when available, CSV histories,
telemetry, timing, energy output, and core plots such as
`initial_conditions.png`, `damage_final.png`, and `energy.png`.

Load an existing run directory without rerunning the solver:

```python
import phast

result = phast.load_result("runs/notched_holed_plate")
metadata = result.metadata()
mesh = result.mesh()
available = result.history_names()
energy = result.history("energy")
visuals = result.visuals()
fields = result.field_names()
has_damage = result.has_field("damage")
damage = result.field("damage", step=-1)  # NumPy array, if stored directly
result.postprocess(fields=["damage", "energy"], skip_gif=True)
```

The current `Result` API is read-only and covers manifests, metadata, mesh
metadata/provenance, CSV histories, visual artifacts, and trajectory field
discovery. Canonical histories include `energy`, `solver_telemetry`, `timing_per_step`,
`load_displacement`, `reaction_force`, `max_damage`, and `response` when the
corresponding CSV columns or files exist. Canonical fields include `damage`,
`displacement`, `history_field`, `stress`, `strain`, `velocity`, and
`acceleration` when a Zarr/H5 trajectory store exists. `result.field()` returns
directly stored raw trajectory datasets as NumPy arrays; use
`torch.as_tensor(result.field("damage"))` for a zero-copy PyTorch view when
training workflows need tensors. `result.mesh()` returns available mesh
metadata from run metadata, manifests, or trajectory mesh groups; full FEM mesh
reconstruction remains a follow-up for mesh-specific APIs. Derived fields such
as von Mises stress remain a follow-up for postprocessing-specific APIs.
`result.postprocess(...)` is an explicit wrapper around the existing
`python -m phast postprocess <run_dir>` command and writes only through the
current postprocessor/output formats.

## Trajectory output

Use Zarr for new work:

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

Write both stores when comparing old and new post-processing:

```yaml
output:
  trajectory: true
  trajectory_format: both
  h5_every: 5
```

The CLI can override the YAML:

```bash
python -m phast run case.yaml --trajectory --trajectory-format both
```

## Geometry and meshes

For built-in examples, put geometry directly in YAML. PhAST compiles the
primitive geometry to a Gmsh mesh, runs the solver, and copies `mesh.msh` and
the generated `mesh.geo` debug artifact into the output directory.

For custom meshes, provide a Gmsh `.msh` file with named physical groups that
match the boundary-condition node-set names in YAML:

```yaml
geometry:
  mesh_path: meshes/my_plate.msh

boundary_conditions:
  - {nodes: left, type: fix, component: 0}
  - {nodes: left, type: fix, component: 1}
  - {nodes: right, type: prescribe, component: 0, value: 0.01}
```

If a user has a `.geo` file, generate the `.msh` with Gmsh first and reference
the `.msh` in YAML. The mesh must preserve physical groups for the boundaries
used by `boundary_conditions`.

Inspect external mesh groups before binding physics to them:

```python
import phast

summary = phast.inspect_mesh("meshes/my_plate.msh")
print(summary["named_groups"])
```

`inspect_mesh()` is read-only and uses meshio. It reports point/cell counts,
meshio point/cell sets when present, and Gmsh physical named groups from
`field_data` plus `gmsh:physical` cell tags. Use this discovery step to map
external names to stable workflow regions such as
`.region("fixed_base", from_mesh="NodeSet-Base")` before assigning materials,
boundary conditions, initial conditions, or outputs.

## What YAML controls

A single YAML file defines:

- geometry or mesh input
- material model and phase-field parameters
- boundary conditions and load protocol
- solver type, tolerances, backend, and device
- output cadence, trajectory format, plots, and animations
- optional acceptance metadata for validation examples

The intended reproduction workflow is the same as a conventional FEM input deck:
author or inspect the setup, save the YAML, validate it, run it, and
inspect the standard result directory. Write new models with
`phast.Problem`; use YAML when the setup becomes a public example, shared
reproduction artifact, or batch/HPC job.

## Schema v2 contract

`schema_version: 2` is an additive workflow-contract schema for compiling
input decks to the same internal `ProblemSpec` used by the Python API. v1 YAML remains supported
and continues through the existing `ProblemConfig`,
`load_config()`, `resolve_config()`, and runner paths.

```yaml
schema_version: 2
name: notched_plate

geometry:
  mesh_path: meshes/notched_plate.msh

regions:
  left: {from_mesh: left}
  right: {from_mesh: right}
  notch: {from_mesh: notch}

materials:
  glass:
    model: phase_field
    E: 210 GPa
    nu: 0.3
    Gc: 3 N/m
    l0: 0.01 mm
    pf_model: AT2

assignments:
  - material: glass
    region: body

initial_conditions:
  - field: damage
    region: notch
    value: 1.0

boundary_conditions:
  - name: clamp_left
    type: fix
    region: left
    dof: x
  - name: pull_right
    type: prescribe
    region: right
    dof: x
    value: 0.01 mm

analysis_steps:
  - name: load
    type: quasi_static
    controls:
      increments: 100
    active_boundary_conditions: [clamp_left, pull_right]

outputs:
  fields:
    - {name: damage, every: 5}
    - {name: displacement, every: 5}
  history:
    - {name: reaction_force, region: right, dof: x, every: 1}
  visuals:
    thumbnail: true
```

This schema currently provides a stable contract adapter and validation
surface. Solver execution still uses existing compatibility runners; the v2
deck is not a new weak-form language and does not promote beta
plasticity/cohesive/interface workflows.
Use `problem_spec_to_schema_v2_dict()` internally when migration tooling needs
to serialize a compiled `ProblemSpec` back to the v2 YAML dictionary shape.
Validate a v2 deck through the workflow contract with:

```bash
python -m phast run path/to/schema_v2.yaml --validate-only
```

Running schema-v2 decks directly is intentionally narrow. Supported
quasi-static phase-field fracture decks can execute by lowering to the
existing v1 `run_config` YAML shape and calling the unchanged runner. Promoted
solid-mechanics examples can also execute when the deck declares
`solver.type: solid_mechanics` and a promoted `solver.example`, for example
`solid_mechanics.linear_plate`; PhAST validates the v2 contract, lowers it to
the existing solid-mechanics YAML runner shape, and calls the unchanged
compatibility runner. Unsupported fracture schema-v2 decks, including
explicit/dynamic v2 decks, remain validate-only.
The v2 validator checks contract names, declared regions, duplicate region,
material, analysis-step, boundary-condition, field/history output, and
postprocess output names, duplicate material assignments, active
boundary-condition references, ambiguous multi-material assignments, duplicate
mesh-to-region mappings, invalid component indices, conflicting displacement
Dirichlet conditions, and execution routes without constructing meshes or
solvers. It also rejects
currently unsupported runner-family combinations such as solid-mechanics
solvers with phase-field materials, phase-field boundary conditions on solid
mechanics solvers, fracture solvers with solid-mechanics materials, or output
requests that are not supported by the selected runner family.

Plasticity/cohesive/interface validation remains intentionally narrower than
the fracture and promoted solid-mechanics YAML path. The curated beta manifest
at
`configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml`
can now be adapted internally with `problem_specs_from_yaml()` into
`validation_script` `ProblemSpec` entries, one per listed run. That proves the
current validation artifacts share the workflow contract surface without
changing `run_config.py`, executing arbitrary YAML commands, or promoting a
general plasticity/cohesive/interface problem schema. Ordinary schema-v2 decks
that use beta material models such as `j2_plasticity` outside a curated
`validation_script` contract are reported by `validate_problem_spec()` as
unsupported.

Allowlisted curated YAML execution paths include the standalone J2 validation
and the diffuse-interphase validation:

```bash
python -m phast run configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml \
  --validation-id j2_validation \
  --output_dir runs/plasticity_interface/j2_validation

python -m phast run configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml \
  --validation-id diffuse_interphase \
  --output_dir runs/plasticity_interface/diffuse_interphase
```

This dispatch calls a known internal validation entry point. It does not
execute arbitrary launcher strings from YAML. Other plasticity/cohesive/
interface manifest entries are contract-adapted and remain runnable through
their existing explicit scripts until each one is deliberately wired and
validated through the common CLI.

## Python workflow API

The fluent `phast.Problem` API shares the same internal workflow contract as
schema-v2 YAML. Supported production execution is intentionally limited to
workflow shapes that lower to existing runners:

- promoted solid-mechanics examples such as `solid_mechanics.linear_plate`
- quasi-static phase-field fracture workflows that lower to v1 `run_config`

Example imported-mesh quasi-static fracture workflow:

```python
import phast

result = (
    phast.Problem("notched plate")
    .mesh("meshes/notched_plate.msh")
    .region("body", kind="domain")
    .region("bottom", from_mesh="bottom")
    .region("top", from_mesh="top")
    .material("glass", region="body", E=210000.0, nu=0.3, Gc=2.7, l0=0.25)
    .boundary_condition("fix", region="bottom", dof="x", name="clamp_x")
    .boundary_condition("fix", region="bottom", dof="y", name="clamp_y")
    .boundary_condition("displacement", region="top", dof="y", value=0.001, name="pull")
    .analysis_step(
        "load",
        kind="quasi_static",
        controls={"protocol": "simple", "num_steps": 1, "dt": 1.0},
        active_boundary_conditions=["clamp_x", "clamp_y", "pull"],
    )
    .solver("quasi_static", max_stagger=1)
    .outputs(histories=[{"name": "reaction_force", "region": "bottom", "dof": "y"}])
    .run(output_dir="runs/notched_plate", return_result=True)
)
```

For setup checks before running, use:

```python
problem = phast.Problem("notched plate").mesh("meshes/notched_plate.msh")
summary = problem.validate_setup()
problem.preview(output="runs/notched_plate/setup_preview.png")
```

`validate_setup()` resolves declared workflow regions against mesh groups when
an imported mesh is present. `preview()` and `plot_setup()` write a static PNG
setup artifact without changing solver outputs. Unsupported fluent workflow
shapes are rejected early instead of being translated into a generic FEM or
weak-form compiler.

## Internal workflow contract

Current YAML and the fluent `phast.Problem` API adapt to the internal
`ProblemSpec` contract before future workflow layers decide how to execute
them. The additive workflow capability registry records supported public
contract names for solver kinds, analysis-step kinds, material models, and
boundary-condition kinds. The registry is advisory in this stage: it validates
the contract surface in tests, but existing `run_config.py` solver loops remain
unchanged. The internal execution-plan adapter can route supported
`ProblemSpec` instances to the existing compatibility runner family without
invoking the solver. The returned plan records the execution boundary:
legacy v1 fracture decks, promoted solid-mechanics decks, supported
schema-v2 quasi-static fracture decks, promoted schema-v2 solid-mechanics
decks, and the existing fluent `Problem` adapter use
`execution_boundary: existing_runner`, while unsupported schema-v2 decks use
`execution_boundary: validate_only` until each compatibility adapter is
designed and tested.

YAML-backed v1 fracture and promoted solid-mechanics specs also retain their
original `source_path`, so `ProblemSpec.run()` can safely delegate back to the
existing compatibility CLI:

```python
spec = problem_spec_from_yaml("examples/dynamic/B3_dynamic_sent/config.yaml")
spec.run(output_dir="runs/b3", validate_only=True)
```

This calls the same `python -m phast run <config>` entry point users already
run manually. Python-built specs should continue to use `Problem.run()`.
Schema-v2 fracture specs may always use `ProblemSpec.run(validate_only=True)`.
Supported quasi-static phase-field fracture specs can now execute through the
v1 `run_config` lowering adapter. Promoted schema-v2 solid-mechanics specs can
execute through the first compatibility adapter when they declare a supported
`solver.example`.
The internal region-reference validator checks that contract-level material,
initial-condition, boundary-condition, and history-output region references
are declared. Writing richer mesh/region provenance into run manifests remains
a separate output-contract change.
Public contract tests also compile YAML-first dynamic, quasi-static, and
promoted solid-mechanics examples to schema-v2 dictionaries and validate the
roundtrip without running solvers.
Use `validate_problem_spec()` internally to collect capability, region, and
execution-route issues without constructing meshes or solvers.
