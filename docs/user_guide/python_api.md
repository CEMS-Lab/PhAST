# Python API

Use the fluent `phast.Problem` API when you are designing a new model in
Python. Use YAML when the setup needs to become a reproducible input deck for
public examples, CI, shared runs, or HPC submission.

## Fluent API Mental Model

A `phast.Problem` is a forward-simulation recipe. You describe:

1. the geometry or imported mesh,
2. named regions,
3. materials assigned to regions,
4. initial and boundary conditions,
5. one or more analysis steps,
6. solver settings,
7. requested outputs,
8. where the run should write its result directory.

The usual workflow is:

```python
import phast

problem = (
    phast.Problem("linear plate")
    .geometry("structured_grid", nx=40, ny=12, length=1.0, height=0.2)
    .region("body", kind="domain")
    .material("steel", model="solid_mechanics", region="body", E=2.1e11, nu=0.3)
    .analysis_step("load", kind="solid_mechanics", controls={"tip_force_y": -1.0e3})
    .solver("solid_mechanics", example="solid_mechanics.linear_plate")
    .outputs(fields=["displacement", "von_mises"], histories=["response"], plots=True)
)

problem.validate_setup()
result = problem.run(output_dir="runs/linear_plate", return_result=True)

print(result.metadata())
print(result.history_names())
print(result.visuals())
```

`Problem.run(...)` routes only to supported solver paths. PhAST is not an
arbitrary weak-form compiler: if a workflow is outside the supported capability
matrix, validation should fail early rather than silently translating it into
an unsupported solve.

## Python or YAML?

| Task | Recommended surface |
|---|---|
| Explore a new setup interactively | Fluent `phast.Problem` |
| Write a script that builds several related models | Fluent `phast.Problem` |
| Share an exact example with another user | YAML `config.yaml` |
| Submit the same run to a cluster queue | YAML `config.yaml` |
| Keep a public benchmark reproducible over time | YAML `config.yaml` plus result manifests |
| Inspect a completed run | `phast.load_result(path)` |

Both paths write standard result directories that can be inspected with
`phast.load_result(...)`.

## Common Method Map

| Method | Purpose |
|---|---|
| `.geometry(...)` | Create a built-in geometry or mesh recipe. |
| `.mesh(...)` | Use an external Gmsh mesh. |
| `.region(...)` | Name a domain, boundary, or imported mesh group. |
| `.material(...)` | Assign material parameters to a region. |
| `.initial_condition(...)` | Seed supported initial fields such as damage. |
| `.boundary_condition(...)` | Apply named fixed, prescribed, traction, or force-style conditions. |
| `.fix(...)` | Convenience helper for fixed displacement components. |
| `.prescribe(...)` | Convenience helper for prescribed displacement components. |
| `.neumann(...)` | Convenience helper for force/traction-style loading. |
| `.analysis_step(...)` | Define the loading or solve step. |
| `.solver(...)` | Select solver family and numerical controls. |
| `.outputs(...)` | Request fields, histories, plots, and trajectories. |
| `.device(...)` | Select CPU/CUDA/MPS execution where supported. |
| `.validate_setup()` | Check regions and setup consistency before running. |
| `.preview(...)` / `.plot_setup(...)` | Write a setup preview image when supported. |
| `.run(...)` | Execute a supported forward workflow. |
| `.to_spec()` | Build the validated representation used by advanced checks and tooling. |

Most users only need the fluent methods above. Public examples keep YAML as
the canonical run artifact.

## Built-in Geometry Example

```python
import phast

problem = (
    phast.Problem("SENT plate")
    .geometry("rectangular_sent", W=100, H=40, a=50, h_crack=0.5, h_coarse=4.0)
    .material("glass_borden", l0=0.5, energy_split="spectral")
    .initial_condition("damage", region="notch", value=1.0)
    .boundary_condition("fix", region="left", dof="x", name="left_x")
    .boundary_condition("prescribe", region="right", dof="x", value=0.01, name="pull")
    .analysis_step(
        "load",
        kind="quasi_static",
        controls={"protocol": "simple", "num_steps": 4},
        active_boundary_conditions=["left_x", "pull"],
    )
    .solver("quasi_static", preconditioner="jacobi", backend="auto")
    .outputs(fields=["damage", "displacement"], histories=["reaction_force"], plots=True)
)

problem.validate_setup()
```

For published examples and benchmark reruns, prefer the checked-in YAML deck in
`examples/` so that every user starts from the same input file.

## Imported Mesh Example

External meshes should use named Gmsh physical groups. Inspect those names
before assigning materials and boundary conditions:

```python
import phast

summary = phast.inspect_mesh("meshes/notched_plate.msh")
print(summary["named_groups"])
```

Then bind those groups to workflow regions:

```python
problem = (
    phast.Problem("imported notched plate")
    .mesh("meshes/notched_plate.msh")
    .region("body", from_mesh="Domain")
    .region("left", from_mesh="Left")
    .region("right", from_mesh="Right")
    .material("glass", region="body", E=210000.0, nu=0.3, Gc=2.7, l0=0.25)
    .boundary_condition("fix", region="left", dof="xy", name="clamp")
    .boundary_condition("displacement", region="right", dof="y", value=0.001, name="pull")
    .analysis_step(
        "load",
        kind="quasi_static",
        controls={"protocol": "simple", "num_steps": 1},
        active_boundary_conditions=["clamp", "pull"],
    )
    .outputs(
        fields=["damage", "displacement"],
        histories=[{"name": "reaction_force", "region": "right", "dof": "y"}],
        plots=True,
    )
)

problem.validate_setup()
problem.preview(output="runs/imported_notched_plate/setup.png")
```

`validate_setup()` checks declared regions against mesh groups when enough mesh
metadata is available. `preview()` writes a static setup image for review before
launching a longer solve.

## Boundary Conditions

Use explicit named boundary conditions when a load step activates several
conditions:

```python
problem = (
    phast.Problem("plate")
    .boundary_condition("fix", region="left", dof="xy", name="clamp")
    .boundary_condition("displacement", region="right", dof="x", value=0.01, name="pull")
    .boundary_condition("traction", region="top", dof="y", value=1.0, name="top_load")
    .analysis_step(
        "load",
        kind="quasi_static",
        controls={"num_steps": 10},
        active_boundary_conditions=["clamp", "pull", "top_load"],
    )
)
```

Use shorthand helpers for compact examples:

```python
problem.fix("left", dof="xy")
problem.prescribe("right", dof="x", value=0.01)
problem.neumann("top", dof="y", value=1.0)
```

For displacement-style conditions, specify the degree of freedom explicitly
with `dof="x"`, `dof="y"`, or `dof="xy"`.

## Outputs and Result Inspection

Request outputs in the model:

```python
problem.outputs(
    fields=["damage", "displacement"],
    histories=[{"name": "reaction_force", "region": "right", "dof": "y"}],
    plots=True,
    trajectory=True,
)
```

Inspect a completed run without rerunning it:

```python
import phast

result = phast.load_result("runs/notched_plate")
print(result.metadata())
print(result.manifest())
print(result.history_names())
print(result.visuals())
print(result.field_names())

if result.has_field("damage"):
    damage = result.field("damage", step=-1)
```

`Result` is read-only. It exposes manifests, metadata, CSV histories, visual
artifacts, mesh metadata, and stored trajectory fields. It does not silently
derive missing fields.

## Capability Boundaries

The fluent API is the recommended Python authoring surface, but execution is
bounded by the supported workflows in the capability matrix. Current public
examples keep YAML as the reproducible run surface. If you need a durable input
deck, author with Python if convenient, then record the final setup as a YAML
example with a standard result bundle.

Next pages:

- [Fluent authoring tutorial](../tutorial/fluent_authoring_guide.md)
- [Setting up new problems](setup_problems.md)
- [YAML workflow](yaml_workflow.md)
- [Results API](results_api.md)
- [Capability matrix](capability_matrix.md)
