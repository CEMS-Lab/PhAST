# Python API

Use the fluent `phast.Problem` API to author new models. Use YAML decks for public examples, reproducibility, batch/HPC runs, and sharing exact simulations.

The public Python API uses domain names: `Problem`, `Geometry`, `Mesh`,
`Region`, `Material`, `InitialCondition`, `BoundaryCondition`, `AnalysisStep`,
`SolverSettings`, `Outputs`, and `Result`. Internal validated objects still
use `*Spec` names where that makes adapter and validation code clearer.

The normal forward path is: author with `Problem`, compile or inspect the
`ProblemSpec`, save/validate a YAML deck when exact reproduction is needed,
then run through the supported runner and inspect the standard result
directory.

The existing fluent `Problem` API remains backward-compatible:

```python
from phast import Problem

problem = (
    Problem("SENT")
    .geometry("rectangular_sent", W=100, H=40, a=50, h_crack=0.5)
    .material("glass_borden", l0=0.5, energy_split="spectral")
    .fix("left", dof="x")
    .neumann("top", dof="y", value=1.0)
    .loading(protocol="simple", t_total=80e-6)
    .solver(dt_safety=0.8)
    .output(trajectory=True, h5_every=5)
)

spec = problem.to_spec()
```

Use an imported mesh with the same fluent API:

```python
problem = (
    Problem("Imported mesh SENT")
    .mesh("meshes/notched_plate.msh")
    .material("glass_borden", l0=0.5)
    .initial_condition("damage", region="notch", value=1.0)
    .boundary_condition("prescribe", region="right", dof="x", value=0.01)
    .analysis_step(
        "load",
        kind="quasi_static",
        controls={"protocol": "simple", "num_steps": 4},
    )
    .outputs(trajectory=True, h5_every=5, plots=True)
)

assert problem.to_spec().mesh.path == "meshes/notched_plate.msh"
assert problem.to_spec().initial_conditions[0].field == "damage"
assert problem.to_spec().boundary_conditions[0].kind == "prescribe"
assert problem.to_spec().analysis_steps[0].name == "load"
assert problem.to_spec().outputs.fields[0].name == "trajectory"
```

The fluent `Problem.initial_condition()` path currently supports damage
preseeding through the existing YAML-compatible `initial_conditions` config.
Other initial-state fields remain validation/design work until the solver paths
consume them directly.

`Problem.boundary_condition(kind, region=..., dof=..., value=...)` is the
domain-named equivalent of the existing `fix()`, `prescribe()`, and
`neumann()` convenience methods. The shorthand methods remain supported.
`Problem.analysis_step(name, kind=..., controls=...)` sets the primary step
through the existing loading and solver configuration path.
`Problem.outputs(...)` is the domain-named alias for the existing
`Problem.output(...)` method.

Workflow helper objects and the existing public `Material` object provide
clean names and convert to the internal contract with `to_spec()`:

```python
from phast import (
    AnalysisStep,
    BoundaryCondition,
    FieldOutput,
    Geometry,
    HistoryOutput,
    InitialCondition,
    Material,
    Mesh,
    Outputs,
    Postprocess,
    Region,
    SolverSettings,
)

plate = Geometry.rectangle(width=1.0, height=0.5, units="mm")
mesh = Mesh("meshes/notched_plate.msh", kind="gmsh", element_order=1)
left = Region("left", from_mesh="left")
glass = Material(E=210000.0, nu=0.3, Gc=2.7, l0=0.5)
notch_seed = InitialCondition.damage(region="notch", value=1.0)
pull = BoundaryCondition.displacement(
    name="pull_right",
    region="right",
    dof="x",
    value="0.01 mm",
)
step = AnalysisStep(
    "load",
    kind="quasi_static",
    controls={"increments": 100},
    active_boundary_conditions=["pull_right"],
)
solver = SolverSettings("quasi_static", nonlinear_tolerance=1e-8)
outputs = Outputs(
    fields=[
        FieldOutput("damage"),
        {"name": "displacement", "every": 5},
    ],
    history=[
        HistoryOutput("reaction_force", region="right", dof="x"),
    ],
    visuals={
        "thumbnail": True,
        "damage_final": Postprocess("damage_final", step=-1),
    },
)

geometry_spec = plate.to_spec()
mesh_spec = mesh.to_spec()
region_spec = left.to_spec()
material_spec = glass.to_spec(name="glass", region="body")
bc_spec = pull.to_spec()
solver_spec = solver.to_spec()
output_spec = outputs.to_spec()
```

`Material` remains the core public solver material class. Its `to_spec()`
method is an additive workflow-contract bridge; it does not replace material
construction in existing solvers.

`Problem.run()` remains backward-compatible and returns the solver object by
default. When a run directory is requested, use `return_result=True` to receive
the same read-only `Result` object exposed by `phast.load_result(path)`:

```python
result = problem.run(
    output_dir="runs/sent",
    verbose=False,
    return_result=True,
)
metadata = result.metadata()
fields = result.field_names()
```

The #700 production workflow layer supports the domain-named fluent surface for
promoted runner paths. The calls compile to `ProblemSpec` first; execution still
uses the existing supported runners rather than a universal weak-form compiler:

```python
result = (
    phast.Problem("notched plate")
    .mesh("mesh.msh")
    .region("body", from_mesh="Domain")
    .region("left", from_mesh="Left")
    .region("right", from_mesh="Right")
    .material("glass", region="body", E=210000.0, nu=0.3, Gc=2.7, l0=0.25)
    .boundary_condition("fix", region="left", dof="x", name="clamp")
    .boundary_condition(
        "displacement",
        region="right",
        dof="y",
        value=0.001,
        name="pull",
    )
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
    .run(output_dir="runs/notched_plate", return_result=True)
)
```

`Result.postprocess(...)` is available when users explicitly want to invoke the
existing postprocess CLI from Python:

```python
result.postprocess(fields=["damage", "energy"], skip_gif=True)
```

Direct `ProblemSpec.run()` is a guarded YAML compatibility bridge, not a new
Python solver API. Specs compiled from existing v1 fracture YAML or promoted
solid-mechanics YAML retain their original source path and can delegate to
`python -m phast run <config>`. Schema-v2 fracture specs can use
`ProblemSpec.run(validate_only=True)` for contract validation; supported
quasi-static phase-field fracture specs can execute through the v1 `run_config`
lowering adapter. Promoted schema-v2 solid-mechanics specs can execute when
they declare a supported `solver.example`. Specs created from the fluent Python
API do not have a YAML source path, so Python users should continue to run them
with `Problem.run()`.

Internal validation is available through `validate_problem_spec()`, which
collects unsupported capability names, missing region references, and missing
execution routes without constructing meshes or solvers.

For reproducible public examples and cluster runs, keep the checked-in
`config.yaml` as the canonical input deck and use the canonical contract in
`docs/user_guide/example_contract.md`. The compatibility references remain
`docs/STANDARD_OUTPUTS.md`, `docs/visualisation_requirements.md`, and
`docs/visualization-output.md`.
