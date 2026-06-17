# Build Your First Model with `phast.Problem`

This tutorial shows the fluent Python workflow from model definition to result
inspection. It uses a small solid-mechanics example because it is fast enough
for a first local run.

If you have used Abaqus, COMSOL, FEniCS, or a similar FEM package, read
`phast.Problem` as the model builder. You define the mesh, named regions,
materials, boundary conditions, analysis step, solver controls, and output
requests, then submit the run.

PhAST is unit-agnostic. This tutorial uses one consistent set of units inside
the example; your own models must do the same.

## 1. Create the Model

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
```

The model reads like a finite-element setup:

| Call | Meaning |
|---|---|
| `.geometry(...)` | Build a structured plate mesh. |
| `.region(...)` | Name the domain region. |
| `.material(...)` | Assign elastic material properties. |
| `.analysis_step(...)` | Define the loading step. |
| `.solver(...)` | Select the promoted solid-mechanics runner. |
| `.outputs(...)` | Request fields, histories, and plots. |

## 2. Validate Before Running

```python
problem.validate_setup()
```

Validation checks the setup without launching the full solve. For imported
meshes, this is where you catch missing or misspelled region names before
submitting a longer job.

For YAML configurations, the equivalent preflight is:

```bash
python -m phast run examples/solid_mechanics_beta/linear_plate/config.yaml --validate-only
```

## 3. Run the Solve

```python
result = problem.run(output_dir="runs/linear_plate", return_result=True)
```

The run writes a standard result directory. The exact files depend on the
workflow, but promoted examples should include metadata, manifests, history
CSVs, and visual artifacts.

## 4. Inspect the Result

```python
print(result.metadata())
print(result.history_names())
print(result.visuals())
```

You can load the same run later without rerunning the solver:

```python
import phast

result = phast.load_result("runs/linear_plate")
```

`Result` is read-only. It reports what was written by the run; it does not
invent missing derived fields.

## 5. Use YAML for Reproducible Runs

The equivalent public example is kept as a YAML declarative configuration:

```bash
python -m phast run examples/solid_mechanics_beta/linear_plate/config.yaml \
  --output_dir runs/linear_plate
```

Use the fluent API while designing the model. Use YAML when the setup is ready
to share, rerun in CI, or submit to a cluster queue.

## 6. Imported Mesh Pattern

For a mesh generated outside PhAST:

```python
import phast

summary = phast.inspect_mesh("meshes/notched_plate.msh")
print(summary["named_groups"])

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
    .outputs(fields=["damage", "displacement"], histories=["reaction_force"], plots=True)
)

problem.validate_setup()
problem.preview(output="runs/imported_notched_plate/setup.png")
```

Use `inspect_mesh(...)` first, bind mesh groups to clean region names, then use
those region names everywhere else.

## 7. From Tutorial to Your Own Case

For a new engineering or research case:

1. Start from the closest solved example in `examples/`.
2. Confirm the example validates with `--validate-only`.
3. Recreate the setup with `phast.Problem` if you want interactive Python
   authoring.
4. Keep the final reproducible run as `config.yaml`.
5. Request the outputs you need before launching the full solve.
6. Keep the whole result directory; it is the equivalent of the solver result
   database and is what `phast.load_result(...)` reads.

Do not treat the fluent API as an unrestricted symbolic weak-form system. It is
an authoring interface for supported PhAST workflows.

## Next Steps

- [Python API reference](../user_guide/python_api.md)
- [Setting up new problems](../user_guide/setup_problems.md)
- [YAML workflow](../user_guide/yaml_workflow.md)
- [Results API](../user_guide/results_api.md)
- [Capability matrix](../user_guide/capability_matrix.md)
