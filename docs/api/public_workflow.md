# Public Workflow API

PhAST exposes a user-facing workflow API around domain nouns. The validation
and adapter layers use `*Spec` names internally, but users should think in
terms of problems, regions, materials, loads, steps, outputs, and results.

| Public noun | Purpose | Primary docs |
|---|---|---|
| `phast.Problem` | Fluent authoring entry point for new models. | [Python API](../user_guide/python_api.md) |
| `Geometry` / `Mesh` | Define generated geometry or import an existing mesh. | [Setting up problems](../user_guide/setup_problems.md) |
| `phast.Region` | Name physical groups, mesh sets, and reusable application regions. | [Python API](../user_guide/python_api.md) |
| `phast.Material` | Material parameter container used by the public API and declarative adapters. | [Configuration](../user_guide/configuration.md) |
| `Problem.material(...)` | Fluent material assignment helper for presets, constitutive parameters, and target regions. | [Python API](../user_guide/python_api.md) |
| `phast.InitialCondition` | Seed fields such as initial damage where supported. | [Python API](../user_guide/python_api.md) |
| `phast.BoundaryCondition` | Apply fix, prescribe, traction, symmetry, and Neumann-style conditions. | [Capability matrix](../user_guide/capability_matrix.md) |
| `phast.AnalysisStep` | Select the solution type, loading protocol, and active conditions. | [YAML workflow](../user_guide/yaml_workflow.md) |
| `phast.Outputs` | Request fields, histories, visuals, trajectories, and manifests. | [Curated example contract](../user_guide/example_contract.md) |
| `phast.Result` / `phast.load_result()` | Inspect completed run directories without rerunning solvers. | [Public API reference](../user_guide/public_api_reference.md) |
| `phast.ResultLoadError` | Clear error for missing result directories, unknown fields, and reserved postprocess methods. | [Public API reference](../user_guide/public_api_reference.md) |

## Authoring Boundary

Use the fluent `phast.Problem` API to author new models. Use YAML
configurations for public examples, reproducibility, batch/HPC runs, and
sharing reviewable simulations. Public examples retain YAML as the primary
rerun interface unless an equivalent fluent pathway is documented and tested.

```python
import phast

problem = (
    phast.Problem("notched plate")
    .mesh("mesh.msh")
    .region("body", from_mesh="Domain")
    .region("left", from_mesh="Left")
    .region("right", from_mesh="Right")
    .material("glass", region="body", E=210000.0, nu=0.3, Gc=2.7, l0=0.25)
    .boundary_condition("fix", region="left", dof="x", name="clamp")
    .boundary_condition("displacement", region="right", dof="y", value=0.001, name="pull")
    .analysis_step(
        "load",
        kind="quasi_static",
        controls={"protocol": "simple", "num_steps": 1},
        active_boundary_conditions=["clamp", "pull"],
    )
    .outputs(fields=["damage", "displacement"], histories=["reaction_force"], plots=True)
)

spec = problem.to_spec()
```

The `ProblemSpec` contract is an implementation detail of the workflow layer.
It lets PhAST validate YAML, fluent Python, curated solid-mechanics
examples, and result inspection through one common representation while
keeping the public interface centered on domain concepts rather than
implementation data
structures. Use the [capability matrix](../user_guide/capability_matrix.md) as
the public boundary for supported workflows.

## Execution Boundary

Supported execution routes through the curated solver paths documented in the
[capability matrix](../user_guide/capability_matrix.md). Schema-v2 and fluent
helpers validate and lower only where a supported runner exists. If a workflow
is marked beta, scaffold, optional-backend, or unsupported, keep it out of
public examples unless the corresponding contract tests and visual manifests
are present.

For a promoted solid-mechanics example, advanced tooling may execute a
Python-built `ProblemSpec` through the same YAML runner. From the repository
root, the linear-plate fluent companion provides a complete setup:

```bash
PYTHONPATH=src:. python examples/solid_mechanics_beta/linear_plate/run_fluent.py --run --output-dir runs/linear_plate_python
```

The script reuses `fluent_setup.build_problem()` and calls
`run_problem_spec(problem.to_spec(), output_dir=...)`. Calling
`run_problem_spec(spec, validate_only=True)` checks the lowered YAML without
solving or writing a result directory. This bridge is limited to Python-built
specs for the published linear-plate runner. That runner clamps the left edge
and applies a vertical point force at the right-edge mid-height node; it does
not apply user-declared boundary conditions. Additional materials, load steps,
initial conditions, imported meshes, and options the runner cannot represent
are rejected before execution. This does not make arbitrary `ProblemSpec`
instances executable. For routine use, prefer `Problem.run(...)` or the
checked-in `config.yaml`.

The linear-plate companion declares `units="m"` in `Problem.geometry(...)`
to match its SI input values and metre-labelled displacement output. This is
metadata for the built-in geometry path, not automatic unit conversion.

## Output Boundary

Every curated public example should expose a flat, predictable result bundle:
`config.yaml`, `run_manifest.json`, `visual_manifest.json`, representative
PNG/MP4 artifacts, CSV histories where relevant, and Zarr-first trajectory
outputs when the run stores fields. See
[Curated example contract](../user_guide/example_contract.md) for the artifact
contract.
