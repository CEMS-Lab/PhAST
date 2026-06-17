# Public Workflow API

PhAST exposes a user-facing workflow API around domain nouns. The internal
validated contract still uses `*Spec` names in adapter and validation modules,
but users should normally think in terms of problems, regions, materials,
loads, steps, outputs, and results.

| Public noun | Purpose | Primary docs |
|---|---|---|
| `phast.Problem` | Fluent authoring entry point for new models. | [Python API](../user_guide/python_api.md) |
| `Geometry` / `Mesh` | Define generated geometry or import an existing mesh. | [Setting up problems](../user_guide/setup_problems.md) |
| `phast.Region` | Name physical groups, mesh sets, and reusable application regions. | [Python API](../user_guide/python_api.md) |
| `phast.Material` | Low-level material parameter container exported for direct material objects. | [Problem types](../user_guide/problem_types.md) |
| `Problem.material(...)` | Fluent material assignment helper for presets, constitutive parameters, and target regions. | [Python API](../user_guide/python_api.md) |
| `phast.InitialCondition` | Seed fields such as initial damage where supported. | [Python API](../user_guide/python_api.md) |
| `phast.BoundaryCondition` | Apply fix, prescribe, traction, symmetry, and Neumann-style conditions. | [Capability matrix](../user_guide/capability_matrix.md) |
| `phast.AnalysisStep` | Select the solution type, loading protocol, and active conditions. | [YAML workflow](../user_guide/yaml_workflow.md) |
| `phast.Outputs` | Request fields, histories, visuals, trajectories, and manifests. | [Output standards](../output_standards/index.md) |
| `phast.Result` / `phast.load_result()` | Inspect completed run directories without rerunning solvers. | [Results API](../user_guide/results_api.md) |
| `phast.ResultLoadError` | Clear error for missing result directories, unknown fields, and reserved postprocess methods. | [Results API](../user_guide/results_api.md) |

## Authoring Boundary

Use the fluent `phast.Problem` API to author new models. Use YAML decks for public examples, reproducibility, batch/HPC runs, and sharing exact simulations. The public examples keep YAML as the canonical rerun surface until each fluent lowering path is promoted and covered by tests.

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

The `ProblemSpec` contract is intentionally internal-facing. It lets PhAST
validate YAML, fluent Python, promoted solid-mechanics examples, and result
inspection through one common representation while not turning the project into an arbitrary weak-form compiler.
Use the [capability matrix](../user_guide/capability_matrix.md) as the public
boundary before documenting internal `*Spec` types on user-facing pages.

## Execution Boundary

Supported execution still routes through the curated solver paths documented in
the [capability matrix](../user_guide/capability_matrix.md). Schema-v2 and
fluent helpers validate and lower only where a supported runner exists. If a
workflow is marked beta, scaffold, optional-backend, or unsupported, keep it out
of public examples unless the corresponding contract tests and visual manifests
are present.

## Output Boundary

Every promoted public example should expose a flat, predictable result bundle:
`config.yaml`, `run_manifest.json`, `visual_manifest.json`, representative
PNG/MP4 artifacts, CSV histories where relevant, and Zarr-first trajectory
outputs when the run stores fields. See
[Output standards](../output_standards/index.md) for the artifact contract.
