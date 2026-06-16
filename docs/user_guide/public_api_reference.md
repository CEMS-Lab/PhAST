# Public API Reference

PhAST’s public API is organized around **problem-building nouns** and explicit result inspection.
Use `phast.Problem` for model authoring and YAML/deck runners for public reproducibility.

## `phast.Problem`

`Problem` is the fluent authoring entry point:

- geometry/mesh definition
- region declaration
- material assignment
- initial conditions
- boundary/loading setup
- outputs
- execution

```python
import phast

problem = (
    phast.Problem("PMMA Branching")
    .geometry("rectangular_sent", W=32, H=16, a=16, h_crack=0.15, branching=True)
    .material("pmma_bleyer", l0=0.25, energy_split="amor", pf_model="AT1")
    .fix("bottom", dof="xy")
    .boundary_condition("prescribe", region="top", dof="y", value=0.04, name="pull")
    .analysis_step("load", kind="quasi_static", controls={"protocol": "simple", "num_steps": 1})
    .outputs(fields=["damage", "displacement"], histories=["response"], plots=True)
)

result = problem.run(output_dir="runs/pmma_branching", return_result=True)
```

Common methods:

- `.to_spec()`: convert to internal contract for validation/adapters
- `.save(path)` / `Problem.from_yaml(path)`
- `.validate_setup()`: optional preflight checks for mesh-bound regions
- `.preview(output="...")` / `.plot_setup(...)`: setup visuals for geometry/BC setup review
- `.run(output_dir=..., return_result=True)` for supported promoted/public paths

## `phast.Region`

`Region` defines reusable application regions used by material assignment, BCs, and output requests.

```python
import phast

region = phast.Region("left", from_mesh="Left")
material = phast.Material(E=210000.0, nu=0.3, Gc=2.7, l0=0.5)
problem = phast.Problem("notched").region("left", kind="domain").material("glass", region="left")
```

Use `Problem.region(...)` for direct fluent workflows; construct `Region(...)` explicitly when building
declarative reusable recipes.

## `phast.Material`

`Material` carries solver-accepted material properties and presets.
For fracture runs, preset models remain available through `Problem.material(...)` and schema-v2 adapters.

```python
from phast import Material

mat = Material(E=210000.0, nu=0.30, Gc=3e-3, l0=0.5)
mat.to_spec()
```

`phast.Material` is also the public container used by declarative API surfaces and examples that mirror runner conventions.

## `phast.InitialCondition`

Initial conditions initialize field states (current public support:
initial damage seeding and related scalar starts in supported fracture workflows).

```python
from phast import InitialCondition

ic = InitialCondition.damage(region="notch", value=1.0)
```

In fluent workflows, call:

```python
problem = (
    phast.Problem("notched plate")
    .initial_condition("damage", region="notch", value=1.0)
)
```

## `phast.BoundaryCondition`

Boundary primitives express support, prescribed displacements, and force-style loads.
The preferred workflow is to set them through the domain names on `Problem`.

```python
from phast import BoundaryCondition

bc = BoundaryCondition.displacement(name="pull", region="right", dof="y", value=0.02)
```

Accepted names and adapters are documented in
[`docs/user_guide/capability_matrix.md`](capability_matrix.md) and `Problem.boundary_condition(...)`.

## `phast.AnalysisStep`

`AnalysisStep` captures solution intent (`quasi_static`, `explicit`) and
step controls.

```python
from phast import AnalysisStep

step = AnalysisStep("load", kind="quasi_static", controls={"protocol": "simple", "num_steps": 1})
```

Fluent equivalent:

```python
problem = (
    phast.Problem("notched")
    .analysis_step("load", kind="quasi_static", controls={"protocol": "simple", "num_steps": 1})
)
```

## `phast.Outputs` and result artifacts

`Outputs` describes requested numeric output and media artifacts.

```python
from phast import Outputs, FieldOutput, HistoryOutput, Postprocess

outputs = Outputs(
    fields=[FieldOutput("damage"), "displacement"],
    histories=[HistoryOutput("reaction_force", region="right", dof="y")],
    visuals={"thumbnail": True, "damage_final": Postprocess("damage_final")},
)
```

Canonical stored outputs include CSV histories, manifest files, and raw trajectory
fields when requested by the run configuration.

## `phast.Result`

Inspect completed runs without rerunning:

```python
import phast

result = phast.load_result("runs/miehe_tension")
print(result.metadata())
print(result.manifest())
print(result.history_names())
print(result.field_names())
print(result.visuals())
```

`Result` methods include:

- `metadata()`
- `manifest()`
- `mesh()`
- `history_names()`
- `history(name)`
- `visuals()`
- `field_names()`
- `has_field(name)`
- `field(name, step=-1)`

`field(name)` returns **stored raw trajectory fields only** (for example `damage` and
`displacement` when written). Derived quantities (for example von Mises, response transforms)
are only available after post-processing and are not silently synthesized by `field()`.

## `phast.load_result(path)`

Use the load API for inspection:

```python
result = phast.load_result("runs/notched_holed_plate")
```

It returns a read-only result handle with provenance, history, and discovered artifacts.

## Exceptions

- `ResultLoadError`: missing path, invalid field/history names, or unavailable postprocess actions.

## Result boundaries and execution policy

- For public reproducibility and public examples: YAML decks are the canonical distributed artifact.
- For authoring and local iteration: fluent `Problem` is preferred.
- For execution/reproduction: `phast.Problem.run(...)` for supported fluent paths,
  `python -m phast run <config.yaml>` for public deck workflows.
- The API is **not** a universal weak-form FEM compiler; supported paths are bounded by
  [capability matrix](capability_matrix.md) and adapter execution routing.
