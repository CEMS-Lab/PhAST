# Setting Up New Problems

This page explains the pieces of a PhAST forward model: geometry or mesh,
regions, materials, initial conditions, boundary conditions, analysis steps,
solver controls, outputs, validation, and result inspection.

Use the fluent `phast.Problem` API while designing a new setup. Use YAML decks
for reproducible public examples, shared runs, CI, and HPC submission.

## FEM Workflow Map

If you are coming from Abaqus, COMSOL, FEniCS, deal.II, or another FEM code,
the PhAST workflow is the same sequence with different names:

| FEM concept | Abaqus/COMSOL-style term | PhAST fluent API | YAML deck location |
|---|---|---|---|
| Model or study | Model database / component / study | `phast.Problem("name")` | `problem:` |
| Part geometry | Part / geometry sequence | `.geometry(...)` | `geometry:` |
| Imported mesh | Mesh file / imported mesh | `.mesh("mesh.msh")` | `geometry.mesh_path` |
| Named selections | Sets / physical groups / boundaries | `.region(...)` | geometry groups and node sets |
| Material definition | Material card | `.material(...)` | `material:` |
| Material assignment | Section assignment / domain material | `.material(..., region="body")` | material plus region/node-set references |
| Initial crack/notch state | Initial field / predefined field | `.initial_condition("damage", ...)` | `initial_conditions:` |
| Supports | Dirichlet BC / fixed constraint | `.fix(...)` or `.boundary_condition("fix", ...)` | `boundary_conditions:` |
| Prescribed displacement | Displacement BC | `.prescribe(...)` or `.boundary_condition("displacement", ...)` | `boundary_conditions:` |
| Loads/tractions | Neumann load / boundary load | `.neumann(...)` or `.boundary_condition("traction", ...)` | `boundary_conditions:` |
| Step/study | Static, transient, or dynamic step | `.analysis_step(...)` | `loading:` and `solver:` |
| Solver controls | Time step, tolerances, backend | `.solver(...)` | `solver:` |
| Field/history output | Output requests / probes | `.outputs(...)` | `output:` |
| Job submission | Run study / submit job | `.run(...)` or `python -m phast run ...` | CLI command |
| Results | ODB/MED/XDMF/result database | `phast.load_result(path)` | result directory |

PhAST is intentionally narrower than a general-purpose commercial FEM GUI: the
public API routes to supported phase-field fracture and solid-mechanics
workflows. It does not compile arbitrary weak forms from user text.

## Units

PhAST is unit-agnostic, like Abaqus. It does not convert units for you. Choose
one consistent unit system and use it everywhere:

| Quantity | SI example | mm-N-MPa style example |
|---|---|---|
| Length | m | mm |
| Force | N | N |
| Stress/modulus | Pa | MPa = N/mm^2 |
| Fracture energy `Gc` | J/m^2 | N/mm |
| Density | kg/m^3 | tonne/mm^3 when using mm-N-s |
| Time | s | s |

Before running a new case, check that geometry dimensions, material properties,
loads, density, fracture energy, and time-step controls are all expressed in
the same system. The result manifests record the inputs; they do not infer or
repair inconsistent units.

## Setup Checklist

| Step | Question to answer | PhAST surface |
|---|---|---|
| Geometry or mesh | Is this a built-in geometry or an imported Gmsh mesh? | `.geometry(...)` or `.mesh(...)` |
| Regions | What parts of the mesh receive materials, loads, supports, or outputs? | `.region(...)` |
| Materials | Which material model and parameters apply to each region? | `.material(...)` |
| Initial conditions | Does the run start with seeded damage or another supported field? | `.initial_condition(...)` |
| Boundary conditions | Which supports and loads are active? | `.boundary_condition(...)`, `.fix(...)`, `.prescribe(...)`, `.neumann(...)` |
| Analysis step | Is this explicit dynamics, quasi-static fracture, or solid mechanics? | `.analysis_step(...)` |
| Solver | Which backend, tolerance, and solver family should be used? | `.solver(...)` |
| Outputs | Which fields, histories, plots, and trajectory stores are needed? | `.outputs(...)` |
| Preflight | Are region names and requested outputs valid? | `.validate_setup()`, `--validate-only` |
| Inspection | How will the completed run be read? | `phast.load_result(path)` |

## Minimal Fluent Setup

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
```

For the same workflow as a durable input deck:

```bash
python -m phast run examples/solid_mechanics/linear_plate/config.yaml \
  --output_dir runs/linear_plate
```

## Start From a Solved Example

The fastest way to build a trustworthy model is to clone a solved public
example, then change one part at a time:

1. Choose the closest example from `examples/`.
2. Run `python -m phast run <example>/config.yaml --validate-only`.
3. Read the example `README.md` and inspect its checked-in visuals.
4. Copy `config.yaml` into a new working directory.
5. Change geometry or mesh first, then validate again.
6. Change material parameters, then validate again.
7. Change boundary conditions or loading, then validate again.
8. Run a short smoke solve before launching the full case.
9. Inspect the result with `phast.load_result(...)`.

Examples are YAML-first because they are reproducible. When available,
`fluent_setup.py` shows the same model in the Python authoring style.

## Geometry and Meshes

Use built-in geometry names for common benchmark shapes:

```python
problem.geometry("rectangular_sent", W=100, H=40, a=50, h_crack=0.5, h_coarse=4.0)
```

The easiest way to discover geometry arguments is to start from a solved YAML
example and inspect it:

```bash
python -m phast explain-config examples/dynamic/B2_kalthoff_winkler/config.yaml
python -m phast explain-config examples/quasistatic/notched_holed_plate/config.yaml
```

The example `config.yaml` files are the supported reference for each public
geometry recipe.

Use an imported Gmsh mesh when the geometry comes from CAD, Gmsh, or another
meshing workflow:

```python
summary = phast.inspect_mesh("meshes/notched_plate.msh")
print(summary["named_groups"])

problem = (
    phast.Problem("imported plate")
    .mesh("meshes/notched_plate.msh")
    .region("body", from_mesh="Domain")
    .region("left", from_mesh="Left")
    .region("right", from_mesh="Right")
)
```

External meshes should preserve named physical groups for all boundaries used
by materials, boundary conditions, initial conditions, and history outputs.
In Gmsh, those names usually come from `Physical Surface` and `Physical Curve`
declarations:

```text
Physical Surface("Domain") = {1};
Physical Curve("Left") = {2};
Physical Curve("Right") = {3};
```

Bind them to PhAST regions by name:

```python
problem = (
    phast.Problem("imported plate")
    .mesh("meshes/notched_plate.msh")
    .region("body", from_mesh="Domain")
    .region("left", from_mesh="Left")
    .region("right", from_mesh="Right")
)
```

Abaqus `NSET`/`ELSET`, COMSOL selections, and FEniCS boundary markers should be
exported or converted into equivalent Gmsh physical groups before use in PhAST.

## Regions

Regions give stable names to domains and boundaries:

```python
problem.region("body", kind="domain")
problem.region("left", from_mesh="Left")
problem.region("right", from_mesh="Right")
problem.region("notch", from_mesh="Notch")
```

Use short domain names in your model (`left`, `right`, `notch`) even if the
mesh group names are verbose. That keeps materials, loads, and histories easy
to read.

## Materials

Assign materials to regions:

```python
problem.material("glass", region="body", E=210000.0, nu=0.3, Gc=2.7, l0=0.25)
```

For phase-field fracture, common parameters are:

| Parameter | Meaning |
|---|---|
| `E` | Young's modulus |
| `nu` | Poisson ratio |
| `Gc` | Critical fracture energy |
| `l0` | Phase-field length scale |
| `pf_model` | Damage model, for example `AT1` or `AT2` |
| `energy_split` | Tensile/compressive split, for example `spectral` or `amor` |
| `eta_residual` | Residual stiffness used after high damage |

For public benchmark reproduction, start from the checked-in `config.yaml`.
For a real engineering study, choose material parameters from the relevant
material data source and document the unit system with the run.

## Initial Conditions

Seed an initial notch or damaged region when the workflow supports damage
initialization:

```python
problem.initial_condition("damage", region="notch", value=1.0)
```

The public fracture examples use this path to mark a pre-existing notch before
the load step begins.

## Boundary Conditions

Use explicit named boundary conditions for clarity:

```python
problem.boundary_condition("fix", region="left", dof="xy", name="clamp")
problem.boundary_condition("displacement", region="right", dof="x", value=0.01, name="pull")
problem.boundary_condition("traction", region="top", dof="y", value=1.0, name="top_load")
```

Use shorthand helpers in compact examples:

```python
problem.fix("left", dof="xy")
problem.prescribe("right", dof="x", value=0.01)
problem.neumann("top", dof="y", value=1.0)
```

Always specify a degree of freedom for displacement-style conditions:
`dof="x"`, `dof="y"`, or `dof="xy"`.

## Analysis Steps and Solver Settings

The analysis step describes the physical solve:

```python
problem.analysis_step(
    "load",
    kind="quasi_static",
    controls={"protocol": "simple", "num_steps": 10},
    active_boundary_conditions=["clamp", "pull"],
)
```

Current public fluent workflows use one primary analysis step. For staged
loading, use the loading protocol supported by that workflow, or encode the
staged run in the canonical YAML deck for that example. Do not assume chained
`.analysis_step(...)` calls create a full Abaqus-style step sequence unless the
specific workflow documents that behavior.

Common controls:

| Workflow kind | Typical controls | Meaning |
|---|---|---|
| `solid_mechanics` | `tip_force_y`, load magnitude or displacement controls used by the promoted solid runner | Small solid-mechanics examples and material-kernel demonstrations. |
| `quasi_static` | `protocol`, `num_steps`, `dt`, active boundary conditions | Staggered phase-field fracture loading. |
| `explicit` | final time, time-step safety, output cadence, damage update cadence | Dynamic fracture examples; public YAML decks expose the full control set. |

The solver settings select numerical controls:

```python
problem.solver("quasi_static", backend="auto", preconditioner="jacobi", max_stagger=500)
```

For explicit dynamics, the YAML examples expose the full time-step and output
cadence controls and are the recommended starting point for reproducible runs.

## Outputs

Request fields, histories, plots, and trajectory stores:

```python
problem.outputs(
    fields=["damage", "displacement"],
    histories=[{"name": "reaction_force", "region": "right", "dof": "y"}],
    plots=True,
    trajectory=True,
)
```

A promoted example result directory should include the input deck, lockfile,
metadata, history CSVs, standard figures, and a visual manifest. See
[Example result contract](example_contract.md) for output conventions.

## Validate, Run, Inspect

Validate a Python-authored setup:

```python
problem.validate_setup()
problem.preview(output="runs/my_case/setup.png")
```

Validate a YAML deck:

```bash
python -m phast run examples/quasistatic/notched_holed_plate/config.yaml --validate-only
python -m phast explain-config examples/quasistatic/notched_holed_plate/config.yaml
```

Run a YAML deck:

```bash
python -m phast run examples/quasistatic/notched_holed_plate/config.yaml \
  --output_dir runs/notched_holed_plate
```

Inspect the result:

```python
import phast

result = phast.load_result("runs/notched_holed_plate")
print(result.metadata())
print(result.history_names())
print(result.visuals())
```

## What Makes a Good Public Example?

Each public example should be boringly predictable:

- one `config.yaml` input deck,
- one optional `fluent_setup.py` companion when the fluent path is promoted,
- setup visual,
- final field plots,
- response or energy histories,
- animation when the response is meaningful,
- `run_manifest.json`, `run_metadata.json`, and lockfile,
- clear README with the command, runtime expectation, and expected outputs.

Examples that still need large raw trajectories, private HPC archaeology, or
custom one-off scripts should stay out of the public examples tree until they
are promoted.
