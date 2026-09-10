# Tutorials

## Student Notebook Sequence

1. [Dynamic SENT: inspect a complete public example](notebook_dynamic_sent.ipynb)
   connects an existing mesh, named regions, boundary conditions, YAML,
   explicit solver route, retained energy history, and visible crack-growth
   animation. It does not rerun the full dynamic calculation by default.
2. [SENT setup and two-step CPU workflow check](notebook_setup.ipynb)
   teaches geometry, meshing, named regions, boundary conditions, material,
   solver selection, and retained outputs. Its default run is not crack-growth
   validation.
3. [Mesh-resolution diagnostic](notebook_mesh_resolution.ipynb)
   samples an AT2 profile at several $h/\ell_0$ ratios. It is not a solved
   convergence study.
4. [Retained Miehe SENT results](notebook_retained_results.ipynb)
   examines checked-in load-displacement and damage evidence and states the
   current post-processing boundary.

Asymmetric three-point bending and L-shaped panel notebooks are not presented
as public benchmarks because the current repository does not retain the
benchmark-specific evidence required to support those claims.

This page is the onboarding map for new PhAST users. Start with the shortest
validation path, then move to Python authoring, YAML reproduction, and result
inspection.

## Start Here

Launch the step-by-step problem setup notebook in Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CEMS-Lab/PhAST/blob/main/docs/tutorial/problem_setup_walkthrough.ipynb)

The badge opens the current public notebook. Its installation cell checks out
the published `v0.16.2-arxiv.2606.23458` solver release, verifies the cached
checkout, and prints the resolved commit before installation.

| Tutorial | Time | What you learn |
|---|---:|---|
| [Getting started](../getting-started.md) | 10-20 min on a fresh machine | Create an environment, install PyTorch and PhAST, run `doctor`, validate a shipped example, and inspect a result. A prepared teaching environment is faster. |
| [Dynamic SENT example](notebook_dynamic_sent.ipynb) | 20-30 min | Inspect the existing B3 mesh, named regions, loading, explicit solver route, retained histories, and crack-growth animation without presenting retained evidence as a new run. |
| [Phase-field primer](01_phase_field_primer.md) | 15 min | Connect Griffith fracture energy, regularization, degradation, energy splits, history, and the staggered solve. |
| [Problem setup notebook](notebook_setup.ipynb) | 35-50 min | Predict the setup, create and inspect the geometry and named regions, apply conditions and solver settings, run a short solve, change one parameter, and interpret the artifacts. A rendered fallback supports sessions without a working runtime. |
| [Mesh-resolution diagnostic](notebook_mesh_resolution.ipynb) | 10 min | Interpret nodal sampling of an analytical AT2 profile; this is not an FEM convergence study. |
| [Retained-results notebook](notebook_retained_results.ipynb) | 10 min | Inspect checked-in Miehe result evidence without rerunning the full calculation. |
| [Python API](../user_guide/python_api.md) | 10-15 min | Author a model with `phast.Problem` and understand the fluent method map. |
| [Visual glossary](02_visual_glossary.md) | 10 min | Read the picture-first guide to AT1/AT2, energy splits, and `l0`. |
| [Modular FEM and learned damage](03_modular_fem_and_learned_damage.md) | 20 min | Assemble geometry, material, boundary conditions, fracture choices, solver routes, and an audited learned-damage plug-in. |
| [Heterogeneous material fields](05_heterogeneous_material_fields.md) | 15 min | Define element-ordered `E(x)` and `Gc(x)` arrays and solve a bounded AT2 damage teaching problem. |
| [From tutorial to first research study](06_first_research_study.md) | 20 min | Convert a modelling question into a bounded PhAST study with explicit decisions, preflight checks, pilot execution, outputs, and validation evidence. |
| [YAML workflow](../user_guide/yaml_workflow.md) | 10 min | Run a public declarative configuration and understand the standard result directory. |
| [Example gallery](../example-gallery.md) | 5 min | Choose a runnable dynamic, quasi-static, or solid-mechanics example. |
| [Public API reference](../user_guide/public_api_reference.md) | 5 min | Read metadata, histories, visuals, and stored trajectory fields. |

## Recommended Learning Path

1. Install the package (`git clone` + `pip install -e .`) and run `python -m phast doctor`.
2. Validate a public YAML configuration with `--validate-only`.
3. Inspect the [B3 dynamic SENT notebook](notebook_dynamic_sent.ipynb) to connect
   a complete public configuration with visible retained crack propagation.
4. Run one small public example into `runs/<case>`.
5. Inspect the completed run with `phast.load_result(...)`.
6. Build a small model with `phast.Problem`.
7. Read the [visual glossary](02_visual_glossary.md) if the terminology feels abstract.
8. Read [Modular FEM and learned damage](03_modular_fem_and_learned_damage.md)
   before introducing a learned damage proposal.
9. Run [Heterogeneous material fields](05_heterogeneous_material_fields.md)
   before adapting a segmented or multiphase material map.
10. Use [From tutorial to first research study](06_first_research_study.md) to
   record the modelling question, assumptions, observables, and validation target.
11. Move durable studies into a YAML configuration when you need reproducibility or HPC
   submission.

Users coming from Abaqus, COMSOL, FEniCS, or deal.II should read
[Setting up new problems](../user_guide/setup_problems.md) first. It maps familiar
FEM concepts such as parts, mesh sets, materials, loads, steps, jobs, and result
databases to the PhAST fluent API and YAML configuration structure.

## Guided 45-minute session

This route is suitable for a supervised laboratory class or autumn-school
session. Installation should be completed in advance; the rendered notebook and
retained-results notebook provide the fallback when a participant cannot execute
the solver.

| Time | Activity | Evidence produced |
|---:|---|---|
| 0-5 min | State the boundary-value problem and predict the constrained and loaded regions. | Written prediction. |
| 5-12 min | Inspect geometry, mesh, and named physical groups. | Setup figure and group table. |
| 12-20 min | Identify material, phase-field, loading, and solver choices. | Completed modelling-decision table. |
| 20-25 min | Run `--validate-only` and explain what it does not verify. | Configuration preflight record. |
| 25-33 min | Inspect the retained B3 propagation sequence, or execute the bounded two-step setup workflow when the environment is prepared. | A clearly labelled retained animation or a newly generated result directory. |
| 33-39 min | Change one parameter and predict the consequence before rerunning. | Before/after observation. |
| 39-45 min | Inspect manifests, histories, and fields; state one limitation and one next test. | Exit statement suitable for a lab notebook. |

The learning objective is not to produce a validated crack path in 45 minutes.
It is to connect a physical problem statement to a reproducible computational
record and to distinguish execution evidence from scientific evidence.

## Runnable Examples

| Workflow | Entry point | Typical output |
|---|---|---|
| Dynamic-fracture preflight | `python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml --validate-only` | Configuration acceptance report only; no simulation fields are generated. |
| Dynamic-branching preflight | `python -m phast run examples/dynamic/B7_dynamic_crack_branching_comsol/config.yaml --validate-only` | Configuration acceptance report only; retained artifacts are separate evidence. |
| Quasi-static fracture | `python -m phast run examples/quasistatic/notched_holed_plate/config.yaml --output_dir runs/notched_holed_plate` | Final damage, response histories, comparison artifacts, and result manifests. |
| Solid mechanics | `python -m phast run examples/solid_mechanics_beta/linear_plate/config.yaml --output_dir runs/linear_plate` | Displacement/stress plots, response history, and metadata. |
| Heterogeneous AT2 teaching problem | `python examples/heterogeneous_fields/run.py --config examples/heterogeneous_fields/parameters.yaml --output-dir runs/heterogeneous_fields` | Elementwise material CSV, nodal damage CSV, field plots, metadata, and manifests. |

The example gallery lists the current public examples and their expected
artifacts. Longer or beta validation workflows are summarized in the capability
matrix rather than treated as first-run tutorials.

## Python Authoring vs YAML Reproduction

Use Python when you are designing a model:

```python
import phast

result = (
    phast.Problem("linear plate")
    .geometry("structured_grid", nx=40, ny=12, length=1.0, height=0.2)
    .region("body", kind="domain")
    .material("steel", model="solid_mechanics", region="body", E=2.1e11, nu=0.3)
    .analysis_step("load", kind="solid_mechanics", controls={"tip_force_y": -1.0e3})
    .solver("solid_mechanics", example="solid_mechanics.linear_plate")
    .outputs(fields=["displacement", "von_mises"], histories=["response"], plots=True)
    .run(output_dir="runs/linear_plate", return_result=True)
)
```

Use YAML when you want an exact configuration file:

```bash
python -m phast run examples/solid_mechanics_beta/linear_plate/config.yaml \
  --output_dir runs/linear_plate
```

Both paths are inspected the same way:

```python
import phast

result = phast.load_result("runs/linear_plate")
print(result.metadata())
print(result.history_names())
print(result.visuals())
```

## Related Guides

- [Python API](../user_guide/python_api.md)
- [Setting up new problems](../user_guide/setup_problems.md)
- [YAML workflow](../user_guide/yaml_workflow.md)
- [Example result contract](../user_guide/example_contract.md)
- [Capability matrix](../user_guide/capability_matrix.md)
- [Troubleshooting](../troubleshooting.md)
