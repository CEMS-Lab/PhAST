# Tutorials

This page is the onboarding map for new PhAST users. Start with the shortest
validation path, then move to Python authoring, YAML reproduction, and result
inspection.

## Start Here

Launch the step-by-step problem setup notebook in Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CEMS-Lab/PhAST/blob/main/docs/tutorial/problem_setup_walkthrough.ipynb)

| Tutorial | Time | What you learn |
|---|---:|---|
| [Getting started](../getting-started.md) | 5 min | Lightweight install (`pip install -e .`), run `doctor`, validate a shipped example, and inspect a result. |
| [Problem setup notebook](problem_setup_walkthrough.ipynb) | 30-45 min | Create geometry, mesh and inspect named regions, apply initial conditions, supports, loads, solver settings, run a short solve, and post-process artifacts. |
| [Python API](../user_guide/python_api.md) | 10-15 min | Author a model with `phast.Problem` and understand the fluent method map. |
| [Visual glossary](02_visual_glossary.md) | 10 min | Read the picture-first guide to AT1/AT2, energy splits, and `l0`. |
| [Modular FEM and learned damage](03_modular_fem_and_learned_damage.md) | 20 min | Assemble geometry, material, boundary conditions, fracture choices, solver routes, and an audited learned-damage plug-in. |
| [Heterogeneous material fields](05_heterogeneous_material_fields.md) | 15 min | Define element-ordered `E(x)` and `Gc(x)` arrays and solve a bounded AT2 damage teaching problem. |
| [YAML workflow](../user_guide/yaml_workflow.md) | 10 min | Run a public declarative configuration and understand the standard result directory. |
| [Example gallery](../example-gallery.md) | 5 min | Choose a runnable dynamic, quasi-static, or solid-mechanics example. |
| [Public API reference](../user_guide/public_api_reference.md) | 5 min | Read metadata, histories, visuals, and stored trajectory fields. |

## Recommended Learning Path

1. Install the package (`git clone` + `pip install -e .`) and run `python -m phast doctor`.
2. Validate a public YAML configuration with `--validate-only`.
3. Run one small public example into `runs/<case>`.
4. Inspect the completed run with `phast.load_result(...)`.
5. Build a small model with `phast.Problem`.
6. Read the [visual glossary](02_visual_glossary.md) if the terminology feels abstract.
7. Read [Modular FEM and learned damage](03_modular_fem_and_learned_damage.md)
   before introducing a learned damage proposal.
8. Run [Heterogeneous material fields](05_heterogeneous_material_fields.md)
   before adapting a segmented or multiphase material map.
9. Move durable studies into a YAML configuration when you need reproducibility or HPC
   submission.

Users coming from Abaqus, COMSOL, FEniCS, or deal.II should read
[Setting up new problems](../user_guide/setup_problems.md) first. It maps familiar
FEM concepts such as parts, mesh sets, materials, loads, steps, jobs, and result
databases to the PhAST fluent API and YAML configuration structure.

## Runnable Examples

| Workflow | Entry point | Typical output |
|---|---|---|
| Dynamic fracture | `python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml --validate-only` | Dynamic damage fields, histories, metadata, and curated animation assets. |
| Dynamic crack branching | `python -m phast run examples/dynamic/B7_dynamic_crack_branching_comsol/config.yaml --validate-only` | Crack-branching comparison package and visual summaries. |
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

## Tutorial Contract

Each public tutorial should state:

- the command to run from the repository root,
- expected runtime and device,
- what physics and solver path are active,
- which output files should appear,
- how to inspect the result with `phast.load_result(...)`,
- which capability-matrix row supports the claim.

If an example depends on unavailable raw trajectory output, unreleased paper
assets, or a custom diagnostic script to make sense, it should stay out of the
public tutorial path until it has documented implementation and evidence.

## Related Guides

- [Python API](../user_guide/python_api.md)
- [Setting up new problems](../user_guide/setup_problems.md)
- [YAML workflow](../user_guide/yaml_workflow.md)
- [Example result contract](../user_guide/example_contract.md)
- [Capability matrix](../user_guide/capability_matrix.md)
- [Troubleshooting](../troubleshooting.md)
