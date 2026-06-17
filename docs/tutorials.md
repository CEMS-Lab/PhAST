# Tutorials

This page is the onboarding map for new PhAST users. Start with the shortest
validation path, then move to Python authoring, YAML reproduction, and result
inspection.

## Start Here

| Tutorial | Time | What you learn |
|---|---:|---|
| [Getting started](getting-started.md) | 5 min | Install PhAST, run `doctor`, validate a shipped example, and inspect a result. |
| [Build your first model with `phast.Problem`](tutorial/fluent_authoring_guide.md) | 10-15 min | Create, validate, run, and inspect a Python-authored model. |
| [YAML workflow](user_guide/yaml_workflow.md) | 10 min | Run a public input deck and understand the standard result directory. |
| [Example gallery](example-gallery.md) | 5 min | Choose a runnable dynamic, quasi-static, or solid-mechanics example. |
| [Results API](user_guide/results_api.md) | 5 min | Read metadata, histories, visuals, and stored trajectory fields. |

## Recommended Learning Path

1. Install the package and run `python -m phast doctor`.
2. Validate a public YAML deck with `--validate-only`.
3. Run one small public example into `runs/<case>`.
4. Inspect the completed run with `phast.load_result(...)`.
5. Build a small model with `phast.Problem`.
6. Move durable studies into a YAML deck when you need reproducibility or HPC
   submission.

## Runnable Examples

| Workflow | Entry point | Typical output |
|---|---|---|
| Dynamic fracture | `python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml --validate-only` | Dynamic damage fields, histories, metadata, and curated animation assets. |
| Dynamic crack branching | `python -m phast run examples/dynamic/B7_dynamic_crack_branching_comsol/config.yaml --validate-only` | Crack-branching comparison package and visual summaries. |
| Quasi-static fracture | `python -m phast run examples/quasistatic/notched_holed_plate/config.yaml --output_dir runs/notched_holed_plate` | Final damage, response histories, comparison artifacts, and result manifests. |
| Solid mechanics | `python -m phast run examples/solid_mechanics/linear_plate/config.yaml --output_dir runs/linear_plate` | Displacement/stress plots, response history, and metadata. |

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

Use YAML when you want an exact deck:

```bash
python -m phast run examples/solid_mechanics/linear_plate/config.yaml \
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

If an example needs private raw HPC output, unreleased paper assets, or a
custom diagnostic script to make sense, it should stay out of the public
tutorial path until it is promoted.

## Related Guides

- [Python API](user_guide/python_api.md)
- [Setting up new problems](user_guide/setup_problems.md)
- [YAML workflow](user_guide/yaml_workflow.md)
- [Example result contract](user_guide/example_contract.md)
- [Capability matrix](user_guide/capability_matrix.md)
