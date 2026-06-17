# Output Standards

This section summarizes what a PhAST run should write and how those artifacts
are consumed. The promoted-example policy is
`docs/user_guide/example_contract.md`.

Each row names the artifact producer, the primary consumer, and the public
contract surface that keeps the artifact convention reviewable.

| Artifact | Producer | Consumer | Contract surface |
|---|---|---|---|
| `run_manifest.json` | YAML runner, promoted scripts | `phast.load_result`, release review, example README | `examples/PUBLIC_EXAMPLES_CONTRACT.yaml` |
| `visual_manifest.json` | plotting/postprocess layer | docs gallery, `result.visuals()` | example README and `docs/user_guide/example_contract.md` |
| `run_metadata.json` | solver runner | provenance audit, `result.metadata()` | example README and result API docs |
| `history.csv` | solver runner | `result.history_names()`, `result.history(name)` | example README |
| `results.csv` | reaction/load output writer | benchmark comparison, response plots | example README and comparison notes |
| `response.csv` | solid mechanics runner | solid examples, README snippets | example README |
| `energy.csv` | fracture/dynamic runners | energy plots, validation review | example README |
| `solver_telemetry.csv` | iterative solvers | convergence review | example README |
| `timing_per_step.csv` | runner/profiler | performance review | performance and reproducibility docs |
| `training_data.zarr` | trajectory writer | restart/postprocess/ML consumers | external artifact policy |
| PNG/GIF visuals | plot/postprocess layer | docs gallery, review packets | `visual_manifest.json` and example contract |

## Inspection path

```python
import phast

result = phast.load_result("runs/miehe_tension")
print(result.metadata())
print(result.manifest())
print(result.history_names())
print(result.visuals())
```

Stored raw fields are loaded with `result.field(name, step=-1)` only when the
trajectory store contains that field. Derived fields such as von Mises stress
or displacement magnitude require explicit postprocessing and are not silently
invented by the Result API.

## Related pages

- `docs/user_guide/example_contract.md`
- `docs/user_guide/results_api.md`
- `docs/user_guide/yaml_workflow.md`
- `docs/example-gallery.md`
