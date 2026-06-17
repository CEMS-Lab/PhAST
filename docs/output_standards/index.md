# Output Standards

This section summarizes what a PhAST run should write and how those artifacts
are consumed. The promoted-example policy is
`docs/user_guide/example_contract.md`.

Each row names the artifact producer, the primary consumer, and the tests that
keep the contract from drifting.

| Artifact | Producer | Consumer | Enforced by |
|---|---|---|---|
| `run_manifest.json` | YAML runner, promoted scripts | `phast.load_result`, release review, example README | `tests/test_public_examples_contract.py`, `tests/test_result_api.py` |
| `visual_manifest.json` | plotting/postprocess layer | docs gallery, `result.visuals()` | `tests/test_public_examples_contract.py`, `tests/test_result_api.py` |
| `run_metadata.json` | solver runner | provenance audit, `result.metadata()` | `tests/test_result_api.py` |
| `history.csv` | solver runner | `result.history_names()`, `result.history(name)` | `tests/test_result_api.py` |
| `results.csv` | reaction/load output writer | benchmark comparison, response plots | `tests/test_public_examples_contract.py` |
| `response.csv` | solid mechanics runner | solid examples, README snippets | `tests/test_solid_mechanics_yaml_runner.py` |
| `energy.csv` | fracture/dynamic runners | energy plots, validation review | `tests/test_public_examples_contract.py` |
| `solver_telemetry.csv` | iterative solvers | convergence review | `tests/test_public_examples_contract.py` |
| `timing_per_step.csv` | runner/profiler | performance review | `tests/test_public_examples_contract.py` |
| `training_data.zarr` | trajectory writer | restart/postprocess/ML consumers | `tests/test_result_api.py` |
| PNG/GIF/MP4 visuals | plot/postprocess layer | docs gallery, review packets | `visual_manifest.json` and example contract tests |

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
