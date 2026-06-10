# Test Matrix

Use explicit pytest tiers instead of running an unknown full suite when making
customer-facing changes.

## Local Confidence

```bash
python -m pytest -q
```

Default pytest options deselect `slow`, `hpc`, and `benchmark` tests, use short
tracebacks, and print the 20 slowest tests at the end of the run. This is the
normal pre-commit confidence suite.

## Targeted Tiers

```bash
python -m pytest -m fast -q
python -m pytest -m solver -q
python -m pytest -m docs -q
python -m pytest -m "not slow and not hpc and not benchmark" -q
python -m pytest -m "benchmark or artifact" -q
```

Marker meanings:

| Marker | Meaning |
|---|---|
| `fast` | Laptop-safe smoke/regression tests intended to finish quickly. |
| `solver` | Core mechanics, damage, autograd, sparse-solve, and config behavior. |
| `docs` | Documentation, schema, citation, packaging, and drift checks. |
| `benchmark` | Benchmark-style validation or mesh-heavy tests; opt-in by default. |
| `hpc` | Requires or emulates HPC/GPU/cluster-only behavior. |
| `artifact` | Validates generated files, figures, datasets, or stored outputs. |
| `slow` | Long-running tests; deselected by default. |

`tests/conftest.py` applies conservative automatic tier markers by test
filename/path. Explicit per-test or module markers still win; the auto marker
only fills obvious gaps so targeted commands do not silently collect zero or
near-zero tests.

When `pytest-timeout` is installed through the `dev` extra, the same hook adds
timeout metadata to opt-in long tiers: 600 seconds for `benchmark`/`slow` tests
and 1800 seconds for `hpc` tests. Tests with an explicit `@pytest.mark.timeout`
keep their own limit.

## Pre-Merge Matrix

For ordinary code changes:

```bash
python -m pytest -m "not slow and not hpc and not benchmark" -q
```

For code touching solver physics or config parsing:

```bash
python -m pytest -m "not slow and not hpc and not benchmark" -q
python -m pytest tests/test_physics.py tests/test_config_validation.py -q
python -m pytest tests/test_staggered_backend_propagation.py -q
python scripts/generate_reference_yaml.py --check
python scripts/check_docs_drift.py
```

For docs/schema-only changes:

```bash
python -m pytest -m docs -q
python scripts/generate_reference_yaml.py --check
python scripts/generate_json_schema.py --check
python scripts/check_docs_drift.py
python scripts/check_artifact_hygiene.py
```

For benchmark or HPC validation:

```bash
python -m pytest -m "benchmark or artifact" -q
python -m pytest -m hpc -q
```

Run benchmark/HPC tests only when the required mesh files, artifacts, devices,
or cluster environment are available. Paper-quality acceptance still requires
the benchmark compare scripts and recorded run artifacts; a passing unit test is
not enough to claim validation.

## CI Behavior

The GitHub Actions package test job runs:

```bash
python -m pytest tests/ -v -m "not slow and not hpc and not benchmark" --tb=short --durations=20
```

That keeps pull-request feedback progress-friendly while reserving slow,
benchmark, and cluster-dependent coverage for explicit local/HPC runs.
