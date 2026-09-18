# Linear Elastic Plate

## 1. Problem Description

Plane-strain CST cantilever solved with PhAST's sparse autograd linear-solve path. The example compares the finite-element tip displacement against an Euler-Bernoulli estimate, differentiates the tip displacement with respect to Young's modulus, and writes standard displacement, stress, strain, and energy visualisations.


## Run The YAML Configuration

Run commands from the repository root:

```bash
python -m phast run examples/solid_mechanics_beta/linear_plate/config.yaml --validate-only
python -m phast run examples/solid_mechanics_beta/linear_plate/config.yaml
python examples/solid_mechanics_beta/linear_plate/run_fluent.py
```

Use `--output_dir <path>` with `python -m phast run` for local reruns that should not overwrite the reference outputs.

The expected finite-element tip displacement is approximately `-2.024e-06 m`,
with a documented `-14.98%` difference from the Euler-Bernoulli estimate. A
newcomer smoke run should reproduce these recorded values to normal floating-
point tolerance. This comparison is an onboarding regression target, not a
general finite-element accuracy criterion. The example writes field plots but
does not retain reloadable displacement or stress arrays in Zarr/HDF5.

## How The YAML Is Used

`mesh` defines the structured rectangular grid, `material` defines the linear elastic constants, and `loading.tip_force_y` defines the applied tip load. The workflow lowers those blocks to the solid-mechanics example runner, which assembles the CST system, solves the sparse linear problem, writes field plots, and records the response history.

## Run Without YAML

```bash
python examples/solid_mechanics_beta/linear_plate/run_fluent.py --run --output-dir runs/linear_plate
```

`run_fluent.py` uses `fluent_setup.build_problem()` to declare the same
geometry, region, material, load step, solver selection, and requested outputs
in Python. With `--run`, it passes the resulting `ProblemSpec` to the existing
solid-mechanics YAML runner. No separate solver implementation is used.

The checked-in `config.yaml` remains the reference input for shared runs.
For advanced Python tooling, `build_problem().to_spec()` from the existing
`fluent_setup.py` companion can also be passed to
`phast.workflow.run_problem_spec(...)`. For this promoted linear-plate case,
the spec is lowered to the same solid-mechanics YAML runner. Passing
`validate_only=True` checks the configuration without producing result
artifacts; `output_dir` selects a separate result directory. This bridge is
not a general executor for arbitrary Python-built problems.

## How Manual Setup Works

The manual setup mirrors the YAML fields directly: `.geometry(...)` maps to `mesh`, `.material(...)` maps to `material`, `.analysis_step(...)` maps to `loading`, `.solver(...)` selects `solid_mechanics.linear_plate`, and `.outputs(...)` requests the response and field artifacts.

The example uses a consistent SI convention: plate dimensions are `1.0 m`
by `0.2 m`, Young's modulus is `2.1e11 Pa`, and the applied force is
`-1000 N`. The fluent companion records `units="m"` as geometry metadata;
it does not rescale the numeric mesh parameters. The recorded tip displacement
and its `response.csv` label are in metres.

The promoted runner clamps both displacement components on the left edge and
applies the vertical point force at the mid-height node on the right edge.
Those boundary conditions are built into this example. The Python `ProblemSpec`
bridge accepts one full-domain material and one load step on the built-in
structured grid. Explicit boundary conditions, imported meshes and additional
materials or steps produce a validation error before execution.

## Reference Result

| Initial conditions | Response and deformed field |
| --- | --- |
| <img src="initial_conditions.png" width="360"> | <img src="deformed_shape.png" width="360"> |

| Metric | Reference value |
| --- | ---: |
| Tip displacement FE | `-2.024e-06 m` |
| Euler-Bernoulli estimate | `-2.381e-06 m` |
| Relative error | `-14.98 %` |
| Maximum von Mises stress | `1.141e+05 Pa` |
