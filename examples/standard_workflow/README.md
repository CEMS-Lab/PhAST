# One configuration sequence for a first simulation

These small examples teach the same PhAST input structure with two analysis
choices. They exercise the installation and solution sequence; they are not
crack-growth benchmarks or a comparison of dynamic and quasi-static results.

Install PhAST from the repository root with `python -m pip install -e .`.
The [installation guide](https://cems-lab.github.io/PhAST/install.html) explains
environment creation on Linux, macOS, and Windows.

## Choose an analysis

| Input | Physical approximation | Numerical solution |
| --- | --- | --- |
| [Quasi-static SENT](quasistatic_sent/config.yaml) | Inertia is neglected. | Alternate displacement equilibrium and an implicit damage solution within each load increment. |
| [Dynamic SENT](dynamic_sent/config.yaml) | Inertia is retained. | Explicit mechanical time integration and an implicit damage update. |

SENT means single-edge-notched tension. Both inputs use the same ordered
configuration sections, explained in
[CONFIGURATION_STYLE.md](../../CONFIGURATION_STYLE.md).
Choose the closest physical example before changing parameters. Changing only
a solver label is not sufficient to convert a loading experiment into a
different analysis.

## Check and run

Run these commands from the PhAST repository root:

```bash
# Describe the selected formulation without running a simulation.
python -m phast explain-config examples/standard_workflow/quasistatic_sent/config.yaml

# Check whether the input is accepted by this execution interface.
python -m phast run examples/standard_workflow/quasistatic_sent/config.yaml --validate-only

# Solve and keep the new result in its own directory.
python -m phast run examples/standard_workflow/quasistatic_sent/config.yaml --output_dir runs/first_quasistatic

# The dynamic example uses exactly the same command structure.
python -m phast run examples/standard_workflow/dynamic_sent/config.yaml --output_dir runs/first_dynamic
```

Configuration validation is not a numerical convergence or experimental
validation test. Inspect the run log and recorded outputs after a solve.

## Change one physical choice at a time

Copy a configuration, give the copy a descriptive name, and choose a new output
directory. Keep units consistent when changing material or geometric values.
After changing geometry, update affected boundary and material regions and
inspect the new mesh. After changing the mesh or phase-field length, reassess
spatial resolution. A finer dynamic mesh may also require a smaller time step.

For a complete retained crack-growth illustration, use the
[layered DCB example](../two_material_dcb_beta/README.md). DCB means
double-cantilever beam. Its README distinguishes the observed response from
unverified penetration, bypass, or interface-debonding claims.

The [standard simulation tutorial](https://cems-lab.github.io/PhAST/tutorial/07_standard_simulation_workflow.html)
explains the input sections, result inspection, and controlled student
exercises. No custom solver loop is needed for either example.

If a documented step fails, [open an issue](https://github.com/CEMS-Lab/PhAST/issues)
with the input, command, operating system, Python version, PhAST revision,
and complete error output.
