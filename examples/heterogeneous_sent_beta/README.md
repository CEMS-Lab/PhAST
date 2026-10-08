# Heterogeneous single-edge-notched tension

This beta schema-2 example pulls a square plate with a horizontal starter
crack and a circular tougher bulk region. It uses the same section order and
CLI as the layered DCB, without a problem-specific solver loop. Its numerical
values are teaching choices, not a calibrated crack-deflection benchmark.

See the [standard simulation tutorial](../../docs/tutorial/07_standard_simulation_workflow.md)
for Python 3.10+ source installation on macOS/Linux or Windows, and the
[configuration style](../../CONFIGURATION_STYLE.md) for authoring rules.
The primary teaching route uses the two short single-material SENT exercises
before the full DCB; this is an additional multi-material setup.

## Run from the repository root

```bash
python -m phast explain-config examples/heterogeneous_sent_beta/config.yaml
python -m phast run examples/heterogeneous_sent_beta/config.yaml --validate-only
python -m phast run examples/heterogeneous_sent_beta/config.yaml --output_dir runs/heterogeneous_sent
```

Schema-2 explanation dispatch requires the combined CLI check before this
sequence is labelled verified. Legacy `precheck --config` is not a
universal schema-2 command. Preflight does not mesh or solve. Use a new
output directory for a changed case.

## Read the configuration

The section order is `schema_version`, `name`, `reference`, `geometry`,
`regions`, `materials`, `assignments`, `initial_conditions`,
`boundary_conditions`, `analysis_steps`, `solver`, `outputs`.

| Input | Meaning |
|---|---|
| Geometry and mesh | A 1 mm square, `40 x 40` cells split into T3 elements. |
| Starter crack | From `(0, 0.50)` to `(0.40, 0.50) mm`; prescribed `damage = 1`. |
| Inclusion | Circle centred at `(0.65, 0.50) mm`, radius `0.12 mm`. |
| Common elastic properties | `E = 210 MPa`, `nu = 0.30`; plane stress. |
| Fracture resistance | Matrix `Gc = 0.0027 N/mm`; inclusion `Gc = 0.0081 N/mm`. |
| Regularisation | Common `l0 = 0.05 mm`, Amor split, AT2. |
| Mechanical support | Bottom `uy = 0`; bottom-left `ux = 0` removes horizontal rigid translation. |
| Loading | Top `uy` ramps to `0.012 mm`. |
| Load levels | 61 including zero: 60 nominal increments, with extra increments possible after cutbacks. |
| Damage boundary assumption | Top and bottom remain at `damage = 0`; the initial crack remains at `damage = 1`. |

The undamaged top/bottom constraints suppress boundary damage. They are a
stated modelling choice, not the natural phase-field condition. The selected
final displacement does not guarantee a particular crack path or stopping
event.

## Make one controlled edit

| Edit | What to check |
|---|---|
| Rectangle dimensions/origin | Update top/bottom boundary coordinates, the anchor, crack, and disk. |
| `nx` or `ny` | Keep selected nodes/elements non-empty; a narrow crack mask can miss a changed mesh. |
| Disk centre/radius | Update both `circle` and `outside_circle` selectors identically. |
| A material's `E` | Stiffness and stress/energy redistribution change; this does not independently increase toughness. |
| A material's `Gc` | Fracture resistance changes; keep `E` and other inputs fixed to isolate that effect. |
| Top displacement | Check strain/rotation assumptions, convergence, and the response before extending the run. |

Element-centroid selection approximates the material boundary rather than
creating a conforming circle. Use positive `E`/`Gc` and only the damage
initial conditions and history requests accepted by this route. Define `h`
when reporting `h/l0`, and
compare mesh and load-increment sensitivity; no single ratio guarantees
convergence. The starter crack already has maximum damage one, so that
maximum alone is not evidence of propagation.

Quasi-static analysis omits inertia; load factor is not physical time and the
illustrative density is not dynamic material data. Explicit dynamics needs
consistent density/time and a CFL-limited step, but dynamic multi-material
execution is not supported by this route. A smaller step is not physical
validation.

## Outputs and optional trajectory

The runner produces CSV histories and nodal/element fields, setup/material
and final-field PNGs, `damage_evolution.gif`, `final_fields.vtu`, a summary,
and run/visual manifests. The input does **not** request a trajectory.
If field-level replay is needed, add this entry to its existing
`outputs.fields` list in a working copy, retaining the other output requests:

```yaml
- {name: trajectory, format: h5, every: 1}
```

This is a list-entry excerpt, not a complete configuration. HDF5 is the
default trajectory format on this route; use a new directory because its
writer refuses to overwrite `training_data.h5`. Keep large raw stores local.

```python
import phast

result = phast.load_result("runs/heterogeneous_sent")
print(result.metadata())
print(result.history_names())
print(result.visuals())
if result.has_field("damage"):
    damage = result.field("damage", step=-1)
```

## Evidence and limits

No completed-run result bundle is retained in this example folder. Do not
treat the input or the chosen endpoint as verified crack-deflection evidence.
Interpret penetration, shielding, arrest, or deflection only after checking
seed-connected growth, fields, reaction history, convergence, and sensitivity.

The documented multi-material scope is a structured T3 rectangle, CPU
float64, elementwise `E`/`Gc`, common `nu`/`l0` and shared Amor AT2
settings, with prescribed-value constraints. It is bulk phase-field fracture,
not interfacial debonding, a cohesive law, or PF-CZM. Frozen historical course
inputs and separate 3D research are not part of this adapter.

Baseline topology references, not validation data for this heterogeneous case:

- [Miehe, Hofacker, and Welschinger (2010)](https://doi.org/10.1016/j.cma.2010.04.011)
- [PhaseFieldX single-edge-notched tension example](https://phasefieldx.readthedocs.io/en/latest/auto_examples/PhaseFieldFracture/plot_1711.html)

If a step is unclear,
[open a GitHub issue](https://github.com/CEMS-Lab/PhAST/issues/new/choose)
with the input, exact command, OS, Python/PhAST version, and full error.
