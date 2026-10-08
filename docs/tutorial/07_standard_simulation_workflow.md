# One configuration workflow for student simulations

This tutorial teaches how to read, run, and change a PhAST problem without
writing a solver loop. It assumes basic knowledge of stress, strain, and
boundary conditions. Read the [phase-field primer](01_phase_field_primer.md)
if a smooth damage field is unfamiliar.

Use the same sequence for each problem: geometry, regions, materials,
assignments, initial conditions, boundary conditions, analysis, solver, and
outputs. A shared input layout does not make every combination of physics
available.

## 1. Install from source

Install **Python 3.10 or newer** and Git. An internet connection is needed
for installation. These commands install the checkout you clone; they do not
identify it as the latest release. Run subsequent commands from the repository
root, where `pyproject.toml` and `examples/` are located.

### macOS or Linux

```bash
git clone https://github.com/CEMS-Lab/PhAST.git
cd PhAST
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m phast doctor
```

Ensure `python3` selects Python 3.10 or newer. A virtual environment keeps
dependencies separate from other projects; editable installation uses the
source in this checkout.

### Windows PowerShell

```powershell
git clone https://github.com/CEMS-Lab/PhAST.git
cd PhAST
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\Activate.ps1
python -m phast doctor
```

Ensure `py -3` selects Python 3.10 or newer. If local policy blocks
activation, do not change system policy for this lesson. Replace `python`
in subsequent commands with `.\.venv\Scripts\python.exe`; activation
is unnecessary with that form.

### Dependencies installed separately

Editable installation already installs the declared runtime dependencies.
Environments that manage dependencies separately can use this sequence in
the same virtual environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Optional GPU/HPC solvers are not prerequisites for the CPU teaching route.
Install `requirements-docs.txt` only if you need to build the website.

## 2. Follow the three-problem teaching route

| Configuration | Physical question | Evidence boundary |
|---|---|---|
| `examples/standard_workflow/quasistatic_sent/config.yaml` | How is a small single-material single-edge-notched tension (SENT) problem loaded without inertia? | Short setup exercise; adapter and exact CLI checks must be recorded before promotion. |
| `examples/standard_workflow/dynamic_sent/config.yaml` | How does a single-material SENT setup include inertia and explicit time integration? | Short setup exercise; successful execution is not implied by this path or by a completed quasi-static run. |
| `examples/two_material_dcb_beta/config.yaml` | How does a crack interact with a tougher disk inside a weak bulk layer? | Full retained calculation and coarse matched control; qualitative evidence, not calibrated inclusion bypass. |

The first two inputs use the same schema as the DCB example. Their combined
implementation and execution checks remain a release requirement; this page
does not claim those checks have passed. A short exercise may show correct
setup and output without appreciable new crack growth. The additional
`examples/heterogeneous_sent_beta/config.yaml` applies the layout to a
multi-material square plate.

Frozen historical course assets and separate 3D research keep their own
inputs. They are not migrated into this public route without adapters and
evidence.

## 3. Explain, check, and run the same input

Use this exact command shape for every supported standard example:

```bash
python -m phast explain-config CONFIG
python -m phast run CONFIG --validate-only
python -m phast run CONFIG --output_dir RUN
```

`CONFIG` is a complete YAML path and `RUN` is a new output directory.
They are placeholders, not shell variables. For the small quasi-static
exercise, the corresponding commands are:

```bash
python -m phast explain-config examples/standard_workflow/quasistatic_sent/config.yaml
python -m phast run examples/standard_workflow/quasistatic_sent/config.yaml --validate-only
python -m phast run examples/standard_workflow/quasistatic_sent/config.yaml --output_dir runs/standard_quasistatic_sent
```

For the dynamic exercise:

```bash
python -m phast explain-config examples/standard_workflow/dynamic_sent/config.yaml
python -m phast run examples/standard_workflow/dynamic_sent/config.yaml --validate-only
python -m phast run examples/standard_workflow/dynamic_sent/config.yaml --output_dir runs/standard_dynamic_sent
```

For the full fine-mesh DCB reference:

```bash
python -m phast explain-config examples/two_material_dcb_beta/config.yaml
python -m phast run examples/two_material_dcb_beta/config.yaml --validate-only
python -m phast run examples/two_material_dcb_beta/config.yaml --output_dir runs/dcb_fine_reference
```

The explanation command describes the input. `--validate-only` checks
implemented schema/workflow constraints without solving. For multi-material
inputs, this includes constructing the geometric mesh to check region
selections, material assignments, and boundary conditions. The last command
executes the model. Input validation is not evidence of numerical convergence
or physical accuracy. Do not change `schema_version` to hide
an unsupported command; legacy `precheck --config` is not a universal
schema-2 replacement.

The retained DCB reference took about 968 seconds on its recorded CPU run,
not a runtime promise for your machine. Use a fresh directory for each
rerun; its trajectory writer refuses to overwrite `training_data.h5`.

## 4. Read the sections in the same order

Open the {download}`complete DCB input <../../examples/two_material_dcb_beta/config.yaml>`.

| Section | Meaning in this example |
|---|---|
| `schema_version`, `name`, `reference` | Schema 2, the problem identity, and its teaching scope. |
| `geometry` | A 6.0 mm by 1.2 mm rectangle, origin `(0, -0.6)`, with `160 x 48` cells split into three-node triangles (T3). |
| `regions` | Element selections for materials and node selections for the crack, clamp, and loading. |
| `materials` | Named elastic and fracture properties. |
| `assignments` | One material for every element, without gaps or overlap. |
| `initial_conditions` | Starting damage; on this multi-material route the selected values remain constrained. |
| `boundary_conditions` | Right-edge clamp and opposite left-edge opening displacements. |
| `analysis_steps` | Physical analysis, active conditions, and loading controls. |
| `solver` | Algorithms, tolerances, and iteration/cutback limits. |
| `outputs` | Result directory, fields, histories, and visuals. |

Use the {download}`configuration authoring rules <../../CONFIGURATION_STYLE.md>`
for new inputs. Older schema-1 benchmarks use singular `material`,
`loading`, and `output` sections. Do not mix the two layouts or migrate a
file by changing its version number alone.

## 5. Edit geometry and mesh with their dependent selections

Geometry is the specimen shape. A mesh divides it into elements on which
the equations are approximated. Changing `nx` or `ny` refines the
structured mesh without changing the declared dimensions.

| Edit | Related inputs to inspect |
|---|---|
| Rectangle length | The `right_clamp` selector uses `side: right` and follows the edge. Adjust both layer rectangles and reconsider disk and crack positions. |
| Height or origin | Update layer extents, loading bounds, boundary coordinates, and crack coordinates. |
| Disk centre or radius | Keep the `circle` and every complementary `outside_circle` selector identical; check layer clearance. |
| `nx` or `ny` | Confirm that crack/load selections still contain nodes and material regions contain elements. |
| Starter crack | Update its `from`, `to`, and selection `thickness`; do not prescribe a desired future path. |

The line selector's `thickness` is a selection distance, not a physical
crack opening. Changing mesh rows can remove nodes from the mid-plane,
leaving a narrow selection empty. A preflight does not check the generated
mesh; inspect selection errors and the setup/material plots after a run.

Materials are assigned at **element centroids**. A circular selector does
not produce a conforming curved interface. Its discrete representation can
change with the mesh.

State how mesh size `h` is measured when reporting `h/l0`. The fine DCB
maximum triangle edge is about `0.0451 mm`, with `l0 = 0.10 mm`, giving
`h/l0 = 0.451`. This is not a universal convergence guarantee. Compare
forces and crack paths across meshes at fixed physical inputs and examine
load-increment sensitivity. Changing `l0` changes the regularised model,
not merely the mesh.

## 6. Change stiffness and fracture resistance independently

The historical DCB folder name contains "two material", but its input has
three bulk regions:

| Region | `E` [MPa] | `Gc` [N/mm] |
|---|---:|---:|
| Fracture-resistant outer regions | 5000 | 1.00 |
| Weak mid-plane layer | 5000 | 0.04 |
| Circular region | 5000 | 0.12 |

`E` is Young's modulus: it controls elastic stiffness and affects stress
redistribution and stored elastic energy. `Gc` is fracture energy per new
crack area: it controls energetic resistance to damage. Increasing `Gc`
does not increase initial elastic stiffness; increasing `E` does not by
itself make a material tougher. Neither change guarantees a particular path.

For a toughness comparison, change only the disk's `Gc`; keep `E`,
geometry, mesh, loading, and tolerances fixed. For a stiffness question,
vary only `E` first. The retained DCB has equal `E` throughout and is not
evidence of a stiffness-inclusion effect.

The documented multi-material route is limited to structured T3 rectangles,
CPU float64, elementwise `E`/`Gc`, common `nu`/`l0`, shared Amor AT2
constitutive settings, and Dirichlet (prescribed-value) conditions. Keep
plane-stress and other shared settings consistent. A bulk-property boundary
is not an independently calibrated interface or debonding law. Do not infer
dynamic multi-material, arbitrary imported-mesh, or 3D support from this
layout.

### Units

These inputs use mm for lengths/displacements, MPa (`N/mm^2`) for `E`,
and `N/mm` for `Gc`. Strain, damage, and `nu` are dimensionless.
Reactions use the runner's unit-thickness convention; declare out-of-plane
thickness before comparing measured forces. `geometry.units: mm` does not
convert arbitrary bare numbers elsewhere in a schema-2 file.

Dynamics requires consistent time and density. In an mm-N-s system, density
is in tonne/mm^3. The illustrative `rho: 1.0` in the quasi-static DCB/SENT
inputs is not calibrated density and must not be copied into a physical
dynamic model without justification and conversion.

## 7. Separate physical analysis from integration

**Quasi-static analysis** neglects inertia and finds equilibrium at successive
loads. Its load factor is not physical time. The DCB uses
`analysis_steps[0].type: quasi_static` and a staggered quasi-static solver.

**Dynamics** includes inertia, density, and physical time. Explicit integration
advances motion subject to a Courant-Friedrichs-Lewy (CFL) stability limit
that depends on mesh size and wave speed. Changing `E`, density, or mesh
can change that limit. Dynamic physics and numerical integration are separate
choices; not every possible combination is an implemented adapter.

| Control | Meaning |
|---|---|
| DCB `controls.number_of_steps: 121` | 121 nominal load levels including zero: 120 nominal increments, possibly more accepted increments after cutbacks. |
| Final physical time | Requested duration of a dynamic calculation. |
| Maximum step count | A work limit; a short cap may stop before final time or crack growth. |
| Time step / CFL safety factor | Stability and temporal resolution, not physical validation. |
| `maximum_staggered_iterations` | Iteration budget within an increment, not additional physical loading. |
| Trajectory `every` | Saved-state cadence, not integration or load resolution. |

Use the actual controls in the selected exercise, not guessed aliases named
`max_steps` or `final_time`. The legacy dynamic route uses controls such
as `loading.t_total` and `loading.num_steps`; the schema-2 adapter must
document and test its mapping. Do not switch the heterogeneous DCB deck to
`explicit`: this route does not provide dynamic multi-material execution.

Smaller time/load steps are sensitivity checks, not validation by themselves.
Compare responses at the same physical time or load, and inspect convergence,
energy where available, and mesh sensitivity before drawing a conclusion.

## 8. Inspect outputs, not just successful termination

For DCB, inspect `history.csv`, `solver_telemetry.csv`, `summary.json`,
and the setup, material, field, and response plots. This **partial** schema-2
excerpt requests a local trajectory; retain the other output requests:

```yaml
outputs:
  fields:
    - {name: trajectory, format: h5, every: 1}
```

HDF5 is the default when a trajectory is requested on this route. DCB requests
it; heterogeneous SENT leaves it opt-in. PNG/GIF and CSV files alone are not
a reloadable field trajectory.

```python
import phast

result = phast.load_result("runs/dcb_fine_reference")
print(result.metadata())
print(result.history_names())
print(result.visuals())
if result.has_field("damage"):
    damage = result.field("damage", step=-1)
```

A short exercise can demonstrate setup and output without new crack growth.
The starter crack already has `d = 1`, so `max_damage = 1` alone proves
no propagation. Compare the seed-connected damaged region with its initial
extent, inspect fields, and read the reaction history. A front coordinate
overlapping the disk's x-range does not establish penetration or bypass.

Keep large raw `training_data.h5` files in local outputs or a separate,
explicit data release. The online lesson needs only compact retained
figures and numerical records, not public raw HDF5.

## 9. Read the retained fine DCB result

These figures are **retained evidence**, not a new run executed by this page.
Coordinates are in mm; damage ranges from zero to one. Grey outlines show
bulk-material boundaries.

```{figure} ../../examples/two_material_dcb_beta/results/damage_final.png
:alt: Retained DCB damage field with a forked crack in the weak layer near the circular tougher region.
:width: 100%

Final state at 0.30 mm total opening. No seed-connected nodes with damage
at least 0.80 enter the disk. Neither penetration nor completed bypass is
demonstrated; this is qualitative layered-material interaction.
```

```{figure} ../../examples/two_material_dcb_beta/results/damage_evolution.gif
:alt: Retained DCB damage evolution from the starter crack to 0.30 mm total opening, with material boundaries outlined.
:width: 100%

Evolution over load factors zero to one, ending at 0.30 mm total opening.
This is quasi-static loading, not physical time. The static figure above
is the animation fallback. The sequence is not calibrated inclusion-bypass
evidence.
```

| Fine reference quantity | Retained value |
|---|---:|
| Mesh | 160 x 48 cells split into T3 triangles |
| Accepted increments / cutbacks | 120 / 0 |
| Maximum projected damage residual | 6.91e-6 |
| Final reaction, recorded unit-thickness convention | 0.61495 |
| Connected front at `d >= 0.80` | 2.0625 mm |

Sources: {download}`solver summary <../../examples/two_material_dcb_beta/results/summary.json>`
and {download}`numerical audit <../../examples/two_material_dcb_beta/results/numerical_review.json>`.
The historical generating source fingerprint was not recorded at runtime
and remains unknown. A later public-availability fingerprint does not attest
which source generated these fields. Independent stored-field checks found
the damage/history bounded and irreversible and recomputed projected residuals
using history from the final displacement: maximum 6.97e-6 for the fine case
and 5.23e-6 for the control. These final-state checks are distinct from the
solver-recorded maximum over accepted steps and are not independent reruns.

## 10. Reproduce the separate coarse matched pair

Two compact, complete inputs reuse the same runner and section order:

- {download}`tough_region.yaml <../../examples/two_material_dcb_beta/comparisons/tough_region.yaml>`
  uses the full DCB physics/settings with `nx: 120` instead of `160`.
- {download}`uniform_layer.yaml <../../examples/two_material_dcb_beta/comparisons/uniform_layer.yaml>`
  uses that coarse configuration with only disk `Gc: 0.12` changed to
  `0.04`, equal to the weak layer.

The outer regions remain fracture-resistant. "Uniform layer" means no disk
toughness contrast within the layer, not a homogeneous specimen. These are
duplicated model inputs, not duplicated solver code or new solver drivers.
The new inputs request HDF5 for local reruns. The retained coarse tough-region
run did not write HDF5. The retained uniform-layer control did: its HDF5
mesh supplies the original triangle connectivity used by the comparison
plots. Full trajectory files are not included in the compact public results.

```bash
python -m phast explain-config examples/two_material_dcb_beta/comparisons/tough_region.yaml
python -m phast run examples/two_material_dcb_beta/comparisons/tough_region.yaml --validate-only
python -m phast run examples/two_material_dcb_beta/comparisons/tough_region.yaml --output_dir runs/dcb_coarse_tough_region

python -m phast explain-config examples/two_material_dcb_beta/comparisons/uniform_layer.yaml
python -m phast run examples/two_material_dcb_beta/comparisons/uniform_layer.yaml --validate-only
python -m phast run examples/two_material_dcb_beta/comparisons/uniform_layer.yaml --output_dir runs/dcb_coarse_uniform_layer
```

Use these explicit, separate output directories rather than the inherited
default. The fine reference retains its geometry, material parameters, and
loading. The values below describe the retained coarse comparison, not a
new run of these input files.

| At 0.30 mm opening on 120 x 48 cells | Tough region | Uniform-layer control |
|---|---:|---:|
| Disk `Gc` [N/mm] | 0.12 | 0.04 |
| Final reaction, recorded convention | 0.60938 | 0.45847 |
| Connected front at `d >= 0.80` [mm] | 2.10 | 2.40 |
| Accepted increments / cutbacks | 120 / 0 | 123 / 3 |

Only the disk toughness differs in the physical inputs. Cutbacks produce
different accepted load sequences despite matched loading and solver
settings. Both cases fork early. The tougher region has shorter final
advance and a higher final reaction in this discrete comparison, but cannot
be credited with causing the initial branching. The fine reaction
`0.61495` is not the coarse matched tough-region reaction.

See the {download}`matched comparison record <../../examples/two_material_dcb_beta/results/matched_control_review.json>`.
The disk radius is only `2 l0`, and its clearance from the outer regions is
`l0`. Confinement, regularisation, centroid-based boundaries, load-increment
sensitivity, and small-strain/rotation assumptions remain relevant. Two mesh
sizes are not a formal convergence study. Fixed-field review does not close
input-guard or publication checks; no experimental, ASTM, or interface-law
validation is claimed.

## 11. Replot compact retained data without solving

The compact-data postprocessor is separate from the FEM commands above:

```bash
python examples/two_material_dcb_beta/compare_results.py --output-dir runs/dcb_comparison_replot
```

It reads the retained `coarse_tough_history.csv`, `uniform_layer_history.csv`,
`matched_control_fields.csv.gz`, `matched_control_elements.csv.gz`, and
`comparison_provenance.json` under the example's `results/` directory. It
regenerates `matched_control_comparison.png` and `matched_control_damage.png`
in the requested output directory without a FEM solve or raw HDF5.
A separate replot check regenerated both PNGs from the 250532-byte compact
history/field/connectivity payload. This is postprocessing evidence, not a
new FEM calculation.

The histories use their own accepted load grids. Compare at matched opening
rather than equating row numbers. `comparison_provenance.json` must keep
historical source identity explicitly unknown where it was not recorded;
a public-availability fingerprint is not proof of prior generation.

## Ask for help

If a command, configuration section, or interpretation is unclear,
[open a GitHub issue](https://github.com/CEMS-Lab/PhAST/issues/new/choose).
Include the complete input, exact command, OS, Python/PhAST version or
source revision, full error output, and whether you inspected retained
figures or a new result. You need not diagnose solver internals first.
