# From a tutorial to a first research study

This guide describes the transition from running a supplied example to defining
a small, reviewable phase-field fracture study. It is intended for graduate
students who know the finite-element method at an introductory level but are new
to PhAST.

The objective is not to select a model automatically. The objective is to make
every modelling decision visible, test the executable pathway at small scale,
and identify the evidence required before interpreting a calculation as a
research result.

## What this workflow establishes

| Stage | What it establishes | What it does not establish |
|---|---|---|
| Installation checks | The package imports and required baseline dependencies are available. | Correctness of a research model. |
| `--validate-only` | The YAML structure and implemented-option constraints pass preflight. | Successful execution, convergence, or physical validity. |
| Bounded pilot run | The selected public pathway executes and writes inspectable artifacts. | Mesh independence or agreement with an experiment. |
| Retained example evidence | Previously generated outputs can be inspected. | Reproduction on the current machine. |
| Validation study | A declared observable is compared with independent evidence under controlled assumptions. | Universal validity outside the tested regime. |

## 1. Reproduce the baseline route

Begin from the repository root in a Python 3.10-or-newer environment. Python
3.11 is the recommended baseline:

```bash
python -m phast doctor
python run_sanitizer.py
python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml --validate-only
python -m phast run examples/solid_mechanics_beta/linear_plate/config.yaml \
  --output_dir runs/linear_plate
```

Then inspect the completed result:

```python
import phast

result = phast.load_result("runs/linear_plate")
print(result.metadata())
print(result.history_names())
print(result.field_names())
print(result.visuals())
```

Do not begin a new model until this baseline either completes or produces a
retained first traceback that can be diagnosed. See
[Install PhAST](../install.md), [Verify the installation](../verify-install.md),
and [Troubleshooting](../troubleshooting.md).

## 2. State the research question

A computational study should begin with a measurable question, not with a list
of solver options. Write one sentence in the following form:

> Under **specified geometry, material, interface, loading, and boundary
> conditions**, how does **one controlled modelling variable** affect **one
> declared observable**, compared with **one validation target**?

Examples of observables include a force-displacement curve, crack-initiation
load, crack trajectory, dissipated fracture energy, or displacement field.
These observables are not interchangeable. Select the one supported by the
available experimental, analytical, or published comparison data.

## 3. Complete the modelling-decision record

Complete this table before editing a configuration:

| Decision | Required information | Record for the study |
|---|---|---|
| Geometry | Dimensions, thickness assumption, notch or defect definition, symmetry assumptions | |
| Microstructure | Homogeneous, explicit phases, elementwise fields, or segmented image; spatial resolution and registration | |
| Material phases | Elastic constants, density when required, units, source, and phase assignment | |
| Fracture law | Supported AT1/AT2 choice, `G_c`, `l0`, residual stiffness, and parameter source | |
| Energy decomposition | Supported split and reason for selecting it | |
| Interfaces | Perfect bonding, spatially varying bulk properties, or a separately documented beta interface route | |
| Loading | Displacement, traction, velocity, or impact history with units and rate | |
| Boundary conditions | Constrained components, loaded regions, symmetry, and prevention of rigid-body motion | |
| Calibration data | Parameters inferred from data and data excluded from calibration | |
| Validation target | Independent observable, acceptance metric, and tolerance | |
| Mesh | Element type, characteristic size, refinement region, and relation between `h` and `l0` | |
| Outputs | Fields, histories, sampling frequency, visual products, and storage estimate | |
| Compute route | CPU, CUDA, or supported optional backend; dtype and resource estimate | |

An empty entry is an open modelling question. It should not be silently replaced
with a convenient default.

## 4. Select the nearest supported pathway

Use the [capability matrix](../user_guide/capability_matrix.md) and the
[example gallery](../example-gallery.md) before choosing an input file.
The [dynamic SENT notebook](notebook_dynamic_sent.ipynb) provides a concise
worked example of tracing an existing mesh, named regions, loading, solver
choice, retained histories, and animation back to one public configuration.

| Intended study | Appropriate starting point | Boundary |
|---|---|---|
| Installation and general FEM execution | `examples/solid_mechanics_beta/linear_plate/config.yaml` | A mechanics check, not a fracture benchmark. |
| Quasi-static fracture | `examples/quasistatic/notched_holed_plate/config.yaml` or the example identified as closest to the target problem | Requires convergence, mesh, parameter, and comparison evidence for a new claim. |
| Explicit dynamic fracture | `examples/dynamic/B2_kalthoff_winkler/config.yaml` or another curated dynamic example | Requires stable time-step and output-cost assessment. |
| Elementwise heterogeneous properties | `examples/heterogeneous_fields/parameters.yaml` with its script-contract runner | A bounded teaching route, not a general heterogeneous YAML adapter or validated microstructure study. |
| Learned damage proposal | The experimental interface described in [Modular FEM and learned damage](03_modular_fem_and_learned_damage.md) | Replaces only the damage proposal; no trained checkpoint is distributed and classical acceptance/fallback remains authoritative. |

Do not construct a new configuration by combining options from unrelated
examples unless the capability matrix and schema identify that combination as
supported.

## 5. Represent geometry and microstructure

For an imported mesh, physical groups provide the contract between geometry and
the solver. Check that every domain, support, load boundary, notch, and material
region has an unambiguous name. Plot the groups before solving.

For a segmented or multiphase microstructure, record:

- the source image or reconstruction and its physical scale;
- the transformation from pixels or voxels to the two-dimensional model;
- the assignment from segments to phases;
- the mapping from phases to elements;
- the treatment of small features and disconnected regions;
- the ordering used by elementwise material arrays.

The public heterogeneous-fields example demonstrates element-ordered `E(x)`
and `G_c(x)` arrays. It does not by itself establish a calibrated interface
fracture law, arbitrary image ingestion through the general YAML runner, or
three-dimensional microstructure support.

## 6. Select material and fracture assumptions

Use one consistent unit system. Record the source and intended regime of each
parameter. In particular, `G_c` and `l0` should not be treated as arbitrary
numerical controls: they determine the regularized fracture response and may
interact with discretization and calibration.

Select only phase-field models, degradation choices, and strain-energy
decompositions accepted by the installed schema and documented for the chosen
workflow. The [phase-field primer](01_phase_field_primer.md) explains AT1/AT2,
degradation, history, and the supported energy-split terminology.

Interface behaviour requires a separate decision. Perfect bonding, diffuse
variation of bulk properties, and cohesive separation are different physical
models. The presence of beta cohesive or diffuse-interface examples does not
make those routes validated defaults for a new microstructure study.

## 7. Define loading and boundary conditions

Draw the boundary-value problem before encoding it. The drawing should identify
the prescribed displacement or traction, constrained components, symmetry
planes, initial crack or notch, and the observable used for comparison.

Check:

1. Rigid-body modes are removed without over-constraining the specimen.
2. Reaction forces are sampled from the intended named region and component.
3. The loading history, total load, number of steps, and time increment are
   dimensionally consistent.
4. Dynamic cases include density and a stable time-step assessment.
5. Quasi-static cases record staggered and inner-solver convergence information.

## 8. Relate the mesh to the phase-field length scale

The regularized crack is represented over a finite width controlled by `l0`.
The mesh must resolve this field in the expected fracture region. A ratio such
as `h/l0` is a diagnostic, not a universal certificate of convergence.

Perform at least a bounded refinement study before making a quantitative claim.
Keep geometry, material parameters, loading, solver tolerances, and output
definitions fixed while changing the mesh. If `l0` is calibrated rather than
fixed independently, state that explicitly.

Use the [mesh-resolution diagnostic](notebook_mesh_resolution.ipynb) to
understand profile sampling. It is not an FEM convergence study.

## 9. Author and preflight the configuration

Copy the nearest example into a study-specific location, retain its README and
provenance, and change one conceptual group at a time. Keep runnable YAML
separate from manifests, contracts, and templates.

```bash
python -m phast explain-config path/to/config.yaml
python -m phast run path/to/config.yaml --validate-only
```

Record the command and complete output. A successful preflight means only that
the configuration passed the implemented structural checks.

## 10. Execute a bounded pilot

Reduce the mesh or step count only in ways that preserve the purpose of the
pilot. Write to an explicit directory:

```bash
python -m phast run path/to/config.yaml --output_dir runs/my_pilot
```

The pilot should answer engineering questions:

- Did the intended solver and device route run?
- Were named regions and initial conditions applied?
- Did the run complete without non-finite values?
- Were convergence or stable-time-step diagnostics recorded?
- Were the requested fields and histories written?
- Is the storage and runtime cost acceptable before refinement?

It should not be used as quantitative fracture evidence merely because it
completed.

## 11. Inspect the result as a scientific record

Retain the configuration, exact command, environment or package revision,
metadata, lockfile, histories, fields, visual manifest, and comparison output.
Inspect these together rather than selecting only a favourable final image.

For a fracture study, the minimum interpretation normally includes:

- the loading history and reaction or response curve;
- damage-field evolution rather than only the final field;
- energy histories where the workflow provides them;
- solver convergence or time-step information;
- mesh and `l0`;
- the declared validation observable and error measure;
- failed, rejected, or fallback steps where relevant.

Follow [Performance and reproducibility](../performance-reproducibility.md) and
the [example input and output contract](../user_guide/example_contract.md).

## 12. Separate calibration from validation

Parameters fitted to one curve or image cannot be validated against the same
data without qualification. Record:

1. which data were used to select `G_c`, `l0`, elastic properties, interface
   parameters, or other model choices;
2. which independent data were withheld for validation;
3. the observable and acceptance metric defined before inspecting the result;
4. sensitivity to mesh, step size, tolerances, and uncertain parameters.

If independent validation data are unavailable, describe the calculation as an
illustrative, exploratory, or reproduction study rather than as validated
prediction.

## Realistic first milestone

A suitable first milestone is:

> Reproduce one curated public workflow, create one closely related geometry or
> material variation, complete configuration preflight and a bounded pilot,
> archive the result bundle, and compare one declared observable while recording
> mesh, solver, and evidence limitations.

New constitutive laws, unsupported interface formulations, automatic image-to-
mesh pipelines, three-dimensional fracture, a new learned architecture, or a
new execution adapter are software-development projects. They should be scoped
separately from the first modelling milestone.

## Requesting review or assistance

Open a [GitHub issue](https://github.com/CEMS-Lab/PhAST/issues) when a documented
step is unclear, a supplied command fails, or a capability boundary prevents the
intended study. Include:

- operating system and Python version;
- PhAST version or commit;
- the exact command;
- the smallest relevant configuration;
- the first complete traceback;
- expected and observed behaviour;
- whether the problem occurs during installation, preflight, execution, or
  post-processing.

Students, researchers, and developers are invited to propose documentation
corrections, focused examples, tests, and well-bounded solver improvements
through the public issue and pull-request workflow. See
[Contributing](https://github.com/CEMS-Lab/PhAST/blob/main/CONTRIBUTING.md)
before preparing a change.
