# Setting up your problem

This tutorial walks through writing a YAML config from scratch. Every
benchmark in `configs/` is a worked example; the field-by-field
schema lives in `configs/REFERENCE.yaml`. Once you have a config you
can validate it without running:

```bash
python -m phast run my_problem.yaml --validate-only
```

## Step 1 -- pick a geometry

`phast` ships a small library of mesh generators. Each
emits a Gmsh `.msh` file with named node sets (`top`, `bottom`,
`left`, `right`, plus geometry-specific sets like `notch_upper` or
`hole_*`). The full list is in `mesh_generator.py` and `config.py`;
the most-used:

| `geometry.type` | Use | Key parameters |
|-----------------|-----|----------------|
| `rectangular_sent` | Single-edge-notched plates, branching benchmarks | `W`, `H`, `a`, `h_crack`, `h_coarse`, `branching` |
| `miehe_tension` | Square SENT (Miehe 2010) | `L`, `a`, `h_crack`, `h_coarse` |
| `miehe_shear` | Square SENS | `L`, `a`, `h_crack`, `h_coarse` |
| `kalthoff_winkler` | Two-notch impact specimen | `W`, `H`, `theta`, `h_crack` |
| `three_point_bending` | TPB beam | `L`, `H`, `a`, `h_crack` |
| `l_shaped_panel` | L-panel re-entrant corner | `L`, `h_crack` |
| `perforated_sent` | SENT with holes | `W`, `H`, `hole_config` |

```yaml
geometry:
  type: rectangular_sent
  parameters:
    W: 100.0
    H: 40.0
    a: 50.0
    h_crack: 0.5
    h_coarse: 4.0
    branching: true
```

For a custom geometry, write a `.geo` file (or a Python function that
calls the Gmsh API) and register it in `mesh_generator.py` /
`config.py`. The `DOCUMENTATION.md` "Adding a New Geometry" section
has a worked example.

## Step 2 -- pick a material preset

Material presets bundle elastic constants, density, fracture
toughness, regularisation length, and a default energy split. Pick the
preset that matches your reference paper, then override the bits you
care about:

```yaml
material:
  preset: glass_borden       # E=32 GPa, nu=0.20, rho=2450 kg/m^3, Gc=3 J/m^2
  overrides:
    l0: 0.5                  # mm; override regularisation length
    energy_split: spectral
    pf_model: AT2
    eta_residual: 1e-7
```

Common presets (full list in `material.py`):

| Preset | E (MPa) | Gc (N/mm) | Default split | Reference |
|--------|---------|-----------|---------------|-----------|
| `glass_borden` | 32000 | 3e-3 | spectral | Borden 2012 |
| `maraging_steel_kw` | 190000 | 22.13 | spectral | Borden 2012 (Kalthoff) |
| `pmma_bleyer` | 3090 | 0.3 | amor (plane stress) | Bleyer 2017 |
| `miehe_tension` | 210000 | 2.7 | isotropic | Miehe 2010 |
| `l_shaped_concrete` | 25850 | 0.089 | spectral | Ambati 2015 |
| `alumina_kumar` | 335000 | 0.0268 | star_convex | Kumar 2020 |

To define a material inline (no preset), pass every property in
`overrides`:

```yaml
material:
  overrides:
    E: 70000
    nu: 0.23
    rho: 2500.0
    Gc: 0.008
    l0: 0.4
    energy_split: spectral
    pf_model: AT2
```

The unit system is `mm-tonne-s-MPa`, so `Gc` is in `N/mm`
(`1 N/mm = 1000 J/m^2`), density in `tonne/mm^3`, and boundary traction
is in `N/mm` force per boundary length. String-suffixed material, loading,
and boundary-condition values are accepted, for example `"32 GPa"`,
`"80 us"`, `"0.01 mm"`, and `"1 MPa"`. Mixing bare numbers from different
unit systems is the most common cause of "all damage / no damage" runs.

## Step 3 -- choose `pf_model` and `energy_split`

The decision flowchart:

- **Pre-existing notch and Mode I loading** -- AT2 + `isotropic` (or
  `amor` if compressive zones are anticipated).
- **Pre-existing notch, mixed-mode / shear / branching** -- AT2 +
  `spectral`. Most dynamic-fracture benchmarks live here.
- **No pre-crack, study nucleation** -- AT1 + `spectral` or
  `star_convex`. AT1 enforces an elastic threshold so the bulk stays
  intact until the strength is reached.
- **Thin plane-stress benchmark** -- prefer plane-stress `amor` unless you
  explicitly want the reduced 2D in-plane `spectral` projection. Plane-stress
  `spectral` is not a fully condensed 3D plane-stress spectral decomposition
  with damage-dependent out-of-plane strain; plane-strain `spectral` remains
  the mature validated Miehe-style principal-strain path.
- **Convergence trouble with curved cracks** -- swap `spectral` for
  `amor`; Amor's vol-dev split is monotone in the principal strains
  and avoids the eigenvector switching that hurts CG conjugacy.

See the [primer](01_phase_field_primer.md) for the underlying theory.

## Step 4 -- define boundary conditions

`boundary_conditions` is a list of `{nodes, type, component, value}`
entries. `nodes` is a node-set name from the geometry; `component` is
0 (x) or 1 (y); `type` is one of `fix`, `prescribe`, `neumann`, or
`traction`.

```yaml
boundary_conditions:
  - { nodes: bottom, type: fix, component: 0 }    # u_x = 0 on bottom
  - { nodes: bottom, type: fix, component: 1 }    # u_y = 0 on bottom
  - { nodes: top,    type: prescribe, component: 1, value: "0.01 mm" }
  - { nodes: right,  type: traction,  component: 1,
      value: "1 MPa", ramp_type: smooth_step, t_ramp: "10 us" }
```

`prescribe` stores displacement in mm and is scaled by the load factor in
quasi-static. `neumann` is the legacy constant traction form; `traction`
adds explicit `ramp_type`, `t_ramp`, and `t_hold`. Traction is stored as
`N/mm`; stress suffixes such as `MPa` assume unit out-of-plane thickness.

## Step 5 -- choose a solver

Two solver families cover most use cases:

| `solver.solver_type` | Method | Use |
|----------------------|--------|-----|
| `explicit` | Velocity-Verlet (central difference) | Dynamic fracture, impact, branching |
| `quasi_static` | Newton-Raphson with sparse-direct or CG inner solve | SENT, SENS, TPB, L-panel |
| `quasi_static_legacy` | Frozen-secant CG (Newton-skipping) | Compatibility with older accepted runs |

```yaml
loading:
  protocol: simple
  t_total: 80.0e-6      # explicit: physical end time in seconds
  # OR for quasi-static:
  # num_steps: 200
  # dt: 1.0e-4          # displacement increment per step (mm)

solver:
  solver_type: explicit
  dt_safety: 0.8        # CFL fraction
  damage_every: 1       # reference validation; use 2-3 after sensitivity checks
```

Quasi-static defaults that matter:

```yaml
solver:
  solver_type: quasi_static
  stagger_criterion: linf      # robust convergence detection
  stagger_tol: 1.0e-6
  anderson_depth: 3            # 30-50% iteration reduction
  backend: auto                # sparse-direct mechanics when available
  preconditioner: jacobi       # conservative QS damage default
```

For production quasi-static CPU runs, keep damage on Jacobi unless you are
explicitly testing AMG. `preconditioner: auto` may try AMG/GMG and should be
treated as an experimental performance setting for QS fracture.

For the full schema (every option, defaults, allowed values) see
`configs/REFERENCE.yaml`.

## Step 6 -- state the acceptance target

For benchmark or customer-validation runs, add an `acceptance:` block before
submitting to a workstation or HPC queue. This block is structured but
extensible: it records what the run is expected to reproduce, not another set
of solver controls. PhAST validates the standard fields (`status`,
`required_outputs`, and `metrics`) while preserving custom benchmark-specific
keys. `explain-config` prints it, and the resolved config and run lockfile
preserve it for later review.

```yaml
acceptance:
  status: beta
  reference_result: "Borden et al. (2012), Fig. 10"
  required_outputs: [run_lockfile.json, config.yaml, damage_final.png]
  metrics:
    crack_path:
      target: "straight crack to right boundary"
      tolerance: visual
    peak_force:
      target: null
      tolerance: null
      units: N
  notes: "Fill numerical targets once the reference extraction is audited."
```

Keep this block honest: use `scaffold` or `beta` until the benchmark has a
documented reference extraction, a known reaction/load convention, and a
stored comparison artifact.

## Step 7 -- run and interpret output

```bash
python -m phast run my_problem.yaml --device cpu
```

Useful flags:

| Flag | Effect |
|------|--------|
| `--device cpu \| cuda \| mps` | Override the device |
| `--fast` | Skip live plotting; with `--h5`, write legacy H5 for later post-processing |
| `--gif` | Render configured field animations; MP4/raster is the default when ffmpeg is available, and `output.gif_fields` can add `stress` and `displacement` |
| `--h5_every 50` | Snapshot frequency in legacy H5 |
| `--num_steps 5` | Smoke test (override step count) |
| `--validate-only` | Parse + schema-check, do not run |

The run directory (`runs/<config_name>_<timestamp>/`) contains:

- `damage_final.png` -- final damage tricontour
- `sample_*.zarr` or a run-level `.zarr` store -- preferred for new
  neural-operator, replay-buffer, and large dataset-generation runs
- `training_data.h5` -- legacy per-step `u`, `d`, `H`, energy, reaction
  trajectory when `output.h5` / `--h5` is requested
- `results.csv` -- step / displacement / reaction / max(d), if
  `output.reaction_node_set` is set
- `solver_telemetry.csv` -- stagger iterations, linear iterations,
  absolute/relative residuals, and load/time increment
- `load_displacement.png` -- required for quasi-static validation
  problems with a controlled displacement or load
- `staggered_convergence.png` -- required for quasi-static validation
  problems so convergence can be compared with PhaseFieldX-style
  `phasefieldx.conv`
- `config.yaml` -- the resolved config (every default filled in), for
  reproducibility

To re-run post-processing without re-simulating:

```bash
python -m phast postprocess <run_dir> --dpi 300
```

## Templates

`configs/REFERENCE.yaml` is the every-option template. For copy-paste
starting points:

| Use case | Template |
|----------|----------|
| Dynamic SENT | `configs/benchmarks/dynamic/B3_dynamic_sent.yaml` |
| Kalthoff impact | `configs/B2_kalthoff_winkler.yaml` |
| PMMA dynamic branching | `configs/B5_pmma_branching.yaml` |
| COMSOL cross-check | `configs/B7_dynamic_crack_branching_comsol.yaml` |
| Quasi-static L-panel | `configs/QS_lshaped_concrete.yaml` |
| Quasi-static notched plate | `configs/QS_notched_holed_plate.yaml` |

Once a config runs cleanly, head to
[exploration experiments](04_exploration_experiments.md) for ideas on
what to vary next.
