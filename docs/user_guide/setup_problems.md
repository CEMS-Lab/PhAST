# Setting up new problems

Use the fluent `phast.Problem` API to author new models. Use YAML decks for public examples, reproducibility, batch/HPC runs, and sharing exact simulations.

This guide covers the forward setup flow: geometry or mesh, material, Gc,
boundary conditions, loading protocol, solver settings, outputs, validation,
and result inspection. The public result bundle follows
`docs/user_guide/example_contract.md`, with compatibility run-file details in
`docs/STANDARD_OUTPUTS.md`, promoted visual rules in
`docs/visualisation_requirements.md`, and VTU/PV visualization format guidance
in `docs/visualization-output.md`.

Installation and backend selection are covered in `docs/installation.md`,
`docs/getting-started.md`, and
`docs/performance_reproducibility/index.md`. This page stays focused on the
problem definition itself.

## Quick Start

**Fluent `phast.Problem` authoring** (preferred for new models):

```python
import phast

problem = (
    phast.Problem("PMMA Branching")
    .geometry("rectangular_sent", W=32, H=16, a=16,
              h_crack=0.15, h_coarse=2.0, branching=True)
    .material("pmma_bleyer", l0=0.25, energy_split="amor", pf_model="AT1")
    .fix("bottom", dof="xy")
    .prescribe("top", dof="y", value=0.04)
    .loading(protocol="two_step_prestrain", t_total=40e-6,
             prestrain_displacement=0.04)
    .solver(dt_safety=0.8, use_multigrid=True, damage_every=1,
            bounds_method="projected_cg")
    .outputs(trajectory=True, h5_every=50, plots=True)
    .device("cuda")
)
```

**YAML input deck** (public reproducibility, examples, CI, and HPC):

```bash
python -m phast run configs/benchmarks/dynamic/B3_dynamic_sent.yaml \
  --device cpu --fast --output_dir output/b3_dynamic_sent
```

Validate before running:

```bash
python -m phast run configs/benchmarks/dynamic/B3_dynamic_sent.yaml \
  --validate-only
python -m phast explain-config configs/benchmarks/dynamic/B3_dynamic_sent.yaml
```

For public examples and batch runs, the YAML file is the solver input deck: it selects geometry or an imported `.msh`,
material parameters, boundary conditions, loading, solver options, Zarr-first
trajectory output, plots, and acceptance metadata. A run writes the resolved
`config.yaml`, `run_lockfile.json`, metadata, mesh provenance, CSV telemetry,
trajectory stores, and standard figures into the output directory.

**Python API** (for execution, custom automation, and solver extensions):

```python
from phast import Problem

solver = (Problem('My First Crack')
    .geometry('rectangular_sent', W=100, H=40, a=50,
              h_crack=0.125, h_coarse=4.0, branching=True)
    .material('glass_borden', l0=0.25, energy_split='spectral')
    .fix('left', dof='x')
    .neumann('top', dof='y', value=1.0)
    .neumann('bottom', dof='y', value=-1.0)
    .loading(protocol='simple', t_total=80e-6)
    .solver(dt_safety=0.8, use_multigrid=True, damage_every=1)
    .device('cpu')
    .run(output_dir='output/'))
```

**YAML + CLI**:

```bash
python -m phast run configs/benchmarks/dynamic/B3_dynamic_sent.yaml --device cpu --fast
```

**Existing example scripts** (for benchmarks with custom post-processing):

```bash
python -u examples/quasistatic/miehe_tension/run.py --num_steps 5 --plots
```

## Setting Up New Problems

### Option A: Python Problem Builder

The `Problem` class is the forward authoring API. It chains configuration in
one expression and can run supported promoted paths directly. Every method
returns `self`, so calls can be chained.

```python
from phast import Problem

solver = (Problem('PMMA Branching')
    .geometry('rectangular_sent', W=32, H=16, a=16,
              h_crack=0.15, h_coarse=2.0, branching=True)
    .material('pmma_bleyer', l0=0.25, energy_split='amor', pf_model='AT1')
    .fix('bottom', dof='xy')
    .prescribe('top', dof='y', value=0.04)
    .loading(protocol='two_step_prestrain', t_total=40e-6,
             prestrain_displacement=0.04)
    .solver(dt_safety=0.8, use_multigrid=True, damage_every=1,
            bounds_method='projected_cg')
    .outputs(trajectory=True, h5_every=50, plots=True)
    .device('cuda')
    .run(output_dir='output/pmma_branching'))
```

**Saving and loading**: any `Problem` can be serialised to YAML and
loaded back:

```python
prob = Problem('test').geometry('miehe_tension', L=1, a=0.5, h_crack=0.01, h_coarse=0.1)
prob.save('my_problem.yaml')

prob2 = Problem.from_yaml('my_problem.yaml')
prob2.run()
```

### Option B: YAML Input Deck (No Python Driver Required)

Create or save a `.yaml` file and run it directly. This is the public
reproducibility path for examples, sharing, CI, and HPC:

```yaml
# my_problem.yaml
problem:
  name: Shear Plate with Holes
geometry:
  type: perforated_sent
  parameters:
    W: 100
    H: 50
    h_crack: 0.3
    h_coarse: 3.0
    hole_config: 1hole_near
material:
  preset: pmma_bleyer
  overrides:
    l0: 0.5
    energy_split: amor
    pf_model: AT1
boundary_conditions:
- {nodes: bottom, type: fix, component: 0}
- {nodes: bottom, type: fix, component: 1}
- {nodes: top, type: prescribe, component: 0, value: 0.01}
loading:
  protocol: simple
  t_total: 40.0e-6
solver:
  solver_type: explicit
  dt_safety: 0.8
  damage_every: 1
  use_multigrid: true
output:
  fast: true
  print_every: 200
```

```bash
python -m phast run my_problem.yaml --device cuda
```

CLI flags override YAML values, so `--device cuda` overrides whatever
is in the file.

### Available Geometries

| Name | Description | Key Parameters |
|------|-------------|----------------|
| `miehe_tension` | Unit square, horizontal notch (Miehe SENT) | `L`, `a`, `h_crack`, `h_coarse` |
| `miehe_shear` | Unit square, horizontal notch (Miehe shear) | `L`, `a`, `h_crack`, `h_coarse` |
| `rectangular_sent` | Rectangular plate, edge notch | `W`, `H`, `a`, `h_crack`, `h_coarse`, `branching` |
| `kalthoff_winkler` | Impact specimen with angled notch | `W`, `H`, `theta`, `h_crack`, `h_coarse` |
| `perforated_sent` | Notched plate with holes | `W`, `H`, `h_crack`, `h_coarse`, `hole_config` |
| `three_point_bending` | Three-point bending beam | `L`, `H`, `a`, `h_crack`, `h_coarse` |
| `l_shaped_panel` | L-shaped domain with re-entrant corner | `L`, `h_crack`, `h_coarse` |
| `square_plate` | Plain square plate (no notch) | `L`, `h` |
| `plate_with_holes` | Plate with circular holes | `W`, `H`, `holes`, `h_crack`, `h_coarse` |
| `glass_impact_vnotch` | V-notch impact specimen | `W`, `H`, `h_crack`, `h_coarse` |
| `bazant_gap_test` | Gap test specimen | `W`, `H`, `gap`, `h_crack`, `h_coarse` |

### Available Material Presets

Material overrides: any property can be overridden in `.material()` or
YAML `overrides:` — `E`, `nu`, `Gc`, `l0`, `rho`, `energy_split`
(`spectral`, `amor`, `isotropic`), `pf_model` (`AT1`, `AT2`),
`eta_residual`, etc.

### Fracture Energy Gc: values and provenance

**Unit system (code).** All presets use `mm-tonne-s-MPa`, so `Gc` is in
`N/mm`. Conversion: `1 N/mm = 1000 J/m^2 = 1 kJ/m^2`. For reference, real
engineering Gc values are typically in `kJ/m^2`; benchmark-paper values
are often quoted in `J/m^2`.

**Key distinction.** Phase-field benchmark papers almost always use Gc
values *below* the engineering Gc of the named material. The reduction
is deliberate: it shortens the characteristic length `l* = E Gc / sigma_c^2`
so that crack initiation and propagation complete within a tractable
simulation window at a resolvable mesh. Using real engineering Gc would
require 10-100x more compute and is rarely done. The table below lists
both the preset value (matching the cited source) and, where available,
the real material Gc from the fracture-mechanics literature.

| Preset | Preset Gc (N/mm) | Preset Gc (J/m^2) | Source (verified) | Real material Gc (J/m^2) | Notes |
|---|---|---|---|---|---|
| `glass_borden` | 3.0e-3 | 3 | Borden et al. 2012, Table at §4.1 (PDF verified 2026-04-21) | Soda-lime 7-10 (Lawn 1993; Wiederhorn 1969) | Benchmark value; ~3x below real glass to keep branching in window |
| `maraging_steel_kw` | 22.13 | 22,130 | Borden 2012, §4.3 Kalthoff-Winkler (PDF verified) | C300 maraging ~48,000 (from Kc=100 MPa-sqrt(m) via Gc = Kc^2(1-nu^2)/E) | ~2x below real steel; standard Kalthoff-benchmark value |
| `pmma_bleyer` | 0.3 | 300 | Bleyer et al. 2017, §3.1 (PDF verified) | PMMA 350-550 (Kinloch & Young 1983) | Within engineering range; closest to real of the benchmark presets |
| `pmma` (generic) | 0.3 | 300 | — (same as `pmma_bleyer` without plane_stress flag) | same | Generic PMMA; prefer `pmma_bleyer` for branching benchmarks |
| `miehe_tension` | 2.7 | 2,700 | Miehe, Hofacker & Welschinger 2010, §5.1 SENT tension (PDF verified — "λ=121.15, μ=80.77, gc=2.7e-3 kN/mm") | Structural steel 10,000-60,000 | ~4-20x below real steel; numerical benchmark for SENT |
| `miehe_shear` | 2.7 | 2,700 | Miehe et al. 2010, §5.2 SENS shear (same parameter set as §5.1) | same as `miehe_tension` | Same material, shear loading test |
| `alumina_kumar` | 0.0268 | 26.8 | Kumar & Lopez-Pamies 2020, JMPS (source citation pending re-verification from the PDF) | Real alumina 30-60 (Wiederhorn 1969) | In correct order of magnitude for brittle ceramic |
| `brittle_ceramic` | 0.042 | 42 | Generic example (no specific source) | Brittle ceramic 20-100 | Generic; within range for various technical ceramics |
| `soda_lime_glass` | 9.0 | 9,000 | Liu-Lopez-Pamies-Dolbow arXiv:2411.16393, §5.2 Table 6 (PDF verified — E=72 GPa, Gc=9 N/mm) | Soda-lime 7-10 | **Not a typo.** Paper uses an effective Gc ~1000x above classical soda-lime. Likely reflects an effective-toughness from Hopkinson-bar data, not Griffith surface energy. Check the source before using for a different application. |
| `l_shaped_concrete` | 0.089 | 89 | Ambati et al. 2015 / Winkler 2001 (source citation pending re-verification from the Ambati PDF) | Concrete 50-200 | In engineering range |
| `l_shaped_glass` | 0.008 | 8 | Rudshaug et al. 2024, Int. J. Fract. (source citation pending re-verification) | Soda-lime 7-10 | Matches real glass Gc — engineering-faithful value |

**Practical rules for choosing Gc:**

- For *reproducing* a published benchmark, use the preset unchanged
  (matches the cited paper exactly).
- For *production* simulations of a real part, use the engineering Gc
  from the fracture-mechanics literature — not the benchmark value. Pass
  via `overrides: {Gc: ...}` in YAML or `create_material(..., Gc=...)`.
- If the simulation is too slow at engineering Gc, either (a) accept
  that benchmark Gc is an admissible simplification, or (b) use a
  coarser mesh with a larger `l0`, or (c) subcycle the mechanics
  (`damage_every > 1`) — but note that `damage_every > 1` suppresses
  branching in AT1/Amor runs (task #131, verified 2026-04-21).
- Three entries flagged above ("pending re-verification") have the
  code value believed correct but the cited source has not been
  re-read from the PDF by the maintainer since the preset was added.
  Do not cite them as validated literature values in a publication
  without a fresh source check.

### Boundary Condition DSL

**Python API** — string dof names, automatic component expansion:
