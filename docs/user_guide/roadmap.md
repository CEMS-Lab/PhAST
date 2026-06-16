# Extensions & roadmap

Near-, medium-, and long-term roadmap items, plus the YAML configuration
reference (geometries, materials, BC types, loading protocols, Python API).

`A_csr` is extremely ill-conditioned. During crack nucleation, `H` spikes
by 3–5 orders of magnitude on a handful of tip-adjacent elements while the
rest of the mesh remains smooth, driving the condition number of `A_csr`
high enough that pyamg's internal float64 arithmetic overshoots to inf.

The `except (ValueError, FloatingPointError)` in `MultilevelAMGPreconditioner.update`
catches this and returns without rebuilding, so CG falls back to the *previous*
AMG hierarchy (still effective, just not reflecting the latest stiffness).
Once `max(d)` saturates at 1 and the post-clamp irreversibility freezes `H`
at the tip, the condition number stabilises and subsequent rebuilds succeed.

Cosmetic mitigations (do not change physics):
- Rebuild AMG less often: `SolverConfig(amg_rebuild_every=50)` samples stable
  configurations only, avoiding the transition window entirely.
- Switch to Jacobi during transition: `--preconditioner jacobi` costs a few
  extra CG iterations per step but has no setup failure mode.

Symptoms that would indicate a real bug (and are *not* the above):
- `[AMG] WARNING: N non-finite reaction_coeff entries, clamping` — means
  `H` itself is non-finite, usually from a diverged mechanics solver.
- CG iteration count spikes into the hundreds after the warnings, or
  `max(d)` plateaus below 1. The cached hierarchy is no longer effective.
  In that case check for mesh degeneracies (zero-area triangles) or unset
  `damage_every` when subcycling.

**Stagger doesn't converge:**
- Increase `--max_stagger` (default 500)
- Add `--anderson_depth 3` or 5
- Try `--stagger_criterion linf` for more robust convergence detection
- Reduce load increment (increase `--num_steps`)

**MPS (Apple Silicon) issues:**
- Auto-detect now selects CPU over MPS (MPS lacks float64, causing CG ping-pong)
- Use `--device mps` to force MPS if desired
- torch.compile is off by default (use `--compile` to opt in, but expect issues on MPS)

**Memory issues:**
- Use `--preconditioner jacobi` to reduce memory (no coarse matrix)
- Reduce mesh size (`--h_crack` larger, `--h_coarse` larger)
- Check VRAM estimate: `DeviceContext('cuda').estimate_vram_mb(n_nodes)`

## Potential Extensions / Roadmap

Features that could make phast a more complete phase-field research
tool:

### Near-term (low effort)
- **Multi-format mesh import**: Read `.inp` (Abaqus), `.vtk`, `.med` via meshio
  — mesh.py already uses meshio, just needs format-aware node set parsing
- **Neumann boundary conditions**: Traction/pressure loads (currently Dirichlet
  only) — needed for three-point bending with distributed loads
- **AT1 model support**: Linear dissipation `alpha(d) = d` instead of `d^2`
  — material.py already has the flag, damage_solver needs the AT1 weak form
- **Hybrid monolithic solver**: Solve the coupled (u, d) system in one shot
  using block preconditioners — eliminates stagger iteration overhead

### Medium-term (moderate effort)
- **Higher-order solver dispatch**: T6 (quadratic triangles), Q4/Q8/Q9
  (quads) now have tested primitive shape functions, quadrature rules, and
  single-element stiffness assembly; the remaining work is production mechanics,
  damage, mass, preconditioner, and IO dispatch.
- **3D extension**: Tetrahedral elements (T4) — fem_operators.py generalizes
  naturally (3D strain = 6 components, 3D shape functions)
- ~~**Adaptive mesh refinement**~~: **Done** (v0.10.0) — NVB h-refinement with
  conforming closure in `adaptive.py`. See the **Adaptive Mesh Refinement** section
- **Contact/cohesive zone**: Crack-face contact for compression, cohesive
  elements for mixed-mode — needed for realistic shear fracture
- **Thermal coupling**: Thermo-mechanical phase-field for thermal shock fracture

### Long-term (research-level)
- **Ductile fracture**: Phase-field for elasto-plastic fracture (Miehe et al.,
  2016) — requires return mapping for plasticity
- **Multi-physics**: Hydrogen embrittlement, corrosion-driven fracture,
  hydraulic fracturing (pressure-driven cracks)
- **Topology optimization**: Differentiable solver enables gradient-based
  structural optimization with fracture constraints
- **Real-time digital twins**: Neural operator (trained on phast data)
  + phast correction loop for online structural health monitoring

### YAML Configuration System (v0.12.1)

Define a complete simulation in a single YAML file — no Python code needed.

#### Quick start

```bash
# Run a benchmark from YAML
python -m phast run configs/benchmarks/dynamic/B3_dynamic_sent.yaml

# Override settings from CLI
python -m phast run configs/benchmarks/dynamic/B5_pmma_branching.yaml --device cuda --fast

# Post-process results
python -m phast postprocess path/to/run_dir --dpi 300
```

#### Example YAML config (glass crack branching)

```yaml
# configs/benchmarks/dynamic/B3_dynamic_sent.yaml
problem:
  name: "Dynamic SENT"
  reference: "Borden et al. (2012), CMAME"

geometry:
  type: rectangular_sent           # mesh generator function name
  parameters:
    W: 100.0                       # plate width (mm)
    H: 40.0                        # plate height (mm)
    a: 50.0                        # notch length (mm)
    h_crack: 0.25                  # public smoke-scale SENT setup
    h_coarse: 4.0                  # coarse mesh size (mm)
    branching: true                # refine full right half for branching

material:
  preset: glass_borden             # from material.py presets
  overrides:
    l0: 0.5                        # public smoke-scale SENT setup
    energy_split: spectral         # spectral (Miehe) decomposition

boundary_conditions:
  - { nodes: left, type: fix, component: 0 }           # u_x = 0 on left
  - { nodes: top, type: neumann, component: 1, value: 1.0 }    # traction +σ on top
  - { nodes: bottom, type: neumann, component: 1, value: -1.0 } # traction -σ on bottom

loading:
  protocol: simple                 # simple | two_step_prestrain | cyclic
  num_steps: 1500

solver:
  solver_type: explicit            # explicit | quasi_static
  dt_safety: 0.8
  use_multigrid: true

output:
  h5: true
  h5_every: 20
  fast: true                       # solver + legacy H5 only, postprocess later
  print_every: 100
```

#### Available geometries

| Name | Description | Key parameters |
|------|-------------|----------------|
| `miehe_tension` | SENT square plate | L, a, h_crack |
| `miehe_shear` | SENS shear plate | L, a, h_crack |
| `rectangular_sent` | Rectangular SENT (branching) | W, H, a, branching |
| `kalthoff_winkler` | Two-notch impact plate | W, H, a |
| `l_shaped_panel` | L-shaped (Ambati 2015) | L |
| `three_point_bending` | TPB with notch | W, H, a |
| `perforated_sent` | SENT with holes | n_holes, hole_spacing |
| `square_plate` | Plain square | L |

#### Available material presets

| Preset | E (MPa) | Gc (N/mm) | Model | Use case |
|--------|---------|-----------|-------|----------|
| `glass_borden` | 32000 | 3.0 | AT2 spectral | B1, B3, B4 |
| `maraging_steel_kw` | 190000 | 22.13 | AT2 spectral | B2 Kalthoff |
| `pmma_bleyer` | 3090 | 0.3 | AT1 amor PS | B5, B6 Bleyer |
| `l_shaped_concrete` | 25850 | 0.089 | AT2 spectral | L-panel |
| `miehe_tension` | 210000 | 2.7 | AT2 isotropic | QS SENT |

#### Boundary condition types

```yaml
# Fix a DOF to zero
- { nodes: bottom, type: fix, component: 1 }           # u_y = 0

# Prescribe a displacement (scaled by load_factor in QS)
- { nodes: top, type: prescribe, component: 1, value: 1.0 }

# Apply traction (Neumann BC)
- { nodes: top, type: neumann, component: 1, value: 1.0 }  # σ_y = 1 MPa
```

#### Loading protocols

```yaml
# Simple: ramp or hold
loading:
  protocol: simple
  num_steps: 1500

# Two-step: pre-strain then dynamic release (PMMA branching)
loading:
  protocol: two_step_prestrain
