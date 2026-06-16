# Benchmark examples

Run commands, common CLI flags, crack-detection / GIF recording, stagger
convergence criteria, output directory layout, and terminal-output anatomy.


```
[Profiler] Timing breakdown:
  Region                     Total (s)    Calls   Avg (ms)      %
  ---------------------------------------------------------------
  mechanics                     45.230      150     301.53   72.3%
  damage                        12.100      150      80.67   19.3%
  psi_plus                       5.250      150      35.00    8.4%
```

### mesh_generator.py

Generates Gmsh `.geo` files for standard benchmark geometries, then meshes them.
Supports mesh refinement zones around the expected crack path.

| Function | Geometry |
|----------|----------|
| `miehe_tension(path, ...)` | 1x1mm square, horizontal notch at mid-height (SENT) |
| `miehe_shear(path, ...)` | 1x1mm square, horizontal notch at mid-height (SENS) |
| `three_point_bending(path, ...)` | 8x2mm beam, V-notch from bottom center |
| `square_plate(path, ...)` | Plain square (no notch) |

### visualization.py

- `plot_field(mesh, field, ...)` — Tricontour plot of a nodal/element field
- `plot_initial_conditions(mesh, mat, bcs, config, ...)` — 4-panel PNG: mesh,
  BCs, material info, loading schedule
- `plot_final_state(mesh, d, history, ...)` — Final damage + load-displacement
- `GIFRecorder` — Collects per-step frames (3-panel: damage, von Mises stress,
  von Mises strain) and saves an animated GIF
- `compute_von_mises_stress(sxx, syy, sxy)` — Von Mises from stress components
- `compute_von_mises_strain(exx, eyy, gxy)` — Von Mises from strain components

### metrics.py — `PFMBenchMetrics`

Standardized evaluation metrics (PFM-Bench protocol, Hamdi & Lejeune 2026):

```python
from phast.metrics import PFMBenchMetrics, load_fd_csv
m = PFMBenchMetrics(mesh, crack_threshold=0.5)
report = m.evaluate(d_pred, d_ref, fd_pred, fd_ref, energy_pred, energy_ref)
m.print_report(report)
```

| Metric | Method | Description |
|--------|--------|-------------|
| MSE | `mse(d_pred, d_ref)` | Mean squared error |
| Relative L2 | `relative_l2(d_pred, d_ref)` | `\|\|pred-ref\|\| / \|\|ref\|\|` |
| L-inf | `linf(d_pred, d_ref)` | Maximum pointwise error |
| Dice | `dice_coefficient(d_pred, d_ref)` | Crack region overlap (0-1) |
| IoU | `iou(d_pred, d_ref)` | Jaccard index for crack region |
| Histogram KL | `damage_histogram_kl(d_pred, d_ref)` | Distribution divergence |
| Crack path | `crack_path_error(d_pred, d_ref)` | Max lateral deviation [mm] |
| Peak force | `peak_force_error(fd_pred, fd_ref)` | Relative peak force error |
| F-D curve | `fd_curve_error(fd_pred, fd_ref)` | Interpolated curve L2 error |
| Energy | `energy_error(e_pred, e_ref)` | Evolution error + monotonicity |

### io_utils.py

| Function | Description |
|----------|-------------|
| `write_vtu(path, mesh, point_data, cell_data)` | ParaView VTU snapshot |
| `init_zarr(path, mesh, mat)` | Create a Zarr trajectory store with mesh, metadata, legacy step groups, and dense step-major arrays |
| `write_zarr_snapshot(zarr_root, step, mesh, u, d, psi, H, ...)` | Write one snapshot to both Zarr layouts when possible |
| `load_state_from_zarr(path, step=None)` | Load a restart state from the dense Zarr trajectory layout, falling back to legacy step groups |
| `init_h5(path, mesh, mat)` | Create legacy H5 with mesh + metadata + PyG edge_index |
| `write_h5_snapshot(h5f, step, mesh, u, d, psi, H, ...)` | Write legacy per-step H5 data |
| `compute_edge_index(elements)` | Triangle connectivity to PyG bidirectional edges |
| `write_profiler_csv(path, profiler)` | Export profiler timings to CSV |
| `CSVHistory` | Append-mode CSV writer for load-displacement history |

## Benchmark Examples

### Quasi-Static Benchmarks

Three standard benchmarks from Miehe et al. (2010), validated against
[PhaseFieldX](https://phasefieldx.readthedocs.io/):

| Benchmark | Split | Loading | Reference | Expected peak |
|-----------|-------|---------|-----------|--------------|
| [SENT](https://github.com/CEMS-Lab/PhAST/tree/main/examples/quasistatic/miehe_tension) | Isotropic | Tension (u_y) | PFX 1711 | ~0.70 kN |
| SENS | Spectral | Shear (u_x) | PFX 1712 | deferred until public YAML/visual contract promotion |
| TPB | Spectral | Bending (u_y) | PFX 1714 | deferred until public YAML/visual contract promotion |

### Explicit Dynamics Benchmark

| Benchmark | Split | Loading | Solver | Default steps |
|-----------|-------|---------|--------|---------------|
| [SENT Explicit](https://github.com/CEMS-Lab/PhAST/blob/main/configs/benchmarks/dynamic/B3_dynamic_sent.yaml) | Isotropic | Symmetric tension | Velocity-Verlet | 1500 |

Same geometry and material as the quasi-static SENT but solved with explicit
dynamics (CFL-limited timestep). Displacement is applied instantaneously via
a static pre-strain, then stress-wave driven crack propagation follows.
Each step is O(N) — no iterative equilibrium solve — making it the fastest
solver mode and suitable as a rapid training data generator for deep learning.

### Run

```bash
cd PhAST

# Quasi-static public benchmark
python -m phast run examples/quasistatic/miehe_tension/config.yaml

# Explicit-dynamic benchmarks are now driven by YAML configs (no per-benchmark
# run.py). Use the CLI with the matching B*_*.yaml config:
python -m phast run configs/benchmarks/dynamic/B3_dynamic_sent.yaml --device cuda --gif

# Quick smoke test (3 steps)
python -u examples/quasistatic/miehe_tension/run.py --num_steps 3 --plots
```

### Common CLI Flags

All six examples share the same interface:

| Flag | Default | Description |
|------|---------|-------------|
| `--h5` | off | Deprecated compatibility alias for Zarr trajectory snapshots (`training_data.zarr`) |
| `--vtu` | off | VTU snapshots for ParaView |
| `--gif` | off | Animation request; QS drivers write MP4 by default and fall back to GIF when needed |
| `--plots` | off | PNGs: initial conditions, final damage, load-displacement, energy, staggered convergence |
| `--profile` | off | Profiler timing CSV |
| `--all_outputs` | off | Enable VTU, GIF, plots, and profiler; request legacy H5 explicitly with `--h5` when existing postprocessors need it |
| `--num_steps` | varies | Total steps (150/300 quasi-static, 1500 explicit) |
| `--h_crack` | varies | Element size in crack zone |
| `--h_coarse` | varies | Element size far from crack |
| `--device` | auto | Force device (`cuda`, `cuda:N`, `mps`, `cpu`) |
| `--preconditioner` | jacobi | Damage-CG preconditioner: `jacobi`, `spectral`, `gmg`, `amg`, `amgx`, `auto`; `jacobi` is the QS-safe default |
| `--compile` | off | Enable torch.compile (off by default, see above) |
| `--stop_at_crack` | off | Stop one step after max(d) > 0.99 |
| `--stagger_tol` | 1e-8 | Staggered iteration convergence tolerance (quasi-static only) |
| `--max_stagger` | 500 | Max staggered iterations per step (quasi-static only) |
| `--stagger_criterion` | relative | Convergence criterion: `absolute`, `relative`, `am_energy`, `linf`, or `residual` (see below) |
| `--damage_cg_tol` | 1e-6 | CG convergence tolerance for damage solver (quasi-static only) |
| `--mechanics_cg_tol` | 1e-8 | CG convergence tolerance for mechanics solver (quasi-static only) |
| `--energy_split` | (preset) | Override energy split: `isotropic`, `amor`, `spectral`, `star_convex` |
| `--H_cap_factor` | 0 | Optional non-reference H cap = factor * Gc/(2*l0), 0 = off |
| `--disp` | 0.006 | Applied displacement in mm (explicit only) |
| `--dt_safety` | 1.0 | CFL safety factor (explicit only) |
| `--output_dir` | auto (timestamped) | Override output directory |
| `--vtu_every` | 1 | Write VTU every N steps |
| `--gif_frames` | 150 | Max GIF frames (skip interval computed automatically) |
| `--anderson_depth` | 0 | Anderson Acceleration depth (0=off, 3-5 typical) |
| `--multigrid` | **on** | 2-level GMG preconditioner (default, use `--no-multigrid` to disable) |

### Crack Detection & GIF Recording

The GIF records frames through the entire simulation (every `gif_every` steps).
When `max(d) > 0.99` (crack fully developed), a notification is printed but
the GIF and simulation continue recording all remaining steps. The full GIF is
saved at the end of the run.

Use `--stop_at_crack` to stop both the simulation and GIF early (one step after
